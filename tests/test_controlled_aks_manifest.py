from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_controlled_memory_policy_cgbench as runner


FRAME_BUDGET = 64


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selection_row(*, qid: str, video_uid: str, question: str) -> dict:
    candidate_count = 192
    duration = 180.0
    selected_candidate_indices = list(range(0, FRAME_BUDGET * 2, 2))
    selected_source_frame_indices = [index * 25 for index in selected_candidate_indices]
    selected_times = [float(index) for index in selected_candidate_indices]
    encoded_times = runner.canonical_times_payload(selected_times)
    scorer_model_id = "Salesforce/blip-itm-large-coco"
    selector_cost = {
        "candidate_fps": 1.0,
        "candidate_count": candidate_count,
        "decode_s": 1.25,
        "feature_s": 2.5,
        "selection_s": 0.05,
        "total_s": 3.8,
        "decode_cuda_s": 0.0,
        "feature_cuda_s": 2.0,
        "selection_cuda_s": 0.0,
        "peak_vram_bytes": 4096,
        "embedding_model_id": f"{scorer_model_id}:revision:test-revision",
        "embedding_retention_policy": "discard_after_timestamp_selection",
        "raw_video_seconds_read": duration,
        "cost_attribution": runner.AKS_COST_ATTRIBUTION,
        "physical_score_pass_shared_across_budgets": True,
    }
    return {
        "schema_version": runner.AKS_SELECTION_SCHEMA_VERSION,
        "config_sha256": sha256_text("shared-config"),
        "source_manifest_sha256": sha256_text("shared-source-manifest"),
        "dataset": "controlled-test",
        "qid": qid,
        "video_uid": video_uid,
        "video_path": f"videos/{video_uid}.mp4",
        "question_sha256": sha256_text(question),
        "question_aware": True,
        "question_only": True,
        "candidate_fps": 1.0,
        "candidate_count": candidate_count,
        "candidate_frame_indices_sha256": sha256_text(f"candidates:{video_uid}"),
        "candidate_times_sha256": sha256_text(f"candidate-times:{video_uid}"),
        "source_average_fps": 25.0,
        "source_frame_count": 5000,
        "duration": duration,
        "source_video_sha256": sha256_text(f"source-video:{video_uid}"),
        "scorer": {
            "model_id": scorer_model_id,
            "hf_revision": "test-revision",
            "model_artifact_sha256": sha256_text("frozen-scorer-artifact"),
            "frozen": True,
            "question_only": True,
        },
        "selector": {
            "id": "aks_cvpr2025_meanstd",
            "max_depth": 5,
            "threshold": 0.8,
            "ratio": 1,
        },
        "frame_budget": FRAME_BUDGET,
        "selected_candidate_indices": selected_candidate_indices,
        "selected_source_frame_indices": selected_source_frame_indices,
        "selected_times": selected_times,
        "selected_times_sha256": hashlib.sha256(encoded_times).hexdigest(),
        "retained_timestamp_bytes": len(encoded_times),
        "selection_wall_seconds": selector_cost["selection_s"],
        "selector_cost": selector_cost,
    }


def valid_manifest_rows() -> tuple[list[dict], list[dict]]:
    questions = [
        {
            "qid": "question-1",
            "video_uid": "video-1",
            "question": "What happens after the door opens?",
        },
        {
            "qid": "question-2",
            "video_uid": "video-2",
            "question": "Which object is placed on the table?",
        },
    ]
    rows = [
        selection_row(
            qid=question["qid"],
            video_uid=question["video_uid"],
            question=question["question"],
        )
        for question in questions
    ]
    return rows, questions


def write_manifest(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_loader_accepts_full_manifest_while_consuming_current_shard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aks_fixed64.jsonl"
    rows, questions = valid_manifest_rows()
    full_manifest_sha = write_manifest(path, rows)

    consumed, manifest_sha, available = runner.load_aks_query_selections(
        path,
        [questions[0]],
        FRAME_BUDGET,
    )

    assert available == 2
    assert set(consumed) == {"question-1"}
    assert manifest_sha == full_manifest_sha
    selected = consumed["question-1"]
    assert selected["qid"] == questions[0]["qid"]
    assert selected["video_uid"] == questions[0]["video_uid"]
    assert selected["question_sha256"] == sha256_text(questions[0]["question"])
    assert selected["frame_budget"] == FRAME_BUDGET
    assert selected["selected_times_sha256"] == rows[0]["selected_times_sha256"]
    assert selected["selector_cost"]["total_s"] == pytest.approx(3.8)


def test_physical_frame_mapping_accepts_rounded_short_duration_hint() -> None:
    row = selection_row(
        qid="question-last-frame",
        video_uid="video-last-frame",
        question="What happens at the end?",
    )
    # The final selected frame is physical frame 3150 at 25 fps, or 126.0 s.
    # A source manifest may round its duration hint slightly short; that hint is
    # not authoritative when the exact frame index remains in bounds.
    row["source_frame_count"] = 3151
    rounded_source_duration_hint = 125.987
    assert row["selected_times"][-1] - rounded_source_duration_hint == pytest.approx(
        0.013
    )

    runner.validate_aks_selection_against_physical_video(
        row,
        actual_frame_count=3151,
        actual_average_fps=25.0,
    )


def test_physical_frame_mapping_still_rejects_true_out_of_bounds_index() -> None:
    row = selection_row(
        qid="question-out-of-bounds",
        video_uid="video-out-of-bounds",
        question="What happens at the end?",
    )
    row["source_frame_count"] = 3151
    row["selected_source_frame_indices"][-1] = 3151

    with pytest.raises(ValueError, match="outside"):
        runner.validate_aks_selection_against_physical_video(
            row,
            actual_frame_count=3151,
            actual_average_fps=25.0,
        )


def test_physical_frame_mapping_rejects_timestamp_frame_mismatch() -> None:
    row = selection_row(
        qid="question-wrong-time",
        video_uid="video-wrong-time",
        question="What happens at the end?",
    )
    row["selected_times"][-1] += 0.01

    with pytest.raises(ValueError, match="timestamp/source frame mismatch"):
        runner.validate_aks_selection_against_physical_video(
            row,
            actual_frame_count=5000,
            actual_average_fps=25.0,
        )


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("question_sha", "question SHA mismatch"),
        ("budget", "budget mismatch"),
        ("cost_total", "selector_cost total mismatch"),
        ("selected_times_sha", "selected_times hash mismatch"),
    ],
)
def test_loader_rejects_tampered_question_bound_selection(
    tmp_path: Path,
    tamper: str,
    error: str,
) -> None:
    path = tmp_path / f"tampered-{tamper}.jsonl"
    original_rows, questions = valid_manifest_rows()
    rows = copy.deepcopy(original_rows)
    if tamper == "question_sha":
        rows[0]["question_sha256"] = sha256_text("different question")
    elif tamper == "budget":
        rows[0]["frame_budget"] = 32
    elif tamper == "cost_total":
        rows[0]["selector_cost"]["total_s"] += 1.0
    elif tamper == "selected_times_sha":
        rows[0]["selected_times_sha256"] = sha256_text("different timestamps")
    else:  # pragma: no cover - the parametrization is intentionally exhaustive.
        raise AssertionError(tamper)
    write_manifest(path, rows)

    with pytest.raises(ValueError, match=error):
        runner.load_aks_query_selections(path, [questions[0]], FRAME_BUDGET)


def test_selector_cost_events_keep_real_question_and_full_manifest_sha(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "aks_fixed64.jsonl"
    rows, questions = valid_manifest_rows()
    full_manifest_sha = write_manifest(manifest_path, rows)
    consumed, _, _ = runner.load_aks_query_selections(
        manifest_path,
        [questions[0]],
        FRAME_BUDGET,
    )
    ledger_path = tmp_path / "cost_ledger.jsonl"

    runner.append_aks_selector_cost_events(
        ledger_path=ledger_path,
        run_id="test-run",
        commit="test-commit",
        method="ControlledAKSFrames64",
        dataset_version="controlled-test-v1",
        selection_row=consumed["question-1"],
        selection_manifest_sha=full_manifest_sha,
    )

    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(events) == 4
    assert [event["tool_call_index"] for event in events] == [0, 1, 2, 3]
    assert [event["phase"] for event in events] == [
        "observe",
        "retrieve",
        "plan",
        "store",
    ]
    assert [event["gpu_count"] for event in events] == [0, 1, 0, 0]
    assert [event["model_id"] for event in events] == [
        "decord_candidate_video_decoder",
        rows[0]["selector_cost"]["embedding_model_id"],
        rows[0]["selector"]["id"],
        "aks_selected_timestamp_array",
    ]
    assert {event["question_id"] for event in events} == {"question-1"}
    assert all(event["question_id"] != "__video_write__" for event in events)
    assert {event["video_id"] for event in events} == {"video-1"}
    assert all(
        event["query_selection_manifest_sha256"] == full_manifest_sha
        for event in events
    )
    assert events[0]["latency_s"] == pytest.approx(1.25)
    assert events[1]["latency_s"] == pytest.approx(2.5)
    assert events[2]["latency_s"] == pytest.approx(0.05)
    assert events[3]["state_storage_bytes"] == rows[0]["retained_timestamp_bytes"]
