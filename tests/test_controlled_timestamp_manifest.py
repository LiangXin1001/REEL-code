from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_controlled_memory_policy_cgbench as runner
from src.pestate.cost_ledger import CostLedger, LedgerContext


def manifest_row(video_uid: str = "video-1") -> dict:
    times = [float(index) for index in range(64)]
    return {
        "schema_version": "pestate.question_blind_timestamp_manifest.v1",
        "video_uid": video_uid,
        "selector_id": "scene_aware_maxinfo_fixed64_clip_vit_l14_336_maxvol_32x2",
        "question_blind": True,
        "selected_times": times,
        "selected_times_sha256": hashlib.sha256(
            json.dumps(times, separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_candidate_indices": list(range(64)),
        "candidate_times_sha256": "a" * 64,
        "retained_timestamp_bytes": len(
            json.dumps(times, separators=(",", ":")).encode()
        ),
        "decodable_duration_s": 100.0,
        "selector_cost": {
            "candidate_fps": 1.0,
            "candidate_count": 101,
            "decode_s": 1.0,
            "feature_s": 2.0,
            "selection_s": 0.25,
            "total_s": 3.25,
            "decode_cuda_s": 0.0,
            "feature_cuda_s": 1.75,
            "selection_cuda_s": 0.0,
            "peak_vram_bytes": 4096,
            "embedding_model_id": "clip-vit-large-patch14-336:sha256",
            "embedding_retention_policy": "discard_after_timestamp_selection",
        },
    }


def write_manifest(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_loader_accepts_costed_question_blind_fixed_budget(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.jsonl"
    row = manifest_row()
    write_manifest(path, row)

    loaded, digest = runner.load_question_blind_timestamps(path, {"video-1"}, 64)

    assert loaded["video-1"]["selected_times"] == row["selected_times"]
    assert loaded["video-1"]["selector_cost"]["total_s"] == 3.25
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["question", "choices", "right_answer", "clue_intervals"])
def test_loader_rejects_query_side_fields(tmp_path: Path, field: str) -> None:
    path = tmp_path / "timestamps.jsonl"
    row = manifest_row()
    row[field] = "leak"
    write_manifest(path, row)

    with pytest.raises(ValueError, match="query-side fields"):
        runner.load_question_blind_timestamps(path, {"video-1"}, 64)


def test_loader_rejects_free_or_incomplete_selector_accounting(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.jsonl"
    row = manifest_row()
    row.pop("selector_cost")
    write_manifest(path, row)
    with pytest.raises(ValueError, match="no measured selector_cost"):
        runner.load_question_blind_timestamps(path, {"video-1"}, 64)

    row = manifest_row()
    row["selector_cost"]["total_s"] = 0.0
    write_manifest(path, row)
    with pytest.raises(ValueError, match="inconsistent total_s"):
        runner.load_question_blind_timestamps(path, {"video-1"}, 64)


def test_hybrid_cache_sets_are_disjoint_and_cover_both_joint_inputs() -> None:
    assert runner.FRAME_TEXT_HYBRID_VISUAL_MODE in runner.VISUAL_CACHE_MODES
    assert runner.FRAME_TEXT_HYBRID_KV_MODE in runner.KV_CACHE_MODES
    assert runner.VISUAL_CACHE_MODES.isdisjoint(runner.KV_CACHE_MODES)
    assert runner.FRAME_TEXT_HYBRID_MODES <= runner.FRAME_TEXT_WRITE_ONCE_MODES


def test_selector_cost_is_imported_as_one_time_decode_feature_rank_and_store(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "cost.jsonl"
    row = manifest_row()

    runner.append_selector_cost_events(
        ledger_path=ledger,
        run_id="run",
        commit="commit",
        method="ControlledWriteOnceFrameText",
        dataset_version="dataset",
        video_id="video-1",
        timestamp_row=row,
        timestamp_manifest_sha="b" * 64,
    )

    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [event["phase"] for event in events] == ["write", "write", "plan", "store"]
    assert [event["tool_call_index"] for event in events] == [0, 1, 2, 3]
    assert {event["question_id"] for event in events} == {"__video_write__"}
    assert events[0]["decoded_frames"] == 101
    assert events[0]["gpu_seconds"] == 0
    assert events[1]["gpu_seconds"] == pytest.approx(2.0)
    assert events[2]["selected_times_sha256"] == row["selected_times_sha256"]
    assert events[3]["state_storage_bytes"] == row["retained_timestamp_bytes"]


def write_canonical_writer_ledger(path: Path, frame_ids: list[str]) -> None:
    ledger = CostLedger(
        path,
        LedgerContext(
            "canonical-run",
            "canonical-commit",
            "ControlledWriteOnceFrameText",
            "dataset",
            "video-1",
            "__video_write__",
            1,
        ),
    )
    ledger.append(
        phase="write",
        model_id="model",
        input_text_tokens=10,
        output_text_tokens=5,
        visual_tokens=20,
        unique_frame_ids=frame_ids,
        latency_s=2.5,
        active_gpu_count=1,
        selected_times=[0.0, 1.0],
        measurement_note=(
            "Question-blind frame-level text write: one validated entry per fixed "
            "source frame."
        ),
    )


def test_joint_input_imports_exact_canonical_writer_cost(tmp_path: Path) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    source_frame_ids = [
        "video-1:controlled:0:0.000000",
        "video-1:controlled:1:1.000000",
    ]
    frame_ids = [
        "video-1:controlled:hash:native:0:0.000000",
        "video-1:controlled:hash:native:1:1.000000",
    ]
    write_canonical_writer_ledger(source / "cost_ledger.jsonl", source_frame_ids)
    target = tmp_path / "joint.jsonl"
    out_dir = tmp_path / "joint"
    out_dir.mkdir()

    runner.import_prebuilt_frame_text_writer_cost(
        directory=source,
        ledger_path=target,
        run_id="joint-run",
        commit="joint-commit",
        method="ControlledFrameTextVisualCache",
        dataset_version="dataset",
        model_id="model",
        video_id="video-1",
        frame_ids=frame_ids,
        selected_times=[0.0, 1.0],
        selected_times_sha256="a" * 64,
        timestamp_manifest_sha256="b" * 64,
        out_dir=out_dir,
    )

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["method_id"] == "ControlledFrameTextVisualCache"
    assert rows[0]["canonical_text_cost_imported"] is True
    assert rows[0]["gpu_seconds"] == pytest.approx(2.5)
    assert rows[0]["input_text_tokens"] == 10
    assert rows[0]["unique_frame_ids"] == frame_ids
    provenance = json.loads(
        (out_dir / "prebuilt_writer_cost_provenance.jsonl").read_text()
    )
    assert provenance["source_writer_events"] == 1
    assert provenance["imported_writer_events"] == 1


def test_joint_input_rejects_canonical_writer_on_different_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "canonical"
    source.mkdir()
    write_canonical_writer_ledger(
        source / "cost_ledger.jsonl",
        [
            "video-1:controlled:0:0.000000",
            "video-1:controlled:1:1.000000",
        ],
    )

    with pytest.raises(ValueError, match="frame-ID timestamp changed"):
        runner.import_prebuilt_frame_text_writer_cost(
            directory=source,
            ledger_path=tmp_path / "joint.jsonl",
            run_id="joint-run",
            commit="joint-commit",
            method="ControlledFrameTextVisualCache",
            dataset_version="dataset",
            model_id="model",
            video_id="video-1",
            frame_ids=[
                "video-1:controlled:hash:native:0:0.000000",
                "video-1:controlled:hash:native:1:2.000000",
            ],
            selected_times=[0.0, 2.0],
            selected_times_sha256="a" * 64,
            timestamp_manifest_sha256="b" * 64,
            out_dir=tmp_path,
        )
