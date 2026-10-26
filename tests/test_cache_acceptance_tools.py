from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_model_state_cache import shared_prefix_audit
from scripts.validate_visual_cache import visual_cache_receipt


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def warmup(video: str, *, text_tokens: int = 200, visual_tokens: int = 800) -> dict:
    return {
        "question_id": "__video_write__",
        "video_id": video,
        "state_storage_unit": "vllm_multimodal_prefix_tokens",
        "input_text_tokens": text_tokens,
        "visual_tokens": visual_tokens,
    }


def answer(
    video: str,
    qid: str,
    *,
    cached_tokens: int,
    answer_request_count: int = 1,
) -> dict:
    return {
        "video_uid": video,
        "qid": qid,
        "cached_prompt_tokens": cached_tokens,
        "answer_request_count": answer_request_count,
    }


def test_shared_prefix_accepts_only_a_small_per_answer_shortfall() -> None:
    receipt = shared_prefix_audit(
        [
            answer("video-a", "q1", cached_tokens=936),
            answer("video-b", "q2", cached_tokens=1_000),
        ],
        [warmup("video-a"), warmup("video-b")],
        max_shortfall_tokens=128,
    )

    assert receipt["passed"] is True
    assert receipt["largest_shared_prefix_shortfall_tokens"] == 64
    assert receipt["minimum_cached_to_warmup_prompt_ratio"] == pytest.approx(0.936)
    assert receipt["failed_answers"] == 0


def test_shared_prefix_rejects_partial_hits_and_retried_answers() -> None:
    receipt = shared_prefix_audit(
        [answer("video-a", "q1", cached_tokens=800)],
        [warmup("video-a")],
        max_shortfall_tokens=128,
    )
    assert receipt["passed"] is False
    assert receipt["failed_answers"] == 1

    with pytest.raises(ValueError, match="one answer request per question"):
        shared_prefix_audit(
            [answer("video-a", "q1", cached_tokens=1_000, answer_request_count=2)],
            [warmup("video-a")],
            max_shortfall_tokens=128,
        )


def test_shared_prefix_requires_one_warmup_and_exact_video_coverage() -> None:
    with pytest.raises(ValueError, match="expected one prefix warmup per video"):
        shared_prefix_audit(
            [answer("video-a", "q1", cached_tokens=1_000)],
            [warmup("video-a"), warmup("video-a")],
            max_shortfall_tokens=128,
        )

    with pytest.raises(ValueError, match="coverage differs"):
        shared_prefix_audit(
            [answer("video-a", "q1", cached_tokens=1_000)],
            [warmup("video-a"), warmup("video-b")],
            max_shortfall_tokens=128,
        )


def test_model_state_cli_checks_metrics_ledger_and_shared_prefix(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.prom"
    results = tmp_path / "results.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "receipt.json"
    metrics.write_text(
        "# TYPE vllm:prefix_cache_hits counter\n"
        'vllm:prefix_cache_hits_total{model_name="synthetic"} 1900\n',
        encoding="utf-8",
    )
    result_rows = [
        answer("video-a", "q1", cached_tokens=950),
        answer("video-b", "q2", cached_tokens=950),
    ]
    ledger_rows = [
        warmup("video-a"),
        warmup("video-b"),
        {"question_id": "q1", "cached_prompt_tokens": 950},
        {"question_id": "q2", "cached_prompt_tokens": 950},
    ]
    write_jsonl(results, result_rows)
    write_jsonl(ledger, ledger_rows)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_model_state_cache.py"),
            "--metrics",
            str(metrics),
            "--results",
            str(results),
            "--ledger",
            str(ledger),
            "--out",
            str(out),
            "--max-shared-prefix-shortfall-tokens",
            "128",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["prefix_cache_hit_tokens"] == 1_900
    assert receipt["all_request_cached_prompt_tokens"] == 1_900
    assert receipt["counter_difference"] == 0
    assert receipt["strict_shared_prefix_audit"]["passed"] is True
    assert receipt["passed"] is True


def test_model_state_cli_rejects_metrics_ledger_disagreement(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.prom"
    results = tmp_path / "results.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "receipt.json"
    metrics.write_text("vllm:prefix_cache_hits 700\n", encoding="utf-8")
    write_jsonl(results, [answer("video-a", "q1", cached_tokens=950)])
    write_jsonl(
        ledger,
        [warmup("video-a"), {"question_id": "q1", "cached_prompt_tokens": 950}],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_model_state_cache.py"),
            "--metrics",
            str(metrics),
            "--results",
            str(results),
            "--ledger",
            str(ledger),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "model-state cache acceptance failed" in completed.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is False


def visual_inputs(*, cached_tokens: int = 0) -> tuple[list[dict], list[dict]]:
    manifest = [
        {"video_uid": "video-a", "qid": "q1"},
        {"video_uid": "video-b", "qid": "q2"},
    ]
    results = [
        {
            "video_uid": "video-a",
            "qid": "q1",
            "official_reference_times": [1.0],
            "cached_prompt_tokens": cached_tokens,
        },
        {
            "video_uid": "video-b",
            "qid": "q2",
            "official_reference_times": [],
            "cached_prompt_tokens": 0,
        },
    ]
    return manifest, results


def test_visual_cache_receipt_accepts_images_and_content_deduplication() -> None:
    manifest, results = visual_inputs()
    receipt = visual_cache_receipt(
        manifest,
        results,
        [{"item_count": 4}, {"item_count": 4}],
        num_frames=4,
        video_input=False,
    )

    assert receipt == {
        "videos": 2,
        "questions": 2,
        "frames_per_video": 4,
        "visual_input_format": "images",
        "encoder_item_executions": 8,
        "expected_encoder_item_executions": 9,
        "memory_encoder_item_executions": 8,
        "query_reference_encoder_item_executions": 1,
        "content_deduplicated_items": 1,
        "cached_prompt_tokens_max": 0,
        "accepted": True,
    }


@pytest.mark.parametrize(
    ("manifest", "results", "message"),
    [
        ([], [{"video_uid": "video-a", "qid": "q1"}], "manifest must be non-empty"),
        ([{"video_uid": "video-a", "qid": "q1"}], [], "results must be non-empty"),
    ],
)
def test_visual_cache_requires_nonempty_manifest_and_results(
    manifest: list[dict], results: list[dict], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 1}],
            num_frames=1,
            video_input=False,
        )


@pytest.mark.parametrize("source", ["manifest", "results"])
def test_visual_cache_rejects_duplicate_qids(source: str) -> None:
    manifest, results = visual_inputs()
    rows = manifest if source == "manifest" else results
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match=rf"{source} contains duplicate qid: q1"):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 9}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_requires_exact_qid_sets() -> None:
    manifest, results = visual_inputs()
    results[1]["qid"] = "q-extra"

    with pytest.raises(ValueError, match="manifest/result qid sets differ"):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 9}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_requires_matching_video_for_each_qid() -> None:
    manifest, results = visual_inputs()
    results[0]["video_uid"] = "wrong-video"

    with pytest.raises(ValueError, match="video_uid differs for qids"):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 9}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_requires_nonempty_trace() -> None:
    manifest, results = visual_inputs()
    with pytest.raises(ValueError, match="trace must be non-empty"):
        visual_cache_receipt(
            manifest,
            results,
            [],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_counts_only_multimodal_encoder_events() -> None:
    manifest, results = visual_inputs()
    receipt = visual_cache_receipt(
        manifest,
        results,
        [
            {"kind": "model_runner_execution", "encoder_item_count": 99},
            {"kind": "mm_encoder_execution", "item_count": 9},
        ],
        num_frames=4,
        video_input=False,
    )
    assert receipt["encoder_item_executions"] == 9


def test_visual_cache_rejects_item_count_on_another_trace_kind() -> None:
    manifest, results = visual_inputs()
    with pytest.raises(ValueError, match="unexpected trace kind contains item_count"):
        visual_cache_receipt(
            manifest,
            results,
            [{"kind": "model_runner_execution", "item_count": 9}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_strictly_rejects_encoder_reruns() -> None:
    manifest, results = visual_inputs()
    with pytest.raises(ValueError, match="visual-token cache miss"):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 10}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_strictly_rejects_any_lm_prefix_hit() -> None:
    manifest, results = visual_inputs(cached_tokens=1)
    with pytest.raises(ValueError, match="language-model prefix tokens"):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 9}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_rejects_implausibly_low_trace_count() -> None:
    manifest, results = visual_inputs()
    with pytest.raises(ValueError, match="unexpectedly low encoder count"):
        visual_cache_receipt(
            manifest,
            results,
            [{"item_count": 7}],
            num_frames=4,
            video_input=False,
        )


def test_visual_cache_cli_writes_native_video_receipt(tmp_path: Path) -> None:
    manifest_rows, result_rows = visual_inputs()
    manifest = tmp_path / "manifest.jsonl"
    results = tmp_path / "results.jsonl"
    trace = tmp_path / "trace.jsonl"
    out = tmp_path / "nested" / "receipt.json"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(results, result_rows)
    write_jsonl(trace, [{"kind": "mm_encoder_execution", "item_count": 3}])

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_visual_cache.py"),
            "--manifest",
            str(manifest),
            "--results",
            str(results),
            "--trace",
            str(trace),
            "--num-frames",
            "4",
            "--out",
            str(out),
            "--video-input",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["visual_input_format"] == "video"
    assert receipt["memory_encoder_item_executions"] == 2
    assert receipt["query_reference_encoder_item_executions"] == 1
    assert receipt["encoder_item_executions"] == 3
    assert receipt["accepted"] is True
