#!/usr/bin/env python3
"""Recompute HD-EPIC cumulative costs at m=20 and all available questions.

The existing RQ2 curves stop at the first 20 questions of each video.  This
audit goes back to the accepted result and ledger files and sums the events for
the exact question IDs selected at each horizon.  It never estimates an
all-question total from an average per-question cost.

Token fields are deliberately kept separate from GPU time:

* ``reported_input_tokens`` is ``input_text_tokens + visual_tokens`` because
  some backends put image tokens in the former field and others in the latter.
* ``cached_input_tokens`` is the engine-reported ``cached_prompt_tokens``.
* ``uncached_input_tokens`` is their per-event difference.  It means
  "reported prompt tokens not covered by the language-model prefix cache"; it
  does not capture skipped visual-encoder work in the VisualTokens condition.
* ``output_tokens`` is never added to an input-token total.

Every executed event is retained even if its answer is later excluded from an
accuracy denominator.  This script analyzes resources only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/tables/hd_all_questions_costs_20260822"
MANIFEST_EXPECTED_QUESTIONS = 1871
MANIFEST_EXPECTED_VIDEOS = 51
FIRST_HORIZON = 20

MODEL_CONFIGS = {
    "Qwen3-VL-8B": ROOT / "outputs/tables/rq2_valid_pairs_hd_qwen3vl_20260820",
    "InternVL3-8B": ROOT / "outputs/tables/rq2_valid_pairs_hd_internvl3_20260820",
    "Gemma-3-12B-IT": ROOT / "outputs/tables/rq2_valid_pairs_hd_gemma3_20260820",
}

REPRESENTATION_ORDER = (
    "RepeatedFrames",
    "FrameText",
    "VisualTokens",
    "KVCache",
)

METRICS = (
    "gpu_seconds",
    "reported_input_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def require_accepted(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing acceptance file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if "accepted=true" not in lines:
        raise ValueError(f"source is not accepted: {path}")


def qid_of(row: dict[str, Any]) -> str:
    for key in ("qid", "question_id", "ID", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"record has no question ID: {row}")


def numeric(row: dict[str, Any], field: str) -> float:
    value = row.get(field, 0)
    if value is None:
        return 0.0
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"non-numeric {field} for video={row.get('video_id')} "
            f"qid={row.get('question_id')}: {value!r}"
        )
    return float(value)


def event_cost(row: dict[str, Any]) -> dict[str, float]:
    reported = numeric(row, "input_text_tokens") + numeric(row, "visual_tokens")
    cached = numeric(row, "cached_prompt_tokens")
    tolerance = 1e-9
    if cached < -tolerance or cached > reported + tolerance:
        raise ValueError(
            "cached_prompt_tokens falls outside [0, reported_input_tokens] "
            f"for video={row.get('video_id')} qid={row.get('question_id')}: "
            f"cached={cached}, reported={reported}"
        )
    return {
        "gpu_seconds": numeric(row, "gpu_seconds"),
        "reported_input_tokens": reported,
        "cached_input_tokens": cached,
        "uncached_input_tokens": max(0.0, reported - cached),
        "output_tokens": numeric(row, "output_text_tokens"),
    }


def sum_cost(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    total = {metric: 0.0 for metric in METRICS}
    for event in events:
        values = event_cost(event)
        for metric in METRICS:
            total[metric] += values[metric]
    return total


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    rows = read_jsonl(path)
    if len(rows) != MANIFEST_EXPECTED_QUESTIONS:
        raise ValueError(
            f"manifest has {len(rows)} questions, expected {MANIFEST_EXPECTED_QUESTIONS}"
        )
    qid_to_video: dict[str, str] = {}
    by_video_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qid = qid_of(row)
        if qid in qid_to_video:
            raise ValueError(f"duplicate manifest question ID: {qid}")
        video_id = str(row.get("video_uid", "")).strip()
        if not video_id:
            raise ValueError(f"manifest question {qid} lacks video_uid")
        qid_to_video[qid] = video_id
        by_video_rows[video_id].append(row)
    if len(by_video_rows) != MANIFEST_EXPECTED_VIDEOS:
        raise ValueError(
            f"manifest has {len(by_video_rows)} videos, expected {MANIFEST_EXPECTED_VIDEOS}"
        )

    by_video: dict[str, list[str]] = {}
    for video_id, video_rows in by_video_rows.items():
        video_rows.sort(key=lambda row: int(row["within_video_question_order"]))
        orders = [int(row["within_video_question_order"]) for row in video_rows]
        if orders not in (
            list(range(len(video_rows))),
            list(range(1, len(video_rows) + 1)),
        ):
            raise ValueError(f"non-contiguous question order for video={video_id}: {orders}")
        if len(video_rows) < FIRST_HORIZON:
            raise ValueError(
                f"video={video_id} has {len(video_rows)} questions, fewer than {FIRST_HORIZON}"
            )
        by_video[video_id] = [qid_of(row) for row in video_rows]
    return qid_to_video, by_video


def load_existing_m20_means(path: Path) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for method in REPRESENTATION_ORDER:
        method_rows = [
            row for row in rows if row["method"] == method and int(row["m"]) == FIRST_HORIZON
        ]
        indexed = {row["metric"]: float(row["mean_per_video"]) for row in method_rows}
        required = {"gpu_seconds", "output_text_tokens"}
        has_reported = "reported_input_tokens" in indexed
        has_components = {"input_text_tokens", "visual_tokens"} <= set(indexed)
        if not required <= set(indexed) or not (has_reported or has_components):
            raise ValueError(f"existing m=20 table lacks metrics for {method}: {path}")
        output[(method, "gpu_seconds")] = indexed["gpu_seconds"]
        output[(method, "reported_input_tokens")] = (
            indexed["reported_input_tokens"]
            if has_reported
            else indexed["input_text_tokens"] + indexed["visual_tokens"]
        )
        output[(method, "output_tokens")] = indexed["output_text_tokens"]
    return output


def summary_rows(per_video: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_video:
        groups[(row["model"], row["representation"], row["horizon"])].append(row)

    output: list[dict[str, Any]] = []
    horizon_order = {"first20": 0, "all_questions": 1}
    model_order = {name: index for index, name in enumerate(MODEL_CONFIGS)}
    rep_order = {name: index for index, name in enumerate(REPRESENTATION_ORDER)}
    for key in sorted(
        groups,
        key=lambda value: (
            model_order[value[0]], rep_order[value[1]], horizon_order[value[2]]
        ),
    ):
        model, representation, horizon = key
        rows = groups[key]
        qcounts = [int(row["questions"]) for row in rows]
        out: dict[str, Any] = {
            "model": model,
            "representation": representation,
            "horizon": horizon,
            "videos": len(rows),
            "questions_total": sum(qcounts),
            "questions_per_video_mean": fmean(qcounts),
            "questions_per_video_min": min(qcounts),
            "questions_per_video_max": max(qcounts),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            out[f"{metric}_mean_per_video"] = fmean(values)
            out[f"{metric}_total"] = sum(values)
        reported = float(out["reported_input_tokens_mean_per_video"])
        cached = float(out["cached_input_tokens_mean_per_video"])
        out["cached_input_fraction"] = cached / reported if reported else 0.0
        output.append(out)
    return output


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["model"], row["representation"], row["horizon"]): row for row in summary
    }
    output: list[dict[str, Any]] = []
    for model in MODEL_CONFIGS:
        for representation in REPRESENTATION_ORDER:
            first = indexed[(model, representation, "first20")]
            all_questions = indexed[(model, representation, "all_questions")]
            row: dict[str, Any] = {
                "model": model,
                "representation": representation,
                "first20_questions_total": first["questions_total"],
                "all_questions_total": all_questions["questions_total"],
                "all_questions_per_video_mean": all_questions["questions_per_video_mean"],
                "all_questions_per_video_range": (
                    f"{all_questions['questions_per_video_min']}-"
                    f"{all_questions['questions_per_video_max']}"
                ),
            }
            for metric in METRICS:
                first_value = float(first[f"{metric}_mean_per_video"])
                all_value = float(all_questions[f"{metric}_mean_per_video"])
                row[f"first20_{metric}_mean_per_video"] = first_value
                row[f"all_{metric}_mean_per_video"] = all_value
                row[f"all_minus_first20_{metric}_mean_per_video"] = all_value - first_value
                row[f"all_over_first20_{metric}"] = (
                    all_value / first_value if first_value else None
                )
            output.append(row)
    return output


def compact_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["model"], row["representation"], row["horizon"]): row for row in summary
    }
    output: list[dict[str, Any]] = []
    for model in MODEL_CONFIGS:
        repeated_first = indexed[(model, "RepeatedFrames", "first20")]
        repeated_all = indexed[(model, "RepeatedFrames", "all_questions")]
        for representation in REPRESENTATION_ORDER:
            first = indexed[(model, representation, "first20")]
            all_questions = indexed[(model, representation, "all_questions")]
            first_gpu = float(first["gpu_seconds_mean_per_video"])
            all_gpu = float(all_questions["gpu_seconds_mean_per_video"])
            repeated_first_gpu = float(repeated_first["gpu_seconds_mean_per_video"])
            repeated_all_gpu = float(repeated_all["gpu_seconds_mean_per_video"])
            all_reported = float(all_questions["reported_input_tokens_mean_per_video"])
            all_cached = float(all_questions["cached_input_tokens_mean_per_video"])
            output.append(
                {
                    "model": model,
                    "representation": representation,
                    "gpu_s_first20": round(first_gpu, 1),
                    "gpu_saving_vs_raw_first20_pct": round(
                        100.0 * (repeated_first_gpu - first_gpu) / repeated_first_gpu, 1
                    ),
                    "gpu_s_all": round(all_gpu, 1),
                    "gpu_saving_vs_raw_all_pct": round(
                        100.0 * (repeated_all_gpu - all_gpu) / repeated_all_gpu, 1
                    ),
                    "reported_input_k_all": round(
                        all_reported / 1000, 1
                    ),
                    "cached_input_k_all": round(
                        all_cached / 1000, 1
                    ),
                    "cached_input_pct_all": round(
                        100.0 * all_cached / all_reported if all_reported else 0.0, 1
                    ),
                    "uncached_input_k_all": round(
                        float(all_questions["uncached_input_tokens_mean_per_video"]) / 1000,
                        1,
                    ),
                    "output_k_all": round(
                        float(all_questions["output_tokens_mean_per_video"]) / 1000,
                        2,
                    ),
                }
            )
    return output


def main() -> None:
    global MODEL_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--include-new-models", action="store_true")
    parser.add_argument(
        "--internvl3-strict-ledgers",
        action="store_true",
        help="Use strict InternVL answers/timings and the provenance-recorded Text write ledger.",
    )
    parser.add_argument(
        "--timemap-hd-cost-tag",
        help=(
            "Use the six hybrid HD lifecycle outputs with this suffix. These "
            "combine the accepted time-map answer ledger with the earlier "
            "accepted Text writer ledger. Requires --include-new-models."
        ),
    )
    args = parser.parse_args()
    if args.timemap_hd_cost_tag and not args.include_new_models:
        parser.error("--timemap-hd-cost-tag requires --include-new-models")
    if args.internvl3_strict_ledgers:
        MODEL_CONFIGS = {
            **MODEL_CONFIGS,
            "InternVL3-8B": ROOT / "outputs/tables/rq2_valid_pairs_hd_internvl3_strict_final_20260824",
        }
    if args.include_new_models:
        MODEL_CONFIGS = {
            **MODEL_CONFIGS,
            "Qwen3-VL-4B": ROOT / "outputs/tables/rq2_valid_pairs_hd_qwen3vl4b_20260823",
            "InternVL3.5-8B": ROOT / "outputs/tables/rq2_valid_pairs_hd_internvl35_8b_20260823",
            "MiniCPM-V-4.5-8B": ROOT / "outputs/tables/rq2_valid_pairs_hd_minicpmv45_20260823",
        }
    if args.timemap_hd_cost_tag:
        slugs = {
            "Qwen3-VL-8B": "qwen3vl",
            "InternVL3-8B": "internvl3",
            "Gemma-3-12B-IT": "gemma3",
            "Qwen3-VL-4B": "qwen3vl4b",
            "InternVL3.5-8B": "internvl35_8b",
            "MiniCPM-V-4.5-8B": "minicpmv45",
        }
        MODEL_CONFIGS = {
            model: ROOT
            / "outputs/tables"
            / f"rq2_valid_pairs_hd_{slug}_{args.timemap_hd_cost_tag}"
            for model, slug in slugs.items()
        }
    out_dir = args.out_dir or (
        ROOT / "outputs/tables/hd_all_questions_costs_all6_20260823"
        if args.include_new_models
        else DEFAULT_OUT
    )

    manifest_path: Path | None = None
    qid_to_video: dict[str, str] | None = None
    by_video: dict[str, list[str]] | None = None
    per_video: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    m20_checks: list[dict[str, Any]] = []

    for model, analysis_dir in MODEL_CONFIGS.items():
        require_accepted(analysis_dir / "acceptance.txt")
        metadata_path = analysis_dir / "bootstrap_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_manifest = Path(metadata["manifest"])
        if manifest_path is None:
            manifest_path = current_manifest
            qid_to_video, by_video = load_manifest(manifest_path)
        elif current_manifest != manifest_path:
            current_qid_to_video, current_by_video = load_manifest(current_manifest)
            if current_qid_to_video != qid_to_video or current_by_video != by_video:
                raise ValueError(
                    "accepted analyses use non-equivalent manifests: "
                    f"{manifest_path} vs {current_manifest}"
                )
        assert qid_to_video is not None and by_video is not None

        existing = load_existing_m20_means(analysis_dir / "lifecycle_metric_ci.csv")
        result_sources = metadata.get("result_sources", {})
        run_sources = metadata.get("run_sources", {})
        if set(result_sources) != set(REPRESENTATION_ORDER):
            raise ValueError(
                f"{model} representation coverage mismatch: {sorted(result_sources)}"
            )

        for representation in REPRESENTATION_ORDER:
            results_path = Path(result_sources[representation])
            run_dir = results_path.parent
            answer_ledger_path = run_dir / "cost_ledger.jsonl"
            write_ledger_path = answer_ledger_path
            lifecycle_provenance_path: Path | None = None
            if representation in run_sources:
                lifecycle_path = Path(run_sources[representation])
                lifecycle_provenance_path = lifecycle_path.parent / "lifecycle_provenance.json"
                if lifecycle_provenance_path.is_file():
                    lifecycle_provenance = json.loads(
                        lifecycle_provenance_path.read_text(encoding="utf-8")
                    )
                    answer_ledger_path = Path(lifecycle_provenance["answer_ledger"])
                    write_ledger_path = Path(
                        lifecycle_provenance.get("write_ledger")
                        or lifecycle_provenance["answer_ledger"]
                    )
                    if Path(lifecycle_provenance["results"]) != results_path:
                        raise ValueError(
                            "lifecycle/result provenance mismatch for "
                            f"{model}/{representation}: {lifecycle_provenance_path}"
                        )
            answer_acceptance_path = answer_ledger_path.parent / "acceptance.txt"
            write_acceptance_path = write_ledger_path.parent / "acceptance.txt"
            require_accepted(answer_acceptance_path)
            require_accepted(write_acceptance_path)

            results = read_jsonl(results_path)
            result_qids = [qid_of(row) for row in results]
            if len(result_qids) != len(set(result_qids)):
                raise ValueError(f"duplicate result QIDs: {results_path}")
            if set(result_qids) != set(qid_to_video):
                raise ValueError(
                    f"result coverage mismatch for {model}/{representation}: "
                    f"missing={len(set(qid_to_video) - set(result_qids))} "
                    f"extra={len(set(result_qids) - set(qid_to_video))}"
                )

            answer_ledger = read_jsonl(answer_ledger_path)
            write_ledger = (
                answer_ledger
                if write_ledger_path == answer_ledger_path
                else read_jsonl(write_ledger_path)
            )
            answer_events_selected = [
                row for row in answer_ledger
                if str(row.get("question_id") or "").strip() != "__video_write__"
            ]
            build_events_selected = [
                row for row in write_ledger
                if str(row.get("question_id") or "").strip() == "__video_write__"
            ]
            ledger = answer_events_selected + build_events_selected
            # Accepted runs may retain traceable answer-format failures.  Their
            # resource use belongs in this analysis even though the answers are
            # excluded from an accuracy denominator elsewhere.
            error_events = [row for row in ledger if str(row.get("error") or "").strip()]
            build_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
            question_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for event in ledger:
                video_id = str(event.get("video_id", "")).strip()
                qid = str(event.get("question_id", "")).strip()
                if video_id not in by_video:
                    raise ValueError(f"ledger event has unknown video_id={video_id}: {answer_ledger_path}")
                if qid == "__video_write__":
                    build_events[video_id].append(event)
                    continue
                if qid not in qid_to_video:
                    raise ValueError(f"ledger event has unknown qid={qid}: {answer_ledger_path}")
                if qid_to_video[qid] != video_id:
                    raise ValueError(
                        f"ledger video/QID mismatch qid={qid}: expected={qid_to_video[qid]} "
                        f"got={video_id}"
                    )
                question_events[qid].append(event)
            missing_events = [qid for qid in qid_to_video if not question_events[qid]]
            if missing_events:
                raise ValueError(
                    f"ledger lacks events for {len(missing_events)} questions: {answer_ledger_path}"
                )
            if representation != "RepeatedFrames":
                missing_builds = [video for video in by_video if not build_events[video]]
                if missing_builds:
                    raise ValueError(
                        f"shared representation lacks builds for {len(missing_builds)} videos: "
                        f"{write_ledger_path}"
                    )

            source_records.append(
                {
                    "model": model,
                    "representation": representation,
                    "results": str(results_path),
                    "ledger": str(answer_ledger_path),
                    "acceptance": str(answer_acceptance_path),
                    "answer_ledger": str(answer_ledger_path),
                    "write_ledger": str(write_ledger_path),
                    "answer_acceptance": str(answer_acceptance_path),
                    "write_acceptance": str(write_acceptance_path),
                    "lifecycle_provenance": (
                        str(lifecycle_provenance_path)
                        if lifecycle_provenance_path is not None
                        else ""
                    ),
                    "ledger_events": len(ledger),
                    "build_events": sum(len(rows) for rows in build_events.values()),
                    "answer_events": sum(len(rows) for rows in question_events.values()),
                    "traceable_error_events_included": len(error_events),
                }
            )

            for video_id, ordered_qids in sorted(by_video.items()):
                for horizon, selected_qids in (
                    ("first20", ordered_qids[:FIRST_HORIZON]),
                    ("all_questions", ordered_qids),
                ):
                    selected_answer_events = [
                        event for qid in selected_qids for event in question_events[qid]
                    ]
                    selected_build_events = list(build_events[video_id])
                    build_cost = sum_cost(selected_build_events)
                    answer_cost = sum_cost(selected_answer_events)
                    total_cost = {
                        metric: build_cost[metric] + answer_cost[metric] for metric in METRICS
                    }
                    row: dict[str, Any] = {
                        "model": model,
                        "representation": representation,
                        "horizon": horizon,
                        "video_id": video_id,
                        "questions": len(selected_qids),
                        "build_events": len(selected_build_events),
                        "answer_events": len(selected_answer_events),
                        "ledger_events": len(selected_build_events) + len(selected_answer_events),
                    }
                    for metric in METRICS:
                        row[metric] = total_cost[metric]
                        row[f"build_{metric}"] = build_cost[metric]
                        row[f"answer_{metric}"] = answer_cost[metric]
                    per_video.append(row)

        model_summary = summary_rows([row for row in per_video if row["model"] == model])
        model_index = {
            (row["representation"], row["horizon"]): row for row in model_summary
        }
        for representation in REPRESENTATION_ORDER:
            first = model_index[(representation, "first20")]
            for metric in ("gpu_seconds", "reported_input_tokens", "output_tokens"):
                actual = float(first[f"{metric}_mean_per_video"])
                expected = existing[(representation, metric)]
                difference = actual - expected
                if abs(difference) > 1e-7:
                    raise ValueError(
                        f"m=20 reconstruction mismatch for {model}/{representation}/{metric}: "
                        f"actual={actual}, expected={expected}, difference={difference}"
                    )
                m20_checks.append(
                    {
                        "model": model,
                        "representation": representation,
                        "metric": metric,
                        "reconstructed": actual,
                        "accepted_table": expected,
                        "difference": difference,
                    }
                )

    assert manifest_path is not None and by_video is not None
    summary = summary_rows(per_video)
    comparisons = comparison_rows(summary)
    compact = compact_rows(summary)
    gpu_table = [
        {
            "model": row["model"],
            "representation": row["representation"],
            "gpu_s_first20": row["gpu_s_first20"],
            "gpu_saving_vs_raw_first20_pct": row[
                "gpu_saving_vs_raw_first20_pct"
            ],
            "gpu_s_all_questions": row["gpu_s_all"],
            "gpu_saving_vs_raw_all_questions_pct": row[
                "gpu_saving_vs_raw_all_pct"
            ],
        }
        for row in compact
    ]
    token_table = [
        {
            "model": row["model"],
            "representation": row["representation"],
            "reported_input_k_all_questions": row["reported_input_k_all"],
            "cached_input_k_all_questions": row["cached_input_k_all"],
            "cached_input_pct_all_questions": row["cached_input_pct_all"],
            "uncached_input_k_all_questions": row["uncached_input_k_all"],
            "output_k_all_questions": row["output_k_all"],
        }
        for row in compact
    ]

    expected_per_video_rows = len(MODEL_CONFIGS) * len(REPRESENTATION_ORDER) * 2 * len(by_video)
    if len(per_video) != expected_per_video_rows:
        raise ValueError(
            f"per-video output has {len(per_video)} rows, expected {expected_per_video_rows}"
        )
    if len(summary) != len(MODEL_CONFIGS) * len(REPRESENTATION_ORDER) * 2:
        raise ValueError(f"summary has unexpected row count: {len(summary)}")
    if any(int(row["questions_total"]) != 1020 for row in summary if row["horizon"] == "first20"):
        raise ValueError("first20 summaries do not all cover exactly 51 x 20 questions")
    if any(
        int(row["questions_total"]) != MANIFEST_EXPECTED_QUESTIONS
        for row in summary
        if row["horizon"] == "all_questions"
    ):
        raise ValueError("all-question summaries do not all cover all 1,871 questions")

    for row in per_video:
        reported = float(row["reported_input_tokens"])
        separated = float(row["cached_input_tokens"]) + float(row["uncached_input_tokens"])
        if abs(reported - separated) > 1e-7:
            raise ValueError("reported input does not equal cached plus uncached input")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_video_horizon_costs.csv", per_video)
    write_csv(out_dir / "summary_by_horizon.csv", summary)
    write_csv(out_dir / "all_vs_first20.csv", comparisons)
    write_csv(out_dir / "paper_table_compact.csv", compact)
    write_csv(out_dir / "paper_gpu_table.csv", gpu_table)
    write_csv(out_dir / "paper_token_table.csv", token_table)
    write_csv(out_dir / "m20_reconstruction_checks.csv", m20_checks)
    write_csv(out_dir / "sources.csv", source_records)

    summary_index = {
        (row["model"], row["representation"], row["horizon"]): row for row in summary
    }
    gpu_rankings: dict[str, dict[str, list[str]]] = {}
    gpu_savings_vs_raw: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODEL_CONFIGS:
        gpu_rankings[model] = {}
        gpu_savings_vs_raw[model] = {}
        for horizon in ("first20", "all_questions"):
            ordered = sorted(
                REPRESENTATION_ORDER,
                key=lambda representation: float(
                    summary_index[(model, representation, horizon)][
                        "gpu_seconds_mean_per_video"
                    ]
                ),
            )
            gpu_rankings[model][horizon] = ordered
            raw_gpu = float(
                summary_index[(model, "RepeatedFrames", horizon)][
                    "gpu_seconds_mean_per_video"
                ]
            )
            gpu_savings_vs_raw[model][horizon] = {
                representation: 100.0
                * (
                    raw_gpu
                    - float(
                        summary_index[(model, representation, horizon)][
                            "gpu_seconds_mean_per_video"
                        ]
                    )
                )
                / raw_gpu
                for representation in REPRESENTATION_ORDER
            }

    payload = {
        "accepted": True,
        "purpose": "direct ledger summation at first 20 and all available HD-EPIC questions",
        "manifest": str(manifest_path),
        "videos": len(by_video),
        "questions": sum(len(qids) for qids in by_video.values()),
        "questions_per_video": {
            "mean": fmean(len(qids) for qids in by_video.values()),
            "min": min(len(qids) for qids in by_video.values()),
            "max": max(len(qids) for qids in by_video.values()),
        },
        "models": list(MODEL_CONFIGS),
        "representations": list(REPRESENTATION_ORDER),
        "horizons": {
            "first20": "the exact first 20 manifest-ordered questions per video",
            "all_questions": "every manifest question for that video (variable video horizon)",
        },
        "resource_policy": "sum every selected build and answer ledger event; no accuracy filtering",
        "traceable_error_events_included": sum(
            int(source["traceable_error_events_included"]) for source in source_records
        ),
        "internvl3_strict_ledgers": args.internvl3_strict_ledgers,
        "timemap_hd_cost_tag": args.timemap_hd_cost_tag,
        "token_definitions": {
            "reported_input_tokens": "input_text_tokens + visual_tokens; no cross-model visual/text split",
            "cached_input_tokens": "engine-reported cached_prompt_tokens (language-model prefix cache)",
            "uncached_input_tokens": "per-event reported_input_tokens - cached_input_tokens",
            "warning": (
                "uncached_input_tokens does not capture skipped visual-encoder work in "
                "VisualTokens; use measured GPU seconds for computation"
            ),
            "output_tokens": "output_text_tokens, reported separately and never added to input",
        },
        "direct_summation": True,
        "estimated_from_average_per_question": False,
        "m20_reconstruction_max_abs_difference": max(
            abs(float(row["difference"])) for row in m20_checks
        ),
        "gpu_rankings_low_to_high": gpu_rankings,
        "gpu_savings_vs_repeated_frames_percent": gpu_savings_vs_raw,
        "sources": source_records,
        "files": {
            "per_video": "per_video_horizon_costs.csv",
            "summary": "summary_by_horizon.csv",
            "comparison": "all_vs_first20.csv",
            "compact_table": "paper_table_compact.csv",
            "gpu_table": "paper_gpu_table.csv",
            "token_table": "paper_token_table.csv",
            "m20_checks": "m20_reconstruction_checks.csv",
            "sources": "sources.csv",
        },
    }
    (out_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "acceptance.txt").write_text(
        "accepted=true\n"
        "passed=true\n"
        "dataset=HD-EPIC-51\n"
        f"videos={len(by_video)}\n"
        f"questions={sum(len(qids) for qids in by_video.values())}\n"
        f"models={len(MODEL_CONFIGS)}\n"
        f"representations={len(REPRESENTATION_ORDER)}\n"
        f"internvl3_strict_ledgers={str(args.internvl3_strict_ledgers).lower()}\n"
        f"timemap_hd_cost_tag={args.timemap_hd_cost_tag or ''}\n"
        "horizons=first20,all_questions\n"
        "direct_ledger_summation=true\n"
        "estimated_from_average_per_question=false\n"
        "m20_reconstruction=true\n"
        "token_inputs_and_outputs_separate=true\n",
        encoding="utf-8",
    )
    print(json.dumps({"accepted": True, "out_dir": str(out_dir), "rows": len(per_video)}))


if __name__ == "__main__":
    main()
