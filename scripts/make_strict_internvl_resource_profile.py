#!/usr/bin/env python3
"""Create auditable CG and HD resource-table rows for strict InternVL3 runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFECYCLE_ROOT = (
    ROOT / "outputs/tables/rq2_strict_lifecycle_inputs_internvl3_final_20260824"
)
DEFAULT_HD_DIR = ROOT / "outputs/tables/hd_all_questions_costs_strict_lifecycle_20260824"
DEFAULT_OUT = ROOT / "outputs/tables/rq2_resource_profiles_original3_strict_lifecycle_20260824"
LEGACY_PROFILE = ROOT / "outputs/tables/rq1_frame_text_profiles_20260820/profiles.csv"
LEGACY_PROFILE_JSON = ROOT / "outputs/tables/rq1_frame_text_profiles_20260820/profiles.json"
CONDITIONS = {
    "Frames": "raw",
    "Text": "text",
    "Visual": "visual",
    "KV": "state",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_accepted(path: Path) -> None:
    if not path.is_file() or "accepted=true" not in path.read_text(encoding="utf-8").splitlines():
        raise ValueError(f"source is not accepted: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cg_rows(lifecycle_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    compute: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for representation, condition in CONDITIONS.items():
        lifecycle_dir = lifecycle_root / "cg" / condition / "lifecycle"
        lifecycle_path = lifecycle_dir / "lifecycle_per_video.json"
        provenance_path = lifecycle_dir / "lifecycle_provenance.json"
        rows = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        endpoint = [row for row in rows if int(row["m"]) == 6]
        if len(endpoint) != 50 or len({row["video_id"] for row in endpoint}) != 50:
            raise ValueError(f"CG endpoint coverage is not 50 videos: {lifecycle_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        answer_ledger = Path(provenance["answer_ledger"])
        write_ledger = Path(provenance.get("write_ledger") or provenance["answer_ledger"])
        require_accepted(answer_ledger.parent / "acceptance.txt")
        require_accepted(write_ledger.parent / "acceptance.txt")
        build_by_video: dict[str, float] = defaultdict(float)
        for event in read_jsonl(write_ledger):
            if str(event.get("question_id") or "") == "__video_write__":
                build_by_video[str(event["video_id"])] += float(event.get("gpu_seconds") or 0.0)
        video_ids = {str(row["video_id"]) for row in endpoint}
        if representation != "Frames" and set(build_by_video) != video_ids:
            raise ValueError(f"CG build coverage differs for {representation}")
        build = fmean(build_by_video.get(video_id, 0.0) for video_id in video_ids)
        total = fmean(float(row["gpu_seconds"]) for row in endpoint)
        compute.append({
            "dataset": "CG MQ-50",
            "model": "InternVL3-8B",
            "representation": representation,
            "videos": 50,
            "questions_per_video": 6,
            "build_gpu_s_per_video": build,
            "answer_gpu_s_per_question": (total - build) / 6.0,
            "cumulative_gpu_s_per_video": total,
        })
        tokens.append({
            "dataset": "CG MQ-50",
            "model": "InternVL3-8B",
            "representation": representation,
            "videos": 50,
            "questions_per_video": 6,
            "reported_input_k_per_video": fmean(float(row["reported_input_tokens"]) for row in endpoint) / 1000.0,
            "cached_input_k_per_video": fmean(float(row["cached_prompt_tokens"]) for row in endpoint) / 1000.0,
            "uncached_input_k_per_video": fmean(float(row["uncached_prompt_tokens"]) for row in endpoint) / 1000.0,
            "output_k_per_video": fmean(float(row["output_text_tokens"]) for row in endpoint) / 1000.0,
        })
        sources.append({
            "dataset": "CG MQ-50",
            "model": "InternVL3-8B",
            "representation": representation,
            "lifecycle": str(lifecycle_path.resolve()),
            "lifecycle_provenance": str(provenance_path.resolve()),
            "answer_ledger": str(answer_ledger.resolve()),
            "write_ledger": str(write_ledger.resolve()),
        })
    return compute, tokens, sources


def hd_rows(hd_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    require_accepted(hd_dir / "acceptance.txt")
    rows = list(csv.DictReader((hd_dir / "per_video_horizon_costs.csv").open(encoding="utf-8")))
    rows = [row for row in rows if row["model"] == "InternVL3-8B"]
    sources = [
        row for row in csv.DictReader((hd_dir / "sources.csv").open(encoding="utf-8"))
        if row["model"] == "InternVL3-8B"
    ]
    compute: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[(row["representation"], row["horizon"])].append(row)
    raw_all = fmean(float(row["gpu_seconds"]) for row in indexed[("RepeatedFrames", "all_questions")])
    hd_names = {
        "RepeatedFrames": "Frames",
        "FrameText": "Text",
        "VisualTokens": "Visual",
        "KVCache": "KV",
    }
    for source_name, representation in hd_names.items():
        first = indexed[(source_name, "first20")]
        all_rows = indexed[(source_name, "all_questions")]
        if len(first) != 51 or len(all_rows) != 51:
            raise ValueError(f"HD coverage differs for {source_name}")
        build = fmean(float(row["build_gpu_seconds"]) for row in first)
        total20 = fmean(float(row["gpu_seconds"]) for row in first)
        total_all = fmean(float(row["gpu_seconds"]) for row in all_rows)
        compute.append({
            "dataset": "HD-EPIC-51",
            "model": "InternVL3-8B",
            "representation": representation,
            "videos": 51,
            "build_gpu_s_per_video": build,
            "answer_gpu_s_per_question_first20": (total20 - build) / 20.0,
            "cumulative_gpu_s_first20": total20,
            "cumulative_gpu_s_all_questions": total_all,
            "saving_vs_frames_all_percent": 100.0 * (raw_all - total_all) / raw_all,
        })
        tokens.append({
            "dataset": "HD-EPIC-51",
            "model": "InternVL3-8B",
            "representation": representation,
            "videos": 51,
            "questions_mean_per_video": fmean(float(row["questions"]) for row in all_rows),
            "reported_input_k_per_video": fmean(float(row["reported_input_tokens"]) for row in all_rows) / 1000.0,
            "cached_input_k_per_video": fmean(float(row["cached_input_tokens"]) for row in all_rows) / 1000.0,
            "uncached_input_k_per_video": fmean(float(row["uncached_input_tokens"]) for row in all_rows) / 1000.0,
            "output_k_per_video": fmean(float(row["output_tokens"]) for row in all_rows) / 1000.0,
        })
    for row in sources:
        row["dataset"] = "HD-EPIC-51"
    return compute, tokens, sources


def complete_cg_rows(
    strict_compute: list[dict[str, Any]], strict_tokens: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    strict_c = {row["representation"]: row for row in strict_compute}
    strict_t = {row["representation"]: row for row in strict_tokens}
    legacy_names = {
        "Repeated frames": "Frames",
        "Frame-level captions": "Text",
        "Visual tokens": "Visual",
        "KV cache": "KV",
    }
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(LEGACY_PROFILE.open(encoding="utf-8")):
        if row["dataset"] != "CG MQ-50":
            continue
        representation = legacy_names[row["representation"]]
        if row["model"] == "InternVL3-8B":
            compute = strict_c[representation]
            tokens = strict_t[representation]
            source_kind = "strict_20260823_answer_timing"
            build = compute["build_gpu_s_per_video"]
            answer = compute["answer_gpu_s_per_question"]
            total = compute["cumulative_gpu_s_per_video"]
            prompt = tokens["reported_input_k_per_video"]
            hit = tokens["cached_input_k_per_video"]
            miss = tokens["uncached_input_k_per_video"]
            output = tokens["output_k_per_video"]
        else:
            source_kind = "accepted_20260820_profile"
            build = float(row["build_gpu_s_per_video"])
            answer = float(row["answer_gpu_s_per_question"])
            total = float(row["total_gpu_s_per_video"])
            prompt = float(row["total_reported_input_tokens_per_video"]) / 1000.0
            hit = float(row["total_cached_prompt_tokens_per_video"]) / 1000.0
            miss = prompt - hit
            output = float(row["total_output_text_tokens_per_video"]) / 1000.0
        rows.append({
            "dataset": "CG MQ-50",
            "model": row["model"],
            "representation": representation,
            "source_kind": source_kind,
            "videos": 50,
            "questions_per_video": 6,
            "build_gpu_s_per_video": build,
            "answer_gpu_s_per_question": answer,
            "cumulative_gpu_s_per_video": total,
            "reported_input_k_per_video": prompt,
            "cached_input_k_per_video": hit,
            "uncached_input_k_per_video": miss,
            "output_k_per_video": output,
        })
    if len(rows) != 12:
        raise ValueError(f"complete CG profile has {len(rows)} rows, expected 12")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument("--hd-dir", type=Path, default=DEFAULT_HD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    require_accepted(
        ROOT / "outputs/tables/rq2_valid_pairs_cg_internvl3_strict_final_20260824/acceptance.txt"
    )
    cg_compute, cg_tokens, cg_sources = cg_rows(args.lifecycle_root)
    hd_compute, hd_tokens, hd_sources = hd_rows(args.hd_dir)
    complete_cg = complete_cg_rows(cg_compute, cg_tokens)
    legacy_sources = json.loads(LEGACY_PROFILE_JSON.read_text(encoding="utf-8"))["sources"]
    retained_legacy_sources = [
        row for row in legacy_sources
        if row["dataset"] == "CG MQ-50" and row["model"] != "InternVL3-8B"
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "cg_complete_resource_profile.csv", complete_cg)
    write_csv(args.out_dir / "cg_strict_compute_profile.csv", cg_compute)
    write_csv(args.out_dir / "cg_strict_token_profile.csv", cg_tokens)
    write_csv(args.out_dir / "hd_strict_compute_profile.csv", hd_compute)
    write_csv(args.out_dir / "hd_strict_token_profile.csv", hd_tokens)
    write_csv(args.out_dir / "cg_strict_sources.csv", cg_sources)
    write_csv(args.out_dir / "hd_strict_sources.csv", hd_sources)
    write_csv(args.out_dir / "cg_retained_legacy_sources.csv", retained_legacy_sources)
    payload = {
        "accepted": True,
        "model": "InternVL3-8B",
        "resource_policy": "all executed build and answer events; no answer filtering",
        "text_composition": "strict answer ledger plus provenance-recorded accepted write ledger",
        "cg_rows": len(cg_compute),
        "hd_rows": len(hd_compute),
        "complete_cg_rows": len(complete_cg),
        "files": [
            "cg_complete_resource_profile.csv", "cg_strict_compute_profile.csv",
            "cg_strict_token_profile.csv", "hd_strict_compute_profile.csv",
            "hd_strict_token_profile.csv", "cg_strict_sources.csv",
            "hd_strict_sources.csv",
            "cg_retained_legacy_sources.csv",
        ],
    }
    (args.out_dir / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "acceptance.txt").write_text(
        "accepted=true\nmodels=Qwen3-VL-8B,InternVL3-8B,Gemma-3-12B-IT\n"
        "strict_replaced_model=InternVL3-8B\ndatasets=CG_MQ50,HD_EPIC_51\n"
        "representations=Frames,Text,Visual,KV\nresource_policy=all_executed_events\n"
        "text_answer_and_write_ledgers_separate=true\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
