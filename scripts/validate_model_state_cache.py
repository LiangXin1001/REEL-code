#!/usr/bin/env python3
"""Validate agreement between per-answer and server prefix-cache counters."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFIX_HITS = re.compile(
    r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


def shared_prefix_audit(
    results: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    *,
    max_shortfall_tokens: int,
) -> dict[str, Any]:
    """Verify that each answer reused nearly the complete warmed prefix.

    A merely positive cache counter can pass when only the first block was
    reused.  For Text+Prefix-State we instead compare every answer with the
    exact per-video warmup prompt recorded in the ledger.  A small shortfall is
    allowed for block granularity and the divergent warmup/answer suffix.
    """
    if max_shortfall_tokens < 0:
        raise ValueError("max_shortfall_tokens must be non-negative")
    warmups: dict[str, list[int]] = defaultdict(list)
    for row in ledger_rows:
        if (
            str(row.get("question_id") or "") == "__video_write__"
            and row.get("state_storage_unit") == "vllm_multimodal_prefix_tokens"
        ):
            video = str(row.get("video_id") or "").strip()
            if not video:
                raise ValueError("prefix warmup ledger event has no video_id")
            prompt_tokens = int(row.get("input_text_tokens") or 0) + int(
                row.get("visual_tokens") or 0
            )
            if prompt_tokens <= 0:
                raise ValueError(f"prefix warmup has no prompt tokens for video={video}")
            warmups[video].append(prompt_tokens)
    duplicate_warmups = {
        video: values for video, values in warmups.items() if len(values) != 1
    }
    if duplicate_warmups:
        raise ValueError(
            f"expected one prefix warmup per video, observed={duplicate_warmups}"
        )
    if not warmups:
        raise ValueError("strict shared-prefix audit found no prefix warmup events")

    per_answer: list[dict[str, Any]] = []
    for row in results:
        video = str(row.get("video_uid") or "").strip()
        if video not in warmups:
            raise ValueError(f"answer has no matching prefix warmup: video={video}")
        request_count = int(row.get("answer_request_count") or 1)
        if request_count != 1:
            raise ValueError(
                "strict shared-prefix audit requires one answer request per question; "
                f"video={video} qid={row.get('qid')} requests={request_count}"
            )
        warmed = warmups[video][0]
        cached = int(row.get("cached_prompt_tokens") or 0)
        shortfall = max(warmed - cached, 0)
        per_answer.append({
            "video_uid": video,
            "qid": str(row.get("qid") or ""),
            "warmup_prompt_tokens": warmed,
            "cached_prompt_tokens": cached,
            "shared_prefix_shortfall_tokens": shortfall,
            "passed": shortfall <= max_shortfall_tokens,
        })
    result_videos = {row["video_uid"] for row in per_answer}
    if result_videos != set(warmups):
        raise ValueError(
            "prefix warmup/result video coverage differs: "
            f"warmups={sorted(warmups)} results={sorted(result_videos)}"
        )
    failing = [row for row in per_answer if not row["passed"]]
    fractions = [
        row["cached_prompt_tokens"] / row["warmup_prompt_tokens"]
        for row in per_answer
    ]
    return {
        "enabled": True,
        "max_allowed_shared_prefix_shortfall_tokens": max_shortfall_tokens,
        "warmup_videos": len(warmups),
        "answers_checked": len(per_answer),
        "largest_shared_prefix_shortfall_tokens": max(
            (row["shared_prefix_shortfall_tokens"] for row in per_answer),
            default=0,
        ),
        "minimum_cached_to_warmup_prompt_ratio": min(fractions, default=0.0),
        "failed_answers": len(failing),
        "failure_examples": failing[:20],
        "passed": bool(per_answer) and not failing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-shared-prefix-shortfall-tokens",
        type=int,
        help=(
            "Optional strict Text+Prefix-State check. Every answer must reuse all but at "
            "most this many tokens of its exact per-video warmup prompt. A value of 128 "
            "allows cache-block/template suffix differences while rejecting image-only hits."
        ),
    )
    args = parser.parse_args()

    metrics = args.metrics.read_text(encoding="utf-8")
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ledger_rows = [
        json.loads(line)
        for line in args.ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metric_hits = sum(float(match.group(1)) for match in PREFIX_HITS.finditer(metrics))
    per_answer = [int(row.get("cached_prompt_tokens") or 0) for row in rows]
    reported_hits = sum(per_answer)
    ledger_hits = sum(int(row.get("cached_prompt_tokens") or 0) for row in ledger_rows)
    passed = bool(rows) and all(value > 0 for value in per_answer)
    passed = passed and metric_hits > 0 and abs(metric_hits - ledger_hits) <= 0.5
    strict_prefix = None
    if args.max_shared_prefix_shortfall_tokens is not None:
        strict_prefix = shared_prefix_audit(
            rows,
            ledger_rows,
            max_shortfall_tokens=args.max_shared_prefix_shortfall_tokens,
        )
        passed = passed and bool(strict_prefix["passed"])
    payload = {
        "questions": len(rows),
        "all_answers_report_cache_hits": bool(rows) and all(value > 0 for value in per_answer),
        "prefix_cache_hit_tokens": metric_hits,
        "answer_cached_prompt_tokens": reported_hits,
        "all_request_cached_prompt_tokens": ledger_hits,
        "non_answer_cached_prompt_tokens": ledger_hits - reported_hits,
        "counter_difference": metric_hits - ledger_hits,
        "strict_shared_prefix_audit": strict_prefix,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"model-state cache acceptance failed: {payload}")


if __name__ == "__main__":
    main()
