#!/usr/bin/env python3
"""Validate that Visual reuses encoder outputs without LM prefix-cache hits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def visual_cache_receipt(
    manifest: list[dict[str, Any]],
    results: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    num_frames: int,
    video_input: bool,
) -> dict[str, Any]:
    """Build the controlled-run Visual cache acceptance receipt.

    Each video's write request may execute the encoder once per image, or once
    for the whole video when native video input is used.  Official reference
    images attached to answer requests are charged separately.  More encoder
    executions than that upper bound indicate a cache miss or rerun.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    if not manifest:
        raise ValueError("manifest must be non-empty")
    if not results:
        raise ValueError("results must be non-empty")

    def index_questions(
        rows: list[dict[str, Any]], *, source: str
    ) -> dict[str, str]:
        indexed: dict[str, str] = {}
        for row in rows:
            qid = str(row.get("qid") or "").strip()
            if not qid:
                raise ValueError(f"{source} row has no qid")
            if qid in indexed:
                raise ValueError(f"{source} contains duplicate qid: {qid}")
            video = str(row.get("video_uid") or "").strip()
            if not video:
                raise ValueError(f"{source} row has no video_uid: qid={qid}")
            indexed[qid] = video
        return indexed

    manifest_questions = index_questions(manifest, source="manifest")
    result_questions = index_questions(results, source="results")
    if set(result_questions) != set(manifest_questions):
        missing = sorted(set(manifest_questions) - set(result_questions))
        extra = sorted(set(result_questions) - set(manifest_questions))
        raise ValueError(
            f"manifest/result qid sets differ: missing={missing} extra={extra}"
        )
    mismatched_videos = [
        qid
        for qid, video in result_questions.items()
        if video != manifest_questions[qid]
    ]
    if mismatched_videos:
        raise ValueError(
            "manifest/result video_uid differs for qids: "
            f"{sorted(mismatched_videos)}"
        )

    if not traces:
        raise ValueError("trace must be non-empty")
    encoded_items = 0
    for row in traces:
        kind_missing = "kind" not in row
        kind = row.get("kind")
        encoder_event = kind_missing or kind == "mm_encoder_execution"
        if "item_count" in row and not encoder_event:
            raise ValueError(
                "unexpected trace kind contains item_count: "
                f"kind={kind!r}"
            )
        if encoder_event:
            encoded_items += int(row.get("item_count") or 0)

    videos = set(manifest_questions.values())
    reference_items = sum(
        len(row.get("official_reference_times") or []) for row in results
    )
    memory_items = len(videos) if video_input else len(videos) * num_frames
    expected_items = memory_items + reference_items
    if encoded_items > expected_items:
        raise ValueError(
            f"visual-token cache miss: encoded_items={encoded_items} "
            f"maximum={expected_items}"
        )
    if encoded_items < int(expected_items * 0.90):
        raise ValueError(
            f"unexpectedly low encoder count: encoded_items={encoded_items} "
            f"maximum={expected_items}"
        )

    cached_prompt = [int(row.get("cached_prompt_tokens") or 0) for row in results]
    if any(cached_prompt):
        raise ValueError(
            "visual-token condition unexpectedly reused language-model prefix tokens: "
            f"max={max(cached_prompt)}"
        )

    return {
        "videos": len(videos),
        "questions": len(results),
        "frames_per_video": num_frames,
        "visual_input_format": "video" if video_input else "images",
        "encoder_item_executions": encoded_items,
        "expected_encoder_item_executions": expected_items,
        "memory_encoder_item_executions": memory_items,
        "query_reference_encoder_item_executions": reference_items,
        "content_deduplicated_items": expected_items - encoded_items,
        "cached_prompt_tokens_max": max(cached_prompt, default=0),
        "accepted": True,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one Visual run from its manifest, results, and vLLM "
            "multimodal-encoder trace."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--video-input",
        action="store_true",
        help="Count one native-video encoder item per video instead of one per frame.",
    )
    args = parser.parse_args()

    try:
        receipt = visual_cache_receipt(
            read_jsonl(args.manifest),
            read_jsonl(args.results),
            read_jsonl(args.trace),
            num_frames=args.num_frames,
            video_input=args.video_input,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"visual cache acceptance failed: {exc}") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
