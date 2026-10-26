#!/usr/bin/env python3
"""Validate frozen HD-EPIC videos and add runtime metadata to the MQ10 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def video_metadata(path: Path) -> tuple[int, float, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
        ok, image = cap.read()
    finally:
        cap.release()
    if frames < 2 or fps <= 0 or width <= 0 or height <= 0 or not ok or image is None:
        raise ValueError(f"invalid video metadata or unreadable midpoint: {path}")
    return frames, (frames - 1) / fps, fps, width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=50)
    parser.add_argument("--questions-per-video", type=int, default=10)
    parser.add_argument(
        "--allow-variable-questions",
        action="store_true",
        help="Allow different question counts per video, using --questions-per-video as a minimum.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_uid"])].append(row)
    if len(by_video) != args.expected_videos:
        raise SystemExit(f"expected {args.expected_videos} videos, found {len(by_video)}")
    if args.allow_variable_questions:
        wrong_counts = {
            video: len(items) for video, items in by_video.items()
            if len(items) < args.questions_per_video
        }
    else:
        wrong_counts = {
            video: len(items) for video, items in by_video.items()
            if len(items) != args.questions_per_video
        }
    if wrong_counts:
        raise SystemExit(f"questions-per-video mismatch: {wrong_counts}")

    metadata: dict[str, dict] = {}
    for video_id, items in sorted(by_video.items()):
        relative = Path(str(items[0].get("video_relpath") or items[0].get("video_path") or ""))
        path = relative if relative.is_absolute() else args.video_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        frames, duration, fps, width, height = video_metadata(path)
        metadata[video_id] = {
            "video_path": str(path.resolve()),
            "total_frames": frames,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "bytes": path.stat().st_size,
        }

    output = []
    for row in rows:
        item = dict(row)
        item.update({key: value for key, value in metadata[str(row["video_uid"])].items() if key in {"video_path", "total_frames", "duration"}})
        output.append(item)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    durations = [item["duration"] for item in metadata.values()]
    question_counts = [len(items) for items in by_video.values()]
    summary = {
        "status": "validated",
        "videos": len(metadata),
        "questions": len(output),
        "questions_per_video": None if args.allow_variable_questions else args.questions_per_video,
        "minimum_questions_per_video": args.questions_per_video,
        "question_count_per_video": {
            "min": min(question_counts),
            "median": statistics.median(question_counts),
            "mean": statistics.mean(question_counts),
            "max": max(question_counts),
        },
        "participants": dict(sorted(Counter(video.split("-", 1)[0] for video in metadata).items())),
        "duration_seconds": {"min": min(durations), "mean": sum(durations) / len(durations), "max": max(durations)},
        "total_video_bytes": sum(item["bytes"] for item in metadata.values()),
        "manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_manifest": str(args.manifest),
        "video_root": str(args.video_root),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
