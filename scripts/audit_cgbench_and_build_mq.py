#!/usr/bin/env python3
"""Audit official CG-Bench annotations and deterministically build MQ manifests.

The script deliberately separates annotation validity from media validity.  A
manifest is never marked runnable unless the corresponding video has passed an
ffprobe check and, when requested, a subtitle file has been found.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov")
SUBTITLE_SUFFIXES = (".srt", ".vtt", ".ass", ".ssa", ".json")
REQUIRED_FIELDS = (
    "qid",
    "video_uid",
    "question",
    "answer",
    "choices",
    "right_answer",
    "clue_intervals",
    "duration",
    "domain",
    "sub_category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--subtitle-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 300])
    parser.add_argument("--questions-per-video", type=int, default=6)
    parser.add_argument("--require-subtitles", action="store_true")
    parser.add_argument(
        "--allow-missing-media",
        action="store_true",
        help="Write annotation-only provisional manifests when media are absent.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def index_media(root: Path | None, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if root is None or not root.is_dir():
        return {}
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.stem, path)
    return index


def ffprobe_duration(path: Path) -> tuple[bool, float | None, str | None]:
    if shutil.which("ffprobe") is None:
        return False, None, "ffprobe_not_found"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return False, None, completed.stderr.strip() or "ffprobe_failed"
    try:
        return True, float(completed.stdout.strip()), None
    except ValueError:
        return False, None, "ffprobe_invalid_duration"


def qtype(sub_category: str) -> str:
    if sub_category == "Hallucination":
        return "hallucination"
    if sub_category.endswith("Perception"):
        return "perception"
    if sub_category.endswith("Cognition"):
        return "reasoning"
    return "other"


def duration_bin(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 20:
        return "10_20m"
    if minutes < 30:
        return "20_30m"
    if minutes < 45:
        return "30_45m"
    return "45_80m"


def valid_question(row: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "", [])]
    if missing:
        return missing
    if not isinstance(row["choices"], list) or not row["choices"]:
        missing.append("choices_not_nonempty_list")
    if not isinstance(row["clue_intervals"], list) or not row["clue_intervals"]:
        missing.append("clue_intervals_not_nonempty_list")
    else:
        for interval in row["clue_intervals"]:
            if not (
                isinstance(interval, list)
                and len(interval) == 2
                and all(isinstance(value, (int, float)) for value in interval)
                and interval[0] <= interval[1]
            ):
                missing.append("invalid_clue_interval")
                break
    return missing


@dataclass(frozen=True)
class VideoRecord:
    video_uid: str
    duration_s: float
    domain: str
    qtype: str
    question_count: int
    valid_question_count: int
    has_video: bool
    video_playable: bool
    has_subtitle: bool
    video_path: str | None
    subtitle_path: str | None
    media_error: str | None


def main() -> None:
    args = parse_args()
    if (
        args.video_root is not None
        and not args.allow_missing_media
        and shutil.which("ffprobe") is None
    ):
        raise SystemExit(
            "ffprobe is required to build a runnable CG-Bench cohort; "
            "install FFmpeg or use --allow-missing-media only for a provisional "
            "annotation-only manifest"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("CG-Bench annotations must be a JSON list")

    video_index = index_media(args.video_root, VIDEO_SUFFIXES)
    subtitle_index = index_media(args.subtitle_root, SUBTITLE_SUFFIXES)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_rows: list[dict[str, Any]] = []
    normalized_questions: list[dict[str, Any]] = []

    for row in questions:
        problems = valid_question(row)
        video_uid = str(row.get("video_uid", ""))
        if problems:
            invalid_rows.append({"qid": row.get("qid"), "video_uid": video_uid, "problems": problems})
            continue
        normalized = dict(row)
        normalized["qid"] = str(normalized["qid"])
        normalized["video_uid"] = video_uid
        normalized["qtype"] = qtype(str(normalized["sub_category"]))
        normalized_questions.append(normalized)
        by_video[video_uid].append(normalized)

    video_records: list[VideoRecord] = []
    for video_uid in sorted(by_video):
        rows = by_video[video_uid]
        annotated_duration = float(rows[0]["duration"])
        video_path = video_index.get(video_uid)
        subtitle_path = subtitle_index.get(video_uid)
        playable = False
        observed_duration = None
        media_error = "video_missing"
        if video_path is not None:
            playable, observed_duration, media_error = ffprobe_duration(video_path)
            if playable and observed_duration is not None:
                if abs(observed_duration - annotated_duration) > 3.0:
                    media_error = f"duration_mismatch:{observed_duration:.3f}!={annotated_duration:.3f}"
                    playable = False
        qtypes = Counter(row["qtype"] for row in rows)
        primary_qtype = qtypes.most_common(1)[0][0]
        video_records.append(
            VideoRecord(
                video_uid=video_uid,
                duration_s=annotated_duration,
                domain=str(rows[0]["domain"]),
                qtype=primary_qtype,
                question_count=len(rows),
                valid_question_count=len(rows),
                has_video=video_path is not None,
                video_playable=playable,
                has_subtitle=subtitle_path is not None,
                video_path=str(video_path) if video_path else None,
                subtitle_path=str(subtitle_path) if subtitle_path else None,
                media_error=media_error,
            )
        )

    usable = [
        record
        for record in video_records
        if record.valid_question_count >= args.questions_per_video
        and (record.video_playable or args.allow_missing_media)
        and (record.has_subtitle or not args.require_subtitles or args.allow_missing_media)
    ]

    summary = {
        "source": str(args.annotations.resolve()),
        "source_sha256": sha256(args.annotations),
        "seed": args.seed,
        "questions_total": len(questions),
        "questions_valid": len(normalized_questions),
        "questions_invalid": len(invalid_rows),
        "videos_total": len(video_records),
        "videos_with_video_file": sum(record.has_video for record in video_records),
        "videos_playable": sum(record.video_playable for record in video_records),
        "videos_with_subtitle": sum(record.has_subtitle for record in video_records),
        "eligible_annotation_only": sum(
            record.valid_question_count >= args.questions_per_video for record in video_records
        ),
        "eligible_under_requested_media_rule": len(usable),
        "require_subtitles": args.require_subtitles,
        "allow_missing_media": args.allow_missing_media,
        "question_type_counts": dict(sorted(Counter(row["qtype"] for row in normalized_questions).items())),
        "subcategory_counts": dict(sorted(Counter(row["sub_category"] for row in normalized_questions).items())),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in normalized_questions).items())),
        "clue_coverage": {
            "questions_with_nonempty_clues": sum(bool(row["clue_intervals"]) for row in normalized_questions),
            "mean_clue_windows_per_question": sum(len(row["clue_intervals"]) for row in normalized_questions)
            / max(1, len(normalized_questions)),
            "questions_with_out_of_range_clue": sum(
                any(start < 0 or end > float(row["duration"]) for start, end in row["clue_intervals"])
                for row in normalized_questions
            ),
        },
    }
    write_json(args.out_dir / "cgbench_audit_summary.json", summary)
    write_json(args.out_dir / "cgbench_invalid_questions.json", invalid_rows)

    with (args.out_dir / "cgbench_per_video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(VideoRecord.__dataclass_fields__))
        writer.writeheader()
        for record in video_records:
            writer.writerow(record.__dict__)

    if any(size > len(usable) for size in args.sizes):
        raise RuntimeError(
            f"Requested MQ size exceeds eligible videos: requested={args.sizes}, eligible={len(usable)}. "
            "Use --allow-missing-media only for provisional annotation-only manifests."
        )

    # Nested, deterministic selections: MQ-10 is a subset of MQ-50, itself a subset of MQ-300.
    grouped: dict[tuple[str, str, str], list[VideoRecord]] = defaultdict(list)
    for record in usable:
        grouped[(duration_bin(record.duration_s), record.qtype, str(min(record.question_count, 15)))].append(record)
    rng = random.Random(args.seed)
    for records in grouped.values():
        rng.shuffle(records)
    # The round-robin order must not be lexicographic: otherwise MQ-10 would
    # systematically overselect the first alphabetic strata.  It is shuffled
    # once from the recorded seed and then kept fixed for all nested manifests.
    ordered_groups = sorted(grouped)
    rng.shuffle(ordered_groups)

    selected: list[VideoRecord] = []
    round_index = 0
    while len(selected) < max(args.sizes):
        progressed = False
        for key in ordered_groups:
            bucket = grouped[key]
            if round_index < len(bucket) and len(selected) < max(args.sizes):
                selected.append(bucket[round_index])
                progressed = True
        if not progressed:
            raise RuntimeError("Stratified selection exhausted before reaching requested size")
        round_index += 1

    for size in sorted(args.sizes):
        chosen = selected[:size]
        video_rows: list[dict[str, Any]] = []
        question_rows: list[dict[str, Any]] = []
        for video_order, record in enumerate(chosen):
            candidate_questions = list(by_video[record.video_uid])
            local_rng = random.Random(f"{args.seed}:{record.video_uid}")
            local_rng.shuffle(candidate_questions)
            chosen_questions = candidate_questions[: args.questions_per_video]
            record_dict = record.__dict__.copy()
            record_dict.update(
                {
                    "manifest_name": f"CG-Bench-MQ-{size}",
                    "video_order": video_order,
                    "question_ids_in_fixed_order": [row["qid"] for row in chosen_questions],
                    "runnable": bool(record.video_playable and (record.has_subtitle or not args.require_subtitles)),
                    "provisional_reason": None
                    if record.video_playable and (record.has_subtitle or not args.require_subtitles)
                    else "annotation_only_until_official_media_is_downloaded_and_verified",
                }
            )
            video_rows.append(record_dict)
            for question_order, question in enumerate(chosen_questions):
                question_row = dict(question)
                question_row.update(
                    {
                        "manifest_name": f"CG-Bench-MQ-{size}",
                        "video_order": video_order,
                        "within_video_question_order": question_order,
                        "runnable": record_dict["runnable"],
                        "video_path": record.video_path,
                        "subtitle_path": record.subtitle_path,
                    }
                )
                question_rows.append(question_row)
        prefix = args.out_dir / f"CG-Bench-MQ-{size}"
        write_jsonl(prefix.with_suffix(".videos.jsonl"), video_rows)
        write_jsonl(prefix.with_suffix(".questions.jsonl"), question_rows)
        write_json(
            prefix.with_suffix(".provenance.json"),
            {
                "manifest_name": f"CG-Bench-MQ-{size}",
                "selection_rule": "nested deterministic round-robin over duration_bin x majority_qtype x capped_question_count",
                "seed": args.seed,
                "questions_per_video": args.questions_per_video,
                "source_sha256": summary["source_sha256"],
                "media_rule": "video must pass ffprobe; subtitles required only when --require-subtitles",
                "provisional": not all(row["runnable"] for row in video_rows),
            },
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
