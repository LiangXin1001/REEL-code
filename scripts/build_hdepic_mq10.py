#!/usr/bin/env python3
"""Build an annotation-only HD-EPIC repeated-question cohort."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def round_robin_videos(eligible: list[str], count: int) -> list[str]:
    by_participant: dict[str, list[str]] = collections.defaultdict(list)
    for video_id in eligible:
        by_participant[video_id.split("-", 1)[0]].append(video_id)
    selected: list[str] = []
    participants = sorted(by_participant)
    while len(selected) < count:
        progressed = False
        for participant in participants:
            if by_participant[participant] and len(selected) < count:
                selected.append(by_participant[participant].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def round_robin_questions(rows: list[dict], count: int) -> list[dict]:
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_task[row["question_type"]].append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: row["qid"])
    selected: list[dict] = []
    tasks = sorted(by_task)
    while len(selected) < count:
        progressed = False
        for task in tasks:
            if by_task[task] and len(selected) < count:
                selected.append(by_task[task].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--video-ids-out", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=50)
    parser.add_argument("--questions-per-video", type=int, default=10)
    parser.add_argument(
        "--include-all-eligible-questions",
        action="store_true",
        help=(
            "Use --questions-per-video only as the eligibility threshold, then "
            "retain every eligible full-video question for each selected video."
        ),
    )
    args = parser.parse_args()

    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted(args.annotations.glob("*.json")):
        task = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        for qid, source in payload.items():
            inputs = source["inputs"]
            if len(inputs) != 1:
                continue
            visual_input = next(iter(inputs.values()))
            if "start_time" in visual_input or "end_time" in visual_input:
                continue
            video_id = str(visual_input["id"])
            correct_idx = int(source["correct_idx"])
            choices = [str(value) for value in source["choices"]]
            by_video[video_id].append(
                {
                    "qid": str(qid),
                    "video_uid": video_id,
                    "video_relpath": f"HD-EPIC/Videos/{video_id[:3]}/{video_id}.mp4",
                    "question": str(source["question"]),
                    "choices": choices,
                    "right_answer": chr(65 + correct_idx),
                    "correct_idx": correct_idx,
                    "question_type": task,
                    "source_dataset": "HD-EPIC-VQA",
                    "official_inputs": inputs,
                    "evidence_intervals": [],
                    "evidence_status": "not uniformly available for full-video prototypes",
                }
            )

    eligible = sorted(
        video_id
        for video_id, rows in by_video.items()
        if len(rows) >= args.questions_per_video
    )
    selected_videos = round_robin_videos(eligible, args.videos)
    if len(selected_videos) < args.videos:
        raise SystemExit(f"only {len(selected_videos)} eligible videos")

    output_rows: list[dict] = []
    for video_id in selected_videos:
        requested = len(by_video[video_id]) if args.include_all_eligible_questions else args.questions_per_video
        rows = round_robin_questions(by_video[video_id], requested)
        if len(rows) != requested:
            raise RuntimeError(f"selection underflow for {video_id}")
        for order, row in enumerate(rows, start=1):
            output_rows.append(row | {"within_video_question_order": order})

    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    task_counts = collections.Counter(row["question_type"] for row in output_rows)
    participant_counts = collections.Counter(video_id[:3] for video_id in selected_videos)
    question_counts = [len(by_video[video_id]) for video_id in selected_videos]
    if args.include_all_eligible_questions:
        question_policy = (
            f"all questions retained after requiring at least {args.questions_per_video} "
            "eligible full-video questions per video"
        )
        questions_per_video: int | None = None
    else:
        question_policy = (
            f"exactly {args.questions_per_video} questions selected round-robin "
            "across sorted official prototypes"
        )
        questions_per_video = args.questions_per_video
    summary = {
        "status": "annotation_only; video files not yet validated",
        "videos": len(selected_videos),
        "questions": len(output_rows),
        "questions_per_video": questions_per_video,
        "minimum_questions_per_video": args.questions_per_video,
        "question_count_per_video": {
            "min": min(question_counts),
            "median": statistics.median(question_counts),
            "mean": statistics.mean(question_counts),
            "max": max(question_counts),
        },
        "eligible_full_video_count": len(eligible),
        "participants": dict(sorted(participant_counts.items())),
        "question_types": dict(sorted(task_counts.items())),
        "selected_video_ids": selected_videos,
        "manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selection_rule": (
            f"Single-video official full-video inputs only; {args.videos} videos selected "
            "round-robin across participants from lexicographically sorted eligible "
            f"IDs; {question_policy}. Question order is deterministic round-robin across "
            "sorted official prototypes."
        ),
        "evidence_policy": (
            "Answer accuracy and resource profiles are valid for the complete cohort. "
            "Evidence coverage requires a separately declared prototype-specific mapping."
        ),
    }
    for path in (args.out, args.summary, args.video_ids_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.video_ids_out.write_text("\n".join(selected_videos) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
