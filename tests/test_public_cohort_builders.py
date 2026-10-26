import collections
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_cgbench_and_build_mq as cg_builder
import build_hdepic_mq10 as hd_builder
import materialize_hdepic_mq10 as hd_materializer


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_main(monkeypatch, module, arguments: list[object]) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__), *(str(value) for value in arguments)])
    module.main()


def _cg_annotations(video_count: int = 9, questions_per_video: int = 8) -> list[dict]:
    rows = []
    for video_index in range(video_count):
        video_uid = f"synthetic-video-{video_index:02d}"
        for question_index in range(questions_per_video):
            rows.append(
                {
                    "qid": f"{video_uid}-q{question_index:02d}",
                    "video_uid": video_uid,
                    "question": f"Question {question_index} for {video_uid}?",
                    "answer": "choice 0",
                    "choices": ["choice 0", "choice 1", "choice 2", "choice 3"],
                    "right_answer": "A",
                    "clue_intervals": [[1.0, 2.0]],
                    "duration": 900.0,
                    "domain": "synthetic",
                    "sub_category": "Spatial Perception",
                }
            )
    return rows


def _run_cg_builder(monkeypatch, annotations: Path, out_dir: Path, seed: int) -> None:
    _run_main(
        monkeypatch,
        cg_builder,
        [
            "--annotations",
            annotations,
            "--out-dir",
            out_dir,
            "--seed",
            seed,
            "--sizes",
            3,
            7,
            "--questions-per-video",
            6,
            "--allow-missing-media",
        ],
    )


def test_cg_selection_is_seeded_nested_and_has_fixed_six_question_order(
    tmp_path: Path, monkeypatch
):
    annotations = tmp_path / "cg_annotations.json"
    annotations.write_text(json.dumps(_cg_annotations()), encoding="utf-8")

    first = tmp_path / "first"
    repeated = tmp_path / "repeated"
    alternate = tmp_path / "alternate"
    _run_cg_builder(monkeypatch, annotations, first, seed=20260722)
    _run_cg_builder(monkeypatch, annotations, repeated, seed=20260722)
    _run_cg_builder(monkeypatch, annotations, alternate, seed=20260723)

    first_videos_3 = _read_jsonl(first / "CG-Bench-MQ-3.videos.jsonl")
    first_videos_7 = _read_jsonl(first / "CG-Bench-MQ-7.videos.jsonl")
    repeated_videos_7 = _read_jsonl(repeated / "CG-Bench-MQ-7.videos.jsonl")
    alternate_videos_7 = _read_jsonl(alternate / "CG-Bench-MQ-7.videos.jsonl")

    first_ids_3 = [row["video_uid"] for row in first_videos_3]
    first_ids_7 = [row["video_uid"] for row in first_videos_7]
    assert first_ids_3 == first_ids_7[:3]
    assert first_videos_7 == repeated_videos_7
    assert first_videos_7 != alternate_videos_7

    questions_3 = _read_jsonl(first / "CG-Bench-MQ-3.questions.jsonl")
    questions_7 = _read_jsonl(first / "CG-Bench-MQ-7.questions.jsonl")
    repeated_questions_7 = _read_jsonl(repeated / "CG-Bench-MQ-7.questions.jsonl")
    assert questions_7 == repeated_questions_7

    counts = collections.Counter(row["video_uid"] for row in questions_7)
    assert counts == {video_uid: 6 for video_uid in first_ids_7}

    def question_order(rows: list[dict], video_uid: str) -> list[str]:
        selected = [row for row in rows if row["video_uid"] == video_uid]
        assert [row["within_video_question_order"] for row in selected] == list(range(6))
        return [row["qid"] for row in selected]

    videos_7_by_id = {row["video_uid"]: row for row in first_videos_7}
    for video_uid in first_ids_3:
        order = question_order(questions_3, video_uid)
        assert order == question_order(questions_7, video_uid)
        assert order == videos_7_by_id[video_uid]["question_ids_in_fixed_order"]

    provenance = json.loads(
        (first / "CG-Bench-MQ-7.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["seed"] == 20260722
    assert provenance["questions_per_video"] == 6


def test_cg_runnable_cohort_requires_ffprobe(tmp_path: Path, monkeypatch) -> None:
    annotations = tmp_path / "cg_annotations.json"
    annotations.write_text(json.dumps(_cg_annotations()), encoding="utf-8")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    monkeypatch.setattr(cg_builder.shutil, "which", lambda _command: None)

    with pytest.raises(SystemExit, match="ffprobe is required"):
        _run_main(
            monkeypatch,
            cg_builder,
            [
                "--annotations",
                annotations,
                "--out-dir",
                tmp_path / "out",
                "--video-root",
                video_root,
                "--sizes",
                1,
            ],
        )


def _hd_task_payload(video_ids: list[str], task: str) -> dict[str, dict]:
    return {
        f"{task}-{video_id}": {
            "inputs": {"video": {"id": video_id}},
            "question": f"{task} question for {video_id}?",
            "choices": ["first", "second", "third"],
            "correct_idx": 1,
        }
        for video_id in video_ids
    }


def _run_hd_builder(monkeypatch, annotations: Path, output_root: Path) -> None:
    _run_main(
        monkeypatch,
        hd_builder,
        [
            "--annotations",
            annotations,
            "--out",
            output_root / "manifest.jsonl",
            "--summary",
            output_root / "summary.json",
            "--video-ids-out",
            output_root / "video_ids.txt",
            "--videos",
            51,
            "--questions-per-video",
            2,
        ],
    )


def test_hd_selection_is_deterministic_for_51_videos(tmp_path: Path, monkeypatch):
    annotations = tmp_path / "hd_annotations"
    annotations.mkdir()
    video_ids = [
        f"P{participant:02d}-{index:03d}"
        for participant in range(1, 4)
        for index in range(17)
    ]
    for task in ("action", "object"):
        (annotations / f"{task}.json").write_text(
            json.dumps(_hd_task_payload(video_ids, task)), encoding="utf-8"
        )

    first = tmp_path / "first_hd"
    repeated = tmp_path / "repeated_hd"
    _run_hd_builder(monkeypatch, annotations, first)
    _run_hd_builder(monkeypatch, annotations, repeated)

    first_manifest = _read_jsonl(first / "manifest.jsonl")
    repeated_manifest = _read_jsonl(repeated / "manifest.jsonl")
    assert first_manifest == repeated_manifest
    assert (first / "video_ids.txt").read_bytes() == (repeated / "video_ids.txt").read_bytes()

    selected_ids = (first / "video_ids.txt").read_text(encoding="utf-8").splitlines()
    assert len(selected_ids) == 51
    assert selected_ids[:6] == [
        "P01-000",
        "P02-000",
        "P03-000",
        "P01-001",
        "P02-001",
        "P03-001",
    ]

    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    for row in first_manifest:
        by_video[row["video_uid"]].append(row)
    assert set(by_video) == set(selected_ids)
    assert all(len(rows) == 2 for rows in by_video.values())
    assert all(
        [row["within_video_question_order"] for row in rows] == [1, 2]
        for rows in by_video.values()
    )
    assert all(
        [row["question_type"] for row in rows] == ["action", "object"]
        for rows in by_video.values()
    )

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["videos"] == 51
    assert summary["questions"] == 102
    assert summary["participants"] == {"P01": 17, "P02": 17, "P03": 17}


def test_hd_materializer_preserves_order_without_reading_real_video(
    tmp_path: Path, monkeypatch
):
    video_root = tmp_path / "media"
    video_root.mkdir()
    rows = []
    for video_uid in ("P01-001", "P02-001"):
        relative_path = Path("videos") / f"{video_uid}.mp4"
        path = video_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-placeholder")
        for order in (1, 2):
            rows.append(
                {
                    "qid": f"{video_uid}-q{order}",
                    "video_uid": video_uid,
                    "video_relpath": str(relative_path),
                    "within_video_question_order": order,
                }
            )

    manifest = tmp_path / "annotation_only.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        hd_materializer,
        "video_metadata",
        lambda _path: (101, 10.0, 10.0, 640, 480),
    )

    output = tmp_path / "materialized.jsonl"
    summary_path = tmp_path / "materialized_summary.json"
    _run_main(
        monkeypatch,
        hd_materializer,
        [
            "--manifest",
            manifest,
            "--video-root",
            video_root,
            "--out",
            output,
            "--summary",
            summary_path,
            "--expected-videos",
            2,
            "--questions-per-video",
            2,
        ],
    )

    materialized = _read_jsonl(output)
    assert [row["qid"] for row in materialized] == [row["qid"] for row in rows]
    assert all(row["total_frames"] == 101 for row in materialized)
    assert all(row["duration"] == 10.0 for row in materialized)
    assert all(Path(row["video_path"]).is_absolute() for row in materialized)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "validated"
    assert summary["videos"] == 2
    assert summary["questions"] == 4
    assert summary["questions_per_video"] == 2
