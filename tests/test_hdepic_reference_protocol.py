from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import run_controlled_memory_policy_cgbench as runner


def test_attach_frame_times_preserves_writer_facts() -> None:
    repository = (
        "Frame 0 | entities: cup | action/state: still | visible text: none\n"
        "Frame 1 | entities: drawer | action/state: opens | visible text: none"
    )

    timed = runner.attach_frame_times(repository, [0.0, 4044.933])

    assert timed.splitlines() == [
        "Frame 0 | time: 0.000s | entities: cup | action/state: still | visible text: none",
        "Frame 1 | time: 4044.933s | entities: drawer | action/state: opens | visible text: none",
    ]
    assert "entities: cup" in timed
    assert "action/state: opens" in timed


def test_attach_frame_times_requires_one_time_per_repository_line() -> None:
    repository = (
        "Frame 0 | entities: cup | action/state: still | visible text: none\n"
        "Frame 1 | entities: drawer | action/state: opens | visible text: none"
    )

    with pytest.raises(ValueError, match="schema violation"):
        runner.attach_frame_times(repository, [0.0])


def test_bbox_reference_uses_question_time_not_choice_times() -> None:
    row = {
        "question": (
            "Where was <BBOX 100 200 300 400> at "
            "<TIME 00:14:46.8 video 1> moved?"
        ),
        "choices": ["<TIME 00:01:00.0 video 1>", "elsewhere"],
    }

    assert runner.official_bbox_reference_times(row) == [886.8]


def test_hdepic_text_normalization_does_not_enable_visual_reference_by_default() -> None:
    row = {
        "question": "What is <BBOX 100 200 300 400> at <TIME 00:00:06.5 video 1>?",
        "choices": ["one", "two"],
    }

    normalized = runner.prepare_hdepic_row(row)
    diagnostic = runner.prepare_hdepic_row(row, include_bbox_reference=True)

    assert normalized["_official_bbox_reference_times"] == []
    assert diagnostic["_official_bbox_reference_times"] == [6.5]
    assert normalized["_official_bbox_question"] == row["question"]
    assert "<TIME" not in normalized["question"]
    assert "<BBOX" not in normalized["question"]


def test_draw_official_bbox_marks_a_qid_specific_copy(tmp_path: Path) -> None:
    path = tmp_path / "reference.jpg"
    image = np.full((140, 140, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    question = "Find <BBOX 140.8 281.6 704.0 844.8> at <TIME 00:00:01.0 video 1>."

    runner.draw_official_bbox(str(path), question)

    marked = cv2.imread(str(path))
    assert marked is not None
    # BBOX coordinates are y1, x1, y2, x2 on the official 1408-square canvas.
    # The upper-left corner therefore maps to approximately (x=28, y=14).
    assert int(marked[14, 28, 2]) > int(marked[14, 28, 1]) + 80


def test_reference_content_labels_query_frame_as_outside_saved_64() -> None:
    content = runner.reference_content(["/tmp/reference.jpg"], [6.5])

    assert len(content) == 3
    assert "not part of the saved source frames" in content[0]["text"]
    assert content[1]["text"] == "Question Reference 0 at 6.500s:"
