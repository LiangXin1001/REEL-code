import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_strict_question_type_heatmaps as heatmaps


def test_extreme_selection_keeps_four_lowest_then_four_highest():
    panel = {
        "question_types": [f"type_{index}" for index in range(10)],
        "labels": [f"Type {index}" for index in range(10)],
        "values": np.asarray([[float(index)] * 6 for index in range(10)]),
    }

    selected = heatmaps.select_extreme_rows(panel)

    assert selected["question_types"] == [
        "type_0", "type_1", "type_2", "type_3",
        "type_6", "type_7", "type_8", "type_9",
    ]
    assert selected["groups"] == ["lowest"] * 4 + ["highest"] * 4


def test_asset_suffix_is_safe_and_explicit():
    assert heatmaps.normalized_suffix("timemap_final_v2_20260826") == (
        "_timemap_final_v2_20260826"
    )
    with pytest.raises(ValueError):
        heatmaps.normalized_suffix("../formal")
