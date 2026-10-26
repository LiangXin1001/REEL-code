import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_glm_tariff_paper_assets import (  # noqa: E402
    CURRENT_HIT_MULTIPLIER,
    CURRENT_OUTPUT_MULTIPLIER,
    crossing_assets,
    endpoint_token_rows,
    heatmap_assets,
    threshold_assets,
    tier_flags,
    winner_for,
)


INPUT = ROOT / "tests/fixtures/glm_tariff"


def test_current_official_prices_choose_text_at_all_six_endpoints():
    endpoint = endpoint_token_rows(INPUT)
    for dataset in ("CG MQ-50", "HD-EPIC-51"):
        for model in ("Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT"):
            assert winner_for(
                endpoint,
                dataset,
                model,
                CURRENT_HIT_MULTIPLIER,
                CURRENT_OUTPUT_MULTIPLIER,
            ) == "FrameText"


def test_crossing_table_uses_both_tiers_and_new_hd_qwen_crossing():
    rows, latex = crossing_assets(INPUT)
    hd_qwen = next(row for row in rows if row["dataset"] == "HD" and row["model"] == "Qwen3-VL-8B")
    assert hd_qwen["uses_both_official_input_tiers"] is True
    assert hd_qwen["no_cache_FrameText_crossing"] == 1
    assert hd_qwen["observed_hits_FrameText_crossing"] == 1
    assert hd_qwen["observed_hits_KVCache_crossing"] == 2
    cg_intern = next(
        row
        for row in rows
        if row["dataset"] == "CG" and row["model"] == "InternVL3-8B"
    )
    assert cg_intern["uses_both_official_input_tiers"] is True
    assert r"InternVL$^{\ast}$" in latex
    assert "both official input-length tiers" in latex
    assert "extrapolat" not in latex


def test_price_sensitivity_boundary_identifies_nearest_cg_gemma_flip():
    endpoint = endpoint_token_rows(INPUT)
    rows, _ = threshold_assets(endpoint, tier_flags(INPUT))
    cg_gemma = next(
        row
        for row in rows
        if row["dataset"] == "CG" and row["model"] == "Gemma-3-12B-IT"
    )
    assert cg_gemma[
        "hit_price_multiplier_boundary_at_current_output_price"
    ] == pytest.approx(0.8256977473)
    assert cg_gemma[
        "output_price_multiplier_boundary_at_current_hit_price"
    ] == pytest.approx(1.5646200709)


def test_heatmap_marks_current_tariff_as_six_text_winners():
    rows, matrix, _ = heatmap_assets(endpoint_token_rows(INPUT))
    current = next(row for row in rows if row["is_current_glm_tariff"])
    assert current["frame_text_winners"] == 6
    assert current["kv_cache_winners"] == 0
    assert matrix.shape == (6, 6)


def test_cli_writes_paper_assets(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_glm_tariff_paper_assets.py"),
            "--input-dir",
            str(INPUT),
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
    )
    accepted = json.loads((tmp_path / "acceptance.json").read_text(encoding="utf-8"))
    assert accepted["accepted"] is True
    assert (tmp_path / "glm_tariff_sensitivity_heatmap.pdf").stat().st_size > 1000
    assert (tmp_path / "monetary_crossing_table.tex").is_file()
