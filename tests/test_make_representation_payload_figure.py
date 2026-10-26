from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_representation_payload_figure.py"
MODELS = ("Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path.parent / "acceptance.txt").write_text("accepted=true\n", encoding="utf-8")


def test_synthetic_two_panel_smoke(tmp_path: Path) -> None:
    payload_rows = []
    for dataset, videos, dataset_scale in (("CG", 50, 1.0), ("HD-EPIC", 51, 1.2)):
        for model_index, model in enumerate(MODELS, start=1):
            for representation, base in (
                ("Text", 8_000),
                ("Visual", 120_000_000),
                ("KV", 1_000_000_000),
                ("State total", 1_120_000_000),
            ):
                median = base * model_index * dataset_scale
                payload_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "representation": representation,
                    "videos": videos,
                    "unit": "bytes_per_video",
                    "mean_bytes": median,
                    "median_bytes": median,
                    "min_bytes": median * 0.8,
                    "max_bytes": median * 1.2,
                })
    payload_csv = tmp_path / "payload" / "representation_payload.csv"
    write_csv(payload_csv, payload_rows)

    cg_rows = []
    cg_names = {
        "Raw": "Repeated frames",
        "Text": "Frame-level captions",
        "Visual": "Visual tokens",
        "KV": "KV cache",
    }
    cg_costs = {"Raw": 100.0, "Text": 70.0, "Visual": 85.0, "KV": 40.0}
    for model_index, model in enumerate(MODELS, start=1):
        for representation, name in cg_names.items():
            cg_rows.append({
                "dataset": "CG MQ-50",
                "model": model,
                "representation": name,
                "horizon": 6,
                "total_gpu_s_per_video": cg_costs[representation] * model_index,
            })
    cg_csv = tmp_path / "cg" / "profiles.csv"
    write_csv(cg_csv, cg_rows)

    hd_rows = []
    hd_names = {"Raw": "RepeatedFrames", "Text": "FrameText", "Visual": "VisualTokens", "KV": "KVCache"}
    hd_savings = {"Raw": 0.0, "Text": 80.0, "Visual": 20.0, "KV": 65.0}
    for model in MODELS:
        for representation, name in hd_names.items():
            hd_rows.append({
                "model": model,
                "representation": name,
                "gpu_saving_vs_raw_all_questions_pct": hd_savings[representation],
            })
    hd_csv = tmp_path / "hd" / "paper_gpu_table.csv"
    write_csv(hd_csv, hd_rows)

    accuracy_rows = []
    accuracy_names = {
        "Text": "Frame-level captions",
        "Visual": "Visual-token cache",
        "KV": "KV cache",
    }
    differences = {"Text": -8.0, "Visual": 0.0, "KV": 1.0}
    for dataset, questions in (("CG MQ-50", 300), ("HD-EPIC-51", 1871)):
        for model in MODELS:
            for representation, name in accuracy_names.items():
                accuracy_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "representation": name,
                    "questions": questions,
                    "difference_pp": differences[representation],
                })
    accuracy_csv = tmp_path / "accuracy" / "paired_accuracy_ci.csv"
    write_csv(accuracy_csv, accuracy_rows)

    out_dir = tmp_path / "figure"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--payload-csv",
            str(payload_csv),
            "--cg-profile-csv",
            str(cg_csv),
            "--hd-all-cost-csv",
            str(hd_csv),
            "--accuracy-csv",
            str(accuracy_csv),
            "--out-dir",
            str(out_dir),
            "--stem",
            "synthetic_payload",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    reported = json.loads(result.stdout)
    assert reported["accepted"] is True
    assert reported["points"] == 18
    assert set(reported["panels"]) == {"CG", "HD"}
    for suffix in ("cg", "hd"):
        assert (out_dir / f"synthetic_payload_{suffix}.pdf").stat().st_size > 1_000
        assert not (out_dir / f"synthetic_payload_{suffix}.png").exists()
    assert not (out_dir / "synthetic_payload.pdf").exists()
    assert not (out_dir / "synthetic_payload.png").exists()
    with (out_dir / "synthetic_payload.data.csv").open(newline="") as handle:
        data_rows = list(csv.DictReader(handle))
    assert len(data_rows) == 18
    assert {row["dataset"] for row in data_rows} == {"CG", "HD"}
    assert {row["representation"] for row in data_rows} == {
        "Text",
        "Visual",
        "State total",
    }
    qwen_cg_state = next(
        row
        for row in data_rows
        if row["dataset"] == "CG"
        and row["model"] == "Qwen3-VL-8B"
        and row["representation"] == "State total"
    )
    assert float(qwen_cg_state["median_bytes"]) == 1_120_000_000
    manifest = json.loads((out_dir / "synthetic_payload.sources.json").read_text())
    assert manifest["accepted"] is True
    assert manifest["panels"] == {"CG": "m=6", "HD": "all questions"}
    assert set(manifest["outputs"]) == {"CG", "HD", "data"}


def test_missing_result_fails_explicitly(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--payload-csv",
            str(tmp_path / "missing.csv"),
            "--cg-profile-csv",
            str(tmp_path / "also-missing.csv"),
            "--hd-all-cost-csv",
            str(tmp_path / "also-missing-hd.csv"),
            "--accuracy-csv",
            str(tmp_path / "also-missing-accuracy.csv"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "required result does not exist" in result.stderr


def test_kv_only_payload_is_not_treated_as_full_state(tmp_path: Path) -> None:
    rows = []
    for dataset, videos in (("CG", 50), ("HD-EPIC", 51)):
        for model in MODELS:
            for representation, median in (
                ("Text", 8_000),
                ("Visual", 120_000_000),
                ("KV", 1_000_000_000),
            ):
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "representation": representation,
                    "videos": videos,
                    "unit": "bytes_per_video",
                    "median_bytes": median,
                    "min_bytes": median * 0.8,
                    "max_bytes": median * 1.2,
                })
    payload_csv = tmp_path / "payload" / "representation_payload.csv"
    write_csv(payload_csv, rows)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--payload-csv",
            str(payload_csv),
            "--cg-profile-csv",
            str(tmp_path / "not-needed-cg.csv"),
            "--hd-all-cost-csv",
            str(tmp_path / "not-needed-hd.csv"),
            "--accuracy-csv",
            str(tmp_path / "not-needed-accuracy.csv"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "KV-only rows cannot be plotted as the retained state" in result.stderr
