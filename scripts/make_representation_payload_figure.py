#!/usr/bin/env python3
"""Plot retained-state size against endpoint compute saving and accuracy.

The figure consumes only accepted, derived CSVs.  It deliberately keeps the
three quantities separate: the x axis is retained bytes, the y axis is GPU
time saved relative to repeated frame input, and color is the paired accuracy
difference.  The plotted State size must include both retained visual encoder
outputs and reached KV pages; a KV-only payload is never substituted for it.
No missing dataset/model/representation cell is imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reel-paper-mpl")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

from paper_plot_style import EDGE_WIDTH, style_colorbar, style_xy_axis


MODELS = (
    ("Qwen3-VL-8B", "Qwen"),
    ("InternVL3-8B", "InternVL"),
    ("Gemma-3-12B-IT", "Gemma"),
)
MODEL_LABELS = dict(MODELS)
DATASETS = ("CG", "HD")
METHOD_REPRESENTATIONS = ("Text", "Visual", "KV")
PLOT_REPRESENTATIONS = ("Text", "Visual", "State total")
METHOD_FOR_PLOT = {"Text": "Text", "Visual": "Visual", "State total": "KV"}
MARKERS = {"Text": "o", "Visual": "^", "State total": "s"}
GRID = "#D9E1E8"
MUTED = "#536273"

# Each PDF is included in one half of a single WACV column.  Keep the panel
# wide enough that the black plotting frame reads as a landscape rectangle;
# the previous 1.72 x 2.40 in canvas made both panels look tall and narrow.
PANEL_FIGSIZE = (1.95, 1.85)
PANEL_MARGINS = {
    "left": 0.25,
    "right": 0.98,
    "bottom": 0.31,
    "top": 0.84,
}


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def canonical_dataset(value: str) -> str:
    aliases = {
        "cg": "CG",
        "cgmq50": "CG",
        "hd": "HD",
        "hdepic": "HD",
        "hdepic51": "HD",
    }
    key = normalized(value)
    if key not in aliases:
        raise ValueError(f"unrecognized dataset label: {value!r}")
    return aliases[key]


def canonical_representation(value: str) -> str:
    aliases = {
        "frames": "Raw",
        "repeatedframes": "Raw",
        "raw": "Raw",
        "text": "Text",
        "frametext": "Text",
        "frameleveltext": "Text",
        "framelevelcaptions": "Text",
        "visual": "Visual",
        "visualtokens": "Visual",
        "visualtokencache": "Visual",
        "kv": "KV",
        "kvcache": "KV",
        "state": "State total",
        "statetotal": "State total",
        "kvstate": "State total",
        "visualpluskv": "State total",
        "combinedvisualpluskv": "State total",
    }
    key = normalized(value)
    if key not in aliases:
        raise ValueError(f"unrecognized representation label: {value!r}")
    return aliases[key]


def finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} is not finite: {value!r}")
    return parsed


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required result does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def require_fields(rows: list[dict[str, str]], fields: set[str], path: Path) -> None:
    missing = fields - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")


def require_accepted(path: Path) -> None:
    """Require an accepted sidecar next to a derived CSV.

    Existing paper tables use ``accepted=true`` in acceptance.txt, while the
    representation-payload validator writes ``accepted: true`` to JSON and a
    leading PASS to text.  Supporting both preserves the original audit
    formats without weakening the check.
    """
    json_path = path.parent / "acceptance.json"
    text_path = path.parent / "acceptance.txt"
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"invalid acceptance JSON: {json_path}") from exc
        if payload.get("accepted") is not True:
            raise ValueError(f"refusing unaccepted source: {path}")
        return
    if text_path.is_file():
        lines = [
            line.strip()
            for line in text_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fields = {
            key.strip().lower(): value.strip().lower()
            for line in lines
            if "=" in line
            for key, value in [line.split("=", 1)]
        }
        if fields.get("accepted") == "true" or (lines and lines[0] == "PASS"):
            return
        raise ValueError(f"refusing unaccepted source: {path}")
    raise ValueError(
        f"missing acceptance sidecar next to {path}; expected acceptance.json or acceptance.txt"
    )


def expected_keys(representations: tuple[str, ...]) -> set[tuple[str, str, str]]:
    return {
        (dataset, model, representation)
        for dataset in DATASETS
        for model, _ in MODELS
        for representation in representations
    }


def read_payload(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = read_rows(path)
    require_accepted(path)
    require_fields(
        rows,
        {
            "dataset",
            "model",
            "representation",
            "videos",
            "unit",
            "median_bytes",
            "min_bytes",
            "max_bytes",
        },
        path,
    )
    available: dict[tuple[str, str, str], dict[str, float]] = {}
    expected_videos = {"CG": 50, "HD": 51}
    for row in rows:
        dataset = canonical_dataset(row["dataset"])
        model = row["model"]
        representation = canonical_representation(row["representation"])
        if model not in MODEL_LABELS or representation not in {
            "Text",
            "Visual",
            "KV",
            "State total",
        }:
            raise ValueError(
                f"unexpected payload cell: {(dataset, model, representation)}"
            )
        key = (dataset, model, representation)
        if key in available:
            raise ValueError(f"duplicate payload cell: {key}")
        if row["unit"] != "bytes_per_video":
            raise ValueError(f"{key} has unit={row['unit']!r}, expected bytes_per_video")
        videos = int(finite_float(row["videos"], f"{key}.videos"))
        if videos != expected_videos[dataset]:
            raise ValueError(
                f"{key} has {videos} videos, expected {expected_videos[dataset]}"
            )
        minimum = finite_float(row["min_bytes"], f"{key}.min_bytes")
        median = finite_float(row["median_bytes"], f"{key}.median_bytes")
        maximum = finite_float(row["max_bytes"], f"{key}.max_bytes")
        if not 0 < minimum <= median <= maximum:
            raise ValueError(
                f"invalid payload order for {key}: min={minimum}, median={median}, max={maximum}"
            )
        available[key] = {
            "min_bytes": minimum,
            "median_bytes": median,
            "max_bytes": maximum,
        }

    selected: dict[tuple[str, str, str], dict[str, float]] = {}
    missing_state: list[tuple[str, str]] = []
    for dataset in DATASETS:
        for model, _ in MODELS:
            for representation in ("Text", "Visual"):
                key = (dataset, model, representation)
                if key in available:
                    selected[key] = available[key]
            state_key = (dataset, model, "State total")
            kv_key = (dataset, model, "KV")
            if state_key not in available:
                missing_state.append((dataset, model))
                continue
            selected[state_key] = available[state_key]
            visual = available.get((dataset, model, "Visual"))
            state = available[state_key]
            if visual and any(
                state[f"{statistic}_bytes"] < visual[f"{statistic}_bytes"]
                for statistic in ("min", "median", "max")
            ):
                raise ValueError(
                    "State total is smaller than its Visual component for "
                    f"{(dataset, model)}"
                )
            kv = available.get(kv_key)
            if kv and any(
                state[f"{statistic}_bytes"] < kv[f"{statistic}_bytes"]
                for statistic in ("min", "median", "max")
            ):
                raise ValueError(
                    "State total is smaller than its KV-only component for "
                    f"{(dataset, model)}"
                )

    if missing_state:
        kv_only = [pair for pair in missing_state if (*pair, "KV") in available]
        raise ValueError(
            "payload table is missing State total rows for "
            f"{missing_state}. State total must include Visual backing storage plus "
            "reached KV pages; KV-only rows cannot be plotted as the retained state. "
            f"KV-only rows found for {kv_only}."
        )

    expected = expected_keys(PLOT_REPRESENTATIONS)
    if set(selected) != expected:
        raise ValueError(
            "payload coverage mismatch: "
            f"missing={sorted(expected - set(selected))}, "
            f"extra={sorted(set(selected) - expected)}"
        )
    allowed = expected | expected_keys(("KV",))
    if set(available) - allowed:
        raise ValueError(f"unexpected payload rows: {sorted(set(available)-allowed)}")
    return selected


def read_cg_savings(path: Path) -> dict[tuple[str, str], float]:
    rows = read_rows(path)
    require_accepted(path)
    fields = set(rows[0])
    common = {"dataset", "model", "representation"}
    legacy = {"horizon", "total_gpu_s_per_video"}
    strict = {"questions_per_video", "cumulative_gpu_s_per_video"}
    if legacy <= fields:
        horizon_field = "horizon"
        cost_field = "total_gpu_s_per_video"
    elif strict <= fields:
        horizon_field = "questions_per_video"
        cost_field = "cumulative_gpu_s_per_video"
    else:
        raise ValueError(
            f"{path} must provide either {sorted(legacy)} or {sorted(strict)}"
        )
    require_fields(rows, common | {horizon_field, cost_field}, path)
    costs: dict[tuple[str, str], float] = {}
    for row in rows:
        if canonical_dataset(row["dataset"]) != "CG":
            continue
        model = row["model"]
        if model not in MODEL_LABELS:
            raise ValueError(f"unexpected CG model: {model!r}")
        representation = canonical_representation(row["representation"])
        if int(finite_float(row[horizon_field], "CG.horizon")) != 6:
            raise ValueError(f"CG endpoint must be m=6: {row}")
        key = (model, representation)
        if key in costs:
            raise ValueError(f"duplicate CG cost cell: {key}")
        cost = finite_float(row[cost_field], f"CG/{key}.GPU-s")
        if cost <= 0:
            raise ValueError(f"CG/{key} GPU time must be positive")
        costs[key] = cost
    expected = {
        (model, representation)
        for model, _ in MODELS
        for representation in ("Raw",) + METHOD_REPRESENTATIONS
    }
    if set(costs) != expected:
        raise ValueError(
            f"CG cost coverage mismatch: missing={sorted(expected-set(costs))}, "
            f"extra={sorted(set(costs)-expected)}"
        )
    savings: dict[tuple[str, str], float] = {}
    for model, _ in MODELS:
        raw = costs[(model, "Raw")]
        for representation in METHOD_REPRESENTATIONS:
            savings[(model, representation)] = 100.0 * (
                raw - costs[(model, representation)]
            ) / raw
    return savings


def read_hd_savings(path: Path) -> dict[tuple[str, str], float]:
    rows = read_rows(path)
    require_accepted(path)
    require_fields(rows, {"model", "representation"}, path)
    fields = set(rows[0])
    direct_field = "gpu_saving_vs_raw_all_questions_pct"
    cost_field = "all_gpu_seconds_mean_per_video"
    if direct_field not in fields and cost_field not in fields:
        raise ValueError(
            f"{path} must contain {direct_field!r} or {cost_field!r}"
        )

    values: dict[tuple[str, str], float] = {}
    for row in rows:
        model = row["model"]
        if model not in MODEL_LABELS:
            # The accepted all-question table may contain the three expansion
            # backbones.  This figure deliberately keeps the original
            # three-model payload matrix, just as read_accuracy does below.
            continue
        representation = canonical_representation(row["representation"])
        key = (model, representation)
        if key in values:
            raise ValueError(f"duplicate HD cost cell: {key}")
        field = direct_field if direct_field in fields else cost_field
        value = finite_float(row[field], f"HD/{key}.{field}")
        if field == cost_field and value <= 0:
            raise ValueError(f"HD/{key} GPU time must be positive")
        values[key] = value
    expected = {
        (model, representation)
        for model, _ in MODELS
        for representation in ("Raw",) + METHOD_REPRESENTATIONS
    }
    if set(values) != expected:
        raise ValueError(
            f"HD cost coverage mismatch: missing={sorted(expected-set(values))}, "
            f"extra={sorted(set(values)-expected)}"
        )

    savings: dict[tuple[str, str], float] = {}
    for model, _ in MODELS:
        if direct_field in fields:
            if abs(values[(model, "Raw")]) > 0.11:
                raise ValueError(f"HD Raw saving is not zero for {model}")
            for representation in METHOD_REPRESENTATIONS:
                savings[(model, representation)] = values[(model, representation)]
        else:
            raw = values[(model, "Raw")]
            for representation in METHOD_REPRESENTATIONS:
                savings[(model, representation)] = 100.0 * (
                    raw - values[(model, representation)]
                ) / raw
    return savings


def read_accuracy(path: Path) -> dict[tuple[str, str, str], float]:
    rows = read_rows(path)
    require_accepted(path)
    fields = set(rows[0])
    legacy_fields = {"dataset", "model", "representation", "questions", "difference_pp"}
    strict_fields = {
        "dataset",
        "model",
        "representation",
        "invalid_answer_policy",
        "total_questions",
        "net_delta",
    }
    if not (legacy_fields <= fields or strict_fields <= fields):
        raise ValueError(
            f"{path} must provide either {sorted(legacy_fields)} or "
            f"{sorted(strict_fields)}"
        )
    strict_endpoint = strict_fields <= fields
    indexed: dict[tuple[str, str, str], float] = {}
    expected_questions = {"CG": 300, "HD": 1871}
    for row in rows:
        if strict_endpoint and row["invalid_answer_policy"] != "paired_valid":
            continue
        dataset = canonical_dataset(row["dataset"])
        model = row["model"]
        representation = canonical_representation(row["representation"])
        # The strict endpoint file also contains the three expansion backbones;
        # this figure intentionally retains the original three-backbone matrix.
        if model not in MODEL_LABELS:
            continue
        if representation not in METHOD_REPRESENTATIONS:
            raise ValueError(
                f"unexpected accuracy cell: {(dataset, model, representation)}"
            )
        key = (dataset, model, representation)
        if key in indexed:
            raise ValueError(f"duplicate accuracy cell: {key}")
        question_field = "total_questions" if strict_endpoint else "questions"
        questions = int(finite_float(row[question_field], f"{key}.{question_field}"))
        if questions != expected_questions[dataset]:
            raise ValueError(
                f"{key} has {questions} questions, expected full workload "
                f"of {expected_questions[dataset]}"
            )
        if strict_endpoint:
            indexed[key] = 100.0 * finite_float(row["net_delta"], f"{key}.net_delta")
        else:
            indexed[key] = finite_float(row["difference_pp"], f"{key}.difference_pp")
    expected = expected_keys(METHOD_REPRESENTATIONS)
    if set(indexed) != expected:
        raise ValueError(
            "accuracy coverage mismatch: "
            f"missing={sorted(expected-set(indexed))}, "
            f"extra={sorted(set(indexed)-expected)}"
        )
    return indexed


def byte_tick(value: float, _position: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g}GB"
    if value >= 1e6:
        return f"{value / 1e6:g}MB"
    if value >= 1e3:
        return f"{value / 1e3:g}KB"
    return f"{value:g} B"


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_points(
    payload: dict[tuple[str, str, str], dict[str, float]],
    cg_savings: dict[tuple[str, str], float],
    hd_savings: dict[tuple[str, str], float],
    accuracy: dict[tuple[str, str, str], float],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    saving_sources = {"CG": cg_savings, "HD": hd_savings}
    for dataset in DATASETS:
        for model, model_label in MODELS:
            for representation in PLOT_REPRESENTATIONS:
                key = (dataset, model, representation)
                method = METHOD_FOR_PLOT[representation]
                points.append({
                    "dataset": dataset,
                    "model": model,
                    "model_label": model_label,
                    "representation": representation,
                    **payload[key],
                    "gpu_saving_vs_raw_pct": saving_sources[dataset][
                        (model, method)
                    ],
                    "difference_pp": accuracy[(dataset, model, method)],
                })
    if len(points) != 18:
        raise AssertionError(f"internal error: expected 18 points, got {len(points)}")
    return points


def plot_panel(
    points: list[dict[str, Any]],
    dataset: str,
    out_dir: Path,
    stem: str,
    norm: TwoSlopeNorm,
    cmap: Any,
    x_limits: tuple[float, float],
) -> Path:
    """Render one title-free panel for composition with LaTeX subcaption."""

    panel = [row for row in points if row["dataset"] == dataset]
    # Each panel is placed at roughly half of a WACV column.  The landscape
    # canvas and wider axes box keep the pair compact when composed side by
    # side, while leaving dedicated bands for the legend and colorbar.
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.8, zorder=0)
    for row in panel:
        color = cmap(norm(float(row["difference_pp"])))
        y = float(row["gpu_saving_vs_raw_pct"])
        ax.hlines(
            y,
            float(row["min_bytes"]),
            float(row["max_bytes"]),
            color="#667885",
            linewidth=0.9,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            float(row["median_bytes"]),
            y,
            marker=MARKERS[str(row["representation"])],
            s=48,
            color=color,
            edgecolor="#475866",
            linewidth=EDGE_WIDTH,
            zorder=3,
        )
        model_label = str(row["model_label"])
        label_on_left = float(row["median_bytes"]) >= 1e9
        ax.annotate(
            model_label,
            (float(row["median_bytes"]), y),
            xytext=(-3, -4 if model_label == "InternVL" else 3)
            if label_on_left
            else (3, -4 if model_label == "InternVL" else 3),
            textcoords="offset points",
            fontsize=6.0,
            fontweight="bold",
            color="#26323D",
            ha="right" if label_on_left else "left",
            va="top" if model_label == "InternVL" else "bottom",
        )

    y_values = [float(row["gpu_saving_vs_raw_pct"]) for row in panel]
    span = max(y_values) - min(y_values)
    padding = max(5.0, 0.12 * max(span, 1.0))
    ax.set_ylim(min(y_values + [0.0]) - padding, max(y_values + [0.0]) + padding)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xscale("log")
    ax.set_xlim(*x_limits)
    ax.xaxis.set_major_locator(FixedLocator([1e4, 1e6, 1e8, 1e10]))
    ax.xaxis.set_major_formatter(FuncFormatter(byte_tick))
    ax.set_xlabel("Retained bytes/video (log)", fontsize=8.2, color="#111111")
    ax.set_ylabel("GPU time saved (%)", fontsize=8.2, color="#111111")
    style_xy_axis(ax, tick_size=7.6)
    ax.tick_params(length=2.8)

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MARKERS[representation],
            markersize=6.2,
            markerfacecolor="#AAB7C2",
            markeredgecolor="#475866",
            markeredgewidth=EDGE_WIDTH,
            label="Prefix State" if representation == "State total" else representation,
        )
        for representation in PLOT_REPRESENTATIONS
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        fontsize=7.2,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        handletextpad=0.3,
        columnspacing=0.42,
    )
    colorbar_ax = fig.add_axes([0.25, 0.065, 0.69, 0.032])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    colorbar.locator = MaxNLocator(nbins=5)
    colorbar.update_ticks()
    style_colorbar(colorbar, tick_size=7.0)
    colorbar.ax.tick_params(length=1.8, pad=1)
    colorbar.set_label("Accuracy change (points)", fontsize=7.2, labelpad=1)

    fig.subplots_adjust(**PANEL_MARGINS)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = dataset.lower()
    pdf = out_dir / f"{stem}_{suffix}.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf


def plot(
    points: list[dict[str, Any]], out_dir: Path, stem: str
) -> dict[str, Path]:
    max_abs_accuracy = max(abs(float(row["difference_pp"])) for row in points)
    color_limit = max(5.0, 5.0 * math.ceil(max_abs_accuracy / 5.0))
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap("RdBu")
    all_min = min(float(row["min_bytes"]) for row in points)
    all_max = max(float(row["max_bytes"]) for row in points)
    x_limits = (all_min / 1.45, all_max * 1.45)
    outputs: dict[str, Path] = {}
    for dataset in DATASETS:
        pdf = plot_panel(
            points, dataset, out_dir, stem, norm, cmap, x_limits
        )
        outputs[dataset] = pdf
    return outputs


def write_data(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "dataset",
        "model",
        "model_label",
        "representation",
        "median_bytes",
        "min_bytes",
        "max_bytes",
        "gpu_saving_vs_raw_pct",
        "difference_pp",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the two-panel representation-payload paper figure."
    )
    parser.add_argument("--payload-csv", type=Path, required=True)
    parser.add_argument("--cg-profile-csv", type=Path, required=True)
    parser.add_argument("--hd-all-cost-csv", type=Path, required=True)
    parser.add_argument("--accuracy-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="representation_payload_tradeoff")
    args = parser.parse_args()

    inputs = (
        args.payload_csv,
        args.cg_profile_csv,
        args.hd_all_cost_csv,
        args.accuracy_csv,
    )
    payload = read_payload(args.payload_csv)
    cg_savings = read_cg_savings(args.cg_profile_csv)
    hd_savings = read_hd_savings(args.hd_all_cost_csv)
    accuracy = read_accuracy(args.accuracy_csv)
    points = build_points(payload, cg_savings, hd_savings, accuracy)
    panel_outputs = plot(points, args.out_dir, args.stem)
    data_path = args.out_dir / f"{args.stem}.data.csv"
    write_data(data_path, points)
    manifest_path = args.out_dir / f"{args.stem}.sources.json"
    manifest_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "panels": {"CG": "m=6", "HD": "all questions"},
                "points": len(points),
                "inputs": [
                    {"path": str(path.resolve()), "sha256": source_digest(path)}
                    for path in inputs
                ],
                "outputs": {
                    dataset: str(path.resolve())
                    for dataset, path in panel_outputs.items()
                } | {
                    "data": str(data_path.resolve()),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "points": len(points),
                "panels": {
                    dataset: str(path.resolve())
                    for dataset, path in panel_outputs.items()
                },
                "data": str(data_path.resolve()),
                "manifest": str(manifest_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
