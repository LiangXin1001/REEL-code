#!/usr/bin/env python3
"""Generate compact, source-validated figures used by the REEL paper."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reel-paper-mpl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "paper" / "reel_wacv2027" / "figures" / "generated"

# NPG palette from ggsci. Four conditions use red, blue, orange, and dark blue.
NPG_RED = "#E64B35"
NPG_BLUE = "#4DBBD5"
NPG_ORANGE = "#F39B7F"
NPG_DARK_BLUE = "#3C5488"
GRID = "#D9E1E8"
MUTED = "#536273"


def require_accepted(directory: Path) -> None:
    """Reject a derived result directory unless its audit explicitly accepted it."""
    acceptance = directory / "acceptance.txt"
    if not acceptance.is_file():
        raise ValueError(f"Missing acceptance record: {acceptance}")
    fields = {
        key.strip(): value.strip()
        for line in acceptance.read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    if fields.get("accepted", "").lower() != "true":
        raise ValueError(f"Refusing unaccepted source: {directory}")


def read_curves(path: Path) -> dict[str, list[float]]:
    values: dict[str, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "gpu_seconds":
                values.setdefault(row["method"], {})[int(row["m"])] = float(row["mean_per_video"])
    return {method: [by_m[m] for m in range(1, 7)] for method, by_m in values.items()}


def read_curve_with_ci(
    path: Path, method: str, expected_m: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_accepted(path.parent)
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == method and row["metric"] == "gpu_seconds":
                rows.append(row)
    rows.sort(key=lambda row: int(row["m"]))
    if [int(row["m"]) for row in rows] != expected_m:
        raise ValueError(
            f"{path} does not contain prefixes {expected_m[0]}--{expected_m[-1]} "
            f"for {method}"
        )
    return tuple(
        np.asarray([float(row[key]) for row in rows])
        for key in ("mean_per_video", "ci_low", "ci_high")
    )


def style(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=.6, alpha=.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, stem: str, left: float = .18) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=left, right=.98, bottom=.20, top=.72)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def released_system_profiles() -> None:
    """Show first-question versus six-question cost for released systems.

    The sidecar is a provenance manifest rather than an accepted controlled
    comparison.  Validate both the manifest coverage and every plotted prefix;
    the figure deliberately labels these points as descriptive profiles.
    """
    source = TABLES / "released_system_lifecycle_gpu_seconds.csv"
    manifest_path = TABLES / "released_system_lifecycle_sources.json"
    names = ["LongVideo-R1", "LongVU", "LongVT", "LangRepo", "StreamingTOM"]
    if not manifest_path.is_file():
        raise ValueError(f"Missing released-system provenance: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if set(names) - set(manifest):
        raise ValueError("Released-system provenance does not cover every method")
    curves = read_curves(source)
    if set(curves) != set(names) or any(len(curves[name]) != 6 for name in names):
        raise ValueError("Released-system profile must contain m=1--6 for five methods")
    appendix_endpoints = {
        "LongVideo-R1": (53.4, 301.9),
        "LongVU": (24.2, 144.8),
        "LongVT": (10.2, 42.2),
        "LangRepo": (62.7, 76.1),
        "StreamingTOM": (12.7, 14.0),
    }
    for name, expected in appendix_endpoints.items():
        observed = (curves[name][0], curves[name][5])
        if any(abs(a - b) > .051 for a, b in zip(observed, expected)):
            raise ValueError(f"{name} G1/G6 does not match the audited appendix table")

    behavior = {
        "LongVideo-R1": "Question-time pipeline",
        "LongVU": "Question-time pipeline",
        "LongVT": "Question-time pipeline",
        "LangRepo": "Build and reuse",
        "StreamingTOM": "Build and reuse",
    }
    colors = {
        "Question-time pipeline": NPG_RED,
        "Build and reuse": NPG_BLUE,
    }
    markers = {
        "Question-time pipeline": "o",
        "Build and reuse": "s",
    }
    label_offsets = {
        "LongVideo-R1": (-3.0, -18.0, "right"),
        "LongVU": (2.0, 5.0, "left"),
        "LongVT": (2.0, 8.0, "left"),
        "LangRepo": (-2.0, 10.0, "right"),
        "StreamingTOM": (2.0, -1.0, "left"),
    }

    fig, ax = plt.subplots(figsize=(3.25, 2.65))
    ax.plot([0, 70], [0, 70], color=MUTED, linestyle="--", linewidth=.8,
            alpha=.75, zorder=1)
    ax.text(43, 51, "$G_6=G_1$", color=MUTED, fontsize=5.5,
            ha="left", va="bottom")
    for name in names:
        g1, g6 = curves[name][0], curves[name][5]
        group = behavior[name]
        ax.scatter(g1, g6, s=34, color=colors[group], marker=markers[group],
                   edgecolor="white", linewidth=.5, zorder=3)
        dx, dy, ha = label_offsets[name]
        ax.annotate(name, (g1, g6), xytext=(g1 + dx, g6 + dy),
                    textcoords="data", fontsize=6.1, color="#26323D",
                    ha=ha, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRID,
                                "linewidth": .55, "shrinkA": 1.5,
                                "shrinkB": 2.5})
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 325)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel("First-question cost, $G_1$ (GPU-s/video)", fontsize=7)
    ax.set_ylabel("Six-question cost, $G_6$ (GPU-s/video)", fontsize=7)
    style(ax, "both")
    handles = [
        Line2D([], [], linestyle="none", marker=markers[group], markersize=5.2,
               markerfacecolor=colors[group], markeredgecolor="white", label=group)
        for group in ("Question-time pipeline", "Build and reuse")
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, ncol=2,
              loc="lower center", bbox_to_anchor=(.5, 1.01),
              borderaxespad=0, handletextpad=.4, columnspacing=.8)
    fig.subplots_adjust(left=.18, right=.98, bottom=.18, top=.86)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "released_system_profiles.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "released_system_profiles.png", dpi=240,
                bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def controlled_lifecycle(model: str, source: Path, stem: str) -> None:
    values = read_curves(source)
    methods = ["WriteOnceText", "BlindReplayText", "QueryConditionedText"]
    labels = ["Shared captions", "Recomputed captions", "Question-specific captions"]
    colors = [NPG_BLUE, NPG_DARK_BLUE, NPG_RED]
    styles = ["-", "--", "-"]
    m = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    for method, label, color, line_style in zip(methods, labels, colors, styles):
        ax.plot(m, values[method], marker="o", markersize=4.0, linewidth=1.6,
                linestyle=line_style, color=color, label=label)
    ax.set_xticks(m)
    ax.set_xlabel("Questions per video ($m$)", fontsize=8)
    ax.set_ylabel("Cumulative GPU-s / video", fontsize=8)
    style(ax)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0,
              columnspacing=.9, handlelength=2.0)
    save(fig, stem, left=.20)


def reuse_vs_question_aware(source: Path, stem: str) -> None:
    values = read_curves(source)
    m = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    ax.plot(m, values["SharedCaptions"], marker="o", markersize=4.2,
            linewidth=1.7, color=NPG_BLUE, label="Shared captions")
    ax.plot(m, values["QuestionAwareObservation"], marker="o", markersize=4.2,
            linewidth=1.7, color=NPG_RED, label="Coarse scan + local 8")
    ax.set_xticks(m)
    ax.set_xlabel("Questions per video ($m$)", fontsize=8)
    ax.set_ylabel("Cumulative GPU-s / video", fontsize=8)
    style(ax)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0,
              columnspacing=1.0, handlelength=2.0)
    save(fig, stem, left=.20)


def model_state_lifecycle(source: Path, stem: str) -> None:
    values = read_curves(source)
    m = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    ax.plot(m, values["QueryTimeRaw"], marker="o", markersize=4.2,
            linewidth=1.7, color=NPG_RED, label="Repeated frame input")
    ax.plot(m, values["ModelStateCache"], marker="o", markersize=4.2,
            linewidth=1.7, color=NPG_BLUE, label="Multimodal KV cache")
    ax.set_xticks(m)
    ax.set_xlabel("Questions per video ($m$)", fontsize=8)
    ax.set_ylabel("Cumulative GPU-s / video", fontsize=8)
    style(ax)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0,
              columnspacing=1.0, handlelength=2.0)
    save(fig, stem, left=.20)


def model_state_delta_lifecycle() -> None:
    sources = {
        "Qwen3-VL": TABLES / "modelstate64_qwen_mq50_allprefix_20260811" / "lifecycle_metric_ci.csv",
        "InternVL3": TABLES / "modelstate64_internvl_mq50_allprefix_20260811" / "lifecycle_metric_ci.csv",
    }
    colors = {"Qwen3-VL": NPG_BLUE, "InternVL3": NPG_RED}
    m = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(3.2, 2.15))
    ax.axhline(0, color=GRID, linewidth=1.1, zorder=0)
    for label, source in sources.items():
        values = read_curves(source)
        delta = np.asarray(values["ModelStateCache"]) - np.asarray(values["QueryTimeRaw"])
        ax.plot(m, delta, marker="o", markersize=4.0, linewidth=1.7,
                color=colors[label], label=label)
    ax.set_xticks(m)
    ax.set_xlabel("Questions per video ($m$)", fontsize=8)
    ax.set_ylabel("State $-$ repeated GPU-s", fontsize=8)
    style(ax)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0,
              columnspacing=1.0, handlelength=2.0)
    save(fig, "modelstate64_delta_lifecycle", left=.20)


def evidence_window_representation_gap() -> None:
    """Compare the Raw--Text gap under uniform and clue-window observation."""
    source = TABLES / "cgbench_evidence64_formal_20260822" / "analysis"
    require_accepted(source)
    with (source / "evidence64_paper_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = ["Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT"]
    indexed = {row["model"]: row for row in rows}
    if set(indexed) != set(expected):
        raise ValueError("Evidence-window analysis does not cover the three paper models")

    labels = ["Qwen", "InternVL", "Gemma"]
    y = np.arange(len(expected))[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 1.88))
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=.8, zorder=1)
    for offset, prefix, label, color, marker in (
        (.10, "uniform", "Uniform 64", NPG_DARK_BLUE, "o"),
        (-.10, "evidence_window", "Clue-window 64", NPG_RED, "D"),
    ):
        values = np.asarray([
            float(indexed[model][f"{prefix}_text_minus_raw_pp"])
            for model in expected
        ])
        low = np.asarray([
            float(indexed[model][f"{prefix}_text_minus_raw_ci_low_pp"])
            for model in expected
        ])
        high = np.asarray([
            float(indexed[model][f"{prefix}_text_minus_raw_ci_high_pp"])
            for model in expected
        ])
        ax.errorbar(
            values, y + offset, xerr=np.vstack((values - low, high - values)),
            fmt=marker, markersize=4.2, color=color, ecolor=color,
            elinewidth=1.0, capsize=2.0, markeredgecolor="white",
            markeredgewidth=.45, label=label, zorder=3,
        )
    for row_index, model in enumerate(expected):
        uniform = float(indexed[model]["uniform_text_minus_raw_pp"])
        evidence = float(indexed[model]["evidence_window_text_minus_raw_pp"])
        ax.plot([uniform, evidence], [y[row_index] + .10, y[row_index] - .10],
                color=GRID, linewidth=.8, zorder=0)

    ax.set_yticks(y, labels)
    ax.set_xlim(-20, 10)
    ax.set_xticks([-20, -10, 0, 10])
    ax.set_xlabel("Text $-$ Repeated Frames accuracy (percentage points)", fontsize=6.8)
    style(ax, "x")
    ax.tick_params(labelsize=6.5)
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="lower center",
              bbox_to_anchor=(.5, 1.02), borderaxespad=0,
              columnspacing=.8, handletextpad=.35)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=.21, right=.98, bottom=.25, top=.76)
    fig.savefig(OUT / "evidence_window_representation_gap.pdf",
                bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "evidence_window_representation_gap.png", dpi=240,
                bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def representation_lifecycle() -> None:
    models = {
        "Qwen3-VL-8B": "qwen3vl",
        "InternVL3-8B": "internvl3",
        "Gemma-3-12B-IT": "gemma3",
    }
    datasets = {
        "CG MQ-50": ("cg", 6),
        "HD-EPIC-51": ("hd", 20),
    }
    methods = {
        "Repeated frames": "RepeatedFrames",
        "Frame-level text": "FrameText",
        "Visual-token cache": "VisualTokens",
        "KV cache": "KVCache",
    }
    colors = {
        "Repeated frames": NPG_RED,
        "Frame-level text": NPG_BLUE,
        "Visual-token cache": NPG_ORANGE,
        "KV cache": NPG_DARK_BLUE,
    }
    # A 3x2 layout remains legible at the 3.25-inch WACV column width.  Columns
    # are datasets and rows are models, so no result is dropped to save space.
    # Two dataset rows by three backbone columns: three panels per row, with
    # every figure still fitting a single WACV column.
    fig, axes = plt.subplots(
        2, 3, figsize=(3.25, 2.70), squeeze=False, sharey="row"
    )
    for row_idx, (dataset, (dataset_key, horizon)) in enumerate(datasets.items()):
        expected_m = list(range(1, horizon + 1))
        m = np.asarray(expected_m)
        for col_idx, (model, model_key) in enumerate(models.items()):
            ax = axes[row_idx, col_idx]
            source_dir = (
                TABLES
                / (
                    f"rq2_valid_pairs_{dataset_key}_internvl3_strict_final_20260824"
                    if model_key == "internvl3"
                    else f"rq2_valid_pairs_{dataset_key}_{model_key}_20260820"
                )
            )
            source = source_dir / "lifecycle_metric_ci.csv"
            for label, method in methods.items():
                mean, low, high = read_curve_with_ci(source, method, expected_m)
                color = colors[label]
                ax.fill_between(m, low, high, color=color, alpha=.11, linewidth=0)
                mark_every = None if horizon == 6 else [0, 4, 9, 14, 19]
                ax.plot(m, mean, marker="o", markersize=2.2, linewidth=1.05,
                        markevery=mark_every, color=color, label=label, zorder=3)
            short_model = {"Qwen3-VL-8B": "Qwen", "InternVL3-8B": "InternVL",
                           "Gemma-3-12B-IT": "Gemma"}[model]
            if row_idx == 0:
                ax.set_title(short_model, fontsize=5.6, fontweight="bold", pad=3)
            short_dataset = "CG" if dataset_key == "cg" else "HD-EPIC"
            ax.text(.04, .90, short_dataset, transform=ax.transAxes, fontsize=4.5,
                    color=MUTED, fontweight="bold", ha="left", va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78,
                          "pad": .7})
            if horizon == 6:
                ax.set_xticks([1, 3, 6])
            else:
                ax.set_xticks([1, 10, 20])
            style(ax)
            ax.tick_params(labelsize=4.3, pad=1.0)
    fig.supxlabel("Questions per video ($m$)", fontsize=5.2, y=.02)
    fig.supylabel("Cumulative allocated GPU-s / video", fontsize=5.2, x=.012)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_labels = ["Repeated frames", "Frame text", "Visual tokens", "KV cache"]
    fig.legend(handles, legend_labels, frameon=False, fontsize=4.4, ncol=4,
               loc="upper center", bbox_to_anchor=(.54, .995),
               columnspacing=.55, handlelength=1.2, handletextpad=.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=.18, right=.985, bottom=.12, top=.82,
                        hspace=.38, wspace=.34)
    fig.savefig(OUT / "representation_gpu_lifecycle.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "representation_gpu_lifecycle.png", dpi=240,
                bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def new_model_representation_lifecycle() -> None:
    """Plot the same RQ2 lifecycle matrix for the three added backbones."""
    models = {
        "Qwen-4B": "qwen3vl4b",
        "InternVL3.5": "internvl35_8b",
        "MiniCPM": "minicpmv45",
    }
    datasets = {"CG": ("cg", 6), "HD-EPIC": ("hd", 20)}
    methods = {
        "Repeated frames": "RepeatedFrames",
        "Frame text": "FrameText",
        "Visual tokens": "VisualTokens",
        "KV cache": "KVCache",
    }
    colors = {
        "Repeated frames": NPG_RED,
        "Frame text": NPG_BLUE,
        "Visual tokens": NPG_ORANGE,
        "KV cache": NPG_DARK_BLUE,
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(3.25, 2.70), squeeze=False, sharey="row"
    )
    for row_index, (dataset_label, (dataset_slug, horizon)) in enumerate(
        datasets.items()
    ):
        expected_m = list(range(1, horizon + 1))
        m = np.asarray(expected_m)
        for column_index, (model_label, model_slug) in enumerate(models.items()):
            ax = axes[row_index, column_index]
            source = TABLES / (
                f"rq2_valid_pairs_{dataset_slug}_{model_slug}_20260823/"
                "lifecycle_metric_ci.csv"
            )
            for label, method in methods.items():
                mean, low, high = read_curve_with_ci(source, method, expected_m)
                color = colors[label]
                ax.fill_between(m, low, high, color=color, alpha=.11, linewidth=0)
                mark_every = None if horizon == 6 else [0, 4, 9, 14, 19]
                ax.plot(
                    m, mean, marker="o", markersize=2.2, linewidth=1.05,
                    markevery=mark_every, color=color, label=label, zorder=3,
                )
            if row_index == 0:
                ax.set_title(model_label, fontsize=5.4, fontweight="bold", pad=3)
            ax.text(
                .04, .90, dataset_label, transform=ax.transAxes, fontsize=4.5,
                color=MUTED, fontweight="bold", ha="left", va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78,
                      "pad": .7},
            )
            ax.set_xticks([1, 3, 6] if horizon == 6 else [1, 10, 20])
            style(ax)
            ax.tick_params(labelsize=4.3, pad=1.0)
    fig.supxlabel("Questions per video ($m$)", fontsize=5.2, y=.02)
    fig.supylabel("Cumulative allocated GPU-s / video", fontsize=5.2, x=.012)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, list(methods), frameon=False, fontsize=4.4, ncol=4,
        loc="upper center", bbox_to_anchor=(.54, .995), columnspacing=.55,
        handlelength=1.2, handletextpad=.25,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=.18, right=.985, bottom=.12, top=.82, hspace=.38, wspace=.34
    )
    fig.savefig(
        OUT / "representation_gpu_lifecycle_new_models.pdf",
        bbox_inches="tight", pad_inches=.03,
    )
    plt.close(fig)


def accounting_timeline() -> None:
    """Draw the one-time-versus-repeated accounting boundary."""
    fig, ax = plt.subplots(figsize=(3.25, 1.72))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.25)
    ax.axis("off")

    def box(x: float, y: float, width: float, height: float, text_value: str,
            face: str, edge: str) -> None:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            linewidth=.75, facecolor=face, edgecolor=edge,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text_value,
                ha="center", va="center", fontsize=5.7, color="#22303A",
                linespacing=1.05)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
            linewidth=.7, color="#748391", shrinkA=1, shrinkB=1,
        ))

    ax.text(1.08, 3.31, "Reprocess\nframes", ha="right", va="center",
            fontsize=6.2, fontweight="bold", color="#9B3338")
    top_x = [1.65, 3.65, 6.10, 8.10]
    top_labels = ["64 frames\n+ answer 1", "64 frames\n+ answer 2", r"$\cdots$", "64 frames\n+ answer $m$"]
    top_widths = [1.55, 1.55, .75, 1.55]
    for x, label, width in zip(top_x, top_labels, top_widths):
        box(x, 2.85, width, .90, label, "#FBE7E3", "#D36A59")
    for left, left_width, right in zip(top_x, top_widths, top_x[1:]):
        arrow(left + left_width, 3.30, right, 3.30)
    ax.text(9.90, 2.56, r"$G_m=C_1+\cdots+C_m$", ha="right", va="center",
            fontsize=5.2, color="#536273")

    ax.text(1.08, 1.42, "Build once,\nthen reuse", ha="right", va="center",
            fontsize=6.2, fontweight="bold", color="#2F6C86")
    bottom_x = [1.35, 3.65, 5.65, 7.32, 8.72]
    bottom_labels = ["Build once\nText / Visual / KV", "Read +\nanswer 1",
                     "Read +\nanswer 2", r"$\cdots$", "Read +\nanswer $m$"]
    bottom_widths = [1.85, 1.40, 1.40, .65, 1.25]
    for index, (x, label, width) in enumerate(zip(bottom_x, bottom_labels, bottom_widths)):
        if index == 0:
            box(x, .96, width, .94, label, "#DDEEF5", "#4D91AD")
        else:
            box(x, .96, width, .94, label, "#E4F2E9", "#5A9A71")
    for left, left_width, right in zip(bottom_x, bottom_widths, bottom_x[1:]):
        arrow(left + left_width, 1.43, right, 1.43)
    ax.text(9.90, .68, r"$G_m=B+P_1+\cdots+P_m$", ha="right", va="center",
            fontsize=5.2, color="#536273")
    ax.annotate("count once", xy=(2.28, .96), xytext=(2.28, .30),
                ha="center", va="center", fontsize=5.0, color="#2F6C86",
                arrowprops={"arrowstyle": "-[", "color": "#4D91AD", "lw": .7})
    ax.annotate("count for every question", xy=(6.45, .96), xytext=(6.45, .30),
                ha="center", va="center", fontsize=5.0, color="#39704D",
                arrowprops={"arrowstyle": "-[", "color": "#5A9A71", "lw": .7})

    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=.01, right=.995, bottom=.02, top=.99)
    fig.savefig(OUT / "ledger.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "ledger.png", dpi=240, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def question_type_heatmaps() -> None:
    """Plot all accepted Text-minus-Raw question-type differences.

    Counts are deliberately not embedded in the cells.  A dagger on the row
    label flags an official type with fewer than 30 questions.  The color and
    printed value encode the paired percentage-point difference itself, while
    bold type marks cells whose paired 95% interval excludes zero.
    """
    source = TABLES / "frame_text_by_question_type_20260820"
    require_accepted(source)
    with (source / "by_question_type.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (source / "reconstruction_audit.csv").open(newline="") as handle:
        audit_rows = list(csv.DictReader(handle))

    models = ["Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT"]
    model_labels = ["Qwen", "InternVL", "Gemma"]
    datasets = {
        "CG MQ-50": {
            "stem": "question_type_heatmap_cg",
            "height": 3.55,
            "labels": {
                "2D Spatial Perception": "2D spatial perception",
                "Entity Cognition": "Entity cognition",
                "Entity Perception": "Entity perception",
                "Event Cognition": "Event cognition",
                "Event Perception": "Event perception",
                "Hallucination": "Hallucination",
                "Scene Cognition": "Scene cognition",
                "Scene Perception": "Scene perception",
                "Text Cognition": "Text cognition",
                "Text Perception": "Text perception",
                "Time Cognition": "Time cognition",
                "Time Perception": "Time perception",
            },
        },
        "HD-EPIC-51": {
            "stem": "question_type_heatmap_hd",
            "height": 4.30,
            "labels": {
                "3d_perception_fixture_interaction_counting": "Fixture interaction counting",
                "3d_perception_fixture_location": "Fixture location",
                "3d_perception_object_contents_retrieval": "Object contents retrieval",
                "3d_perception_object_location": "Object location",
                "ingredient_ingredient_adding_localization": "Ingredient adding localization",
                "ingredient_ingredient_retrieval": "Ingredient retrieval",
                "ingredient_ingredients_order": "Ingredient order",
                "nutrition_video_nutrition_estimation": "Nutrition estimation",
                "object_motion_object_movement_itinerary": "Object movement itinerary",
                "recipe_following_activity_recognition": "Activity recognition",
                "recipe_multi_recipe_recognition": "Multi-recipe recognition",
                "recipe_multi_step_localization": "Multi-step localization",
                "recipe_prep_localization": "Preparation localization",
                "recipe_rough_step_localization": "Rough-step localization",
                "recipe_step_localization": "Step localization",
                "recipe_step_recognition": "Step recognition",
            },
        },
    }
    expected_rows = sum(len(spec["labels"]) for spec in datasets.values()) * len(models)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} accepted question-type rows, got {len(rows)}")
    if len(audit_rows) != len(datasets) * len(models):
        raise ValueError("Question-type reconstruction audit has incomplete model coverage")
    for row in audit_rows:
        if abs(float(row["reconstruction_error"])) >= 1e-12:
            raise ValueError(f"Question-type results do not reconstruct: {row}")

    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["official_question_type"])
        if key in indexed:
            raise ValueError(f"Duplicate question-type row: {key}")
        indexed[key] = row
    expected_keys = {
        (dataset, model, question_type)
        for dataset, spec in datasets.items()
        for model in models
        for question_type in spec["labels"]
    }
    if set(indexed) != expected_keys:
        raise ValueError(
            f"Question-type coverage mismatch: missing={expected_keys-set(indexed)}, "
            f"extra={set(indexed)-expected_keys}"
        )

    limit = 35.0  # shared scale: the largest accepted magnitude is 33.3 pp
    cmap = plt.get_cmap("RdBu")  # losses are red; gains are blue
    OUT.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, dict[str, object]] = {}
    for dataset, spec in datasets.items():
        # Put consistent text losses first, mixed model directions in the
        # middle, and consistent gains last.  The mean breaks ties within each
        # group.  This preserves every cell while making the red-to-blue
        # structure much easier to scan.
        def type_sort_key(question_type: str) -> tuple[int, float]:
            differences = [
                float(indexed[(dataset, model, question_type)]["paired_difference"])
                for model in models
            ]
            if all(value <= 0 for value in differences) and any(
                value < 0 for value in differences
            ):
                direction_group = 0
            elif all(value >= 0 for value in differences) and any(
                value > 0 for value in differences
            ):
                direction_group = 2
            else:
                direction_group = 1
            return direction_group, float(np.mean(differences))

        question_types = sorted(
            spec["labels"],
            key=type_sort_key,
        )
        type_labels: list[str] = []
        for question_type in question_types:
            totals = {
                int(indexed[(dataset, model, question_type)]["total_questions"])
                for model in models
            }
            if len(totals) != 1:
                raise ValueError(
                    f"Question-type total changes across models: "
                    f"{dataset}/{question_type}: {sorted(totals)}"
                )
            total = totals.pop()
            label = spec["labels"][question_type]
            type_labels.append(label + (r"$^\dagger$" if total < 30 else ""))
        values = np.asarray([
            [100.0 * float(indexed[(dataset, model, question_type)]["paired_difference"])
             for model in models]
            for question_type in question_types
        ])
        low = np.asarray([
            [100.0 * float(indexed[(dataset, model, question_type)]["ci_low"])
             for model in models]
            for question_type in question_types
        ])
        high = np.asarray([
            [100.0 * float(indexed[(dataset, model, question_type)]["ci_high"])
             for model in models]
            for question_type in question_types
        ])
        rendered[dataset] = {
            "labels": type_labels,
            "values": values,
            "low": low,
            "high": high,
        }

        fig, ax = plt.subplots(figsize=(3.25, spec["height"]))
        image_artist = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit,
                                 aspect="auto", interpolation="none")
        ax.set_xticks(np.arange(len(models)), model_labels)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=6.2, pad=2, length=0)
        ax.set_yticks(np.arange(len(question_types)), type_labels)
        ax.tick_params(axis="y", labelsize=5.7, pad=2, length=0)
        ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(question_types), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                text_color = "white" if abs(value) >= 18 else "#1C2730"
                ax.text(col_idx, row_idx, f"{value:+.1f}", ha="center", va="center",
                        fontsize=5.6, color=text_color,
                        fontweight="bold" if low[row_idx, col_idx] > 0 or
                        high[row_idx, col_idx] < 0 else "normal")

        colorbar = fig.colorbar(image_artist, ax=ax, orientation="horizontal",
                                fraction=.045, pad=.075, aspect=22)
        colorbar.set_ticks([-30, -15, 0, 15, 30])
        colorbar.ax.tick_params(labelsize=5.4, length=2, pad=1)
        colorbar.outline.set_visible(False)
        colorbar.set_label(r"Text $-$ repeated frames: $\Delta$ accuracy (pp)",
                           fontsize=5.8, labelpad=1)
        fig.subplots_adjust(left=.53, right=.985, bottom=.16 if dataset == "CG MQ-50" else .13,
                            top=.89)
        stem = f"{spec['stem']}_full"
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.03)
        fig.savefig(OUT / f"{stem}.png", dpi=240,
                    bbox_inches="tight", pad_inches=.03)
        plt.close(fig)

    # A dedicated one-column asset remains readable after placing the two
    # datasets side by side.  Short labels are used only in this compact view;
    # the individual figures above retain the full official display names.
    compact_labels = {
        "Fixture interaction counting": "Fixture count",
        "Object contents retrieval": "Object contents",
        "Ingredient adding localization": "Adding loc.",
        "Ingredient retrieval": "Ingredient retrieval",
        "Nutrition estimation": "Nutrition est.",
        "Object movement itinerary": "Movement route",
        "Activity recognition": "Activity recog.",
        "Multi-recipe recognition": "Multi-recipe recog.",
        "Multi-step localization": "Multi-step loc.",
        "Preparation localization": "Preparation loc.",
        "Rough-step localization": "Rough-step loc.",
        "Step localization": "Step loc.",
        "Step recognition": "Step recog.",
        "2D spatial perception": "2D spatial",
    }
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 3.0))
    combined_titles = {"CG MQ-50": "CG", "HD-EPIC-51": "HD-EPIC"}
    image_artist = None
    for ax, dataset in zip(axes, datasets):
        panel = rendered[dataset]
        values = panel["values"]
        low = panel["low"]
        high = panel["high"]
        labels = [
            compact_labels.get(label.replace(r"$^\dagger$", ""),
                               label.replace(r"$^\dagger$", ""))
            + (r"$^\dagger$" if r"$^\dagger$" in label else "")
            for label in panel["labels"]
        ]
        image_artist = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit,
                                 aspect="auto", interpolation="none")
        ax.set_xticks(np.arange(len(models)), model_labels)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=4.6, pad=1, length=0)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.tick_params(axis="y", labelsize=4.0, pad=1, length=0)
        ax.set_title(combined_titles[dataset], fontsize=5.8,
                     fontweight="bold", pad=12)
        ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                excludes_zero = low[row_idx, col_idx] > 0 or high[row_idx, col_idx] < 0
                ax.text(col_idx, row_idx, f"{value:+.1f}", ha="center", va="center",
                        fontsize=3.65, color="white" if abs(value) >= 18 else "#1C2730",
                        fontweight="bold" if excludes_zero else "normal")

    fig.subplots_adjust(left=.255, right=.99, bottom=.15, top=.90, wspace=1.25)
    colorbar_ax = fig.add_axes([.34, .055, .54, .025])
    colorbar = fig.colorbar(image_artist, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([-30, 0, 30])
    colorbar.ax.tick_params(labelsize=4.2, length=1.5, pad=1)
    colorbar.outline.set_visible(False)
    colorbar.set_label(r"Text $-$ repeated frames: $\Delta$ accuracy (pp)",
                       fontsize=4.5, labelpad=1)
    fig.savefig(OUT / "question_type_heatmaps_full.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(OUT / "question_type_heatmaps_full.png", dpi=300,
                bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def question_type_top10_heatmaps() -> None:
    """Plot the five types at each end of the full red-to-blue ordering.

    This deliberately emphasizes the contrast between question types that lose
    and gain most consistently across models.  Complete official-taxonomy
    heatmaps are generated separately for the appendix by
    :func:`question_type_heatmaps`.
    """
    source = TABLES / "frame_text_by_question_type_20260820"
    require_accepted(source)
    with (source / "by_question_type.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    models = ["Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT"]
    model_labels = ["Qwen", "Intern", "Gemma"]
    dataset_specs = {
        "CG MQ-50": {
            "stem": "question_type_heatmap_cg",
            "labels": {
                "2D Spatial Perception": "2D spatial",
                "Entity Cognition": "Entity cognition",
                "Entity Perception": "Entity perception",
                "Event Cognition": "Event cognition",
                "Event Perception": "Event perception",
                "Hallucination": "Hallucination",
                "Scene Cognition": "Scene cognition",
                "Scene Perception": "Scene perception",
                "Text Cognition": "Text cognition",
                "Text Perception": "Text perception",
                "Time Cognition": "Time cognition",
                "Time Perception": "Time perception",
            },
        },
        "HD-EPIC-51": {
            "stem": "question_type_heatmap_hd",
            "labels": {
                "3d_perception_fixture_interaction_counting": "Fixture count",
                "3d_perception_fixture_location": "Fixture location",
                "3d_perception_object_contents_retrieval": "Object contents",
                "3d_perception_object_location": "Object location",
                "ingredient_ingredient_adding_localization": "Adding loc.",
                "ingredient_ingredient_retrieval": "Ingredient retrieval",
                "ingredient_ingredients_order": "Ingredient order",
                "nutrition_video_nutrition_estimation": "Nutrition est.",
                "object_motion_object_movement_itinerary": "Movement route",
                "recipe_following_activity_recognition": "Activity recog.",
                "recipe_multi_recipe_recognition": "Multi-recipe recog.",
                "recipe_multi_step_localization": "Multi-step loc.",
                "recipe_prep_localization": "Preparation loc.",
                "recipe_rough_step_localization": "Rough-step loc.",
                "recipe_step_localization": "Step loc.",
                "recipe_step_recognition": "Step recognition",
            },
        },
    }
    indexed = {
        (row["dataset"], row["model"], row["official_question_type"]): row
        for row in rows
    }
    limit = 35.0
    cmap = plt.get_cmap("RdBu")
    for dataset, spec in dataset_specs.items():
        available_types = {
            row["official_question_type"]
            for row in rows if row["dataset"] == dataset
        }
        totals: dict[str, int] = {}
        for question_type in available_types:
            observed = {
                int(indexed[(dataset, model, question_type)]["total_questions"])
                for model in models
            }
            if len(observed) != 1:
                raise ValueError(f"Question-type count changes across models: {dataset}/{question_type}")
            totals[question_type] = observed.pop()
        def sort_key(question_type: str) -> tuple[int, float]:
            differences = [
                float(indexed[(dataset, model, question_type)]["paired_difference"])
                for model in models
            ]
            if all(value <= 0 for value in differences) and any(value < 0 for value in differences):
                group = 0
            elif all(value >= 0 for value in differences) and any(value > 0 for value in differences):
                group = 2
            else:
                group = 1
            return group, float(np.mean(differences))

        ordered = sorted(available_types, key=sort_key)
        selected = ordered[:5] + ordered[-5:]
        values = np.asarray([
            [100 * float(indexed[(dataset, model, q)]["paired_difference"])
             for model in models]
            for q in selected
        ])
        low = np.asarray([
            [100 * float(indexed[(dataset, model, q)]["ci_low"])
             for model in models]
            for q in selected
        ])
        high = np.asarray([
            [100 * float(indexed[(dataset, model, q)]["ci_high"])
             for model in models]
            for q in selected
        ])
        labels = [
            spec["labels"][q] + (r"$^\dagger$" if totals[q] < 30 else "")
            for q in selected
        ]

        fig, ax = plt.subplots(figsize=(1.70, 2.55))
        image_artist = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit,
                                 aspect="auto", interpolation="none")
        ax.set_xticks(np.arange(len(models)), model_labels)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=4.4, pad=1, length=0)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.tick_params(axis="y", labelsize=4.1, pad=1, length=0)
        ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                excludes_zero = low[row_idx, col_idx] > 0 or high[row_idx, col_idx] < 0
                ax.text(col_idx, row_idx, f"{value:+.1f}", ha="center", va="center",
                        fontsize=3.65, color="white" if abs(value) >= 18 else "#1C2730",
                        fontweight="bold" if excludes_zero else "normal")
        colorbar = fig.colorbar(image_artist, ax=ax, orientation="horizontal",
                                fraction=.045, pad=.075, aspect=18)
        colorbar.set_ticks([-30, 0, 30])
        colorbar.ax.tick_params(labelsize=4.0, length=1.5, pad=1)
        colorbar.outline.set_visible(False)
        colorbar.set_label(r"Text $-$ frames (pp)", fontsize=4.2, labelpad=1)
        fig.subplots_adjust(left=.49, right=.99, bottom=.14, top=.93)
        stem = spec["stem"]
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.02)
        fig.savefig(OUT / f"{stem}.png", dpi=300,
                    bbox_inches="tight", pad_inches=.02)
        plt.close(fig)


def rq1_accuracy_forest() -> None:
    """Plot one horizontal panel per backbone for the two evaluation workloads."""
    source = TABLES / "rq1_quality_both_datasets_frame_text_20260820"
    acceptance = source / "acceptance.txt"
    if not acceptance.is_file() or "accepted=true" not in acceptance.read_text().splitlines():
        raise ValueError(f"Refusing unaccepted two-dataset RQ1 source: {source}")
    with (source / "paired_accuracy_ci.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    datasets = ["CG MQ-50", "HD-EPIC-51"]
    models = ["Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT"]
    methods = [
        ("Frame-level captions", "Text"),
        ("Visual-token cache", "Visual tokens"),
        ("KV cache", "KV cache"),
    ]
    dataset_styles = {
        "CG MQ-50": (NPG_RED, "o", "CG MQ-50"),
        "HD-EPIC-51": (NPG_BLUE, "s", "HD-EPIC-51"),
    }
    y = np.arange(len(methods))
    expected = {(dataset, model, method) for dataset in datasets for model in models
                for method, _ in methods}
    observed = {(row["dataset"], row["model"], row["representation"]) for row in rows}
    if observed != expected:
        raise ValueError(
            f"two-dataset RQ1 figure coverage mismatch: missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )

    indexed = {
        (row["dataset"], row["model"], row["representation"]): row for row in rows
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.28), sharex=True, sharey=True)
    model_labels = ["Qwen", "InternVL", "Gemma"]
    for ax, model, model_label in zip(axes, models, model_labels):
        ax.axvline(-3, color=NPG_RED, linestyle="--", linewidth=1.0, zorder=1)
        ax.axvline(0, color=MUTED, linestyle=":", linewidth=1.0, zorder=1)
        for dataset_index, dataset in enumerate(datasets):
            color, marker, _ = dataset_styles[dataset]
            offset = -.10 if dataset_index == 0 else .10
            for method_index, (method, _) in enumerate(methods):
                row = indexed[(dataset, model, method)]
                mean, low, high = [
                    float(row[key])
                    for key in ("difference_pp", "ci_low_pp", "ci_high_pp")
                ]
                ax.errorbar(
                    mean,
                    y[method_index] + offset,
                    xerr=np.asarray([[mean - low], [high - mean]]),
                    fmt=marker,
                    markersize=3.1,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=.35,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=1.8,
                    capthick=1.0,
                    zorder=3,
                )
        ax.set_xlim(-18, 10)
        ax.set_xticks([-15, 0, 10])
        ax.set_yticks(y, [label for _, label in methods])
        ax.set_ylim(y[-1] + .45, -.45)
        style(ax, "x")
        ax.tick_params(axis="both", labelsize=4.2, pad=1.0)
        ax.spines["bottom"].set_linewidth(.65)
        ax.spines["left"].set_linewidth(.65)
        ax.set_title(model_label, fontsize=5.0, fontweight="bold", pad=3)

    handles = [
        Line2D([], [], linestyle="none", marker=dataset_styles[dataset][1],
               markersize=3.8, markerfacecolor=dataset_styles[dataset][0],
               markeredgecolor="white", label=dataset_styles[dataset][2])
        for dataset in datasets
    ]
    fig.legend(handles=handles, frameon=False, fontsize=4.3, ncol=2,
               loc="upper center", bbox_to_anchor=(.52, 1.02),
               handletextpad=.4, columnspacing=1.2)
    fig.supxlabel(r"$\Delta$ accuracy from Repeated Frames (pp)",
                  fontsize=4.7, y=.015)
    fig.subplots_adjust(left=.20, right=.99, bottom=.27, top=.74, wspace=.34)
    fig.savefig(OUT / "rq1_quality_models.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / "rq1_quality_models.png", dpi=240,
                bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def model_state_budget_tradeoff(model: str, stem: str) -> None:
    frames = np.array([8, 32, 64])
    answer_accuracy = []
    gpu_saved = []
    for budget in frames:
        directory = TABLES / f"modelstate{budget}_{model}_mq50_20260805"
        with (directory / "lifecycle_metric_ci.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        answer = next(
            row for row in rows
            if row["method"] == "ModelStateCache" and row["m"] == "6"
            and row["metric"] == "acc_qa"
        )
        with (directory / "lifecycle_pair_diffs_ci.csv").open(newline="") as handle:
            pair_rows = list(csv.DictReader(handle))
        cost = next(
            row for row in pair_rows
            if row["m"] == "6" and row["metric"] == "gpu_seconds"
        )
        answer_accuracy.append(100 * float(answer["mean_per_video"]))
        gpu_saved.append(-float(cost["difference_a_minus_b"]))

    fig, left = plt.subplots(figsize=(3.2, 2.45))
    right = left.twinx()
    line_a = left.plot(frames, answer_accuracy, marker="o", markersize=4.2,
                       linewidth=1.7, color=NPG_BLUE, label="Answer accuracy")
    line_c = right.plot(frames, gpu_saved, marker="o", markersize=4.2,
                        linewidth=1.7, color=NPG_RED, label="GPU-s saved")
    left.set_xticks(frames)
    left.set_xlabel("Retained frames", fontsize=8)
    left.set_ylabel("Answer accuracy (%)", fontsize=8, color=NPG_BLUE)
    right.set_ylabel("GPU-s saved / video", fontsize=8, color=NPG_RED)
    style(left)
    right.spines[["top"]].set_visible(False)
    right.spines["right"].set_color(GRID)
    right.tick_params(colors=NPG_RED, labelsize=8)
    left.tick_params(axis="y", colors=NPG_BLUE)
    left.legend(line_a + line_c, [line.get_label() for line in line_a + line_c],
                frameon=False, fontsize=7.0, ncol=2, loc="lower center",
                bbox_to_anchor=(.5, 1.02), borderaxespad=0,
                columnspacing=.9, handlelength=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=.18, right=.82, bottom=.20, top=.72)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    representation_lifecycle()
    new_model_representation_lifecycle()
    accounting_timeline()
    evidence_window_representation_gap()
    question_type_heatmaps()
    question_type_top10_heatmaps()
    rq1_accuracy_forest()
    released_system_profiles()
    controlled_lifecycle(
        "Qwen3-VL-8B",
        TABLES / "blindreplay_qwen_mq50_allprefix_20260811" / "lifecycle_metric_ci.csv",
        "qwen_lifecycle",
    )
    controlled_lifecycle(
        "InternVL3-8B",
        TABLES / "blindreplay_internvl_mq50_allprefix_20260811" / "lifecycle_metric_ci.csv",
        "intern_lifecycle",
    )
    reuse_vs_question_aware(
        TABLES / "shared_vs_question_aware_qwen_20260811" / "lifecycle_metric_ci.csv",
        "shared_vs_question_aware_qwen",
    )
    reuse_vs_question_aware(
        TABLES / "shared_vs_question_aware_internvl_20260811" / "lifecycle_metric_ci.csv",
        "shared_vs_question_aware_internvl",
    )
    model_state_lifecycle(
        TABLES / "modelstate64_qwen_mq50_allprefix_20260811" / "lifecycle_metric_ci.csv",
        "modelstate64_qwen_lifecycle",
    )
    model_state_lifecycle(
        TABLES / "modelstate64_internvl_mq50_allprefix_20260811" / "lifecycle_metric_ci.csv",
        "modelstate64_internvl_lifecycle",
    )
    model_state_delta_lifecycle()
    model_state_budget_tradeoff("qwen", "modelstate_budget_qwen")
    model_state_budget_tradeoff("internvl", "modelstate_budget_internvl")
    print(OUT)
