#!/usr/bin/env python3
"""Render publication-safe strict question-type heatmaps for all six models.

All six columns use the accepted strict-format results.  Exact intervals and
denominators stay in the companion CSV files rather than being squeezed into
heatmap cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reel-strict-type-mpl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from paper_plot_style import style_colorbar, style_heatmap_axis


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tables/frame_text_by_question_type_strict_all6_20260823"
REWEIGHT_SOURCE = ROOT / "outputs/tables/official_question_mix_reweighting_strict_all6_20260823"
OUT = ROOT / "paper/reel_wacv2027/figures/generated"

MODEL_ORDER = (
    "Qwen3-VL-8B",
    "InternVL3-8B",
    "Gemma-3-12B-IT",
    "Qwen3-VL-4B",
    "InternVL3.5-8B",
    "MiniCPM-V-4.5-8B",
)
MODEL_LABELS = (
    "Qwen\n8B",
    "Intern\n3",
    "Gemma\n12B",
    "Qwen\n4B",
    "Intern\n3.5",
    "MiniCPM\n8B",
)
PENDING = frozenset()

DATASET_SPECS = {
    "CG MQ-50": {
        "title": "CG-Bench",
        "stem": "question_type_heatmap_cg_strict_all6",
        "height": 3.75,
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
        "title": "HD-EPIC",
        "stem": "question_type_heatmap_hd_strict_all6",
        "height": 4.55,
        "labels": {
            "3d_perception_fixture_interaction_counting": "Fixture interaction count",
            "3d_perception_fixture_location": "Fixture location",
            "3d_perception_object_contents_retrieval": "Object contents",
            "3d_perception_object_location": "Object location",
            "ingredient_ingredient_adding_localization": "Ingredient adding loc.",
            "ingredient_ingredient_retrieval": "Ingredient retrieval",
            "ingredient_ingredients_order": "Ingredient order",
            "nutrition_video_nutrition_estimation": "Nutrition estimation",
            "object_motion_object_movement_itinerary": "Movement itinerary",
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


def require_accepted(path: Path) -> dict[str, str]:
    fields = {
        key.strip(): value.strip()
        for line in (path / "acceptance.txt").read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    if fields.get("accepted") != "true":
        raise ValueError(f"unaccepted source: {path}")
    return fields


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_panels(
    source: Path = SOURCE,
    reweight_source: Path = REWEIGHT_SOURCE,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    require_accepted(source)
    require_accepted(reweight_source)
    rows = read_csv(source / "by_official_question_type.csv")
    coverage = read_csv(source / "model_dataset_strict_coverage.csv")
    reweight = read_csv(reweight_source / "aggregate_weighting_comparison.csv")
    indexed = {
        (row["dataset"], row["model"], row["official_question_type"]): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("duplicate strict question-type rows")

    panels: dict[str, dict] = {}
    usable_models = [model for model in MODEL_ORDER if model not in PENDING]
    for dataset, spec in DATASET_SPECS.items():
        official_types = list(spec["labels"])
        expected = {
            (dataset, model, question_type)
            for model in MODEL_ORDER
            for question_type in official_types
        }
        if expected - set(indexed):
            raise ValueError(f"missing strict cells for {dataset}: {expected-set(indexed)}")

        def ordering(question_type: str) -> tuple[int, float]:
            values = [
                float(indexed[(dataset, model, question_type)]["text_minus_raw"])
                for model in usable_models
            ]
            if all(value <= 0 for value in values) and any(value < 0 for value in values):
                direction = 0
            elif all(value >= 0 for value in values) and any(value > 0 for value in values):
                direction = 2
            else:
                direction = 1
            return direction, float(np.mean(values))

        official_types.sort(key=ordering)
        values = np.full((len(official_types), len(MODEL_ORDER)), np.nan)
        labels: list[str] = []
        for row_index, question_type in enumerate(official_types):
            totals = {
                int(indexed[(dataset, model, question_type)]["total_questions"])
                for model in MODEL_ORDER
            }
            if len(totals) != 1:
                raise ValueError(f"type denominator changes across models: {dataset}/{question_type}")
            total = totals.pop()
            labels.append(spec["labels"][question_type] + (r"$^\dagger$" if total < 30 else ""))
            for column_index, model in enumerate(MODEL_ORDER):
                row = indexed[(dataset, model, question_type)]
                if model in PENDING:
                    if row["display_allowed"].lower() != "false":
                        raise ValueError(f"pending model was not masked: {dataset}/{model}")
                    continue
                values[row_index, column_index] = 100 * float(row["text_minus_raw"])
        panels[dataset] = {
            "question_types": official_types,
            "labels": labels,
            "values": values,
        }
    return panels, coverage, reweight


def draw_panel(
    ax,
    dataset: str,
    panel: dict,
    *,
    compact: bool = False,
) -> object:
    values = panel["values"]
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#E8EBEE")
    artist = ax.imshow(
        values,
        cmap=cmap,
        vmin=-40,
        vmax=40,
        aspect="auto",
        interpolation="none",
    )
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_LABELS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=5.4 if compact else 6.0, pad=2, length=0)
    ax.set_yticks(np.arange(len(panel["labels"])), panel["labels"])
    ax.tick_params(axis="y", labelsize=4.8 if compact else 5.8, pad=2, length=0)
    ax.set_xticks(np.arange(-.5, len(MODEL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(panel["labels"]), 1), minor=True)
    style_heatmap_axis(ax, tick_size=4.8 if compact else 5.8)
    ax.tick_params(axis="x", labelsize=5.8 if compact else 6.0, pad=2, length=0)
    ax.tick_params(axis="y", labelsize=5.4 if compact else 5.8, pad=2, length=0)
    panel_id = "(a)" if dataset == next(iter(DATASET_SPECS)) else "(b)"
    ax.text(0.0, 1.13, panel_id, transform=ax.transAxes,
            fontsize=7.0 if compact else 7.5, fontweight="bold",
            ha="left", va="bottom")

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if np.isnan(value):
                continue
            color = "white" if abs(value) >= 22 else "#17232D"
            ax.text(
                column_index,
                row_index,
                f"{value:+.0f}" if compact else f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=3.4 if compact else 4.7,
                color=color,
            )
    for column_index, model in enumerate(MODEL_ORDER):
        if model not in PENDING:
            continue
        ax.text(
            column_index,
            (values.shape[0] - 1) / 2,
            "pending",
            rotation=90,
            ha="center",
            va="center",
            fontsize=4.0 if compact else 4.7,
            color="#697680",
            fontweight="bold",
        )
        ax.add_patch(Rectangle(
            (column_index - .5, -.5),
            1,
            values.shape[0],
            fill=False,
            edgecolor="#BCC4CA",
            linewidth=.8,
            hatch="///",
        ))
    return artist


def select_extreme_rows(panel: dict, count_each: int = 4) -> dict:
    """Select the lowest and highest equal-model mean rows, without overlap."""
    values = panel["values"]
    means = np.nanmean(values, axis=1)
    order = np.argsort(means, kind="stable")
    if len(order) < 2 * count_each:
        raise ValueError("not enough question types for the requested extremes")
    selected = np.concatenate((order[:count_each], order[-count_each:]))
    return {
        "question_types": [panel["question_types"][index] for index in selected],
        "labels": [panel["labels"][index] for index in selected],
        "values": values[selected],
        "means": means[selected],
        "groups": ["lowest"] * count_each + ["highest"] * count_each,
    }


def write_extreme_rows(
    extreme_panels: dict[str, dict],
    source: Path,
    output_path: Path,
) -> None:
    """Save the plotted values and deterministic selection rule for audit."""
    type_rows = read_csv(source / "by_official_question_type.csv")
    type_index = {
        (row["dataset"], row["model"], row["official_question_type"]): row
        for row in type_rows
    }
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "selection_group",
            "display_order",
            "official_question_type",
            "display_label",
            "total_questions",
            "equal_model_mean_pp",
            *MODEL_ORDER,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, panel in extreme_panels.items():
            for index, question_type in enumerate(panel["question_types"]):
                source_row = type_index[(dataset, MODEL_ORDER[0], question_type)]
                output_row = {
                    "dataset": dataset,
                    "selection_group": panel["groups"][index],
                    "display_order": index + 1,
                    "official_question_type": question_type,
                    "display_label": DATASET_SPECS[dataset]["labels"][question_type],
                    "total_questions": source_row["total_questions"],
                    "equal_model_mean_pp": f"{panel['means'][index]:.6f}",
                }
                output_row.update({
                    model: f"{panel['values'][index, model_index]:.6f}"
                    for model_index, model in enumerate(MODEL_ORDER)
                })
                writer.writerow(output_row)


def signed(value: float) -> str:
    return f"{value:+.1f}" if value else "0.0"


def interval_cell(row: dict, point: str, low: str, high: str) -> str:
    return (
        "$"
        + signed(100 * float(row[point]))
        + r"\,["
        + signed(100 * float(row[low]))
        + ","
        + signed(100 * float(row[high]))
        + "]$"
    )


def write_latex_helpers(
    panels: dict[str, dict],
    coverage_rows: list[dict],
    reweight_rows: list[dict],
    source: Path,
    out: Path,
    suffix: str,
) -> None:
    """Write optional rows; no manuscript section is modified here."""
    out.mkdir(parents=True, exist_ok=True)
    coverage_index = {(row["dataset"], row["model"]): row for row in coverage_rows}
    coverage_lines = []
    for dataset in DATASET_SPECS:
        for model in MODEL_ORDER:
            row = coverage_index[(dataset, model)]
            status = "Pending rerun" if model in PENDING else "Usable"
            coverage_lines.append(
                f"{dataset} & {model} & {int(row['paired_valid_questions'])} & "
                f"{100*float(row['paired_valid_coverage']):.1f} & {status} \\\\"
            )
    (out / f"question_type_strict_coverage_rows{suffix}.tex").write_text(
        "\n".join(coverage_lines) + "\n\\bottomrule\n"
    )

    type_rows = read_csv(source / "by_official_question_type.csv")
    type_index = {
        (row["dataset"], row["model"], row["official_question_type"]): row
        for row in type_rows
    }
    usable_models = [model for model in MODEL_ORDER if model not in PENDING]
    type_lines = []
    for dataset, panel in panels.items():
        first = True
        for question_type in panel["question_types"]:
            reference = type_index[(dataset, usable_models[0], question_type)]
            cells = [
                interval_cell(
                    type_index[(dataset, model, question_type)],
                    "text_minus_raw",
                    "ci_low",
                    "ci_high",
                )
                for model in usable_models
            ]
            dataset_label = ("CG" if dataset == "CG MQ-50" else "HD") if first else ""
            first = False
            display = DATASET_SPECS[dataset]["labels"][question_type]
            type_lines.append(
                f"{dataset_label} & {display} & {int(reference['total_questions'])} & "
                + " & ".join(cells)
                + r" \\"
            )
        if dataset != list(panels)[-1]:
            type_lines.append(r"\midrule")
    (out / f"question_type_heterogeneity_strict_rows{suffix}.tex").write_text(
        "\n".join(type_lines) + "\n\\bottomrule\n"
    )

    reweight_lines = []
    for row in reweight_rows:
        if row["included_in_publication_claim"].lower() != "true":
            continue
        if row["estimable"].lower() != "true":
            continue
        reweight_lines.append(
            f"{row['dataset']} & {row['model']} & "
            + interval_cell(
                row,
                "cohort_question_weighted",
                "cohort_ci_low",
                "cohort_ci_high",
            )
            + " & "
            + interval_cell(
                row,
                "official_frequency_reweighted",
                "official_frequency_ci_low",
                "official_frequency_ci_high",
            )
            + " & "
            + interval_cell(
                row,
                "equal_type_weighted",
                "equal_type_ci_low",
                "equal_type_ci_high",
            )
            + r" \\"
        )
    (out / f"question_type_weighting_comparison_rows{suffix}.tex").write_text(
        "\n".join(reweight_lines) + "\n\\bottomrule\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--reweight-source-dir", type=Path, default=REWEIGHT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--stem-suffix",
        default="",
        help=(
            "Suffix appended to every generated asset stem, for example "
            "timemap_final_v2_20260826."
        ),
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow replacing files with the requested stems.",
    )
    return parser.parse_args(argv)


def normalized_suffix(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError("--stem-suffix may contain only letters, digits, ., _, and -")
    return "_" + value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source_dir.resolve()
    reweight_source = args.reweight_source_dir.resolve()
    out = args.out_dir.resolve()
    suffix = normalized_suffix(args.stem_suffix)
    panels, coverage, reweight = build_panels(source, reweight_source)

    individual_stems = {
        dataset: f"{spec['stem']}{suffix}"
        for dataset, spec in DATASET_SPECS.items()
    }
    appendix_stem = f"question_type_heatmaps_strict_all6_appendix{suffix}"
    extreme_stem = f"question_type_heatmaps_extremes_strict_all6{suffix}"
    provenance_stem = f"question_type_heatmaps_strict_all6{suffix}"
    requested_paths = [
        *(out / f"{stem}.{extension}"
          for stem in individual_stems.values()
          for extension in ("pdf", "png")),
        *(out / f"{appendix_stem}.{extension}" for extension in ("pdf", "png")),
        *(out / f"{extreme_stem}.{extension}" for extension in ("pdf", "png")),
        out / f"{extreme_stem}.data.csv",
        out / f"question_type_strict_coverage_rows{suffix}.tex",
        out / f"question_type_heterogeneity_strict_rows{suffix}.tex",
        out / f"question_type_weighting_comparison_rows{suffix}.tex",
        out / f"{provenance_stem}.sources.json",
        out / f"{provenance_stem}.acceptance.txt",
    ]
    existing = [path for path in requested_paths if path.exists()]
    if existing and not args.allow_existing_output:
        raise FileExistsError(
            "refusing to overwrite requested heatmap assets: "
            + ", ".join(str(path) for path in existing)
        )
    out.mkdir(parents=True, exist_ok=True)

    for dataset, spec in DATASET_SPECS.items():
        fig, ax = plt.subplots(figsize=(3.35, spec["height"]))
        artist = draw_panel(ax, dataset, panels[dataset])
        colorbar = fig.colorbar(
            artist,
            ax=ax,
            orientation="horizontal",
            fraction=.04,
            pad=.075,
            aspect=24,
        )
        colorbar.set_ticks([-40, -20, 0, 20, 40])
        style_colorbar(colorbar, tick_size=6.0)
        colorbar.ax.tick_params(length=2, pad=1)
        colorbar.set_label(
            r"Text $-$ Frames accuracy (percentage points)",
            fontsize=6.2,
            labelpad=1,
        )
        fig.text(
            .985, .012, r"$^\dagger$ fewer than 30 cohort questions",
            ha="right", va="bottom", fontsize=4.5, color="#596873",
        )
        fig.subplots_adjust(left=.48, right=.985, bottom=.17, top=.86)
        stem = individual_stems[dataset]
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=.03)
        fig.savefig(out / f"{stem}.png", dpi=300,
                    bbox_inches="tight", pad_inches=.03)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.65))
    artist = None
    for ax, dataset in zip(axes, DATASET_SPECS):
        artist = draw_panel(ax, dataset, panels[dataset], compact=True)
    fig.subplots_adjust(left=.23, right=.995, bottom=.18, top=.91, wspace=.74)
    colorbar_ax = fig.add_axes([.35, .085, .43, .018])
    colorbar = fig.colorbar(artist, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([-40, -20, 0, 20, 40])
    style_colorbar(colorbar, tick_size=5.8)
    colorbar.ax.tick_params(length=2, pad=1)
    colorbar.set_label(
        r"Text $-$ Frames accuracy (percentage points)",
        fontsize=6.2,
        labelpad=1,
    )
    fig.text(
        .995,
        .012,
        r"$^\dagger$ fewer than 30 cohort questions. Exact CIs and strict paired coverage are reported separately.",
        ha="right",
        va="bottom",
        fontsize=4.6,
        color="#596873",
    )
    fig.savefig(
        out / f"{appendix_stem}.pdf",
        bbox_inches="tight",
        pad_inches=.03,
    )
    fig.savefig(
        out / f"{appendix_stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=.03,
    )
    plt.close(fig)

    extreme_panels = {
        dataset: select_extreme_rows(panel)
        for dataset, panel in panels.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    artist = None
    for ax, dataset in zip(axes, DATASET_SPECS):
        artist = draw_panel(ax, dataset, extreme_panels[dataset], compact=True)
        ax.axhline(3.5, color="#25313A", linewidth=1.0)
    fig.subplots_adjust(left=.215, right=.995, bottom=.25, top=.86, wspace=.68)
    colorbar_ax = fig.add_axes([.34, .105, .45, .026])
    colorbar = fig.colorbar(artist, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([-40, -20, 0, 20, 40])
    style_colorbar(colorbar, tick_size=5.8)
    colorbar.ax.tick_params(length=2, pad=1)
    colorbar.set_label(
        r"Text $-$ Frames accuracy (percentage points)",
        fontsize=6.2,
        labelpad=1,
    )
    fig.savefig(
        out / f"{extreme_stem}.pdf",
        bbox_inches="tight",
        pad_inches=.03,
    )
    fig.savefig(
        out / f"{extreme_stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=.03,
    )
    plt.close(fig)
    write_extreme_rows(extreme_panels, source, out / f"{extreme_stem}.data.csv")
    write_latex_helpers(panels, coverage, reweight, source, out, suffix)

    provenance = {
        "accepted": True,
        "question_type_source": {
            "path": str(source),
            "acceptance_sha256": sha256(source / "acceptance.txt"),
            "table_sha256": sha256(source / "by_official_question_type.csv"),
        },
        "reweight_source": {
            "path": str(reweight_source),
            "acceptance_sha256": sha256(reweight_source / "acceptance.txt"),
            "table_sha256": sha256(
                reweight_source / "aggregate_weighting_comparison.csv"
            ),
        },
        "selection_rule": (
            "For each dataset, take the four lowest and four highest "
            "equal-model mean Text-minus-Frames effects."
        ),
        "main_extreme_stem": extreme_stem,
        "appendix_full_stem": appendix_stem,
        "individual_stems": individual_stems,
    }
    (out / f"{provenance_stem}.sources.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (out / f"{provenance_stem}.acceptance.txt").write_text(
        "accepted=true\n"
        "datasets=2\n"
        "models=6\n"
        "main_rows_per_dataset=8\n"
        "extreme_low_rows=4\n"
        "extreme_high_rows=4\n"
        "appendix_full_taxonomies=true\n"
        f"question_type_source={source}\n"
        f"reweight_source={reweight_source}\n"
        f"main_extreme_stem={extreme_stem}\n"
        f"appendix_full_stem={appendix_stem}\n"
    )
    print("accepted=true")
    print("single_column_figures=2")
    print("appendix_two_panel_figure=true")
    print("main_extreme_two_panel_figure=true")
    print("pending_columns_masked=none")


if __name__ == "__main__":
    main()
