#!/usr/bin/env python3
"""Create compact paper assets for the counterfactual GLM tariff analysis.

"Sensitivity" has one concrete meaning here: keep every measured token count
fixed, change only the price rule, and check whether the cheapest
representation changes. The analysis scales the two official input-length
tiers together, so a multiplier of one is the current tariff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from paper_plot_style import style_colorbar, style_heatmap_axis  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "outputs/tables/glm_tariff_counterfactual_strict_lifecycle_20260824"
)
DATASETS = ("CG MQ-50", "HD-EPIC-51")
MODELS = ("Qwen3-VL-8B", "InternVL3-8B", "Gemma-3-12B-IT")
REPRESENTATIONS = ("RepeatedFrames", "FrameText", "VisualTokens", "KVCache")
CACHE_SCENARIOS = ("no_cache", "observed_hits")
TIER_RATES = {
    "lt32k": {"miss": 5.0, "hit": 1.2, "output": 22.0},
    "ge32k": {"miss": 7.0, "hit": 1.8, "output": 26.0},
}
CURRENT_HIT_MULTIPLIER = 1.0
CURRENT_OUTPUT_MULTIPLIER = 1.0
HEATMAP_HIT_MULTIPLIERS = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)
HEATMAP_OUTPUT_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0, 8.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_accepted(directory: Path) -> Path:
    acceptance = directory / "acceptance.txt"
    if not acceptance.is_file():
        raise ValueError(f"counterfactual source has no acceptance record: {acceptance}")
    fields = {
        key.strip(): value.strip()
        for line in acceptance.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    if fields.get("accepted", "").lower() != "true":
        raise ValueError(f"counterfactual source is not accepted: {acceptance}")
    return acceptance


def endpoint_token_rows(input_dir: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = read_csv(input_dir / "mean_cumulative_cost.csv")
    observed = [row for row in rows if row["cache_scenario"] == "observed_hits"]
    endpoint_by_workload = {
        (dataset, model): max(
            int(row["m"])
            for row in observed
            if row["dataset"] == dataset and row["model"] == model
        )
        for dataset in DATASETS
        for model in MODELS
    }
    output: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in observed:
        workload = (row["dataset"], row["model"])
        if int(row["m"]) != endpoint_by_workload[workload]:
            continue
        values: dict[str, float] = {
            "m": float(row["m"]),
            "miss": float(row["missed_input_tokens_mean_per_video"]),
            "hit": float(row["cache_hit_input_tokens_mean_per_video"]),
            "output": float(row["output_tokens_mean_per_video"]),
        }
        for tier in TIER_RATES:
            for field in ("missed_input_tokens", "cache_hit_input_tokens", "output_tokens"):
                values[f"{tier}_{field}"] = float(
                    row[f"{tier}_{field}_mean_per_video"]
                )
        output[(row["dataset"], row["model"], row["representation"])] = values
    expected = len(DATASETS) * len(MODELS) * len(REPRESENTATIONS)
    if len(output) != expected:
        raise ValueError(f"found {len(output)} endpoint rows, expected {expected}")
    return output


def official_charge_components(row: dict[str, float]) -> dict[str, float]:
    return {
        component: sum(
            row[f"{tier}_{field}"] * TIER_RATES[tier][component]
            for tier in TIER_RATES
        )
        for component, field in (
            ("miss", "missed_input_tokens"),
            ("hit", "cache_hit_input_tokens"),
            ("output", "output_tokens"),
        )
    }


def price_units(
    row: dict[str, float], hit_multiplier: float, output_multiplier: float
) -> float:
    components = official_charge_components(row)
    return (
        components["miss"]
        + hit_multiplier * components["hit"]
        + output_multiplier * components["output"]
    )


def winner_for(
    endpoint: dict[tuple[str, str, str], dict[str, float]],
    dataset: str,
    model: str,
    hit_multiplier: float,
    output_multiplier: float,
) -> str:
    values = {
        representation: price_units(
            endpoint[(dataset, model, representation)], hit_multiplier, output_multiplier
        )
        for representation in REPRESENTATIONS
    }
    return min(values, key=values.get)


def tier_flags(input_dir: Path) -> dict[tuple[str, str], bool]:
    flags: dict[tuple[str, str], bool] = {}
    for row in read_csv(input_dir / "sources.csv"):
        key = (row["dataset"], row["model"])
        uses_long_tier = int(row["requests_using_32k_plus_tier"]) > 0
        flags[key] = flags.get(key, False) or uses_long_tier
    return flags


def crossing_assets(input_dir: Path) -> tuple[list[dict[str, Any]], str]:
    rows = read_csv(input_dir / "monetary_break_even.csv")
    indexed = {
        (row["dataset"], row["model"], row["cache_scenario"], row["representation"]): row
        for row in rows
    }
    flags = tier_flags(input_dir)
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            item: dict[str, Any] = {
                "dataset": "CG" if dataset.startswith("CG") else "HD",
                "model": model,
                "uses_both_official_input_tiers": flags[(dataset, model)],
            }
            for scenario in CACHE_SCENARIOS:
                for representation in REPRESENTATIONS[1:]:
                    row = indexed[(dataset, model, scenario, representation)]
                    crossing = row["first_sustained_monetary_crossing"]
                    endpoint = int(row["max_common_m"])
                    key = f"{scenario}_{representation}"
                    item[f"{key}_crossing"] = int(crossing) if crossing else ""
                    item[f"{key}_display"] = crossing if crossing else f">{endpoint}"
            output.append(item)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\caption{Counterfactual monetary break-even under the GLM tariff. "
        r"Entries give the earliest sustained question index; $>{H}$ means no "
        r"crossing by the common endpoint $H$. No cache applies one hypothetical "
        r"cache policy to every representation. Ledger hits discounts only the "
        r"prefix-cache hits measured by our local engine.}",
        r"\label{tab:glm-tariff-crossing}",
        r"\begin{tabular}{llccc@{\hspace{4pt}}ccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{No cache} & \multicolumn{3}{c}{Ledger hits} \\",
        r"Dataset & Model & Text & Visual & KV & Text & Visual & KV \\",
        r"\midrule",
    ]
    for row in output:
        model = {
            "Qwen3-VL-8B": "Qwen",
            "InternVL3-8B": "InternVL",
            "Gemma-3-12B-IT": "Gemma",
        }[row["model"]]
        if row["uses_both_official_input_tiers"]:
            model += r"$^{\ast}$"
        values = [
            row[f"{scenario}_{representation}_display"]
            for scenario in CACHE_SCENARIOS
            for representation in REPRESENTATIONS[1:]
        ]
        latex_values = [f"${value}$" if str(value).startswith(">") else str(value) for value in values]
        lines.append(
            f"{row['dataset']} & {model} & " + " & ".join(latex_values) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\parbox{\columnwidth}{\scriptsize $^{\ast}$Rows include requests "
        r"from both official input-length tiers; unmarked rows use the $<32$k "
        r"tier. Prices are "
        r"selected per request. Miss, hit, and output tokens remain separate. "
        r"GLM's implicit automatic hits require an actual API run and are not "
        r"reconstructed here.}",
        r"\end{table}",
        "",
    ])
    return output, "\n".join(lines)


def threshold_assets(
    endpoint: dict[tuple[str, str, str], dict[str, float]],
    flags: dict[tuple[str, str], bool],
) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            text = endpoint[(dataset, model, "FrameText")]
            kv = endpoint[(dataset, model, "KVCache")]
            raw = endpoint[(dataset, model, "RepeatedFrames")]
            text_charge = official_charge_components(text)
            kv_charge = official_charge_components(kv)
            raw_charge = official_charge_components(raw)
            if kv_charge["hit"] <= 0 or text_charge["output"] <= kv_charge["output"]:
                raise ValueError(f"cannot derive Text/KV price boundary for {dataset}/{model}")
            hit_threshold = (
                text_charge["miss"]
                - kv_charge["miss"]
                + CURRENT_OUTPUT_MULTIPLIER
                * (text_charge["output"] - kv_charge["output"])
            ) / kv_charge["hit"]
            output_threshold = (
                kv_charge["miss"]
                + CURRENT_HIT_MULTIPLIER * kv_charge["hit"]
                - text_charge["miss"]
            ) / (text_charge["output"] - kv_charge["output"])
            if text_charge["output"] <= raw_charge["output"]:
                raise ValueError(f"cannot derive no-cache Text/Frames boundary for {dataset}/{model}")
            no_cache_output_threshold = (
                raw_charge["miss"] - text_charge["miss"]
            ) / (text_charge["output"] - raw_charge["output"])
            current_winner = winner_for(
                endpoint,
                dataset,
                model,
                CURRENT_HIT_MULTIPLIER,
                CURRENT_OUTPUT_MULTIPLIER,
            )
            output.append({
                "dataset": "CG" if dataset.startswith("CG") else "HD",
                "model": model,
                "uses_both_official_input_tiers": flags[(dataset, model)],
                "current_hit_price_multiplier": CURRENT_HIT_MULTIPLIER,
                "current_output_price_multiplier": CURRENT_OUTPUT_MULTIPLIER,
                "current_endpoint_winner": current_winner,
                "hit_price_multiplier_boundary_at_current_output_price": hit_threshold,
                "text_wins_when_hit_multiplier_is_at_least_boundary": True,
                "output_price_multiplier_boundary_at_current_hit_price": output_threshold,
                "text_wins_when_output_multiplier_is_at_most_boundary": True,
                "no_cache_text_vs_frames_output_multiplier_boundary": no_cache_output_threshold,
            })

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Price-rule sensitivity at the common endpoint. Multipliers "
        r"scale both official input-length tiers. Keeping output prices fixed, "
        r"Text is cheaper than KV when the hit-price multiplier is at least "
        r"$h^{\dagger}$. Keeping hit prices fixed, Text is cheaper when the "
        r"output-price multiplier is at most $o^{\dagger}$. "
        r"The current tariff therefore selects Text in all six workloads; the "
        r"table shows how far each workload is from changing winner.}",
        r"\label{tab:glm-tariff-sensitivity}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Model & $h^{\dagger}$ & $o^{\dagger}$ & No-cache $o^{\dagger}_0$ \\",
        r"\midrule",
    ]
    for row in output:
        model = {
            "Qwen3-VL-8B": "Qwen",
            "InternVL3-8B": "InternVL",
            "Gemma-3-12B-IT": "Gemma",
        }[row["model"]]
        if row["uses_both_official_input_tiers"]:
            model += r"$^{\ast}$"
        hit_boundary = row["hit_price_multiplier_boundary_at_current_output_price"]
        hit_display = r"$\leq 0$" if hit_boundary <= 0 else f"{hit_boundary:.3f}"
        lines.append(
            f"{row['dataset']} & {model} & "
            f"{hit_display} & "
            f"{row['output_price_multiplier_boundary_at_current_hit_price']:.1f} & "
            f"{row['no_cache_text_vs_frames_output_multiplier_boundary']:.1f} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\parbox{\columnwidth}{\scriptsize A multiplier of one is the official "
        r"price. $o^{\dagger}_0$ compares Text with Repeated Frames when no "
        r"condition receives a cache discount. $^{\ast}$Rows include requests "
        r"from both official input-length tiers.}",
        r"\end{table}",
        "",
    ])
    return output, "\n".join(lines)


def heatmap_assets(
    endpoint: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[list[dict[str, Any]], np.ndarray, list[list[str]]]:
    rows: list[dict[str, Any]] = []
    matrix = np.zeros((len(HEATMAP_OUTPUT_MULTIPLIERS), len(HEATMAP_HIT_MULTIPLIERS)))
    annotations: list[list[str]] = []
    for row_index, output_multiplier in enumerate(HEATMAP_OUTPUT_MULTIPLIERS):
        annotation_row: list[str] = []
        for column_index, hit_multiplier in enumerate(HEATMAP_HIT_MULTIPLIERS):
            winners = [
                winner_for(endpoint, dataset, model, hit_multiplier, output_multiplier)
                for dataset in DATASETS
                for model in MODELS
            ]
            counts = Counter(winners)
            text_count = counts["FrameText"]
            kv_count = counts["KVCache"]
            matrix[row_index, column_index] = text_count
            annotation = f"{text_count}/{kv_count}"
            if sum(counts[name] for name in ("FrameText", "KVCache")) != len(winners):
                annotation += "$^{+}$"
            annotation_row.append(annotation)
            rows.append({
                "hit_price_multiplier": hit_multiplier,
                "output_price_multiplier": output_multiplier,
                "frame_text_winners": text_count,
                "kv_cache_winners": kv_count,
                "visual_token_winners": counts["VisualTokens"],
                "repeated_frame_winners": counts["RepeatedFrames"],
                "winner_counts_json": json.dumps(counts, sort_keys=True),
                "is_current_glm_tariff": (
                    hit_multiplier == CURRENT_HIT_MULTIPLIER
                    and output_multiplier == CURRENT_OUTPUT_MULTIPLIER
                ),
            })
        annotations.append(annotation_row)
    return rows, matrix, annotations


def plot_heatmap(
    path_base: Path,
    matrix: np.ndarray,
    annotations: list[list[str]],
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    cmap = plt.get_cmap("YlGnBu", 7)
    norm = BoundaryNorm(np.arange(-0.5, 7.0, 1.0), cmap.N)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(HEATMAP_HIT_MULTIPLIERS)))
    ax.set_xticklabels([f"{value:g}" for value in HEATMAP_HIT_MULTIPLIERS], fontsize=7.4)
    ax.set_yticks(range(len(HEATMAP_OUTPUT_MULTIPLIERS)))
    ax.set_yticklabels([f"{value:g}" for value in HEATMAP_OUTPUT_MULTIPLIERS], fontsize=7.4)
    ax.set_xlabel("Cache-hit price multiplier", fontsize=8.4)
    ax.set_ylabel("Output price multiplier", fontsize=8.4)
    ax.set_xticks(np.arange(-.5, len(HEATMAP_HIT_MULTIPLIERS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(HEATMAP_OUTPUT_MULTIPLIERS), 1), minor=True)
    style_heatmap_axis(ax, tick_size=7.4)
    for row_index, row in enumerate(annotations):
        for column_index, value in enumerate(row):
            color = "white" if matrix[row_index, column_index] >= 4 else "black"
            ax.text(
                column_index, row_index, value,
                ha="center", va="center", fontsize=6.5, color=color,
            )
    current_column = HEATMAP_HIT_MULTIPLIERS.index(CURRENT_HIT_MULTIPLIER)
    current_row = HEATMAP_OUTPUT_MULTIPLIERS.index(CURRENT_OUTPUT_MULTIPLIER)
    ax.add_patch(Rectangle(
        (current_column - 0.49, current_row - 0.49), 0.98, 0.98,
        fill=False, edgecolor="#d62728", linewidth=1.6,
    ))
    colorbar = fig.colorbar(image, ax=ax, ticks=range(7), fraction=0.055, pad=0.03)
    style_colorbar(colorbar, tick_size=7.2)
    colorbar.set_label("Text winners", fontsize=7.6)
    fig.text(
        0.01, 0.005,
        r"Red box: official prices (1, 1); rates are selected per request.",
        fontsize=5.8,
    )
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.22, top=0.96)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def paper_notes(thresholds: list[dict[str, Any]]) -> str:
    hit_row = max(
        thresholds,
        key=lambda row: float(row["hit_price_multiplier_boundary_at_current_output_price"]),
    )
    output_row = min(
        thresholds,
        key=lambda row: float(row["output_price_multiplier_boundary_at_current_hit_price"]),
    )
    hit_label = f"{hit_row['dataset']}-{hit_row['model']}"
    output_label = f"{output_row['dataset']}-{output_row['model']}"
    hit_boundary = float(hit_row["hit_price_multiplier_boundary_at_current_output_price"])
    output_boundary = float(output_row["output_price_multiplier_boundary_at_current_hit_price"])
    return f"""# GLM tariff analysis: paper notes

## What sensitivity means

We keep every measured Miss, Hit, and Out count fixed and change only their
prices.  A hit-price multiplier scales the official hit prices in both input
tiers; an output-price multiplier does the same to both official output prices.
If the cheapest representation changes, the conclusion is sensitive to the
provider's price rule.  This is not another model run.

## Fair cache comparison

- **No cache** is the main controlled counterfactual: all four representations
  pay the missed-input rate for every prompt token.
- **Ledger hits** is a mechanism comparison: only cache hits measured by our
  local engine receive the hit price.
- GLM uses implicit automatic caching.  An actual GLM bill may therefore cache
  Raw, Text, or Visual prompts too.  Only a new GLM run can report those hit
  counts; the present ledgers cannot reconstruct them fairly.

## Main result in plain language

Under the current GLM ratios, Text is the cheapest endpoint in all six
workloads.  This conclusion is not equally stable everywhere.  The closest
boundaries are recomputed from the selected source rather than hard-coded.
The closest hit-price boundary is {hit_label} at
{hit_boundary:.3f} times the current hit price; the closest output-price
boundary is {output_label} at {output_boundary:.2f} times the current output
price.  Crossing either boundary makes KV cheaper than Text in that workload.
Under no cache, Text remains cheaper than Repeated Frames over a much wider
output-price range.

Visual never wins anywhere in the plotted grid.  The reason is specific to
token billing: its internal encoder reuse reduces GPU work but does not reduce
the logical prompt charged by the tariff.

## Scope warning

This is a counterfactual price-rule sensitivity analysis, not an observed GLM
bill. Token counts come from Qwen, InternVL, and Gemma rather than GLM. We apply
the official <32k prices (5/1.2/22 RMB/M for Miss/Hit/Out) or 32k+ prices
(7/1.8/26) separately to each request. Rows that contain at least one long-tier
request are marked in the generated tables; every observed request remains
within the stated 200k context. The counts still use non-GLM tokenization.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_INPUT / "paper_assets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    acceptance = require_accepted(args.input_dir)

    expected_outputs = (
        "paper_monetary_crossing.csv",
        "monetary_crossing_table.tex",
        "tariff_sensitivity_thresholds.csv",
        "tariff_sensitivity_table.tex",
        "tariff_sensitivity_grid.csv",
        "glm_tariff_sensitivity_heatmap.pdf",
        "glm_tariff_sensitivity_heatmap.png",
        "PAPER_NOTES.md",
        "acceptance.json",
    )
    if not args.overwrite:
        existing = [args.out_dir / name for name in expected_outputs if (args.out_dir / name).exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite paper assets: {existing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint_token_rows(args.input_dir)
    flags = tier_flags(args.input_dir)
    crossings, crossing_tex = crossing_assets(args.input_dir)
    thresholds, threshold_tex = threshold_assets(endpoint, flags)
    grid, matrix, annotations = heatmap_assets(endpoint)

    write_csv(args.out_dir / "paper_monetary_crossing.csv", crossings)
    (args.out_dir / "monetary_crossing_table.tex").write_text(
        crossing_tex, encoding="utf-8"
    )
    write_csv(args.out_dir / "tariff_sensitivity_thresholds.csv", thresholds)
    (args.out_dir / "tariff_sensitivity_table.tex").write_text(
        threshold_tex, encoding="utf-8"
    )
    write_csv(args.out_dir / "tariff_sensitivity_grid.csv", grid)
    plot_heatmap(args.out_dir / "glm_tariff_sensitivity_heatmap", matrix, annotations)
    (args.out_dir / "PAPER_NOTES.md").write_text(
        paper_notes(thresholds), encoding="utf-8"
    )
    input_names = (
        "mean_cumulative_cost.csv",
        "monetary_break_even.csv",
        "sources.csv",
    )
    output_names = expected_outputs[:-1]
    (args.out_dir / "acceptance.json").write_text(
        json.dumps({
            "accepted": True,
            "source": str(args.input_dir.resolve()),
            "source_acceptance": {
                "path": str(acceptance.resolve()),
                "sha256": sha256(acceptance),
            },
            "inputs": [
                {
                    "path": str((args.input_dir / name).resolve()),
                    "sha256": sha256(args.input_dir / name),
                }
                for name in input_names
            ],
            "outputs": [
                {
                    "path": str((args.out_dir / name).resolve()),
                    "sha256": sha256(args.out_dir / name),
                }
                for name in output_names
            ],
            "crossing_rows": len(crossings),
            "threshold_rows": len(thresholds),
            "sensitivity_grid_cells": len(grid),
            "current_price_multipliers": {
                "cache_hit": CURRENT_HIT_MULTIPLIER,
                "output": CURRENT_OUTPUT_MULTIPLIER,
            },
            "figure_counts": "each cell is Text/KV endpoint winners among six workloads",
            "tier_scope": (
                "asterisks mark model--dataset rows with at least one request "
                "in each official input-length tier"
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
