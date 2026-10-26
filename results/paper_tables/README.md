# REEL aggregate paper tables

This directory contains de-identified aggregate results used by the REEL
camera-ready analysis. The tables contain model- and dataset-level summaries,
confidence intervals, lifecycle decisions, and storage measurements. They do
not contain question or video identifiers, per-example predictions, raw model
responses, prompts, media, cluster job logs, personal metadata, or local file
system paths.

These files are a compact results release, not a copy of either benchmark.
Users must obtain CG-Bench and HD-EPIC from their official sources and follow
the applicable terms.

## Provenance

The tables were exported from the following accepted analysis bundles:

- `unified_prompt_main_intermediate/20261024_v1`
- `unified_prompt_lifecycle_core/20261024_v1`
- `unified_prompt_quality_diagnostics/20261024_v2`
- `unified_prompt_question_type/20261024_v1` (strict paired-valid analysis)
- `unified_prompt_scope_storage/20261024_v1`

The main matrix used the final answer prompt with SHA-256
`baa64a3423f0e47e457b01e87fb8739d924424f7d9e8bcf60a0c508238e27ae9`.

For the main CG cost table, Frames, Visual, and Prefix State were taken from
one coherent batch for each model: Qwen3-VL-8B and Gemma-3-12B-IT use the
2026-10-22 batch; InternVL3-8B and Qwen3-VL-4B use the 2026-08-23 batch; and
InternVL3.5-8B and MiniCPM-V-4.5-8B use the 2026-10-24 batch. Text uses the
timestamped 2026-10-22 Balanced24 run for all six models. HD uses the accepted
`20261023_slim_v1` cells. This same-batch rule prevents normal timing drift
between runs from being mistaken for a representation-cost difference.

The summary ablation retains its own 2026-10-22 Frames runs rather than
silently substituting the main-table Frames runs. Their answer agreement is
reported in `summary_ablation_frames_agreement.csv`. The separate
`cost_sensitivity.csv` and `batch_timing_variation.csv` tables show when a
between-path cost difference is smaller than observed between-batch timing
variation.

## Files

- `main_table_selection.csv`: lowest-cost quality-passing condition for each
  model and dataset.
- `main_four_path_metrics.csv`: aggregate quality and paired-change metrics
  for Frames, Text, Visual, and Prefix State.
- `cost_sensitivity.csv`: three pairwise cost contrasts for each of the 12
  model-dataset workloads, compared with observed batch variation.
- `selection_paths.csv`: GPU-only and quality-constrained lifecycle paths.
- `cost_components.csv`: build, common-horizon query, and all-question costs.
- `quality_profiles_vs_frames.csv`: aggregate quality effects of the evaluated
  text-memory profiles relative to Frames.
- `clue_window_interactions.csv`: aggregate matched clue-window interaction
  estimates.
- `question_type_by_official_type.csv`: strict paired-valid results by official
  question type.
- `representation_storage.csv`: measured representation sizes by workload.
- `storage_budget_endpoint_counts.csv`: selected endpoint counts across storage
  budgets.
- `cg_uniform32_uniform64_four_path.csv`: CG results for 32- and 64-frame
  budgets across all four paths.
- `summary_ablation_frames_agreement.csv`: aggregate correctness agreement
  between main-table and summary-ablation Frames runs.
- `batch_timing_variation.csv`: aggregate per-path timing ranges across
  observed batches.

Eleven CSV files are byte-for-byte copies of their accepted aggregate source
tables, with two stable public filename changes (`cost_components.csv` and
`question_type_by_official_type.csv`). Two files were column-projected for
public release: `summary_ablation_frames_agreement.csv` drops the two local run
directory columns, and `batch_timing_variation.csv` drops `observations_json`,
which contained local run paths. No rows or retained values were changed.

`manifest.json` records the expected data-row count and key columns for every
CSV. Verify file integrity from this directory with:

```bash
sha256sum -c SHA256SUMS
```
