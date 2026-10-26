# Reproducing REEL

There are two distinct reproduction levels:

1. **Released aggregate results:** available now and CPU-verifiable from
   `results/paper_tables/`.
2. **End-to-end GPU inference:** requires external benchmark media, official
   annotations, the six pinned checkpoints, and an exact cohort manifest.

The canonical cohort manifest contents and a legally redistributable GPU smoke
fixture are not tracked in this staging repository. The CPU-only cohort
builders are included, so an authorized user can reconstruct the selected IDs
and order from official annotations. The workflow is still not a
download-free, one-command claim. See [`DATA.md`](DATA.md) for construction
rules, hashes, and path-dependent hash caveats.

## Fixed experimental contract

For one model and benchmark, change only the retained representation:

| Paper path | `--mode` | Server cache condition |
| --- | --- | --- |
| Frames | `query_time_raw` | prefix caching off, multimodal cache 0 GiB |
| Text | `write_once_frame_text` | prefix caching off, multimodal cache 0 GiB |
| Visual | `write_once_visual_token_cache` | prefix caching off, multimodal cache 4 GiB, encoder cache enabled |
| Prefix State | `write_once_model_state_cache` | prefix caching on, multimodal cache 4 GiB |

Keep the following fixed across all four paths:

- identical video IDs, question IDs, question order, choice order, and correct
  answers;
- 64 identical question-blind uniform timestamps per video;
- the checkpoint revision, processor, context limit, prompt, generation
  settings, and hardware;
- one common, text-only server warm-up before the measured runner;
- temperature 0, maximum 384 answer tokens, and two malformed-answer repair
  attempts; and
- one synchronous client request at a time.

Text uses the `balanced24` profile, writes eight frames per chunk, and has a
1,024-token frame-writer ceiling. Visual and Prefix State each use the
runner's one-token question-blind warm-up. MiniCPM uses video input and has
thinking disabled. InternVL uses exactly one 448 x 448 patch per timestamp.
Visual additionally loads the released vLLM encoder-cache instrumentation;
setting its cache-size environment variable without loading the patch is not
equivalent. The full service contract is in [`MODELS.md`](MODELS.md).

## 1. Verify the aggregate release

Install the CPU environment from [`INSTALL.md`](INSTALL.md), then run:

```bash
reel-verify-results
(cd results/paper_tables && sha256sum -c SHA256SUMS)
```

The verifier checks hashes, schemas, row counts, and the complete two-dataset
by six-model workload matrix. It does not regenerate aggregate CSVs from
private question-level predictions.

The package records an important timing decision. Frames, Visual, and Prefix
State in a main-table workload come from one coherent batch. For CG,
Qwen3-VL-8B and Gemma use the 2026-10-22 batch; InternVL3 and Qwen3-VL-4B use
the 2026-08-23 batch; InternVL3.5 and MiniCPM use the 2026-10-24 batch. Text
uses the timestamped 2026-10-22 Balanced24 runs. HD uses the accepted
`20261023_slim_v1` cells. The summary ablation intentionally retains its own
2026-10-22 Frames runs; correctness agreement with main-table Frames is in
`summary_ablation_frames_agreement.csv`.

`cost_sensitivity.csv` reports Visual minus Frames, Prefix State minus Frames,
and Prefix State minus Visual for all 12 model-dataset workloads.
`batch_timing_variation.csv` provides the observed same-path batch variation
used to flag contrasts smaller than timing drift.

## 2. Build the cohort manifests

All three cohort tools expose their complete interface locally:

```bash
python scripts/audit_cgbench_and_build_mq.py --help
python scripts/build_hdepic_mq10.py --help
python scripts/materialize_hdepic_mq10.py --help
```

Keep generated manifests outside Git. For CG MQ-50, point the builder at the
official annotation JSON, media, and subtitle directories. The runnable-media
check requires FFmpeg's `ffprobe`; verify it before building:

```bash
ffprobe -version

export REEL_BUILD_ROOT=/absolute/path/to/reel-build
export CG_ANNOTATIONS=/absolute/path/to/cgbench/annotations.json
export CG_VIDEO_ROOT=/absolute/path/to/cgbench/videos
export CG_SUBTITLE_ROOT=/absolute/path/to/cgbench/subtitles

python scripts/audit_cgbench_and_build_mq.py \
  --annotations "$CG_ANNOTATIONS" \
  --video-root "$CG_VIDEO_ROOT" \
  --subtitle-root "$CG_SUBTITLE_ROOT" \
  --out-dir "$REEL_BUILD_ROOT/cg" \
  --seed 20260722 \
  --sizes 50 \
  --questions-per-video 6 \
  --require-subtitles
```

The inference manifest is
`$REEL_BUILD_ROOT/cg/CG-Bench-MQ-50.questions.jsonl`. Require 50 videos, 300
rows, six rows per video, the source snapshot hash in `DATA.md`, and the
expected selected IDs/order. Its byte hash also includes local media/subtitle
paths and JSON serialization, so the accepted-run hash is reproducible only
under the same path-normalization convention.

For HD-EPIC-51, first build the annotation-only path-relative manifest, then
materialize it against the downloaded media:

```bash
export HDEPIC_ANNOTATIONS=/absolute/path/to/hd-epic/annotations
export HDEPIC_VIDEO_ROOT=/absolute/path/to/hd-epic-download-root

python scripts/build_hdepic_mq10.py \
  --annotations "$HDEPIC_ANNOTATIONS" \
  --out "$REEL_BUILD_ROOT/hd/questions.jsonl" \
  --summary "$REEL_BUILD_ROOT/hd/questions.summary.json" \
  --video-ids-out "$REEL_BUILD_ROOT/hd/video_ids.txt" \
  --videos 51 \
  --questions-per-video 20 \
  --include-all-eligible-questions

sha256sum "$REEL_BUILD_ROOT/hd/questions.jsonl"

python scripts/materialize_hdepic_mq10.py \
  --manifest "$REEL_BUILD_ROOT/hd/questions.jsonl" \
  --video-root "$HDEPIC_VIDEO_ROOT" \
  --out "$REEL_BUILD_ROOT/hd/questions.materialized.jsonl" \
  --summary "$REEL_BUILD_ROOT/hd/materialized.summary.json" \
  --expected-videos 51 \
  --questions-per-video 20 \
  --allow-variable-questions
```

The path-relative `questions.jsonl` should have SHA-256
`0c25ee80492909c026949c110519befce398ad06a6f770626e1424702efce3ae`,
51 videos, and 1,871 rows. The materialized output should retain those IDs and
order and report 70,533,652,710 total video bytes for the accepted media
snapshot. Its byte hash is local-path dependent.

## 3. Prepare an end-to-end run

1. Install the correct GPU environment from [`INSTALL.md`](INSTALL.md).
2. Download one exact checkpoint revision and configure its service using
   [`MODELS.md`](MODELS.md).
3. Download the benchmark from its official source.
4. Run the applicable cohort builder above and validate its source hash,
   selected IDs, order, counts, and path-relative hash before materialization.
5. Split by `video_uid` into five video-disjoint shards. Keep every question
   for a video in one shard and preserve `within_video_question_order`.

Each shard uses one GPU and a separate vLLM process. Do not put two shards on
one server or allow requests from multiple clients to overlap, because that
changes cache residency and timing.

Before starting one shard, set its local values. The example assumes the
runner's frame cache is below `REEL_RUN_ROOT`, which is also the directory
vLLM is permitted to read:

```bash
export REEL_DATA_ROOT=/absolute/path/to/reel-data
export REEL_RUN_ROOT=/absolute/path/to/reel-runs
export REEL_MODEL_DIR=/absolute/path/to/pinned-checkpoint
export REEL_MODEL_FAMILY=qwen3vl
export REEL_DATASET=cg
export REEL_SHARD_MANIFEST=/absolute/path/to/questions.shard-0-of-5.jsonl
export REEL_MODE=query_time_raw
export REEL_DATASET_VERSION=cg-mq50
export REEL_RUN_ID=cg-model-frames-shard0

case "$REEL_MODEL_FAMILY" in
  qwen3vl)       export REEL_SERVED_MODEL=qwen3-vl-8b;   export REEL_VISUAL_TOKEN_COUNTER=qwen ;;
  qwen3vl4b)     export REEL_SERVED_MODEL=qwen3-vl-4b;   export REEL_VISUAL_TOKEN_COUNTER=qwen ;;
  internvl3)     export REEL_SERVED_MODEL=internvl3-8b;  export REEL_VISUAL_TOKEN_COUNTER=unavailable ;;
  internvl35_8b) export REEL_SERVED_MODEL=internvl3.5-8b; export REEL_VISUAL_TOKEN_COUNTER=unavailable ;;
  gemma3)        export REEL_SERVED_MODEL=gemma-3-12b-it; export REEL_VISUAL_TOKEN_COUNTER=unavailable ;;
  minicpmv45)    export REEL_SERVED_MODEL=minicpm-v-4.5; export REEL_VISUAL_TOKEN_COUNTER=unavailable ;;
  *) echo "unsupported model family: $REEL_MODEL_FAMILY" >&2; exit 2 ;;
esac
```

The family names and served-model names above are fixed by the released
launcher. It also derives the context limit, GPU-memory utilization,
multimodal limit, cache flags, and model-specific processor arguments from
the model family, dataset, and mode. Do not override them independently.

## 4. Start the model service

Use a fresh vLLM process and run directory for each cache condition. The
released launcher is the canonical service entry point: it verifies the exact
checkpoint revision and vLLM version, fixes every service argument, installs
the Visual-only instrumentation when needed, and writes a path-redacted launch
receipt. In the service shell, run:

```bash
python scripts/launch_reel_vllm.py \
  --model-family "$REEL_MODEL_FAMILY" \
  --dataset "$REEL_DATASET" \
  --mode "$REEL_MODE" \
  --model-dir "$REEL_MODEL_DIR" \
  --allowed-media-root "$REEL_RUN_ROOT" \
  --run-dir "$REEL_RUN_ROOT/$REEL_RUN_ID" \
  --port 8000 \
  --execute
```

Do not bypass the launcher or set `PESTATE_RESOURCE_TRACE=1` for timing runs;
resource tracing synchronizes CUDA and changes timing. For Visual, confirm the
server log reports installation of the encoder patch and the intended minimum
token capacity.

After the service is ready, use a second shell to verify its model identity
and perform the same one-request, text-only server warm-up before every
measured run:

```bash
python scripts/launch_reel_vllm.py warmup \
  --model-family "$REEL_MODEL_FAMILY" \
  --port 8000 \
  --out "$REEL_RUN_ROOT/$REEL_RUN_ID/server_warmup_acceptance.json" \
  --execute
```

Require `accepted: true` in the warm-up receipt. This unmetered server warm-up
is common to all four paths and is separate from the per-video, one-token
question-blind construction request made by the Visual and Prefix State runner
modes.

For MiniCPM, use the one-video limit, the `minicpmv45` family, and pass
`--visual-input-format video` to the runner. For InternVL, select its matching
family so the fixed one-patch processor arguments are applied. Do not
substitute default service settings. Every path needs
`--allowed-local-media-path` because the runner sends local media as `file://`
URLs.

## 5. Run one shard

With the values from step 3 still set, choose the model- and dataset-specific
runner flags and run the controlled client:

```bash
RUNNER_MODEL_ARGS=()
if test "$REEL_MODEL_FAMILY" = minicpmv45; then
  RUNNER_MODEL_ARGS=(--visual-input-format video)
fi
RUNNER_DATASET_ARGS=()
if test "$REEL_DATASET" = hd; then
  RUNNER_DATASET_ARGS=(--expand-official-visual-references)
fi

python scripts/run_controlled_memory_policy_cgbench.py \
  --mode "$REEL_MODE" \
  --manifest "$REEL_SHARD_MANIFEST" \
  --out-dir "$REEL_RUN_ROOT/$REEL_RUN_ID" \
  --frame-cache "$REEL_RUN_ROOT/$REEL_RUN_ID/frames" \
  --workspace-root "$REEL_DATA_ROOT" \
  --base-url http://127.0.0.1:8000/v1 \
  --model "$REEL_SERVED_MODEL" \
  --visual-model-path "$REEL_MODEL_DIR" \
  --run-id "$REEL_RUN_ID" \
  --dataset-version "$REEL_DATASET_VERSION" \
  --num-frames 64 \
  --writer-chunk-size 8 \
  --frame-text-profile balanced24 \
  --frame-writer-tokens 1024 \
  --max-answer-tokens 384 \
  --answer-repair-attempts 2 \
  --visual-token-counter "$REEL_VISUAL_TOKEN_COUNTER" \
  --allow-result-errors \
  "${RUNNER_MODEL_ARGS[@]}" "${RUNNER_DATASET_ARGS[@]}"
```

Change `REEL_MODE` and use a new run directory, then repeat the launcher and
server warm-up for the other three paths. The arrays above add native video
input for MiniCPM and official reference normalization for HD-EPIC. The
diagnostic `--include-hdepic-bbox-reference-frame` option is not part of the
main four-path experiment.

The runner writes `results.jsonl` and `cost_ledger.jsonl` under `--out-dir`,
plus path-specific acceptance and provenance files. It refuses to overwrite an
existing `results.jsonl`; use a new run directory rather than deleting or
mixing partial output.

While the service is still running, create a cache receipt for Visual or
Prefix State. For Visual:

```bash
VISUAL_INPUT_ARGS=()
if test "$REEL_MODEL_FAMILY" = minicpmv45; then
  VISUAL_INPUT_ARGS=(--video-input)
fi
python scripts/validate_visual_cache.py \
  --manifest "$REEL_SHARD_MANIFEST" \
  --results "$REEL_RUN_ROOT/$REEL_RUN_ID/results.jsonl" \
  --trace "$REEL_RUN_ROOT/$REEL_RUN_ID/encoder_trace.jsonl" \
  --num-frames 64 \
  --out "$REEL_RUN_ROOT/$REEL_RUN_ID/visual_token_cache_acceptance.json" \
  "${VISUAL_INPUT_ARGS[@]}"
```

For Prefix State:

```bash
curl --fail http://127.0.0.1:8000/metrics \
  --output "$REEL_RUN_ROOT/$REEL_RUN_ID/metrics.prom"
python scripts/validate_model_state_cache.py \
  --metrics "$REEL_RUN_ROOT/$REEL_RUN_ID/metrics.prom" \
  --results "$REEL_RUN_ROOT/$REEL_RUN_ID/results.jsonl" \
  --ledger "$REEL_RUN_ROOT/$REEL_RUN_ID/cost_ledger.jsonl" \
  --max-shared-prefix-shortfall-tokens 128 \
  --out "$REEL_RUN_ROOT/$REEL_RUN_ID/cache_acceptance.json"
```

Both validators exit nonzero on a failed cache contract. Keep their receipts
with the shard; they may contain benchmark identifiers and are not part of the
public aggregate release.

## 6. Accept and combine shards

Before combining a configuration, require all five shards to satisfy:

- no missing or duplicate `qid`;
- exactly 50 videos and 300 questions for CG MQ-50, with six questions for
  every video;
- exactly 51 videos and 1,871 questions for HD-EPIC-51, with 20--86 questions
  per video;
- the expected checkpoint revision, context limit, processor settings, and
  cache flags in every receipt;
- an accepted common server-warm-up receipt created before the measured
  runner;
- complete result and cost-ledger rows, including any repair calls;
- no unreviewed decode, transport, or measurement failure; and
- identical timestamp and input fingerprints across the four paths.

Visual additionally requires an accepted
`visual_token_cache_acceptance.json`; Prefix State requires an accepted
`cache_acceptance.json` whose `strict_shared_prefix_audit.passed` field is
true with at most 128 shortfall tokens. A successful model answer is not
evidence that the intended cache was actually reused.

For InternVL3.5, the receipt must show a 32,768-token context and one 448 x 448
patch per timestamp. For MiniCPM, it must show a 40,960-token context, vLLM
0.15.0, one video input, and thinking disabled. Visual receipts must show
prefix caching off, a 4 GiB multimodal cache, and a 65,536-token minimum
encoder cache. Prefix State receipts must show prefix caching on and a 4 GiB
multimodal cache. If vLLM computes a larger native encoder budget, the
instrumentation keeps that larger value. For the accepted Qwen HD Visual run,
vLLM reported an effective 153,600-token budget.

Concatenate shards only after validation, preserving the canonical manifest
order. Do not use filename order as question order, and do not silently remove
failed rows. The five-shard merge must contain the same `qid` set as the source
manifest.

## 7. Cost comparison rule

For every model/benchmark workload, collect Frames, Visual, and Prefix State
in the same operational batch. Normal run-to-run timing drift can otherwise
reverse a small cost ordering. Report all three pairwise differences and place
them next to the observed same-model, same-path batch variation. Flag any
difference whose magnitude is smaller than that variation. A conclusion that
changes when a different batch is selected must be labeled batch-sensitive.

The released aggregate tables implement this rule. They do not include raw
per-question responses or local run paths.

## Current limit of public regeneration

The repository validates the final aggregate package and releases the cohort
builders and analysis utilities that are safe to redistribute. It does not
include canonical manifest contents, benchmark assets, model weights, raw
experiment directories, or a complete public raw-run-to-camera-ready
orchestration layer. Until a legal smoke fixture and author-approved release
metadata are supplied, treat the GPU workflow as a precise protocol record
rather than a turnkey bundled reproduction.
