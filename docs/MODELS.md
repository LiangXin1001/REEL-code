# Models and serving environments

Model weights are not included. Download the exact revisions below from the
official Hugging Face repositories and follow each model's license, gated
access rules, and acceptable-use terms.

## Exact checkpoints

| Paper name | Official repository and revision | Runtime | CG context | HD context |
| --- | --- | --- | ---: | ---: |
| Qwen3-VL-8B | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b), `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | vLLM 0.11 | 65,536 | 131,072 |
| Qwen3-VL-4B | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17), `ebb281ec70b05090aa6165b016eac8ec08e71b17` | vLLM 0.11 | 65,536 | 131,072 |
| InternVL3-8B | [`OpenGVLab/InternVL3-8B`](https://huggingface.co/OpenGVLab/InternVL3-8B/tree/853e3a797a661694b1b8ece0cb72dc2b23e3dac9), `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | vLLM 0.11 | 32,768 | 32,768 |
| Gemma-3-12B-IT | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it/tree/96b6f1eccf38110c56df3a15bffe176da04bfd80), `96b6f1eccf38110c56df3a15bffe176da04bfd80` | vLLM 0.11 | 32,768 | 32,768 |
| InternVL3.5-8B | [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B/tree/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79), `9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79` | vLLM 0.11 | 32,768 | 32,768 |
| MiniCPM-V-4.5 | [`openbmb/MiniCPM-V-4_5`](https://huggingface.co/openbmb/MiniCPM-V-4_5/tree/daef484c35ec93210ec93c5e901f8f3e9b78ee34), `daef484c35ec93210ec93c5e901f8f3e9b78ee34` | vLLM 0.15 | 40,960 | 40,960 |

Context values are vLLM `--max-model-len` values in tokens. They are not the
number of generated answer tokens.

One reproducible way to materialize a checkpoint is:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir /absolute/path/to/Qwen3-VL-8B-Instruct
```

Repeat with the repository and full revision from the table. Record the
resolved directory and revision in private run provenance, but do not commit
the weight directory or a path-bearing provenance file.

## Tested software and hardware

The first five models used:

- Python 3.11.15;
- PyTorch 2.8.0+cu126;
- Transformers 4.57.0; and
- vLLM 0.11.0.

MiniCPM-V-4.5 used:

- Python 3.12.15;
- PyTorch 2.9.1+cu128;
- Transformers 4.57.1; and
- vLLM 0.15.0.

The accepted MiniCPM jobs recorded Python 3.12.15. If a manuscript appendix
still states Python 3.11.15 for this service, correct that discrepancy before
public release. The direct package records are
`configs/environments/openvlm_vllm011_requirements.txt` and
`configs/environments/openvlm_vllm015_requirements.txt`; CUDA-tagged PyTorch
wheels still require a platform-appropriate package index.

Each task used one NVIDIA L40S GPU (48 GiB), 12 CPU cores, and 112 GiB host
memory. Models ran in BF16 without quantization. vLLM selected the KV-cache
dtype with `--kv-cache-dtype auto`. Smaller GPUs and tensor-parallel serving
were not validated, so 48 GiB is the tested capacity, not a claimed universal
minimum.

## Common service contract

Start a fresh server for each cache condition. The shared vLLM arguments were:

```text
--dtype bfloat16
--generation-config vllm
--trust-remote-code
--kv-cache-dtype auto
--gpu-memory-utilization 0.88
```

Qwen3-VL on HD-EPIC used `--gpu-memory-utilization 0.96`; other cells used
0.88. Set `--max-model-len` from the checkpoint table. Image-based services
accepted up to 68 images per prompt (`--limit-mm-per-prompt '{"image":68}'`),
which covers the 64 study frames plus protocol allowance. The client issued
one synchronous request at a time. Separate shards ran in separate server
processes on separate GPUs.

The cache arguments are part of the experimental condition:

| Path | Prefix caching | Multimodal processor cache | Encoder-output cache | Extra service flag |
| --- | --- | ---: | ---: | --- |
| Frames | off | 0 GiB | off | none |
| Text | off | 0 GiB | off | none |
| Visual | off | 4 GiB | minimum 65,536 tokens | `--enable-prompt-tokens-details` |
| Prefix State | on | 4 GiB | server-managed | `--enable-prompt-tokens-details` |

Use `--no-enable-prefix-caching --mm-processor-cache-gb 0` for Frames and
Text. Use `--no-enable-prefix-caching --mm-processor-cache-gb 4` for Visual,
and load the included
[`vllm_encoder_trace`](../instrumentation/vllm_encoder_trace/README.md) patch
with `PESTATE_ENCODER_CACHE_TOKENS=65536`. The patch SHA-256 is
`211d6525ce9bb446fd900c791473dcdb1c82dcdeca586b06a5ea6584e2d9430c`.
The environment variable alone does nothing unless that `sitecustomize.py`
directory is first on `PYTHONPATH`. The patch applies
`max(vLLM_native_budget, 65536)`, so 65,536 is a required lower bound rather
than an exact capacity. The accepted Qwen HD main run requested the same
65,536 minimum and vLLM profiled an effective 153,600-token budget. Record both
values from the server log. Use `--enable-prefix-caching
--mm-processor-cache-gb 4` for Prefix State. Do not compare paths served with
a different cache contract.

After each fresh service becomes ready and before the measured runner starts,
send exactly one common, text-only warm-up request using the released
launcher's `warmup` subcommand. This request is unmetered and identical across
all four paths. It is separate from the per-video construction request below.

All four paths send local images or video through `file://` URLs. Start vLLM
with `--allowed-local-media-path` set to the parent of the runner's frame cache.
For the runner's token accounting, pass `--visual-token-counter qwen` only for
the two Qwen checkpoints and `--visual-token-counter unavailable` for the
other four models.

Visual and Prefix State each perform one question-blind, one-token warm-up per
video. Their reported construction cost includes visual encoding, language
model prefill, that generated token, and measured overhead.

## Model-specific preprocessing

### Qwen3-VL-8B and Qwen3-VL-4B

Use the processor shipped with the pinned checkpoint. Supply the 64 timestamps
as image items and use the context limit from the table. No additional dynamic
patch override was used.

### Gemma-3-12B-IT

Use the released Gemma SigLIP processor and supply the 64 timestamps as image
items. Use a 32,768-token context.

### InternVL3-8B and InternVL3.5-8B

Every timestamp is exactly one 448 x 448 patch. Start vLLM with:

```text
--mm-processor-kwargs '{"max_dynamic_patch":1,"dynamic_image_size":false}'
```

Do not enable dynamic tiling: it changes both visual token count and cost. Both
models use a 32,768-token context.

### MiniCPM-V-4.5

MiniCPM is the only main-table model served as one video input rather than 64
independent image items. The required settings are:

```text
--limit-mm-per-prompt '{"video":1}'
--default-chat-template-kwargs '{"enable_thinking":false}'
```

Pass `--visual-input-format video` to the REEL runner, use vLLM 0.15.0, and use
a 40,960-token context. Thinking must remain disabled for both construction
and answering calls.

## Generation settings

The main four-path runs use temperature 0, at most 384 answer tokens, and two
repair attempts only when the initial answer cannot be parsed as an option.
All retry tokens and latency remain charged to the question. Text uses the
Balanced24 profile with eight frames per writer chunk and a 1,024-token
frame-writer ceiling. These settings are repeated in
[`REPRODUCE.md`](REPRODUCE.md) because changing them changes the evaluated
condition.
