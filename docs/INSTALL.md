# Installation

Use separate environments for CPU analysis, vLLM 0.11, and vLLM 0.15. The two
GPU stacks pin incompatible PyTorch and vLLM versions and must not be merged.

## CPU analysis and tests

The CPU path needs Linux or macOS, Git, and Python 3.11 or newer. Python 3.11
is the tested analysis version. From a clean checkout:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,dev]'
python -m pytest -q
reel-verify-results
```

`reel-verify-results` checks the released CSV schemas, row counts, workload
coverage, and SHA-256 values. GNU/Linux users can independently check the same
files with:

```bash
(cd results/paper_tables && sha256sum -c SHA256SUMS)
```

These commands require no benchmark files, model weights, network API, or
GPU. To inspect the tree for release hazards, run:

```bash
reel-release-audit
```

The strict audit requires the release license and citation metadata. See
[`RELEASE_AUDIT.md`](RELEASE_AUDIT.md) for checks that still need manual
review before publication.

Building the runnable CG-Bench cohort has one additional system dependency:
`ffprobe`, which is distributed with FFmpeg. Confirm it is available before
running the cohort builder:

```bash
ffprobe -version
```

The released CPU tests and aggregate-result verification do not require
FFmpeg.

## GPU serving prerequisites

GPU reproduction was tested on Linux with one NVIDIA L40S (48 GiB) per task.
You also need:

- an NVIDIA driver compatible with the selected CUDA/PyTorch wheel;
- enough local disk for one pinned model revision;
- legally obtained CG-Bench or HD-EPIC files; and
- access to the model repositories, including any required Hugging Face terms.

The requirements files record direct runtime versions. They are not complete
cross-platform lockfiles: CUDA-tagged PyTorch wheels and low-level NVIDIA
libraries depend on the host. After installing, record `python --version`,
`python -m pip freeze`, `nvidia-smi`, and the checkpoint revision with the run.

## vLLM 0.11 environment

Use this environment for Qwen3-VL-8B, Qwen3-VL-4B, InternVL3-8B,
InternVL3.5-8B, and Gemma-3-12B-IT:

```bash
python3.11 -m venv .venv-vllm011
source .venv-vllm011/bin/activate
python -m pip install --upgrade pip
python -m pip install -r configs/environments/openvlm_vllm011_requirements.txt
python -m pip install -e .
python -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)'
```

The accepted environment reported Python 3.11.15, PyTorch 2.8.0+cu126,
Transformers 4.57.0, and vLLM 0.11.0. Install the CUDA 12.6-compatible PyTorch
wheel appropriate for the host if the default package index does not provide
that build.

## vLLM 0.15 environment

Use this environment only for MiniCPM-V-4.5:

```bash
python3.12 -m venv .venv-vllm015
source .venv-vllm015/bin/activate
python -m pip install --upgrade pip
python -m pip install -r configs/environments/openvlm_vllm015_requirements.txt
python -m pip install -e .
python -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)'
```

The accepted environment reported Python 3.12.15, PyTorch 2.9.1+cu128,
Transformers 4.57.1, and vLLM 0.15.0. Install the CUDA 12.8-compatible PyTorch
wheel appropriate for the host when necessary. `molmo-utils` is not a REEL
dependency and is intentionally absent from this environment record.

## External assets

Download each checkpoint at the full revision in [`MODELS.md`](MODELS.md).
Download benchmark files from the official sources in [`DATA.md`](DATA.md).
Keep both outside the Git checkout. Never commit access tokens, weight files,
media, generated frame caches, or path-bearing run directories.

Once the environment and assets are ready, use the service settings in
[`MODELS.md`](MODELS.md) and the path-to-mode commands in
[`REPRODUCE.md`](REPRODUCE.md).
