# Third-party code and resources

Third-party repositories, datasets, and model checkpoints are referenced, not
vendored. This document describes the boundary of the release candidate; it
does not replace any upstream license or access agreement.

## MaxVol fallback

`src/pestate/maxvol.py` contains a small NumPy fallback adapted from
`maxvolpy` 0.3.8 by Alexander Mikhalev.

- Upstream package: <https://pypi.org/project/maxvolpy/0.3.8/>
- Upstream license: MIT
- Included attribution: the source file retains the upstream copyright notice
  and complete MIT License text
- REEL changes: deterministic initialization, recomputation after row swaps,
  and only the subset needed by the blockwise selector

No `maxvolpy` package or source tree is copied into the repository beyond the
attributed fallback in that file.

## Adaptive Keyframe Sampling (AKS)

`src/pestate/aks.py` implements the selection rule described in
*Adaptive Keyframe Sampling for Long Video Understanding*.

- Paper: <https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html>
- Authors' repository: <https://github.com/ncTimTang/AKS>

The REEL file is an independent implementation and does not copy the authors'
released source. It is not described as a formally isolated clean-room
implementation. The locally audited upstream snapshot had no top-level
license file, so no upstream AKS source is redistributed here. Users who run
the authors' implementation must inspect its current terms themselves.

## MaxInfo reference

The question-blind diversity selector is informed by the public MaxInfo work:

- Project repository: <https://github.com/FusionBrainLab/MaxInfo>

No MaxInfo source tree or model artifact is included. REEL's released selector
utilities and provenance checks live in this repository; users must obtain any
external MaxInfo components separately and follow their current terms.

## Models and serving libraries

The evaluated checkpoints are third-party works. Their code, weights, model
cards, and licenses are not redistributed. Exact official repositories and
revisions are listed in [`MODELS.md`](MODELS.md).

REEL uses external packages including:

- vLLM: <https://github.com/vllm-project/vllm>
- OpenAI Python client: <https://github.com/openai/openai-python>
- PyTorch: <https://github.com/pytorch/pytorch>
- Transformers: <https://github.com/huggingface/transformers>

They are installed as dependencies under their own licenses. The two direct
serving-version records in `configs/environments/` do not vendor those
packages.

`instrumentation/vllm_encoder_trace/sitecustomize.py` is REEL-authored runtime
instrumentation that imports and monkey-patches private vLLM interfaces. It
does not copy vLLM source. It is pinned to the tested vLLM 0.11.0/0.15.0 APIs;
using it with another version requires a new compatibility and cache-reuse
audit.

## Benchmarks

CG-Bench and HD-EPIC media, annotations, subtitles, and manifests are not
included. Official sources and cohort provenance are recorded in
[`DATA.md`](DATA.md). No benchmark asset may be added as a smoke fixture
without explicit redistribution permission.

## Baseline repositories

No baseline repository is copied into this release. The released AKS and
MaxInfo references above are links and local interoperability records, not
bundled upstream checkouts. Private comparison environments, local patches,
and nested Git repositories remain outside the release boundary.

## REEL's own license

REEL-authored code is licensed under Apache-2.0; see the root
[`LICENSE`](../LICENSE). Third-party components retain their own licenses. In
particular, the MaxVol-derived file retains its embedded MIT notice. The root
license does not relicense datasets, model weights, benchmark assets, or other
external works that are not distributed by this repository.
