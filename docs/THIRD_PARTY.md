# Third-party code and resources

This file must be completed before public release. Third-party repositories,
datasets, and model checkpoints are referenced rather than vendored.

## MaxVol fallback

`src/pestate/maxvol.py` contains a small adapted fallback based on
`maxvolpy` 0.3.8 by Alexander Mikhalev. The source file itself retains the
upstream copyright notice and complete MIT License text, as required.

- Upstream: https://pypi.org/project/maxvolpy/0.3.8/
- License: MIT
- Changes: deterministic initialization and recomputation after row swaps

## Adaptive Keyframe Sampling (AKS)

`src/pestate/aks.py` is an independent reimplementation of the selection rule
described in *Adaptive Keyframe Sampling for Long Video Understanding*.
It does not copy the authors' released source, but this repository does not
claim a formally isolated clean-room process. The upstream repository is not
vendored here and currently has no top-level license in the local audited
snapshot.

## Models and serving libraries

REEL evaluates third-party vision-language models and uses external serving
libraries. Their code, weights, and licenses are not redistributed here.
Users must obtain each checkpoint under its original terms. Exact revisions
will be listed in `docs/MODELS.md` after camera-ready verification.

## Benchmarks

CG-Bench and HD-EPIC assets are not included. Users must download them from
their official sources and comply with their licenses or access agreements.

## Baseline repositories

No baseline repository should be copied into this repository. The final
version of this file should record, for each baseline, its official URL,
tested commit, license, and any local patch applied by REEL.
