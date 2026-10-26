# REEL

**REEL: Same Frames, Different Answers and Different Costs**

Accepted at WACV 2027, Datasets and Evaluation Track.

REEL evaluates long-video reuse under repeated questions. It holds the
observed frames fixed, changes only the saved representation, and reports:

1. paired changes in which questions are answered correctly;
2. cumulative GPU work as more questions are asked; and
3. retained storage in bytes.

> **Private release staging.** This repository is not ready to be made public.
> The author list, license, camera-ready paper, public data package, and final
> reproduction commands still require author approval.

## Current release candidate

This first whitelist contains:

- the small `pestate` Python package used by the controlled runner;
- the controlled CG-Bench memory-policy runner;
- answer-validity and paper-analysis utilities;
- selected figure-generation scripts; and
- tests that do not require private videos or model checkpoints.

It intentionally does **not** contain raw datasets, videos, model weights,
API responses, cluster logs, full experiment outputs, third-party source
trees, or internal paper-planning documents.

## Repository layout

```text
src/pestate/              Core utilities and selectors
scripts/                  Whitelisted runner and paper-analysis scripts
tests/                    Redistributable unit tests
configs/environments/     Direct inference dependency versions
docs/                     Release, data, model, and third-party notes
results/paper_tables/     Reserved for small anonymized aggregate tables
assets/figures/           Reserved for final paper figures
examples/                 Reserved for legal smoke-test fixtures
```

## Local checks

The lightweight parser test can run without video data or a GPU:

```bash
python -m pip install -e '.[analysis,dev]'
PYTHONPATH=src:scripts python -m pytest -q tests/test_vlm_parse.py
```

The GPU inference environment is deliberately documented separately from the
analysis environment. See `configs/environments/` and `docs/MODELS.md`.

## Before public release

Read [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md). In particular, do not
publish this repository until the authors choose a code license and approve
the final author/citation metadata.
