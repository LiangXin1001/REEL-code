# REEL

**REEL: Same Frames, Different Answers and Different Costs**

Accepted at WACV 2027, Datasets and Evaluation Track.

REEL studies repeated-question long-video inference. For a fixed set of 64
question-blind timestamps, it changes only what is retained after the video is
first processed. It then measures answer changes, cumulative GPU work, and
retained bytes.

> **This is still a private release candidate.** REEL-authored code is
> licensed under Apache-2.0, and software citation metadata is included.
> Keep the repository private until the remaining checklist in
> [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md) is complete.

## Evaluated representations

| Paper name | Runner mode | Retained state |
| --- | --- | --- |
| Frames | `query_time_raw` | 64 decoded frames are supplied again for every question |
| Text | `write_once_frame_text` | one question-blind Balanced24 text record per frame |
| Visual | `write_once_visual_token_cache` | reusable multimodal encoder outputs, without prefix KV reuse |
| Prefix State | `write_once_model_state_cache` | multimodal encoder outputs plus reusable prefix KV state |

The main evaluation uses the same video IDs, question IDs, question order,
answer choices, and timestamps across these four paths. See
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the protocol and runner mapping.

## What is included

- the controlled runner and the small `pestate` utility package;
- the pinned vLLM encoder-cache instrumentation used by the Visual path;
- CPU-only cohort audit/build/materialization tools for CG-Bench and HD-EPIC;
- CPU-only analysis and figure-generation scripts;
- redistributable tests and synthetic test fixtures;
- direct dependency versions for the two serving environments; and
- de-identified aggregate paper tables in `results/paper_tables/`.

The repository does **not** redistribute CG-Bench or HD-EPIC videos or
annotations, model weights, per-question predictions, raw model responses,
private prompt-audit artifacts, private run manifests, cluster logs, or
third-party source trees. The formal prompt templates needed by the runner are
included in the released source.

Obtain the data and checkpoints from their owners under their original terms:

- [`docs/DATA.md`](docs/DATA.md) records the two benchmark cohorts and sources.
- [`docs/MODELS.md`](docs/MODELS.md) pins all six checkpoint revisions and
  serving settings.
- [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) records external code and
  attribution boundaries.

## Quick CPU verification

Python 3.11 is recommended for the analysis environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,dev]'
python -m pytest -q
reel-verify-results
(cd results/paper_tables && sha256sum -c SHA256SUMS)
```

These checks do not download videos or checkpoints and do not require a GPU.
Detailed CPU and GPU installation instructions are in
[`docs/INSTALL.md`](docs/INSTALL.md).

## Reproduction status

The aggregate result package is self-checking and can be inspected without
private assets. Full GPU reproduction additionally requires legally obtained
benchmark media and annotations, the pinned model revisions, and a locally
built manifest matching the published cohort provenance. The cohort builders
are included, but the canonical manifest contents and a legally
redistributable GPU smoke fixture are not tracked here. CG selection IDs and
order are reproducible; its byte hash additionally depends on the same local
path and JSON-serialization convention. The repository therefore does not
claim a download-free, one-command reproduction. See
[`docs/DATA.md`](docs/DATA.md) for commands and hashes.

## Repository layout

```text
src/pestate/              Core utilities and selectors
scripts/                  Cohort tools, controlled runner, and analysis scripts
instrumentation/          Pinned vLLM Visual-cache patch and usage notes
tests/                    CPU tests and synthetic fixtures
configs/environments/     Direct GPU-serving dependency versions
docs/                     Installation and reproducibility records
results/paper_tables/     De-identified aggregate paper tables
```

## Citation

Until the paper is public, cite this repository as software using
[`CITATION.cff`](CITATION.cff). It intentionally contains no DOI or arXiv ID.
The camera-ready paper citation and public URL will be added when available.

## License

REEL-authored code is licensed under the [Apache License 2.0](LICENSE). The
software author listed in the release metadata is Xin Liang. Third-party works
retain their own licenses; see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

Before changing the GitHub repository's visibility, resolve every remaining
item in [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md), including the
clean-environment build, legal GPU smoke test, and final number and
attribution review.
