# Public-release audit

## Decision

Create a new repository with a new Git history and copy files from a
whitelist. Do not make the working PEState repository public, do not fork it
for release, and do not run `git add -A` from that repository.

## Why the working repository cannot be uploaded

The working tree is about 5.5 GB and mixes source code with research state:

- `outputs/` is about 3.6 GB and contains logs, model responses, audit
  materials, API-batch artifacts, and thousands of stale symbolic links to
  personal cluster paths;
- `third_party/` is about 1.6 GB and contains nested repositories, videos,
  annotations, checkpoints, and projects whose redistribution terms differ;
- `data/`, `models/`, environment directories, literature PDFs, archives, and
  paper working files should not be copied wholesale;
- many SLURM scripts and some configs contain `/data/...` or `/u/...` paths;
  and
- deleting a sensitive or internal file in a later commit does not remove it
  from the old repository history.

The first-pass pattern scan did not find a common high-confidence private-key
or API-token format in the whitelisted text files. This is not a guarantee.

## Current whitelist

- `src/pestate/*.py`
- `scripts/run_controlled_memory_policy_cgbench.py`
- `scripts/lifecycle_answer_validity.py`
- `scripts/make_paper_diagram_figures.py`
- `scripts/make_paper_result_figures.py`
- `scripts/make_strict_question_type_heatmaps.py`
- `scripts/analyze_hd_all_question_costs.py`
- `scripts/make_strict_internvl_resource_profile.py`
- `scripts/make_glm_tariff_paper_assets.py`
- `scripts/make_representation_payload_figure.py`
- `scripts/paper_plot_style.py`
- selected tests that exercise these files without private assets

This is a starting point, not proof that every paper result is reproducible
from the current repository.

## Never copy wholesale

- `outputs/`, `data/`, `models/`, `third_party/`, `third_party_env/`, `.deps/`
- raw or licensed videos and annotations
- model caches, weights, API responses, and batch request/result files
- SLURM logs and job-specific launch scripts
- internal research plans, audit packs, writing studies, and reviewer material
- Overleaf ZIP files and LaTeX build artifacts
- literature PDFs or other copyrighted papers

## Required before changing GitHub visibility to public

- [ ] Authors choose and approve the repository license.
- [ ] Camera-ready author order and citation are added.
- [ ] Every candidate file receives a second manual path/secret review.
- [ ] `gitleaks` or an equivalent secret scanner passes on the full new history.
- [ ] No tracked file exceeds the intended repository size limit.
- [ ] Third-party attribution and licenses are complete.
- [ ] A clean-environment install and CPU smoke test pass.
- [ ] At least one documented GPU smoke test passes on legally obtained data.
- [ ] Released aggregate tables contain no private paths or raw model output.
- [ ] Website numbers match the camera-ready manuscript.
