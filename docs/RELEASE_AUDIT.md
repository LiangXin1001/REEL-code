# Public-release audit

## Release boundary

This repository is the only candidate for the public REEL code release. It
was created from a whitelist with a new Git history. The private PEState
working repository must never be made public, forked for release, or copied
with a broad command such as `git add -A`.

The release candidate contains:

- the small `pestate` package and controlled experiment runner;
- the exact REEL-authored vLLM encoder-cache instrumentation used by Visual;
- CPU analysis and figure-generation scripts;
- tests that do not require private assets;
- exact direct dependency versions for the two GPU serving environments; and
- small, de-identified aggregate result tables.

It intentionally excludes raw videos and annotations, model weights, model
responses, per-question records, cluster launch files and logs, private paper
sources, third-party source trees, and the private experiment workspace.

## Why the working repository is not releasable

The working repository mixes source code with several gigabytes of research
state. Its history and working tree include licensed assets, nested upstream
repositories, job logs, checkpoints, API artifacts, and files containing
personal cluster paths. Deleting those files in a later commit would not
remove them from Git history.

## Generated-file warning

Several experiment scripts record resolved input and output paths in their
provenance manifests. This is useful during a run, but those generated
manifests can expose a local username or cluster layout. Do not commit a run
directory directly. Reduce it to an aggregate table, remove path-bearing
columns, then run both verification commands below.

## Automated checks

From the repository root:

```bash
python -m pip install -e '.[analysis,dev]'
python -m pytest -q
python -m ruff check src scripts tests instrumentation compat
reel-verify-results
reel-release-audit
```

The audit checks the release tree for symbolic links, oversized files,
generated build products, common credential formats, private absolute path
patterns, and AI-author attribution comments. It complements, but does not
replace, a dedicated full-history secret scanner.

## Required before changing GitHub visibility

- [x] Add the author-approved Apache-2.0 `LICENSE`.
- [x] Add author-approved software citation metadata in `CITATION.cff`.
- [x] Rewrite the candidate history so every public-branch commit uses the
      GitHub-provided `noreply` address.
- [x] Run the strict `reel-release-audit` check.
- [x] Run Gitleaks over the complete new history and current candidate tree.
- [ ] Repeat the install, CPU tests, aggregate-result verification, and package
      build in a clean environment.
- [ ] Document one successful GPU smoke test using legally obtained data.
- [ ] Complete a second manual review of every tracked file and third-party
      attribution.
- [ ] Confirm that released numbers match the camera-ready manuscript.

The public paper URL and preferred paper citation should be added after they
exist. Their absence does not block a software-only release citation.

Do not change repository visibility, create a public release, or upload public
assets until every item above is resolved and approved by an author. A push to
the private staging repository should still be a separate, reviewed action.
