# Verification status

Verified on 2026-10-25 in the private release-candidate checkout.

## Passed checks

- all Python files under `src/`, `scripts/`, `tests/`, `instrumentation/`, and
  `compat/` compile;
- the complete CPU suite passes: `185 passed`;
- Ruff reports no static-check finding in the released Python source and tests;
- the aggregate package passes schema, row-count, semantic-coverage, and
  SHA-256 checks: 13 tables, 479 data rows, and 14 checked hashes;
- the strict release-tree audit scans 83 tracked or non-ignored candidate
  files with no warning and no path, credential, AI-attribution,
  symbolic-link, generated-artifact, or oversized-file finding;
- Gitleaks 8.30.1 reports no secret in the complete candidate history or in
  the current candidate tree;
- every commit on the rewritten public branch uses the GitHub-provided
  `noreply` address for both author and committer metadata;
- `CITATION.cff` validates against the official CFF 1.2.0 JSON schema;
- `git diff --check` reports no whitespace error outside the authenticated
  aggregate CSV payloads, whose CRLF bytes are intentionally preserved by
  `.gitattributes` and checked by the released SHA-256 manifest;
- an offline sdist and wheel build into a temporary output directory completes
  successfully; the sdist contains `LICENSE`, `CITATION.cff`, the released
  scripts, documentation, aggregate tables, instrumentation, and test
  fixtures; the wheel carries the Apache-2.0 license and author metadata; and
  both installed command-line entry points pass from a temporary virtual
  environment; and
- the public runner is byte-identical to the accepted working copy, with
  SHA-256
  `0d155b08119ff07b24541f4b1c94c81400ed6e1c070c89a35b430f8567e92484`.

The 64-frame final answer prompt is locked by a unit test at SHA-256
`baa64a3423f0e47e457b01e87fb8739d924424f7d9e8bcf60a0c508238e27ae9`.

Commands used:

```bash
PYTHONPATH=src:scripts python -m compileall -q src scripts tests instrumentation compat
python -m ruff check src scripts tests instrumentation compat
PYTHONPATH=src:scripts python -m pytest -q
PYTHONPATH=src python -m pestate.paper_results results/paper_tables
PYTHONPATH=src python -m pestate.release_audit .
gitleaks git .
gitleaks dir .
git diff --check -- . ':(exclude)results/paper_tables/*.csv'
```

The strict audit passes with no finding and no warning. The root license is the
standard Apache-2.0 text, and the software-only citation metadata names Xin
Liang as the author without inventing a DOI or arXiv identifier.

## Still required before publication

This receipt is not final release certification. Complete the unchecked items
in [`RELEASE_AUDIT.md`](RELEASE_AUDIT.md), including a clean-environment
build, a legally obtained GPU smoke test, and final number/attribution review.
No GPU job was run as part of this CPU verification.
