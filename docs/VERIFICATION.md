# Verification status

Verified on 2026-10-18 in the private staging workspace:

- all Python files under `src/`, `scripts/`, and `tests/` compile;
- 76 selected CPU tests pass;
- no symbolic links are present;
- no file is larger than 10 MB;
- the current `src/`, `scripts/`, `tests/`, and `configs/` whitelist contains
  no `/data/...`, `/u/...`, username, or lab-path matches; and
- a first-pass scan found no common private-key, GitHub-token, AWS-key, or
  OpenAI-style key pattern.

Test command:

```bash
PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_vlm_parse.py \
  tests/test_aks.py \
  tests/test_time_utils.py \
  tests/test_controlled_aks_manifest.py \
  tests/test_controlled_timestamp_manifest.py \
  tests/test_hdepic_reference_protocol.py \
  tests/test_make_glm_tariff_paper_assets.py \
  tests/test_strict_question_type_heatmaps.py \
  tests/test_make_representation_payload_figure.py
```

Result: `76 passed`.

This is a staging check, not final release certification. GPU inference,
end-to-end dataset loading, full paper reproduction, license approval, and a
dedicated full-history secret scanner remain outstanding.
