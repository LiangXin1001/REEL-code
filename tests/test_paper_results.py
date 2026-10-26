from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from pestate.paper_results import (
    CANONICAL_ROW_COUNTS,
    ResultVerificationError,
    main,
    verify_results,
)


DATASETS = ("CG", "HD")
MODELS = tuple(f"model-{index}" for index in range(6))
REPRESENTATIONS = ("frames", "balanced24", "visual", "prefix")
CONTRASTS = ("visual-frames", "prefix-frames", "prefix-visual")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _semantic_rows(name: str) -> list[dict[str, object]] | None:
    workloads = [(dataset, model) for dataset in DATASETS for model in MODELS]
    if name in {"main_table_selection.csv", "selection_paths.csv"}:
        return [
            {"dataset": dataset, "model": model, "selection": "Prefix State"}
            for dataset, model in workloads
        ]
    if name == "main_four_path_metrics.csv":
        return [
            {
                "dataset": dataset,
                "model": model,
                "representation": representation,
                "gpu_seconds": index,
            }
            for dataset, model in workloads
            for index, representation in enumerate(REPRESENTATIONS)
        ]
    if name == "cost_sensitivity.csv":
        return [
            {
                "dataset": dataset,
                "model": model,
                "contrast": contrast,
                "difference": index,
            }
            for dataset, model in workloads
            for index, contrast in enumerate(CONTRASTS)
        ]
    return None


def _rewrite_checksums(root: Path) -> None:
    names = sorted(
        path.name for path in root.iterdir() if path.suffix == ".csv"
    ) + ["manifest.json"]
    lines = [
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}"
        for name in names
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bundle(root: Path) -> None:
    root.mkdir()
    files: dict[str, dict[str, object]] = {}
    for name, expected_rows in CANONICAL_ROW_COUNTS.items():
        rows = _semantic_rows(name)
        if rows is None:
            rows = [
                {"row_id": index, "value": f"value-{index}"}
                for index in range(expected_rows)
            ]
        _write_csv(root / name, rows)
        files[name] = {
            "rows": expected_rows,
            "required_columns": list(rows[0]),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "reel.paper_tables.v1", "files": files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _rewrite_checksums(root)


def test_valid_bundle_checks_schema_rows_semantics_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "paper_tables"
    _write_bundle(root)

    report = verify_results(root)

    assert report.accepted is True
    assert report.table_count == 13
    assert report.total_rows == sum(CANONICAL_ROW_COUNTS.values()) == 479
    assert report.checked_hashes == 14


def test_console_command_emits_machine_readable_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "paper_tables"
    _write_bundle(root)

    assert main([str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is True
    assert payload["table_count"] == 13


def test_tampered_table_is_rejected_before_use(tmp_path: Path) -> None:
    root = tmp_path / "paper_tables"
    _write_bundle(root)
    with (root / "main_table_selection.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered,row,value\n")

    with pytest.raises(ResultVerificationError, match="SHA-256 mismatch"):
        verify_results(root)


def test_duplicate_workload_fails_even_with_updated_checksum(tmp_path: Path) -> None:
    root = tmp_path / "paper_tables"
    _write_bundle(root)
    rows = _semantic_rows("main_table_selection.csv")
    assert rows is not None
    rows[-1] = dict(rows[0])
    _write_csv(root / "main_table_selection.csv", rows)
    _rewrite_checksums(root)

    with pytest.raises(ResultVerificationError, match="full 2-dataset x 6-model matrix"):
        verify_results(root)


def test_checksum_paths_cannot_escape_result_directory(tmp_path: Path) -> None:
    root = tmp_path / "paper_tables"
    _write_bundle(root)
    with (root / "SHA256SUMS").open("a", encoding="utf-8") as handle:
        handle.write("0" * 64 + "  ../private.csv\n")

    with pytest.raises(ResultVerificationError, match="stay inside"):
        verify_results(root)


def test_unmanifested_csv_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "paper_tables"
    _write_bundle(root)
    (root / "unreviewed.csv").write_text("name,value\nextra,1\n", encoding="utf-8")

    with pytest.raises(ResultVerificationError, match="absent from manifest"):
        verify_results(root)
