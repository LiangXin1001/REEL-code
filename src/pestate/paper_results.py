"""Validate the small, public aggregate tables released with REEL.

The verifier intentionally does not read raw answers, videos, or model logs.
Each released CSV is described by ``manifest.json`` and authenticated by the
independent ``SHA256SUMS`` file in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = "reel.paper_tables.v1"

# These are the camera-ready aggregate tables.  The manifest remains the
# source of truth for columns and may describe additional public CSVs, but it
# may not omit or change the row count of a canonical table.
CANONICAL_ROW_COUNTS: dict[str, int] = {
    "main_table_selection.csv": 12,
    "main_four_path_metrics.csv": 48,
    "cost_sensitivity.csv": 36,
    "selection_paths.csv": 12,
    "cost_components.csv": 48,
    "quality_profiles_vs_frames.csv": 48,
    "clue_window_interactions.csv": 3,
    "question_type_by_official_type.csv": 168,
    "representation_storage.csv": 24,
    "storage_budget_endpoint_counts.csv": 8,
    "cg_uniform32_uniform64_four_path.csv": 24,
    "summary_ablation_frames_agreement.csv": 12,
    "batch_timing_variation.csv": 36,
}
EXPECTED_REPRESENTATIONS = frozenset({"frames", "balanced24", "visual", "prefix"})
EXPECTED_COST_CONTRASTS = frozenset(
    {"visual-frames", "prefix-frames", "prefix-visual"}
)

_SHA256_LINE = re.compile(r"^([0-9a-f]{64})[ \t]+\*?(.+)$")


class ResultVerificationError(ValueError):
    """Raised when a released result bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class ResultVerificationReport:
    results_dir: str
    table_count: int
    total_rows: int
    checked_hashes: int
    schema_version: str = SCHEMA_VERSION
    accepted: bool = True


@dataclass(frozen=True)
class _Table:
    name: str
    header: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_member(root: Path, name: str) -> Path:
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ResultVerificationError(f"result bundle member is a symbolic link: {name}")
    if not path.is_file():
        raise ResultVerificationError(f"result bundle member is missing or not regular: {name}")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ResultVerificationError(f"result bundle member escapes its directory: {name}") from exc
    return path


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultVerificationError(f"{label} must be a non-empty path string")
    if value != value.strip() or "\\" in value:
        raise ResultVerificationError(f"{label} is not a canonical POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ResultVerificationError(f"{label} must stay inside the result directory: {value!r}")
    if path.as_posix() != value:
        raise ResultVerificationError(f"{label} is not canonical: {value!r}")
    return path.as_posix()


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ResultVerificationError(f"missing regular manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultVerificationError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ResultVerificationError(
            f"manifest schema_version must be {SCHEMA_VERSION!r}"
        )
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ResultVerificationError("manifest.files must be a non-empty object")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, spec in files.items():
        name = _safe_relative_path(raw_name, label="manifest file")
        if not name.endswith(".csv"):
            raise ResultVerificationError(f"manifest entry is not a CSV: {name}")
        if not isinstance(spec, dict):
            raise ResultVerificationError(f"manifest spec must be an object: {name}")
        rows = spec.get("rows")
        columns = spec.get("required_columns")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ResultVerificationError(f"manifest rows must be a non-negative integer: {name}")
        if (
            not isinstance(columns, list)
            or not columns
            or any(
                not isinstance(column, str)
                or not column.strip()
                or column != column.strip()
                for column in columns
            )
            or len(columns) != len(set(columns))
        ):
            raise ResultVerificationError(
                f"manifest required_columns must be unique non-empty strings: {name}"
            )
        normalized[name] = {"rows": rows, "required_columns": tuple(columns)}

    missing = sorted(set(CANONICAL_ROW_COUNTS) - set(normalized))
    if missing:
        raise ResultVerificationError(f"manifest omits canonical tables: {missing}")
    changed = {
        name: {"manifest": normalized[name]["rows"], "expected": expected}
        for name, expected in CANONICAL_ROW_COUNTS.items()
        if normalized[name]["rows"] != expected
    }
    if changed:
        raise ResultVerificationError(f"canonical row counts changed: {changed}")
    return normalized


def _load_checksums(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ResultVerificationError(f"missing regular checksum file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultVerificationError(f"cannot read checksum file {path}: {exc}") from exc
    if not lines:
        raise ResultVerificationError("SHA256SUMS is empty")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise ResultVerificationError(
                f"SHA256SUMS:{line_number} is not '<sha256>  <relative-path>'"
            )
        digest, raw_name = match.groups()
        name = _safe_relative_path(raw_name, label=f"SHA256SUMS:{line_number}")
        if name in checksums:
            raise ResultVerificationError(f"duplicate SHA256SUMS entry: {name}")
        checksums[name] = digest
    return checksums


def _read_table(path: Path, spec: dict[str, Any]) -> _Table:
    if not path.is_file() or path.is_symlink():
        raise ResultVerificationError(f"missing regular table: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ResultVerificationError(f"CSV is empty: {path.name}") from exc
            if not header or any(not column.strip() for column in header):
                raise ResultVerificationError(f"CSV has an empty header field: {path.name}")
            if len(header) != len(set(header)):
                raise ResultVerificationError(f"CSV has duplicate columns: {path.name}")
            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise ResultVerificationError(
                        f"{path.name}:{line_number} has {len(values)} fields; expected {len(header)}"
                    )
                if all(not value.strip() for value in values):
                    raise ResultVerificationError(f"{path.name}:{line_number} is blank")
                rows.append(dict(zip(header, values)))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ResultVerificationError(f"cannot read CSV {path}: {exc}") from exc

    required = set(spec["required_columns"])
    missing = sorted(required - set(header))
    if missing:
        raise ResultVerificationError(f"{path.name} is missing required columns: {missing}")
    if len(rows) != spec["rows"]:
        raise ResultVerificationError(
            f"{path.name} has {len(rows)} data rows; expected {spec['rows']}"
        )
    return _Table(path.name, tuple(header), tuple(rows))


def _column(table: _Table, aliases: Iterable[str]) -> str:
    for name in aliases:
        if name in table.header:
            return name
    raise ResultVerificationError(
        f"{table.name} needs one of the semantic columns {tuple(aliases)}"
    )


def _values(table: _Table, column: str) -> set[str]:
    values = {row[column].strip() for row in table.rows}
    if "" in values:
        raise ResultVerificationError(f"{table.name}.{column} contains an empty value")
    return values


def _combos(table: _Table) -> set[tuple[str, str]]:
    dataset = _column(table, ("dataset", "benchmark"))
    model = _column(table, ("model", "model_id"))
    pairs = {(row[dataset].strip(), row[model].strip()) for row in table.rows}
    if any(not left or not right for left, right in pairs):
        raise ResultVerificationError(f"{table.name} has an empty dataset/model identity")
    return pairs


def _verify_semantics(tables: dict[str, _Table]) -> None:
    main = tables["main_table_selection.csv"]
    main_pairs = _combos(main)
    dataset = _column(main, ("dataset", "benchmark"))
    model = _column(main, ("model", "model_id"))
    datasets = _values(main, dataset)
    models = _values(main, model)
    expected_pairs = set(itertools.product(datasets, models))
    if len(datasets) != 2 or len(models) != 6 or main_pairs != expected_pairs:
        raise ResultVerificationError(
            "main_table_selection.csv must contain the full 2-dataset x 6-model matrix"
        )
    if len(main.rows) != len(main_pairs):
        raise ResultVerificationError("main_table_selection.csv repeats a dataset/model row")

    selection_pairs = _combos(tables["selection_paths.csv"])
    if len(selection_pairs) != 12 or selection_pairs != main_pairs:
        raise ResultVerificationError("selection_paths.csv does not match the 12 main workloads")

    four_path = tables["main_four_path_metrics.csv"]
    four_pairs = _combos(four_path)
    representation = _column(
        four_path, ("representation", "path", "condition", "candidate")
    )
    representations = _values(four_path, representation)
    triples = {
        (dataset_name, model_name, row[representation].strip())
        for row in four_path.rows
        for dataset_name, model_name in [
            (
                row[_column(four_path, ("dataset", "benchmark"))].strip(),
                row[_column(four_path, ("model", "model_id"))].strip(),
            )
        ]
    }
    if (
        four_pairs != main_pairs
        or representations != EXPECTED_REPRESENTATIONS
        or len(triples) != 48
        or any(
            {path for dataset_name, model_name, path in triples if (dataset_name, model_name) == pair}
            != representations
            for pair in main_pairs
        )
    ):
        raise ResultVerificationError(
            "main_four_path_metrics.csv must contain four paths for each of the 12 workloads"
        )

    sensitivity = tables["cost_sensitivity.csv"]
    sensitivity_pairs = _combos(sensitivity)
    contrast = _column(sensitivity, ("contrast", "comparison", "cost_contrast"))
    contrasts = _values(sensitivity, contrast)
    sensitivity_dataset = _column(sensitivity, ("dataset", "benchmark"))
    sensitivity_model = _column(sensitivity, ("model", "model_id"))
    sensitivity_triples = {
        (
            row[sensitivity_dataset].strip(),
            row[sensitivity_model].strip(),
            row[contrast].strip(),
        )
        for row in sensitivity.rows
    }
    if (
        sensitivity_pairs != main_pairs
        or contrasts != EXPECTED_COST_CONTRASTS
        or len(sensitivity_triples) != 36
        or any(
            {
                comparison
                for dataset_name, model_name, comparison in sensitivity_triples
                if (dataset_name, model_name) == pair
            }
            != contrasts
            for pair in main_pairs
        )
    ):
        raise ResultVerificationError(
            "cost_sensitivity.csv must contain three contrasts for each of the 12 workloads"
        )


def verify_results(results_dir: str | Path) -> ResultVerificationReport:
    """Validate one public result bundle and return its compact receipt."""

    root = Path(results_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ResultVerificationError(f"result directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    specs = _load_manifest(manifest_path)
    observed_csvs = {
        path.relative_to(root).as_posix() for path in root.rglob("*.csv")
    }
    extra_csvs = sorted(observed_csvs - set(specs))
    if extra_csvs:
        raise ResultVerificationError(
            f"CSV files are present but absent from manifest.json: {extra_csvs}"
        )
    checksums = _load_checksums(root / "SHA256SUMS")
    expected_hashes = set(specs) | {"manifest.json"}
    if set(checksums) != expected_hashes:
        missing = sorted(expected_hashes - set(checksums))
        extra = sorted(set(checksums) - expected_hashes)
        raise ResultVerificationError(
            f"SHA256SUMS coverage mismatch: missing={missing}, extra={extra}"
        )
    for name, expected in checksums.items():
        path = _regular_member(root, name)
        observed = _sha256(path)
        if observed != expected:
            raise ResultVerificationError(
                f"SHA-256 mismatch for {name}: expected={expected}, observed={observed}"
            )

    tables = {
        name: _read_table(_regular_member(root, name), spec)
        for name, spec in specs.items()
    }
    _verify_semantics(tables)
    return ResultVerificationReport(
        results_dir=str(root),
        table_count=len(tables),
        total_rows=sum(len(table.rows) for table in tables.values()),
        checked_hashes=len(checksums),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path("results/paper_tables"),
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON receipt.")
    args = parser.parse_args(argv)
    try:
        report = verify_results(args.results_dir)
    except ResultVerificationError as exc:
        if args.json:
            print(json.dumps({"accepted": False, "error": str(exc)}, indent=2))
        else:
            print(f"REEL result verification failed: {exc}", file=sys.stderr)
        return 1
    payload = asdict(report)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "REEL result verification passed: "
            f"tables={report.table_count} rows={report.total_rows} "
            f"hashes={report.checked_hashes}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
