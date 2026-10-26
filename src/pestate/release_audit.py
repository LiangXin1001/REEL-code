"""Audit a REEL checkout before changing its GitHub visibility to public.

This is a current-tree safety check, not a substitute for a full-history
scanner such as gitleaks.  In a Git checkout it scans tracked files plus
untracked, non-ignored files: the same set most likely to enter the next
commit.  Outside Git it scans the supplied directory recursively.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024

_GENERATED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".deps",
        "build",
        "cache",
        "checkpoints",
        "data",
        "dist",
        "models",
        "outputs",
        "paper",
        "runs",
        "logs",
        "third_party",
        "third_party_env",
        "wandb",
    }
)
_GENERATED_FILE_NAMES = frozenset({".coverage", ".DS_Store", ".env", ".netrc"})
_GENERATED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".egg-info",
    ".safetensors",
    ".ckpt",
    ".key",
    ".pem",
    ".pth",
    ".pt",
    ".token",
    ".bin",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
)

# These formats are especially likely to be benchmark inputs or generated run
# artifacts.  Size and secret scans are not sufficient protection: a small,
# path-relative JSONL manifest or one extracted frame can still disclose
# licensed question/video content without containing a credential or private
# absolute path.
_SENSITIVE_DATA_SUFFIXES = (
    ".jsonl",
    ".ndjson",
    ".mp4",
    ".mkv",
    ".webm",
    ".avi",
    ".mov",
    ".m4v",
    ".flv",
    ".wmv",
    ".srt",
    ".vtt",
    ".ass",
    ".ssa",
    ".sub",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)
_README_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}
)

_PRIVATE_UNIX_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:data|u|home|scratch|mnt)/"
    r"(?!\.\.\.)(?![<{\$])[A-Za-z0-9._-]+(?:/[^\s\"'`<>]*)?"
)
_PRIVATE_WINDOWS_PATH = re.compile(
    r"\b[A-Za-z]:\\Users\\(?!Public(?:\\|\b))[^\\\s\"'`<>]+(?:\\[^\s\"'`<>]*)?"
)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "credential_in_url",
        re.compile(r"https?://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
    ),
)
_GENERIC_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\b"
    r"\s*[:=]\s*[\"']([^\"']{8,})[\"']"
)
_AI_ATTRIBUTION = re.compile(
    r"(?i)\bby\s*:?[ \t]*(?:claude|codex|chatgpt|copilot|gemini)\b"
)
_PLACEHOLDER_CREDENTIALS = frozenset(
    {
        "changeme",
        "dummy",
        "empty",
        "example",
        "none",
        "placeholder",
        "redacted",
        "test-token",
        "your-api-key",
    }
)


@dataclass(frozen=True)
class AuditFinding:
    code: str
    path: str
    message: str
    line: int | None = None


@dataclass(frozen=True)
class ReleaseAuditReport:
    root: str
    scanned_files: int
    findings: tuple[AuditFinding, ...]
    warnings: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "root": self.root,
            "scanned_files": self.scanned_files,
            "findings": [asdict(finding) for finding in self.findings],
            "warnings": list(self.warnings),
        }


def _candidate_paths(root: Path) -> tuple[list[Path], bool]:
    """Return public-candidate paths and whether Git supplied the list."""

    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            paths = []
            for raw_name in result.stdout.split(b"\0"):
                if not raw_name:
                    continue
                name = os.fsdecode(raw_name)
                path = root / name
                # ``git ls-files --cached`` also reports tracked files that
                # have been deleted from the working tree. They cannot enter
                # the next commit and should not be treated as irregular
                # files. Keep broken symlinks so the symlink check can still
                # reject them.
                if not path.exists() and not path.is_symlink():
                    continue
                paths.append(path)
            return sorted(set(paths), key=lambda path: path.as_posix()), True

    paths = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_file() or path.is_symlink():
            paths.append(path)
    return sorted(paths, key=lambda path: path.as_posix()), False


def _generated_reason(relative: Path) -> str | None:
    parts = relative.parts
    for part in parts:
        if part in _GENERATED_DIRECTORY_NAMES or part.endswith(".egg-info"):
            return f"generated directory {part!r}"
    name = relative.name
    lower_name = name.lower()
    if name in _GENERATED_FILE_NAMES or name.startswith(".coverage."):
        return f"generated file {name!r}"
    if name.startswith("slurm-") and name.endswith((".out", ".err")):
        return f"cluster log {name!r}"
    sensitive_suffix = next(
        (suffix for suffix in _SENSITIVE_DATA_SUFFIXES if lower_name.endswith(suffix)),
        None,
    )
    if sensitive_suffix is not None:
        parts = relative.parts
        explicit_public_fixture = parts[:3] == ("tests", "fixtures", "public")
        documentation_asset = (
            sensitive_suffix in _README_IMAGE_SUFFIXES
            and parts[:2] == ("docs", "assets")
        )
        if explicit_public_fixture or documentation_asset:
            return None
        return (
            f"unreviewed manifest, media, subtitle, or extracted-frame artifact "
            f"with suffix {sensitive_suffix!r}; only reviewed synthetic files under "
            "tests/fixtures/public/ and documentation images under docs/assets/ are allowed"
        )
    if lower_name.endswith(_GENERATED_SUFFIXES):
        return f"generated or large runtime artifact {name!r}"
    return None


def _text_findings(path: Path, relative: Path) -> Iterable[AuditFinding]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        yield AuditFinding("unreadable_file", relative.as_posix(), str(exc))
        return
    if b"\0" in raw:
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _AI_ATTRIBUTION.search(line):
            yield AuditFinding(
                "ai_attribution",
                relative.as_posix(),
                "contains an AI-author attribution that should not appear in released source",
                line_number,
            )
        if _PRIVATE_UNIX_PATH.search(line) or _PRIVATE_WINDOWS_PATH.search(line):
            yield AuditFinding(
                "private_path",
                relative.as_posix(),
                "contains a user- or cluster-specific absolute path",
                line_number,
            )
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                yield AuditFinding(
                    "secret_pattern",
                    relative.as_posix(),
                    f"matches a {label} pattern",
                    line_number,
                )
        generic = _GENERIC_SECRET.search(line)
        if generic is not None:
            value = generic.group(1).strip().lower()
            if value not in _PLACEHOLDER_CREDENTIALS:
                yield AuditFinding(
                    "secret_assignment",
                    relative.as_posix(),
                    "contains a non-placeholder credential assignment",
                    line_number,
                )


def audit_release(
    root: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ReleaseAuditReport:
    """Audit a prospective public checkout without changing it."""

    release_root = Path(root).resolve()
    if not release_root.is_dir():
        raise ValueError(f"release root is not a directory: {release_root}")
    if isinstance(max_file_bytes, bool) or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")

    findings: list[AuditFinding] = []
    warnings: list[str] = []
    license_path = release_root / "LICENSE"
    citation_path = release_root / "CITATION.cff"

    def nonempty_regular(path: Path) -> bool:
        try:
            return path.is_file() and not path.is_symlink() and bool(path.read_bytes().strip())
        except OSError:
            return False

    if not nonempty_regular(license_path):
        findings.append(
            AuditFinding("missing_license", "LICENSE", "a public release needs a final LICENSE")
        )
    if not nonempty_regular(citation_path):
        findings.append(
            AuditFinding(
                "missing_citation",
                "CITATION.cff",
                "a public release needs final citation metadata",
            )
        )

    paths, from_git = _candidate_paths(release_root)
    if not from_git:
        warnings.append("Git file discovery was unavailable; scanned the directory tree")
    scanned = 0
    for path in paths:
        relative = path.relative_to(release_root)
        display = relative.as_posix()
        scanned += 1
        if path.is_symlink():
            findings.append(
                AuditFinding("symbolic_link", display, "symbolic links are not allowed")
            )
            continue
        if not path.is_file():
            findings.append(
                AuditFinding("non_regular_file", display, "candidate is not a regular file")
            )
            continue
        reason = _generated_reason(relative)
        if reason is not None:
            findings.append(AuditFinding("generated_artifact", display, reason))
        try:
            size = path.stat().st_size
        except OSError as exc:
            findings.append(AuditFinding("unreadable_file", display, str(exc)))
            continue
        if size > max_file_bytes:
            findings.append(
                AuditFinding(
                    "large_file",
                    display,
                    f"file is {size} bytes; limit is {max_file_bytes}",
                )
            )
            continue
        findings.extend(_text_findings(path, relative))

    findings.sort(key=lambda item: (item.path, item.line or 0, item.code))
    return ReleaseAuditReport(
        root=str(release_root),
        scanned_files=scanned,
        findings=tuple(findings),
        warnings=tuple(warnings),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_BYTES / (1024 * 1024),
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON receipt.")
    args = parser.parse_args(argv)
    if args.max_file_mb <= 0:
        parser.error("--max-file-mb must be positive")
    report = audit_release(
        args.root,
        max_file_bytes=max(1, int(args.max_file_mb * 1024 * 1024)),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    elif report.accepted:
        print(
            f"REEL release audit passed: files={report.scanned_files} "
            f"warnings={len(report.warnings)}"
        )
        for warning in report.warnings:
            print(f"warning: {warning}")
    else:
        print(
            f"REEL release audit failed: findings={len(report.findings)}",
            file=sys.stderr,
        )
        for finding in report.findings:
            location = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"{finding.code}: {location}: {finding.message}", file=sys.stderr)
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
