from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pestate.release_audit import audit_release, main


def _write_final_metadata(root: Path) -> None:
    (root / "LICENSE").write_text("Example license for test fixtures.\n", encoding="utf-8")
    (root / "CITATION.cff").write_text(
        "cff-version: 1.2.0\ntitle: REEL test fixture\n",
        encoding="utf-8",
    )


def test_clean_directory_passes_with_final_metadata(tmp_path: Path) -> None:
    _write_final_metadata(tmp_path)
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    report = audit_release(tmp_path)

    assert report.accepted is True
    assert report.findings == ()
    assert report.scanned_files == 3


def test_missing_release_metadata_is_rejected(tmp_path: Path) -> None:
    report = audit_release(tmp_path)

    assert {finding.code for finding in report.findings} == {
        "missing_license",
        "missing_citation",
    }


def test_git_discovery_ignores_tracked_files_deleted_from_worktree(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    obsolete = tmp_path / "obsolete.txt"
    obsolete.write_text("remove before release\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "obsolete.txt"], check=True)
    obsolete.unlink()
    _write_final_metadata(tmp_path)

    report = audit_release(tmp_path)

    assert report.accepted is True
    assert report.scanned_files == 2


def test_audit_detects_paths_secrets_large_files_symlinks_and_build_outputs(
    tmp_path: Path,
) -> None:
    _write_final_metadata(tmp_path)
    private_path = "/" + "data" + "/user/private/run.jsonl"
    token = "ghp_" + "a" * 40
    (tmp_path / "unsafe.py").write_text(
        f'PATH = "{private_path}"\nTOKEN = "{token}"\n',
        encoding="utf-8",
    )
    (tmp_path / "large.csv").write_bytes(b"x" * 256)
    generated = tmp_path / "build"
    generated.mkdir()
    (generated / "artifact.txt").write_text("generated\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "unsafe.py")

    report = audit_release(tmp_path, max_file_bytes=128)
    codes = {finding.code for finding in report.findings}

    assert report.accepted is False
    assert {
        "private_path",
        "secret_pattern",
        "large_file",
        "symbolic_link",
        "generated_artifact",
    } <= codes


def test_audit_rejects_ai_author_attribution(tmp_path: Path) -> None:
    _write_final_metadata(tmp_path)
    for name, line in {
        "claude.py": "# Edited by " + "Claude\n",
        "codex.py": "# by " + "Codex\n",
    }.items():
        (tmp_path / name).write_text(line, encoding="utf-8")

    report = audit_release(tmp_path)

    assert report.accepted is False
    assert [(finding.code, finding.path, finding.line) for finding in report.findings] == [
        ("ai_attribution", "claude.py", 1),
        ("ai_attribution", "codex.py", 1),
    ]


def test_audit_rejects_unreviewed_manifests_media_subtitles_and_frames(
    tmp_path: Path,
) -> None:
    _write_final_metadata(tmp_path)
    unsafe = {
        "questions.jsonl": b'{"qid":"private-question"}\n',
        "clip.webm": b"synthetic-video-placeholder",
        "captions.srt": b"1\n00:00:00,000 --> 00:00:01,000\nprivate subtitle\n",
        "frame.JPG": b"synthetic-frame-placeholder",
    }
    for name, payload in unsafe.items():
        (tmp_path / name).write_bytes(payload)

    report = audit_release(tmp_path)
    generated_paths = {
        finding.path
        for finding in report.findings
        if finding.code == "generated_artifact"
    }

    assert report.accepted is False
    assert generated_paths == set(unsafe)
    assert all(
        "unreviewed manifest, media, subtitle, or extracted-frame artifact"
        in finding.message
        for finding in report.findings
        if finding.path in unsafe
    )


def test_audit_allows_explicit_public_fixtures_and_documentation_images(
    tmp_path: Path,
) -> None:
    _write_final_metadata(tmp_path)
    fixture_root = tmp_path / "tests" / "fixtures" / "public"
    fixture_root.mkdir(parents=True)
    (fixture_root / "synthetic.jsonl").write_text(
        '{"kind":"synthetic"}\n', encoding="utf-8"
    )
    (fixture_root / "tiny.webm").write_bytes(b"reviewed-synthetic-placeholder")
    asset_root = tmp_path / "docs" / "assets"
    asset_root.mkdir(parents=True)
    (asset_root / "overview.png").write_bytes(b"reviewed-documentation-placeholder")

    report = audit_release(tmp_path)

    assert report.accepted is True
    assert report.findings == ()


def test_console_command_returns_nonzero_and_json_findings(
    tmp_path: Path, capsys
) -> None:
    assert main([str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is False
    assert {finding["code"] for finding in payload["findings"]} == {
        "missing_license",
        "missing_citation",
    }
