from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

from scripts import launch_reel_vllm as launcher


def make_model(root: Path, family: str, *, revision: str | None = None) -> Path:
    model = root / f"model-{family}"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"synthetic"}\n', encoding="utf-8")
    metadata = model / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    metadata.joinpath("config.json.metadata").write_text(
        (revision or launcher.MODEL_SPECS[family].revision) + "\nsynthetic-etag\n",
        encoding="utf-8",
    )
    return model


def runtime_for(family: str) -> dict[str, object]:
    return {
        "probe_status": "ok",
        "executable_exists": True,
        "executable_sha256": "0" * 64,
        "python": "3.11.15",
        "torch": "synthetic",
        "transformers": "synthetic",
        "vllm": launcher.MODEL_SPECS[family].vllm_version,
    }


def parse_args(
    tmp_path: Path,
    family: str,
    dataset: str,
    mode: str,
    *,
    model: Path | None = None,
) -> object:
    media = tmp_path / "media"
    media.mkdir(parents=True, exist_ok=True)
    return launcher.make_parser().parse_args(
        [
            "--model-family",
            family,
            "--dataset",
            dataset,
            "--mode",
            mode,
            "--model-dir",
            str(model or make_model(tmp_path, family)),
            "--allowed-media-root",
            str(media),
            "--run-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ]
    )


def flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("family", sorted(launcher.MODEL_SPECS))
@pytest.mark.parametrize("dataset", ["cg", "hd"])
@pytest.mark.parametrize("mode", sorted(launcher.MODES))
def test_all_service_cells_use_the_frozen_contract(
    tmp_path: Path, family: str, dataset: str, mode: str
) -> None:
    cell = tmp_path / family / dataset / mode
    args = parse_args(cell, family, dataset, mode)
    plan = launcher.prepare_launch(
        args,
        base_environment={"PATH": os.environ.get("PATH", "")},
        runtime_probe=runtime_for(family),
    )
    spec = launcher.MODEL_SPECS[family]
    mode_spec = launcher.MODES[mode]

    assert flag_value(plan.command, "--max-model-len") == str(
        spec.context_by_dataset[dataset]
    )
    assert flag_value(plan.command, "--gpu-memory-utilization") == (
        f"{spec.gpu_memory_by_dataset[dataset]:.2f}"
    )
    assert flag_value(plan.command, "--dtype") == "bfloat16"
    assert flag_value(plan.command, "--kv-cache-dtype") == "auto"
    assert "--allowed-local-media-path" in plan.command
    assert flag_value(plan.command, "--mm-processor-cache-gb") == str(
        mode_spec["mm_processor_cache_gb"]
    )
    expected_prefix_flag = (
        "--enable-prefix-caching"
        if mode_spec["prefix_caching"]
        else "--no-enable-prefix-caching"
    )
    assert expected_prefix_flag in plan.command
    assert ("--enable-prompt-tokens-details" in plan.command) is bool(
        mode_spec["prompt_token_details"]
    )
    assert plan.receipt["checkpoint"]["verified"] is True
    assert plan.receipt["runtime"]["version_verified"] is True
    assert plan.receipt["pre_measurement_warmup"]["required"] is True
    assert plan.receipt["pre_measurement_warmup"]["completed_by_launcher"] is False


def test_visual_enables_only_the_exact_instrumentation_contract(tmp_path: Path) -> None:
    args = parse_args(
        tmp_path,
        "qwen3vl",
        "hd",
        "write_once_visual_token_cache",
    )
    inherited = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(
            [str(launcher.INSTRUMENTATION_DIR), "/synthetic/pythonpath"]
        ),
        "PESTATE_ENCODER_TRACE": "/stale/trace",
        "PESTATE_ENCODER_CACHE_TOKENS": "131072",
        "PESTATE_RESOURCE_TRACE": "1",
    }
    plan = launcher.prepare_launch(
        args,
        base_environment=inherited,
        runtime_probe=runtime_for("qwen3vl"),
    )

    assert plan.environment["PESTATE_ENCODER_CACHE_TOKENS"] == "65536"
    assert plan.environment["PESTATE_RESOURCE_TRACE"] == "0"
    assert plan.environment["PESTATE_ENCODER_TRACE"].endswith("encoder_trace.jsonl")
    assert plan.environment["PYTHONPATH"].split(os.pathsep) == [
        str(launcher.INSTRUMENTATION_DIR.resolve()),
        "/synthetic/pythonpath",
    ]
    assert plan.receipt["encoder_cache_contract"] == {
        "instrumentation_enabled": True,
        "instrumentation_sha256": launcher.INSTRUMENTATION_SHA256,
        "minimum_tokens": 65536,
        "minimum_is_lower_bound": True,
        "resource_trace_enabled": False,
    }
    assert flag_value(plan.command, "--max-model-len") == "131072"
    assert flag_value(plan.command, "--gpu-memory-utilization") == "0.96"


def test_prefix_removes_visual_environment_and_uses_internvl_one_patch(
    tmp_path: Path,
) -> None:
    args = parse_args(
        tmp_path,
        "internvl35_8b",
        "cg",
        "write_once_model_state_cache",
    )
    plan = launcher.prepare_launch(
        args,
        base_environment={
            "PYTHONPATH": str(launcher.INSTRUMENTATION_DIR),
            "PESTATE_ENCODER_TRACE": "/stale/trace",
            "PESTATE_ENCODER_CACHE_TOKENS": "65536",
            "PESTATE_RESOURCE_TRACE": "1",
        },
        runtime_probe=runtime_for("internvl35_8b"),
    )

    assert "--enable-prefix-caching" in plan.command
    assert flag_value(plan.command, "--mm-processor-cache-gb") == "4"
    assert flag_value(plan.command, "--max-model-len") == "32768"
    assert flag_value(plan.command, "--mm-processor-kwargs") == (
        '{"max_dynamic_patch":1,"dynamic_image_size":false}'
    )
    assert "PYTHONPATH" not in plan.environment
    assert not any(key.startswith("PESTATE_") for key in plan.environment)


def test_minicpm_uses_video_input_disables_thinking_and_pins_vllm_015(
    tmp_path: Path,
) -> None:
    args = parse_args(tmp_path, "minicpmv45", "hd", "query_time_raw")
    plan = launcher.prepare_launch(
        args,
        base_environment={},
        runtime_probe=runtime_for("minicpmv45"),
    )

    assert flag_value(plan.command, "--max-model-len") == "40960"
    assert flag_value(plan.command, "--limit-mm-per-prompt") == '{"video":1}'
    assert flag_value(plan.command, "--default-chat-template-kwargs") == (
        '{"enable_thinking":false}'
    )
    assert plan.receipt["runtime"]["required_vllm"] == "0.15.0"
    assert "--no-enable-prefix-caching" in plan.command
    assert flag_value(plan.command, "--mm-processor-cache-gb") == "0"


def test_receipt_is_path_redacted_by_default(tmp_path: Path) -> None:
    args = parse_args(
        tmp_path,
        "qwen3vl4b",
        "cg",
        "write_once_visual_token_cache",
    )
    plan = launcher.prepare_launch(
        args,
        base_environment={},
        runtime_probe=runtime_for("qwen3vl4b"),
    )
    encoded = json.dumps(plan.receipt, sort_keys=True)

    assert str(tmp_path.resolve()) not in encoded
    assert str(launcher.INSTRUMENTATION_DIR.resolve()) not in encoded
    assert sys.executable not in encoded
    assert "<MODEL_DIR>" in plan.receipt["command"]
    assert "<ALLOWED_MEDIA_ROOT>" in plan.receipt["command"]
    assert plan.receipt["path_recording_opt_in"] is False
    assert set(plan.receipt["assets"]["model_directory"]) == {
        "exists",
        "is_directory",
        "path_sha256",
        "config_json_sha256",
    }


def test_snapshot_path_is_accepted_as_revision_evidence(tmp_path: Path) -> None:
    revision = launcher.MODEL_SPECS["gemma3"].revision
    model = tmp_path / "hf" / "snapshots" / revision / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    evidence = launcher.revision_evidence(model, revision)

    assert evidence["verified"] is True
    assert evidence["evidence_sources"] == ["huggingface_snapshot_path"]


def test_wrong_checkpoint_or_runtime_revision_fails_closed(tmp_path: Path) -> None:
    model = make_model(tmp_path, "gemma3", revision="1" * 40)
    args = parse_args(
        tmp_path,
        "gemma3",
        "cg",
        "query_time_raw",
        model=model,
    )
    with pytest.raises(ValueError, match="checkpoint revision mismatch"):
        launcher.prepare_launch(args, runtime_probe=runtime_for("gemma3"))

    good_root = tmp_path / "good"
    good_args = parse_args(good_root, "gemma3", "cg", "query_time_raw")
    bad_runtime = runtime_for("gemma3") | {"vllm": "0.15.0"}
    with pytest.raises(ValueError, match="vLLM version mismatch"):
        launcher.prepare_launch(good_args, runtime_probe=bad_runtime)


def test_cli_dry_run_writes_a_private_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = make_model(tmp_path, "internvl3")
    media = tmp_path / "media"
    media.mkdir()
    run = tmp_path / "run"
    monkeypatch.setattr(
        launcher, "probe_runtime", lambda _: runtime_for("internvl3")
    )

    result = launcher.main(
        [
            "--model-family",
            "internvl3",
            "--dataset",
            "cg",
            "--mode",
            "query_time_raw",
            "--model-dir",
            str(model),
            "--allowed-media-root",
            str(media),
            "--run-dir",
            str(run),
            "--dry-run",
        ]
    )

    assert result == 0
    receipt_path = run / "service_launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["action"] == "dry_run"
    assert receipt["checkpoint"]["verified"] is True
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert str(tmp_path.resolve()) not in receipt_path.read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["schema_version"] == (
        "reel.vllm_service_launch.v1"
    )


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.encoded = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _amount: int = -1) -> bytes:
        return self.encoded


def test_warmup_dry_run_never_sends_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_urlopen(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run attempted an HTTP request")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_urlopen)
    out = tmp_path / "server_warmup_plan.json"

    result = launcher.main(
        [
            "warmup",
            "--model-family",
            "qwen3vl",
            "--port",
            "8123",
            "--out",
            str(out),
            "--dry-run",
        ]
    )

    assert result == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["action"] == "dry_run"
    assert receipt["accepted"] is False
    assert receipt["planned_only"] is True
    assert receipt["request_contract"] == {
        "model": "qwen3-vl-8b",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
    }


def test_warmup_execute_checks_model_and_sends_one_text_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, float]] = []

    def fake_urlopen(
        request: object, *, timeout: float
    ) -> FakeHTTPResponse:
        calls.append((request, timeout))
        if isinstance(request, str):
            assert request == "http://127.0.0.1:8456/v1/models"
            return FakeHTTPResponse({"data": [{"id": "minicpm-v-4.5"}]})
        assert isinstance(request, urllib.request.Request)
        return FakeHTTPResponse(
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1},
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = tmp_path / "server_warmup_acceptance.json"

    result = launcher.main(
        [
            "warmup",
            "--model-family",
            "minicpmv45",
            "--port",
            "8456",
            "--out",
            str(out),
            "--execute",
        ]
    )

    assert result == 0
    assert len(calls) == 2
    completion_requests = [
        request for request, _ in calls if isinstance(request, urllib.request.Request)
    ]
    assert len(completion_requests) == 1
    request = completion_requests[0]
    assert request.full_url == "http://127.0.0.1:8456/v1/chat/completions"
    assert request.get_method() == "POST"
    assert json.loads(request.data or b"") == {
        "model": "minicpm-v-4.5",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
    }
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["accepted"] is True
    assert receipt["unmetered"] is True
    assert receipt["scope"] == (
        "common_text_only_server_warmup_before_the_measured_runner"
    )
    assert receipt["usage"] == {"prompt_tokens": 8, "completion_tokens": 1}
    assert out.stat().st_mode & 0o777 == 0o600
