#!/usr/bin/env python3
"""Build and optionally execute the pinned REEL vLLM service command.

The safe default is a CPU-only dry run.  It validates the checkpoint revision,
freezes the model/dataset/path service contract, and writes a path-redacted JSON
receipt.  Pass ``--execute`` only inside the matching GPU environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTATION_DIR = ROOT / "instrumentation" / "vllm_encoder_trace"
INSTRUMENTATION_PATH = INSTRUMENTATION_DIR / "sitecustomize.py"
INSTRUMENTATION_SHA256 = (
    "211d6525ce9bb446fd900c791473dcdb1c82dcdeca586b06a5ea6584e2d9430c"
)
ENCODER_CACHE_MINIMUM_TOKENS = 65_536
HEX_REVISION = re.compile(r"[0-9a-fA-F]{40}")

MODES: dict[str, dict[str, Any]] = {
    "query_time_raw": {
        "paper_path": "frames",
        "prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "prompt_token_details": False,
        "encoder_instrumentation": False,
    },
    "write_once_frame_text": {
        "paper_path": "text",
        "prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "prompt_token_details": False,
        "encoder_instrumentation": False,
    },
    "write_once_visual_token_cache": {
        "paper_path": "visual",
        "prefix_caching": False,
        "mm_processor_cache_gb": 4,
        "prompt_token_details": True,
        "encoder_instrumentation": True,
    },
    "write_once_model_state_cache": {
        "paper_path": "prefix_state",
        "prefix_caching": True,
        "mm_processor_cache_gb": 4,
        "prompt_token_details": True,
        "encoder_instrumentation": False,
    },
}


@dataclass(frozen=True)
class ModelSpec:
    paper_name: str
    repository: str
    revision: str
    served_model_name: str
    vllm_version: str
    context_by_dataset: Mapping[str, int]
    gpu_memory_by_dataset: Mapping[str, float]
    multimodal_limit: Mapping[str, int]
    extra_arguments: tuple[str, ...] = ()


IMAGE_LIMIT = {"image": 68}
DEFAULT_MEMORY = {"cg": 0.88, "hd": 0.88}

MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3vl": ModelSpec(
        paper_name="Qwen3-VL-8B",
        repository="Qwen/Qwen3-VL-8B-Instruct",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        served_model_name="qwen3-vl-8b",
        vllm_version="0.11.0",
        context_by_dataset={"cg": 65_536, "hd": 131_072},
        gpu_memory_by_dataset={"cg": 0.88, "hd": 0.96},
        multimodal_limit=IMAGE_LIMIT,
    ),
    "qwen3vl4b": ModelSpec(
        paper_name="Qwen3-VL-4B",
        repository="Qwen/Qwen3-VL-4B-Instruct",
        revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        served_model_name="qwen3-vl-4b",
        vllm_version="0.11.0",
        context_by_dataset={"cg": 65_536, "hd": 131_072},
        gpu_memory_by_dataset={"cg": 0.88, "hd": 0.96},
        multimodal_limit=IMAGE_LIMIT,
    ),
    "internvl3": ModelSpec(
        paper_name="InternVL3-8B",
        repository="OpenGVLab/InternVL3-8B",
        revision="853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
        served_model_name="internvl3-8b",
        vllm_version="0.11.0",
        context_by_dataset={"cg": 32_768, "hd": 32_768},
        gpu_memory_by_dataset=DEFAULT_MEMORY,
        multimodal_limit=IMAGE_LIMIT,
        extra_arguments=(
            "--mm-processor-kwargs",
            '{"max_dynamic_patch":1,"dynamic_image_size":false}',
        ),
    ),
    "internvl35_8b": ModelSpec(
        paper_name="InternVL3.5-8B",
        repository="OpenGVLab/InternVL3_5-8B",
        revision="9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79",
        served_model_name="internvl3.5-8b",
        vllm_version="0.11.0",
        context_by_dataset={"cg": 32_768, "hd": 32_768},
        gpu_memory_by_dataset=DEFAULT_MEMORY,
        multimodal_limit=IMAGE_LIMIT,
        extra_arguments=(
            "--mm-processor-kwargs",
            '{"max_dynamic_patch":1,"dynamic_image_size":false}',
        ),
    ),
    "gemma3": ModelSpec(
        paper_name="Gemma-3-12B-IT",
        repository="google/gemma-3-12b-it",
        revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
        served_model_name="gemma-3-12b-it",
        vllm_version="0.11.0",
        context_by_dataset={"cg": 32_768, "hd": 32_768},
        gpu_memory_by_dataset=DEFAULT_MEMORY,
        multimodal_limit=IMAGE_LIMIT,
    ),
    "minicpmv45": ModelSpec(
        paper_name="MiniCPM-V-4.5",
        repository="openbmb/MiniCPM-V-4_5",
        revision="daef484c35ec93210ec93c5e901f8f3e9b78ee34",
        served_model_name="minicpm-v-4.5",
        vllm_version="0.15.0",
        context_by_dataset={"cg": 40_960, "hd": 40_960},
        gpu_memory_by_dataset=DEFAULT_MEMORY,
        multimodal_limit={"video": 1},
        extra_arguments=(
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}',
        ),
    ),
}


@dataclass
class LaunchPlan:
    command: list[str]
    environment: dict[str, str]
    receipt: dict[str, Any]
    receipt_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_argument(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def revision_evidence(model_dir: Path, expected: str) -> dict[str, Any]:
    """Require an exact HF commit from download metadata or a snapshot path."""

    observed: list[tuple[str, str]] = []
    metadata = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata.is_file():
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].strip():
            raise ValueError("config.json.metadata is empty")
        candidate = lines[0].strip()
        if HEX_REVISION.fullmatch(candidate) is None:
            raise ValueError("config.json.metadata does not start with a full commit hash")
        observed.append(("huggingface_download_metadata", candidate.lower()))

    # Check both spellings: resolve() follows a symlink into the HF cache, while
    # absolute() preserves an explicitly supplied snapshots/<revision> path.
    paths = {model_dir.absolute(), model_dir.resolve()}
    for candidate_path in paths:
        parts = candidate_path.parts
        for index, part in enumerate(parts[:-1]):
            if part != "snapshots":
                continue
            candidate = parts[index + 1]
            if HEX_REVISION.fullmatch(candidate):
                observed.append(("huggingface_snapshot_path", candidate.lower()))

    if not observed:
        raise ValueError(
            "cannot verify checkpoint revision: use an HF snapshots/<40-hex>/ "
            "directory or keep .cache/huggingface/download/config.json.metadata"
        )
    mismatches = sorted({revision for _, revision in observed if revision != expected})
    if mismatches:
        raise ValueError(
            f"checkpoint revision mismatch: expected {expected}, observed {mismatches}"
        )
    source_order = ("huggingface_download_metadata", "huggingface_snapshot_path")
    sources = [source for source in source_order if any(row[0] == source for row in observed)]
    return {
        "verified": True,
        "expected_revision": expected,
        "observed_revision": expected,
        "evidence_sources": sources,
    }


def resolve_executable(command: str) -> Path | None:
    if os.sep in command or (os.altsep and os.altsep in command):
        candidate = Path(command).expanduser()
        return candidate.resolve() if candidate.is_file() else None
    found = shutil.which(command)
    return Path(found).resolve() if found else None


def probe_runtime(python_command: str) -> dict[str, Any]:
    executable = resolve_executable(python_command)
    base: dict[str, Any] = {
        "probe_status": "unavailable",
        "executable_exists": executable is not None,
        "executable_sha256": sha256_file(executable) if executable else None,
        "python": None,
        "torch": None,
        "transformers": None,
        "vllm": None,
    }
    if executable is None:
        return base
    probe_environment = dict(os.environ)
    for variable in (
        "PESTATE_ENCODER_TRACE",
        "PESTATE_ENCODER_CACHE_TOKENS",
        "PESTATE_RESOURCE_TRACE",
    ):
        probe_environment.pop(variable, None)
    probe_pythonpath = _without_instrumentation(probe_environment.get("PYTHONPATH"))
    if probe_pythonpath:
        probe_environment["PYTHONPATH"] = os.pathsep.join(probe_pythonpath)
    else:
        probe_environment.pop("PYTHONPATH", None)
    code = """
import importlib.metadata as metadata
import json
import sys

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": version("torch"),
    "transformers": version("transformers"),
    "vllm": version("vllm"),
}))
"""
    try:
        completed = subprocess.run(
            [str(executable), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=probe_environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return base
    if completed.returncode != 0:
        return base
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return base
    return base | {"probe_status": "ok", **payload}


def path_record(path: Path, *, record_paths: bool) -> dict[str, Any]:
    resolved = path.resolve()
    payload: dict[str, Any] = {
        "exists": resolved.exists(),
        "is_directory": resolved.is_dir(),
        "path_sha256": sha256_text(str(resolved)),
    }
    if record_paths:
        payload["path"] = str(resolved)
    return payload


def sanitized_command(
    command: Sequence[str],
    *,
    runtime_python: str,
    model_dir: Path,
    media_root: Path,
    record_paths: bool,
) -> list[str]:
    if record_paths:
        return list(command)
    replacements = {
        runtime_python: "<RUNTIME_PYTHON>",
        str(model_dir): "<MODEL_DIR>",
        str(media_root): "<ALLOWED_MEDIA_ROOT>",
    }
    return [replacements.get(value, value) for value in command]


def _without_instrumentation(pythonpath: str | None) -> list[str]:
    target = INSTRUMENTATION_DIR.resolve()
    kept: list[str] = []
    for entry in (pythonpath or "").split(os.pathsep):
        if not entry:
            continue
        if Path(entry).expanduser().resolve() != target:
            kept.append(entry)
    return kept


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and launch one pinned REEL vLLM service. The default is "
            "a CPU-only dry run; use --execute for the real server."
        ),
        epilog=(
            "After the server is ready, run this script's 'warmup' subcommand "
            "before starting the measured client."
        ),
    )
    parser.add_argument("--model-family", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--dataset", choices=("cg", "hd"), required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--allowed-media-root",
        type=Path,
        required=True,
        help="Existing parent directory of every file:// image or video sent to vLLM.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--runtime-python",
        default=sys.executable,
        help="Python executable from the pinned vLLM environment.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Defaults to RUN_DIR/service_launch_receipt.json.",
    )
    parser.add_argument(
        "--encoder-trace",
        type=Path,
        help="Visual only; defaults to RUN_DIR/encoder_trace.jsonl.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Validate and write the receipt without starting vLLM (default).",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Replace this process with the validated vLLM server command.",
    )
    parser.set_defaults(execute=False)
    parser.add_argument(
        "--record-paths",
        action="store_true",
        help="Opt in to storing absolute local paths in the launch receipt.",
    )
    parser.add_argument(
        "--overwrite-receipt",
        action="store_true",
        help="Allow replacement of an existing launch receipt.",
    )
    return parser


def prepare_launch(
    args: argparse.Namespace,
    *,
    base_environment: Mapping[str, str] | None = None,
    runtime_probe: Mapping[str, Any] | None = None,
) -> LaunchPlan:
    spec = MODEL_SPECS[args.model_family]
    mode = MODES[args.mode]
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")

    model_dir = args.model_dir.expanduser().resolve()
    media_root = args.allowed_media_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError("model directory does not exist or is not a directory")
    if not media_root.is_dir():
        raise ValueError("allowed media root does not exist or is not a directory")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError("model directory has no config.json")
    revision = revision_evidence(model_dir, spec.revision)

    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = (args.receipt or run_dir / "service_launch_receipt.json").expanduser().resolve()
    if receipt_path.exists() and not args.overwrite_receipt:
        raise ValueError("launch receipt already exists; choose a new file or pass --overwrite-receipt")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    if args.encoder_trace is not None and not mode["encoder_instrumentation"]:
        raise ValueError("--encoder-trace is valid only for write_once_visual_token_cache")
    trace_path = (args.encoder_trace or run_dir / "encoder_trace.jsonl").expanduser().resolve()
    if mode["encoder_instrumentation"]:
        if trace_path.exists():
            raise ValueError("encoder trace already exists; use a fresh trace path")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if not INSTRUMENTATION_PATH.is_file():
            raise ValueError("released encoder instrumentation is missing")
        observed_instrumentation_sha = sha256_file(INSTRUMENTATION_PATH)
        if observed_instrumentation_sha != INSTRUMENTATION_SHA256:
            raise ValueError(
                "released encoder instrumentation hash mismatch: "
                f"expected {INSTRUMENTATION_SHA256}, observed {observed_instrumentation_sha}"
            )
    else:
        observed_instrumentation_sha = None

    runtime = dict(
        runtime_probe
        if runtime_probe is not None
        else probe_runtime(args.runtime_python)
    )
    observed_vllm = runtime.get("vllm")
    if observed_vllm is not None and observed_vllm != spec.vllm_version:
        raise ValueError(
            f"vLLM version mismatch: expected {spec.vllm_version}, observed {observed_vllm}"
        )
    if args.execute and observed_vllm is None:
        raise ValueError("cannot execute because the selected runtime's vLLM version is unavailable")

    command = [
        args.runtime_python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_dir),
        "--served-model-name",
        spec.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-model-len",
        str(spec.context_by_dataset[args.dataset]),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--gpu-memory-utilization",
        f"{spec.gpu_memory_by_dataset[args.dataset]:.2f}",
        "--generation-config",
        "vllm",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json_argument(spec.multimodal_limit),
        "--allowed-local-media-path",
        str(media_root),
    ]
    command.append(
        "--enable-prefix-caching"
        if mode["prefix_caching"]
        else "--no-enable-prefix-caching"
    )
    command.extend(["--mm-processor-cache-gb", str(mode["mm_processor_cache_gb"])])
    if mode["prompt_token_details"]:
        command.append("--enable-prompt-tokens-details")
    command.extend(spec.extra_arguments)

    environment = dict(
        base_environment if base_environment is not None else os.environ
    )
    pythonpath_entries = _without_instrumentation(environment.get("PYTHONPATH"))
    for variable in (
        "PESTATE_ENCODER_TRACE",
        "PESTATE_ENCODER_CACHE_TOKENS",
        "PESTATE_RESOURCE_TRACE",
    ):
        environment.pop(variable, None)
    if mode["encoder_instrumentation"]:
        environment["PESTATE_ENCODER_TRACE"] = str(trace_path)
        environment["PESTATE_ENCODER_CACHE_TOKENS"] = str(ENCODER_CACHE_MINIMUM_TOKENS)
        environment["PESTATE_RESOURCE_TRACE"] = "0"
        pythonpath_entries.insert(0, str(INSTRUMENTATION_DIR.resolve()))
    if pythonpath_entries:
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    else:
        environment.pop("PYTHONPATH", None)

    warnings: list[str] = []
    if observed_vllm is None:
        warnings.append(
            "vLLM is unavailable in the selected runtime; dry-run command construction "
            "was checked, but execution remains disabled"
        )
    command_for_receipt = sanitized_command(
        command,
        runtime_python=args.runtime_python,
        model_dir=model_dir,
        media_root=media_root,
        record_paths=args.record_paths,
    )
    runtime_receipt = dict(runtime)
    executable = resolve_executable(args.runtime_python)
    runtime_receipt["required_vllm"] = spec.vllm_version
    runtime_receipt["version_verified"] = observed_vllm == spec.vllm_version
    runtime_receipt.pop("executable_path", None)
    if args.record_paths and executable is not None:
        runtime_receipt["executable_path"] = str(executable)

    environment_receipt: dict[str, Any]
    if mode["encoder_instrumentation"]:
        environment_receipt = {
            "PESTATE_ENCODER_TRACE": (
                str(trace_path) if args.record_paths else "<ENCODER_TRACE>"
            ),
            "PESTATE_ENCODER_CACHE_TOKENS": str(ENCODER_CACHE_MINIMUM_TOKENS),
            "PESTATE_RESOURCE_TRACE": "0",
            "PYTHONPATH_first_entry": (
                str(INSTRUMENTATION_DIR.resolve())
                if args.record_paths
                else "<RELEASE_ENCODER_INSTRUMENTATION>"
            ),
        }
    else:
        environment_receipt = {
            "removed_variables": [
                "PESTATE_ENCODER_TRACE",
                "PESTATE_ENCODER_CACHE_TOKENS",
                "PESTATE_RESOURCE_TRACE",
            ],
            "release_instrumentation_removed_from_PYTHONPATH": True,
        }

    assets = {
        "model_directory": path_record(model_dir, record_paths=args.record_paths)
        | {"config_json_sha256": sha256_file(config_path)},
        "allowed_media_root": path_record(media_root, record_paths=args.record_paths),
        "run_directory": path_record(run_dir, record_paths=args.record_paths),
    }
    if mode["encoder_instrumentation"]:
        assets["encoder_trace"] = path_record(
            trace_path, record_paths=args.record_paths
        )

    receipt = {
        "schema_version": "reel.vllm_service_launch.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "action": "execute" if args.execute else "dry_run",
        "path_recording_opt_in": bool(args.record_paths),
        "model_family": args.model_family,
        "paper_model": spec.paper_name,
        "dataset": args.dataset,
        "mode": args.mode,
        "paper_path": mode["paper_path"],
        "checkpoint": {
            "repository": spec.repository,
            **revision,
        },
        "service_contract": {
            "served_model_name": spec.served_model_name,
            "host": "127.0.0.1",
            "port": args.port,
            "max_model_len": spec.context_by_dataset[args.dataset],
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "gpu_memory_utilization": spec.gpu_memory_by_dataset[args.dataset],
            "generation_config": "vllm",
            "trust_remote_code": True,
            "multimodal_limit": dict(spec.multimodal_limit),
            "prefix_caching": mode["prefix_caching"],
            "mm_processor_cache_gb": mode["mm_processor_cache_gb"],
            "prompt_token_details": mode["prompt_token_details"],
            "model_specific_arguments": list(spec.extra_arguments),
        },
        "encoder_cache_contract": {
            "instrumentation_enabled": mode["encoder_instrumentation"],
            "instrumentation_sha256": observed_instrumentation_sha,
            "minimum_tokens": (
                ENCODER_CACHE_MINIMUM_TOKENS
                if mode["encoder_instrumentation"]
                else None
            ),
            "minimum_is_lower_bound": bool(mode["encoder_instrumentation"]),
            "resource_trace_enabled": False,
        },
        "runtime": runtime_receipt,
        "assets": assets,
        "environment_contract": environment_receipt,
        "pre_measurement_warmup": {
            "required": True,
            "completed_by_launcher": False,
            "receipt_schema": "reel.server_warmup_acceptance.v1",
            "scope": "common_text_only_server_warmup_before_the_measured_runner",
            "instruction": "run this script's warmup subcommand after service readiness",
        },
        "command": command_for_receipt,
        "warnings": warnings,
        "configuration_validated": True,
        "execution_ready": observed_vllm == spec.vllm_version,
    }
    return LaunchPlan(command, environment, receipt, receipt_path)


def write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def launch_main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        plan = prepare_launch(args)
        write_receipt(plan.receipt_path, plan.receipt)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan.receipt, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    os.execvpe(plan.command[0], plan.command, plan.environment)
    raise AssertionError("os.execvpe returned unexpectedly")


def make_warmup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch_reel_vllm.py warmup",
        description=(
            "Verify a local REEL vLLM service and perform the one unmetered, "
            "text-only warm-up used before every accepted main run. The safe "
            "default writes a plan without sending any request."
        ),
    )
    parser.add_argument("--model-family", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Write the planned warm-up receipt without making HTTP requests (default).",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the model check and one warm-up completion request.",
    )
    parser.set_defaults(execute=False)
    parser.add_argument(
        "--overwrite-receipt",
        action="store_true",
        help="Allow replacement of an existing warm-up receipt.",
    )
    return parser


def warmup_request_contract(model_family: str) -> dict[str, Any]:
    return {
        "model": MODEL_SPECS[model_family].served_model_name,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
    }


def execute_warmup(
    *,
    model_family: str,
    port: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Check the local service identity, then send exactly one completion."""

    model = MODEL_SPECS[model_family].served_model_name
    base_url = f"http://127.0.0.1:{port}/v1"
    with urllib.request.urlopen(
        f"{base_url}/models", timeout=timeout_seconds
    ) as response:
        models_payload = json.load(response)
    served = {
        row.get("id")
        for row in models_payload.get("data", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    if model not in served:
        raise ValueError(
            f"wrong vLLM service on port {port}: expected {model}, served {sorted(served)}"
        )

    payload = warmup_request_contract(model_family)
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    if not result.get("choices"):
        raise ValueError("server warm-up returned no choice")
    return {
        "accepted": True,
        "model": model,
        "unmetered": True,
        "scope": "common_text_only_server_warmup_before_the_measured_runner",
        "usage": result.get("usage") or {},
    }


def warmup_main(argv: Sequence[str] | None = None) -> int:
    parser = make_warmup_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65_535:
        parser.error("port must be between 1 and 65535")
    if not 0 < args.timeout_seconds <= 3_600:
        parser.error("timeout seconds must be in (0, 3600]")
    out = args.out.expanduser().resolve()
    if out.exists() and not args.overwrite_receipt:
        parser.error("warm-up receipt already exists; choose a new file or pass --overwrite-receipt")
    out.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "schema_version": "reel.server_warmup_acceptance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "action": "execute" if args.execute else "dry_run",
        "model_family": args.model_family,
        "model": MODEL_SPECS[args.model_family].served_model_name,
        "endpoint": {
            "host": "127.0.0.1",
            "port": args.port,
            "api_prefix": "/v1",
        },
        "request_contract": warmup_request_contract(args.model_family),
        "unmetered": True,
        "scope": "common_text_only_server_warmup_before_the_measured_runner",
    }
    if args.execute:
        try:
            receipt = base | execute_warmup(
                model_family=args.model_family,
                port=args.port,
                timeout_seconds=args.timeout_seconds,
            )
        except (OSError, ValueError) as exc:
            failed = base | {
                "accepted": False,
                "error_type": type(exc).__name__,
                "usage": {},
            }
            write_receipt(out, failed)
            parser.error(f"warm-up failed; failure receipt written: {type(exc).__name__}")
    else:
        receipt = base | {
            "accepted": False,
            "planned_only": True,
            "usage": {},
        }
    write_receipt(out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "warmup":
        return warmup_main(values[1:])
    if values and values[0] == "launch":
        values = values[1:]
    return launch_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
