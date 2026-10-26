"""Append-only, per-operation cost accounting for formal REEL runs.

The ledger deliberately records an event when it happens.  It is not a
post-hoc estimate derived from final answers.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "run_id", "git_commit", "method_id", "dataset_version", "video_id", "question_id",
    "phase", "model_id", "cache_status", "tool_call_index",
    "input_text_tokens", "output_text_tokens", "visual_tokens",
    "decoded_frames", "repeated_frames", "unique_frame_ids",
    "raw_video_seconds_read", "latency_s", "gpu_count", "gpu_seconds",
    "peak_vram_bytes", "api_cost_usd", "error",
)

VALID_PHASES = {"write", "store", "plan", "retrieve", "observe", "verify", "answer"}


def sampled_visible_gpu_memory_bytes() -> int:
    """Best-effort device memory sample at an accounting-event boundary.

    This is deliberately labelled as a boundary sample in the ledger; it must
    not be reported as an isolated per-request peak when another process shares
    the device.
    """

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return max(values, default=0) * 1024 * 1024


@dataclass(frozen=True)
class LedgerContext:
    run_id: str
    git_commit: str
    method_id: str
    dataset_version: str
    video_id: str
    question_id: str
    gpu_count: int


class CostLedger:
    """A small, strict writer for one question's append-only cost events."""

    def __init__(self, path: str | Path, context: LedgerContext) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for field in (
            "run_id", "git_commit", "method_id", "dataset_version", "video_id", "question_id",
        ):
            if not str(getattr(context, field)).strip():
                raise ValueError(f"LedgerContext.{field} must be non-empty")
        if context.gpu_count < 0:
            raise ValueError("LedgerContext.gpu_count must be non-negative")
        self.context = context
        self._tool_call_index = 0

    def append(
        self,
        *,
        phase: str,
        model_id: str,
        cache_status: str = "miss",
        input_text_tokens: int = 0,
        output_text_tokens: int = 0,
        visual_tokens: int = 0,
        decoded_frames: int = 0,
        repeated_frames: int = 0,
        unique_frame_ids: list[str] | None = None,
        raw_video_seconds_read: float = 0.0,
        latency_s: float = 0.0,
        active_gpu_count: int | None = None,
        peak_vram_bytes: int = 0,
        api_cost_usd: float = 0.0,
        error: str = "",
        measurement_note: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        if phase not in VALID_PHASES:
            raise ValueError(f"Unsupported ledger phase: {phase}")
        numeric = {
            "input_text_tokens": input_text_tokens,
            "output_text_tokens": output_text_tokens,
            "visual_tokens": visual_tokens,
            "decoded_frames": decoded_frames,
            "repeated_frames": repeated_frames,
            "raw_video_seconds_read": raw_video_seconds_read,
            "latency_s": latency_s,
            "peak_vram_bytes": peak_vram_bytes,
            "api_cost_usd": api_cost_usd,
        }
        for name, value in numeric.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"Invalid non-negative numeric ledger field {name}={value!r}")

        gpu_count = self.context.gpu_count if active_gpu_count is None else active_gpu_count
        if gpu_count < 0:
            raise ValueError(f"active_gpu_count must be non-negative, got {gpu_count}")
        row: dict[str, Any] = {
            **self.context.__dict__,
            "phase": phase,
            "model_id": model_id,
            "cache_status": cache_status,
            "tool_call_index": self._tool_call_index,
            "input_text_tokens": int(input_text_tokens),
            "output_text_tokens": int(output_text_tokens),
            "visual_tokens": int(visual_tokens),
            "decoded_frames": int(decoded_frames),
            "repeated_frames": int(repeated_frames),
            "unique_frame_ids": list(unique_frame_ids or []),
            "raw_video_seconds_read": float(raw_video_seconds_read),
            "latency_s": float(latency_s),
            "gpu_count": int(gpu_count),
            "allocated_gpu_count": self.context.gpu_count,
            "gpu_seconds": float(latency_s) * int(gpu_count),
            "peak_vram_bytes": int(peak_vram_bytes),
            "api_cost_usd": float(api_cost_usd),
            "error": error,
            "measurement_note": measurement_note,
            **extra,
        }
        missing = [name for name in REQUIRED_FIELDS if name not in row]
        if missing:
            raise RuntimeError(f"Ledger writer omitted required fields: {missing}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        self._tool_call_index += 1
        return row
