from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENCODER_INSTRUMENTATION_SHA256 = (
    "211d6525ce9bb446fd900c791473dcdb1c82dcdeca586b06a5ea6584e2d9430c"
)


def test_encoder_instrumentation_matches_accepted_runtime() -> None:
    path = ROOT / "instrumentation/vllm_encoder_trace/sitecustomize.py"

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        EXPECTED_ENCODER_INSTRUMENTATION_SHA256
    )
