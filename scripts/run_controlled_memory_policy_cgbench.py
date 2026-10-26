#!/usr/bin/env python3
"""Run matched memory-policy controls for an OpenAI-compatible VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pestate.cost_ledger import CostLedger, LedgerContext, sampled_visible_gpu_memory_bytes
from pestate.openai_vlm import OpenAIVLMClient
from pestate.time_utils import selected_times
from pestate.video_frames import extract_frames_at_times, oracle_window_times
from pestate.visual_tokens import qwen_visual_tokens
from pestate.vlm_parse import parse_evidence_response


ANSWER_PROMPT_TEMPLATE = """Answer using only the supplied visual input or repository.
Return exactly one JSON object:
{"answer":"A","evidence_spans":[{"role":"direct_answer","start_frame":0,"end_frame":0,"key_frames":[0],"evidence":"at most 20 words"}],"rationale":"at most 20 words"}
The value "A" above is only a format example. Replace it with exactly one uppercase
option label. Never copy a list such as "A|B|C" into the answer field.
Frame indices refer to the <FRAME_COUNT> supplied source frames and are zero-based. Cite only evidence that supports the answer. If evidence is insufficient, return an empty evidence_spans list."""


def answer_prompt_for_frames(frame_count: int) -> str:
    if frame_count < 1:
        raise ValueError("answer prompt requires at least one frame")
    return ANSWER_PROMPT_TEMPLATE.replace("<FRAME_COUNT>", str(frame_count))


# Backward compatibility for the fixed-64 scripts that import this module.
ANSWER_PROMPT = answer_prompt_for_frames(64)

ANSWER_REPAIR_PROMPT = """Your previous response did not fully follow the required JSON schema.
Re-check the supplied evidence and options. The answer must be exactly one of: {valid_labels}.
Return exactly one JSON object using this schema, with no other text:
{{"answer":"A","evidence_spans":[],"rationale":"at most 20 words"}}
The value "A" is only a format example; replace it with your one chosen label.
Do not write the option text, "None", or an explanation in the answer field."""


def minimal_answer_repair_response_format(valid_labels: list[str]) -> dict:
    """Build the constrained schema used only after a malformed first answer.

    The repair prompt already requires an empty evidence list.  Encoding that
    same requirement as a JSON schema prevents a format-only retry from
    spending its entire token budget on repeated evidence and being truncated.
    """
    expected = [chr(65 + index) for index in range(len(valid_labels))]
    if not valid_labels or valid_labels != expected:
        raise ValueError(
            "valid_labels must be a non-empty contiguous prefix of A-Z"
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "minimal_answer_repair",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": valid_labels},
                    "evidence_spans": {
                        "type": "array",
                        "items": {"type": "object"},
                        "maxItems": 0,
                    },
                    "rationale": {"type": "string", "maxLength": 160},
                },
                "required": ["answer", "evidence_spans", "rationale"],
                "additionalProperties": False,
            },
        },
    }

WRITE_PROMPT = """Questions are hidden. Build a reusable factual video repository from these chronological source frames. Preserve concrete entities, actions, visible text, state changes, and their source frame indices. Do not speculate. Return concise timeline entries, each beginning with the source frame range it describes."""

FRAME_LEVEL_WRITE_PROMPT = """Questions are hidden. Build a reusable factual record from these chronological source frames.
Output exactly one physical line for every supplied source frame, in the same order, using this schema:
Frame <index> | entities: <visible people and objects> | action/state: <visible actions and state> | visible text: <legible text or none>
Do not merge frames, skip frames, repeat frames, add headings, or use frame ranges. Record only visible facts and do not speculate.
Each line must contain at most 24 words total: list at most five entities, describe the action/state in at most eight words, and copy at most eight words of visible text. Use "none" when a field has no visible content."""

FRAME_LEVEL_EXPANDED48_WRITE_PROMPT = """Questions are hidden. Build a reusable factual record from these chronological source frames.
Output exactly one physical line for every supplied source frame, in the same order, using this schema:
Frame <index> | entities: <visible people and objects> | action/state: <visible actions and state> | visible text: <legible text or none>
Do not merge frames, skip frames, repeat frames, add headings, or use frame ranges. Record only visible facts and do not speculate.
For every frame with visible content, write 32 to 48 words total by recording more of the visible people and objects, actions or state, and legible text in the same schema. A fully blank or black frame may use fewer words. List at most ten entities, describe the action/state in at most sixteen words, and copy at most sixteen words of visible text. Use "none" when a field has no visible content. Do not pad, repeat, explain, or speculate."""

FRAME_LEVEL_DETAIL24_WRITE_PROMPT = """Questions are hidden. Build a reusable factual record from these chronological source frames.
Output exactly one physical line for every supplied source frame, in the same order, using this schema:
Frame <index> | entities: <visible people and objects> | action/state: <visible actions and state> | visible text: <legible text or none>
Do not merge frames, skip frames, repeat frames, add headings, or use frame ranges. Record only visible facts and do not speculate.
Each line must contain at most 24 words total: list at most five entities, describe the action/state in at most eight words, and copy at most eight words of visible text. Use "none" when a field has no visible content.
Within the same length limit, prioritize object contents, counts, colors, spatial relations, package or label details, and exact visible text. If space is tight, omit generic scene descriptions before these details."""

FRAME_LEVEL_WRITE_PROMPTS = {
    "balanced24": FRAME_LEVEL_WRITE_PROMPT,
    "expanded48": FRAME_LEVEL_EXPANDED48_WRITE_PROMPT,
    "detail24": FRAME_LEVEL_DETAIL24_WRITE_PROMPT,
}


def frame_level_write_prompts(profile: str) -> tuple[str, str]:
    question_hidden = FRAME_LEVEL_WRITE_PROMPTS[profile]
    context_available = question_hidden.replace(
        "Questions are hidden. Build a reusable factual record",
        "Build a factual record",
        1,
    )
    return question_hidden, context_available

MATCHED_WRITE_PROMPT = """Write question-relevant or generally useful facts from these chronological source frames using exactly this schema for every line:
Frames <start>-<end> | entities: <visible entities> | actions: <visible actions> | state/text: <state changes or visible text>
Use source-frame indices, report only visible facts, and do not speculate. Keep the schema identical regardless of whether question context is available."""

SELECTION_PROMPT = """Select the source frames most likely to contain visual evidence needed to answer the question. Return exactly one JSON object:
{"selected_frames":[0,1],"reason":"at most 20 words"}
Use only valid source-frame indices, do not repeat an index, and select no more than the requested budget. This is evidence selection, not answering."""

COARSE_SELECTION_PROMPT = """Use these low-frequency frames to decide where the video should be inspected more densely. Return exactly one JSON object:
{"selected_frames":[0,1],"observations":"at most 80 words of question-relevant facts visible in the coarse frames"}
Use only valid source-frame indices, do not repeat an index, and select no more than the requested budget. Record only visible facts; do not guess the answer. The selected frames are temporal anchors for a later local observation."""

MODEL_STATE_WARMUP = "Initialize reusable model state for later questions."

# These modes save more than one artifact from the same question-blind pass.
# Prefix state already depends on retained visual encoder outputs, so
# Text+Prefix State is the meaningful "all available state" condition; a separate
# Visual+KV mode would store the same visual backing twice without giving the
# answerer new evidence.
FRAME_TEXT_HYBRID_VISUAL_MODE = "write_once_frame_text_visual_cache"
FRAME_TEXT_HYBRID_KV_MODE = "write_once_frame_text_model_state_cache"
AKS_QUERY_TIME_RAW_MODE = "aks_query_time_raw"
AKS_SELECTION_SCHEMA_VERSION = "pestate.query_aware_aks_selection.v1"
AKS_TOP_LEVEL_FIELDS = {
    "schema_version",
    "config_sha256",
    "source_manifest_sha256",
    "dataset",
    "qid",
    "video_uid",
    "video_path",
    "question_sha256",
    "question_aware",
    "question_only",
    "candidate_fps",
    "candidate_count",
    "candidate_frame_indices_sha256",
    "candidate_times_sha256",
    "source_average_fps",
    "source_frame_count",
    "duration",
    "source_video_sha256",
    "scorer",
    "selector",
    "frame_budget",
    "selected_candidate_indices",
    "selected_source_frame_indices",
    "selected_times",
    "selected_times_sha256",
    "retained_timestamp_bytes",
    "selection_wall_seconds",
    "selector_cost",
}
AKS_SELECTOR_COST_FIELDS = {
    "candidate_fps",
    "candidate_count",
    "decode_s",
    "feature_s",
    "selection_s",
    "total_s",
    "decode_cuda_s",
    "feature_cuda_s",
    "selection_cuda_s",
    "peak_vram_bytes",
    "embedding_model_id",
    "embedding_retention_policy",
    "raw_video_seconds_read",
    "cost_attribution",
    "physical_score_pass_shared_across_budgets",
}
AKS_COST_ATTRIBUTION = "full_standalone_per_question_policy"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
FRAME_TEXT_HYBRID_MODES = {
    FRAME_TEXT_HYBRID_VISUAL_MODE,
    FRAME_TEXT_HYBRID_KV_MODE,
}
FRAME_TEXT_WRITE_ONCE_MODES = {
    "write_once_frame_text",
    "write_once_frame_text_masked",
    *FRAME_TEXT_HYBRID_MODES,
}
VISUAL_CACHE_MODES = {
    "write_once_visual_token_cache",
    FRAME_TEXT_HYBRID_VISUAL_MODE,
}
KV_CACHE_MODES = {
    "write_once_model_state_cache",
    FRAME_TEXT_HYBRID_KV_MODE,
}
STATE_CACHE_MODES = VISUAL_CACHE_MODES | KV_CACHE_MODES
TIME_TAG = re.compile(r"<TIME\s+([0-9:.]+)\s+video\s+\d+>")
TIME_PAIR = re.compile(
    r"<TIME\s+([0-9:.]+)\s+video\s+\d+>\s+(?:to|and)\s+"
    r"<TIME\s+([0-9:.]+)\s+video\s+\d+>"
)
BBOX_TAG = re.compile(
    r"<BBOX\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)>"
)
FRAME_LEVEL_ENTRY = re.compile(
    r"^(?:Frame\s+)?(\d+)\s*\|\s*\S.+$",
    re.IGNORECASE,
)


def normalize_frame_level_line(line: str, index: int) -> str:
    """Normalize the optional literal `Frame` prefix without changing content."""
    if re.match(r"^Frame\s+", line, re.IGNORECASE):
        return line
    return f"Frame {index} | {line.split('|', 1)[1].strip()}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_times_payload(times: list[float]) -> bytes:
    return json.dumps(
        times,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def require_sha256(value: object, field: str) -> str:
    text = value if isinstance(value, str) else ""
    if SHA256_HEX.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def aks_component_id(component: dict, kind: str) -> str:
    """Return a readable ID when available and a stable ID otherwise."""
    for field in ("id", "method_id", "model_id", "name"):
        value = component.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = json.dumps(
        component,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{kind}:sha256:{hashlib.sha256(payload).hexdigest()[:16]}"


def load_aks_query_selections(
    path: Path,
    expected_rows: list[dict],
    frame_budget: int,
) -> tuple[dict[str, dict], str, int]:
    """Load strict, question-bound AKS selections for one answer shard."""
    if not path.is_file():
        raise FileNotFoundError(path)

    expected_by_qid: dict[str, dict] = {}
    for source_row_number, source_row in enumerate(expected_rows, start=1):
        if not isinstance(source_row, dict):
            raise ValueError(f"source manifest row {source_row_number} is not an object")
        qid = str(source_row.get("qid") or "")
        if not qid or qid in expected_by_qid:
            raise ValueError(f"source manifest has an invalid or duplicate qid={qid!r}")
        expected_by_qid[qid] = source_row
    if not expected_by_qid:
        raise ValueError("source manifest contains no questions")

    selections: dict[str, dict] = {}
    config_hashes: set[str] = set()
    source_manifest_hashes: set[str] = set()
    datasets: set[str] = set()
    for row_number, row in enumerate(read_jsonl(path), start=1):
        if not isinstance(row, dict):
            raise ValueError(f"AKS selection row {row_number} is not an object")
        if set(row) != AKS_TOP_LEVEL_FIELDS:
            missing = sorted(AKS_TOP_LEVEL_FIELDS - set(row))
            extra = sorted(set(row) - AKS_TOP_LEVEL_FIELDS)
            raise ValueError(
                f"AKS selection row {row_number} has an unexpected schema; "
                f"missing={missing} extra={extra}"
            )
        if row["schema_version"] != AKS_SELECTION_SCHEMA_VERSION:
            raise ValueError(
                f"AKS selection row {row_number} has unsupported schema_version="
                f"{row['schema_version']!r}"
            )

        qid = row["qid"]
        if not isinstance(qid, str) or not qid or qid.strip() != qid:
            raise ValueError(f"AKS selection row {row_number} has invalid qid={qid!r}")
        if qid in selections:
            raise ValueError(f"AKS selection manifest has duplicate qid={qid!r}")
        video_id = row["video_uid"]
        if not isinstance(video_id, str) or not video_id or video_id.strip() != video_id:
            raise ValueError(f"AKS selection row for {qid} has invalid video_uid")
        if not isinstance(row["video_path"], str) or not row["video_path"].strip():
            raise ValueError(f"AKS selection row for {qid} has invalid video_path")
        if not isinstance(row["dataset"], str) or not row["dataset"].strip():
            raise ValueError(f"AKS selection row for {qid} has invalid dataset")
        if row["question_aware"] is not True or row["question_only"] is not True:
            raise ValueError(
                f"AKS selection row for {qid} must be question-aware and question-only"
            )

        config_sha = require_sha256(row["config_sha256"], f"config_sha256[{qid}]")
        source_manifest_sha = require_sha256(
            row["source_manifest_sha256"], f"source_manifest_sha256[{qid}]"
        )
        require_sha256(row["question_sha256"], f"question_sha256[{qid}]")
        require_sha256(
            row["candidate_frame_indices_sha256"],
            f"candidate_frame_indices_sha256[{qid}]",
        )
        require_sha256(row["candidate_times_sha256"], f"candidate_times_sha256[{qid}]")
        require_sha256(row["source_video_sha256"], f"source_video_sha256[{qid}]")
        config_hashes.add(config_sha)
        source_manifest_hashes.add(source_manifest_sha)
        datasets.add(row["dataset"])

        for field in ("candidate_count", "source_frame_count", "frame_budget"):
            if type(row[field]) is not int or int(row[field]) < 1:
                raise ValueError(f"AKS selection row for {qid} has invalid {field}")
        if int(row["frame_budget"]) != frame_budget:
            raise ValueError(
                f"AKS selection budget mismatch for {qid}: "
                f"manifest={row['frame_budget']} requested={frame_budget}"
            )
        candidate_count = int(row["candidate_count"])
        source_frame_count = int(row["source_frame_count"])
        if candidate_count < frame_budget or source_frame_count < candidate_count:
            raise ValueError(f"AKS selection row for {qid} has too few candidate frames")

        numeric_top_level = ("candidate_fps", "source_average_fps", "duration", "selection_wall_seconds")
        for field in numeric_top_level:
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"AKS selection row for {qid} has invalid {field}")
        if not math.isclose(float(row["candidate_fps"]), 1.0, abs_tol=1e-12):
            raise ValueError(f"AKS selection row for {qid} was not built from a 1-fps scan")

        for field, upper_bound in (
            ("selected_candidate_indices", candidate_count),
            ("selected_source_frame_indices", source_frame_count),
        ):
            values = row[field]
            if (
                not isinstance(values, list)
                or len(values) != frame_budget
                or any(type(value) is not int for value in values)
                or values != sorted(values)
                or len(set(values)) != frame_budget
                or values[0] < 0
                or values[-1] >= upper_bound
            ):
                raise ValueError(f"AKS selection row for {qid} has invalid {field}")

        raw_times = row["selected_times"]
        if (
            not isinstance(raw_times, list)
            or len(raw_times) != frame_budget
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in raw_times
            )
        ):
            raise ValueError(f"AKS selection row for {qid} has invalid selected_times")
        times = [float(value) for value in raw_times]
        if times != sorted(times) or len(set(times)) != frame_budget:
            raise ValueError(f"AKS selected_times for {qid} are not strictly chronological")
        if times[-1] > float(row["duration"]) + 1e-3:
            raise ValueError(f"AKS selected_times for {qid} exceed the declared duration")
        encoded_times = canonical_times_payload(times)
        selected_times_sha = hashlib.sha256(encoded_times).hexdigest()
        if row["selected_times_sha256"] != selected_times_sha:
            raise ValueError(f"AKS selected_times hash mismatch for {qid}")
        if (
            type(row["retained_timestamp_bytes"]) is not int
            or int(row["retained_timestamp_bytes"]) != len(encoded_times)
        ):
            raise ValueError(f"AKS retained timestamp byte count mismatch for {qid}")

        for field in ("scorer", "selector"):
            if not isinstance(row[field], dict) or not row[field]:
                raise ValueError(f"AKS selection row for {qid} has invalid {field}")
            aks_component_id(row[field], field)
        scorer = row["scorer"]
        if (
            not isinstance(scorer.get("model_id"), str)
            or not scorer["model_id"].strip()
            or not isinstance(scorer.get("hf_revision"), str)
            or not scorer["hf_revision"].strip()
            or scorer.get("frozen") is not True
            or scorer.get("question_only") is not True
        ):
            raise ValueError(f"AKS scorer provenance is incomplete for {qid}")
        require_sha256(
            scorer.get("model_artifact_sha256"),
            f"scorer.model_artifact_sha256[{qid}]",
        )
        selector = row["selector"]
        if (
            type(selector.get("max_depth")) is not int
            or int(selector["max_depth"]) < 1
            or type(selector.get("ratio")) is not int
            or int(selector["ratio"]) < 1
            or isinstance(selector.get("threshold"), bool)
            or not isinstance(selector.get("threshold"), (int, float))
            or not math.isfinite(float(selector["threshold"]))
        ):
            raise ValueError(f"AKS selector provenance is incomplete for {qid}")

        costs = row["selector_cost"]
        if not isinstance(costs, dict) or set(costs) != AKS_SELECTOR_COST_FIELDS:
            raise ValueError(f"AKS selection row for {qid} has an unexpected selector_cost schema")
        numeric_cost_fields = AKS_SELECTOR_COST_FIELDS - {
            "embedding_model_id",
            "embedding_retention_policy",
            "cost_attribution",
            "physical_score_pass_shared_across_budgets",
        }
        for field in numeric_cost_fields:
            value = costs[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"AKS selector_cost for {qid} has invalid {field}")
        if type(costs["candidate_count"]) is not int:
            raise ValueError(f"AKS selector_cost for {qid} has non-integer candidate_count")
        if int(costs["candidate_count"]) != candidate_count or not math.isclose(
            float(costs["candidate_fps"]), float(row["candidate_fps"]), abs_tol=1e-12
        ):
            raise ValueError(f"AKS selector_cost candidate audit mismatch for {qid}")
        if any(float(costs[field]) <= 0 for field in ("decode_s", "feature_s", "selection_s", "total_s")):
            raise ValueError(f"AKS selector_cost for {qid} is missing measured wall time")
        if (
            float(costs["feature_cuda_s"]) <= 0
            or type(costs["peak_vram_bytes"]) is not int
            or int(costs["peak_vram_bytes"]) <= 0
        ):
            raise ValueError(f"AKS selector_cost for {qid} is missing measured GPU scoring cost")
        if not math.isclose(float(costs["decode_cuda_s"]), 0.0, abs_tol=1e-12):
            raise ValueError(f"AKS candidate decode for {qid} is not declared CPU-only")
        if not math.isclose(float(costs["selection_cuda_s"]), 0.0, abs_tol=1e-12):
            raise ValueError(f"AKS selection for {qid} is not declared CPU-only")
        component_total = sum(
            float(costs[field]) for field in ("decode_s", "feature_s", "selection_s")
        )
        if not math.isclose(float(costs["total_s"]), component_total, abs_tol=1e-9):
            raise ValueError(f"AKS selector_cost total mismatch for {qid}")
        if not math.isclose(
            float(row["selection_wall_seconds"]),
            float(costs["selection_s"]),
            abs_tol=1e-9,
        ):
            raise ValueError(f"AKS selection wall-time mismatch for {qid}")
        if not isinstance(costs["embedding_model_id"], str) or not costs["embedding_model_id"].strip():
            raise ValueError(f"AKS selector_cost for {qid} has no embedding model")
        if str(scorer["model_id"]) not in str(costs["embedding_model_id"]):
            raise ValueError(f"AKS selector_cost for {qid} does not identify its scorer")
        if costs["embedding_retention_policy"] != "discard_after_timestamp_selection":
            raise ValueError(f"AKS selector_cost for {qid} retained temporary embeddings")
        if float(costs["raw_video_seconds_read"]) <= 0 or not math.isclose(
            float(costs["raw_video_seconds_read"]),
            float(row["duration"]),
            abs_tol=1e-6,
        ):
            raise ValueError(f"AKS selector_cost for {qid} did not charge the full video read")
        if costs["cost_attribution"] != AKS_COST_ATTRIBUTION:
            raise ValueError(f"AKS selector_cost for {qid} has the wrong cost attribution")
        if costs["physical_score_pass_shared_across_budgets"] is not True:
            raise ValueError(f"AKS selector_cost for {qid} lacks the physical reuse declaration")

        expected = expected_by_qid.get(qid)
        if expected is not None:
            if str(expected.get("video_uid") or "") != video_id:
                raise ValueError(f"AKS video_uid mismatch for source question {qid}")
            expected_question_sha = hashlib.sha256(
                str(expected.get("question") or "").encode("utf-8")
            ).hexdigest()
            if row["question_sha256"] != expected_question_sha:
                raise ValueError(f"AKS question SHA mismatch for {qid}")

        selections[qid] = row | {
            "selected_times": times,
            "selected_times_sha256": selected_times_sha,
        }

    if len(config_hashes) != 1:
        raise ValueError(f"AKS selection manifest mixes config hashes: {sorted(config_hashes)}")
    if len(source_manifest_hashes) != 1:
        raise ValueError(
            "AKS selection manifest mixes source manifest hashes: "
            f"{sorted(source_manifest_hashes)}"
        )
    if len(datasets) != 1:
        raise ValueError(f"AKS selection manifest mixes datasets: {sorted(datasets)}")
    missing = set(expected_by_qid) - set(selections)
    if missing:
        raise ValueError(f"AKS selection manifest is missing qids: {sorted(missing)}")
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        {qid: selections[qid] for qid in expected_by_qid},
        manifest_sha,
        len(selections),
    )


def load_question_blind_timestamps(
    path: Path | None,
    expected_videos: set[str],
    num_frames: int,
) -> tuple[dict[str, dict], str]:
    """Load one fixed timestamp set per video without accepting query leakage."""
    if path is None:
        return {}, ""
    if not path.is_file():
        raise FileNotFoundError(path)
    forbidden = {
        "qid", "question", "questions", "choices", "answer", "right_answer",
        "clue_intervals", "evidence", "query", "prompt",
    }
    index: dict[str, dict] = {}
    for row_number, row in enumerate(read_jsonl(path), start=1):
        if not isinstance(row, dict):
            raise ValueError(f"timestamp manifest row {row_number} is not an object")
        leaked = forbidden & set(row)
        if leaked:
            raise ValueError(
                f"timestamp manifest row {row_number} contains query-side fields: {sorted(leaked)}"
            )
        video_id = str(row.get("video_uid") or "")
        if not video_id or video_id in index:
            raise ValueError(f"invalid or duplicate timestamp row for video={video_id!r}")
        if row.get("question_blind") is not True:
            raise ValueError(f"timestamp row for {video_id} is not declared question-blind")
        selector_id = str(row.get("selector_id") or "").strip()
        if not selector_id:
            raise ValueError(f"timestamp row for {video_id} has no selector_id")
        raw_times = row.get("selected_times")
        if not isinstance(raw_times, list) or len(raw_times) != num_frames:
            raise ValueError(
                f"timestamp row for {video_id} has {len(raw_times) if isinstance(raw_times, list) else 'invalid'} "
                f"times, expected {num_frames}"
            )
        times = [float(value) for value in raw_times]
        if any(not math.isfinite(value) or value < 0 for value in times):
            raise ValueError(f"timestamp row for {video_id} contains invalid values")
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError(f"timestamp row for {video_id} is not strictly chronological")
        computed_sha = hashlib.sha256(
            json.dumps(times, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        declared_sha = str(row.get("selected_times_sha256") or "")
        if declared_sha and declared_sha != computed_sha:
            raise ValueError(f"timestamp hash mismatch for {video_id}")
        selector_cost = row.get("selector_cost")
        if not isinstance(selector_cost, dict):
            raise ValueError(
                f"timestamp row for {video_id} has no measured selector_cost"
            )
        required_cost_fields = {
            "candidate_fps", "candidate_count", "decode_s", "feature_s",
            "selection_s", "total_s", "decode_cuda_s", "feature_cuda_s",
            "selection_cuda_s", "peak_vram_bytes", "embedding_model_id",
            "embedding_retention_policy",
        }
        if set(selector_cost) != required_cost_fields:
            raise ValueError(
                f"timestamp row for {video_id} has an unexpected selector_cost schema"
            )
        numeric_cost_fields = required_cost_fields - {
            "embedding_model_id", "embedding_retention_policy"
        }
        if any(
            not isinstance(selector_cost[field], (int, float))
            or not math.isfinite(float(selector_cost[field]))
            or float(selector_cost[field]) < 0
            for field in numeric_cost_fields
        ):
            raise ValueError(f"timestamp row for {video_id} has invalid selector costs")
        if int(selector_cost["candidate_count"]) < num_frames:
            raise ValueError(f"timestamp row for {video_id} scanned too few candidates")
        selected_candidate_indices = row.get("selected_candidate_indices")
        if (
            not isinstance(selected_candidate_indices, list)
            or len(selected_candidate_indices) != num_frames
            or not all(isinstance(value, int) for value in selected_candidate_indices)
            or selected_candidate_indices != sorted(selected_candidate_indices)
            or len(set(selected_candidate_indices)) != num_frames
            or selected_candidate_indices[0] < 0
            or selected_candidate_indices[-1] >= int(selector_cost["candidate_count"])
        ):
            raise ValueError(
                f"timestamp row for {video_id} has invalid selected candidate indices"
            )
        component_total = sum(
            float(selector_cost[field])
            for field in ("decode_s", "feature_s", "selection_s")
        )
        if not math.isclose(
            float(selector_cost["total_s"]), component_total, abs_tol=1e-9
        ):
            raise ValueError(f"timestamp row for {video_id} has inconsistent total_s")
        if not str(selector_cost["embedding_model_id"]).strip():
            raise ValueError(f"timestamp row for {video_id} has no embedding model")
        if selector_cost["embedding_retention_policy"] not in {
            "discard_after_timestamp_selection", "retained_and_charged"
        }:
            raise ValueError(
                f"timestamp row for {video_id} has no embedding retention policy"
            )
        if int(row.get("retained_timestamp_bytes") or 0) <= 0:
            raise ValueError(f"timestamp row for {video_id} has no retained byte count")
        index[video_id] = row | {
            "selected_times": times,
            "selected_times_sha256": computed_sha,
        }
    missing = expected_videos - set(index)
    if missing:
        raise ValueError(f"timestamp manifest is missing videos: {sorted(missing)}")
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {video: index[video] for video in expected_videos}, manifest_sha


def append_selector_cost_events(
    *,
    ledger_path: Path,
    run_id: str,
    commit: str,
    method: str,
    dataset_version: str,
    video_id: str,
    timestamp_row: dict,
    timestamp_manifest_sha: str,
) -> None:
    """Import measured, immutable selector work into each end-to-end path."""

    costs = timestamp_row["selector_cost"]
    candidate_count = int(costs["candidate_count"])
    candidate_sha = str(timestamp_row.get("candidate_times_sha256") or "unknown")
    candidate_frame_ids = [
        f"{video_id}:selector_candidate:{candidate_sha[:12]}:{index}"
        for index in range(candidate_count)
    ]
    context = LedgerContext(
        run_id,
        commit,
        method,
        dataset_version,
        video_id,
        "__video_write__",
        1,
    )
    common = {
        "selector_id": str(timestamp_row["selector_id"]),
        "question_blind": True,
        "timestamp_manifest_sha256": timestamp_manifest_sha,
        "selector_cost_source": "frozen_timestamp_manifest",
        "candidate_fps": float(costs["candidate_fps"]),
        "candidate_count": candidate_count,
    }
    selector_ledger = CostLedger(ledger_path, context)
    selector_ledger.append(
        phase="write",
        model_id="question_blind_video_decoder",
        cache_status="miss",
        decoded_frames=candidate_count,
        unique_frame_ids=candidate_frame_ids,
        raw_video_seconds_read=float(timestamp_row.get("decodable_duration_s") or 0),
        latency_s=float(costs["decode_s"]),
        active_gpu_count=0,
        measurement_note=(
            "Measured one-time candidate scan imported from the frozen question-blind "
            "timestamp manifest."
        ),
        **common,
    )
    selector_ledger.append(
        phase="write",
        model_id=str(costs["embedding_model_id"]),
        cache_status="miss",
        unique_frame_ids=candidate_frame_ids,
        latency_s=float(costs["feature_s"]),
        active_gpu_count=1,
        peak_vram_bytes=int(costs["peak_vram_bytes"]),
        cuda_seconds=float(costs["feature_cuda_s"]),
        embedding_retention_policy=str(costs["embedding_retention_policy"]),
        measurement_note=(
            "Measured one-time frozen visual-feature extraction imported from the "
            "timestamp manifest; candidate embeddings are not retained."
        ),
        **common,
    )
    selector_ledger.append(
        phase="plan",
        model_id="maxvol_or_semantic_change_selector",
        cache_status="miss",
        unique_frame_ids=[
            candidate_frame_ids[int(index)]
            for index in timestamp_row.get("selected_candidate_indices", [])
        ],
        latency_s=float(costs["selection_s"]),
        active_gpu_count=0,
        selected_times=list(timestamp_row["selected_times"]),
        selected_times_sha256=str(timestamp_row["selected_times_sha256"]),
        measurement_note=(
            "Measured fixed-budget ranking imported from the frozen question-blind "
            "timestamp manifest."
        ),
        **common,
    )
    selector_ledger.append(
        phase="store",
        model_id="timestamp_array",
        cache_status="write",
        active_gpu_count=0,
        state_storage_bytes=int(timestamp_row.get("retained_timestamp_bytes") or 0),
        selected_times_sha256=str(timestamp_row["selected_times_sha256"]),
        measurement_note=(
            "The selector retains only the chosen timestamps; temporary candidate "
            "embeddings are excluded from persistent memory."
        ),
        **common,
    )


def append_aks_selector_cost_events(
    *,
    ledger_path: Path,
    run_id: str,
    commit: str,
    method: str,
    dataset_version: str,
    selection_row: dict,
    selection_manifest_sha: str,
) -> None:
    """Charge one question the full standalone AKS selection lifecycle."""
    qid = str(selection_row["qid"])
    video_id = str(selection_row["video_uid"])
    costs = selection_row["selector_cost"]
    candidate_count = int(selection_row["candidate_count"])
    budget = int(selection_row["frame_budget"])
    observation_policy = (
        f"aks_query_aware_fixed{budget}:"
        f"{str(selection_row['config_sha256'])[:12]}"
    )
    selected_frame_ids = [
        f"{video_id}:aks_query:{qid}:{source_index}:{stamp:.6f}"
        for source_index, stamp in zip(
            selection_row["selected_source_frame_indices"],
            selection_row["selected_times"],
        )
    ]
    common = {
        "selector_id": aks_component_id(selection_row["selector"], "selector"),
        "scorer_id": aks_component_id(selection_row["scorer"], "scorer"),
        "observation_policy": observation_policy,
        "question_aware": True,
        "question_only": True,
        "selector_budget": budget,
        "selector_candidate_count": candidate_count,
        "selector_retained_timestamp_bytes": int(
            selection_row["retained_timestamp_bytes"]
        ),
        "candidate_fps": float(selection_row["candidate_fps"]),
        "candidate_frame_indices_sha256": str(
            selection_row["candidate_frame_indices_sha256"]
        ),
        "candidate_times_sha256": str(selection_row["candidate_times_sha256"]),
        "question_sha256": str(selection_row["question_sha256"]),
        "source_manifest_sha256": str(selection_row["source_manifest_sha256"]),
        "source_video_sha256": str(selection_row["source_video_sha256"]),
        "query_selection_manifest_sha256": selection_manifest_sha,
        "selector_cost_source": "frozen_query_aware_aks_manifest",
        "selector_cost_attribution": str(costs["cost_attribution"]),
        "physical_score_pass_shared_across_budgets": bool(
            costs["physical_score_pass_shared_across_budgets"]
        ),
        "selector_cost_imported_in_full": True,
    }
    ledger = CostLedger(
        ledger_path,
        LedgerContext(
            run_id,
            commit,
            method,
            dataset_version,
            video_id,
            qid,
            1,
        ),
    )
    ledger.append(
        phase="observe",
        model_id="decord_candidate_video_decoder",
        cache_status="miss",
        decoded_frames=candidate_count,
        unique_frame_ids=[],
        raw_video_seconds_read=float(costs["raw_video_seconds_read"]),
        latency_s=float(costs["decode_s"]),
        active_gpu_count=0,
        cuda_seconds=float(costs["decode_cuda_s"]),
        candidate_frame_ids_omitted_from_ledger=True,
        measurement_note=(
            "Imported full per-question CPU candidate decode from the frozen AKS "
            "manifest. Candidate frame IDs are represented by their count and hashes."
        ),
        **common,
    )
    ledger.append(
        phase="retrieve",
        model_id=str(costs["embedding_model_id"]),
        cache_status="miss",
        repeated_frames=candidate_count,
        unique_frame_ids=[],
        latency_s=float(costs["feature_s"]),
        active_gpu_count=1,
        peak_vram_bytes=int(costs["peak_vram_bytes"]),
        cuda_seconds=float(costs["feature_cuda_s"]),
        embedding_retention_policy=str(costs["embedding_retention_policy"]),
        candidate_frame_ids_omitted_from_ledger=True,
        measurement_note=(
            "Imported full per-question BLIP query-frame scoring cost. A physically "
            "shared score pass across budgets is charged in full to this condition."
        ),
        **common,
    )
    ledger.append(
        phase="plan",
        model_id=aks_component_id(selection_row["selector"], "selector"),
        cache_status="miss",
        unique_frame_ids=selected_frame_ids,
        latency_s=float(costs["selection_s"]),
        active_gpu_count=0,
        cuda_seconds=float(costs["selection_cuda_s"]),
        selected_candidate_indices=list(selection_row["selected_candidate_indices"]),
        selected_source_frame_indices=list(
            selection_row["selected_source_frame_indices"]
        ),
        selected_times=list(selection_row["selected_times"]),
        selected_times_sha256=str(selection_row["selected_times_sha256"]),
        measurement_note=(
            "Imported per-question CPU AKS ranking and fixed-budget keyframe selection."
        ),
        **common,
    )
    ledger.append(
        phase="store",
        model_id="aks_selected_timestamp_array",
        cache_status="write",
        active_gpu_count=0,
        state_storage_bytes=int(selection_row["retained_timestamp_bytes"]),
        state_budget_frames=budget,
        state_storage_unit="per_question_selected_timestamp_json",
        selected_times_sha256=str(selection_row["selected_times_sha256"]),
        retention_scope="current_question_only",
        reusable_across_questions=False,
        measurement_note=(
            "The query-time selector retains only this question's chosen timestamps; "
            "candidate embeddings are discarded."
        ),
        **common,
    )


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_prebuilt_repository(directory: Path, video_id: str) -> Path:
    """Resolve one composed or sharded repository without silently choosing a duplicate."""
    filename = f"{video_id}.repository.txt"
    candidates = [path for path in (directory / filename, *directory.glob(f"shard_*/run/{filename}")) if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one prebuilt repository for {video_id} under {directory}; "
            f"found={list(map(str, candidates))}"
        )
    return candidates[0]


def resolve_prebuilt_cost_ledger(directory: Path) -> Path:
    """Resolve one canonical Text ledger without double-reading composed shards."""
    direct = directory / "cost_ledger.jsonl"
    if direct.is_file():
        return direct
    composed = directory / "run" / "cost_ledger.jsonl"
    if composed.is_file():
        return composed
    candidates = sorted(directory.glob("shard_*/run/cost_ledger.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no canonical Text cost ledger under {directory}")
    # Multiple shard ledgers are one logical source.  The caller filters by
    # video and rejects duplicate writer events below.
    return candidates[0] if len(candidates) == 1 else directory


def prebuilt_cost_ledger_rows(directory: Path) -> tuple[list[dict], list[Path]]:
    resolved = resolve_prebuilt_cost_ledger(directory)
    paths = (
        sorted(resolved.glob("shard_*/run/cost_ledger.jsonl"))
        if resolved.is_dir()
        else [resolved]
    )
    return [row for path in paths for row in read_jsonl(path)], paths


def import_prebuilt_frame_text_writer_cost(
    *,
    directory: Path,
    ledger_path: Path,
    run_id: str,
    commit: str,
    method: str,
    dataset_version: str,
    model_id: str,
    video_id: str,
    frame_ids: list[str],
    selected_times_sha256: str,
    timestamp_manifest_sha256: str,
    out_dir: Path,
) -> None:
    """Charge the exact canonical Text writer once in a joint-input run.

    A joint run reads the already frozen Text bytes so every compared condition
    receives identical language.  The writer is still part of construction and
    therefore cannot be free.  We import only the source writer calls: selector
    work and frame decoding are measured directly by the joint run.
    """
    source_rows, source_paths = prebuilt_cost_ledger_rows(directory)
    allowed_notes = (
        "Question-blind frame-level text write:",
        "Frame-level text repair for source indices omitted",
    )
    writer_rows = [
        row
        for row in source_rows
        if str(row.get("video_id") or "") == video_id
        and str(row.get("question_id") or "") == "__video_write__"
        and str(row.get("phase") or "") == "write"
        and str(row.get("measurement_note") or "").startswith(allowed_notes)
    ]
    if not writer_rows:
        raise ValueError(f"canonical Text ledger has no writer calls for {video_id}")
    if any(str(row.get("model_id") or "") != model_id for row in writer_rows):
        raise ValueError(
            f"canonical Text writer model differs from joint model for {video_id}"
        )
    if any(str(row.get("error") or "") for row in writer_rows):
        raise ValueError(f"canonical Text writer contains failed calls for {video_id}")
    observed_frame_ids = {
        str(frame_id)
        for row in writer_rows
        for frame_id in row.get("unique_frame_ids", [])
    }
    if observed_frame_ids != set(frame_ids):
        raise ValueError(
            "canonical Text writer did not use the joint run's exact selected frames: "
            f"video={video_id} expected={len(frame_ids)} observed={len(observed_frame_ids)}"
        )

    reserved = {
        "run_id", "git_commit", "method_id", "dataset_version", "video_id",
        "question_id", "phase", "model_id", "cache_status", "tool_call_index",
        "input_text_tokens", "output_text_tokens", "visual_tokens",
        "decoded_frames", "repeated_frames", "unique_frame_ids",
        "raw_video_seconds_read", "latency_s", "gpu_count",
        "allocated_gpu_count", "gpu_seconds", "peak_vram_bytes",
        "api_cost_usd", "error", "measurement_note",
    }
    context = LedgerContext(
        run_id,
        commit,
        method,
        dataset_version,
        video_id,
        "__video_write__",
        1,
    )
    imported = []
    for source_row in writer_rows:
        extras = {
            key: value for key, value in source_row.items() if key not in reserved
        }
        extras.update({
            "canonical_text_cost_imported": True,
            "canonical_text_source_method_id": str(source_row.get("method_id") or ""),
            "canonical_text_source_run_id": str(source_row.get("run_id") or ""),
            "selected_times_sha256": selected_times_sha256,
            "timestamp_manifest_sha256": timestamp_manifest_sha256,
        })
        imported.append(
            CostLedger(ledger_path, context).append(
                phase="write",
                model_id=model_id,
                cache_status=str(source_row.get("cache_status") or "miss"),
                input_text_tokens=int(source_row.get("input_text_tokens") or 0),
                output_text_tokens=int(source_row.get("output_text_tokens") or 0),
                visual_tokens=int(source_row.get("visual_tokens") or 0),
                decoded_frames=int(source_row.get("decoded_frames") or 0),
                repeated_frames=int(source_row.get("repeated_frames") or 0),
                unique_frame_ids=[str(value) for value in source_row.get("unique_frame_ids", [])],
                raw_video_seconds_read=float(source_row.get("raw_video_seconds_read") or 0),
                latency_s=float(source_row.get("latency_s") or 0),
                active_gpu_count=int(source_row.get("gpu_count") or 0),
                peak_vram_bytes=int(source_row.get("peak_vram_bytes") or 0),
                api_cost_usd=float(source_row.get("api_cost_usd") or 0),
                error="",
                measurement_note=(
                    "Imported measured canonical Text construction: "
                    f"{source_row.get('measurement_note') or ''}"
                ),
                **extras,
            )
        )
    append_jsonl(
        out_dir / "prebuilt_writer_cost_provenance.jsonl",
        {
            "video_uid": video_id,
            "source_ledger_paths": [str(path.resolve()) for path in source_paths],
            "source_ledger_sha256": {
                str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
            "source_writer_events": len(writer_rows),
            "imported_writer_events": len(imported),
            "imported_gpu_seconds": sum(float(row["gpu_seconds"]) for row in imported),
            "selected_times_sha256": selected_times_sha256,
            "timestamp_manifest_sha256": timestamp_manifest_sha256,
        },
    )


def validate_frame_level_entries(text: str, expected_indices: list[int]) -> list[str]:
    """Require one numbered physical output line per supplied frame."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    accepted: dict[int, str] = {}
    malformed: list[str] = []
    for line in lines:
        match = FRAME_LEVEL_ENTRY.match(line)
        if match is None:
            malformed.append(line)
        else:
            index = int(match.group(1))
            if index in expected_indices and index not in accepted:
                accepted[index] = normalize_frame_level_line(line, index)
            else:
                malformed.append(line)
    observed = [index for index in expected_indices if index in accepted]
    if malformed or observed != expected_indices:
        raise ValueError(
            "frame-level writer schema violation: "
            f"expected_indices={expected_indices} observed_indices={observed} "
            f"malformed_lines={malformed[:3]}"
        )
    return [accepted[index] for index in expected_indices]


def format_clock_time(seconds: float) -> str:
    """Format a video timestamp without dropping sub-second precision."""
    if seconds < 0:
        raise ValueError(f"video timestamp must be non-negative: {seconds}")
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def attach_frame_times(repository: str, times: list[float]) -> str:
    """Attach deterministic absolute time metadata to a validated frame repository.

    The writer's semantic output is reused verbatim.  Only sampler metadata that
    accompanied the same images on the Raw path is inserted, so an answerer can
    resolve question timestamps without asking the writer to describe more facts.
    """
    expected_indices = list(range(len(times)))
    lines = validate_frame_level_entries(repository, expected_indices)
    timed_lines: list[str] = []
    for index, (line, stamp) in enumerate(zip(lines, times)):
        _, payload = line.split("|", 1)
        timed_lines.append(
            f"Frame {index} | time: {float(stamp):.3f}s | {payload.strip()}"
        )
    return "\n".join(timed_lines)


def collect_frame_level_entries(text: str, expected_indices: list[int]) -> tuple[dict[int, str], list[str]]:
    """Collect the first valid line for each expected frame and audit the rest."""
    accepted: dict[int, str] = {}
    rejected: list[str] = []
    expected = set(expected_indices)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = FRAME_LEVEL_ENTRY.match(line)
        if match is None:
            rejected.append(line)
            continue
        index = int(match.group(1))
        if index not in expected or index in accepted:
            rejected.append(line)
            continue
        accepted[index] = normalize_frame_level_line(line, index)
    return accepted, rejected


def source_revision() -> str:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or f"source_sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"


def sampled_video_path(paths: list[str]) -> Path:
    """Materialize an exact sampled-frame sequence for video-only VLM APIs."""
    digest = hashlib.sha256("\n".join(str(Path(path).resolve()) for path in paths).encode()).hexdigest()[:16]
    target = Path(paths[0]).resolve().parent / f"sampled_{digest}.mp4"
    if target.exists() and target.stat().st_size > 0:
        return target
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read sampled frame {paths[0]}")
    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create sampled video {target}")
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not read sampled frame {path}")
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return target


def media_content(
    paths: list[str], indices: list[int], times: list[float], visual_input_format: str
) -> list[dict]:
    if visual_input_format == "video":
        mapping = ", ".join(
            f"clip frame {position}=source frame {index} ({times[index]:.3f}s)"
            for position, index in enumerate(indices)
        )
        video_path = sampled_video_path(paths)
        return [
            {"type": "text", "text": f"Chronological sampled clip mapping: {mapping}."},
            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
        ]
    content: list[dict] = []
    for path, index in zip(paths, indices):
        content.append({"type": "text", "text": f"Source Frame {index} at {times[index]:.3f}s:"})
        content.append({"type": "image_url", "image_url": {"url": f"file://{Path(path).resolve()}"}})
    return content


def tagged_time_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid HD-EPIC time tag: {value}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def normalize_hdepic_tags(text: str) -> str:
    """Match the text formatting used by the official HD-EPIC evaluator."""
    def replace_time(match: re.Match) -> str:
        seconds = tagged_time_seconds(match.group(1))
        if seconds >= 3600:
            return time.strftime("%H:%M:%S", time.gmtime(seconds))
        return time.strftime("%M:%S", time.gmtime(seconds))

    def replace_bbox(match: re.Match) -> str:
        coordinates = [int(float(value) / 1408.0 * 1000) for value in match.groups()]
        return f"({', '.join(str(value) for value in coordinates)})"

    return BBOX_TAG.sub(replace_bbox, TIME_TAG.sub(replace_time, text))


def official_reference_times(row: dict) -> list[float]:
    """Return explicit question/choice timestamps exposed by the HD-EPIC interface."""
    texts = [str(row.get("question") or ""), *(str(value) for value in row.get("choices", []))]
    values: list[float] = []
    for text in texts:
        for match in TIME_TAG.finditer(text):
            value = tagged_time_seconds(match.group(1))
            if all(abs(value - seen) > 1e-6 for seen in values):
                values.append(value)
        for match in TIME_PAIR.finditer(text):
            start = tagged_time_seconds(match.group(1))
            end = tagged_time_seconds(match.group(2))
            midpoint = (start + end) / 2.0
            if all(abs(midpoint - seen) > 1e-6 for seen in values):
                values.append(midpoint)
    return values


def official_bbox_reference_times(row: dict) -> list[float]:
    """Return the exact question frame needed to resolve an HD-EPIC BBOX referent."""
    question = str(row.get("question") or "")
    if BBOX_TAG.search(question) is None:
        return []
    match = TIME_TAG.search(question)
    if match is None:
        raise ValueError("HD-EPIC question contains <BBOX> without a matching <TIME>")
    return [tagged_time_seconds(match.group(1))]


def prepare_hdepic_row(row: dict, *, include_bbox_reference: bool = False) -> dict:
    """Preserve visual-reference metadata before replacing HD-EPIC markup."""
    raw_question = str(row.get("question") or "")
    return row | {
        "question": normalize_hdepic_tags(raw_question),
        "choices": [
            normalize_hdepic_tags(str(value)) for value in row.get("choices", [])
        ],
        "_official_bbox_question": raw_question,
        "_official_bbox_reference_times": (
            official_bbox_reference_times(row) if include_bbox_reference else []
        ),
    }


def draw_official_bbox(path: str, question: str) -> None:
    match = BBOX_TAG.search(question)
    if match is None:
        return
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read HD-EPIC reference frame {path}")
    y1, x1, y2, x2 = (float(value) for value in match.groups())
    height, width = frame.shape[:2]
    left = round(max(0.0, min(1408.0, x1)) * width / 1408.0)
    right = round(max(0.0, min(1408.0, x2)) * width / 1408.0)
    top = round(max(0.0, min(1408.0, y1)) * height / 1408.0)
    bottom = round(max(0.0, min(1408.0, y2)) * height / 1408.0)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), max(2, height // 350))
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"Could not write marked HD-EPIC reference frame {path}")


def reference_content(paths: list[str], times: list[float]) -> list[dict]:
    content: list[dict] = [{
        "type": "text",
        "text": (
            "Question-linked reference frames follow. A red box marks the object referred "
            "to by the question. These frames identify the query object; they are not part "
            "of the saved source frames and are not source-frame citation indices."
        ),
    }]
    for index, (path, stamp) in enumerate(zip(paths, times)):
        content.append({"type": "text", "text": f"Question Reference {index} at {stamp:.3f}s:"})
        content.append({"type": "image_url", "image_url": {"url": f"file://{Path(path).resolve()}"}})
    return content


def visual_tokens(paths: list[str], model_path: Path, max_pixels: int | None) -> int:
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return qwen_visual_tokens(images, model_path, max_pixels=max_pixels)
    finally:
        for image in images:
            image.close()


def parse_selected_frames(text: str, num_frames: int, budget: int) -> list[int]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"selector returned no JSON object: {text[:200]!r}")
        payload = json.loads(text[start:end + 1])
    raw = payload.get("selected_frames") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f"selector returned no selected_frames list: {text[:200]!r}")
    selected: list[int] = []
    for value in raw:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < num_frames and index not in selected:
            selected.append(index)
        if len(selected) == budget:
            break
    if not selected:
        raise ValueError(f"selector returned no valid frame indices: {text[:200]!r}")
    return sorted(selected)


def uniform_fallback_indices(num_frames: int, budget: int) -> list[int]:
    count = min(num_frames, budget)
    if count == 1:
        return [0]
    return [round(index * (num_frames - 1) / (count - 1)) for index in range(count)]


def local_times_around_anchors(
    anchors: list[float], duration: float, coarse_spacing: float, frame_budget: int
) -> list[float]:
    """Sample a fixed number of local frames around question-selected anchors."""
    if not anchors or frame_budget < 1:
        raise ValueError("local sampling requires anchors and a positive frame budget")
    half_window = max(2.0, min(30.0, coarse_spacing / 2.0))
    counts = [frame_budget // len(anchors)] * len(anchors)
    for index in range(frame_budget % len(anchors)):
        counts[index] += 1
    values: list[float] = []
    for anchor, count in zip(anchors, counts):
        left = max(0.0, anchor - half_window)
        right = min(duration, anchor + half_window)
        if count == 1:
            values.append(anchor)
        else:
            values.extend(left + step * (right - left) / (count - 1) for step in range(count))
    rounded = sorted({round(float(value), 6) for value in values})
    if len(rounded) < frame_budget:
        fallback = [
            round(index * duration / max(1, frame_budget - 1), 6)
            for index in range(frame_budget)
        ]
        rounded = sorted({*rounded, *fallback})
    if len(rounded) < frame_budget:
        raise RuntimeError(f"could not construct {frame_budget} distinct local timestamps")
    # Keep the selected local neighborhoods when deduplication adds fallback
    # points, choosing an evenly spread fixed-size subset only if necessary.
    if len(rounded) > frame_budget:
        chosen = uniform_fallback_indices(len(rounded), frame_budget)
        rounded = [rounded[index] for index in chosen]
    return rounded


def cached_prompt_tokens(usage: dict) -> int:
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    return int(details.get("cached_tokens") or 0)


def metadata(path: Path, duration_hint: float) -> tuple[int, float, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    finally:
        cap.release()
    duration = duration_hint if duration_hint > 0 else (frames - 1) / fps
    if frames < 2 or not math.isfinite(fps) or fps <= 0 or duration <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return frames, duration, fps


def validate_aks_selection_against_physical_video(
    selection: dict,
    *,
    actual_frame_count: int,
    actual_average_fps: float,
) -> None:
    """Verify AKS timestamps against the physical frames they identify.

    The source frame numbers and frame count are authoritative.  A dataset's
    duration hint may be rounded and can therefore be slightly shorter than the
    timestamp of the final physical frame.
    """
    qid = str(selection.get("qid") or "<unknown>")
    declared_frame_count = int(selection["source_frame_count"])
    if declared_frame_count != actual_frame_count:
        raise ValueError(
            f"AKS source frame count mismatch for {qid}: "
            f"manifest={declared_frame_count} actual={actual_frame_count}"
        )
    if (
        not math.isfinite(actual_average_fps)
        or actual_average_fps <= 0
        or not math.isclose(
            float(selection["source_average_fps"]),
            actual_average_fps,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(
            f"AKS source average FPS mismatch for {qid}: "
            f"manifest={selection['source_average_fps']} actual={actual_average_fps}"
        )

    source_indices = selection["selected_source_frame_indices"]
    times = selection["selected_times"]
    if len(source_indices) != len(times):
        raise ValueError(f"AKS source frame/timestamp count mismatch for {qid}")
    for source_index, timestamp in zip(source_indices, times):
        if type(source_index) is not int or not 0 <= source_index < actual_frame_count:
            raise ValueError(
                f"AKS source frame index {source_index!r} is outside "
                f"[0, {actual_frame_count}) for {qid}"
            )
        physical_timestamp = source_index / actual_average_fps
        if not math.isclose(
            float(timestamp), physical_timestamp, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"AKS timestamp/source frame mismatch for {qid}: "
                f"frame={source_index} timestamp={float(timestamp):.9f} "
                f"physical_timestamp={physical_timestamp:.9f}"
            )


def extract_frames_at_source_indices(
    video_path: str | Path,
    source_frame_indices: list[int],
    out_dir: str | Path,
    total_frames: int,
    *,
    max_side: int | None = None,
) -> list[str]:
    """Decode every requested physical source frame or fail the whole request."""
    if total_frames < 1:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if max_side is not None and max_side <= 0:
        raise ValueError(f"max_side must be positive, got {max_side}")
    if not source_frame_indices:
        raise ValueError("source_frame_indices must not be empty")
    if any(type(index) is not int for index in source_frame_indices):
        raise ValueError("source_frame_indices must contain only integers")
    if any(
        left >= right
        for left, right in zip(source_frame_indices, source_frame_indices[1:])
    ):
        raise ValueError("source_frame_indices must be strictly increasing and unique")
    invalid = [
        index for index in source_frame_indices if not 0 <= index < total_frames
    ]
    if invalid:
        raise ValueError(
            f"source frame indices outside [0, {total_frames}): {invalid[:10]}"
        )

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution_label = f"maxside{max_side}" if max_side is not None else "native"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    paths: list[str] = []
    try:
        reported_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if reported_total_frames > 0 and reported_total_frames != total_frames:
            raise RuntimeError(
                f"video frame count changed for {video_path}: "
                f"metadata={total_frames} decoder={reported_total_frames}"
            )
        for output_index, source_index in enumerate(source_frame_indices):
            frame_path = output_dir / (
                f"frame_{output_index:03d}_source_{source_index:09d}_"
                f"{resolution_label}.jpg"
            )
            if frame_path.exists():
                cached = cv2.imread(str(frame_path))
                if cached is None:
                    raise RuntimeError(f"cached source frame is unreadable: {frame_path}")
                if max_side is not None and max(cached.shape[:2]) > max_side:
                    raise RuntimeError(
                        f"cached source frame exceeds max_side={max_side}: {frame_path}"
                    )
                paths.append(str(frame_path))
                continue

            if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                raise RuntimeError(
                    f"decoder refused source frame seek {source_index} for {video_path}"
                )
            ok, frame = capture.read()
            position_after_read = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            decoded_source_index = round(position_after_read) - 1
            if (
                not ok
                or frame is None
                or not math.isfinite(position_after_read)
                or decoded_source_index != source_index
            ):
                raise RuntimeError(
                    f"failed exact source-frame decode for {video_path}: "
                    f"requested={source_index} decoded={decoded_source_index} "
                    f"position_after_read={position_after_read!r}"
                )
            if max_side is not None:
                height, width = frame.shape[:2]
                longest = max(height, width)
                if longest > max_side:
                    scale = max_side / longest
                    frame = cv2.resize(
                        frame,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write exact source frame: {frame_path}")
            if cv2.imread(str(frame_path)) is None:
                raise RuntimeError(f"Written source frame is unreadable: {frame_path}")
            paths.append(str(frame_path))
    finally:
        capture.release()

    if len(paths) != len(source_frame_indices):
        raise RuntimeError(
            f"exact source-frame extraction was incomplete for {video_path}: "
            f"requested={len(source_frame_indices)} decoded={len(paths)}"
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "write_once_text", "write_once_frame_text", "write_once_frame_text_masked",
            "blind_replay_text", "query_conditioned_text", "blind_replay_frame_text",
            "query_conditioned_frame_text",
            "blind_replay_text_matched", "query_conditioned_text_matched",
            "query_time_raw", "write_once_raw_cache", "write_once_visual_token_cache",
            "write_once_model_state_cache",
            FRAME_TEXT_HYBRID_VISUAL_MODE, FRAME_TEXT_HYBRID_KV_MODE,
            "query_aware_selection", AKS_QUERY_TIME_RAW_MODE,
            "coarse_to_fine_selection",
            "coarse_vlm_local_selection", "annotated_evidence",
            "annotated_evidence_frame_text",
            "annotated_evidence_query_conditioned_frame_text",
        ),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--timestamp-manifest",
        type=Path,
        help=(
            "Optional one-row-per-video question-blind timestamp manifest. When omitted, "
            "the historical uniform sampler is used."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument(
        "--prebuilt-repository-dir",
        type=Path,
        help=(
            "For write_once_frame_text only, reuse validated <video_uid>.repository.txt "
            "files from this directory and rerun only the complete answer phase."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--visual-model-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--writer-chunk-size", type=int, default=8)
    parser.add_argument("--max-answer-tokens", type=int, default=384)
    parser.add_argument(
        "--answer-repair-attempts",
        type=int,
        default=0,
        help=(
            "Retry an answer only when the model failed to return a valid option letter. "
            "All retry latency and tokens are included in the answer cost event."
        ),
    )
    parser.add_argument(
        "--answer-repair-json-schema",
        action="store_true",
        help=(
            "Constrain only malformed-answer repair calls to the minimal JSON "
            "schema already stated by the repair prompt. The initial answer call "
            "and its evidence schema are unchanged."
        ),
    )
    parser.add_argument("--max-writer-tokens", type=int, default=256)
    parser.add_argument(
        "--frame-writer-tokens", type=int, default=512,
        help="Completion-token ceiling for each eight-frame, frame-level writer call.",
    )
    parser.add_argument(
        "--frame-text-profile",
        choices=tuple(FRAME_LEVEL_WRITE_PROMPTS),
        default="balanced24",
        help=(
            "Frame-level text-writing instruction. balanced24 reproduces the formal "
            "condition; expanded48 changes only the requested capacity; detail24 keeps "
            "the 24-word limit but prioritizes fine visual details."
        ),
    )
    parser.add_argument(
        "--allow-result-errors",
        action="store_true",
        help=(
            "Declare that downstream compose is expected to audit traceable model-output "
            "failures as incorrect. Collection always preserves them; compose rejects them "
            "unless its explicit error audit passes."
        ),
    )
    parser.add_argument(
        "--matched-writer-tokens",
        type=int,
        default=192,
        help="Exact completion-token budget used by both matched RQ3 writers.",
    )
    parser.add_argument("--selection-frames", type=int, default=8)
    parser.add_argument("--answer-frames", type=int, default=8)
    parser.add_argument("--max-selector-tokens", type=int, default=128)
    parser.add_argument(
        "--stream-timing",
        action="store_true",
        help="Use streaming responses and record TTFT/E2E/TPOT in ledger rows.",
    )
    parser.add_argument("--visual-token-counter", choices=("qwen", "unavailable"), default="qwen")
    parser.add_argument(
        "--visual-input-format", choices=("images", "video"), default="images",
        help="Use video for backends such as LLaVA-NeXT-Video that do not accept image items.",
    )
    parser.add_argument("--visual-max-pixels", type=int)
    parser.add_argument(
        "--frame-max-side",
        type=int,
        help=(
            "Downscale each extracted source frame so its longest side is at most this "
            "many pixels. The selected timestamps and aspect ratio are unchanged."
        ),
    )
    parser.add_argument(
        "--expand-official-visual-references", action="store_true",
        help=(
            "Normalize HD-EPIC <TIME>/<BBOX> tags using the official evaluator's text format."
        ),
    )
    parser.add_argument(
        "--include-hdepic-bbox-reference-frame",
        action="store_true",
        help=(
            "Diagnostic only: add the exact question-time BBOX frame as identical "
            "query-side visual context for every representation."
        ),
    )
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--dataset-version", default="cgbench-controlled-public-subtitle-v1")
    parser.add_argument(
        "--disable-json-response-format",
        action="store_true",
        help="Rely on the explicit JSON prompt when a backend mishandles JSON-object guided decoding.",
    )
    parser.add_argument(
        "--stop-string", action="append", default=[],
        help="Backend-specific decoded stop string; may be repeated.",
    )
    parser.add_argument(
        "--stop-token-id", action="append", type=int, default=[],
        help="Backend-specific token id that terminates generation; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answer_prompt_text = answer_prompt_for_frames(args.num_frames)
    frame_level_write_prompt, frame_level_context_write_prompt = (
        frame_level_write_prompts(args.frame_text_profile)
    )
    if not 1 <= args.selection_frames <= args.num_frames:
        raise ValueError("selection_frames must be between 1 and num_frames")
    if args.answer_frames < 1:
        raise ValueError("answer_frames must be positive")
    if args.answer_repair_attempts < 0:
        raise ValueError("answer_repair_attempts must be non-negative")
    if args.answer_repair_json_schema and args.disable_json_response_format:
        raise ValueError(
            "answer_repair_json_schema requires JSON response-format support"
        )
    if args.mode == AKS_QUERY_TIME_RAW_MODE:
        if args.num_frames not in {32, 64, 128}:
            raise ValueError("aks_query_time_raw requires --num-frames 32, 64, or 128")
        if args.selection_manifest is None:
            raise ValueError("--selection-manifest is required for aks_query_time_raw")
        if args.timestamp_manifest is not None:
            raise ValueError(
                "aks_query_time_raw selects frames separately for every question and "
                "cannot use --timestamp-manifest"
            )
        if args.prebuilt_repository_dir is not None:
            raise ValueError(
                "aks_query_time_raw is a Frames/query-time raw condition and cannot "
                "reuse a prebuilt repository"
            )
    elif args.mode != "coarse_to_fine_selection" and args.selection_manifest is not None:
        raise ValueError(
            "--selection-manifest is supported only for aks_query_time_raw and "
            "coarse_to_fine_selection"
        )
    if args.prebuilt_repository_dir is not None:
        if args.mode not in {"write_once_frame_text", *FRAME_TEXT_HYBRID_MODES}:
            raise ValueError(
                "--prebuilt-repository-dir is supported only for frame-level Text "
                "and its joint-input conditions"
            )
        if not args.prebuilt_repository_dir.is_dir():
            raise FileNotFoundError(args.prebuilt_repository_dir)
    if args.frame_max_side is not None and args.frame_max_side <= 0:
        raise ValueError("frame_max_side must be positive")
    if args.include_hdepic_bbox_reference_frame and not args.expand_official_visual_references:
        raise ValueError(
            "--include-hdepic-bbox-reference-frame requires "
            "--expand-official-visual-references"
        )
    source_rows = read_jsonl(args.manifest)
    rows = source_rows
    if args.expand_official_visual_references:
        rows = [
            prepare_hdepic_row(
                row,
                include_bbox_reference=args.include_hdepic_bbox_reference_frame,
            )
            for row in rows
        ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_uid"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["within_video_question_order"]))
    timestamp_rows, timestamp_manifest_sha = load_question_blind_timestamps(
        args.timestamp_manifest,
        set(grouped),
        args.num_frames,
    )
    aks_selections: dict[str, dict] = {}
    aks_selection_manifest_sha = ""
    aks_selection_available_rows = 0
    if args.mode == AKS_QUERY_TIME_RAW_MODE:
        if args.selection_manifest is None:
            raise AssertionError("validated AKS mode lost its selection manifest")
        (
            aks_selections,
            aks_selection_manifest_sha,
            aks_selection_available_rows,
        ) = load_aks_query_selections(
            args.selection_manifest, source_rows, args.num_frames
        )
    if args.mode == "coarse_to_fine_selection":
        if not args.selection_manifest:
            raise ValueError("--selection-manifest is required for coarse_to_fine_selection")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if (args.out_dir / "results.jsonl").exists():
            raise FileExistsError(args.out_dir / "results.jsonl")
        coarse_selections = {str(row["qid"]): row for row in read_jsonl(args.selection_manifest)}
        if set(coarse_selections) != {str(row["qid"]) for row in rows}:
            raise ValueError("coarse selection manifest does not match the question manifest")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        coarse_selections = {}
    results_path = args.out_dir / "results.jsonl"
    ledger_path = args.out_dir / "cost_ledger.jsonl"
    if args.timestamp_manifest is not None:
        (args.out_dir / "timestamp_manifest_provenance.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "question_blind": True,
                    "source_path": str(args.timestamp_manifest.resolve()),
                    "source_sha256": timestamp_manifest_sha,
                    "videos": len(timestamp_rows),
                    "frames_per_video": args.num_frames,
                    "selectors": sorted({
                        str(row["selector_id"]) for row in timestamp_rows.values()
                    }),
                    "selected_times_sha256_by_video": {
                        video: str(timestamp_rows[video]["selected_times_sha256"])
                        for video in sorted(timestamp_rows)
                    },
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    if args.mode == AKS_QUERY_TIME_RAW_MODE:
        selector_descriptors = {
            json.dumps(
                row["selector"],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ): row["selector"]
            for row in aks_selections.values()
        }
        scorer_descriptors = {
            json.dumps(
                row["scorer"],
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ): row["scorer"]
            for row in aks_selections.values()
        }
        (args.out_dir / "query_selection_manifest_provenance.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "schema_version": AKS_SELECTION_SCHEMA_VERSION,
                    "question_aware": True,
                    "question_only": True,
                    "representation": "frames",
                    "lifecycle": "query_time_raw",
                    "cross_question_reuse": False,
                    "source_path": str(args.selection_manifest.resolve()),
                    "source_sha256": aks_selection_manifest_sha,
                    "query_selection_manifest_sha256": aks_selection_manifest_sha,
                    "source_manifest_sha256": next(iter({
                        str(row["source_manifest_sha256"])
                        for row in aks_selections.values()
                    })),
                    "config_sha256": next(iter({
                        str(row["config_sha256"])
                        for row in aks_selections.values()
                    })),
                    "dataset": next(iter({
                        str(row["dataset"]) for row in aks_selections.values()
                    })),
                    "available_rows": aks_selection_available_rows,
                    "consumed_rows": len(aks_selections),
                    "frame_budget": args.num_frames,
                    "frames_per_question": args.num_frames,
                    "selector_ids": sorted({
                        aks_component_id(row["selector"], "selector")
                        for row in aks_selections.values()
                    }),
                    "scorer_ids": sorted({
                        aks_component_id(row["scorer"], "scorer")
                        for row in aks_selections.values()
                    }),
                    "selector_descriptors": [
                        selector_descriptors[key] for key in sorted(selector_descriptors)
                    ],
                    "scorer_descriptors": [
                        scorer_descriptors[key] for key in sorted(scorer_descriptors)
                    ],
                    "selector_cost_attribution": AKS_COST_ATTRIBUTION,
                    "full_standalone_selector_cost_charged_per_question": True,
                    "physical_score_pass_shared_across_budgets": True,
                    "selected_times_sha256_by_qid": {
                        qid: str(aks_selections[qid]["selected_times_sha256"])
                        for qid in sorted(aks_selections)
                    },
                    "answer_input_source_frame_indices_by_qid": {
                        qid: list(
                            aks_selections[qid]["selected_source_frame_indices"]
                        )
                        for qid in sorted(aks_selections)
                    },
                    "total_retained_timestamp_bytes": sum(
                        int(row["retained_timestamp_bytes"])
                        for row in aks_selections.values()
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    client = OpenAIVLMClient(args.base_url, args.model, timeout=600)
    commit = source_revision()
    method = {
        "write_once_text": "ControlledWriteOnceText",
        "write_once_frame_text": "ControlledWriteOnceFrameText",
        "write_once_frame_text_masked": "ControlledWriteOnceFrameTextMasked",
        "blind_replay_text": "ControlledBlindReplayText",
        "query_conditioned_text": "ControlledQueryConditionedText",
        "query_conditioned_frame_text": "ControlledQueryConditionedFrameText",
        "blind_replay_frame_text": "ControlledBlindReplayFrameText",
        "blind_replay_text_matched": "ControlledBlindReplayTextMatched",
        "query_conditioned_text_matched": "ControlledQueryConditionedTextMatched",
        "query_time_raw": "ControlledQueryTimeRaw",
        "write_once_raw_cache": "ControlledWriteOnceRawCache",
        "write_once_visual_token_cache": "ControlledVisualTokenCache",
        "write_once_model_state_cache": "ControlledModelStateCache",
        FRAME_TEXT_HYBRID_VISUAL_MODE: "ControlledFrameTextVisualCache",
        FRAME_TEXT_HYBRID_KV_MODE: "ControlledFrameTextModelStateCache",
        "query_aware_selection": "ControlledQueryAwareSelect",
        AKS_QUERY_TIME_RAW_MODE: "ControlledAKSQueryTimeRaw",
        "coarse_to_fine_selection": "ControlledClipCoarseToFine",
        "coarse_vlm_local_selection": "ControlledCoarseVLMLocal",
        "annotated_evidence": "ControlledAnnotatedEvidence",
        "annotated_evidence_frame_text": "ControlledAnnotatedEvidenceFrameText",
        "annotated_evidence_query_conditioned_frame_text": (
            "ControlledAnnotatedEvidenceQueryConditionedFrameText"
        ),
    }[args.mode]
    failures = 0

    for video_id, questions in sorted(grouped.items()):
        video_path = Path(str(questions[0]["video_path"]))
        if not video_path.is_absolute():
            video_path = args.workspace_root / video_path
        total_frames, duration, source_average_fps = metadata(
            video_path, float(questions[0].get("duration") or 0)
        )
        timestamp_row = timestamp_rows.get(video_id)
        if timestamp_row is None:
            times = selected_times(total_frames, duration, args.num_frames, sampling="uniform")
            observation_policy = "uniform"
            selected_times_sha = hashlib.sha256(
                json.dumps(times, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        else:
            times = list(timestamp_row["selected_times"])
            if times[-1] > duration + 1e-3:
                raise ValueError(
                    f"timestamp {times[-1]:.6f}s exceeds duration {duration:.6f}s for {video_id}"
                )
            observation_policy = str(timestamp_row["selector_id"])
            selected_times_sha = str(timestamp_row["selected_times_sha256"])
            append_selector_cost_events(
                ledger_path=ledger_path,
                run_id=args.run_id,
                commit=commit,
                method=method,
                dataset_version=args.dataset_version,
                video_id=video_id,
                timestamp_row=timestamp_row,
                timestamp_manifest_sha=timestamp_manifest_sha,
            )
        resolution_label = f"maxside{args.frame_max_side}" if args.frame_max_side else "native"
        frame_ids = [
            f"{video_id}:controlled:{selected_times_sha[:12]}:{resolution_label}:{index}:{stamp:.6f}"
            for index, stamp in enumerate(times)
        ]
        repository = ""
        answer_repository = ""
        repository_path: Path | None = None
        shared_paths: list[str] = []

        if args.mode in {
            "write_once_text", "write_once_frame_text", "write_once_frame_text_masked",
            "write_once_raw_cache", "write_once_visual_token_cache",
            "write_once_model_state_cache", *FRAME_TEXT_HYBRID_MODES,
        } and (
            args.prebuilt_repository_dir is None
            or args.mode in FRAME_TEXT_HYBRID_MODES
        ):
            shared_dir = args.frame_cache / video_id / args.mode
            decode_started = time.perf_counter()
            shared_paths = extract_frames_at_times(
                str(video_path), times, shared_dir, total_frames, duration,
                max_side=args.frame_max_side,
            )
            decode_latency = time.perf_counter() - decode_started
            ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, "__video_write__", 1))
            ledger.append(
                phase="write", model_id="opencv_video_decoder", cache_status="miss",
                decoded_frames=args.num_frames, unique_frame_ids=frame_ids,
                latency_s=decode_latency, active_gpu_count=0, selected_times=times,
                measurement_note={
                    "write_once_raw_cache": "Question-blind CPU frame decode for a raw cache.",
                    "write_once_visual_token_cache": (
                        "Question-blind CPU frame decode before visual-encoder cache warmup."
                    ),
                    "write_once_model_state_cache": "Question-blind CPU frame decode before model-state cache warmup.",
                    "write_once_text": "Question-blind CPU frame decode before writing the text repository.",
                    "write_once_frame_text": (
                        "Question-blind CPU frame decode before writing one text entry per source frame."
                    ),
                    "write_once_frame_text_masked": (
                        "Masked-question CPU frame decode before writing one text entry per source frame."
                    ),
                    FRAME_TEXT_HYBRID_VISUAL_MODE: (
                        "Question-blind CPU frame decode before writing Text and retaining visual encoder outputs."
                    ),
                    FRAME_TEXT_HYBRID_KV_MODE: (
                        "Question-blind CPU frame decode before writing Text and retaining multimodal prefix state."
                    ),
                }[args.mode],
                observation_policy=observation_policy,
                timestamp_manifest_sha256=timestamp_manifest_sha,
            )

        if args.prebuilt_repository_dir is not None:
            repository_path = resolve_prebuilt_repository(
                args.prebuilt_repository_dir, video_id
            )
            repository = repository_path.read_text(encoding="utf-8").strip()
            validate_frame_level_entries(repository, list(range(args.num_frames)))
            if args.mode in FRAME_TEXT_HYBRID_MODES:
                import_prebuilt_frame_text_writer_cost(
                    directory=args.prebuilt_repository_dir,
                    ledger_path=ledger_path,
                    run_id=args.run_id,
                    commit=commit,
                    method=method,
                    dataset_version=args.dataset_version,
                    model_id=args.model,
                    video_id=video_id,
                    frame_ids=frame_ids,
                    selected_times_sha256=selected_times_sha,
                    timestamp_manifest_sha256=timestamp_manifest_sha,
                    out_dir=args.out_dir,
                )
        elif args.mode == "write_once_text" or args.mode in FRAME_TEXT_WRITE_ONCE_MODES:
            frame_level = args.mode in FRAME_TEXT_WRITE_ONCE_MODES
            masked_frame_level = args.mode == "write_once_frame_text_masked"
            chunks = []
            for start in range(0, args.num_frames, args.writer_chunk_size):
                end = min(args.num_frames, start + args.writer_chunk_size)
                paths = shared_paths[start:end]
                indices = list(range(start, end))
                writer_prompt = (
                    f"{frame_level_context_write_prompt}\nQuestion context: [MASKED]\n"
                    f"Required source-frame indices: {', '.join(map(str, indices))}."
                    if masked_frame_level
                    else f"{frame_level_write_prompt}\nRequired source-frame indices: "
                    f"{', '.join(map(str, indices))}."
                    if frame_level
                    else WRITE_PROMPT
                )
                messages = [{"role": "user", "content": [{"type": "text", "text": writer_prompt}, *media_content(paths, indices, times, args.visual_input_format)]}]
                started = time.perf_counter()
                response = client.chat(
                    messages,
                    max_tokens=(args.frame_writer_tokens if frame_level else args.max_writer_tokens),
                    temperature=0.0,
                    stop=args.stop_string or None,
                    stop_token_ids=args.stop_token_id or None,
                )
                latency = time.perf_counter() - started
                usage = response.get("usage") or {}
                writer_cached_prompt_tokens = cached_prompt_tokens(usage)
                chunk_visual_tokens = visual_tokens(paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, "__video_write__", 1))
                ledger.append(
                    phase="write", model_id=args.model, cache_status="miss",
                    input_text_tokens=max(0, int(usage.get("prompt_tokens") or 0) - chunk_visual_tokens),
                    output_text_tokens=int(usage.get("completion_tokens") or 0),
                    visual_tokens=chunk_visual_tokens,
                    decoded_frames=0,
                    unique_frame_ids=frame_ids[start:end],
                    latency_s=latency,
                    peak_vram_bytes=sampled_visible_gpu_memory_bytes(),
                    cached_prompt_tokens=writer_cached_prompt_tokens,
                    selected_times=times[start:end],
                    measurement_note=(
                        "Question-blind frame-level text write: one validated entry per fixed source frame."
                        if frame_level
                        else "Question-blind text repository write over fixed source frames."
                    ),
                )
                if frame_level:
                    collected, rejected = collect_frame_level_entries(str(response["text"]), indices)
                    append_jsonl(args.out_dir / "frame_writer_chunks.jsonl", {
                        "video_uid": video_id,
                        "start_frame": start,
                        "end_frame": end - 1,
                        "expected_indices": indices,
                        "attempt": 0,
                        "rejected_lines": rejected,
                        "raw_response": response["text"],
                    })
                    for repair_attempt in range(1, 4):
                        missing = [index for index in indices if index not in collected]
                        if not missing:
                            break
                        repair_paths = [shared_paths[index] for index in missing]
                        repair_prompt = (
                            f"{frame_level_context_write_prompt}\nQuestion context: [MASKED]\n"
                            if masked_frame_level
                            else f"{frame_level_write_prompt}\n"
                        ) + (
                            f"This is a repair request. Generate complete schema lines for these "
                            f"missing source-frame indices: {', '.join(map(str, missing))}. "
                            "Do not output a bare list of indices, and do not describe any other frames."
                        )
                        repair_messages = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": repair_prompt},
                                *media_content(repair_paths, missing, times, args.visual_input_format),
                            ],
                        }]
                        repair_started = time.perf_counter()
                        repair_response = client.chat(
                            repair_messages,
                            max_tokens=args.frame_writer_tokens,
                            temperature=0.0,
                            stop=args.stop_string or None,
                            stop_token_ids=args.stop_token_id or None,
                        )
                        repair_latency = time.perf_counter() - repair_started
                        repair_usage = repair_response.get("usage") or {}
                        repair_cached_prompt_tokens = cached_prompt_tokens(repair_usage)
                        repair_visual_tokens = (
                            visual_tokens(
                                repair_paths,
                                args.visual_model_path,
                                args.visual_max_pixels,
                            )
                            if args.visual_token_counter == "qwen"
                            else 0
                        )
                        repair_ledger = CostLedger(
                            ledger_path,
                            LedgerContext(
                                args.run_id,
                                commit,
                                method,
                                args.dataset_version,
                                video_id,
                                "__video_write__",
                                1,
                            ),
                        )
                        repair_ledger.append(
                            phase="write",
                            model_id=args.model,
                            cache_status="miss",
                            input_text_tokens=max(
                                0,
                                int(repair_usage.get("prompt_tokens") or 0)
                                - repair_visual_tokens,
                            ),
                            output_text_tokens=int(repair_usage.get("completion_tokens") or 0),
                            visual_tokens=repair_visual_tokens,
                            decoded_frames=0,
                            unique_frame_ids=[frame_ids[index] for index in missing],
                            latency_s=repair_latency,
                            peak_vram_bytes=sampled_visible_gpu_memory_bytes(),
                            cached_prompt_tokens=repair_cached_prompt_tokens,
                            selected_times=[times[index] for index in missing],
                            measurement_note=(
                                "Frame-level text repair for source indices omitted by the "
                                "initial deterministic writer call."
                            ),
                        )
                        repaired, repair_rejected = collect_frame_level_entries(
                            str(repair_response["text"]), missing
                        )
                        collected.update(repaired)
                        append_jsonl(args.out_dir / "frame_writer_chunks.jsonl", {
                            "video_uid": video_id,
                            "start_frame": start,
                            "end_frame": end - 1,
                            "expected_indices": missing,
                            "attempt": repair_attempt,
                            "rejected_lines": repair_rejected,
                            "raw_response": repair_response["text"],
                        })
                    missing = [index for index in indices if index not in collected]
                    if missing:
                        raise ValueError(
                            "frame-level writer repair exhausted: "
                            f"expected_indices={indices} missing_indices={missing}"
                        )
                    chunks.extend(collected[index] for index in indices)
                else:
                    chunks.append(f"Frames {start}-{end - 1}:\n{response['text']}")
            repository = ("\n" if frame_level else "\n\n").join(chunks)
            (args.out_dir / f"{video_id}.repository.txt").write_text(repository + "\n", encoding="utf-8")

        answer_repository = repository
        if args.mode in FRAME_TEXT_WRITE_ONCE_MODES:
            answer_repository = attach_frame_times(repository, times)
            timed_path = args.out_dir / f"{video_id}.repository_with_time.txt"
            timed_path.write_text(answer_repository + "\n", encoding="utf-8")
            semantic_bytes = len(repository.encode("utf-8"))
            answer_bytes = len(answer_repository.encode("utf-8"))
            metadata_row = {
                "video_uid": video_id,
                "source_frame_count": args.num_frames,
                "frame_time_metadata_attached": True,
                "semantic_repository_sha256": hashlib.sha256(
                    repository.encode("utf-8")
                ).hexdigest(),
                "answer_repository_sha256": hashlib.sha256(
                    answer_repository.encode("utf-8")
                ).hexdigest(),
                "semantic_repository_bytes": semantic_bytes,
                "answer_repository_bytes": answer_bytes,
                "time_metadata_bytes": answer_bytes - semantic_bytes,
                "timestamped_repository_path": str(timed_path.resolve()),
            }
            append_jsonl(args.out_dir / "frame_time_metadata.jsonl", metadata_row)
            if repository_path is not None:
                append_jsonl(
                    args.out_dir / "prebuilt_repository_provenance.jsonl",
                    metadata_row
                    | {
                        "repository_path": str(repository_path.resolve()),
                        "sha256": hashlib.sha256(repository.encode("utf-8")).hexdigest(),
                    },
                )

        if args.mode in STATE_CACHE_MODES:
            warmup_content = [
                {"type": "text", "text": answer_prompt_text},
                *media_content(shared_paths, list(range(args.num_frames)), times, args.visual_input_format),
                *(
                    [{
                        "type": "text",
                        # Keep this content item byte-identical to the one used
                        # at answer time. Prefix caching depends on the boundary
                        # after the saved repository as well as on its contents.
                        "text": f"Repository:\n{answer_repository}\n\n",
                    }]
                    if args.mode == FRAME_TEXT_HYBRID_KV_MODE
                    else []
                ),
                {"type": "text", "text": MODEL_STATE_WARMUP},
            ]
            started = time.perf_counter()
            warmup_call = client.chat_stream_timed if args.stream_timing else client.chat
            warmup_response = warmup_call(
                [{"role": "user", "content": warmup_content}],
                max_tokens=1,
                temperature=0.0,
                stop=args.stop_string or None,
                stop_token_ids=args.stop_token_id or None,
            )
            latency = time.perf_counter() - started
            usage = warmup_response.get("usage") or {}
            warmup_visual_tokens = (
                visual_tokens(shared_paths, args.visual_model_path, args.visual_max_pixels)
                if args.visual_token_counter == "qwen" else 0
            )
            ledger = CostLedger(
                ledger_path,
                LedgerContext(
                    args.run_id, commit, method, args.dataset_version,
                    video_id, "__video_write__", 1,
                ),
            )
            ledger.append(
                phase="write", model_id=args.model, cache_status="miss",
                input_text_tokens=max(0, int(usage.get("prompt_tokens") or 0) - warmup_visual_tokens),
                output_text_tokens=int(usage.get("completion_tokens") or 0),
                visual_tokens=warmup_visual_tokens, decoded_frames=0,
                unique_frame_ids=frame_ids, latency_s=latency,
                peak_vram_bytes=sampled_visible_gpu_memory_bytes(), selected_times=times,
                cached_prompt_tokens=cached_prompt_tokens(usage),
                state_budget_frames=args.num_frames,
                state_storage_unit=(
                    "vllm_visual_encoder_outputs"
                    if args.mode in VISUAL_CACHE_MODES
                    else "vllm_multimodal_prefix_tokens"
                ),
                measurement_note=(
                    "Question-blind visual-encoder warmup. Prefix caching is disabled, so later "
                    "questions reuse visual encoder outputs but recompute language-model prefill."
                    if args.mode in VISUAL_CACHE_MODES
                    else (
                        "Question-blind multimodal prefix warmup. The cached prefix contains both "
                        "the visual state and the saved Text repository."
                        if args.mode == FRAME_TEXT_HYBRID_KV_MODE
                        else "Question-blind multimodal prefix warmup. The server retains multimodal "
                        "encoder outputs and prefix KV state for reuse by later questions."
                    )
                ),
                observation_policy=observation_policy,
                timestamp_manifest_sha256=timestamp_manifest_sha,
                **(warmup_response.get("timing") or {}),
            )

        for row in questions:
            qid = str(row["qid"])
            options = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(row["choices"]))
            conditioned_notes = ""
            blind_replay_repository = ""
            selected_frame_indices: list[int] = []
            selection_fallback_reason = ""
            answer_frame_ids = frame_ids
            answer_times = times
            answer_input_source_frame_indices: list[int] = []
            result_selected_times = times
            result_selected_times_sha = selected_times_sha
            result_observation_policy = observation_policy
            query_selection_manifest_sha = ""
            selector_cost_imported_in_full = False
            selector_candidate_count = 0
            selector_retained_timestamp_bytes = 0
            selector_cost_attribution = ""
            physical_score_pass_shared_across_budgets = False
            selected_anchor_indices: list[int] = []
            selected_anchor_times: list[float] = []
            coarse_scan_notes = ""
            reference_times = [
                float(value)
                for value in row.get("_official_bbox_reference_times", [])
            ]
            reference_paths: list[str] = []
            reference_frame_ids: list[str] = []
            if reference_times:
                reference_dir = args.frame_cache / video_id / f"question_reference_{qid}"
                reference_started = time.perf_counter()
                reference_paths = extract_frames_at_times(
                    str(video_path),
                    reference_times,
                    reference_dir,
                    total_frames,
                    duration,
                    max_side=args.frame_max_side,
                )
                reference_latency = time.perf_counter() - reference_started
                if len(reference_paths) != len(reference_times):
                    raise RuntimeError(
                        f"Expected {len(reference_times)} HD-EPIC question-reference frames "
                        f"for {video_id}/{qid}, got {len(reference_paths)}"
                    )
                raw_reference_question = str(
                    row.get("_official_bbox_question") or row.get("question") or ""
                )
                for path in reference_paths:
                    draw_official_bbox(path, raw_reference_question)
                reference_frame_ids = [
                    f"{video_id}:question_reference:{qid}:{stamp:.6f}"
                    for stamp in reference_times
                ]
                CostLedger(
                    ledger_path,
                    LedgerContext(
                        args.run_id,
                        commit,
                        method,
                        args.dataset_version,
                        video_id,
                        qid,
                        1,
                    ),
                ).append(
                    phase="observe",
                    model_id="opencv_video_decoder",
                    cache_status="miss",
                    decoded_frames=len(reference_paths),
                    unique_frame_ids=reference_frame_ids,
                    latency_s=reference_latency,
                    active_gpu_count=0,
                    selected_times=reference_times,
                    measurement_note=(
                        "Question-time decode of the exact HD-EPIC BBOX reference frame. "
                        "This query-side referent is supplied identically to every representation."
                    ),
                )
            answer_repeated_frames = (
                args.num_frames
                if args.mode in {
                    "query_time_raw",
                    "write_once_raw_cache",
                    "annotated_evidence",
                    AKS_QUERY_TIME_RAW_MODE,
                }
                else 0
            )
            if args.mode in {"write_once_text", "write_once_frame_text", "write_once_frame_text_masked"}:
                content = f"{answer_prompt_text}\n\nRepository:\n{answer_repository}\n\nQuestion:\n{row['question']}\n\nOptions:\n{options}"
                messages = [{"role": "user", "content": content}]
                paths = []
                decoded = 0
                tokens = 0
                cache_status = "hit"
            elif args.mode in FRAME_TEXT_HYBRID_MODES:
                # Joint-input conditions expose both the saved frame-level Text record
                # and the same retained visual evidence to every question.  This is a
                # real fusion input, not an ensemble over separately produced answers.
                content = [
                    {"type": "text", "text": answer_prompt_text},
                    *media_content(
                        shared_paths,
                        list(range(args.num_frames)),
                        times,
                        args.visual_input_format,
                    ),
                    {
                        "type": "text",
                        # This exact item also appears in the warmup request for
                        # Text+Prefix State. Put the question in a new item so it
                        # cannot change tokenization at the repository boundary.
                        "text": f"Repository:\n{answer_repository}\n\n",
                    },
                    {
                        "type": "text",
                        "text": f"Question:\n{row['question']}\n\nOptions:\n{options}",
                    },
                ]
                messages = [{"role": "user", "content": content}]
                paths = shared_paths
                decoded = 0
                tokens = (
                    visual_tokens(paths, args.visual_model_path, args.visual_max_pixels)
                    if args.visual_token_counter == "qwen"
                    else 0
                )
                cache_status = "hit"
            elif args.mode in {"blind_replay_text", "blind_replay_text_matched"}:
                matched_writer = args.mode == "blind_replay_text_matched"
                query_dir = args.frame_cache / video_id / f"blind_replay_{qid}"
                decode_started = time.perf_counter()
                replay_paths = extract_frames_at_times(
                    str(video_path), times, query_dir, total_frames, duration,
                    max_side=args.frame_max_side,
                )
                if len(replay_paths) != args.num_frames:
                    raise RuntimeError(
                        f"Expected {args.num_frames} replay frames for {video_id}/{qid}, got {len(replay_paths)}"
                    )
                decode_latency = time.perf_counter() - decode_started
                decode_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                decode_ledger.append(
                    phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                    decoded_frames=args.num_frames, unique_frame_ids=frame_ids,
                    latency_s=decode_latency, active_gpu_count=0, selected_times=times,
                    measurement_note="Repeated question-blind CPU decode before replaying the generic text writer.",
                )
                replay_chunks = []
                for start in range(0, args.num_frames, args.writer_chunk_size):
                    end = min(args.num_frames, start + args.writer_chunk_size)
                    chunk_paths = replay_paths[start:end]
                    indices = list(range(start, end))
                    writer_prompt = (
                        f"{MATCHED_WRITE_PROMPT}\n\nQuestion context: unavailable."
                        if matched_writer else WRITE_PROMPT
                    )
                    writer_messages = [{"role": "user", "content": [{"type": "text", "text": writer_prompt}, *media_content(chunk_paths, indices, times, args.visual_input_format)]}]
                    writer_started = time.perf_counter()
                    writer_response = client.chat(
                        writer_messages,
                        max_tokens=(args.matched_writer_tokens if matched_writer else args.max_writer_tokens),
                        min_tokens=(args.matched_writer_tokens if matched_writer else None),
                        temperature=0.0,
                        stop=args.stop_string or None,
                        stop_token_ids=args.stop_token_id or None,
                    )
                    writer_latency = time.perf_counter() - writer_started
                    writer_usage = writer_response.get("usage") or {}
                    writer_visual_tokens = visual_tokens(chunk_paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                    writer_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                    writer_ledger.append(
                        phase="observe", model_id=args.model, cache_status="miss",
                        input_text_tokens=max(0, int(writer_usage.get("prompt_tokens") or 0) - writer_visual_tokens),
                        output_text_tokens=int(writer_usage.get("completion_tokens") or 0),
                        visual_tokens=writer_visual_tokens, decoded_frames=0,
                        repeated_frames=len(chunk_paths), unique_frame_ids=frame_ids[start:end],
                        latency_s=writer_latency, peak_vram_bytes=sampled_visible_gpu_memory_bytes(),
                        selected_times=times[start:end],
                        measurement_note=(
                            "Budget-locked question-blind writer using the matched RQ3 schema."
                            if matched_writer
                            else "Repeated question-blind generic repository write over matched source frames."
                        ),
                    )
                    replay_chunks.append(f"Frames {start}-{end - 1}:\n{writer_response['text']}")
                blind_replay_repository = "\n\n".join(replay_chunks)
                repository_label = "Matched evidence notes" if matched_writer else "Repository"
                content = f"{answer_prompt_text}\n\n{repository_label}:\n{blind_replay_repository}\n\nQuestion:\n{row['question']}\n\nOptions:\n{options}"
                messages = [{"role": "user", "content": content}]
                paths = []
                decoded = 0
                tokens = 0
                cache_status = "miss"
            elif args.mode in {
                "query_conditioned_text",
                "query_conditioned_text_matched",
                "blind_replay_frame_text",
                "query_conditioned_frame_text",
                "annotated_evidence_frame_text",
                "annotated_evidence_query_conditioned_frame_text",
            }:
                matched_writer = args.mode == "query_conditioned_text_matched"
                annotated_frame_text = args.mode in {
                    "annotated_evidence_frame_text",
                    "annotated_evidence_query_conditioned_frame_text",
                }
                annotated_question_aware = (
                    args.mode == "annotated_evidence_query_conditioned_frame_text"
                )
                frame_level_conditioned = args.mode in {
                    "blind_replay_frame_text",
                    "query_conditioned_frame_text",
                    "annotated_evidence_frame_text",
                    "annotated_evidence_query_conditioned_frame_text",
                }
                frame_level_context = (
                    "[MASKED]"
                    if args.mode in {"blind_replay_frame_text", "annotated_evidence_frame_text"}
                    else str(row["question"])
                )
                if annotated_frame_text:
                    windows = [
                        {"start": float(interval[0]), "end": float(interval[1])}
                        for interval in row.get("clue_intervals", [])
                        if len(interval) == 2 and float(interval[1]) > float(interval[0])
                    ]
                    write_times = oracle_window_times(windows, args.num_frames)
                    if len(write_times) != args.num_frames:
                        raise RuntimeError(
                            f"Expected {args.num_frames} annotated-evidence times for "
                            f"{video_id}/{qid}, got {len(write_times)}"
                        )
                    write_frame_ids = [
                        f"{video_id}:annotated_evidence:{index}:{stamp:.6f}"
                        for index, stamp in enumerate(write_times)
                    ]
                    answer_times = write_times
                    result_selected_times = write_times
                    answer_frame_ids = write_frame_ids
                    query_dir = args.frame_cache / video_id / f"{args.mode}_{qid}"
                else:
                    write_times = times
                    write_frame_ids = frame_ids
                    query_dir = args.frame_cache / video_id / f"query_conditioned_{qid}"
                decode_started = time.perf_counter()
                paths = extract_frames_at_times(
                    str(video_path), write_times, query_dir, total_frames, duration,
                    max_side=args.frame_max_side,
                )
                if len(paths) != args.num_frames:
                    raise RuntimeError(
                        f"Expected {args.num_frames} writer frames for {video_id}/{qid}, "
                        f"got {len(paths)}"
                    )
                decode_latency = time.perf_counter() - decode_started
                decode_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                decode_ledger.append(
                    phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                    decoded_frames=args.num_frames, unique_frame_ids=write_frame_ids,
                    latency_s=decode_latency, active_gpu_count=0, selected_times=write_times,
                    measurement_note=(
                        "Diagnostic question-time decode sampled inside annotated evidence intervals "
                        f"before a {'question-aware' if annotated_question_aware else 'question-blind'} "
                        "frame-to-text write."
                        if annotated_frame_text
                        else "Question-time CPU frame decode before conditioned evidence extraction."
                    ),
                )
                chunks = []
                for start in range(0, args.num_frames, args.writer_chunk_size):
                    end = min(args.num_frames, start + args.writer_chunk_size)
                    chunk_paths = paths[start:end]
                    indices = list(range(start, end))
                    conditioned_prompt = (
                        f"{frame_level_write_prompt}\nRequired source-frame indices: "
                        f"{', '.join(map(str, indices))}."
                        if args.mode == "annotated_evidence_frame_text"
                        else f"{frame_level_context_write_prompt}\nQuestion context: "
                        f"{frame_level_context}\nRequired source-frame "
                        f"indices: {', '.join(map(str, indices))}."
                        if frame_level_conditioned
                        else f"{MATCHED_WRITE_PROMPT}\n\nQuestion context: {row['question']}"
                        if matched_writer
                        else f"{WRITE_PROMPT}\nThe current question is known: {row['question']}\nExtract evidence relevant to this question while preserving source frame indices."
                    )
                    writer_messages = [{"role": "user", "content": [{"type": "text", "text": conditioned_prompt}, *media_content(chunk_paths, indices, write_times, args.visual_input_format)]}]
                    writer_started = time.perf_counter()
                    writer_response = client.chat(
                        writer_messages,
                        max_tokens=(
                            args.frame_writer_tokens
                            if frame_level_conditioned
                            else args.matched_writer_tokens
                            if matched_writer
                            else args.max_writer_tokens
                        ),
                        min_tokens=(args.matched_writer_tokens if matched_writer else None),
                        temperature=0.0,
                        stop=args.stop_string or None,
                        stop_token_ids=args.stop_token_id or None,
                    )
                    writer_latency = time.perf_counter() - writer_started
                    writer_usage = writer_response.get("usage") or {}
                    writer_visual_tokens = visual_tokens(chunk_paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                    writer_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                    writer_ledger.append(
                        phase="observe", model_id=args.model, cache_status="miss",
                        input_text_tokens=max(0, int(writer_usage.get("prompt_tokens") or 0) - writer_visual_tokens),
                        output_text_tokens=int(writer_usage.get("completion_tokens") or 0),
                        visual_tokens=writer_visual_tokens,
                        decoded_frames=0,
                        repeated_frames=len(chunk_paths),
                        unique_frame_ids=write_frame_ids[start:end],
                        latency_s=writer_latency,
                        peak_vram_bytes=sampled_visible_gpu_memory_bytes(), selected_times=write_times[start:end],
                        measurement_note=(
                            (
                                "Question-aware frame-level text write: one validated entry per fixed source frame."
                                if args.mode == "query_conditioned_frame_text"
                                else (
                                    "Annotated-evidence frame-level text write with the question available: "
                                    "one validated entry per evidence frame."
                                    if annotated_question_aware
                                    else
                                    "Annotated-evidence frame-level text write with the question hidden: "
                                    "one validated entry per evidence frame."
                                    if annotated_frame_text
                                    else "Masked-question frame-level text replay: one validated entry per fixed source frame."
                                )
                            )
                            if frame_level_conditioned
                            else
                            "Budget-locked question-aware writer using the matched RQ3 schema."
                            if matched_writer
                            else "Question-conditioned evidence extraction over the matched source frames."
                        ),
                    )
                    if frame_level_conditioned:
                        collected, rejected = collect_frame_level_entries(
                            str(writer_response["text"]), indices
                        )
                        append_jsonl(args.out_dir / "frame_writer_chunks.jsonl", {
                            "video_uid": video_id,
                            "qid": qid,
                            "start_frame": start,
                            "end_frame": end - 1,
                            "expected_indices": indices,
                            "attempt": 0,
                            "rejected_lines": rejected,
                            "raw_response": writer_response["text"],
                        })
                        for repair_attempt in range(1, 4):
                            missing = [index for index in indices if index not in collected]
                            if not missing:
                                break
                            repair_paths = [paths[index] for index in missing]
                            repair_prompt = (
                                (
                                    f"{frame_level_write_prompt}\n"
                                    if args.mode == "annotated_evidence_frame_text"
                                    else f"{frame_level_context_write_prompt}\nQuestion context: "
                                    f"{frame_level_context}\n"
                                )
                                + "This is a repair request. Generate complete "
                                f"schema lines for these missing source-frame indices: "
                                f"{', '.join(map(str, missing))}. Do not output a bare list of "
                                "indices, and do not describe any other frames."
                            )
                            repair_messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": repair_prompt},
                                    *media_content(
                                        repair_paths,
                                        missing,
                                        write_times,
                                        args.visual_input_format,
                                    ),
                                ],
                            }]
                            repair_started = time.perf_counter()
                            repair_response = client.chat(
                                repair_messages,
                                max_tokens=args.frame_writer_tokens,
                                temperature=0.0,
                                stop=args.stop_string or None,
                                stop_token_ids=args.stop_token_id or None,
                            )
                            repair_latency = time.perf_counter() - repair_started
                            repair_usage = repair_response.get("usage") or {}
                            repair_visual_tokens = (
                                visual_tokens(
                                    repair_paths,
                                    args.visual_model_path,
                                    args.visual_max_pixels,
                                )
                                if args.visual_token_counter == "qwen"
                                else 0
                            )
                            repair_ledger = CostLedger(
                                ledger_path,
                                LedgerContext(
                                    args.run_id,
                                    commit,
                                    method,
                                    args.dataset_version,
                                    video_id,
                                    qid,
                                    1,
                                ),
                            )
                            repair_ledger.append(
                                phase="observe",
                                model_id=args.model,
                                cache_status="miss",
                                input_text_tokens=max(
                                    0,
                                    int(repair_usage.get("prompt_tokens") or 0)
                                    - repair_visual_tokens,
                                ),
                                output_text_tokens=int(
                                    repair_usage.get("completion_tokens") or 0
                                ),
                                visual_tokens=repair_visual_tokens,
                                decoded_frames=0,
                                repeated_frames=len(repair_paths),
                                unique_frame_ids=[write_frame_ids[index] for index in missing],
                                latency_s=repair_latency,
                                peak_vram_bytes=sampled_visible_gpu_memory_bytes(),
                                selected_times=[write_times[index] for index in missing],
                                measurement_note=(
                                    "Frame-level context-control repair for source indices "
                                    "omitted by the initial deterministic writer call."
                                ),
                            )
                            repaired, repair_rejected = collect_frame_level_entries(
                                str(repair_response["text"]), missing
                            )
                            collected.update(repaired)
                            append_jsonl(args.out_dir / "frame_writer_chunks.jsonl", {
                                "video_uid": video_id,
                                "qid": qid,
                                "start_frame": start,
                                "end_frame": end - 1,
                                "expected_indices": missing,
                                "attempt": repair_attempt,
                                "rejected_lines": repair_rejected,
                                "raw_response": repair_response["text"],
                            })
                        missing = [index for index in indices if index not in collected]
                        if missing:
                            raise ValueError(
                                "question-aware frame-level writer repair exhausted: "
                                f"expected_indices={indices} missing_indices={missing}"
                            )
                        chunks.extend(collected[index] for index in indices)
                    else:
                        chunks.append(f"Frames {start}-{end - 1}:\n{writer_response['text']}")
                conditioned_notes = ("\n" if frame_level_conditioned else "\n\n").join(chunks)
                notes_label = (
                    "Repository"
                    if frame_level_conditioned
                    else "Matched evidence notes"
                    if matched_writer
                    else "Question-conditioned evidence notes"
                )
                content = f"{answer_prompt_text}\n\n{notes_label}:\n{conditioned_notes}\n\nQuestion:\n{row['question']}\n\nOptions:\n{options}"
                messages = [{"role": "user", "content": content}]
                paths = []
                decoded = 0
                tokens = 0
                cache_status = "miss" if annotated_frame_text else "hit"
            elif args.mode in {"write_once_raw_cache", "write_once_visual_token_cache"}:
                content = [{"type": "text", "text": answer_prompt_text}, *media_content(shared_paths, list(range(args.num_frames)), times, args.visual_input_format), {"type": "text", "text": f"Question:\n{row['question']}\n\nOptions:\n{options}"}]
                messages = [{"role": "user", "content": content}]
                paths = shared_paths
                decoded = 0
                tokens = visual_tokens(paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                cache_status = "hit"
            elif args.mode == "write_once_model_state_cache":
                content = [
                    {"type": "text", "text": answer_prompt_text},
                    *media_content(shared_paths, list(range(args.num_frames)), times, args.visual_input_format),
                    {"type": "text", "text": f"Question:\n{row['question']}\n\nOptions:\n{options}"},
                ]
                messages = [{"role": "user", "content": content}]
                paths = shared_paths
                decoded = 0
                tokens = visual_tokens(paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                cache_status = "hit"
            elif args.mode == "annotated_evidence":
                windows = [
                    {"start": float(interval[0]), "end": float(interval[1])}
                    for interval in row.get("clue_intervals", [])
                    if len(interval) == 2 and float(interval[1]) > float(interval[0])
                ]
                answer_times = oracle_window_times(windows, args.num_frames)
                if len(answer_times) != args.num_frames:
                    raise RuntimeError(
                        f"Expected {args.num_frames} annotated-evidence times for {video_id}/{qid}, "
                        f"got {len(answer_times)}"
                    )
                result_selected_times = answer_times
                query_dir = args.frame_cache / video_id / f"annotated_evidence_{qid}"
                decode_started = time.perf_counter()
                paths = extract_frames_at_times(
                    str(video_path), answer_times, query_dir, total_frames, duration,
                    max_side=args.frame_max_side,
                )
                decode_latency = time.perf_counter() - decode_started
                if len(paths) != args.num_frames:
                    raise RuntimeError(
                        f"Expected {args.num_frames} annotated-evidence frames for {video_id}/{qid}, "
                        f"got {len(paths)}"
                    )
                answer_frame_ids = [
                    f"{video_id}:annotated_evidence:{index}:{stamp:.6f}"
                    for index, stamp in enumerate(answer_times)
                ]
                CostLedger(
                    ledger_path,
                    LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1),
                ).append(
                    phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                    decoded_frames=args.num_frames, unique_frame_ids=answer_frame_ids,
                    latency_s=decode_latency, active_gpu_count=0, selected_times=answer_times,
                    measurement_note=(
                        "Diagnostic question-time decode sampled inside annotated evidence intervals."
                    ),
                )
                content = [
                    {"type": "text", "text": answer_prompt_text},
                    *media_content(
                        paths, list(range(args.num_frames)), answer_times, args.visual_input_format
                    ),
                    {"type": "text", "text": f"Question:\n{row['question']}\n\nOptions:\n{options}"},
                ]
                messages = [{"role": "user", "content": content}]
                decoded = 0
                tokens = (
                    visual_tokens(paths, args.visual_model_path, args.visual_max_pixels)
                    if args.visual_token_counter == "qwen" else 0
                )
                cache_status = "miss"
            elif args.mode == AKS_QUERY_TIME_RAW_MODE:
                selection = aks_selections[qid]
                if Path(str(selection["video_path"])).resolve() != video_path.resolve():
                    raise ValueError(
                        f"AKS video path mismatch for source question {video_id}/{qid}"
                    )
                validate_aks_selection_against_physical_video(
                    selection,
                    actual_frame_count=total_frames,
                    actual_average_fps=source_average_fps,
                )
                chosen_times = list(selection["selected_times"])
                answer_input_source_frame_indices = list(
                    selection["selected_source_frame_indices"]
                )
                append_aks_selector_cost_events(
                    ledger_path=ledger_path,
                    run_id=args.run_id,
                    commit=commit,
                    method=method,
                    dataset_version=args.dataset_version,
                    selection_row=selection,
                    selection_manifest_sha=aks_selection_manifest_sha,
                )
                query_dir = (
                    args.frame_cache
                    / video_id
                    / f"aks_query_{qid}_k{args.num_frames}"
                )
                decode_started = time.perf_counter()
                paths = extract_frames_at_source_indices(
                    video_path,
                    answer_input_source_frame_indices,
                    query_dir,
                    total_frames,
                    max_side=args.frame_max_side,
                )
                decode_latency = time.perf_counter() - decode_started
                if len(paths) != args.num_frames:
                    raise RuntimeError(
                        f"Expected {args.num_frames} AKS frames for {video_id}/{qid}, "
                        f"got {len(paths)}"
                    )
                answer_times = chosen_times
                result_selected_times = chosen_times
                result_selected_times_sha = str(selection["selected_times_sha256"])
                result_observation_policy = (
                    f"aks_query_aware_fixed{args.num_frames}:"
                    f"{str(selection['config_sha256'])[:12]}"
                )
                query_selection_manifest_sha = aks_selection_manifest_sha
                selector_cost_imported_in_full = True
                selector_candidate_count = int(selection["candidate_count"])
                selector_retained_timestamp_bytes = int(
                    selection["retained_timestamp_bytes"]
                )
                selector_cost_attribution = str(
                    selection["selector_cost"]["cost_attribution"]
                )
                physical_score_pass_shared_across_budgets = bool(
                    selection["selector_cost"][
                        "physical_score_pass_shared_across_budgets"
                    ]
                )
                selected_frame_indices = list(range(args.num_frames))
                answer_frame_ids = [
                    f"{video_id}:aks_query:{qid}:{source_index}:{stamp:.6f}"
                    for source_index, stamp in zip(
                        selection["selected_source_frame_indices"], chosen_times
                    )
                ]
                CostLedger(
                    ledger_path,
                    LedgerContext(
                        args.run_id,
                        commit,
                        method,
                        args.dataset_version,
                        video_id,
                        qid,
                        1,
                    ),
                ).append(
                    phase="observe",
                    model_id="opencv_selected_frame_decoder",
                    cache_status="miss",
                    decoded_frames=args.num_frames,
                    unique_frame_ids=answer_frame_ids,
                    latency_s=decode_latency,
                    active_gpu_count=0,
                    selected_times=chosen_times,
                    selected_times_sha256=result_selected_times_sha,
                    answer_input_source_frame_indices=(
                        answer_input_source_frame_indices
                    ),
                    observation_policy=result_observation_policy,
                    query_selection_manifest_sha256=query_selection_manifest_sha,
                    selector_cost_imported_in_full=True,
                    measurement_note=(
                        "Question-time CPU decode of the AKS-selected frames for the raw "
                        "VLM answer. This is separate from the selector's full candidate scan."
                    ),
                )
                content = [
                    {"type": "text", "text": answer_prompt_text},
                    *media_content(
                        paths,
                        selected_frame_indices,
                        chosen_times,
                        args.visual_input_format,
                    ),
                    {
                        "type": "text",
                        "text": f"Question:\n{row['question']}\n\nOptions:\n{options}",
                    },
                ]
                messages = [{"role": "user", "content": content}]
                decoded = 0
                tokens = (
                    visual_tokens(
                        paths, args.visual_model_path, args.visual_max_pixels
                    )
                    if args.visual_token_counter == "qwen"
                    else 0
                )
                cache_status = "miss"
                answer_repeated_frames = args.num_frames
            elif args.mode in {"query_aware_selection", "coarse_vlm_local_selection"}:
                query_dir = args.frame_cache / video_id / f"query_select_{qid}"
                decode_started = time.perf_counter()
                candidate_paths = extract_frames_at_times(
                    str(video_path), times, query_dir, total_frames, duration,
                    max_side=args.frame_max_side,
                )
                if len(candidate_paths) != args.num_frames:
                    raise RuntimeError(
                        f"Expected {args.num_frames} candidate frames for {video_id}, got {len(candidate_paths)}"
                    )
                decode_latency = time.perf_counter() - decode_started
                decode_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                decode_ledger.append(
                    phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                    decoded_frames=args.num_frames, unique_frame_ids=frame_ids,
                    latency_s=decode_latency, active_gpu_count=0, selected_times=times,
                    measurement_note="Question-time CPU decode of the matched candidate frame pool before question-aware selection.",
                )
                selector_content = [
                    {"type": "text", "text": f"{COARSE_SELECTION_PROMPT if args.mode == 'coarse_vlm_local_selection' else SELECTION_PROMPT}\nSelect at most {args.selection_frames} of the {args.num_frames} source frames.\n\nQuestion:\n{row['question']}\n\nOptions:\n{options}"},
                    *media_content(candidate_paths, list(range(args.num_frames)), times, args.visual_input_format),
                ]
                if reference_paths:
                    selector_content.extend(reference_content(reference_paths, reference_times))
                selector_started = time.perf_counter()
                selector_response = client.chat(
                    [{"role": "user", "content": selector_content}],
                    max_tokens=args.max_selector_tokens,
                    temperature=0.0,
                    response_format=(
                        None if args.disable_json_response_format else {"type": "json_object"}
                    ),
                    stop=args.stop_string or None,
                    stop_token_ids=args.stop_token_id or None,
                )
                selector_latency = time.perf_counter() - selector_started
                try:
                    selected_frame_indices = parse_selected_frames(
                        str(selector_response["text"]), args.num_frames, args.selection_frames
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    selection_fallback_reason = str(exc)
                    selected_frame_indices = uniform_fallback_indices(
                        args.num_frames, args.selection_frames
                    )
                if args.mode == "coarse_vlm_local_selection":
                    try:
                        selector_payload = json.loads(str(selector_response["text"]))
                    except json.JSONDecodeError:
                        start = str(selector_response["text"]).find("{")
                        end = str(selector_response["text"]).rfind("}")
                        selector_payload = (
                            json.loads(str(selector_response["text"])[start:end + 1])
                            if start >= 0 and end > start else {}
                        )
                    coarse_scan_notes = str(selector_payload.get("observations") or "").strip()
                selector_usage = selector_response.get("usage") or {}
                selector_visual_tokens = (
                    visual_tokens(candidate_paths + reference_paths, args.visual_model_path, args.visual_max_pixels)
                    if args.visual_token_counter == "qwen" else 0
                )
                selector_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                selector_ledger.append(
                    phase="observe", model_id=args.model, cache_status="miss",
                    input_text_tokens=max(0, int(selector_usage.get("prompt_tokens") or 0) - selector_visual_tokens),
                    output_text_tokens=int(selector_usage.get("completion_tokens") or 0),
                    visual_tokens=selector_visual_tokens, decoded_frames=0,
                    repeated_frames=args.num_frames + len(reference_paths),
                    unique_frame_ids=frame_ids + reference_frame_ids,
                    latency_s=selector_latency, peak_vram_bytes=sampled_visible_gpu_memory_bytes(),
                    selected_times=times + reference_times, key_frames=selected_frame_indices,
                    selection_fallback_reason=selection_fallback_reason,
                    measurement_note=(
                        "Question-aware anchor selection over a low-frequency frame scan before local observation."
                        if args.mode == "coarse_vlm_local_selection"
                        else f"Question-aware evidence-frame selection over the matched "
                        f"{args.num_frames}-frame candidate pool."
                    ),
                )
                selected_anchor_indices = list(selected_frame_indices)
                selected_anchor_times = [times[index] for index in selected_frame_indices]
                if args.mode == "coarse_vlm_local_selection":
                    coarse_spacing = duration / max(1, args.num_frames - 1)
                    answer_times = local_times_around_anchors(
                        selected_anchor_times, duration, coarse_spacing, args.answer_frames
                    )
                    local_dir = args.frame_cache / video_id / f"query_local_{qid}"
                    local_started = time.perf_counter()
                    paths = extract_frames_at_times(
                        str(video_path), answer_times, local_dir, total_frames, duration,
                        max_side=args.frame_max_side,
                    )
                    local_latency = time.perf_counter() - local_started
                    if len(paths) != args.answer_frames:
                        raise RuntimeError(
                            f"Expected {args.answer_frames} local frames for {video_id}/{qid}, got {len(paths)}"
                        )
                    answer_frame_ids = [
                        f"{video_id}:coarse_vlm_local:{index}:{stamp:.6f}"
                        for index, stamp in enumerate(answer_times)
                    ]
                    CostLedger(
                        ledger_path,
                        LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1),
                    ).append(
                        phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                        decoded_frames=len(paths), unique_frame_ids=answer_frame_ids,
                        latency_s=local_latency, active_gpu_count=0, selected_times=answer_times,
                        selected_anchor_indices=selected_anchor_indices,
                        anchor_times=selected_anchor_times,
                        measurement_note="Question-time dense local decode around VLM-selected temporal anchors.",
                    )
                    selected_frame_indices = list(range(args.answer_frames))
                    result_selected_times = answer_times
                else:
                    paths = [candidate_paths[index] for index in selected_frame_indices]
                    answer_frame_ids = [frame_ids[index] for index in selected_frame_indices]
                    answer_times = [times[index] for index in selected_frame_indices]
                image_times = answer_times if args.mode == "coarse_vlm_local_selection" else times
                content = [
                    {"type": "text", "text": answer_prompt_text},
                    *media_content(paths, selected_frame_indices, image_times, args.visual_input_format),
                    {"type": "text", "text": (
                        f"Coarse-scan facts (coarse frame IDs differ from the local frame IDs above):\n"
                        f"{coarse_scan_notes or 'No usable fact was recorded.'}\n\n"
                        f"Question:\n{row['question']}\n\nOptions:\n{options}"
                        if args.mode == "coarse_vlm_local_selection"
                        else f"Question:\n{row['question']}\n\nOptions:\n{options}"
                    )},
                ]
                messages = [{"role": "user", "content": content}]
                decoded = 0
                tokens = visual_tokens(paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                cache_status = "miss"
                answer_repeated_frames = len(paths)
            elif args.mode == "coarse_to_fine_selection":
                selection = coarse_selections[qid]
                chosen_times = [float(value) for value in selection["answer_input_times"]]
                if len(chosen_times) != args.num_frames:
                    raise RuntimeError(f"Expected {args.num_frames} local frames for {qid}, got {len(chosen_times)}")
                query_dir = args.frame_cache / video_id / f"coarse_local_{qid}"
                decode_started = time.perf_counter()
                paths = extract_frames_at_times(
                    str(video_path), chosen_times, query_dir, total_frames, duration,
                    max_side=args.frame_max_side,
                )
                decode_latency = time.perf_counter() - decode_started
                answer_times = chosen_times
                result_selected_times = chosen_times
                answer_frame_ids = [f"{video_id}:coarse_local:{index}:{stamp:.6f}" for index, stamp in enumerate(chosen_times)]
                selected_frame_indices = list(range(args.num_frames))
                decode_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                decode_ledger.append(
                    phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                    decoded_frames=args.num_frames, unique_frame_ids=answer_frame_ids,
                    latency_s=decode_latency, active_gpu_count=0, selected_times=chosen_times,
                    selected_anchor_indices=selection.get("selected_anchor_indices") or [],
                    anchor_times=selection.get("anchor_times") or [],
                    measurement_note="Question-time local dense decode around CLIP-retrieved temporal anchors.",
                )
                content = [
                    {"type": "text", "text": answer_prompt_text},
                    *media_content(paths, list(range(args.num_frames)), chosen_times, args.visual_input_format),
                    {"type": "text", "text": f"Question:\n{row['question']}\n\nOptions:\n{options}"},
                ]
                messages = [{"role": "user", "content": content}]
                decoded = 0
                tokens = visual_tokens(paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                cache_status = "miss"
                answer_repeated_frames = args.num_frames
            else:
                query_dir = args.frame_cache / video_id / f"query_{qid}"
                decode_started = time.perf_counter()
                paths = extract_frames_at_times(
                    str(video_path), times, query_dir, total_frames, duration,
                    max_side=args.frame_max_side,
                )
                decode_latency = time.perf_counter() - decode_started
                decode_ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                decode_ledger.append(
                    phase="observe", model_id="opencv_video_decoder", cache_status="miss",
                    decoded_frames=args.num_frames, unique_frame_ids=frame_ids,
                    latency_s=decode_latency, active_gpu_count=0, selected_times=times,
                    measurement_note="Question-time CPU frame decode before raw VLM observation.",
                )
                content = [{"type": "text", "text": answer_prompt_text}, *media_content(paths, list(range(args.num_frames)), times, args.visual_input_format), {"type": "text", "text": f"Question:\n{row['question']}\n\nOptions:\n{options}"}]
                messages = [{"role": "user", "content": content}]
                decoded = 0
                tokens = visual_tokens(paths, args.visual_model_path, args.visual_max_pixels) if args.visual_token_counter == "qwen" else 0
                cache_status = "miss"
            if reference_paths:
                additions = reference_content(reference_paths, reference_times)
                message_content = messages[0]["content"]
                if isinstance(message_content, str):
                    messages[0]["content"] = [{"type": "text", "text": message_content}, *additions]
                else:
                    message_content.extend(additions)
                if args.visual_token_counter == "qwen":
                    tokens += visual_tokens(
                        reference_paths, args.visual_model_path, args.visual_max_pixels
                    )
                answer_repeated_frames += len(reference_paths)
                answer_frame_ids = list(answer_frame_ids) + reference_frame_ids
            started = time.perf_counter()
            try:
                answer_call = client.chat_stream_timed if args.stream_timing else client.chat
                response = answer_call(
                    messages,
                    max_tokens=args.max_answer_tokens,
                    temperature=0.0,
                    response_format=(
                        None if args.disable_json_response_format else {"type": "json_object"}
                    ),
                    stop=args.stop_string or None,
                    stop_token_ids=args.stop_token_id or None,
                )
                parse_frame_count = (
                    len(answer_times) if args.mode == "coarse_vlm_local_selection" else args.num_frames
                )
                parsed = parse_evidence_response(str(response["text"]), num_frames=parse_frame_count)
                answer_responses = [response]
                valid_label_values = [
                    chr(65 + index) for index in range(len(row["choices"]))
                ]
                valid_labels = ", ".join(valid_label_values)
                for _ in range(args.answer_repair_attempts):
                    if parsed.answer and not parsed.parse_error:
                        break
                    repair_messages = [
                        *messages,
                        {"role": "assistant", "content": str(response["text"])},
                        {
                            "role": "user",
                            "content": ANSWER_REPAIR_PROMPT.format(valid_labels=valid_labels),
                        },
                    ]
                    response = answer_call(
                        repair_messages,
                        max_tokens=args.max_answer_tokens,
                        temperature=0.0,
                        response_format=(
                            minimal_answer_repair_response_format(valid_label_values)
                            if args.answer_repair_json_schema
                            else None
                            if args.disable_json_response_format
                            else {"type": "json_object"}
                        ),
                        stop=args.stop_string or None,
                        stop_token_ids=args.stop_token_id or None,
                    )
                    answer_responses.append(response)
                    parsed = parse_evidence_response(
                        str(response["text"]), num_frames=parse_frame_count
                    )
                latency = time.perf_counter() - started
                error = "" if parsed.answer else str(parsed.parse_error or "missing_answer")
                usage = {
                    key: sum(
                        int((item.get("usage") or {}).get(key) or 0)
                        for item in answer_responses
                    )
                    for key in ("prompt_tokens", "completion_tokens")
                }
                reused_prompt_tokens = sum(
                    cached_prompt_tokens(item.get("usage") or {})
                    for item in answer_responses
                )
                answer_request_count = len(answer_responses)
                billed_visual_tokens = tokens * answer_request_count
                answer_timing = response.get("timing") or {}
                if answer_request_count > 1 and args.stream_timing:
                    timing_rows = [item.get("timing") or {} for item in answer_responses]
                    answer_timing = {
                        "ttft_s": sum(float(item.get("ttft_s") or 0) for item in timing_rows),
                        "e2e_stream_s": sum(
                            float(item.get("e2e_stream_s") or 0) for item in timing_rows
                        ),
                        "tpot_s": None,
                        "stream_request_timings": timing_rows,
                    }
                ledger = CostLedger(ledger_path, LedgerContext(args.run_id, commit, method, args.dataset_version, video_id, qid, 1))
                ledger.append(
                    phase="answer", model_id=args.model, cache_status=cache_status,
                    input_text_tokens=max(0, int(usage.get("prompt_tokens") or 0) - billed_visual_tokens),
                    output_text_tokens=int(usage.get("completion_tokens") or 0), visual_tokens=billed_visual_tokens,
                    decoded_frames=decoded,
                    repeated_frames=answer_repeated_frames * answer_request_count,
                    unique_frame_ids=answer_frame_ids, latency_s=latency,
                    peak_vram_bytes=sampled_visible_gpu_memory_bytes(), error=error,
                    selected_times=answer_times + reference_times,
                    evidence_spans=list(parsed.evidence_spans), key_frames=list(parsed.key_frames),
                    cached_prompt_tokens=reused_prompt_tokens,
                    state_budget_frames=(
                        args.num_frames
                        if args.mode in STATE_CACHE_MODES
                        else 0
                    ),
                    state_storage_unit=(
                        "vllm_visual_encoder_outputs"
                        if args.mode in VISUAL_CACHE_MODES
                        else "vllm_multimodal_prefix_tokens"
                        if args.mode in KV_CACHE_MODES
                        else ""
                    ),
                    measurement_note=(
                        "Joint Text+Visual answer: the model reads the saved frame-level Text record "
                        "and reuses question-blind visual encoder outputs; language-model prefill is "
                        "recomputed."
                        if args.mode == FRAME_TEXT_HYBRID_VISUAL_MODE
                        else "Joint Text+Prefix State answer: the model reads the saved "
                        "frame-level Text record from the reused multimodal prefix and receives "
                        "the same visual evidence."
                        if args.mode == FRAME_TEXT_HYBRID_KV_MODE
                        else
                        "Matched controlled answer reusing question-blind visual encoder outputs; "
                        "language-model prefill is recomputed."
                        if args.mode == "write_once_visual_token_cache"
                        else "Matched controlled answer reusing a question-blind multimodal prefix state."
                        if args.mode == "write_once_model_state_cache"
                        else "Question-time raw-frame answer over the per-question "
                        "AKS-selected visual evidence."
                        if args.mode == AKS_QUERY_TIME_RAW_MODE
                        else "Matched controlled answer with input-mappable source-frame citations."
                    ),
                    answer_request_count=answer_request_count,
                    observation_policy=result_observation_policy,
                    timestamp_manifest_sha256=timestamp_manifest_sha,
                    query_selection_manifest_sha256=query_selection_manifest_sha,
                    selector_cost_imported_in_full=selector_cost_imported_in_full,
                    selector_candidate_count=selector_candidate_count,
                    selector_retained_timestamp_bytes=(
                        selector_retained_timestamp_bytes
                    ),
                    selector_cost_attribution=selector_cost_attribution,
                    physical_score_pass_shared_across_budgets=(
                        physical_score_pass_shared_across_budgets
                    ),
                    **(
                        {
                            "answer_input_source_frame_indices": (
                                answer_input_source_frame_indices
                            )
                        }
                        if args.mode == AKS_QUERY_TIME_RAW_MODE
                        else {}
                    ),
                    **answer_timing,
                )
                append_jsonl(results_path, {
                    "qid": qid, "video_uid": video_id, "within_video_question_order": row["within_video_question_order"],
                    "prediction": parsed.answer, "right_answer": str(row["right_answer"]),
                    "correct": parsed.answer == str(row["right_answer"]), "selected_times": result_selected_times,
                    "answer_input_times": answer_times, "selected_frame_indices": selected_frame_indices,
                    **(
                        {
                            "answer_input_source_frame_indices": (
                                answer_input_source_frame_indices
                            )
                        }
                        if args.mode == AKS_QUERY_TIME_RAW_MODE
                        else {}
                    ),
                    "evidence_spans": list(parsed.evidence_spans), "key_frames": list(parsed.key_frames),
                    "conditioned_notes": conditioned_notes,
                    "blind_replay_repository": blind_replay_repository,
                    "cached_prompt_tokens": reused_prompt_tokens,
                    "state_budget_frames": (
                        args.num_frames
                        if args.mode in STATE_CACHE_MODES
                        else 0
                    ),
                    "state_storage_unit": (
                        "vllm_visual_encoder_outputs"
                        if args.mode in VISUAL_CACHE_MODES
                        else "vllm_multimodal_prefix_tokens"
                        if args.mode in KV_CACHE_MODES
                        else ""
                    ),
                    "representation_input": (
                        "text+visual"
                        if args.mode == FRAME_TEXT_HYBRID_VISUAL_MODE
                        else "text+prefix_state"
                        if args.mode == FRAME_TEXT_HYBRID_KV_MODE
                        else args.mode
                    ),
                    "observation_policy": result_observation_policy,
                    "selected_times_sha256": result_selected_times_sha,
                    "timestamp_manifest_sha256": timestamp_manifest_sha,
                    "query_selection_manifest_sha256": query_selection_manifest_sha,
                    "selector_cost_imported_in_full": selector_cost_imported_in_full,
                    "selector_candidate_count": selector_candidate_count,
                    "selector_retained_timestamp_bytes": (
                        selector_retained_timestamp_bytes
                    ),
                    "selector_cost_attribution": selector_cost_attribution,
                    "physical_score_pass_shared_across_budgets": (
                        physical_score_pass_shared_across_budgets
                    ),
                    "selection_fallback_reason": selection_fallback_reason,
                    "selected_anchor_indices": selected_anchor_indices,
                    "selected_anchor_times": selected_anchor_times,
                    "official_reference_times": reference_times,
                    "official_reference_frame_ids": reference_frame_ids,
                    "official_reference_paths": [
                        str(Path(path).resolve()) for path in reference_paths
                    ],
                    "frame_time_metadata_attached": (
                        args.mode in FRAME_TEXT_WRITE_ONCE_MODES
                    ),
                    "coarse_scan_notes": coarse_scan_notes,
                    "structured_evidence_parse_error": str(parsed.parse_error or ""),
                    "raw_response": response["text"],
                    "answer_attempt_responses": [item["text"] for item in answer_responses],
                    "answer_request_count": answer_request_count,
                    "answer_response_format": (
                        "none" if args.disable_json_response_format else "json_object"
                    ),
                    "answer_repair_response_format": (
                        "minimal_json_schema_v1"
                        if args.answer_repair_json_schema
                        else "none"
                        if args.disable_json_response_format
                        else "json_object"
                    ),
                    "frame_max_side": args.frame_max_side,
                    "error": error,
                })
                failures += int(bool(error))
            except Exception as exc:
                failures += 1
                append_jsonl(results_path, {
                    "qid": qid,
                    "video_uid": video_id,
                    "correct": False,
                    "frame_max_side": args.frame_max_side,
                    "observation_policy": result_observation_policy,
                    "selected_times": result_selected_times,
                    "answer_input_times": answer_times,
                    "selected_frame_indices": selected_frame_indices,
                    **(
                        {
                            "answer_input_source_frame_indices": (
                                answer_input_source_frame_indices
                            )
                        }
                        if args.mode == AKS_QUERY_TIME_RAW_MODE
                        else {}
                    ),
                    "selected_times_sha256": result_selected_times_sha,
                    "query_selection_manifest_sha256": query_selection_manifest_sha,
                    "selector_cost_imported_in_full": selector_cost_imported_in_full,
                    "selector_candidate_count": selector_candidate_count,
                    "selector_retained_timestamp_bytes": (
                        selector_retained_timestamp_bytes
                    ),
                    "selector_cost_attribution": selector_cost_attribution,
                    "physical_score_pass_shared_across_budgets": (
                        physical_score_pass_shared_across_budgets
                    ),
                    "error": repr(exc),
                })
    if args.mode in FRAME_TEXT_HYBRID_MODES:
        completed = read_jsonl(results_path)
        expected_input = (
            "text+visual"
            if args.mode == FRAME_TEXT_HYBRID_VISUAL_MODE
            else "text+prefix_state"
        )
        bad = [
            str(row.get("qid") or "")
            for row in completed
            if row.get("representation_input") != expected_input
            or int(row.get("state_budget_frames") or 0) != args.num_frames
            or row.get("frame_time_metadata_attached") is not True
            or str(row.get("observation_policy") or "") == ""
        ]
        if bad:
            raise RuntimeError(
                f"joint-input acceptance failed for {len(bad)} results: {bad[:10]}"
            )
        metadata_by_video = {
            str(row["video_uid"]): row
            for row in read_jsonl(args.out_dir / "frame_time_metadata.jsonl")
        }
        if set(metadata_by_video) != set(grouped):
            raise RuntimeError(
                "joint-input repository coverage mismatch: "
                f"expected={sorted(grouped)} observed={sorted(metadata_by_video)}"
            )
        (args.out_dir / "joint_input_acceptance.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "representation_input": expected_input,
                    "questions": len(completed),
                    "videos": len(metadata_by_video),
                    "frames_per_video": args.num_frames,
                    "question_blind_text": True,
                    "question_blind_visual_state": True,
                    "ensemble_or_vote": False,
                    "canonical_text_reused_byte_for_byte": (
                        args.prebuilt_repository_dir is not None
                    ),
                    "canonical_text_writer_cost_imported": (
                        args.prebuilt_repository_dir is not None
                    ),
                    "canonical_text_source_directory": (
                        str(args.prebuilt_repository_dir.resolve())
                        if args.prebuilt_repository_dir is not None
                        else ""
                    ),
                    "visual_backing_is_part_of_prefix_state": (
                        args.mode == FRAME_TEXT_HYBRID_KV_MODE
                    ),
                    "answer_repository_sha256_by_video": {
                        video: str(metadata_by_video[video]["answer_repository_sha256"])
                        for video in sorted(metadata_by_video)
                    },
                    "timestamp_manifest_sha256": timestamp_manifest_sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if failures:
        # Collection and acceptance are deliberately separate. Preserve every
        # raw model response and finish any cache-specific technical checks;
        # compose_controlled_condition_cpu.sh later rejects errors by default,
        # or accepts only explicitly audited model-output format failures.
        print(
            json.dumps({
                "warning": "controlled run contains result errors",
                "result_error_count": failures,
                "allow_result_errors_requested": args.allow_result_errors,
            }),
            file=sys.stderr,
        )
    if (
        args.mode in FRAME_TEXT_WRITE_ONCE_MODES
        and args.prebuilt_repository_dir is None
    ):
        writer_calls_per_video = (args.num_frames + args.writer_chunk_size - 1) // args.writer_chunk_size
        (args.out_dir / "frame_text_schema_acceptance.json").write_text(
            json.dumps({
                "accepted": True,
                "videos": len(grouped),
                "source_frames_per_video": args.num_frames,
                "writer_chunk_size": args.writer_chunk_size,
                "writer_calls_per_video": writer_calls_per_video,
                "validated_entries_per_video": args.num_frames,
                "total_validated_entries": len(grouped) * args.num_frames,
                "frame_writer_tokens": args.frame_writer_tokens,
                "frame_max_side": args.frame_max_side,
                "frame_text_profile": args.frame_text_profile,
                "prompt_sha256": hashlib.sha256(
                    (
                        frame_level_context_write_prompt
                        if args.mode == "write_once_frame_text_masked"
                        else frame_level_write_prompt
                    ).encode()
                ).hexdigest(),
                "question_context": "masked" if args.mode == "write_once_frame_text_masked" else "hidden",
                "joint_input": (
                    "text+visual"
                    if args.mode == FRAME_TEXT_HYBRID_VISUAL_MODE
                    else "text+prefix_state"
                    if args.mode == FRAME_TEXT_HYBRID_KV_MODE
                    else ""
                ),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.mode in FRAME_TEXT_WRITE_ONCE_MODES:
        metadata_rows = read_jsonl(args.out_dir / "frame_time_metadata.jsonl")
        expected_videos = sorted(grouped)
        observed_videos = sorted(str(row["video_uid"]) for row in metadata_rows)
        if observed_videos != expected_videos:
            raise RuntimeError(
                "frame-time metadata coverage mismatch: "
                f"expected={expected_videos} observed={observed_videos}"
            )
        if not all(
            row.get("frame_time_metadata_attached") is True
            and int(row.get("source_frame_count") or 0) == args.num_frames
            and int(row.get("time_metadata_bytes") or 0) > 0
            for row in metadata_rows
        ):
            raise RuntimeError(f"invalid frame-time metadata records: {metadata_rows[:3]}")
        (args.out_dir / "frame_time_metadata_acceptance.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "videos": len(metadata_rows),
                    "source_frames_per_video": args.num_frames,
                    "metadata_source": (
                        "question_blind_timestamp_manifest"
                        if args.timestamp_manifest is not None
                        else "deterministic_uniform_sampler_timestamps"
                    ),
                    "observation_policy": (
                        sorted({str(row["selector_id"]) for row in timestamp_rows.values()})
                        if args.timestamp_manifest is not None
                        else ["uniform"]
                    ),
                    "timestamp_manifest_sha256": timestamp_manifest_sha,
                    "writer_rerun_required": False,
                    "total_semantic_repository_bytes": sum(
                        int(row["semantic_repository_bytes"]) for row in metadata_rows
                    ),
                    "total_answer_repository_bytes": sum(
                        int(row["answer_repository_bytes"]) for row in metadata_rows
                    ),
                    "total_time_metadata_bytes": sum(
                        int(row["time_metadata_bytes"]) for row in metadata_rows
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.mode in {"write_once_frame_text", *FRAME_TEXT_HYBRID_MODES} and args.prebuilt_repository_dir is not None:
        provenance_rows = read_jsonl(args.out_dir / "prebuilt_repository_provenance.jsonl")
        expected_videos = sorted(grouped)
        observed_videos = sorted(str(row["video_uid"]) for row in provenance_rows)
        if observed_videos != expected_videos:
            raise RuntimeError(
                "prebuilt repository provenance mismatch: "
                f"expected={expected_videos} observed={observed_videos}"
            )
        (args.out_dir / "prebuilt_repository_acceptance.json").write_text(
            json.dumps({
                "accepted": True,
                "source_directory": str(args.prebuilt_repository_dir.resolve()),
                "videos": len(provenance_rows),
                "source_frames_per_video": args.num_frames,
                "answer_only_rerun": args.mode == "write_once_frame_text",
                "canonical_text_writer_cost_imported": args.mode in FRAME_TEXT_HYBRID_MODES,
                "frame_time_metadata_attached": True,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.mode in {
        "blind_replay_frame_text",
        "query_conditioned_frame_text",
        "annotated_evidence_frame_text",
        "annotated_evidence_query_conditioned_frame_text",
    }:
        completed = read_jsonl(results_path)
        for result in completed:
            validate_frame_level_entries(
                str(result.get("conditioned_notes") or ""),
                list(range(args.num_frames)),
            )
        (args.out_dir / "frame_text_schema_acceptance.json").write_text(
            json.dumps({
                "accepted": True,
                "questions": len(completed),
                "source_frames_per_question": args.num_frames,
                "writer_chunk_size": args.writer_chunk_size,
                "validated_entries_per_question": args.num_frames,
                "total_validated_entries": len(completed) * args.num_frames,
                "frame_writer_tokens": args.frame_writer_tokens,
                "frame_text_profile": args.frame_text_profile,
                "question_aware": args.mode in {
                    "query_conditioned_frame_text",
                    "annotated_evidence_query_conditioned_frame_text",
                },
                "per_question": True,
                "question_context": (
                    "hidden"
                    if args.mode == "annotated_evidence_frame_text"
                    else "real"
                    if args.mode in {
                        "query_conditioned_frame_text",
                        "annotated_evidence_query_conditioned_frame_text",
                    }
                    else "masked"
                ),
                "prompt_sha256": hashlib.sha256(
                    (
                        frame_level_write_prompt
                        if args.mode == "annotated_evidence_frame_text"
                        else frame_level_context_write_prompt
                    ).encode()
                ).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
