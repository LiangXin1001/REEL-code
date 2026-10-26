from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


FRAME_RE = re.compile(r"\b[Ff]rame\s*[:#-]?\s*(-?\d+)\b")
FRAME_VALUE_RE = re.compile(r"-?\d+")
ANSWER_JSON_RE = re.compile(r'"answer"\s*:\s*"([A-H])"', re.IGNORECASE)
ANSWER_TEXT_RE = re.compile(r"\b([A-H])\b")
VALID_ANSWER_LABELS = set("ABCDEFGH")
VALID_EVIDENCE_ROLES = {"before", "transition", "after", "direct_answer"}


@dataclass(frozen=True)
class EvidenceResponse:
    answer: str = ""
    confidence: float | None = None
    evidence_sufficiency: str = "unknown"
    need_more_evidence: bool | None = None
    evidence_spans: tuple[dict[str, Any], ...] = ()
    invalid_evidence_spans: tuple[dict[str, Any], ...] = ()
    key_frames: tuple[int, ...] = ()
    invalid_key_frames: tuple[dict[str, Any], ...] = ()
    missing_evidence: tuple[str, ...] = ()
    relevant_moments: tuple[dict[str, Any], ...] = ()
    rationale: str = ""
    parse_error: str = ""


def parse_jsonish(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def parse_frame_number(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = FRAME_VALUE_RE.search(str(value))
    return int(match.group(0)) if match else None


def _append_invalid(
    invalid_items: list[dict[str, Any]],
    item: dict[str, Any],
    reason: str,
    index: int,
) -> None:
    invalid_items.append(
        {
            "index": index,
            "reason": reason,
            "raw": item,
        }
    )


def normalize_evidence_spans(
    spans,
    num_frames: int | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    if not isinstance(spans, list):
        return (), (), (), ()

    valid_spans: list[dict[str, Any]] = []
    invalid_spans: list[dict[str, Any]] = []
    all_key_frames: list[int] = []
    invalid_key_frames: list[dict[str, Any]] = []

    for span_index, span in enumerate(spans):
        if not isinstance(span, dict):
            _append_invalid(invalid_spans, {"value": span}, "span_not_object", span_index)
            continue

        role = str(span.get("role") or "").strip().lower()
        if role not in VALID_EVIDENCE_ROLES:
            role = "unknown"

        start_frame = parse_frame_number(span.get("start_frame"))
        end_frame = parse_frame_number(span.get("end_frame"))
        if start_frame is None or end_frame is None:
            _append_invalid(invalid_spans, span, "missing_start_or_end", span_index)
            continue
        if start_frame > end_frame:
            _append_invalid(invalid_spans, span, "start_after_end", span_index)
            continue
        if num_frames is not None and (
            start_frame < 0 or end_frame < 0 or start_frame >= num_frames or end_frame >= num_frames
        ):
            _append_invalid(invalid_spans, span, "frame_oob", span_index)
            continue

        raw_key_frames = span.get("key_frames")
        if isinstance(raw_key_frames, list):
            key_values = raw_key_frames
        elif raw_key_frames is None:
            key_values = []
            invalid_key_frames.append(
                {
                    "span_index": span_index,
                    "reason": "missing_key_frames",
                    "raw": span,
                }
            )
        else:
            key_values = [raw_key_frames]

        key_frames: list[int] = []
        for key_index, raw_key in enumerate(key_values):
            frame_no = parse_frame_number(raw_key)
            if frame_no is None:
                invalid_key_frames.append(
                    {
                        "span_index": span_index,
                        "key_index": key_index,
                        "reason": "invalid_key_frame",
                        "raw": raw_key,
                    }
                )
                continue
            if num_frames is not None and (frame_no < 0 or frame_no >= num_frames):
                invalid_key_frames.append(
                    {
                        "span_index": span_index,
                        "key_index": key_index,
                        "reason": "key_frame_oob",
                        "raw": raw_key,
                    }
                )
                continue
            if frame_no < start_frame or frame_no > end_frame:
                invalid_key_frames.append(
                    {
                        "span_index": span_index,
                        "key_index": key_index,
                        "reason": "key_frame_outside_span",
                        "raw": raw_key,
                    }
                )
                continue
            key_frames.append(frame_no)

        key_frames = sorted(set(key_frames))
        all_key_frames.extend(key_frames)
        valid_spans.append(
            {
                "role": role,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "key_frames": key_frames,
                "evidence": str(span.get("evidence") or "").strip(),
            }
        )

    return (
        tuple(valid_spans),
        tuple(invalid_spans),
        tuple(sorted(set(all_key_frames))),
        tuple(invalid_key_frames),
    )


def extract_cited_frame_numbers(raw_response: str) -> list[int]:
    frames: list[int] = []
    parsed = parse_jsonish(raw_response or "")
    spans = parsed.get("evidence_spans")
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            span_added = False
            key_frames = span.get("key_frames")
            if isinstance(key_frames, list):
                for value in key_frames:
                    frame_no = parse_frame_number(value)
                    if frame_no is not None:
                        frames.append(frame_no)
                        span_added = True
            elif key_frames is not None:
                frame_no = parse_frame_number(key_frames)
                if frame_no is not None:
                    frames.append(frame_no)
                    span_added = True
            if not span_added:
                for field in ("start_frame", "end_frame"):
                    frame_no = parse_frame_number(span.get(field))
                    if frame_no is not None:
                        frames.append(frame_no)
    moments = parsed.get("relevant_moments")
    if isinstance(moments, list):
        for moment in moments:
            if not isinstance(moment, dict):
                continue
            value = moment.get("frame")
            frame_no = parse_frame_number(value)
            if frame_no is not None:
                frames.append(frame_no)
    for match in FRAME_RE.finditer(raw_response or ""):
        frames.append(int(match.group(1)))
    return sorted(set(frames))


def parse_answer(text: str) -> str:
    if not text:
        return ""
    json_match = ANSWER_JSON_RE.search(text)
    if json_match:
        return json_match.group(1).upper()
    parsed = parse_jsonish(text)
    if parsed:
        answer = str(parsed.get("answer") or "").strip().upper()
        return answer if answer in VALID_ANSWER_LABELS else ""
    text_match = ANSWER_TEXT_RE.search(text)
    return text_match.group(1).upper() if text_match else ""


def parse_evidence_response(text: str, num_frames: int | None = None) -> EvidenceResponse:
    parsed = parse_jsonish(text or "")
    if not parsed:
        answer = parse_answer(text)
        return EvidenceResponse(answer=answer, parse_error="json_parse_failed")

    answer = str(parsed.get("answer") or "").strip().upper()
    answer_error = "" if answer in VALID_ANSWER_LABELS else "invalid_answer"
    if answer_error:
        answer = ""

    confidence = None
    raw_confidence = parsed.get("confidence")
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = None

    sufficiency = str(parsed.get("evidence_sufficiency") or "unknown").strip().lower()
    raw_need_more = parsed.get("need_more_evidence")
    if isinstance(raw_need_more, bool):
        need_more = raw_need_more
    elif raw_need_more is None:
        need_more = None
    else:
        need_more = str(raw_need_more).strip().lower() in {"true", "yes", "1"}

    moments = parsed.get("relevant_moments")
    if isinstance(moments, list):
        relevant_moments = tuple(item for item in moments if isinstance(item, dict))
    else:
        relevant_moments = ()
    evidence_spans, invalid_spans, key_frames, invalid_key_frames = normalize_evidence_spans(
        parsed.get("evidence_spans"),
        num_frames=num_frames,
    )
    missing = parsed.get("missing_evidence")
    if isinstance(missing, list):
        missing_evidence = tuple(str(item).strip().lower() for item in missing if str(item).strip())
    else:
        missing_evidence = ()

    return EvidenceResponse(
        answer=answer,
        confidence=confidence,
        evidence_sufficiency=sufficiency,
        need_more_evidence=need_more,
        evidence_spans=evidence_spans,
        invalid_evidence_spans=invalid_spans,
        key_frames=key_frames,
        invalid_key_frames=invalid_key_frames,
        missing_evidence=missing_evidence,
        relevant_moments=relevant_moments,
        rationale=str(parsed.get("rationale") or ""),
        parse_error=answer_error,
    )


def normalized_evidence_sufficiency(row: dict) -> str:
    value = row.get("evidence_sufficiency")
    if value is None:
        parsed = parse_jsonish(row.get("raw_response") or "")
        value = parsed.get("evidence_sufficiency")
    return str(value or "unknown").strip().lower()
