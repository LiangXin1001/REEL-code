#!/usr/bin/env python3
"""Shared answer-validity rules for lifecycle aggregation and pairwise analysis."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Mapping


ANSWER_FORMAT_ERRORS = frozenset(
    {"missing_answer", "invalid_answer", "json_parse_failed"}
)


def _raw_json_answer(result_row: Mapping[str, Any]) -> tuple[str, str]:
    """Return the exact raw JSON answer and a failure label.

    Older ledgers may contain a normalized single-letter ``prediction`` even
    when the model returned ``A|B|...``.  When a raw response is available, it
    is therefore the authority for answer-format validity.
    """
    if "raw_response" not in result_row:
        return "", ""
    text = str(result_row.get("raw_response") or "").strip()
    if not text:
        return "", "missing_answer"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return "", "json_parse_failed"
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return "", "json_parse_failed"
    if not isinstance(parsed, dict):
        return "", "json_parse_failed"
    answer = str(parsed.get("answer") or "").strip().upper()
    return (answer, "") if answer else ("", "missing_answer")


def _correct_of(qid: str, row: Mapping[str, Any]) -> bool:
    value = row.get("correct")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"result qid={qid} has no boolean correctness value")


def valid_choice_labels(qid: str, manifest_row: Mapping[str, Any]) -> set[str]:
    choices = manifest_row.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"manifest has no answer choices: qid={qid}")
    if len(choices) > 26:
        raise ValueError(f"manifest has more than 26 answer choices: qid={qid}")
    return {chr(ord("A") + index) for index in range(len(choices))}


def answer_format_failure(
    qid: str,
    result_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> str:
    """Return a format-failure label; raise instead of hiding technical errors."""
    correct = _correct_of(qid, result_row)
    error = str(result_row.get("error") or "").strip()
    if error and error not in ANSWER_FORMAT_ERRORS:
        raise ValueError(
            f"technical error cannot be excluded: qid={qid} error={error}"
        )
    if error and correct:
        raise ValueError(
            f"answer-format failure is marked correct: qid={qid} error={error}"
        )

    # This field is specifically the structured-answer parser status.  Some
    # runners salvage a choice from malformed JSON, so a non-empty value must
    # still be excluded even when ``prediction`` looks usable.
    structured_error = str(
        result_row.get("structured_evidence_parse_error") or ""
    ).strip()

    # Validate every field before returning.  This prevents an allowed answer
    # error from masking a second, technical error in the same row.
    if error:
        return error
    if structured_error:
        return "structured_evidence_parse_error"

    prediction = str(result_row.get("prediction") or "").strip().upper()
    if not prediction:
        return "empty_prediction"
    valid_labels = valid_choice_labels(qid, manifest_row)
    if prediction not in valid_labels:
        return "invalid_answer"
    raw_answer, raw_failure = _raw_json_answer(result_row)
    if raw_failure:
        return raw_failure
    if "raw_response" in result_row and (
        raw_answer not in valid_labels or raw_answer != prediction
    ):
        return "invalid_answer"
    return ""


def question_outcome(
    qid: str,
    result_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    grounded_correct: bool | None = None,
) -> dict[str, Any]:
    failure = answer_format_failure(qid, result_row, manifest_row)
    outcome: dict[str, Any] = {
        "qid": qid,
        "valid_answer": not failure,
        "answer_format_failure": failure,
        "correct": _correct_of(qid, result_row),
        "grounded_correct": grounded_correct,
    }
    return outcome


def index_outcomes(
    outcomes: Iterable[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        qid = str(outcome.get("qid") or "").strip()
        if not qid:
            raise ValueError(f"{label} has a question outcome without qid")
        if qid in indexed:
            raise ValueError(f"{label} has duplicate question outcome qid={qid}")
        valid = outcome.get("valid_answer")
        if not isinstance(valid, bool):
            raise ValueError(f"{label} qid={qid} lacks boolean valid_answer")
        failure = str(outcome.get("answer_format_failure") or "")
        if valid == bool(failure):
            raise ValueError(
                f"{label} qid={qid} has inconsistent validity/failure fields"
            )
        if not isinstance(outcome.get("correct"), bool):
            raise ValueError(f"{label} qid={qid} lacks boolean correct")
        indexed[qid] = outcome
    return indexed


def failure_type_counts(outcomes: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(outcome.get("answer_format_failure") or "")
        for outcome in outcomes
        if not bool(outcome.get("valid_answer"))
    )
    counts.pop("", None)
    return dict(sorted(counts.items()))
