"""Deterministic adaptive keyframe selection from query--frame scores.

This is an independent reimplementation of the recursive selection rule
described in *Adaptive Keyframe Sampling for Long Video Understanding*
(CVPR 2025). It does not contain source copied from the authors' released
repository. The compatibility notes below describe observed public behavior;
they should not be read as a claim of a formally isolated clean-room process.

At each node, the selector compares the mean of up to the *global* formal
budget's best scores with the mean of all scores in that node.  This detail is
deliberately compatible with the released ``meanstd`` behavior: the contrast
calculation does not shrink its top-k value as recursion gets deeper.  A large
gap means that the useful frames are concentrated, so the node keeps its local
quota of highest-scoring frames.  Otherwise, the node is split in time and its
quota is divided between the two children.  This trades query relevance for
temporal coverage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np


SUPPORTED_BUDGETS = frozenset({32, 64, 128})
DEFAULT_MAX_DEPTH = 5
DEFAULT_THRESHOLD = 0.8
DEFAULT_RATIO = 1


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _scores(values: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a one-dimensional numeric sequence") from exc
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not len(scores):
        raise ValueError("scores must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    return scores


def _stable_top_positions(
    scores: np.ndarray,
    source_indices: np.ndarray,
    count: int,
) -> np.ndarray:
    """Rank by descending score and then by the earliest source index."""

    if count < 1 or count > len(scores):
        raise ValueError(
            f"cannot select {count} rows from a segment containing {len(scores)} candidates"
        )
    order = np.lexsort((source_indices, -scores))
    return np.asarray(order[:count], dtype=np.int64)


def select_aks_indices(
    scores: Sequence[float] | np.ndarray,
    *,
    budget: int,
    max_depth: int = DEFAULT_MAX_DEPTH,
    threshold: float = DEFAULT_THRESHOLD,
    ratio: int = DEFAULT_RATIO,
) -> tuple[list[int], dict[str, Any]]:
    """Select a fixed number of source indices with the AKS recursion.

    Parameters
    ----------
    scores:
        One finite query--frame relevance score for every source candidate.
    budget:
        Formal REEL frame budget. It must be 32, 64, or 128.
    max_depth:
        Maximum number of temporal bisections.  The released AKS default is 5.
    threshold:
        Stop splitting when ``top_mean - segment_mean`` is strictly greater
        than this value.  ``top_mean`` always uses up to ``budget`` scores,
        matching the released implementation, while terminal selection uses
        the node's smaller quota.  Scores are min--max normalized first.
    ratio:
        Keep every ``ratio``-th input candidate before selection.  The formal
        protocol uses 1.  Returned values always index the original ``scores``.

    Returns
    -------
    selected, diagnostics:
        ``selected`` contains exactly ``budget`` increasing, unique source
        indices.  ``diagnostics`` is JSON-serializable apart from ordinary
        integer dictionary keys and records every terminal recursion node.

    Notes
    -----
    Constant scores are normalized to zeros.  They therefore recurse to the
    coverage limit and deterministically select the earliest tied candidates
    within each terminal time bin.
    """

    formal_budget = _positive_integer(budget, name="budget")
    if formal_budget not in SUPPORTED_BUDGETS:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_BUDGETS))
        raise ValueError(f"budget must be one of: {allowed}")
    depth_limit = _positive_integer(max_depth, name="max_depth")
    stride = _positive_integer(ratio, name="ratio")
    try:
        stop_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite non-negative number") from exc
    if not np.isfinite(stop_threshold) or stop_threshold < 0:
        raise ValueError("threshold must be a finite non-negative number")

    raw_scores = _scores(scores)
    candidate_indices = np.arange(0, len(raw_scores), stride, dtype=np.int64)
    candidate_scores = raw_scores[candidate_indices]
    if len(candidate_scores) < formal_budget:
        raise ValueError(
            "selection requires at least "
            f"{formal_budget} candidates after ratio={stride}, got {len(candidate_scores)}"
        )

    score_min = float(candidate_scores.min())
    score_max = float(candidate_scores.max())
    score_range = score_max - score_min
    constant_scores = score_range == 0.0
    if constant_scores:
        normalized = np.zeros_like(candidate_scores, dtype=np.float64)
    else:
        normalized = (candidate_scores - score_min) / score_range

    selected: list[int] = []
    terminal_segments: list[dict[str, Any]] = []
    split_count = 0

    def visit(start: int, stop: int, node_budget: int, depth: int) -> None:
        nonlocal split_count

        segment_scores = normalized[start:stop]
        segment_indices = candidate_indices[start:stop]
        if len(segment_scores) < node_budget:
            raise RuntimeError(
                "AKS recursion assigned more selections than candidates to a time segment"
            )
        # Released AKS keeps ``n`` fixed at the global frame budget while it
        # recursively evaluates smaller segments.  If a segment has fewer than
        # ``n`` candidates, all of them contribute and the contrast is zero.
        contrast_top_count = min(formal_budget, len(segment_scores))
        contrast_top_positions = _stable_top_positions(
            segment_scores,
            segment_indices,
            contrast_top_count,
        )
        segment_mean = float(segment_scores.mean())
        top_mean = float(segment_scores[contrast_top_positions].mean())
        contrast = top_mean - segment_mean

        if contrast > stop_threshold:
            stop_reason = "relevance_concentrated"
        elif depth >= depth_limit:
            stop_reason = "max_depth"
        else:
            midpoint = start + (stop - start) // 2
            left_budget = node_budget // 2
            right_budget = node_budget - left_budget
            if (
                left_budget > 0
                and midpoint - start >= left_budget
                and stop - midpoint >= right_budget
            ):
                split_count += 1
                visit(start, midpoint, left_budget, depth + 1)
                visit(midpoint, stop, right_budget, depth + 1)
                return
            # With a power-of-two formal budget and at least ``budget`` input
            # candidates this guard is unreachable.  Keeping it explicit makes
            # exact-budget behavior safe if the recursion is later generalized.
            stop_reason = "capacity_guard"

        selection_positions = _stable_top_positions(
            segment_scores,
            segment_indices,
            node_budget,
        )
        chosen = sorted(
            int(segment_indices[position]) for position in selection_positions
        )
        selected.extend(chosen)
        terminal_segments.append(
            {
                "depth": int(depth),
                "candidate_position_start": int(start),
                "candidate_position_stop": int(stop),
                "source_index_start": int(segment_indices[0]),
                "source_index_end": int(segment_indices[-1]),
                "candidate_count": int(len(segment_scores)),
                "allocated_budget": int(node_budget),
                "contrast_top_count": int(contrast_top_count),
                "segment_mean": segment_mean,
                "top_mean": top_mean,
                "top_minus_segment_mean": contrast,
                "stop_reason": stop_reason,
                "selected_indices": chosen,
            }
        )

    visit(0, len(candidate_scores), formal_budget, 0)
    selected.sort()
    if len(selected) != formal_budget or len(set(selected)) != formal_budget:
        raise RuntimeError(
            "AKS violated its exact-budget invariant: "
            f"selected={len(selected)} unique={len(set(selected))} budget={formal_budget}"
        )
    if selected[0] < 0 or selected[-1] >= len(raw_scores):
        raise RuntimeError("AKS produced an out-of-range source index")

    stop_reason_counts = Counter(
        str(segment["stop_reason"]) for segment in terminal_segments
    )
    depth_counts = Counter(int(segment["depth"]) for segment in terminal_segments)
    diagnostics: dict[str, Any] = {
        "algorithm": "aks_recursive_relevance_coverage",
        "input_count": int(len(raw_scores)),
        "candidate_count": int(len(candidate_scores)),
        "budget": formal_budget,
        "ratio": stride,
        "max_depth": depth_limit,
        "threshold": stop_threshold,
        "raw_score_min": score_min,
        "raw_score_max": score_max,
        "raw_score_range": score_range,
        "constant_scores": constant_scores,
        "normalized_score_min": float(normalized.min()),
        "normalized_score_max": float(normalized.max()),
        "split_count": int(split_count),
        "terminal_count": int(len(terminal_segments)),
        "max_reached_depth": int(max(depth_counts, default=0)),
        "terminal_depth_counts": dict(sorted(depth_counts.items())),
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "selected_count": int(len(selected)),
        "selected_indices": list(selected),
        "terminal_segments": terminal_segments,
    }
    return selected, diagnostics


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_RATIO",
    "DEFAULT_THRESHOLD",
    "SUPPORTED_BUDGETS",
    "select_aks_indices",
]
