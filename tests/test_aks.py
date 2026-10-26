from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pestate.aks import select_aks_indices  # noqa: E402


@pytest.mark.parametrize("budget", [32, 64, 128])
def test_supported_budgets_return_exact_ordered_unique_indices(budget: int) -> None:
    scores = np.sin(np.arange(320, dtype=np.float64) / 13.0)

    selected, diagnostics = select_aks_indices(scores, budget=budget)

    assert selected == sorted(selected)
    assert len(selected) == len(set(selected)) == budget
    assert diagnostics["selected_indices"] == selected
    assert diagnostics["selected_count"] == budget
    assert diagnostics["budget"] == budget


def test_constant_scores_use_deterministic_temporal_coverage() -> None:
    scores = np.ones(320, dtype=np.float64)

    first, diagnostics = select_aks_indices(scores, budget=32)
    second, _ = select_aks_indices(scores, budget=32)

    assert first == second
    assert diagnostics["constant_scores"] is True
    assert diagnostics["split_count"] == 31
    assert diagnostics["terminal_count"] == 32
    assert diagnostics["terminal_depth_counts"] == {5: 32}
    assert diagnostics["stop_reason_counts"] == {"max_depth": 32}
    # Every depth-5 temporal bin contributes one frame, instead of all ties
    # collapsing to the start of the video.
    assert first == [10 * index for index in range(32)]


def test_candidate_count_equal_to_budget_selects_every_candidate() -> None:
    scores = np.linspace(1.0, -1.0, 32)

    selected, diagnostics = select_aks_indices(scores, budget=32)

    assert selected == list(range(32))
    assert diagnostics["candidate_count"] == 32


def test_short_candidate_pool_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least 64 candidates"):
        select_aks_indices(np.zeros(63), budget=64)

    with pytest.raises(ValueError, match="after ratio=2, got 64"):
        select_aks_indices(np.zeros(128), budget=128, ratio=2)


def test_ties_choose_earliest_source_indices() -> None:
    scores = np.zeros(320, dtype=np.float64)
    scores[:40] = 1.0

    selected, diagnostics = select_aks_indices(scores, budget=32)

    # The normalized top-32 mean is 1 and the global mean is 40/320, so the
    # root stops.  Stable tie breaking must take the first 32 frames.
    assert selected == list(range(32))
    assert diagnostics["split_count"] == 0
    assert diagnostics["stop_reason_counts"] == {"relevance_concentrated": 1}


def test_recursion_splits_flat_region_but_stops_on_concentrated_region() -> None:
    scores = np.zeros(512, dtype=np.float64)
    # The right half is uniformly high, keeping the root contrast below 0.8.
    # The left half has 32 isolated high scores, so its global-top-32 contrast
    # exceeds 0.8 and it stops while the right half continues for coverage.
    scores[:32] = 1.0
    scores[256:] = 1.0

    selected, diagnostics = select_aks_indices(scores, budget=32)

    assert len(selected) == len(set(selected)) == 32
    assert diagnostics["split_count"] > 1
    assert diagnostics["stop_reason_counts"]["relevance_concentrated"] == 1
    assert diagnostics["stop_reason_counts"]["max_depth"] == 16
    concentrated = [
        segment
        for segment in diagnostics["terminal_segments"]
        if segment["stop_reason"] == "relevance_concentrated"
    ]
    assert len(concentrated) == 1
    assert concentrated[0]["source_index_start"] == 0
    assert concentrated[0]["source_index_end"] == 255
    assert concentrated[0]["allocated_budget"] == 16
    assert concentrated[0]["contrast_top_count"] == 32


def _reference_released_rule(
    scores: np.ndarray,
    *,
    budget: int,
    max_depth: int = 5,
    threshold: float = 0.8,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Small independent specification of the released recursive rule."""

    values = np.asarray(scores, dtype=np.float64)
    low = float(values.min())
    span = float(values.max()) - low
    normalized = np.zeros_like(values) if span == 0.0 else (values - low) / span
    selected: list[int] = []
    terminal_nodes: list[tuple[int, int, int]] = []

    # Breadth-first traversal is intentionally different from the production
    # implementation's recursive traversal.  Only the mathematical partition
    # and deterministic rank rule are shared.
    frontier: list[tuple[int, int, int]] = [(0, len(values), 0)]
    while frontier:
        next_frontier: list[tuple[int, int, int]] = []
        for start, stop, depth in frontier:
            segment = normalized[start:stop]
            source = np.arange(start, stop, dtype=np.int64)
            ranking = np.lexsort((source, -segment))
            contrast_count = min(budget, len(segment))
            contrast = float(segment[ranking[:contrast_count]].mean() - segment.mean())
            if contrast <= threshold and depth < max_depth:
                midpoint = start + (stop - start) // 2
                next_frontier.extend(
                    [(start, midpoint, depth + 1), (midpoint, stop, depth + 1)]
                )
                continue
            quota = budget // (2**depth)
            selected.extend(int(source[position]) for position in ranking[:quota])
            terminal_nodes.append((start, stop, depth))
        frontier = next_frontier
    return sorted(selected), sorted(terminal_nodes)


@pytest.mark.parametrize("budget", [32, 64, 128])
def test_matches_independent_released_rule_reference(budget: int) -> None:
    rng = np.random.default_rng(20260827 + budget)
    cases = [
        rng.normal(size=640),
        np.zeros(640, dtype=np.float64),
        np.concatenate(
            [
                np.ones(32),
                np.zeros(288),
                np.ones(320),
            ]
        ),
        np.repeat(np.arange(10, dtype=np.float64), 64),
    ]

    for scores in cases:
        expected_indices, expected_nodes = _reference_released_rule(
            scores,
            budget=budget,
        )
        selected, diagnostics = select_aks_indices(scores, budget=budget)
        observed_nodes = sorted(
            (
                int(segment["candidate_position_start"]),
                int(segment["candidate_position_stop"]),
                int(segment["depth"]),
            )
            for segment in diagnostics["terminal_segments"]
        )

        assert selected == expected_indices
        assert observed_nodes == expected_nodes
        assert len(selected) == len(set(selected)) == budget


@pytest.mark.parametrize("budget", [0, 31, 33, 256, True, 32.0])
def test_invalid_budget_is_rejected(budget: object) -> None:
    with pytest.raises(ValueError, match="budget"):
        select_aks_indices(np.zeros(256), budget=budget)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 0}, "max_depth"),
        ({"max_depth": True}, "max_depth"),
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"ratio": 0}, "ratio"),
        ({"ratio": 1.5}, "ratio"),
    ],
)
def test_invalid_selector_parameters_are_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_aks_indices(np.zeros(256), budget=32, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [[0.0, 1.0]],
        [0.0, float("nan")],
        [0.0, float("inf")],
        ["not-a-number"] * 128,
    ],
)
def test_invalid_scores_are_rejected(scores: object) -> None:
    with pytest.raises(ValueError, match="scores"):
        select_aks_indices(scores, budget=32)  # type: ignore[arg-type]


def test_ratio_returns_indices_in_original_input_space() -> None:
    scores = np.zeros(640, dtype=np.float64)
    scores[:80:2] = 1.0

    selected, diagnostics = select_aks_indices(scores, budget=32, ratio=2)

    assert selected == list(range(0, 64, 2))
    assert diagnostics["input_count"] == 640
    assert diagnostics["candidate_count"] == 320
    assert diagnostics["ratio"] == 2
