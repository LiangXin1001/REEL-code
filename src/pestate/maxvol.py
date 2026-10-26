"""Deterministic, NumPy-only row selection for the MaxInfo frame policy.

The row-swap loop in :func:`maxvol_rows` is adapted from ``py_maxvol`` in
``maxvolpy`` 0.3.8, written by Alexander Mikhalev.  The upstream package is
released under the MIT License.  We keep this small fallback in-tree because
the published 0.3.8 extension does not build reliably on current Python
versions.  Our initialization uses deterministic modified Gram--Schmidt and
our implementation recomputes the coefficient matrix after a swap; these are
documented changes from the upstream routine.

Upstream project: https://pypi.org/project/maxvolpy/0.3.8/
Upstream 0.3.8 source-distribution SHA-256:
df4adb1bbc091e960ec5f3e5b53c3909ae428f5e48cf7ab95a01ea982a1a1980

MIT License

Copyright (c) 2015 Alexander Mikhalev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np


DEFAULT_BUDGETS = (32, 64, 128)
DEFAULT_FRAMES_PER_BLOCK = 2


def _matrix(value: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def _initial_rows(matrix: np.ndarray) -> np.ndarray:
    """Choose a nonsingular starting submatrix without random tie breaking."""

    rows, rank = matrix.shape
    residual = matrix.copy()
    selected: list[int] = []
    available = np.ones(rows, dtype=bool)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    threshold = np.finfo(np.float64).eps * max(matrix.shape) * scale
    for _ in range(rank):
        scores = np.einsum("ij,ij->i", residual, residual)
        scores[~available] = -np.inf
        pivot = int(np.argmax(scores))
        norm = float(np.sqrt(max(0.0, scores[pivot])))
        if not np.isfinite(norm) or norm <= threshold:
            raise np.linalg.LinAlgError("matrix does not have full column rank")
        selected.append(pivot)
        available[pivot] = False
        direction = residual[pivot] / norm
        residual -= np.outer(residual @ direction, direction)
    return np.asarray(selected, dtype=np.int64)


def _coefficients(matrix: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    basis = matrix[pivots]
    try:
        coefficients = np.linalg.solve(basis.T, matrix.T).T
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("selected MaxVol rows are singular") from exc
    coefficients[pivots] = np.eye(len(pivots), dtype=matrix.dtype)
    return coefficients


def maxvol_rows(
    matrix: np.ndarray,
    *,
    tol: float = 1.05,
    max_iters: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a high-volume square row submatrix.

    Parameters follow ``maxvolpy.maxvol.py_maxvol`` for the subset needed by
    this project.  ``matrix`` must have at least as many rows as columns and
    full column rank.  Ties always resolve to the earliest row.
    """

    values = _matrix(matrix)
    rows, rank = values.shape
    if rows < rank:
        raise ValueError(f"matrix has fewer rows ({rows}) than columns ({rank})")
    if not np.isfinite(tol) or tol < 1.0:
        raise ValueError("tol must be finite and at least 1")
    if max_iters < 0:
        raise ValueError("max_iters must be non-negative")
    if rows == rank:
        pivots = np.arange(rows, dtype=np.int64)
        return pivots, np.eye(rows, dtype=values.dtype)

    pivots = _initial_rows(values)
    coefficients = _coefficients(values, pivots)
    for _ in range(max_iters):
        row, column = np.unravel_index(
            int(np.argmax(np.abs(coefficients))), coefficients.shape
        )
        if abs(float(coefficients[row, column])) <= tol:
            break
        pivots[column] = row
        if len(set(map(int, pivots))) != rank:
            raise RuntimeError("MaxVol attempted to select one row twice")
        coefficients = _coefficients(values, pivots)
    else:
        maximum = float(np.abs(coefficients).max())
        if maximum > tol:
            raise RuntimeError(
                f"MaxVol did not converge in {max_iters} iterations; max coefficient={maximum:.6g}"
            )

    coefficients[pivots] = np.eye(rank, dtype=values.dtype)
    return pivots.copy(), coefficients


def _normalized_rows(embeddings: np.ndarray) -> np.ndarray:
    values = _matrix(embeddings, name="embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("embeddings contain a zero-norm row")
    return values / norms


def _time_blocks(row_count: int, blocks: int, per_block: int) -> list[np.ndarray]:
    if blocks <= 0 or per_block <= 0:
        raise ValueError("blocks and per_block must be positive")
    budget = blocks * per_block
    if row_count < budget:
        raise ValueError(f"selection requires at least {budget} candidate frames, got {row_count}")
    groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(np.arange(row_count), blocks)]
    if any(len(group) < per_block for group in groups):
        raise ValueError(f"each of {blocks} time blocks must contain at least {per_block} candidates")
    return groups


def blockwise_maxvol_indices(
    embeddings: np.ndarray,
    *,
    blocks: int = 32,
    per_block: int = 2,
    tol: float = 1.05,
    max_iters: int = 100,
) -> list[int]:
    """Select ``per_block`` diverse CLIP rows from each contiguous block."""

    values = _normalized_rows(embeddings)
    groups = _time_blocks(values.shape[0], blocks, per_block)
    if values.shape[1] < per_block:
        raise ValueError(
            f"embeddings need at least {per_block} dimensions for blockwise MaxVol"
        )

    selected: list[int] = []
    for group in groups:
        if len(group) == per_block:
            local = np.arange(per_block, dtype=np.int64)
        else:
            # The leading left singular vectors form a rank-k row embedding.
            # Applying MaxVol to this basis chooses a D-optimal, diverse subset.
            left, _, _ = np.linalg.svd(values[group], full_matrices=False)
            basis = left[:, :per_block]
            local, _ = maxvol_rows(basis, tol=tol, max_iters=max_iters)
        selected.extend(int(group[index]) for index in local)
    return sorted(selected)


def maxvol_budget_sweep_indices(
    embeddings: np.ndarray,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    frames_per_block: int = DEFAULT_FRAMES_PER_BLOCK,
    tol: float = 1.05,
    max_iters: int = 100,
) -> dict[int, list[int]]:
    """Run the chunked rank-2 MaxVol policy for several frame budgets.

    The caller supplies one shared candidate embedding matrix.  Budget ``K``
    is instantiated as ``K / frames_per_block`` chronological blocks, with
    ``frames_per_block`` rows selected independently from every block.  This
    helper never decodes frames or computes embeddings, so a budget sweep can
    reuse one physical candidate/CLIP pass while keeping each policy output
    logically independent.
    """

    values = _normalized_rows(embeddings)
    raw_budgets = list(budgets)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_budgets
    ):
        raise ValueError("every budget must be an integer")
    requested = [int(value) for value in raw_budgets]
    if not requested:
        raise ValueError("budgets must be non-empty")
    if isinstance(frames_per_block, bool) or not isinstance(frames_per_block, Integral):
        raise ValueError("frames_per_block must be a positive integer")
    frames_per_block = int(frames_per_block)
    if frames_per_block <= 0:
        raise ValueError("frames_per_block must be a positive integer")
    if len(set(requested)) != len(requested):
        raise ValueError("budgets must be unique")
    if any(value <= 0 or value % frames_per_block for value in requested):
        raise ValueError("every budget must be positive and divisible by frames_per_block")
    if values.shape[0] < max(requested):
        raise ValueError(
            f"largest budget requires {max(requested)} candidates, got {values.shape[0]}"
        )

    return {
        budget: blockwise_maxvol_indices(
            values,
            blocks=budget // frames_per_block,
            per_block=frames_per_block,
            tol=tol,
            max_iters=max_iters,
        )
        for budget in requested
    }


def semantic_change_indices(
    embeddings: np.ndarray,
    *,
    blocks: int = 32,
    per_block: int = 2,
) -> list[int]:
    """Select the largest adjacent CLIP changes within each time block."""

    values = _normalized_rows(embeddings)
    groups = _time_blocks(values.shape[0], blocks, per_block)
    selected: list[int] = []
    for group in groups:
        block = values[group]
        scores = np.zeros(len(group), dtype=np.float64)
        if len(group) > 1:
            scores[1:] = 1.0 - np.einsum("ij,ij->i", block[1:], block[:-1])
        # lexsort makes the tie rule explicit: larger change first, then time.
        ranked = np.lexsort((group, -scores))
        selected.extend(int(group[index]) for index in ranked[:per_block])
    return sorted(selected)


def semantic_set_metrics(
    candidate_embeddings: np.ndarray,
    selected_indices: Sequence[int],
) -> dict[str, float | int]:
    """Measure coverage of all candidates and redundancy within a selection.

    Coverage is the mean, over every 1-fps candidate, of its largest cosine
    similarity to a selected frame.  Redundancy is the mean nearest-neighbor
    cosine similarity among selected frames (self matches are excluded).
    """

    candidates = _normalized_rows(candidate_embeddings)
    indices = np.asarray(list(selected_indices), dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("selected_indices must be a non-empty one-dimensional sequence")
    if len(set(map(int, indices))) != len(indices):
        raise ValueError("selected_indices contains duplicates")
    if int(indices.min()) < 0 or int(indices.max()) >= len(candidates):
        raise ValueError("selected_indices contains an out-of-range row")

    selected = candidates[indices]
    # Bound temporary matrices for multi-hour videos.  A full candidate by
    # candidate cosine matrix can otherwise consume several gigabytes.
    coverage_sum = 0.0
    for start in range(0, len(candidates), 512):
        similarities = np.clip(candidates[start : start + 512] @ selected.T, -1.0, 1.0)
        coverage_sum += float(np.max(similarities, axis=1).sum())
    coverage = coverage_sum / len(candidates)
    if len(selected) == 1:
        redundancy = 0.0
    else:
        nearest_sum = 0.0
        for start in range(0, len(selected), 512):
            stop = min(len(selected), start + 512)
            within = np.clip(selected[start:stop] @ selected.T, -1.0, 1.0)
            within[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            nearest_sum += float(np.max(within, axis=1).sum())
        redundancy = nearest_sum / len(selected)
    return {
        "count": int(len(selected)),
        "coverage_mean_cosine": coverage,
        "redundancy_mean_nearest_cosine": redundancy,
    }


__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_FRAMES_PER_BLOCK",
    "blockwise_maxvol_indices",
    "maxvol_rows",
    "maxvol_budget_sweep_indices",
    "semantic_change_indices",
    "semantic_set_metrics",
]
