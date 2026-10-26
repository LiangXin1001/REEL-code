"""Shared visual constants for axis-based figures in the PEState paper.

Keep this file small: figure-specific layout stays in each generator, while
frame, grid, edge, and auxiliary-text styling is defined once here.
"""

from __future__ import annotations

from typing import Any


INK = "#111111"
GRID = "#D9DEE3"
EDGE_WIDTH = 0.8
GRID_WIDTH = 0.55
AXIS_LABEL_SIZE = 8.2
TICK_LABEL_SIZE = 7.2
LEGEND_SIZE = 7.0


def style_xy_axis(
    ax: Any,
    *,
    tick_size: float = TICK_LABEL_SIZE,
    grid_which: str = "major",
    grid_alpha: float = 0.82,
) -> None:
    """Apply the paper standard to a conventional x/y axis."""

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(EDGE_WIDTH)
    ax.grid(
        True,
        which=grid_which,
        axis="both",
        color=GRID,
        linewidth=GRID_WIDTH,
        alpha=grid_alpha,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        colors=INK,
        labelsize=tick_size,
        width=EDGE_WIDTH,
    )


def style_heatmap_axis(ax: Any, *, tick_size: float = 5.2) -> None:
    """Draw a black outer box and a two-direction cell grid."""

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(EDGE_WIDTH)
    ax.grid(
        True,
        which="minor",
        axis="both",
        color=GRID,
        linewidth=GRID_WIDTH,
    )
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", colors=INK, labelsize=tick_size)


def style_colorbar(colorbar: Any, *, tick_size: float = 7.0) -> None:
    """Keep colorbar text legible and its outline consistent with plot edges."""

    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor(INK)
    colorbar.outline.set_linewidth(EDGE_WIDTH)
    colorbar.ax.tick_params(
        colors=INK,
        labelsize=tick_size,
        width=EDGE_WIDTH,
    )
