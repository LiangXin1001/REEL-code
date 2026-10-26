#!/usr/bin/env python3
"""Generate REEL overview and resource-accounting diagrams as PDF/PNG assets."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reel-diagram-mpl")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "reel_wacv2027"
ASSETS = PAPER / "figures" / "assets"
OUT = PAPER / "figures" / "generated"

NAVY = "#0B274E"
BLUE = "#195DA5"
TEAL = "#007E89"
GREEN = "#228158"
ORANGE = "#CF6F2B"
PURPLE = "#654B98"
RED = "#B64343"
PALE = "#EFF6FD"
GREY = "#D8E1EB"


def box(ax, x, y, w, h, text, edge, face="white", size=8, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=NAVY, fontweight=weight, zorder=3)
    return patch


def arrow(ax, start, end, color=NAVY, width=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=width, color=color, zorder=1))


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=.02)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(10.5, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    source = box(ax, .035, .68, .20, .18,
                 "LONG VIDEO $V$\nREPEATED QUESTIONS $q_1, \\ldots, q_m$",
                 NAVY, PALE, 8.1, "bold")

    observe = box(ax, .30, .70, .20, .15,
                  "observe after each question", ORANGE, "#FFF8F1", 7.8)
    write = box(ax, .30, .43, .20, .18,
                "build before questions\nthen reuse", GREEN, "#F2FAF6", 7.8)
    box(ax, .56, .70, .17, .15,
        "repeated visual\ncomputation", ORANGE, "#FFF8F1", 7.7)
    box(ax, .56, .43, .17, .18,
        "saved representation\nreused by later questions", GREEN, "#F2FAF6", 7.7)
    box(ax, .80, .56, .17, .22,
        "CUMULATIVE RESOURCE $R_k(m)$\nconstruction cost amortized?",
        PURPLE, "#F7F4FB", 7.5, "bold")

    arrow(ax, (source.get_x() + source.get_width(), .77), (observe.get_x(), .775), ORANGE)
    arrow(ax, (source.get_x() + source.get_width(), .72), (write.get_x(), .52), GREEN)
    arrow(ax, (.50, .775), (.56, .775), ORANGE)
    arrow(ax, (.50, .52), (.56, .52), GREEN)
    arrow(ax, (.73, .775), (.80, .70), ORANGE)
    arrow(ax, (.73, .52), (.80, .63), GREEN)

    box(ax, .56, .12, .17, .17,
        "PAIRED ANSWERS\nwhich questions change?\nquality within margin?",
        RED, "#FFF6F6", 6.9, "bold")
    box(ax, .80, .10, .17, .20,
        "USEFUL REUSE\nlower resource\n+ quality maintained",
        NAVY, PALE, 7.1, "bold")
    arrow(ax, (.885, .56), (.885, .30), PURPLE)
    arrow(ax, (.73, .205), (.80, .205), RED)
    ax.text(.035, .19,
            "RQ1 compares which questions remain answerable from each saved form.\n"
            "RQ2 tests when cumulative cost becomes lower without losing quality.",
            ha="left", va="center", fontsize=7.9, color=NAVY)
    save(fig, "overview")


def ledger():
    fig, ax = plt.subplots(figsize=(10.5, 1.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.04, .91, "ACTUAL INFERENCE OPERATIONS", fontsize=8.5,
            fontweight="bold", color=NAVY, ha="left")
    stages = [
        (.05, "decode", BLUE, "#F2F7FC"),
        (.18, "observe", ORANGE, "#FFF8F1"),
        (.31, "retrieve", GREEN, "#F2FAF6"),
        (.44, "answer", PURPLE, "#F7F4FB"),
    ]
    patches = [box(ax, x, .60, .095, .18, text, edge, face, 8)
               for x, text, edge, face in stages]
    for left, right in zip(patches, patches[1:]):
        arrow(ax, (left.get_x() + left.get_width(), .69), (right.get_x(), .69))

    row = box(ax, .045, .18, .51, .22,
              "append one row at call time\nphase | cache | frames/tokens | latency | GPU/API | error",
              PURPLE, "white", 7.5)
    for patch in patches:
        arrow(ax, (patch.get_x() + patch.get_width()/2, patch.get_y()),
              (patch.get_x() + patch.get_width()/2, row.get_y() + row.get_height()), PURPLE, .8)

    validate = box(ax, .65, .43, .16, .24, "fixed data list + complete\nquestions and resource fields",
                   RED, "#FFF6F6", 7.5)
    result = box(ax, .86, .43, .12, .24, "resource curves\nthat can be checked",
                 NAVY, PALE, 7.5)
    arrow(ax, (row.get_x() + row.get_width(), row.get_y() + row.get_height()/2),
          (validate.get_x(), validate.get_y() + validate.get_height()/2))
    arrow(ax, (validate.get_x() + validate.get_width(), validate.get_y() + validate.get_height()/2),
          (result.get_x(), result.get_y() + result.get_height()/2))
    ax.text(.73, .24, "incomplete resource log $\\Rightarrow$ reject resource claim", color=RED,
            fontsize=7.3, ha="center", va="center")
    save(fig, "ledger")


def evaluation_gap():
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rows = ["Conventional LVQA", "CG-Bench", "HERBench", "Cost-of-Pass", "REEL"]
    columns = [
        "Multiple questions\nper video",
        "Charges shared\npreprocessing",
        "Cumulative\ncost $C(m)$",
        "Evidence\nlocalization",
        "Evidence\nsufficiency",
        "Controlled reuse\ndecomposition",
    ]
    # 0: absent, 0.5: partial or available in the data but not the protocol, 1: included.
    values = [
        [0, .5, 0, .5, 0, 0],
        [1, 0, 0, 1, .5, 0],
        [1, 0, 0, 1, .5, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    colors = {0: RED, .5: ORANGE, 1: BLUE}

    left, right = .22, .975
    top, bottom = .78, .14
    col_w = (right - left) / len(columns)
    row_h = (top - bottom) / len(rows)
    ax.add_patch(Rectangle((.02, top), right - .02, .16, facecolor=PALE,
                           edgecolor="none", zorder=0))
    for j, label in enumerate(columns):
        x = left + (j + .5) * col_w
        ax.text(x, top + .08, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=NAVY)

    for i, (row, vals) in enumerate(zip(rows, values)):
        y = top - (i + .5) * row_h
        if row == "REEL":
            ax.add_patch(Rectangle((.02, y - row_h / 2), right - .02, row_h,
                                   facecolor="#F2F7FC", edgecolor="none", zorder=0))
        ax.text(.035, y, row, ha="left", va="center", fontsize=8.2,
                fontweight="bold" if row == "REEL" else "normal", color=NAVY)
        for j, value in enumerate(vals):
            x = left + (j + .5) * col_w
            ax.add_patch(Circle((x, y), .018, facecolor=colors[value],
                                edgecolor="white", linewidth=.8, zorder=2))
        ax.plot([.02, right], [y - row_h / 2, y - row_h / 2], color=GREY,
                linewidth=.55, zorder=1)

    ax.plot([left, left], [bottom, top + .16], color=GREY, linewidth=.7)
    legend = [(BLUE, "Included"), (ORANGE, "Partial / data only"), (RED, "Not included")]
    for x, (color, label) in zip([.30, .50, .72], legend):
        ax.add_patch(Circle((x, .055), .012, facecolor=color, edgecolor="white", linewidth=.6))
        ax.text(x + .022, .055, label, ha="left", va="center", fontsize=7.3, color=NAVY)
    save(fig, "evaluation_gap")


def paired_comparisons():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(.5, .955,
            "FIXED WITHIN EACH PAIR: model | video | questions | frame budget | hardware",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=NAVY)

    lanes = [
        (.76, "REUSE", "Regenerate the same\ncaptions", "Reuse the same\ncaptions",
         "change: reuse", "measure: repeated computation", BLUE),
        (.55, "QUESTION AVAILABILITY", "Question hidden\nfrom writer", "Question visible\nto writer",
         "change: question visible", "measure: relevant information", ORANGE),
        (.34, "CACHE LOCATION", "Repeated frame\ninput", "Pixel cache     KV cache",
         "change: cache level", "measure: I/O vs. GPU inference", PURPLE),
        (.13, "EVIDENCE INPUT", "Uniform / retrieved\nframes", "Annotated-evidence\nframes",
         "change: source timestamps", "measure: gain from better evidence", RED),
    ]
    for y, label, left_text, right_text, change, measure, color in lanes:
        ax.text(.025, y, label, ha="left", va="center", fontsize=7.5,
                fontweight="bold", color=color)
        box(ax, .22, y - .065, .20, .13, left_text, color, "white", 7.7)
        box(ax, .59, y - .065, .20, .13, right_text, color,
            PALE if color == BLUE else "white", 7.7)
        arrow(ax, (.42, y), (.59, y), color, 1.3)
        ax.text(.505, y + .035, change, ha="center", va="center", fontsize=7.0,
                color=color, fontweight="bold")
        ax.text(.86, y, measure, ha="left", va="center", fontsize=7.5, color=NAVY)
        if y > .13:
            ax.plot([.02, .98], [y - .105, y - .105], color=GREY, linewidth=.55)
    save(fig, "paired_comparisons")


def caption_triangle():
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax, .34, .70, .32, .19,
        "RECOMPUTED CAPTIONS\nquestion hidden\nrecompute per query",
        BLUE, "#F2F7FC", 10.3, "bold",
    )
    box(
        ax, .03, .11, .31, .19,
        "SHARED CAPTIONS\nquestion hidden\nbuild once and reuse",
        TEAL, "#F1FAFA", 10.3, "bold",
    )
    box(
        ax, .66, .11, .31, .19,
        "QUESTION-SPECIFIC CAPTIONS\nquestion visible\nrecompute per query",
        ORANGE, "#FFF8F1", 9.8, "bold",
    )

    def double_arrow(start, end, color):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="<->", mutation_scale=10,
            linewidth=1.3, color=color, zorder=1,
        ))

    double_arrow((.405, .70), (.285, .30), TEAL)
    ax.text(.25, .55, "REUSE EFFECT", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=TEAL, rotation=55)
    ax.text(.15, .49, "same content", ha="center", va="center",
            fontsize=8.7, color=NAVY, rotation=55)

    double_arrow((.595, .70), (.715, .30), ORANGE)
    ax.text(.75, .55, "QUESTION AVAILABILITY", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=ORANGE, rotation=-55)
    ax.text(.86, .50, "both recomputed", ha="center", va="center",
            fontsize=8.7, color=NAVY, rotation=-55)

    double_arrow((.34, .205), (.66, .205), RED)
    ax.text(.50, .40, "ACTUAL REUSE TRADE-OFF", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=RED)
    ax.text(.50, .045, "resource amortization + maintained quality?", ha="center",
            va="center", fontsize=8.7, color=NAVY)
    save(fig, "caption_condition_triangle")


if __name__ == "__main__":
    overview()
    ledger()
    evaluation_gap()
    paired_comparisons()
    caption_triangle()
    print(OUT)
