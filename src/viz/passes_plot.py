"""
passes_plot — Player-pass visualization in our brand style.

Inspired by the Opta Analyst player-pass figure (see figures/Messi_pases.jpg)
but adapted to the project's dark theme + the Diagonality-3D narrative:

  - The colour of each pass arrow encodes its **direction class**
    (forward / diagonal / sideways / backward) instead of just success.
    Gold = diagonal, the SV signature.
  - Unsuccessful passes use the SAME colour but a dashed line and lower
    alpha, so the visual balance (per-class) does not change.
  - The footer is split in **three uniform blocks**: pass-class legend,
    attacking direction, and Opta-style stats (passes / accuracy / DIAG).

The renderer is data-agnostic: it takes a passes DataFrame and draws.
Wiring to real cached events is the caller's responsibility.

Schema expected for the passes DataFrame (one row per pass):
    x, y                 — origin in TRACAB meters (centered, [-52.5, 52.5])
    x_receiver, y_receiver — destination in meters
    direction_class      — "forward" | "diagonal" | "sideways" | "backward"
    successful           — bool (True if completed)

The "successful" column is what the renderer actually checks; if the input
has Bundesliga-style "evaluation" strings, map them upstream:
    successful = evaluation.isin({"successfullyCompleted", "successful"})
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mplsoccer import Pitch

from .common import (
    BG, WHITE, FONT, PKW, DIRECTION_COLORS,
)


# Visual constants tuned for figsize = (16, 10.4) at dpi 200.
ARROW_LW = 1.6
ARROW_ALPHA_OK = 0.92
ARROW_ALPHA_FAIL = 0.55
ARROW_HEAD_LEN = 8
ARROW_HEAD_WIDTH = 6
DIRECTION_ORDER = ["diagonal", "forward", "sideways", "backward"]
DIRECTION_LABELS = {
    "diagonal": "Diagonal",
    "forward":  "Forward",
    "sideways": "Sideways",
    "backward": "Backward",
}


# ── Arrow drawing ──────────────────────────────────────────────────────


def _draw_pass_arrow(
    ax: plt.Axes,
    x0: float, y0: float, x1: float, y1: float,
    color: str,
    successful: bool,
    lw: float = ARROW_LW,
):
    """One pass = one FancyArrowPatch. Solid for OK, dashed for fail.
    Both share the same colour so the per-class density is preserved."""
    style = "-" if successful else (0, (3, 2))   # dash pattern for fail
    alpha = ARROW_ALPHA_OK if successful else ARROW_ALPHA_FAIL
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=f"-|>,head_length={ARROW_HEAD_LEN},"
                   f"head_width={ARROW_HEAD_WIDTH}",
        color=color, lw=lw, alpha=alpha, linestyle=style,
        mutation_scale=1.0, zorder=4,
        capstyle="round",
    )
    ax.add_patch(arrow)


# ── Header ─────────────────────────────────────────────────────────────


def _draw_header(
    fig: plt.Figure,
    ax_header: plt.Axes,
    title: str,
    subtitle: str,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,
):
    """Top strip: optional team logo (left of title), title + subtitle
    (centre-left), optional project logo (top-right). All on the figure
    background, no axes lines visible."""
    ax_header.set_facecolor(BG)
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1)
    ax_header.set_xticks([]); ax_header.set_yticks([])
    for s in ax_header.spines.values():
        s.set_visible(False)

    text_x = 0.04
    if team_logo_path is not None and Path(team_logo_path).exists():
        try:
            img = plt.imread(team_logo_path)
            ab = AnnotationBbox(
                OffsetImage(img, zoom=0.10),
                (0.03, 0.55), frameon=False,
                box_alignment=(0.0, 0.5),
            )
            ax_header.add_artist(ab)
            text_x = 0.10
        except Exception:
            pass

    ax_header.text(
        text_x, 0.72, title, color=WHITE,
        fontsize=22, fontweight="bold", fontfamily=FONT,
        ha="left", va="center",
    )
    ax_header.text(
        text_x, 0.28, subtitle, color=WHITE,
        fontsize=11, fontweight="normal", fontfamily=FONT,
        ha="left", va="center", alpha=0.78,
    )

    if project_logo_path is not None and Path(project_logo_path).exists():
        try:
            img = plt.imread(project_logo_path)
            ab = AnnotationBbox(
                OffsetImage(img, zoom=0.16),
                (0.985, 0.5), frameon=False,
                box_alignment=(1.0, 0.5),
            )
            ax_header.add_artist(ab)
        except Exception:
            pass


# ── Footer (3 uniform blocks) ──────────────────────────────────────────


def _block_directions(ax: plt.Axes):
    """Block 1: 4 colored mini-arrows + labels + dashed-fail caveat."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 4 rows distributed vertically, centered horizontally
    n = len(DIRECTION_ORDER)
    y_top, y_bot = 0.92, 0.20
    arrow_x0, arrow_x1 = 0.10, 0.32
    text_x = 0.38
    rows = []
    for i, key in enumerate(DIRECTION_ORDER):
        y = y_top - i * (y_top - y_bot) / max(n - 1, 1)
        rows.append((y, key))

    for y, key in rows:
        color = DIRECTION_COLORS[key]
        arrow = FancyArrowPatch(
            (arrow_x0, y), (arrow_x1, y),
            arrowstyle=f"-|>,head_length={ARROW_HEAD_LEN},"
                       f"head_width={ARROW_HEAD_WIDTH}",
            color=color, lw=ARROW_LW + 0.4, alpha=ARROW_ALPHA_OK,
            transform=ax.transAxes, mutation_scale=1.0,
        )
        ax.add_patch(arrow)
        ax.text(
            text_x, y, DIRECTION_LABELS[key], color=WHITE,
            fontsize=10, fontweight="bold", fontfamily=FONT,
            ha="left", va="center", transform=ax.transAxes,
        )

    # "Dashed = unsuccessful" caveat at the very bottom
    ax.text(
        0.5, 0.04, "(dashed line · unsuccessful)", color=WHITE,
        fontsize=8.5, fontweight="normal", fontfamily=FONT,
        ha="center", va="bottom", alpha=0.65, transform=ax.transAxes,
    )


def _block_attacking_direction(ax: plt.Axes, attacking_right: bool):
    """Block 2: ▶▶▶▶ attacking-direction icon + label."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 4 stacked triangles, one per "step", soft cyan/white pulse
    n = 4
    y_row = 0.62
    base_x = 0.30
    step_w = 0.10
    triangle_w = 0.06
    for i in range(n):
        cx = base_x + i * step_w
        # Triangle pointing right (or left depending on attacking_right)
        if attacking_right:
            verts = [(cx, y_row + 0.08), (cx + triangle_w, y_row),
                     (cx, y_row - 0.08)]
        else:
            verts = [(cx + triangle_w, y_row + 0.08), (cx, y_row),
                     (cx + triangle_w, y_row - 0.08)]
        alpha = 0.35 + i * 0.18
        tri = mpatches.Polygon(
            verts, closed=True, facecolor=WHITE,
            edgecolor="none", alpha=alpha,
            transform=ax.transAxes,
        )
        ax.add_patch(tri)

    ax.text(
        0.5, 0.30, "Attacking direction", color=WHITE,
        fontsize=10, fontweight="bold", fontfamily=FONT,
        ha="center", va="center", alpha=0.92, transform=ax.transAxes,
    )


def _stat_circle(ax, cx, cy, r, value, label, color):
    """Draw an Opta-style colored circle with `value` inside + label below."""
    circ = Circle((cx, cy), r, facecolor=color, edgecolor="none",
                  alpha=0.92, transform=ax.transAxes, zorder=3)
    ax.add_patch(circ)
    ax.text(cx, cy, value, color=BG,
            fontsize=12, fontweight="bold", fontfamily=FONT,
            ha="center", va="center", transform=ax.transAxes, zorder=4)
    ax.text(cx, cy - r - 0.08, label, color=WHITE,
            fontsize=9, fontweight="normal", fontfamily=FONT,
            ha="center", va="top", transform=ax.transAxes, alpha=0.85)


def _block_stats(
    ax: plt.Axes,
    n_passes: int,
    accuracy_pct: float,
    diag_share_pct: float,
):
    """Block 3: 3 Opta-style number-in-circle stats. Diagonal share gets
    the gold circle to keep the SV emphasis explicit even in the legend."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 3 stats spaced uniformly horizontally, vertically centered
    cy = 0.62
    r = 0.085
    xs = [0.18, 0.50, 0.82]
    _stat_circle(ax, xs[0], cy, r, f"{n_passes}", "passes", WHITE)
    _stat_circle(ax, xs[1], cy, r, f"{accuracy_pct:.0f}%", "accuracy", WHITE)
    _stat_circle(ax, xs[2], cy, r, f"{diag_share_pct:.0f}%",
                 "diagonal", DIRECTION_COLORS["diagonal"])


# ── Main API ───────────────────────────────────────────────────────────


def plot_player_passes(
    passes: pd.DataFrame,
    title: str,
    subtitle: str,
    attacking_right: bool = True,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10.4),
) -> plt.Figure:
    """Render a player-passes figure in the project style.

    Args:
        passes: DataFrame, one row per pass. Columns required:
            x, y, x_receiver, y_receiver, direction_class, successful.
        title: top-of-page title (e.g. "Michael Olise — Passes").
        subtitle: smaller line under title (e.g. match info).
        attacking_right: True if the player attacked toward +x in the
            timeframe being rendered (controls arrow read direction).
        team_logo_path: Optional PNG (transparent BG) for the top-left
            logo next to the title. Skipped if missing.
        project_logo_path: Optional PNG for the top-right project logo
            (Diagonality-3D). Skipped if missing.
        save_path: Optional output file. Saved at dpi=200.

    Layout: 3 horizontal strips
      - row 0 : header   (height 14%)
      - row 1 : pitch    (height 65%)
      - row 2 : footer   (height 21%, split in 3 vertical blocks)
    """
    required = {"x", "y", "x_receiver", "y_receiver",
                "direction_class", "successful"}
    missing = required - set(passes.columns)
    if missing:
        raise ValueError(f"passes DataFrame missing columns: {sorted(missing)}")

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.14, 0.65, 0.21],
        hspace=0.0, wspace=0.0,
        left=0.02, right=0.98, top=0.97, bottom=0.03,
    )

    # ── Header (spans the 3 columns) ─────────────────────────────────
    ax_header = fig.add_subplot(gs[0, :])
    _draw_header(fig, ax_header, title, subtitle,
                 team_logo_path, project_logo_path)

    # ── Pitch (spans the 3 columns) ──────────────────────────────────
    ax_pitch = fig.add_subplot(gs[1, :])
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax_pitch)

    for _, p in passes.iterrows():
        cls = str(p["direction_class"])
        if cls not in DIRECTION_COLORS:
            continue
        color = DIRECTION_COLORS[cls]
        _draw_pass_arrow(
            ax_pitch,
            float(p["x"]), float(p["y"]),
            float(p["x_receiver"]), float(p["y_receiver"]),
            color=color, successful=bool(p["successful"]),
        )

    # ── Footer (3 blocks) ────────────────────────────────────────────
    ax_b1 = fig.add_subplot(gs[2, 0])
    ax_b2 = fig.add_subplot(gs[2, 1])
    ax_b3 = fig.add_subplot(gs[2, 2])

    _block_directions(ax_b1)
    _block_attacking_direction(ax_b2, attacking_right)

    n = int(len(passes))
    if n > 0:
        accuracy = float(passes["successful"].mean()) * 100.0
        diag = float((passes["direction_class"] == "diagonal").mean()) * 100.0
    else:
        accuracy = 0.0; diag = 0.0
    _block_stats(ax_b3, n_passes=n, accuracy_pct=accuracy,
                 diag_share_pct=diag)

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
    return fig
