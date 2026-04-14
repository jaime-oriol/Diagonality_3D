"""
passes_plot — Player-pass visualization in our brand style.

Inspired by the Opta Analyst player-pass figure (figures/Messi_pases.jpg)
but adapted to the project's dark theme + the Diagonality-3D narrative:

  - Pass arrows colored by **direction class** (gold = diagonal, the SV
    signature) instead of just success.
  - Unsuccessful passes use the SAME class colour but a dashed line at
    lower alpha — the per-class density stays readable.
  - Header and footer are anchored to the **exact horizontal extent of
    the pitch** via absolute axis positioning, so the figure feels
    grid-aligned regardless of figsize.
  - Footer = 3 uniform blocks: 2x2 direction legend, attacking-direction
    triangles, Opta-style outlined number-in-circle stats with diagonal
    share highlighted in gold.

Schema expected for the passes DataFrame (one row per pass):
    x, y                 — origin in TRACAB meters (centered, [-52.5, 52.5])
    x_receiver, y_receiver — destination in meters
    direction_class      — "forward" | "diagonal" | "sideways" | "backward"
    successful           — bool (True if completed)
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


# ── Geometry: pitch-aligned absolute layout (figsize-aware) ────────────
# Pitch true aspect = 105 / 68 = 1.544. We size the pitch axis to that
# exact ratio so mplsoccer fills it edge-to-edge, no internal margins.
# Header and footer share the SAME horizontal extent as the pitch.

FIGSIZE = (14.0, 9.5)
PITCH_RATIO = 105.0 / 68.0   # ~1.544

# Fractions of the figure height
H_HEADER = 0.11
H_PITCH = 0.60
H_FOOTER = 0.20
# Vertical spacing
PAD_TOP = 0.02
PAD_HEADER_PITCH = 0.012
PAD_PITCH_FOOTER = 0.012
PAD_BOTTOM = 0.025

# Compute pitch horizontal span from the height ratio so it stays edge-
# aligned with header/footer (constant `pitch_w` as a fraction of figure).
def _layout(figsize=FIGSIZE):
    fw, fh = figsize
    pitch_h_in = H_PITCH * fh
    pitch_w_in = pitch_h_in * PITCH_RATIO
    pitch_w_frac = pitch_w_in / fw
    left = (1.0 - pitch_w_frac) / 2.0
    right = left + pitch_w_frac

    y_footer_bot = PAD_BOTTOM
    y_footer_top = y_footer_bot + H_FOOTER
    y_pitch_bot = y_footer_top + PAD_PITCH_FOOTER
    y_pitch_top = y_pitch_bot + H_PITCH
    y_header_bot = y_pitch_top + PAD_HEADER_PITCH
    y_header_top = y_header_bot + H_HEADER

    # Sanity check: should be <= 1 - PAD_TOP
    assert y_header_top <= 1.0 - PAD_TOP + 1e-6, (
        f"layout overflows: header_top={y_header_top}")

    return {
        "left": left, "right": right, "width": pitch_w_frac,
        "header": (left, y_header_bot, pitch_w_frac, H_HEADER),
        "pitch":  (left, y_pitch_bot, pitch_w_frac, H_PITCH),
        "footer": (left, y_footer_bot, pitch_w_frac, H_FOOTER),
    }


# ── Visual constants for arrows + footer ───────────────────────────────

ARROW_LW = 1.25
ARROW_ALPHA_OK = 0.95
ARROW_ALPHA_FAIL = 0.55
ARROW_HEAD_LEN = 4.0       # in points — small Opta-style head
ARROW_HEAD_WIDTH = 3.2

DIRECTION_ORDER = ["diagonal", "forward", "sideways", "backward"]
DIRECTION_LABELS = {
    "diagonal": "Diagonal",
    "forward":  "Forward",
    "sideways": "Sideways",
    "backward": "Backward",
}


# ── Pass arrows ────────────────────────────────────────────────────────

def _draw_pass_arrow(
    ax: plt.Axes,
    x0: float, y0: float, x1: float, y1: float,
    color: str, successful: bool,
):
    """One pass = thin Opta-style arrow. Solid for OK, dashed + faded
    for fail. Same color either way so the per-class density reads."""
    style = "-" if successful else (0, (3, 2))
    alpha = ARROW_ALPHA_OK if successful else ARROW_ALPHA_FAIL
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=f"-|>,head_length={ARROW_HEAD_LEN},"
                   f"head_width={ARROW_HEAD_WIDTH}",
        color=color, lw=ARROW_LW, alpha=alpha,
        linestyle=style, mutation_scale=1.0,
        capstyle="round", joinstyle="round",
        zorder=4,
    )
    ax.add_patch(arrow)


# ── Header ─────────────────────────────────────────────────────────────

def _draw_header(
    ax: plt.Axes,
    title: str, subtitle: str,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,
):
    """Top strip aligned to pitch width: optional team logo (left edge),
    title + subtitle (left), optional project logo (right edge)."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    text_x = 0.0
    if team_logo_path is not None and Path(team_logo_path).exists():
        try:
            img = plt.imread(team_logo_path)
            ab = AnnotationBbox(
                OffsetImage(img, zoom=0.32),
                (0.005, 0.55), frameon=False,
                box_alignment=(0.0, 0.5),
            )
            ax.add_artist(ab)
            text_x = 0.10
        except Exception:
            pass

    ax.text(
        text_x, 0.72, title, color=WHITE,
        fontsize=20, fontweight="bold", fontfamily=FONT,
        ha="left", va="center",
    )
    ax.text(
        text_x, 0.26, subtitle, color=WHITE,
        fontsize=10.5, fontweight="normal", fontfamily=FONT,
        ha="left", va="center", alpha=0.78,
    )

    if project_logo_path is not None and Path(project_logo_path).exists():
        try:
            img = plt.imread(project_logo_path)
            ab = AnnotationBbox(
                OffsetImage(img, zoom=0.20),
                (1.0, 0.5), frameon=False,
                box_alignment=(1.0, 0.5),
            )
            ax.add_artist(ab)
        except Exception:
            pass


# ── Footer block 1: 2x2 direction-class legend ─────────────────────────

def _block_directions(ax: plt.Axes):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 2x2 grid: top row = diagonal | forward, bottom = sideways | backward
    layout = [
        (0.07, 0.70, "diagonal"),
        (0.55, 0.70, "forward"),
        (0.07, 0.36, "sideways"),
        (0.55, 0.36, "backward"),
    ]
    arrow_dx = 0.16
    text_pad = 0.02
    for ax_left, y, key in layout:
        color = DIRECTION_COLORS[key]
        arrow = FancyArrowPatch(
            (ax_left, y), (ax_left + arrow_dx, y),
            arrowstyle=f"-|>,head_length={ARROW_HEAD_LEN},"
                       f"head_width={ARROW_HEAD_WIDTH}",
            color=color, lw=ARROW_LW + 0.4, alpha=ARROW_ALPHA_OK,
            transform=ax.transAxes, mutation_scale=1.0,
            capstyle="round",
        )
        ax.add_patch(arrow)
        ax.text(
            ax_left + arrow_dx + text_pad, y, DIRECTION_LABELS[key],
            color=WHITE, fontsize=10, fontweight="bold", fontfamily=FONT,
            ha="left", va="center", transform=ax.transAxes,
        )

    ax.text(
        0.5, 0.05, "(dashed line · unsuccessful)", color=WHITE,
        fontsize=8.5, fontweight="normal", fontfamily=FONT,
        ha="center", va="bottom", alpha=0.62, transform=ax.transAxes,
    )


# ── Footer block 2: attacking direction ────────────────────────────────

def _block_attacking_direction(ax: plt.Axes, attacking_right: bool):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    n = 4
    y_row = 0.62
    triangle_w = 0.05
    base_x = 0.30
    step = 0.10
    for i in range(n):
        cx = base_x + i * step
        if attacking_right:
            verts = [(cx, y_row + 0.07), (cx + triangle_w, y_row),
                     (cx, y_row - 0.07)]
        else:
            verts = [(cx + triangle_w, y_row + 0.07), (cx, y_row),
                     (cx + triangle_w, y_row - 0.07)]
        alpha = 0.32 + i * 0.18
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


# ── Footer block 3: Opta-style outlined stat circles ───────────────────

def _stat_circle_with_label(
    ax: plt.Axes,
    cx: float, cy: float,
    value: str, label: str,
    edge_color: str = WHITE,
    text_color: str = WHITE,
    radius: float = 0.075,
):
    """Outlined circle with `value` inside + `label` text to the right.
    Mirrors the Opta `(43) passes` aesthetic. All coords in axes frac."""
    circ = Circle(
        (cx, cy), radius,
        facecolor="none", edgecolor=edge_color, lw=1.6,
        alpha=0.95, transform=ax.transAxes, zorder=3,
    )
    ax.add_patch(circ)
    ax.text(
        cx, cy, value, color=text_color,
        fontsize=11, fontweight="bold", fontfamily=FONT,
        ha="center", va="center", transform=ax.transAxes, zorder=4,
    )
    ax.text(
        cx + radius + 0.04, cy, label, color=WHITE,
        fontsize=10, fontweight="normal", fontfamily=FONT,
        ha="left", va="center", transform=ax.transAxes, alpha=0.88,
    )


def _block_stats(
    ax: plt.Axes,
    n_passes: int,
    accuracy_pct: float,
    diag_share_pct: float,
):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    cy = 0.55
    # 3 stats: passes, accuracy, diagonal share. Diagonal gets gold edge.
    _stat_circle_with_label(ax, 0.05, cy, f"{n_passes}", "passes",
                            edge_color=WHITE)
    _stat_circle_with_label(ax, 0.38, cy, f"{accuracy_pct:.0f}%", "accuracy",
                            edge_color=WHITE)
    _stat_circle_with_label(ax, 0.71, cy, f"{diag_share_pct:.0f}%", "diagonal",
                            edge_color=DIRECTION_COLORS["diagonal"],
                            text_color=DIRECTION_COLORS["diagonal"])


# ── Main API ───────────────────────────────────────────────────────────

def plot_player_passes(
    passes: pd.DataFrame,
    title: str,
    subtitle: str,
    attacking_right: bool = True,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """Render a player-passes figure in the project style.

    Layout: header, pitch and 3-block footer all anchored to the exact
    horizontal extent of the pitch (no overflow into the figure margins).

    Args:
        passes: DataFrame, one row per pass. Required columns:
            x, y, x_receiver, y_receiver, direction_class, successful.
        title: top-of-page title (e.g. "Michael Olise — Passes").
        subtitle: smaller line under title (match info).
        attacking_right: True if the player attacked toward +x.
        team_logo_path: Optional PNG (transparent BG) for the top-left
            logo next to the title. Skipped if missing.
        project_logo_path: Optional PNG for the top-right project logo.
            Skipped if missing.
        save_path: Optional output file. Saved at dpi=200.
    """
    required = {"x", "y", "x_receiver", "y_receiver",
                "direction_class", "successful"}
    missing = required - set(passes.columns)
    if missing:
        raise ValueError(f"passes DataFrame missing columns: {sorted(missing)}")

    L = _layout(figsize)
    fig = plt.figure(figsize=figsize, facecolor=BG)

    ax_header = fig.add_axes(L["header"])
    _draw_header(ax_header, title, subtitle,
                 team_logo_path, project_logo_path)

    ax_pitch = fig.add_axes(L["pitch"])
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax_pitch)

    for _, p in passes.iterrows():
        cls = str(p["direction_class"])
        if cls not in DIRECTION_COLORS:
            continue
        _draw_pass_arrow(
            ax_pitch,
            float(p["x"]), float(p["y"]),
            float(p["x_receiver"]), float(p["y_receiver"]),
            color=DIRECTION_COLORS[cls],
            successful=bool(p["successful"]),
        )

    # Footer split horizontally into 3 equal-width blocks
    fl, fb, fw, fh = L["footer"]
    block_w = fw / 3.0
    ax_b1 = fig.add_axes([fl, fb, block_w, fh])
    ax_b2 = fig.add_axes([fl + block_w, fb, block_w, fh])
    ax_b3 = fig.add_axes([fl + 2 * block_w, fb, block_w, fh])

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
                    bbox_inches=None)
    return fig
