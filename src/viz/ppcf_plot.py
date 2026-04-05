"""
ppcf_plot — Render Immediate Orientation-Aware PPCF on pitch.

Opta Forum aesthetic (dark BG, blue/red divergent cmap, Opta player markers)
adapted to immediate PPCF semantics: alpha is modulated by contrast from the
neutral point 0.5, so only cells where a team genuinely dominates in the
short window are painted. Cells where neither team resolves control within
the immediate horizon fade to transparent, letting the pitch BG show.

This is the visual signature of "immediate, not global" — we don't paint the
whole pitch in asymptotic colors; we paint the zones where orientation and
reaction time actually decide control in the next ~1 second.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from ..ppcf import compute_ppcf_surfaces, default_params
from .common import (
    BG, WHITE, PKW, PPCF_CMAP,
    ATT as ATT_C, ATT_LIGHT, DEF as DEF_C, DEF_LIGHT,
    GK as GK_C, BALL as BALL_C,
)

PE_S = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 20  # Opta-style large markers for overlay contrast


def plot_ppcf_frame(
    orientations_frame: pd.DataFrame,
    ball_xy: tuple,
    attacking_team: int,
    gk_jerseys: dict = None,
    ppcf_att: np.ndarray = None,
    ppcf_def: np.ndarray = None,
    immediate_window: float = 0.8,
    n_grid_x: int = 100,
    alpha_max: float = 0.75,
    title: str = "",
    ax: plt.Axes = None,
    save_path: str = None,
    figsize: tuple = (16, 10.4),
) -> plt.Figure:
    """Render one frame with immediate PPCF heatmap + players + ball.

    Alpha semantics (the immediate signature):
        alpha_per_cell = (ppcf_att + ppcf_def) * alpha_max
    So unresolved cells (nobody reaches in the window) have alpha=0 and
    show the pitch BG through. Resolved cells paint from red (defender)
    through gray (contested) to blue (attacker), with opacity equal to how
    much probability mass has actually been resolved in the window.

    Args:
        orientations_frame: Single-frame slice (output of add_dynamics).
            Must have team, jersey, x, y, shoulder_angle, head_angle,
            shoulder_width, vx, vy.
        ball_xy: (x, y) ball position in meters.
        attacking_team: Team ID of attacking team (0 or 1).
        gk_jerseys: {team_id: jersey}. Default {0:1, 1:1}.
        ppcf_att, ppcf_def: Pre-computed surfaces from compute_ppcf_surfaces.
            If None, computed inside.
        immediate_window: Integration horizon in seconds. Default 0.8s.
        n_grid_x: Grid resolution. Default 100 (~1m cells).
        alpha_max: Peak alpha for fully-resolved cells. Default 0.75.
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    # Compute surfaces if not provided
    if ppcf_att is None or ppcf_def is None:
        params = default_params(immediate_window=immediate_window)
        ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
            orientations_frame, attacking_team, ball_xy,
            gk_jerseys=gk_jerseys, params=params, n_grid_x=n_grid_x,
        )

    # Figure / axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # Resolved mass = fraction of probability mass the model has decided
    # within the window (0 = nobody reaches, 1 = fully contested/decided).
    # This is the alpha channel: unresolved -> transparent -> field BG.
    resolved = np.clip(ppcf_att + ppcf_def, 0.0, 1.0)
    alpha = resolved * alpha_max

    # Color from att-def difference in [-1, 1], mapped to [0, 1] cmap domain
    # (0 -> red/def, 0.5 -> gray, 1 -> blue/att).
    winner = np.clip(ppcf_att - ppcf_def, -1.0, 1.0)
    color_idx = 0.5 + 0.5 * winner

    rgba = PPCF_CMAP(color_idx)
    rgba[..., 3] = alpha

    ax.imshow(
        rgba, origin="lower",
        extent=[-52.5, 52.5, -34.0, 34.0],
        interpolation="spline36",
        zorder=1, aspect="auto",
    )

    # --- Players (Opta Forum style, same as vision_plot) ---
    for _, p in orientations_frame.iterrows():
        t = int(p["team"])
        j = int(p["jersey"])
        x, y = float(p["x"]), float(p["y"])
        is_gk = (j == gk_jerseys.get(t, 1))
        color = GK_C if is_gk else (ATT_C if t == attacking_team else DEF_C)

        # Dot
        ax.plot(x, y, "o", ms=MS, color=color,
                markeredgecolor=WHITE, markeredgewidth=1,
                alpha=0.85, zorder=5)

        # Shoulder bar (body orientation) — shown for ALL players, GKs included
        sa = p.get("shoulder_angle", np.nan)
        if not (isinstance(sa, float) and np.isnan(sa)):
            sw = float(p.get("shoulder_width", 0.45) or 0.45) * 9
            perp_l = sa + np.pi / 2
            perp_r = sa - np.pi / 2
            # GK shoulder bar in white for contrast against black dot
            bar_color = WHITE if is_gk else color
            ax.plot(
                [x + (sw / 2) * np.cos(perp_l), x + (sw / 2) * np.cos(perp_r)],
                [y + (sw / 2) * np.sin(perp_l), y + (sw / 2) * np.sin(perp_r)],
                color=bar_color, linewidth=8, alpha=0.7, zorder=4,
                solid_capstyle="round",
            )

        # Head wedge (gaze direction) — shown for ALL players, GKs included
        ha = p.get("head_angle", np.nan)
        if not (isinstance(ha, float) and np.isnan(ha)):
            if is_gk:
                head_color = WHITE
            else:
                head_color = ATT_LIGHT if t == attacking_team else DEF_LIGHT
            wedge = Wedge(
                (x, y), 5.5,
                np.degrees(ha) - 22.5,
                np.degrees(ha) + 22.5,
                color=head_color, alpha=0.5, zorder=3,
            )
            ax.add_patch(wedge)

        # Jersey number
        ax.text(x, y, str(j), color=WHITE, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

    # --- Ball ---
    if ball_xy is not None and not (np.isnan(ball_xy[0]) or np.isnan(ball_xy[1])):
        ax.plot(ball_xy[0], ball_xy[1], "o", ms=10, color=BALL_C,
                markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    if title:
        ax.set_title(title, color=WHITE, fontsize=14, fontweight="bold", pad=12)

    if save_path:
        fig.savefig(save_path, dpi=400, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")

    return fig
