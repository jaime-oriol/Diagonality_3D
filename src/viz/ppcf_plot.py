"""
ppcf_plot — Render Immediate Orientation-Aware PPCF on pitch.

Opta Forum aesthetic (dark BG, blue/red divergent cmap, Opta player markers)
paired with the reach-field PPCF: each player contributes an anisotropic
Gaussian blob whose sigma is derived from orientation-aware reach (Vater RT
+ Dos'Santos COD applied to the real shoulder angle). The blobs are summed
per team and converted into ppcf shares.

Rendering semantics:
    - color comes from which team dominates the cell (ratio of att to total)
    - alpha comes from the resolved mass at the cell (both teams combined)
    - cells with zero influence from any player fade to the pitch BG
A defender's blob has a visible hole in their blind spot — the literal
image of the diagonality thesis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from ..ppcf import compute_ppcf_surfaces, default_params
from .common import (
    BG, WHITE, PKW, PPCF_CMAP,
    ATT as ATT_C, DEF as DEF_C,
    GK as GK_C, BALL as BALL_C,
)

PE_S = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 20  # Opta-style large markers for overlay contrast


def plot_ppcf_frame(
    orientations_frame: pd.DataFrame,
    attacking_team: int,
    ball_xy: tuple = None,
    gk_jerseys: dict = None,
    ppcf_att: np.ndarray = None,
    ppcf_def: np.ndarray = None,
    immediate_window: float = None,
    n_grid_x: int = 100,
    alpha_max: float = 1.0,
    title: str = "",
    ax: plt.Axes = None,
    save_path: str = None,
    figsize: tuple = (16, 10.4),
) -> plt.Figure:
    """Render one frame with reach-field PPCF heatmap + players + ball.

    Alpha = (ppcf_att + ppcf_def) * alpha_max (the resolved mass at the
    cell). Cells with zero influence fade to the pitch BG.
    Color = ratio of attacker share to total resolved, mapped through
    PPCF_CMAP (red = defender dominant, gray = contested, blue = attacker
    dominant).

    Args:
        orientations_frame: Single-frame slice of an orientations DataFrame
            (from compute_orientations + add_dynamics). Must have columns:
            team, jersey, x, y, shoulder_angle. For nicer rendering also:
            head_angle, shoulder_width, vx, vy.
        attacking_team: Team ID of attacking team (0 or 1).
        ball_xy: Optional (x, y) ball position in meters for drawing the
            ball marker. NOT used by the PPCF computation (reach model
            is ball-independent; the ball position is implicit in each
            player's shoulder angle).
        gk_jerseys: {team_id: jersey} for drawing GKs in black. Default
            {0: 1, 1: 1}. Not used by the math.
        ppcf_att, ppcf_def: Pre-computed reach-field shares from
            compute_ppcf_surfaces(). If None, computed inside.
        immediate_window: Override integration horizon (s). Default None
            = use default_params() (2.5s).
        n_grid_x: Grid resolution. Default 100 (~1m cells).
        alpha_max: Peak alpha for fully-resolved cells. Default 0.95.
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    # Compute surfaces if not provided
    if ppcf_att is None or ppcf_def is None:
        if immediate_window is not None:
            params = default_params(immediate_window=immediate_window)
        else:
            params = default_params()
        ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
            orientations_frame, attacking_team,
            params=params, n_grid_x=n_grid_x,
        )

    # Figure / axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # Resolved mass: how much control the reach-field has assigned to the
    # cell (0 = nobody reaches, ~1 = at least one player clearly dominates).
    # Linear alpha: intense center, natural fade at edges. No sqrt boost
    # (which makes everything uniformly faint). Higher alpha_max compensates.
    resolved = np.clip(ppcf_att + ppcf_def, 0.0, 1.0)
    alpha = resolved * alpha_max

    # Color: share of attacker over total resolved.
    # 0 = pure defender (red), 0.5 = contested (gray), 1 = pure attacker (blue).
    total = ppcf_att + ppcf_def
    with np.errstate(divide="ignore", invalid="ignore"):
        color_idx = np.where(total > 1e-9, ppcf_att / total, 0.5)

    rgba = PPCF_CMAP(np.clip(color_idx, 0.0, 1.0))
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

        # Head wedge (gaze direction) — white for all players so it stands
        # out against the team-colored PPCF blob underneath
        ha = p.get("head_angle", np.nan)
        if not (isinstance(ha, float) and np.isnan(ha)):
            wedge = Wedge(
                (x, y), 3.5,
                np.degrees(ha) - 22.5,
                np.degrees(ha) + 22.5,
                color=WHITE, alpha=0.45, zorder=3,
            )
            ax.add_patch(wedge)

        # Jersey number
        ax.text(x, y, str(j), color=WHITE, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

    # --- Velocity arrows (Opta Forum style quiver) ---
    if "vx" in orientations_frame.columns and "vy" in orientations_frame.columns:
        for team_id, team_color in [(attacking_team, ATT_C),
                                     (1 - attacking_team, DEF_C)]:
            tdf = orientations_frame[orientations_frame["team"] == team_id]
            valid = tdf.dropna(subset=["vx", "vy"])
            if not valid.empty:
                ax.quiver(
                    valid["x"].values, valid["y"].values,
                    valid["vx"].values, valid["vy"].values,
                    color=team_color, scale=120, scale_units="width",
                    width=0.003, headwidth=3.5, headlength=4,
                    headaxislength=3.5, alpha=0.55, zorder=3,
                )

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
