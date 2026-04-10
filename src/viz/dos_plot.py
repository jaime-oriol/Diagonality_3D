"""
dos_plot — Render Diagonal Opportunity Surfaces on pitch.

Same aesthetic as ppcf_plot (dark BG, Opta player markers) but with the
DOS_CMAP: dark = no diagonal advantage, warm = high diagonal opportunity.

Rendering semantics:
    - color = DOS value mapped through DOS_CMAP (dark -> red -> yellow)
    - alpha = |DOS| clipped and scaled (cells with zero DOS fade to BG)
    - optional arrows showing the best diagonal direction per zone

Two gating modes:
  1. Legacy (default): attacker-radius mask + behind-ball mask + 20%
     relative threshold. Heuristic. Causes flicker because thresholds
     float with the per-frame max.
  2. Scanning gate (preferred): pass `scanning_memory` (already
     resampled to the DOS grid). DOS is multiplied by it (FOV + 2.5s
     decayed memory of the on-ball player), then clipped against an
     absolute threshold and an absolute display max — both fixed across
     frames so colors are stable and the only animation comes from real
     model dynamics, not from renormalization.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from ..dos import compute_dos_surface
from ..ppcf import default_params
from .common import (
    BG, WHITE, PKW, DOS_CMAP,
    ATT as ATT_C, DEF as DEF_C,
    GK as GK_C, BALL as BALL_C,
)

PE_S = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 20


def plot_dos_frame(
    orientations_frame: pd.DataFrame,
    attacking_team: int,
    ball_xy: tuple,
    attacking_right: bool,
    gk_jerseys: dict = None,
    dos_surface: np.ndarray = None,
    best_direction: np.ndarray = None,
    n_grid_x: int = 50,
    n_directions: int = 24,
    alpha_max: float = 0.9,
    show_arrows: bool = False,
    arrow_grid: int = 5,
    title: str = "",
    ax: plt.Axes = None,
    save_path: str = None,
    figsize: tuple = (16, 10.4),
    vision_smoothing: float = 3.0,
    scanning_memory: Optional[np.ndarray] = None,
    noise_floor: float = 0.0005,
    display_max: float = 0.015,
) -> plt.Figure:
    """Render one frame with DOS heatmap + players + ball + direction arrows.

    Args:
        orientations_frame: Single-frame orientations slice.
        attacking_team: Team ID.
        ball_xy: Ball position (x, y) in meters.
        attacking_right: True if attacking toward +x.
        gk_jerseys: {team_id: jersey} for GK markers.
        dos_surface: Pre-computed DOS from compute_dos_surface(). If None,
            computed inside.
        best_direction: Pre-computed best diagonal direction per cell
            (radians). If None, computed with dos_surface.
        n_grid_x: Grid resolution. Default 50.
        n_directions: Candidate delivery directions. Default 24 (every 15 deg).
        alpha_max: Peak alpha for cells at `display_max` DOS.
        show_arrows: Draw arrows showing best diagonal direction. Default False.
        arrow_grid: Subsample arrows every N cells. Default 5.
        title: Optional title.
        scanning_memory: Optional (n_grid_y, n_grid_x) on-ball scanning
            memory grid in [0, 1], already resampled to the DOS grid.
            If provided, DOS is gated by this mask and the absolute
            thresholds are used (no flicker). If None, falls back to
            the legacy attacker-radius / behind-ball / 20%-relative
            heuristic gating.
        noise_floor: Lower edge of the smoothstep visibility curve. Cells
            with gated DOS at or below this fade to transparent smoothly
            (no hard cliff -> no on/off flicker). Default 0.0005, tuned
            from real Kane goal probe frames.
        display_max: Upper edge of the smoothstep visibility curve. Cells
            with gated DOS at or above this saturate to alpha_max. Fixed
            across frames so the color scale is stable. Default 0.015
            (~ P95 of gated DOS values observed at the goal sequence).
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    # Compute DOS if not provided
    if dos_surface is None or best_direction is None:
        dos_surface, _, best_direction, xgrid, ygrid = compute_dos_surface(
            orientations_frame, attacking_team, ball_xy, attacking_right,
            params=default_params(), n_grid_x=n_grid_x,
            n_directions=n_directions, vision_smoothing=vision_smoothing,
        )
    else:
        n_y, n_x = dos_surface.shape
        dx = 105.0 / n_x
        dy = 68.0 / n_y
        xgrid = np.arange(n_x) * dx - 52.5 + dx / 2.0
        ygrid = np.arange(n_y) * dy - 34.0 + dy / 2.0

    # Figure / axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # DOS heatmap.
    dos_pos = np.clip(dos_surface, 0.0, None).astype(np.float32)
    xx_grid, yy_grid = np.meshgrid(xgrid, ygrid)

    if scanning_memory is not None:
        # ── Principled gate: on-ball FOV + 2.5s decayed memory ──
        # The scanning_memory grid is assumed to already match the DOS
        # grid shape. It encodes "what the current on-ball player has
        # seen / is seeing", in [0, 1]. Multiplying gives an
        # actionability-weighted DOS surface.
        if scanning_memory.shape != dos_pos.shape:
            raise ValueError(
                f"scanning_memory shape {scanning_memory.shape} does not "
                f"match DOS surface shape {dos_pos.shape}"
            )
        dos_gated = dos_pos * scanning_memory.astype(np.float32)
        # Smoothstep visibility curve [noise_floor, display_max]:
        # values <= noise_floor fade to 0, values >= display_max saturate
        # to 1, transition is C^1 continuous (no on/off flicker).
        edge0 = float(noise_floor)
        edge1 = float(max(display_max, edge0 + 1e-9))
        t = np.clip((dos_gated - edge0) / (edge1 - edge0), 0.0, 1.0)
        dos_norm = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    else:
        # ── Legacy heuristic gate (back-compat) ──
        dos_max = max(dos_pos.max(), 1e-9)
        dos_norm = np.clip(dos_pos / dos_max, 0.0, 1.0)

        att_df = orientations_frame[orientations_frame["team"] == attacking_team]
        if len(att_df) > 0:
            att_mask = np.zeros_like(dos_norm, dtype=bool)
            attacker_radius = 12.0
            for _, a in att_df.iterrows():
                dist = np.sqrt((xx_grid - float(a["x"]))**2 + (yy_grid - float(a["y"]))**2)
                att_mask |= (dist < attacker_radius)
            dos_norm[~att_mask] = 0.0

        if ball_xy is not None:
            behind_limit = 20.0
            if attacking_right:
                dos_norm[xx_grid < (ball_xy[0] - behind_limit)] = 0.0
            else:
                dos_norm[xx_grid > (ball_xy[0] + behind_limit)] = 0.0

        dos_norm[dos_norm < 0.2] = 0.0

    rgba = DOS_CMAP(dos_norm)
    rgba[..., 3] = dos_norm * alpha_max

    ax.imshow(
        rgba, origin="lower",
        extent=[-52.5, 52.5, -34.0, 34.0],
        interpolation="spline36",
        zorder=1, aspect="auto",
    )

    # --- Direction arrows (subsampled grid) ---
    if show_arrows and best_direction is not None:
        xx, yy = np.meshgrid(xgrid, ygrid)
        # Subsample
        step_y = max(1, len(ygrid) // arrow_grid)
        step_x = max(1, len(xgrid) // arrow_grid)
        sy = slice(step_y // 2, None, step_y)
        sx = slice(step_x // 2, None, step_x)

        ax_x = xx[sy, sx].ravel()
        ax_y = yy[sy, sx].ravel()
        ax_d = best_direction[sy, sx].ravel()
        ax_dos = dos_pos[sy, sx].ravel()

        # Only draw arrows where DOS > 10% of max
        mask = ax_dos > dos_max * 0.1
        if mask.any():
            u = np.cos(ax_d[mask])
            v = np.sin(ax_d[mask])
            ax.quiver(
                ax_x[mask], ax_y[mask], u, v,
                color="#ffdd00", scale=25, scale_units="width",
                width=0.004, headwidth=4, headlength=5,
                alpha=0.7, zorder=4,
            )

    # --- Players ---
    for _, p in orientations_frame.iterrows():
        t = int(p["team"])
        j = int(p["jersey"])
        x, y = float(p["x"]), float(p["y"])
        is_gk = (j == gk_jerseys.get(t, 1))
        color = GK_C if is_gk else (ATT_C if t == attacking_team else DEF_C)

        ax.plot(x, y, "o", ms=MS, color=color,
                markeredgecolor=WHITE, markeredgewidth=1,
                alpha=0.85, zorder=5)

        # Shoulder bar
        sa = p.get("shoulder_angle", np.nan)
        if not (isinstance(sa, float) and np.isnan(sa)):
            sw = float(p.get("shoulder_width", 0.45) or 0.45) * 9
            perp_l = sa + np.pi / 2
            perp_r = sa - np.pi / 2
            bar_color = WHITE if is_gk else color
            ax.plot(
                [x + (sw / 2) * np.cos(perp_l), x + (sw / 2) * np.cos(perp_r)],
                [y + (sw / 2) * np.sin(perp_l), y + (sw / 2) * np.sin(perp_r)],
                color=bar_color, linewidth=8, alpha=0.7, zorder=4,
                solid_capstyle="round",
            )

        # Head wedge
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

    # --- Velocity arrows ---
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
