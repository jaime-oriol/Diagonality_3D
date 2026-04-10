"""
dos_plot — Render Diagonal Opportunity Surfaces on pitch.

Same aesthetic as ppcf_plot (dark BG, Opta player markers) but with the
DOS_CMAP: dark = no diagonal advantage, warm = high diagonal opportunity.

The DOS surface is gated by the on-ball player's scanning memory
(FOV + 2.5s exponentially decayed history) so only cells the player can
SEE or has scanned recently are painted. A smoothstep visibility curve
maps the gated DOS onto a fixed display range — no per-frame
renormalization, no on/off flicker.

The caller is expected to pass:
  - dos_surface: pre-computed DOS grid (typically already EMA-smoothed
    and gaussian-blurred by the render loop)
  - scanning_memory: same shape as dos_surface, in [0, 1], encoding what
    the current on-ball player has seen
  - noise_floor / display_max: absolute bounds (NOT per-frame), so the
    color scale is comparable across the whole video.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge
from mplsoccer import Pitch

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
    dos_surface: np.ndarray,
    scanning_memory: np.ndarray,
    gk_jerseys: dict = None,
    alpha_max: float = 0.9,
    title: str = "",
    ax: plt.Axes = None,
    save_path: str = None,
    figsize: tuple = (16, 10.4),
    noise_floor: float = 0.0005,
    display_max: float = 0.015,
) -> plt.Figure:
    """Render one frame with DOS heatmap + players + ball.

    Args:
        orientations_frame: Single-frame orientations slice.
        attacking_team: Team ID.
        ball_xy: Ball position (x, y) in meters.
        attacking_right: True if attacking toward +x.
        dos_surface: (n_grid_y, n_grid_x) DOS grid. Typically the EMA'd
            and gaussian-blurred output of the render loop, NOT the raw
            output of compute_dos_surface.
        scanning_memory: (n_grid_y, n_grid_x) on-ball scanning memory in
            [0, 1], already resampled to the DOS grid. The DOS is
            multiplied by this mask before being mapped to color.
        gk_jerseys: {team_id: jersey} for GK markers. Default {0:1, 1:1}.
        alpha_max: Peak alpha for cells at `display_max` DOS.
        title: Optional title.
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

    if scanning_memory.shape != dos_surface.shape:
        raise ValueError(
            f"scanning_memory shape {scanning_memory.shape} does not "
            f"match DOS surface shape {dos_surface.shape}"
        )

    # Figure / axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # ── Gate + smoothstep visibility curve ───────────────────────────
    # Multiply DOS by the on-ball scanning memory to keep only cells the
    # player can see / has scanned recently. Map onto [0, 1] via a C^1
    # smoothstep over [noise_floor, display_max] so there is no on/off
    # cliff and the color scale is fixed cross-frame.
    dos_pos = np.clip(dos_surface, 0.0, None).astype(np.float32)
    dos_gated = dos_pos * scanning_memory.astype(np.float32)
    edge0 = float(noise_floor)
    edge1 = float(max(display_max, edge0 + 1e-9))
    t = np.clip((dos_gated - edge0) / (edge1 - edge0), 0.0, 1.0)
    dos_norm = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    rgba = DOS_CMAP(dos_norm)
    rgba[..., 3] = dos_norm * alpha_max

    ax.imshow(
        rgba, origin="lower",
        extent=[-52.5, 52.5, -34.0, 34.0],
        interpolation="spline36",
        zorder=1, aspect="auto",
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
