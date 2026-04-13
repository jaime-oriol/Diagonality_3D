"""
dos_plot — Render Diagonal Opportunity Surfaces on pitch.

Two-layer overlay with distinct cognitive semantics:

  1. VISIBLE DOS (DOS_CMAP, cold cyan→magenta): opportunities the on-ball
     player is currently seeing or scanned within the last 2.5s.
     Gated by `dos * scanning_memory`.

  2. SHADOW DOS (SHADOW_CMAP, warm amber→gold): opportunities the player
     does NOT see but that lie ahead of the ball inside realistic pass
     or carry range. Gated by `dos * (1 - scanning_memory) * forward_cone`.
     A shadowpass (Hamilton manifesto) is a pass that actually lands
     inside a gold cell.

Both layers use the same smoothstep visibility curve `t^2*(3-2t)` over
absolute `[noise_floor, display_max]` bounds, so the colour scale is
fixed across frames — no per-frame renormalization, no on/off flicker.

The caller is expected to pre-process both surfaces (blur + EMA) per
frame and pass them in. The visible layer takes `dos_surface` + a
`scanning_memory` mask (use all-ones if the caller has already gated).
The shadow layer is optional; pass `shadow_surface` already masked by
`compute_forward_cone_mask` to restrict it to the on-ball player's
offensive arc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from .common import (
    BG, WHITE, PKW, DOS_CMAP, SHADOW_CMAP,
    ATT as ATT_C, DEF as DEF_C,
    GK as GK_C, BALL as BALL_C,
)

PE_S = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 20


def compute_forward_cone_mask(
    ball_xy: tuple,
    attacking_right: bool,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    max_dist_m: float = 35.0,
) -> np.ndarray:
    """Boolean mask (len(ygrid), len(xgrid)) of cells that:
      - are ahead of the ball along the attacking axis
      - lie within `max_dist_m` of the ball (realistic pass / carry range)

    Used to gate the shadow-DOS layer so it only appears in the on-ball
    player's offensive arc, never in the defensive half (avoids distracting
    low-DOS noise behind the ball).
    """
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    xx, yy = np.meshgrid(xgrid, ygrid)
    forward_sign = 1.0 if attacking_right else -1.0
    ahead = (xx - bx) * forward_sign > 0.0
    within_range = np.hypot(xx - bx, yy - by) <= float(max_dist_m)
    return (ahead & within_range).astype(np.float32)


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
    shadow_surface: np.ndarray = None,
    shadow_noise_floor: float = 0.003,
    shadow_display_max: float = 0.025,
    shadow_alpha_max: float = 0.55,
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
        shadow_surface: Optional (n_grid_y, n_grid_x) surface holding the
            "unseen but dangerous" DOS component. Should already be
            ball-forward-cone-masked by the caller (see
            `compute_forward_cone_mask`). If None, no shadow layer is
            rendered. Semantics: `shadow = dos * (1 - memory) * forward_cone`,
            i.e. diagonal opportunities the on-ball player does NOT see
            but that lie ahead of the ball in realistic pass / carry range.
            A Hamilton shadowpass is a pass that actually lands inside a
            cell lit up by this layer.
        shadow_noise_floor: Strict lower edge of the shadow smoothstep.
            Default 0.003 = 6x the visible noise_floor, tuned to suppress
            low-DOS noise in zones the player simply hasn't scanned yet.
        shadow_display_max: Upper edge of the shadow smoothstep. Default
            0.025 (~P95 of the top shadow DOS values observed).
        shadow_alpha_max: Peak alpha for the shadow layer. Default 0.55,
            lower than the visible layer so shadow reads clearly as
            secondary ("what the player could exploit if he scanned").
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    if scanning_memory.shape != dos_surface.shape:
        raise ValueError(
            f"scanning_memory shape {scanning_memory.shape} does not "
            f"match DOS surface shape {dos_surface.shape}"
        )
    if shadow_surface is not None and shadow_surface.shape != dos_surface.shape:
        raise ValueError(
            f"shadow_surface shape {shadow_surface.shape} does not "
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

    # ── Shadow layer (optional) ──────────────────────────────────────
    # Rendered UNDER the visible DOS (lower z) so visible wins in any
    # edge-case overlap, although by construction shadow only lives where
    # the player does NOT see + is ahead of the ball inside pass range.
    if shadow_surface is not None:
        sh_pos = np.clip(shadow_surface, 0.0, None).astype(np.float32)
        sh_edge0 = float(shadow_noise_floor)
        sh_edge1 = float(max(shadow_display_max, sh_edge0 + 1e-9))
        ts = np.clip((sh_pos - sh_edge0) / (sh_edge1 - sh_edge0), 0.0, 1.0)
        sh_norm = (ts * ts * (3.0 - 2.0 * ts)).astype(np.float32)
        sh_rgba = SHADOW_CMAP(sh_norm)
        sh_rgba[..., 3] = sh_norm * shadow_alpha_max
        ax.imshow(
            sh_rgba, origin="lower",
            extent=[-52.5, 52.5, -34.0, 34.0],
            interpolation="spline36",
            zorder=0, aspect="auto",
        )

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
