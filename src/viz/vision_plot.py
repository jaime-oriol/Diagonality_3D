"""
vision_plot — Render vision maps on pitch.

Bekkers mechanics (FOV, occlusion, Gaussians) with Opta Forum aesthetics
(dark BG, bright colors, Opta-style player markers).

Vision cmap: focus team color -> white -> rival color.
Outside FOV = transparent (shows pitch BG normally).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from ..vision import compute_player_vision

# ── Style: shared base from common.py + vision-specific colors ──

from .common import BG, WHITE, FONT, PKW

from .common import ATT as ATT_C, DEF as DEF_C, GK as GK_C, BALL as BALL_C
PE_S = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 17


def plot_vision_frame(
    orientations_frame: pd.DataFrame,
    focus_team: int,
    focus_jersey: int,
    ball_x: float = None,
    ball_y: float = None,
    title: str = "",
    save_path: str = None,
    smoothing: float = 7.0,
    figsize: tuple = (16, 10.4),
    att_team: int = 1,
    gk_jerseys: dict = None,
) -> plt.Figure:
    """Render vision map for one focus player on the pitch.

    Args:
        gk_jerseys: {team_id: jersey} for GK identification. Default {0:1, 1:1}.
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}
    fo = orientations_frame
    focus_row = fo[(fo["team"] == focus_team) & (fo["jersey"] == focus_jersey)]
    if len(focus_row) == 0:
        raise ValueError(f"Player team={focus_team} jersey={focus_jersey} not found")
    focus = focus_row.iloc[0]

    others = fo[fo.index != focus.name]

    # Compute vision grid (Bekkers exact, high res)
    grid = compute_player_vision(
        focus["x"], focus["y"], focus["head_angle"],
        focus.get("speed", 0.0) if not np.isnan(focus.get("speed", 0.0)) else 0.0,
        others["x"].values, others["y"].values,
        others["shoulder_angle"].values,
        shoulder_width=focus.get("shoulder_width", 0.45),
        smoothing=smoothing,
    )

    # Cmap: focus team color -> white -> rival color
    focus_color = DEF_C if focus_team != att_team else ATT_C
    rival_color = ATT_C if focus_team != att_team else DEF_C
    vision_cmap = mcolors.LinearSegmentedColormap.from_list(
        "vis", [rival_color, WHITE, focus_color]
    )
    vision_norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    # Build RGBA:
    # - Outside FOV (grid==0) -> fully transparent (shows pitch BG)
    # - Inside FOV, high vision -> focus team color, opaque
    # - Inside FOV, occlusion shadow -> transparent (shows pitch BG = "black shadow")
    mapped = vision_cmap(vision_norm(grid))  # (H, W, 4) RGBA
    # Alpha: proportional to vision value INSIDE the FOV
    # grid>0.01 means inside FOV. Higher vision = more opaque color.
    # Occlusion drops vision toward 0 inside FOV = alpha drops = black BG shows through = shadow
    inside_fov = grid > 0.005
    mapped[:, :, 3] = np.where(inside_fov, np.clip(grid * 0.7, 0.05, 0.65), 0.0)

    # ── Render ──
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(BG)
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # Vision overlay — transparent outside FOV
    ax.imshow(mapped, origin="lower",
              extent=[-52.5, 52.5, -34, 34],
              interpolation="bilinear",
              zorder=1, aspect="auto")

    # Players (Opta Forum style)
    for _, p in fo.iterrows():
        t = int(p["team"]); j = int(p["jersey"])
        x, y = p["x"], p["y"]
        is_gk = (j == gk_jerseys.get(t, 1))
        color = GK_C if is_gk else (ATT_C if t == att_team else DEF_C)
        is_focus = (t == focus_team and j == focus_jersey)

        # Dot (Opta Forum: "o" with edge)
        ax.plot(x, y, "o", ms=MS + (4 if is_focus else 0), color=color,
                markeredgecolor=WHITE, markeredgewidth=1.5 if is_focus else 1,
                alpha=0.85, zorder=5)

        # Shoulder bar (wide, visible — shows body orientation distinctly from head)
        if not is_gk and not np.isnan(p.get("shoulder_angle", np.nan)):
            sw = p.get("shoulder_width", 0.45) * 7  # big scale for visibility
            perp_l = p["shoulder_angle"] + np.pi / 2
            perp_r = p["shoulder_angle"] - np.pi / 2
            ax.plot([x + (sw/2)*np.cos(perp_l), x + (sw/2)*np.cos(perp_r)],
                    [y + (sw/2)*np.sin(perp_l), y + (sw/2)*np.sin(perp_r)],
                    color=color, linewidth=5, alpha=0.85, zorder=4,
                    solid_capstyle="round")

        # Head wedge (gaze direction — distinct from shoulder bar)
        if not is_gk and not np.isnan(p.get("head_angle", np.nan)):
            r = 6 if is_focus else 4.5
            wedge = Wedge((x, y), r,
                          np.degrees(p["head_angle"]) - 22.5,
                          np.degrees(p["head_angle"]) + 22.5,
                          color=color, alpha=0.5, zorder=3)
            ax.add_patch(wedge)

        # Jersey number
        ax.text(x, y, str(j), color=WHITE, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

    # Ball
    if ball_x is not None and ball_y is not None:
        ax.plot(ball_x, ball_y, "o", ms=8, color=BALL_C,
                markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    if title:
        ax.set_title(title, color=WHITE, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig
