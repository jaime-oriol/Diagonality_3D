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

from .common import BG, WHITE, PKW
from .common import ATT as ATT_C, ATT_LIGHT, DEF as DEF_C, DEF_LIGHT, GK as GK_C, BALL as BALL_C
PE_S = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 20  # Opta-style large markers for overlay contrast


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
    ax: plt.Axes = None,
) -> plt.Figure:
    """Render vision map for one focus player on the pitch.

    Args:
        gk_jerseys: {team_id: jersey} for GK identification. Default {0:1, 1:1}.
        ax: existing axes to draw on (for animation). If None, creates new figure.
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}
    fo = orientations_frame
    focus_row = fo[(fo["team"] == focus_team) & (fo["jersey"] == focus_jersey)]
    if len(focus_row) == 0:
        raise ValueError(f"Player team={focus_team} jersey={focus_jersey} not found")
    focus = focus_row.iloc[0]

    others = fo[fo.index != focus.name]

    # Per-occluder shoulder widths — scaled for visual clarity in viz
    # (analytical model uses raw widths; viz inflates for visible shadows)
    other_sw = others["shoulder_width"].values if "shoulder_width" in others.columns else None
    if other_sw is not None:
        other_sw = other_sw * smoothing

    # Compute vision grid (Bekkers FOV + per-occluder occlusion, high res)
    grid = compute_player_vision(
        focus["x"], focus["y"], focus["head_angle"],
        focus.get("speed", 0.0) if not np.isnan(focus.get("speed", 0.0)) else 0.0,
        others["x"].values, others["y"].values,
        others["shoulder_angle"].values,
        other_shoulder_widths=other_sw,
        smoothing=smoothing,
    )

    # Cmap: deep/dark versions of team colors for the overlay backdrop
    # Rival deep -> white -> focus deep
    DEEP_ATT = "#1a6b8a"   # dark deepskyblue
    DEEP_DEF = "#8b2020"   # dark tomato
    focus_deep = DEEP_DEF if focus_team != att_team else DEEP_ATT
    rival_deep = DEEP_ATT if focus_team != att_team else DEEP_DEF
    vision_cmap = mcolors.LinearSegmentedColormap.from_list(
        "vis", [rival_deep, WHITE, focus_deep]
    )
    vision_norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    # ── Render ──
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # Vision overlay:
    # - Outside FOV (grid=0): transparent -> pitch BG shows
    # - Inside FOV, visible: focus color, alpha=0.5
    # - Inside FOV, occlusion shadow: alpha drops with vision -> BG shows through = dark shadow
    # No gaussian blur — keeps shadows sharp and clean
    mapped = vision_cmap(vision_norm(grid))
    # Alpha tracks vision: high vision = opaque color, low vision (shadow) = transparent = field BG
    mapped[:, :, 3] = np.clip(grid * 0.6, 0, 0.55)

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

        # Shoulder bar (body orientation — team color, slightly muted)
        if not is_gk and not np.isnan(p.get("shoulder_angle", np.nan)):
            sw = p.get("shoulder_width", 0.45) * 9
            perp_l = p["shoulder_angle"] + np.pi / 2
            perp_r = p["shoulder_angle"] - np.pi / 2
            ax.plot([x + (sw/2)*np.cos(perp_l), x + (sw/2)*np.cos(perp_r)],
                    [y + (sw/2)*np.sin(perp_l), y + (sw/2)*np.sin(perp_r)],
                    color=color, linewidth=8, alpha=0.7, zorder=4,
                    solid_capstyle="round")

        # Head wedge (gaze direction — lighter color, stands out)
        if not is_gk and not np.isnan(p.get("head_angle", np.nan)):
            head_color = ATT_LIGHT if t == att_team else DEF_LIGHT
            r = 7 if is_focus else 5.5
            wedge = Wedge((x, y), r,
                          np.degrees(p["head_angle"]) - 22.5,
                          np.degrees(p["head_angle"]) + 22.5,
                          color=head_color, alpha=0.5, zorder=3)
            ax.add_patch(wedge)

        # Jersey number
        ax.text(x, y, str(j), color=WHITE, fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

    # Ball
    if ball_x is not None and ball_y is not None:
        ax.plot(ball_x, ball_y, "o", ms=10, color=BALL_C,
                markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    if title:
        ax.set_title(title, color=WHITE, fontsize=14, fontweight="bold", pad=12)

    if save_path:
        fig.savefig(save_path, dpi=400, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig
