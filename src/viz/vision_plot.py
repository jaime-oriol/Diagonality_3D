"""vision_plot — Render Bekkers vision map on pitch (LIGHT OPTA).

Mecanica Bekkers (FOV + occlusion + gaussianas) sobre identidad CCV
(white BG, marcadores Opta, balon Telstar). El cmap del overlay es:
color del rival -> blanco -> color del jugador focus. Fuera del FOV =
transparente (el campo blanco pasa por debajo).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from ..vision import compute_player_vision
from .common import (
    BG, TEXT, PE_S, PKW,
    ATT as ATT_C, ATT_LIGHT, DEF as DEF_C, DEF_LIGHT, GK as GK_C,
    draw_football,
)

MS = 20


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
    """Render vision map for one focus player sobre BG blanco."""
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}
    fo = orientations_frame
    focus_row = fo[(fo["team"] == focus_team) & (fo["jersey"] == focus_jersey)]
    if len(focus_row) == 0:
        raise ValueError(f"Player team={focus_team} jersey={focus_jersey} not found")
    focus = focus_row.iloc[0]

    others = fo[fo.index != focus.name]

    other_sw = others["shoulder_width"].values if "shoulder_width" in others.columns else None
    if other_sw is not None:
        other_sw = other_sw * smoothing

    grid = compute_player_vision(
        focus["x"], focus["y"], focus["head_angle"],
        focus.get("speed", 0.0) if not np.isnan(focus.get("speed", 0.0)) else 0.0,
        others["x"].values, others["y"].values,
        others["shoulder_angle"].values,
        other_shoulder_widths=other_sw,
        smoothing=smoothing,
    )

    # Cmap: rival oscuro -> blanco -> focus oscuro.
    # Saturados pa que se vean sobre el BG blanco del pitch.
    DEEP_ATT = "#1d4ed8"
    DEEP_DEF = "#b91c1c"
    focus_deep = DEEP_DEF if focus_team != att_team else DEEP_ATT
    rival_deep = DEEP_ATT if focus_team != att_team else DEEP_DEF
    vision_cmap = mcolors.LinearSegmentedColormap.from_list(
        "vis", [rival_deep, "#ffffff", focus_deep]
    )
    vision_norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # Vision overlay: alpha tracks vision -> fuera del FOV = transparente.
    mapped = vision_cmap(vision_norm(grid))
    mapped[:, :, 3] = np.clip(grid * 0.55, 0, 0.55)

    ax.imshow(mapped, origin="lower",
              extent=[-52.5, 52.5, -34, 34],
              interpolation="bilinear",
              zorder=1, aspect="auto")

    for _, p in fo.iterrows():
        t_id = int(p["team"]); j = int(p["jersey"])
        x, y = p["x"], p["y"]
        is_gk = (j == gk_jerseys.get(t_id, 1))
        color = GK_C if is_gk else (ATT_C if t_id == att_team else DEF_C)
        is_focus = (t_id == focus_team and j == focus_jersey)

        ax.plot(x, y, "o", ms=MS + (4 if is_focus else 0), color=color,
                markeredgecolor=TEXT, markeredgewidth=1.5 if is_focus else 1.0,
                alpha=0.95, zorder=5)

        if not is_gk and not np.isnan(p.get("shoulder_angle", np.nan)):
            sw = p.get("shoulder_width", 0.45) * 9
            perp_l = p["shoulder_angle"] + np.pi / 2
            perp_r = p["shoulder_angle"] - np.pi / 2
            ax.plot([x + (sw/2)*np.cos(perp_l), x + (sw/2)*np.cos(perp_r)],
                    [y + (sw/2)*np.sin(perp_l), y + (sw/2)*np.sin(perp_r)],
                    color=color, linewidth=8, alpha=0.75, zorder=4,
                    solid_capstyle="round")

        if not is_gk and not np.isnan(p.get("head_angle", np.nan)):
            head_color = ATT_LIGHT if t_id == att_team else DEF_LIGHT
            r = 7 if is_focus else 5.5
            wedge = Wedge((x, y), r,
                          np.degrees(p["head_angle"]) - 22.5,
                          np.degrees(p["head_angle"]) + 22.5,
                          color=head_color, alpha=0.55, zorder=3)
            ax.add_patch(wedge)

        ax.text(x, y, str(j), color="white", fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

    if ball_x is not None and ball_y is not None:
        draw_football(ax, ball_x, ball_y, r=0.9, zorder=10)

    if title:
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold", pad=12)

    if save_path:
        fig.savefig(save_path, dpi=400, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig
