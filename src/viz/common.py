"""Shared style, constants, and helper functions for all viz modules.

Adapted from Opta Forum 2026 viz style to TRACAB coordinate system.
Pitch: 105x68m, origin at center (0,0). Coordinates in meters.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Wedge, Ellipse
from mplsoccer import Pitch

# --- Style ----------------------------------------------------------------

BG = "#313332"
WHITE = "white"
FONT = "DejaVu Sans"

plt.style.use("default")
plt.rcParams.update({
    "font.family": FONT, "font.size": 10,
    "figure.dpi": 100, "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.facecolor": BG, "figure.facecolor": BG,
    "text.color": WHITE, "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
})

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# --- Colors ---------------------------------------------------------------

ATT = "deepskyblue"
DEF = "tomato"
GK = "black"
BALL = WHITE
VISION = "#00ff88"           # vision cones
BLIND = "#ff3333"            # blind spots
DIAGONAL = "gold"            # diagonal pass/action highlight
THETA_HIGH = "#ff6600"       # high theta (exploitable)
THETA_LOW = "#66ccff"        # low theta (covered)

# Pitch control colormap: DEF red → neutral → ATT blue
PPCF_CMAP = LinearSegmentedColormap.from_list("ppcf", ["#8B0000", "#777777", "#004D98"])

# Vision colormap: invisible → visible (dark → green)
VISION_CMAP = LinearSegmentedColormap.from_list("vision", ["#000000", "#003311", "#00ff88"])

# DOS colormap: no opportunity → high opportunity
DOS_CMAP = LinearSegmentedColormap.from_list("dos", ["#1a1a2e", "#e63946", "#ffdd00"])

# --- Pitch ----------------------------------------------------------------

# TRACAB pitch: 105x68m, center at (0,0)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2  # 52.5
HALF_WIDTH = PITCH_WIDTH / 2    # 34.0

PKW = dict(
    pitch_type="secondspectrum",
    pitch_length=PITCH_LENGTH,
    pitch_width=PITCH_WIDTH,
    pitch_color=BG,
    line_color=WHITE,
    line_zorder=2,
    linewidth=1,
)

# Path effects for text legibility on any background
PE = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
MS = 12  # marker size for players

# --- Figure helpers -------------------------------------------------------

def make_pitch(ax: Optional[plt.Axes] = None,
               figsize: Tuple[float, float] = (16, 10.4)) -> Tuple:
    """Create pitch figure. Returns (pitch, fig, ax)."""
    pitch = Pitch(**PKW)
    if ax is None:
        fig, ax = pitch.draw(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
        pitch.draw(ax=ax)
    return pitch, fig, ax


def save_fig(fig: plt.Figure, name: str, subdir: str = ""):
    """Save figure to FIGURES_DIR/{subdir}/{name}.png."""
    if not name:
        return
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"{name}.png", dpi=400,
        bbox_inches="tight", facecolor=fig.get_facecolor(),
    )


# --- Player plotting ------------------------------------------------------

def plot_players(
    ax: plt.Axes,
    orientations: pd.DataFrame,
    frame: int,
    attacking_team: int,
    gk_jerseys: Optional[dict] = None,
    show_velocity: bool = True,
    show_jersey: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Plot all players with correct colors. Returns (att_df, def_df).

    Args:
        orientations: DataFrame with x, y, team, jersey, vx, vy columns
        frame: frame_number to plot
        attacking_team: 0 or 1
        gk_jerseys: {team: jersey} for GK identification. If None, uses jersey=1.
        show_velocity: draw velocity quiver arrows
        show_jersey: draw jersey numbers
    """
    f = orientations[orientations["frame_number"] == frame]
    if len(f) == 0:
        return pd.DataFrame(), pd.DataFrame()

    defending_team = 1 - attacking_team
    att = f[f["team"] == attacking_team]
    def_ = f[f["team"] == defending_team]

    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    for team_df, color, team_id in [(def_, DEF, defending_team), (att, ATT, attacking_team)]:
        if team_df.empty:
            continue
        gk_j = gk_jerseys.get(team_id, 1)
        outfield = team_df[team_df["jersey"] != gk_j]
        gk = team_df[team_df["jersey"] == gk_j]

        # Outfield players
        if not outfield.empty:
            ax.plot(outfield["x"].values, outfield["y"].values,
                    "o", ms=MS, color=color, markeredgecolor=WHITE,
                    markeredgewidth=1, alpha=0.85, zorder=4, linestyle="None")

        # GK
        if not gk.empty:
            ax.plot(gk["x"].values, gk["y"].values,
                    "o", ms=MS, color=GK, markeredgecolor=WHITE,
                    markeredgewidth=1, alpha=0.85, zorder=4, linestyle="None")

        # Velocity arrows
        if show_velocity and "vx" in team_df.columns:
            valid = team_df.dropna(subset=["vx", "vy"])
            if not valid.empty:
                ax.quiver(valid["x"].values, valid["y"].values,
                          valid["vx"].values, valid["vy"].values,
                          color=color, scale=120, scale_units="width",
                          width=0.003, headwidth=3.5, headlength=4,
                          headaxislength=3.5, alpha=0.55, zorder=3)

        # Jersey numbers
        if show_jersey:
            for _, p in team_df.iterrows():
                label_text(ax, p["x"], p["y"] + 2.0, str(int(p["jersey"])),
                           fontsize=7, color=WHITE)

    return att, def_


def plot_ball(ax: plt.Axes, ball_df: pd.DataFrame, frame: int):
    """Plot ball position."""
    bf = ball_df[ball_df["frame_number"] == frame]
    if len(bf) > 0 and bf.iloc[0].get("ball_exists", True):
        ax.plot(bf.iloc[0]["x"], bf.iloc[0]["y"],
                "o", ms=8, color=BALL, markeredgecolor="black",
                markeredgewidth=0.8, zorder=10)


# --- Text helpers ---------------------------------------------------------

def label_text(ax: plt.Axes, x: float, y: float, text: str,
               fontsize: int = 7, color: str = WHITE, **kw):
    """Place a text label with stroke path effect for legibility."""
    ax.text(x, y, text, fontsize=fontsize, ha="center", color=color,
            fontweight="bold", path_effects=PE, zorder=8, **kw)


def info_panel(fig: plt.Figure, lines: list, x: float = 0.02, y_start: float = 0.95,
               fontsize: int = 11, line_spacing: float = 0.04):
    """Draw an info panel with multiple text lines on the figure."""
    for i, (text, color) in enumerate(lines):
        fig.text(x, y_start - i * line_spacing, text,
                 ha="left", va="top", fontsize=fontsize,
                 color=color, fontweight="bold", fontfamily=FONT,
                 path_effects=PE)


# --- Vision overlay helpers -----------------------------------------------

def draw_vision_cone(ax: plt.Axes, x: float, y: float,
                     head_angle: float, radius: float = 8.0,
                     fov_deg: float = 120.0, color: str = VISION,
                     alpha: float = 0.12):
    """Draw a vision cone (wedge) on the pitch."""
    half_fov = fov_deg / 2
    theta1 = np.degrees(head_angle) - half_fov
    theta2 = np.degrees(head_angle) + half_fov
    wedge = Wedge((x, y), radius, theta1, theta2,
                  alpha=alpha, color=color, zorder=2)
    ax.add_patch(wedge)


def draw_orientation_arrow(ax: plt.Axes, x: float, y: float,
                           angle: float, length: float = 3.0,
                           color: str = WHITE, lw: float = 2.0,
                           style: str = "-|>"):
    """Draw a body orientation arrow from a player position."""
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=4)


def draw_pass_arrow(ax: plt.Axes, x1: float, y1: float,
                    x2: float, y2: float,
                    color: str = WHITE, lw: float = 2.5,
                    style: str = "-|>", alpha: float = 0.9):
    """Draw a pass trajectory arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                alpha=alpha),
                zorder=6)


# --- Legend ---------------------------------------------------------------

def draw_legend(fig: plt.Figure, elements: str = "basic", y_offset: float = 0.0):
    """Draw a legend at the bottom of the figure.

    Args:
        elements: 'basic' (players + ball) or 'full' (+ vision + theta + pass)
    """
    fig.subplots_adjust(bottom=0.18)
    tfig = fig.transFigure
    W, H = fig.get_size_inches()
    s = 10.4 / H  # scale factor for consistent spacing

    r = 0.13
    cw, ch = 2 * r / W, 2 * r / H
    y1 = 0.02 * s + y_offset
    y_sym = -0.01 * s + y_offset
    y_lbl = -0.04 * s + y_offset
    y2 = -0.09 * s + y_offset
    y2_sym = -0.12 * s + y_offset
    y2_lbl = -0.15 * s + y_offset

    gap = 0.08
    L = 0.07

    # Row 1: Players
    p0 = L + cw / 2
    fig.text((p0 + p0 + 2*gap) / 2, y1, "Players", ha="center",
             fontsize=14, color=WHITE, fontweight="bold", fontfamily=FONT)
    for i, (fc, lbl) in enumerate([(ATT, "Attacking"), (DEF, "Defending"), (GK, "GK")]):
        fx = p0 + i * gap
        fig.patches.append(Ellipse((fx, y_sym), cw, ch, transform=tfig,
                                    facecolor=fc, edgecolor=WHITE, linewidth=1.2,
                                    alpha=0.85, zorder=10))
        fig.text(fx, y_lbl, lbl, ha="center", fontsize=12, color=WHITE, fontfamily=FONT)

    if elements == "full":
        # Row 1 right: Vision
        vx = 0.38
        fig.text(vx + 0.04, y1, "Orientation", ha="center",
                 fontsize=14, color=WHITE, fontweight="bold", fontfamily=FONT)
        # Vision cone icon
        fig.patches.append(Wedge((vx, y_sym), r * 1.5 / W, -30, 30, transform=tfig,
                                  alpha=0.3, color=VISION, zorder=10))
        fig.text(vx, y_lbl, "FOV", ha="center", fontsize=12, color=WHITE, fontfamily=FONT)
        # Shoulder arrow icon
        arrow_s, arrow_e = vx + gap, vx + gap + 0.05
        fig.patches.append(FancyArrowPatch(
            (arrow_s, y_sym), (arrow_e, y_sym), transform=tfig,
            arrowstyle="-|>", color=ATT, lw=2.5, mutation_scale=13, zorder=10))
        fig.text((arrow_s + arrow_e) / 2, y_lbl, "Body facing",
                 ha="center", fontsize=12, color=WHITE, fontfamily=FONT)

        # Row 2: Actions + Theta
        fig.text((p0 + p0 + 2*gap) / 2, y2, "Actions", ha="center",
                 fontsize=14, color=WHITE, fontweight="bold", fontfamily=FONT)
        a0s, a0e = L, L + 0.06
        fig.patches.append(FancyArrowPatch(
            (a0s, y2_sym), (a0e, y2_sym), transform=tfig,
            arrowstyle="-|>", color=WHITE, lw=2.5, mutation_scale=15, zorder=10))
        fig.text((a0s + a0e) / 2, y2_lbl, "Pass", ha="center",
                 fontsize=12, color=WHITE, fontfamily=FONT)
        a1s = a0e + 0.03
        a1e = a1s + 0.06
        fig.patches.append(FancyArrowPatch(
            (a1s, y2_sym), (a1e, y2_sym), transform=tfig,
            arrowstyle="-|>", color=DIAGONAL, lw=2.5, mutation_scale=15, zorder=10))
        fig.text((a1s + a1e) / 2, y2_lbl, "Diagonal", ha="center",
                 fontsize=12, color=WHITE, fontfamily=FONT)

        # Theta info
        fig.text(vx, y2, "Theta", ha="center",
                 fontsize=14, color=THETA_HIGH, fontweight="bold", fontfamily=FONT)
        fig.patches.append(Ellipse((vx, y2_sym), cw, ch, transform=tfig,
                                    facecolor="none", edgecolor=THETA_HIGH,
                                    linewidth=2, zorder=10))
        fig.text(vx, y2_lbl, "Blind spot", ha="center",
                 fontsize=12, color=WHITE, fontfamily=FONT)
        fig.patches.append(Ellipse((vx + gap, y2_sym), cw, ch, transform=tfig,
                                    facecolor="none", edgecolor=THETA_LOW,
                                    linewidth=2, zorder=10))
        fig.text(vx + gap, y2_lbl, "Covered", ha="center",
                 fontsize=12, color=WHITE, fontfamily=FONT)
