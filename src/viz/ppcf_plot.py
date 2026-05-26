"""ppcf_plot — Render Immediate Orientation-Aware PPCF on pitch (LIGHT OPTA).

Identidad CCV (white BG + ATT blue / DEF red + balon Telstar) sobre el
reach-field PPCF: cada jugador contribuye una gaussiana 2D anisotropica
cuyo sigma depende del reach orientation-aware (Vater RT + Dos'Santos COD
aplicados al shoulder_angle real del skeleton). Los blobs se suman por
equipo y se convierten en shares de control.

Render semantica:
    - color por equipo dominante (ratio atacante / total)
    - alpha por masa resuelta en la celda (atacante + defensor)
    - celdas sin influencia -> fade al BG blanco del pitch

El blob del defensor tiene un agujero visible en su blind spot — la
imagen literal de la tesis de la diagonalidad.

Dos modos de render segun el caller:
    - ax is None + title/subtitle dados: layout completo CCV-style
      (header con escudo + titulo + JO; pitch; footer con gradiente +
      direccion + equipos + balon Telstar). Para PNGs estaticos.
    - ax is not None (renderers de video): pinta SOLO en el axes del
      pitch (sin header / footer). Para animacion frame-by-frame.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from ..ppcf import compute_ppcf_surfaces, default_params
from .common import (
    BG, TEXT, PE_S, PKW, PPCF_CMAP,
    ATT as ATT_C, DEF as DEF_C, GK as GK_C,
    PITCH_LENGTH, PITCH_WIDTH,
    draw_header, draw_footer_legend, draw_football,
)

MS = 20


# ── Layout CCV-style pa el PNG estatico (pitch-aligned, sin margen lateral)
FIG_W            = 14.0
HEADER_H_IN      = 1.10
FOOTER_H_IN      = 1.30
PAD_TOP_IN       = 0.05
PAD_HDR_PITCH_IN = 0.05
PAD_PCH_FTR_IN   = 0.05
PAD_BOT_IN       = 0.10
PITCH_MARGIN_Y_M = 1.0
PITCH_MARGIN_X_M = 2.0

_DATA_RATIO = (PITCH_LENGTH + 2 * PITCH_MARGIN_X_M) / (PITCH_WIDTH + 2 * PITCH_MARGIN_Y_M)
PITCH_H_IN  = FIG_W / _DATA_RATIO
FIG_H = (PAD_TOP_IN + HEADER_H_IN + PAD_HDR_PITCH_IN
         + PITCH_H_IN + PAD_PCH_FTR_IN
         + FOOTER_H_IN + PAD_BOT_IN)
FIGSIZE_PNG = (FIG_W, FIG_H)


def _layout(figsize: tuple = FIGSIZE_PNG) -> dict:
    """Posiciones absolutas header/pitch/footer (frac fig)."""
    _, fh = figsize
    h_hdr = HEADER_H_IN      / fh
    h_pch = PITCH_H_IN       / fh
    h_ftr = FOOTER_H_IN      / fh
    p_hp  = PAD_HDR_PITCH_IN / fh
    p_pf  = PAD_PCH_FTR_IN   / fh
    p_bot = PAD_BOT_IN       / fh
    y_ftr_bot = p_bot
    y_pch_bot = y_ftr_bot + h_ftr + p_pf
    y_pch_top = y_pch_bot + h_pch
    y_hdr_bot = y_pch_top + p_hp
    return {
        "header": (0.0, y_hdr_bot, 1.0, h_hdr),
        "pitch":  (0.0, y_pch_bot, 1.0, h_pch),
        "footer": (0.0, y_ftr_bot, 1.0, h_ftr),
    }


# ── Render principal del frame ─────────────────────────────────────────

def _draw_pitch_layer(ax, orientations_frame, attacking_team, ball_xy,
                      gk_jerseys, ppcf_att, ppcf_def, alpha_max):
    """Pinta surface + jugadores + balon sobre el axes del pitch (compartido
    por modo PNG y modo video)."""
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    resolved = np.clip(ppcf_att + ppcf_def, 0.0, 1.0)
    alpha = resolved * alpha_max
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

    for _, p in orientations_frame.iterrows():
        t_id = int(p["team"])
        j = int(p["jersey"])
        x, y = float(p["x"]), float(p["y"])
        is_gk = (j == gk_jerseys.get(t_id, 1))
        color = GK_C if is_gk else (ATT_C if t_id == attacking_team else DEF_C)

        ax.plot(x, y, "o", ms=MS, color=color,
                markeredgecolor=TEXT, markeredgewidth=1.0,
                alpha=0.95, zorder=5)

        sa = p.get("shoulder_angle", np.nan)
        if not (isinstance(sa, float) and np.isnan(sa)):
            sw = float(p.get("shoulder_width", 0.45) or 0.45) * 9
            perp_l = sa + np.pi / 2
            perp_r = sa - np.pi / 2
            bar_color = "#555555" if is_gk else color
            ax.plot(
                [x + (sw / 2) * np.cos(perp_l), x + (sw / 2) * np.cos(perp_r)],
                [y + (sw / 2) * np.sin(perp_l), y + (sw / 2) * np.sin(perp_r)],
                color=bar_color, linewidth=8, alpha=0.75, zorder=4,
                solid_capstyle="round",
            )

        ha = p.get("head_angle", np.nan)
        if not (isinstance(ha, float) and np.isnan(ha)):
            wedge = Wedge(
                (x, y), 3.5,
                np.degrees(ha) - 22.5,
                np.degrees(ha) + 22.5,
                color="#333333", alpha=0.30, zorder=3,
            )
            ax.add_patch(wedge)

        ax.text(x, y, str(j), color="white", fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

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
                    headaxislength=3.5, alpha=0.7, zorder=3,
                )

    if ball_xy is not None and not (np.isnan(ball_xy[0]) or np.isnan(ball_xy[1])):
        draw_football(ax, ball_xy[0], ball_xy[1], r=0.9, zorder=10)


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
    title: Optional[str] = None,
    subtitle=None,
    escudo_path: Optional[str] = None,
    att_team_name: Optional[str] = None,
    def_team_name: Optional[str] = None,
    attacking_right: bool = True,
    ax: plt.Axes = None,
    save_path: str = None,
    figsize: tuple = (16, 10.4),
) -> plt.Figure:
    """Render one frame con reach-field PPCF heatmap + players + ball.

    Dos modos:

    1) PNG ESTATICO (ax=None y title is not None):
       Layout CCV-style con header (escudo IZQ + title + sub + JO DCHA),
       pitch en el centro, footer (gradiente PPCF + Direccion + equipos +
       balon Telstar). Usa FIGSIZE_PNG (14 x ~11.6").

    2) VIDEO (ax dado): pinta SOLO en el axes (sin header / footer).
       El renderer de video crea la fig y la ax y los pasa.

    Args:
        att_team_name / def_team_name: nombres para la leyenda del footer
            (PNG mode). Si None, fallback "Atacante" / "Defensor".
        escudo_path: PNG del escudo del equipo atacante (header IZQ).
        attacking_right: orienta los triangulos del footer (PNG mode).
    """
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    if ppcf_att is None or ppcf_def is None:
        params = default_params(immediate_window=immediate_window) if immediate_window is not None else default_params()
        ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
            orientations_frame, attacking_team,
            params=params, n_grid_x=n_grid_x,
        )

    # ── Modo PNG estatico (con header + footer CCV-style) ───────────
    if ax is None and title is not None:
        L = _layout(FIGSIZE_PNG)
        fig = plt.figure(figsize=FIGSIZE_PNG, facecolor=BG)
        hdr_bot = L["header"][1]
        hdr_top = hdr_bot + L["header"][3]
        escudo = escudo_path if (escudo_path and Path(escudo_path).exists()) else None
        draw_header(fig, title=title, subtitle=subtitle,
                    escudo_path=escudo,
                    hdr_band=(hdr_bot, hdr_top))
        ax_pitch = fig.add_axes(L["pitch"])
        _draw_pitch_layer(ax_pitch, orientations_frame, attacking_team,
                          ball_xy, gk_jerseys, ppcf_att, ppcf_def, alpha_max)
        draw_footer_legend(
            fig, L["footer"],
            cmap=PPCF_CMAP,
            bar_left_label="100% Defender",
            bar_right_label="100% Attacker",
            bar_desc="Probability that the team controls this point of the pitch",
            att_team_name=att_team_name or "Attacker",
            def_team_name=def_team_name or "Defender",
            att_color=ATT_C, def_color=DEF_C,
            attacking_right=attacking_right,
        )
        if save_path:
            fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                        bbox_inches=None)
        return fig

    # ── Modo pitch-only (video o caller que ya prepara el ax) ───────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    _draw_pitch_layer(ax, orientations_frame, attacking_team,
                      ball_xy, gk_jerseys, ppcf_att, ppcf_def, alpha_max)

    if title:
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold", pad=12)

    if save_path:
        fig.savefig(save_path, dpi=400, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")

    return fig
