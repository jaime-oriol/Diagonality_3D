"""passes_plot — Player-pass visualization en identidad LIGHT OPTA / CCV.

Mismo concepto que la viz original (flechas coloreadas por direction_class,
footer 3-bloques con leyenda de tipos + direccion + stats) pero migrada al
estilo CCV: fondo blanco, lineas negras, tipografia Chakra Petch, logo JO
en el header via draw_header de common.

Cambios respecto al render dark:
  - Header: escudo del equipo IZQ + titulo + subtitulo (1 o N lineas) +
    logo JO DCHA, todo dibujado por common.draw_header. Sin axes propio.
  - Pass arrows en DIRECTION_COLORS (verde diagonal, azul forward, gris
    sideways, rosa backward) — todos legibles sobre BG blanco.
  - Footer mantiene 40/20/40, pero usa LEGEND para textos suaves y TEXT
    para numeros / acentos.

Schema esperado del DataFrame (one row per pass):
    x, y                  — origen en TRACAB metros (centrado, [-52.5, 52.5])
    x_receiver, y_receiver — destino en metros
    direction_class       — "forward" | "diagonal" | "sideways" | "backward"
    successful            — bool (True si completed)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse
from mplsoccer import Pitch

from .common import (
    BG, TEXT, LEGEND, PKW, DIRECTION_COLORS,
    PITCH_LENGTH, PITCH_WIDTH, draw_header,
)


# ── Geometry: pitch-aligned absolute layout ────────────────────────────
# Replicando el patron del PPCF de CCV: el ancho de la figura coincide con
# el ancho del campo (sin margenes laterales) y header/footer ocupan el
# mismo ancho horizontal. El alto del axes del pitch se deriva del ratio
# datos = (pitch_length + 2·margin_x) / (pitch_width + 2·margin_y) pa que
# set_aspect("equal") encaje sin huecos.

FIG_W            = 14.0
HEADER_H_IN      = 1.10
FOOTER_H_IN      = 1.30
PAD_TOP_IN       = 0.05
PAD_HDR_PITCH_IN = 0.05
PAD_PCH_FTR_IN   = 0.05
PAD_BOT_IN       = 0.10
PITCH_MARGIN_Y_M = 1.0
PITCH_MARGIN_X_M = 2.0      # >=1.5 pa porterias

_DATA_RATIO = (PITCH_LENGTH + 2 * PITCH_MARGIN_X_M) / (PITCH_WIDTH + 2 * PITCH_MARGIN_Y_M)
PITCH_H_IN  = FIG_W / _DATA_RATIO

FIG_H = (PAD_TOP_IN + HEADER_H_IN + PAD_HDR_PITCH_IN
         + PITCH_H_IN + PAD_PCH_FTR_IN
         + FOOTER_H_IN + PAD_BOT_IN)

FIGSIZE = (FIG_W, FIG_H)


def _layout(figsize: tuple = FIGSIZE) -> dict:
    """Posiciones absolutas de header/pitch/footer (frac fig)."""
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
        "left": 0.0, "width": 1.0,
        "header": (0.0, y_hdr_bot, 1.0, h_hdr),
        "pitch":  (0.0, y_pch_bot, 1.0, h_pch),
        "footer": (0.0, y_ftr_bot, 1.0, h_ftr),
    }


# ── Visual constants ────────────────────────────────────────────────────

ARROW_LW = 1.4
ARROW_ALPHA_OK = 0.95
ARROW_ALPHA_FAIL = 0.55
ARROW_STYLE = "->,head_length=5,head_width=4"

DIRECTION_ORDER = ["diagonal", "forward", "sideways", "backward"]
DIRECTION_LABELS = {
    "diagonal": "Diagonal",
    "forward":  "Forward",
    "sideways": "Sideways",
    "backward": "Backward",
}


# ── Pass arrows ────────────────────────────────────────────────────────

def _draw_pass_arrow(ax: plt.Axes,
                     x0: float, y0: float, x1: float, y1: float,
                     color: str, successful: bool):
    """One pass = thin Opta-style arrow. Solid for OK, dashed + faded
    for fail. Same color either way so per-class density reads."""
    style = "-" if successful else (0, (3, 2))
    alpha = ARROW_ALPHA_OK if successful else ARROW_ALPHA_FAIL
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=ARROW_STYLE,
        color=color, lw=ARROW_LW, alpha=alpha,
        linestyle=style, mutation_scale=1.0,
        capstyle="round", joinstyle="round",
        zorder=4,
    )
    ax.add_patch(arrow)


# ── Footer (3 bloques 40/20/40) ────────────────────────────────────────

def _make_block_axes(fig, left, bottom, width, height):
    """Sub-axes limpio (sin spines, lim 0-1) pa pintar leyenda en axes-frac."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def _draw_footer(
    fig: plt.Figure, L: dict,
    passes: pd.DataFrame,
    n_passes: int, accuracy_pct: float, diag_share_pct: float,
    attacking_right: bool,
):
    """Footer = 3 bloques 40/20/40 (mismo split que CCV PPCF).

    block 1 (40%) — leyenda 2x2 de direction_class con conteo por tipo
    block 2 (20%) — TOP: triangulos "Dirección de Ataque"
                   BOT: arrow dashed con conteo de unsuccessful
    block 3 (40%) — 3 mini-cells: passes, accuracy (diana), diagonal share
    """
    W, H = fig.get_size_inches()
    fl, fw = L["left"], L["width"]
    fb, fh = L["footer"][1], L["footer"][3]

    b1_w, b2_w, b3_w = fw * 0.40, fw * 0.20, fw * 0.40
    ax_b1 = _make_block_axes(fig, fl, fb, b1_w, fh)
    ax_b2 = _make_block_axes(fig, fl + b1_w, fb, b2_w, fh)
    ax_b3 = _make_block_axes(fig, fl + b1_w + b2_w, fb, b3_w, fh)

    # ── BLOCK 1: 2x2 cells (arrow + NUM + label) ────────────────────
    counts = passes["direction_class"].value_counts().to_dict() if len(passes) else {}
    cells = [
        (0.10, 0.78, "diagonal"),
        (0.50, 0.78, "forward"),
        (0.10, 0.32, "sideways"),
        (0.50, 0.32, "backward"),
    ]
    arrow_y_off  = 0.15
    arrow_len_ax = 0.27
    for cell_x, cell_y, key in cells:
        color = DIRECTION_COLORS[key]
        ax_y = cell_y + arrow_y_off
        x0 = cell_x + 0.05
        x1 = cell_x + arrow_len_ax + 0.10

        ax_b1.plot([x0, x1], [ax_y, ax_y],
                   color=color, lw=2.2, solid_capstyle="round",
                   transform=ax_b1.transAxes, zorder=10)
        head_dx, head_dy = 0.035, 0.07
        for sign in (+1, -1):
            ax_b1.plot([x1 - head_dx, x1],
                       [ax_y + sign * head_dy, ax_y],
                       color=color, lw=2.2, solid_capstyle="round",
                       transform=ax_b1.transAxes, zorder=10)

        n_cls = int(counts.get(key, 0))
        mid_x = cell_x + arrow_len_ax / 2.0
        gap = 0.012
        ax_b1.text(mid_x - gap, cell_y, f"{n_cls}",
                   ha="right", va="center", fontsize=22,
                   color=TEXT, fontweight="bold",
                   transform=ax_b1.transAxes)
        ax_b1.text(mid_x + gap, cell_y, DIRECTION_LABELS[key],
                   ha="left", va="center", fontsize=13,
                   color=LEGEND, transform=ax_b1.transAxes)

    # ── BLOCK 2: TOP triangulos / BOT dashed arrow ──────────────────
    n_tri, tri_w, tri_h, spacing = 4, 0.10, 0.16, 0.06
    total_w = n_tri * tri_w + (n_tri - 1) * spacing
    start_x = (1.0 - total_w) / 2.0
    y_tri = 0.90
    for i in range(n_tri):
        cx = start_x + i * (tri_w + spacing)
        if attacking_right:
            verts = [(cx, y_tri + tri_h / 2),
                     (cx + tri_w, y_tri),
                     (cx, y_tri - tri_h / 2)]
        else:
            verts = [(cx + tri_w, y_tri + tri_h / 2),
                     (cx, y_tri),
                     (cx + tri_w, y_tri - tri_h / 2)]
        alpha = 0.32 + i * 0.18
        ax_b2.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=TEXT, edgecolor="none",
            alpha=alpha, transform=ax_b2.transAxes, zorder=10))
    ax_b2.text(0.5, 0.725, "Attacking Direction",
               ha="center", va="center", fontsize=13,
               color=LEGEND, transform=ax_b2.transAxes,
               fontweight="bold")

    n_failed = int((~passes["successful"]).sum()) if len(passes) else 0
    cy_bot = 0.32
    ay_bot = cy_bot + arrow_y_off
    x0 = 0.15
    x1 = x0 + arrow_len_ax + 0.45
    ax_b2.plot([x0, x1], [ay_bot, ay_bot],
               color=TEXT, lw=2.2, linestyle=(0, (4, 2)),
               solid_capstyle="round",
               transform=ax_b2.transAxes, zorder=10)
    head_dx, head_dy = 0.035, 0.07
    for sign in (+1, -1):
        ax_b2.plot([x1 - head_dx, x1],
                   [ay_bot + sign * head_dy, ay_bot],
                   color=TEXT, lw=2.2, solid_capstyle="round",
                   transform=ax_b2.transAxes, zorder=10)
    mid_x_b2 = -0.01 + x0 + arrow_len_ax / 2.0
    gap_b2 = 0.012
    ax_b2.text(mid_x_b2 - gap_b2, cy_bot, f"{n_failed}",
               ha="right", va="center", fontsize=22,
               color=TEXT, fontweight="bold",
               transform=ax_b2.transAxes)
    ax_b2.text(mid_x_b2 + gap_b2, cy_bot, "Unsuccessful",
               ha="left", va="center", fontsize=13,
               color=LEGEND, transform=ax_b2.transAxes)

    # ── BLOCK 3: 3 mini-cells stats ─────────────────────────────────
    tfig = fig.transFigure
    CIRCLE_R_IN = 0.32
    cw = 2 * CIRCLE_R_IN / W
    ch = 2 * CIRCLE_R_IN / H

    b3_l = fl + b1_w + b2_w
    cx_pass = b3_l + b3_w * 0.18
    cx_acc  = b3_l + b3_w * 0.50
    cx_diag = b3_l + b3_w * 0.82
    y_circle = fb + fh * 0.75
    y_label  = fb + fh * 0.40

    def _outline_circle(cx, edge):
        fig.patches.append(Ellipse(
            (cx, y_circle), cw, ch, transform=tfig,
            facecolor="none", edgecolor=edge,
            linewidth=2.0, alpha=0.95, zorder=10))

    def _dartboard(cx, n_rings=3, ring_gap=0.26):
        for i in range(n_rings):
            scale = 1.0 - i * ring_gap
            alpha = 0.55 - i * 0.13
            fig.patches.append(Ellipse(
                (cx, y_circle), cw * scale, ch * scale, transform=tfig,
                facecolor="none", edgecolor=DIRECTION_COLORS["backward"],
                linewidth=3.5, alpha=alpha, zorder=9))

    def _stat_cell(cx, value, label, edge_color, dartboard=False):
        if dartboard:
            _dartboard(cx)
        else:
            _outline_circle(cx, edge_color)
        fig.text(cx, y_circle, value,
                 ha="center", va="center", fontsize=18,
                 color=TEXT, fontweight="bold",
                 transform=tfig, zorder=12)
        fig.text(cx, y_label, label,
                 ha="center", va="center", fontsize=13,
                 color=LEGEND, transform=tfig)

    _stat_cell(cx_pass, f"{n_passes}", "Passes", TEXT)
    _stat_cell(cx_acc,  f"{accuracy_pct:.0f}%", "Accuracy", TEXT,
               dartboard=True)
    _stat_cell(cx_diag, f"{diag_share_pct:.0f}%", "Diagonal",
               DIRECTION_COLORS["diagonal"])


# ── Main API ───────────────────────────────────────────────────────────

def plot_player_passes(
    passes: pd.DataFrame,
    title: str,
    subtitle,
    attacking_right: bool = True,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,   # ignorado: el JO va en draw_header
    save_path: Optional[str] = None,
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """Render a player-passes figure en identidad LIGHT OPTA / CCV.

    Layout: header (escudo IZQ + titulo + sub + JO DCHA, via draw_header)
    + pitch + footer 3 bloques. Todo anclado al ancho completo de la figura.

    Args:
        passes: DataFrame, one row per pass. Required columns:
            x, y, x_receiver, y_receiver, direction_class, successful.
        title: titulo (Chakra Petch 700) — ej. "Michael Olise · Passes".
        subtitle: 1 string o lista de strings (lineas adicionales bajo el
            titulo).
        attacking_right: True si el jugador atacaba hacia +x.
        team_logo_path: PNG del escudo del equipo (header IZQ). Si None
            o no existe, el header no pinta escudo (solo titulo + JO).
        project_logo_path: ignorado en esta version — el logo JO se gestiona
            siempre desde common.draw_header (esquina DCHA del header).
        save_path: Optional output file. Guardado a dpi=200.
    """
    required = {"x", "y", "x_receiver", "y_receiver",
                "direction_class", "successful"}
    missing = required - set(passes.columns)
    if missing:
        raise ValueError(f"passes DataFrame missing columns: {sorted(missing)}")

    L = _layout(figsize)
    fig = plt.figure(figsize=figsize, facecolor=BG)

    # ── Header (escudo IZQ + titulo + sub + JO DCHA) ───────────────────
    hdr_bot = L["header"][1]
    hdr_top = hdr_bot + L["header"][3]
    escudo = team_logo_path if (team_logo_path and Path(team_logo_path).exists()) else None
    draw_header(fig, title=title, subtitle=subtitle,
                escudo_path=escudo,
                hdr_band=(hdr_bot, hdr_top))

    # ── Pitch ──────────────────────────────────────────────────────────
    ax_pitch = fig.add_axes(L["pitch"])
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax_pitch)

    for _, p in passes.iterrows():
        cls = str(p["direction_class"])
        if cls not in DIRECTION_COLORS:
            continue
        _draw_pass_arrow(
            ax_pitch,
            float(p["x"]), float(p["y"]),
            float(p["x_receiver"]), float(p["y_receiver"]),
            color=DIRECTION_COLORS[cls],
            successful=bool(p["successful"]),
        )

    # ── Footer ─────────────────────────────────────────────────────────
    n = int(len(passes))
    if n > 0:
        accuracy = float(passes["successful"].mean()) * 100.0
        diag = float((passes["direction_class"] == "diagonal").mean()) * 100.0
    else:
        accuracy = 0.0; diag = 0.0
    _draw_footer(fig, L, passes=passes,
                 n_passes=n, accuracy_pct=accuracy,
                 diag_share_pct=diag,
                 attacking_right=attacking_right)

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches=None)
    return fig
