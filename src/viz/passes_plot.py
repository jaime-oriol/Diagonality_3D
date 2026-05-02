"""
passes_plot — Player-pass visualization in our brand style.

Inspired by the Opta Analyst player-pass figure (figures/Messi_pases.jpg)
but adapted to the project's dark theme + the Diagonality-3D narrative:

  - Pass arrows colored by **direction class** (gold = diagonal, the SV
    signature) instead of just success.
  - Unsuccessful passes use the SAME class colour but a dashed line at
    lower alpha — the per-class density stays readable.
  - Header and footer are anchored to the **exact horizontal extent of
    the pitch** via absolute axis positioning, so the figure feels
    grid-aligned regardless of figsize.
  - Footer = 3 uniform blocks: 2x2 direction legend, attacking-direction
    triangles, Opta-style outlined number-in-circle stats with diagonal
    share highlighted in gold.

Schema expected for the passes DataFrame (one row per pass):
    x, y                 — origin in TRACAB meters (centered, [-52.5, 52.5])
    x_receiver, y_receiver — destination in meters
    direction_class      — "forward" | "diagonal" | "sideways" | "backward"
    successful           — bool (True if completed)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mplsoccer import Pitch

from .common import (
    BG, WHITE, PKW, DIRECTION_COLORS,
)
import matplotlib.font_manager as _fm

# Try to use Inter (Opta-like aesthetic). Fall back to DejaVu Sans
# (always present) if Inter is not installed in the system fonts.
def _resolve_font():
    try:
        for ttf in [
            "/home/jaime/.local/share/fonts/Inter-Regular.ttf",
            "/home/jaime/.local/share/fonts/Inter-Bold.ttf",
            "/home/jaime/.local/share/fonts/Inter-Medium.ttf",
            "/home/jaime/.local/share/fonts/Inter-SemiBold.ttf",
        ]:
            try: _fm.fontManager.addfont(ttf)
            except Exception: pass
        names = {f.name for f in _fm.fontManager.ttflist}
        if "Inter" in names:
            return "Inter"
    except Exception:
        pass
    return "DejaVu Sans"

FONT = _resolve_font()


# ── Geometry: pitch-aligned absolute layout (figsize-aware) ────────────
# Pitch true aspect = 105 / 68 = 1.544. We size the pitch axis to that
# exact ratio so mplsoccer fills it edge-to-edge, no internal margins.
# Header and footer share the SAME horizontal extent as the pitch.

FIGSIZE = (14.0, 9.5)
PITCH_RATIO = 105.0 / 68.0   # ~1.544

# Fractions of the figure height
H_HEADER = 0.10
H_PITCH = 0.70
H_FOOTER = 0.16
# Vertical spacing — header + footer flush against the pitch (Opta-style)
PAD_TOP = 0.015
PAD_HEADER_PITCH = 0.0
PAD_PITCH_FOOTER = 0.005
PAD_BOTTOM = 0.020

# Compute pitch horizontal span from the height ratio so it stays edge-
# aligned with header/footer (constant `pitch_w` as a fraction of figure).
def _layout(figsize=FIGSIZE):
    fw, fh = figsize
    pitch_h_in = H_PITCH * fh
    pitch_w_in = pitch_h_in * PITCH_RATIO
    pitch_w_frac = pitch_w_in / fw
    left = (1.0 - pitch_w_frac) / 2.0
    right = left + pitch_w_frac

    y_footer_bot = PAD_BOTTOM
    y_footer_top = y_footer_bot + H_FOOTER
    y_pitch_bot = y_footer_top + PAD_PITCH_FOOTER
    y_pitch_top = y_pitch_bot + H_PITCH
    y_header_bot = y_pitch_top + PAD_HEADER_PITCH
    y_header_top = y_header_bot + H_HEADER

    # Sanity check: should be <= 1 - PAD_TOP
    assert y_header_top <= 1.0 - PAD_TOP + 1e-6, (
        f"layout overflows: header_top={y_header_top}")

    return {
        "left": left, "right": right, "width": pitch_w_frac,
        "header": (left, y_header_bot, pitch_w_frac, H_HEADER),
        "pitch":  (left, y_pitch_bot, pitch_w_frac, H_PITCH),
        "footer": (left, y_footer_bot, pitch_w_frac, H_FOOTER),
    }


# ── Visual constants for arrows + footer ───────────────────────────────

ARROW_LW = 1.4
ARROW_ALPHA_OK = 0.95
ARROW_ALPHA_FAIL = 0.55
# Open-style arrow head (two strokes meeting at a point — like Opta's `→`)
ARROW_STYLE = "->,head_length=5,head_width=4"

DIRECTION_ORDER = ["diagonal", "forward", "sideways", "backward"]
DIRECTION_LABELS = {
    "diagonal": "Diagonal",
    "forward":  "Forward",
    "sideways": "Sideways",
    "backward": "Backward",
}


# ── Pass arrows ────────────────────────────────────────────────────────

def _draw_pass_arrow(
    ax: plt.Axes,
    x0: float, y0: float, x1: float, y1: float,
    color: str, successful: bool,
):
    """One pass = thin Opta-style arrow. Solid for OK, dashed + faded
    for fail. Same color either way so the per-class density reads."""
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


# ── Header ─────────────────────────────────────────────────────────────

def _draw_header(
    ax: plt.Axes,
    title: str, subtitle: str,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,
):
    """Top strip aligned to pitch width: optional team logo (left edge),
    title + subtitle (left), optional project logo (right edge)."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    text_x = 0.125
    if team_logo_path is not None and Path(team_logo_path).exists():
        try:
            img = plt.imread(team_logo_path)
            ab = AnnotationBbox(
                OffsetImage(img, zoom=0.45),
                (0.0, 0.50), frameon=False,      # X ↑ → escudo a la DERECHA / X ↓ → IZQ (mantén dentro de [0,1])
                box_alignment=(0.0, 0.5),         # ancla LEFT-EDGE del logo en X
            )
            ab.set_clip_on(False)                 # permite render fuera del axes si X<0 o X>1
            ax.add_artist(ab)
            text_x = 0.1
        except Exception:
            pass

    # Subtitle puede ser str (1 línea) o list/tuple (N líneas).
    # Spacing visualmente uniforme: el título es fontsize 25 (ocupa más
    # altura visible) que los subs (fontsize 15), así que el gap
    # title→sub1 debe ser MAYOR que sub1→sub2 para que el espacio en
    # blanco entre líneas sea el mismo a la vista.
    sub_lines = list(subtitle) if isinstance(subtitle, (list, tuple)) else [subtitle]
    n_sub = len(sub_lines)
    title_y       = 0.82 if n_sub >= 2 else 0.55  # ↑ → título más alto
    gap_title_sub = 0.36   # ↑ → más espacio entre TÍTULO y sub1
    gap_sub_sub   = 0.26   # ↑ → más espacio entre sub1 y sub2

    ax.text(
        text_x, title_y, title, color=WHITE,
        fontsize=25, fontweight="bold", fontfamily=FONT,
        ha="left", va="center",
    )
    y = title_y
    for i, line in enumerate(sub_lines):
        y -= gap_title_sub if i == 0 else gap_sub_sub  # 1er sub usa gap mayor; resto, gap entre subs
        ax.text(
            text_x, y, line, color=WHITE,
            fontsize=15, fontweight="normal", fontfamily=FONT,
            ha="left", va="center", alpha=0.85,
        )

    if project_logo_path is not None and Path(project_logo_path).exists():
        try:
            img = plt.imread(project_logo_path)
            # Auto-zoom so the rendered logo has the same visual height
            # regardless of the source PNG resolution. Target ≈ 48 px tall
            # (the size that matched the previous Logo_vizs.png at zoom=0.08).
            TARGET_HEIGHT_PX = 48
            zoom = TARGET_HEIGHT_PX / float(img.shape[0])
            ab = AnnotationBbox(
                OffsetImage(img, zoom=zoom),
                (1.0, 0.42), frameon=False,      # X ↑ → logo a la DERECHA (mantén dentro de [0,1]) / Y ↓ → MÁS ABAJO
                box_alignment=(1.0, 0.5),         # ancla RIGHT-EDGE del logo en X
            )
            ab.set_clip_on(False)                 # permite render fuera del axes si X>1
            ax.add_artist(ab)
        except Exception:
            pass


# ── Footer (Opta-Forum-style figure-coord legend) ───────────────────────
# Following the exact pattern of Jaime's Opta Forum common.py:
#   - All elements positioned in FIGURE coordinates via fig.transFigure.
#   - Real circles via `Ellipse(cw, ch)` with cw=2r/W, ch=2r/H so they
#     stay round regardless of axis aspect.
#   - Each block: header (fontsize 14 bold) on top, symbol(s) in middle,
#     label(s) below symbol — same vertical rhythm as the "Off-Ball Run"
#     entry of the Opta Forum legend.

def _make_block_axes(fig, left, bottom, width, height):
    """A clean axes inside the figure for legend symbols (no spines, lim 0-1)."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def _draw_footer(
    fig: plt.Figure,
    L: dict,
    passes: pd.DataFrame,
    n_passes: int,
    accuracy_pct: float,
    diag_share_pct: float,
    attacking_right: bool,
):
    """Footer = 3 horizontal blocks with widths 40 / 20 / 40 of pitch.

    Each block is a sub-axes so arrows can be drawn in axes coordinates
    (FancyArrowPatch renders correctly there). Real circles inside the
    stats block via `Ellipse(cw=2r/W, ch=2r/H)` so they stay round."""
    from matplotlib.patches import Ellipse

    W, H = fig.get_size_inches()

    fl = L["left"]; fw = L["width"]
    fb = L["footer"][1]
    fh = L["footer"][3]

    # 40 / 20 / 40 split
    b1_w = fw * 0.40
    b2_w = fw * 0.20
    b3_w = fw * 0.40
    ax_b1 = _make_block_axes(fig, fl, fb, b1_w, fh)
    ax_b2 = _make_block_axes(fig, fl + b1_w, fb, b2_w, fh)
    ax_b3 = _make_block_axes(fig, fl + b1_w + b2_w, fb, b3_w, fh)

    # ── BLOCK 1: Pass type (2x2 cells; arrow on top, NUM + label below)
    counts = passes["direction_class"].value_counts().to_dict() if len(passes) else {}
    cells = [
        # (cell_x, cell_y, key)
        # cell_x ↑ → cell se mueve a la DERECHA      | cell_x ↓ → IZQUIERDA
        # cell_y ↑ → texto sube (más cerca arrow)     | cell_y ↓ → texto baja
        (0.10, 0.78, "diagonal"),   # arriba-izq
        (0.5, 0.78, "forward"),    # arriba-der
        (0.10, 0.32, "sideways"),   # abajo-izq
        (0.5, 0.32, "backward"),   # abajo-der
    ]
    arrow_y_off  = 0.15  # ↑ → arrow MÁS ARRIBA respecto al texto / ↓ → arrow PEGADO al texto
    arrow_len_ax = 0.27  # ↑ → arrow MÁS LARGA / ↓ → arrow más corta (head queda más cerca del start)
    for cell_x, cell_y, key in cells:
        color = DIRECTION_COLORS[key]
        ax_y = cell_y + arrow_y_off            # Y final donde se dibuja el arrow (combina cell_y + offset)
        x0 = cell_x + 0.05                     # X inicio shaft (lado izquierdo de la flecha)
        x1 = cell_x + arrow_len_ax + 0.1       # X fin shaft (punta)

        # Shaft horizontal
        ax_b1.plot(
            [x0, x1], [ax_y, ax_y],
            color=color,
            lw=2.2,                            # ↑ → línea MÁS GORDA / ↓ más fina
            solid_capstyle="round",
            transform=ax_b1.transAxes, zorder=10,
        )

        # Open head `>` (dos strokes que convergen en la punta)
        head_dx = 0.035  # ↑ → head MÁS LARGO en X (more pulled back) / ↓ head más corto
        head_dy = 0.07   # ↑ → head MÁS ABIERTO en Y (más V) / ↓ head más cerrado (casi recto)
        for sign in (+1, -1):
            ax_b1.plot(
                [x1 - head_dx, x1],
                [ax_y + sign * head_dy, ax_y],
                color=color,
                lw=2.2,                        # ↑ trazo head MÁS GORDO
                solid_capstyle="round",
                transform=ax_b1.transAxes, zorder=10,
            )

        # NUM + label en MISMA Y baseline, centrados bajo midpoint del arrow
        n_cls = int(counts.get(key, 0))
        mid_x = cell_x + arrow_len_ax / 2.0    # punto medio horizontal del arrow shaft
        gap   = 0.012  # ↑ → más SEPARACIÓN entre NUM y label / ↓ pegados
        ax_b1.text(
            mid_x - gap, cell_y, f"{n_cls}",
            ha="right", va="center",           # NUM termina (right edge) en mid_x - gap
            fontsize=22,                       # ↑ → NUMERO MÁS GORDO / ↓ más pequeño
            color=WHITE, fontweight="bold",
            fontfamily=FONT, transform=ax_b1.transAxes,
        )
        ax_b1.text(
            mid_x + gap, cell_y, DIRECTION_LABELS[key],
            ha="left", va="center",            # label empieza (left edge) en mid_x + gap
            fontsize=12,                       # ↑ → LABEL MÁS GRANDE / ↓ más pequeño
            color=WHITE, fontfamily=FONT,
            transform=ax_b1.transAxes, alpha=0.95,  # alpha ↑ → MÁS BLANCO / ↓ MÁS APAGADO
        )

    # ── BLOCK 2: split en 2 filas — TOP: triangles + "Attacking direction"
    # BOT: dashed arrow + N + "Unsuccessful". Misma cadencia que Block 1.
    n_tri   = 4      # ↑ → MÁS triángulos / ↓ menos
    tri_w   = 0.1   # ↑ → triángulo MÁS ANCHO / ↓ más estrecho
    tri_h   = 0.16   # ↑ → triángulo MÁS ALTO / ↓ más bajo
    spacing = 0.06   # ↑ → MÁS hueco entre triángulos / ↓ pegados
    total_w = n_tri * tri_w + (n_tri - 1) * spacing
    start_x = (1.0 - total_w) / 2.0           # arranca centrado en el bloque
    y_tri   = 0.9   # ↑ → triángulos SUBEN / ↓ BAJAN (tiene que coincidir con arrow Y de block 1 top)
    for i in range(n_tri):
        cx = start_x + i * (tri_w + spacing)
        if attacking_right:                   # apunta a la DERECHA
            verts = [(cx, y_tri + tri_h / 2),
                     (cx + tri_w, y_tri),
                     (cx, y_tri - tri_h / 2)]
        else:                                 # apunta a la IZQUIERDA
            verts = [(cx + tri_w, y_tri + tri_h / 2),
                     (cx, y_tri),
                     (cx + tri_w, y_tri - tri_h / 2)]
        alpha = 0.32 + i * 0.18               # ↑ base / step → triángulos MÁS OPACOS
        ax_b2.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=WHITE, edgecolor="none",
            alpha=alpha, transform=ax_b2.transAxes, zorder=10,
        ))
    ax_b2.text(
        0.5,                                  # X centro horizontal
        0.725,                                 # ↑ → "Attacking direction" SUBE / ↓ BAJA
        "Attacking direction",
        ha="center", va="center",
        fontsize=12,                          # ↑ texto MÁS GRANDE / ↓ más pequeño
        color=WHITE, fontfamily=FONT,
        transform=ax_b2.transAxes,
        alpha=0.95,                           # ↑ MÁS BLANCO / ↓ APAGADO
    )

    # Fila inferior: arrow dashed + NUM + "Unsuccessful"
    n_failed = int((~passes["successful"]).sum()) if len(passes) else 0
    cy_bot = 0.32   # ↑ → texto SUBE / ↓ BAJA (debe coincidir con cell_y bot de block 1)
    ay_bot = cy_bot + arrow_y_off              # arrow Y = texto Y + offset (mismo offset que block 1)
    x0 = 0.15                                  # ↑ → toda la fila se desplaza a la DERECHA / ↓ IZQ
    x1 = x0 + arrow_len_ax + 0.45                     # punta del arrow (mismo length que block 1)

    # Dashed shaft
    ax_b2.plot(
        [x0, x1], [ay_bot, ay_bot],
        color=WHITE,
        lw=2.2,                                # ↑ trazo MÁS GORDO
        linestyle=(0, (4, 2)),                 # patrón dash → (3 on, 2 off). ↑ "on" → guiones MÁS LARGOS / ↑ "off" → más SEPARADOS
        solid_capstyle="round",
        transform=ax_b2.transAxes, zorder=10,
    )
    # Open `>` head — usa los MISMOS head_dx / head_dy del Block 1
    for sign in (+1, -1):
        ax_b2.plot(
            [x1 - head_dx, x1],
            [ay_bot + sign * head_dy, ay_bot],
            color=WHITE, lw=2.2, solid_capstyle="round",
            transform=ax_b2.transAxes, zorder=10,
        )

    # NUM + label centered under arrow midpoint (igual rhythm que block 1)
    mid_x_b2 = -0.01 + x0 + arrow_len_ax / 2.0
    gap_b2   = 0.012   # ↑ MÁS SEPARACIÓN NUM↔label / ↓ pegados
    ax_b2.text(
        mid_x_b2 - gap_b2, cy_bot, f"{n_failed}",
        ha="right", va="center",
        fontsize=22,                           # ↑ NUMERO MÁS GORDO
        color=WHITE, fontweight="bold",
        fontfamily=FONT, transform=ax_b2.transAxes,
    )
    ax_b2.text(
        mid_x_b2 + gap_b2, cy_bot, "Unsuccessful",
        ha="left", va="center",
        fontsize=12,                           # ↑ label MÁS GRANDE (cuidado — bloque B2 estrecho, puede salirse)
        color=WHITE, fontfamily=FONT,
        transform=ax_b2.transAxes, alpha=0.95,
    )

    # ── BLOCK 3: Stats — 3 cells (passes / accuracy-dartboard / diagonal).
    # Círculos en FIGURE coords (Ellipse cw,ch) → permanecen REDONDOS
    # independientemente del aspect del axis.
    tfig = fig.transFigure
    CIRCLE_R_IN = 0.34   # ↑ → CÍRCULOS MÁS GRANDES (en pulgadas) / ↓ más pequeños
    cw = 2 * CIRCLE_R_IN / W                   # ancho círculo en figure-frac (no tocar — derivado)
    ch = 2 * CIRCLE_R_IN / H                   # alto  círculo en figure-frac (no tocar — derivado)

    # Posiciones X de cada cell dentro del block 3 (b3_l..b3_l+b3_w)
    b3_l = fl + b1_w + b2_w
    cx_pass = b3_l + b3_w * 0.18   # ↑ → "passes" se mueve a la DERECHA / ↓ a la IZQ
    cx_acc  = b3_l + b3_w * 0.50   # ↑ → "accuracy" a la DERECHA / ↓ a la IZQ
    cx_diag = b3_l + b3_w * 0.82   # ↑ → "diagonal" a la DERECHA / ↓ a la IZQ

    y_circle = fb + fh * 0.75      # ↑ → círculos SUBEN (más cerca borde superior) / ↓ BAJAN
    y_label  = fb + fh * 0.4      # ↑ → labels SUBEN (más cerca círculos) / ↓ BAJAN (más cerca pie figura)

    def _outline_circle(cx, edge):
        fig.patches.append(Ellipse(
            (cx, y_circle), cw, ch, transform=tfig,
            facecolor="none", edgecolor=edge,
            linewidth=2.0,                     # ↑ → BORDE MÁS GORDO / ↓ más fino
            alpha=0.95, zorder=10,
        ))

    def _dartboard(cx, n_rings=3, ring_gap=0.26):
        """Opta-style target: anillos concéntricos rojos hacia DENTRO."""
        ring_color = "#FF6B6B"  # cambia → otro color base de la diana (e.g. "#FFD700" para gold)
        # n_rings  ↑ → MÁS anillos / ↓ menos
        # ring_gap ↑ → anillos MÁS SEPARADOS hacia adentro / ↓ pegados
        for i in range(n_rings):
            scale = 1.0 - i * ring_gap         # i=0 → ring exterior (full), i++ shrink
            alpha = 0.55 - i * 0.13            # ↑ base → diana MÁS OPACA / ↑ step → desvanece más rápido
            fig.patches.append(Ellipse(
                (cx, y_circle), cw * scale, ch * scale, transform=tfig,
                facecolor="none", edgecolor=ring_color,
                linewidth=3.5,                 # ↑ → ANILLOS MÁS GORDOS / ↓ más finos
                alpha=alpha, zorder=9,
            ))

    def _stat_cell(cx, value, label, edge_color, dartboard=False):
        if dartboard:
            _dartboard(cx)
        else:
            _outline_circle(cx, edge_color)
        fig.text(
            cx, y_circle, value,
            ha="center", va="center",
            fontsize=16,                       # ↑ → NUM/% dentro del círculo MÁS GRANDE
            color=WHITE, fontweight="bold",
            fontfamily=FONT, transform=tfig, zorder=12,
        )
        fig.text(
            cx, y_label, label,
            ha="center", va="center",
            fontsize=12,                       # ↑ → label DEBAJO MÁS GRANDE
            color=WHITE, fontfamily=FONT,
            transform=tfig,
            alpha=0.95,                        # ↑ MÁS BLANCO / ↓ APAGADO
        )

    _stat_cell(cx_pass, f"{n_passes}", "passes", WHITE)
    _stat_cell(cx_acc, f"{accuracy_pct:.0f}%", "accuracy", WHITE,
               dartboard=True)
    _stat_cell(cx_diag, f"{diag_share_pct:.0f}%", "diagonal",
               DIRECTION_COLORS["diagonal"])


# ── Main API ───────────────────────────────────────────────────────────

def plot_player_passes(
    passes: pd.DataFrame,
    title: str,
    subtitle: str,
    attacking_right: bool = True,
    team_logo_path: Optional[str] = None,
    project_logo_path: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = FIGSIZE,
) -> plt.Figure:
    """Render a player-passes figure in the project style.

    Layout: header, pitch and 3-block footer all anchored to the exact
    horizontal extent of the pitch (no overflow into the figure margins).

    Args:
        passes: DataFrame, one row per pass. Required columns:
            x, y, x_receiver, y_receiver, direction_class, successful.
        title: top-of-page title (e.g. "Michael Olise — Passes").
        subtitle: smaller line under title (match info).
        attacking_right: True if the player attacked toward +x.
        team_logo_path: Optional PNG (transparent BG) for the top-left
            logo next to the title. Skipped if missing.
        project_logo_path: Optional PNG for the top-right project logo.
            Skipped if missing.
        save_path: Optional output file. Saved at dpi=200.
    """
    required = {"x", "y", "x_receiver", "y_receiver",
                "direction_class", "successful"}
    missing = required - set(passes.columns)
    if missing:
        raise ValueError(f"passes DataFrame missing columns: {sorted(missing)}")

    L = _layout(figsize)
    fig = plt.figure(figsize=figsize, facecolor=BG)

    ax_header = fig.add_axes(L["header"])
    _draw_header(ax_header, title, subtitle,
                 team_logo_path, project_logo_path)

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

    # Footer drawn directly in figure coordinates (Opta-Forum pattern):
    # circles are real circles, labels are figure-level text, blocks are
    # arranged across the pitch width by computing x-centres in fig coords.
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
