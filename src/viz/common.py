"""common — Estilo + helpers compartidos por todas las vizs.

Identidad LIGHT OPTA (heredada de jaime-oriol/CCV): fondo BLANCO, textos
NEGROS, leyenda gris medio. Paleta azul / rojo saturada (ATT / DEF) y
acentos brand para los heatmaps propios (DOS visible cyan->magenta,
SHADOW amber->red). Tipografia Chakra Petch (cortes angulares Opta).

Exporta:
  - Paleta:       BG, TEXT, LEGEND, GRID, PITCH, WHITE(=TEXT), ATT, DEF,
                  ATT_LIGHT, DEF_LIGHT, GK, NEUTRAL, BALL
  - Cmaps:        PPCF_CMAP (DEF->NEUTRAL->ATT), DOS_CMAP (visible),
                  SHADOW_CMAP, DIRECTION_COLORS (pass-map)
  - Path effects: PE (halo blanco fino), PE_S (halo negro grueso)
  - Tipografia:   FONT_STACK (Chakra Petch + fallbacks) registrado en rcParams
  - Layout:       MASTER_FIGSIZE (16x9), HEADER_BAND (0.85..1.0),
                  PITCH_LENGTH/WIDTH, PKW (mplsoccer kwargs)
  - Helpers:      draw_pitch (numpy puro, alternativa a mplsoccer),
                  draw_header (escudo IZQ + titulo+sub + JO DCHA),
                  draw_football (Telstar para footer / pitch),
                  draw_footer_legend (3-block gradiente + direccion + equipos),
                  add_logo (legacy corner stamp)

WHITE/PE retro-compat: WHITE apunta a TEXT (negro) - codigo viejo que pinta
con WHITE sigue funcionando (negro sobre blanco) salvo en sitios donde se
necesita literal blanco (dorsales sobre marker azul/rojo); ahi se usa el
string "white" hardcodeado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


# ────────────────────────────────────────────────────────────────────────
# REGISTRO DE FUENTES — Chakra Petch al importar el modulo
# ────────────────────────────────────────────────────────────────────────
# Carga Chakra Petch (todos los pesos) en el fontManager de matplotlib pa
# que rcParams pueda usarlo. Mira en figures/fonts/ del repo y en
# ~/.local/share/fonts. Si no existe, matplotlib usa el siguiente del
# FONT_STACK (Inter, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_FONT_DIR = _REPO_ROOT / "figures" / "fonts"
_SYS_FONT_DIR = Path.home() / ".local" / "share" / "fonts"
for _fdir in (_REPO_FONT_DIR, _SYS_FONT_DIR):
    if _fdir.exists():
        for _f in _fdir.glob("ChakraPetch*.ttf"):
            try:
                _fm.fontManager.addfont(str(_f))
            except Exception:
                pass


# ────────────────────────────────────────────────────────────────────────
# PALETA LIGHT OPTA
# ────────────────────────────────────────────────────────────────────────

BG      = "#ffffff"          # fondo BLANCO de toda la viz
TEXT    = "#000000"          # titulos, labels, ticks
LEGEND  = "#666666"          # notas secundarias, mediana inline
GRID    = "#dddddd"          # gridlines + spines (gris muy suave)
PITCH   = "#000000"          # lineas del campo (negro Opta)

WHITE = TEXT                  # alias retro-compat: codigo viejo que usa
                              # WHITE pa markeredgecolor sigue funcionando
                              # (borde negro sobre fondo blanco, correcto).

# Acentos — saturados pa contrastar bien sobre BG blanco
ATT       = "#3b82f6"        # azul vivo light: atacante / PPCF blue zones
DEF       = "#ef4444"        # rojo vivo light: defensor / PPCF red zones
ATT_LIGHT = "#93c5fd"        # head wedge del atacante (gaze)
DEF_LIGHT = "#fca5a5"        # head wedge del defensor (gaze)
GK        = "#000000"        # portero (contraste maximo en pitch claro)
NEUTRAL   = "#e5e7eb"        # zona contestada PPCF (gris claro)
BALL      = "#ffffff"        # balon (circulo blanco con borde negro)


# ────────────────────────────────────────────────────────────────────────
# TIPOGRAFIA
# ────────────────────────────────────────────────────────────────────────
# matplotlib usa el primer font de la lista que encuentre. Chakra Petch
# primero pa tener los cortes angulares Opta-style en mayusculas.
FONT_STACK = ["Chakra Petch", "Inter", "Source Sans 3", "Helvetica Neue",
              "Helvetica", "Arial", "DejaVu Sans"]
FONT = FONT_STACK[0]          # default para fontfamily=FONT en codigo viejo

plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": None,        # SIN tight-crop: PNG sale EXACTO al figsize
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

# Path effects: halo alrededor del texto pa que se lea sobre fondos coloreados.
PE   = [pe.withStroke(linewidth=1.5, foreground=BG), pe.Normal()]      # halo BLANCO fino
PE_S = [pe.withStroke(linewidth=2.2, foreground="black")]               # halo NEGRO grueso (dorsales blancos)


# ────────────────────────────────────────────────────────────────────────
# COLORMAPS
# ────────────────────────────────────────────────────────────────────────

# PPCF (pitch control): defensor (rojo) -> neutro (gris claro) -> atacante (azul).
PPCF_CMAP = LinearSegmentedColormap.from_list("ppcf", [DEF, NEUTRAL, ATT])

# DOS visible (Diagonal Opportunity Surface dentro del scan del portador):
# blanco -> cyan -> purpura -> magenta. Alpha por celda lleva el peso visual,
# asi que el blanco inicial nunca se ve; lo importante es el pico magenta.
DOS_CMAP = LinearSegmentedColormap.from_list(
    "dos", ["#ffffff", "#06b6d4", "#9333ea", "#ec4899"]
)

# DOS shadow (Hamilton shadowpass — diagonal fuera del scan pero delante del balon):
# blanco -> amber -> naranja -> rojo oscuro. Warm urgency sobre fondo blanco.
SHADOW_CMAP = LinearSegmentedColormap.from_list(
    "dos_shadow", ["#ffffff", "#fbbf24", "#f97316", "#b91c1c"]
)

# Direction class (pass map): paleta BRIGHT + saturada al estilo de los
# radares de CCV (#1CA64C, #3A6FE0, #FF6B00, #E8344E — accents de banderas
# nacionales: verde Mexico, azul Francia, oranje Holanda, rojo Portugal).
# Lightness media (no oscura) pa que las flechas chillen sobre BG blanco
# y se distingan entre si por hue + luminance.
DIRECTION_COLORS = {
    "diagonal": "#1CA64C",   # verde vivo (Mexico / Senegal CCV)
    "forward":  "#3A6FE0",   # azul vivo (Ecuador CCV)
    "sideways": "#9333EA",   # violeta vivo — distinto a green/blue/red por hue (no warm, no cool azulado)
    "backward": "#E8344E",   # rojo vivo (Portugal / Belgium CCV)
}


# ────────────────────────────────────────────────────────────────────────
# PITCH DIMENSIONS
# ────────────────────────────────────────────────────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

# mplsoccer kwargs — secondspectrum = centrado en (0, 0), metros.
# Lineas negras finas sobre fondo blanco = identidad LIGHT OPTA.
PKW = dict(
    pitch_type="secondspectrum",
    pitch_length=PITCH_LENGTH,
    pitch_width=PITCH_WIDTH,
    pitch_color=BG,
    line_color=PITCH,
    line_zorder=2,
    linewidth=1.0,
)


# ────────────────────────────────────────────────────────────────────────
# LAYOUT MASTER
# ────────────────────────────────────────────────────────────────────────

MASTER_FIGSIZE = (16.0, 9.0)                # PNG 16:9 a 150dpi -> 2400x1350
HEADER_BAND    = (0.85, 1.0)                # franja vertical del header (frac fig)


# ────────────────────────────────────────────────────────────────────────
# LOGO JO + helpers de imagen
# ────────────────────────────────────────────────────────────────────────

_LOGO_PATH = _REPO_ROOT / "figures" / "logos" / "jo_logo.png"
_JO_LOGO_SCALE = 0.75                       # JO al 75% del alto del escudo (mas discreto)

_ESCUDO_TARGET_PX = 100                     # alto al que se normalizan los logos antes del zoom


def _trim_transparent(img: np.ndarray, alpha_threshold: int = 10) -> np.ndarray:
    """Recorta filas / columnas exteriores con alpha <= threshold.

    Muchos PNG de escudos (Bundesliga 139x181 con 21px top/bot transparentes,
    luukhopman/football-logos) traen padding que reduce el alto VISIBLE del
    crest a ~75% del bbox. Sin recortar, draw_header normaliza al alto del
    bbox y el escudo se ve mas pequeno de lo esperado dentro del header band.
    """
    if img.ndim != 3 or img.shape[2] < 4:
        return img
    alpha = img[..., 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(ys) == 0:
        return img
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return img[y0:y1, x0:x1]


def _load_rgba(path, trim: bool = True) -> Optional[np.ndarray]:
    """Carga PNG como np.array RGBA. Maneja modo Palette de FotMob/sportlogos.

    Por defecto recorta padding transparente exterior (trim=True) pa que el
    escudo / logo llene el alto que se le pide en draw_header.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = np.asarray(Image.open(p).convert("RGBA"))
        return _trim_transparent(img) if trim else img
    except Exception:
        return None


def _normalize_img(img: np.ndarray, target_h: int = _ESCUDO_TARGET_PX) -> np.ndarray:
    """Redimensiona img a target_h px de alto (mantiene aspect ratio).

    Evita que logos con distinta resolucion source salgan de tamano diferente
    al aplicar el mismo zoom en draw_header.
    """
    h = img.shape[0]
    if h == target_h:
        return img
    w = img.shape[1]
    new_w = max(1, int(round(w * target_h / h)))
    return np.asarray(Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS))


# ────────────────────────────────────────────────────────────────────────
# draw_pitch — campo de futbol numpy puro (alternativa a mplsoccer)
# ────────────────────────────────────────────────────────────────────────

def draw_pitch(ax: plt.Axes,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               color: str = PITCH, lw: float = 1.0,
               margin: float = 3.0,
               margin_x: Optional[float] = None) -> None:
    """Dibuja un campo de futbol centrado en (0,0), en metros. Sin mplsoccer.

    Margenes:
      - margin   = metros de margen Y (arriba/abajo del campo)
      - margin_x = metros de margen X (laterales). None -> usa `margin`.
                   IMPORTANTE: las porterias van de +-L a +-(L+1.5), asi que
                   margin_x>=1.5 pa que no se corten los postes traseros.
    """
    if margin_x is None:
        margin_x = margin
    L, W = pitch_length / 2, pitch_width / 2

    # Borde exterior
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], color=color, lw=lw, zorder=2)
    # Linea de medio campo
    ax.plot([0, 0], [-W, W], color=color, lw=lw, zorder=2)
    # Circulo central (FIFA 9.15 m)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec=color, lw=lw, zorder=2))
    ax.plot(0, 0, "o", ms=3, color=color, zorder=2)

    pa_w, pa_d = 20.16, 16.5                # area grande
    ga_w, ga_d = 9.16, 5.5                  # area pequena

    for sign in (-1, 1):
        x0, x1 = sign * L, sign * (L - pa_d)
        ax.plot([x0, x1, x1, x0], [-pa_w, -pa_w, pa_w, pa_w], color=color, lw=lw, zorder=2)
        x2 = sign * (L - ga_d)
        ax.plot([x0, x2, x2, x0], [-ga_w, -ga_w, ga_w, ga_w], color=color, lw=lw, zorder=2)
        ax.plot(sign * (L - 11), 0, "o", ms=3, color=color, zorder=2)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
        arc_x = sign * (L - 11) + 9.15 * np.cos(theta) * sign
        arc_y = 9.15 * np.sin(theta)
        arc_x[np.abs(arc_x) > (L - pa_d)] = np.nan
        ax.plot(arc_x, arc_y, color=color, lw=lw, zorder=2)
        for yy in (-3.66, 3.66):
            ax.plot([sign * L, sign * (L + 1.5)], [yy, yy], color=color, lw=lw * 1.5, zorder=2)
        ax.plot([sign * (L + 1.5), sign * (L + 1.5)], [-3.66, 3.66],
                color=color, lw=lw * 1.5, zorder=2)

    ax.set_xlim(-L - margin_x, L + margin_x)
    ax.set_ylim(-W - margin, W + margin)
    ax.set_aspect("equal")
    ax.axis("off")


# ────────────────────────────────────────────────────────────────────────
# draw_header — escudo IZQ | titulo + subtitulo CENTRO | JO DCHA
# ────────────────────────────────────────────────────────────────────────
_TEXT_FILL_FACTOR = 1.2     # divisor pa convertir alto en pt -> fontsize mpl


def draw_header(fig: plt.Figure, *, title: str,
                subtitle=None,
                escudo_path=None,
                hdr_band: Tuple[float, float] = HEADER_BAND,
                escudo_x: float = 0.04, text_x: float = 0.13, jo_x: float = 0.96,
                logo_h_frac: float = 0.62,
                text_h_frac: float = 0.5,
                title_share: float = 2.0 / 3.0,
                title_size: Optional[float] = None,
                sub_size: Optional[float] = None) -> None:
    """Header PPCF-style (heredado de CCV): escudo grande IZQ + titulo +
    subtitulo (1 o N lineas) + JO DCHA.

    Proporciones (todo relativo al alto del header band):
      - Logos (escudo + JO):  centrados vertical, alto = logo_h_frac del band
      - Bloque titulo + sub:  centrado vertical, alto = text_h_frac del band
      - Dentro del bloque:    titulo = title_share del alto; sub = (1-title_share)

    Si subtitle es lista/tupla, pinta cada elemento como una linea separada
    debajo del titulo (mismo color, mismo fontsize), con el espacio
    proporcionalmente repartido.
    """
    bot, top = hdr_band
    h = top - bot
    fig_h_inches = fig.get_size_inches()[1]
    band_h_inches = h * fig_h_inches

    # Normaliza subtitle a lista de lineas
    if subtitle is None or subtitle == "":
        sub_lines: list[str] = []
    elif isinstance(subtitle, (list, tuple)):
        sub_lines = [str(s) for s in subtitle if s]
    else:
        sub_lines = [str(subtitle)]
    n_sub = max(1, len(sub_lines))

    # Bloque de texto centrado vertical en el band
    sub_share = 1.0 - title_share
    block_bot_frac = 0.5 - text_h_frac / 2.0
    block_top_frac = 0.5 + text_h_frac / 2.0
    boundary_frac = block_bot_frac + sub_share * text_h_frac

    title_center_frac = (boundary_frac + block_top_frac) / 2.0
    title_y = bot + title_center_frac * h
    jo_y    = bot + 0.5 * h           # JO siempre centrado en el band
    # escudo_y se calcula MAS ABAJO una vez sabemos sub_size + n_sub
    # (escudo se alinea al centro del bloque titulo+subtitulos, NO al band).

    if title_size is None:
        title_size = (title_share * text_h_frac * band_h_inches) * 72.0 / _TEXT_FILL_FACTOR + 1
    if sub_size is None:
        # Una sola sub-line ocupa todo el sub_share del bloque (NO dividir por
        # n_sub — eso encogeria el texto a 5-6pt cuando hay 2 lineas). Las
        # lineas extra se apilan debajo extendiendo el bloque mas alla del
        # text_h_frac, absorbidas por el pad header->pitch.
        # +2 = bump explicito de +1pt sobre el +1pt original (subtitulos
        # mas legibles en el header).
        sub_size = (sub_share * text_h_frac * band_h_inches) * 72.0 / _TEXT_FILL_FACTOR + 2

    # Zoom de logos pa que midan logo_h_frac del band
    target_logo_px = logo_h_frac * band_h_inches * fig.dpi

    # ── Centro vertical del bloque de texto (titulo + sub lines) ──
    # Cuando hay subtitulos, el bloque de texto cae mas abajo que el centro
    # del band (el titulo arriba, las subs apiladas debajo). El escudo se
    # alinea a ese centro pa que visualmente acompane al texto y no flote
    # encima de el.
    if sub_lines:
        line_h_inches = sub_size * _TEXT_FILL_FACTOR / 72.0
        line_h_frac = line_h_inches / fig_h_inches
        last_sub_y = title_y - (title_size * _TEXT_FILL_FACTOR / 72.0) / fig_h_inches * 0.55 - (n_sub - 0.5) * line_h_frac
        escudo_y = (title_y + last_sub_y) / 2.0
    else:
        escudo_y = title_y

    # Escudo IZQ
    if escudo_path is not None:
        img = _load_rgba(escudo_path)
        if img is not None:
            img = _normalize_img(img)
            escudo_zoom = target_logo_px / img.shape[0]
            ab = AnnotationBbox(
                OffsetImage(img, zoom=escudo_zoom),
                (escudo_x, escudo_y),
                frameon=False,
                xycoords="figure fraction",
                box_alignment=(0.0, 0.5),
            )
            ab.set_clip_on(False)
            fig.add_artist(ab)

    # Titulo
    fig.text(text_x, title_y, title, ha="left", va="center",
             color=TEXT, fontsize=title_size, fontweight=700)

    # Subtitulo(s) — apiladas bajo el titulo con altura de linea fija
    # derivada del sub_size en pulgadas (asi N lineas se ven legibles
    # aunque rebasen el text_h_frac nominal). Reusamos line_h_frac
    # computado arriba pa anclar el escudo.
    if sub_lines:
        sub_top_y = title_y - (title_size * _TEXT_FILL_FACTOR / 72.0) / fig_h_inches * 0.55
        for i, line in enumerate(sub_lines):
            sub_y = sub_top_y - (i + 0.5) * line_h_frac
            fig.text(text_x, sub_y, line, ha="left", va="center",
                     color=TEXT, fontsize=sub_size, fontweight=500)

    # Logo JO DCHA — centrado en el band (no en el bloque de texto)
    img_jo = _load_rgba(_LOGO_PATH)
    if img_jo is not None:
        img_jo = _normalize_img(img_jo)
        jo_zoom = (target_logo_px * _JO_LOGO_SCALE) / img_jo.shape[0]
        ab = AnnotationBbox(
            OffsetImage(img_jo, zoom=jo_zoom),
            (jo_x, jo_y),
            frameon=False,
            xycoords="figure fraction",
            box_alignment=(1.0, 0.5),
        )
        ab.set_clip_on(False)
        fig.add_artist(ab)


# ────────────────────────────────────────────────────────────────────────
# add_logo — LEGACY: estampa el logo JO en una esquina
# ────────────────────────────────────────────────────────────────────────

def add_logo(fig: plt.Figure, width_frac: float = 0.11,
             margin: float = 0.012, corner: str = "br") -> None:
    """Estampa el logo JO en una esquina (axes dedicado).

    Args:
      width_frac:  ancho del logo en fraccion de figura
      margin:      separacion del borde de la figura en fraccion
      corner:      "br"=bottom-right, "bl"=bottom-left, "tr"=top-right, "tl"=top-left
    """
    if not _LOGO_PATH.exists():
        return
    try:
        img = plt.imread(str(_LOGO_PATH))
        logo_aspect = img.shape[1] / img.shape[0]
        figW, figH = fig.get_size_inches()
        h_frac = width_frac * (figW / figH) / logo_aspect
        x = margin if corner in ("bl", "tl") else 1.0 - margin - width_frac
        y = margin if corner in ("bl", "br") else 1.0 - margin - h_frac
        ax_logo = fig.add_axes([x, y, width_frac, h_frac])
        ax_logo.imshow(img)
        ax_logo.axis("off")
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────
# draw_football — Telstar (pentagonos sobre circulo blanco)
# ────────────────────────────────────────────────────────────────────────

def draw_football(ax: plt.Axes, cx: float, cy: float,
                  r: float = 0.65, zorder: int = 10) -> None:
    """Balon de futbol patron Telstar en coords del axes.

    El axes debe tener set_aspect('equal') si se quiere un balon redondo.
      r:        radio del balon (en las mismas unidades que el axes)
      zorder:   profundidad
    """
    n = 5
    ball = mpatches.Circle((cx, cy), r, facecolor="white", edgecolor="black",
                           linewidth=1.5, zorder=zorder)
    ax.add_patch(ball)
    rp = r * 0.37
    c_ang = np.array([np.pi / 2 + 2 * np.pi * i / n for i in range(n)])
    pts_c = np.column_stack([cx + rp * np.cos(c_ang), cy + rp * np.sin(c_ang)])
    cp = mpatches.Polygon(pts_c, facecolor="black", edgecolor="none",
                          zorder=zorder + 1)
    cp.set_clip_path(ball)
    ax.add_patch(cp)
    r_outer, rp2 = r * 0.73, r * 0.27
    for i in range(n):
        ang = c_ang[i]
        ox = cx + r_outer * np.cos(ang)
        oy = cy + r_outer * np.sin(ang)
        o_ang = np.array([ang + 2 * np.pi * k / n for k in range(n)])
        pts_o = np.column_stack([ox + rp2 * np.cos(o_ang), oy + rp2 * np.sin(o_ang)])
        op = mpatches.Polygon(pts_o, facecolor="black", edgecolor="none",
                              zorder=zorder + 1)
        op.set_clip_path(ball)
        ax.add_patch(op)


def _draw_football_ellipse(ax: plt.Axes, cx: float, cy: float,
                           rx: float, ry: float, zorder: int = 10) -> None:
    """Balon Telstar via Ellipse — pa axes no-cuadrados (footer blocks).

    rx, ry son los radios en coords del axes (0-1 si transAxes).
    """
    n = 5
    ball = mpatches.Ellipse((cx, cy), 2 * rx, 2 * ry,
                            facecolor="white", edgecolor="black",
                            linewidth=1.5, zorder=zorder)
    ax.add_patch(ball)
    c_ang = np.array([np.pi / 2 + 2 * np.pi * i / n for i in range(n)])
    pts_c = np.column_stack([cx + rx * 0.37 * np.cos(c_ang),
                             cy + ry * 0.37 * np.sin(c_ang)])
    cp = mpatches.Polygon(pts_c, facecolor="black", edgecolor="none",
                          zorder=zorder + 1)
    cp.set_clip_path(ball)
    ax.add_patch(cp)
    for i in range(n):
        ang = c_ang[i]
        ox = cx + rx * 0.73 * np.cos(ang)
        oy = cy + ry * 0.73 * np.sin(ang)
        o_ang = np.array([ang + 2 * np.pi * k / n for k in range(n)])
        pts_o = np.column_stack([ox + rx * 0.27 * np.cos(o_ang),
                                 oy + ry * 0.27 * np.sin(o_ang)])
        op = mpatches.Polygon(pts_o, facecolor="black", edgecolor="none",
                              zorder=zorder + 1)
        op.set_clip_path(ball)
        ax.add_patch(op)


# ────────────────────────────────────────────────────────────────────────
# draw_footer_legend — 3 bloques 40/20/40 reusable por todos los renders
# ────────────────────────────────────────────────────────────────────────
# Layout heredado del PPCF de CCV:
#   bloque 1 (40%) - barra gradiente con un cmap + 2 labels extremos + descripcion
#   bloque 2 (20%) - triangulos "Direccion de Ataque"
#   bloque 3 (40%) - leyenda de equipos (atacante + defensor) + balon Telstar

def _make_block_axes(fig, left, bottom, width, height):
    """Sub-axes limpio (sin spines, lim 0-1) pa pintar leyenda en axes-frac."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def draw_footer_legend(
    fig: plt.Figure,
    footer_box: Tuple[float, float, float, float],
    *,
    cmap=PPCF_CMAP,
    bar_left_label: str = "100% Defender",
    bar_right_label: str = "100% Attacker",
    bar_desc: str = "Probability that the team controls this point of the pitch",
    att_team_name: str = "Attacker",
    def_team_name: str = "Defender",
    att_color: str = ATT,
    def_color: str = DEF,
    attacking_right: bool = True,
) -> None:
    """Footer de 3 bloques 40/20/40 (heredado de CCV) anclado a `footer_box`.

    Args:
      footer_box: (left, bottom, width, height) en fraccion de figura.
      cmap:       colormap pa el gradiente del bloque 1.
      bar_*:      textos del gradiente (extremos + descripcion arriba).
      *_team_*:   nombre + color de cada equipo en el bloque 3.
      attacking_right: True -> triangulos hacia DCHA en el bloque 2.
    """
    fl, fb, fw, fh = footer_box
    b1_w, b2_w, b3_w = fw * 0.40, fw * 0.20, fw * 0.40
    ax_b1 = _make_block_axes(fig, fl, fb, b1_w, fh)
    ax_b2 = _make_block_axes(fig, fl + b1_w, fb, b2_w, fh)
    ax_b3 = _make_block_axes(fig, fl + b1_w + b2_w, fb, b3_w, fh)

    # ── BLOCK 1: barra gradiente horizontal + etiquetas ──────────────
    bar_y = 0.75
    bar_h = 0.20
    bar_x0, bar_x1 = 0.10, 0.90
    n_steps = 200
    xs = np.linspace(bar_x0, bar_x1, n_steps + 1)
    for i in range(n_steps):
        c = cmap(i / (n_steps - 1))
        ax_b1.add_patch(mpatches.Rectangle(
            (xs[i], bar_y - bar_h / 2), xs[i + 1] - xs[i], bar_h,
            facecolor=c, edgecolor="none",
            transform=ax_b1.transAxes, zorder=2))
    ax_b1.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y - bar_h / 2), bar_x1 - bar_x0, bar_h,
        facecolor="none", edgecolor=TEXT, lw=0.7,
        transform=ax_b1.transAxes, zorder=3))
    ax_b1.text(bar_x0, bar_y - bar_h / 2 - 0.1, bar_left_label,
               ha="left", va="top", fontsize=12, color=LEGEND,
               transform=ax_b1.transAxes, fontweight="bold")
    ax_b1.text(bar_x1, bar_y - bar_h / 2 - 0.1, bar_right_label,
               ha="right", va="top", fontsize=12, color=LEGEND,
               transform=ax_b1.transAxes, fontweight="bold")
    if bar_desc:
        ax_b1.text(0.5, bar_y + bar_h / 2 + 0.05, bar_desc,
                   ha="center", va="bottom", fontsize=11, color=LEGEND,
                   transform=ax_b1.transAxes)

    # ── BLOCK 2: triangulos Direccion de Ataque ──────────────────────
    n_tri, tri_w, tri_h, spacing = 4, 0.10, 0.16, 0.06
    total_w = n_tri * tri_w + (n_tri - 1) * spacing
    start_x = (1.0 - total_w) / 2.0
    y_tri = 0.76
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
            verts, closed=True, facecolor=att_color, edgecolor="none",
            alpha=alpha, transform=ax_b2.transAxes, zorder=10))
    ax_b2.text(0.5, 0.5, "Attacking Direction",
               ha="center", va="center", fontsize=12, color=LEGEND,
               transform=ax_b2.transAxes, fontweight="bold")

    # ── BLOCK 3: equipos (atacante azul + defensor rojo) + balon Telstar
    y_node = 0.75
    for x_node, x_txt, color, name in (
            (0.07, 0.125, att_color, att_team_name),
            (0.40, 0.455, def_color, def_team_name)):
        ax_b3.plot(x_node, y_node, "o", ms=22, color=color,
                   markeredgecolor=TEXT, markeredgewidth=1.0,
                   alpha=0.95, transform=ax_b3.transAxes,
                   clip_on=False, zorder=5)
        ax_b3.text(x_txt, y_node, name, ha="left", va="center",
                   fontsize=12, color=LEGEND, fontweight="bold",
                   transform=ax_b3.transAxes)
    # Balon Telstar (Ellipse pa compensar aspect no cuadrado del bloque)
    W_in, H_in = fig.get_size_inches()
    b3_aspect = (W_in * b3_w) / (H_in * fh)
    r_y = 0.125
    r_x = r_y / b3_aspect
    x_ball = 0.78
    _draw_football_ellipse(ax_b3, x_ball, y_node, r_x, r_y, zorder=5)
    ax_b3.text(x_ball + r_x + 0.04, y_node, "Ball", ha="left", va="center",
               fontsize=12, color=LEGEND, fontweight="bold",
               transform=ax_b3.transAxes)
