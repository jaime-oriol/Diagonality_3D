"""dos_plot — Render Diagonal Opportunity Surfaces on pitch (LIGHT OPTA).

Dos capas con semantica cognitiva distinta:

  1. VISIBLE DOS (DOS_CMAP, cyan->purpura->magenta): oportunidades que el
     portador esta viendo o ha escaneado en los ultimos 2.5s.
     Gated por `dos * scanning_memory`.

  2. SHADOW DOS (SHADOW_CMAP, amber->rojo): oportunidades que el portador
     NO ve pero que estan delante del balon dentro del cono ofensivo.
     Gated por `dos * (1 - scanning_memory) * forward_cone`.
     Un shadowpass (Hamilton manifesto) es un pase que aterriza dentro
     de una celda amber.

Ambas capas usan smoothstep `t^2*(3-2t)` sobre `[noise_floor, display_max]`
absolutos -> color stable cross-frame, sin parpadeo.

El caller pre-procesa surfaces (blur + EMA) por frame y los pasa aqui.
La capa visible recibe `dos_surface` + un `scanning_memory` (use all-ones
si ya esta gated). La capa shadow es opcional; pasala ya enmascarada
con `compute_forward_cone_mask` pa restringir al cono ofensivo del portador.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from mplsoccer import Pitch

from .common import (
    BG, TEXT, PE_S, PKW, DOS_CMAP, SHADOW_CMAP,
    ATT as ATT_C, DEF as DEF_C,
    GK as GK_C, draw_football,
)

MS = 20      # diametro de marker pa lectura sobre el campo


def compute_forward_cone_mask(
    ball_xy: tuple,
    attacking_right: bool,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    max_dist_m: float = 35.0,
) -> np.ndarray:
    """Boolean mask (len(ygrid), len(xgrid)) de celdas que:
      - estan delante del balon en el eje ataque
      - estan a <= max_dist_m del balon (pase / conduccion realista)

    Se usa pa gatear la capa shadow del DOS y restringirla al arco
    ofensivo del portador (evita ruido de fondo en el campo defensivo).
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
    """Render one frame with DOS heatmap + players + ball sobre BG blanco.

    Args identicos a la version dark: solo cambia la paleta (BG blanco,
    lineas negras, edge negro en jugadores, dorsales blancos con halo).
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # ── Gate + smoothstep visibility curve ──────────────────────────
    dos_pos = np.clip(dos_surface, 0.0, None).astype(np.float32)
    dos_gated = dos_pos * scanning_memory.astype(np.float32)
    edge0 = float(noise_floor)
    edge1 = float(max(display_max, edge0 + 1e-9))
    t = np.clip((dos_gated - edge0) / (edge1 - edge0), 0.0, 1.0)
    dos_norm = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    rgba = DOS_CMAP(dos_norm)
    rgba[..., 3] = dos_norm * alpha_max

    # ── Shadow layer (opcional, bajo la visible) ────────────────────
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

    # ── Players ─────────────────────────────────────────────────────
    for _, p in orientations_frame.iterrows():
        t_id = int(p["team"])
        j = int(p["jersey"])
        x, y = float(p["x"]), float(p["y"])
        is_gk = (j == gk_jerseys.get(t_id, 1))
        color = GK_C if is_gk else (ATT_C if t_id == attacking_team else DEF_C)

        # Dot (edge negro pa contraste sobre fondo blanco)
        ax.plot(x, y, "o", ms=MS, color=color,
                markeredgecolor=TEXT, markeredgewidth=1.0,
                alpha=0.95, zorder=5)

        # Shoulder bar (orientacion del cuerpo)
        sa = p.get("shoulder_angle", np.nan)
        if not (isinstance(sa, float) and np.isnan(sa)):
            sw = float(p.get("shoulder_width", 0.45) or 0.45) * 9
            perp_l = sa + np.pi / 2
            perp_r = sa - np.pi / 2
            # GK shoulder bar oscuro sobre dot oscuro -> usar gris medio
            bar_color = "#555555" if is_gk else color
            ax.plot(
                [x + (sw / 2) * np.cos(perp_l), x + (sw / 2) * np.cos(perp_r)],
                [y + (sw / 2) * np.sin(perp_l), y + (sw / 2) * np.sin(perp_r)],
                color=bar_color, linewidth=8, alpha=0.75, zorder=4,
                solid_capstyle="round",
            )

        # Head wedge (mirada) — gris oscuro semi-trans pa lectura sobre el heatmap
        ha = p.get("head_angle", np.nan)
        if not (isinstance(ha, float) and np.isnan(ha)):
            wedge = Wedge(
                (x, y), 3.5,
                np.degrees(ha) - 22.5,
                np.degrees(ha) + 22.5,
                color="#333333", alpha=0.30, zorder=3,
            )
            ax.add_patch(wedge)

        # Jersey number — blanco con halo negro (PE_S) pa contraste maximo
        ax.text(x, y, str(j), color="white", fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=6,
                path_effects=PE_S)

    # ── Velocity arrows ─────────────────────────────────────────────
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

    # ── Ball (Telstar pattern, mismo render que la leyenda) ─────────
    if ball_xy is not None and not (np.isnan(ball_xy[0]) or np.isnan(ball_xy[1])):
        draw_football(ax, ball_xy[0], ball_xy[1], r=0.9, zorder=10)

    if title:
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold", pad=12)

    if save_path:
        fig.savefig(save_path, dpi=400, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")

    return fig
