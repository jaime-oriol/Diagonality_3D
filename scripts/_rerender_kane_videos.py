"""One-off: re-render the 3 Kane goal 5 videos (DOS / PPCF / Vision) en
identidad LIGHT OPTA (white BG, Chakra Petch, ATT blue / DEF red, Telstar ball).

Carga skeleton + dyn UNA SOLA VEZ y los reusa para los 3 renders. Los 3
mp4s sobrescriben los originales en `results/renders/`. La salida previa
esta ya backupeada en `results/renders_backup_20260526_0828/`.
"""
import sys; sys.path.insert(0, ".")
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball, load_cached_events,
)
from src.dos import compute_dos_surface, default_params as dos_default_params
from src.ppcf import compute_ppcf_surfaces, default_params as ppcf_default_params
from src.ppcf import PITCH_LENGTH, PITCH_WIDTH
from src.possession import build_possession_timeline
from src.scanning import (
    ScanningMemoryConfig, compute_scanning_memory_sequence,
    resample_memory_to_grid,
)
from src.viz.dos_plot import plot_dos_frame, compute_forward_cone_mask
from src.viz.ppcf_plot import plot_ppcf_frame
from src.viz.vision_plot import plot_vision_frame
from src.viz.common import BG

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218
FPS = 50
WINDOW_PRE = 20
WINDOW_POST = 15

start_f = GOAL_F - WINDOW_PRE * FPS
end_f = GOAL_F + WINDOW_POST * FPS

print(f"[{time.strftime('%H:%M:%S')}] Loading match info + cache...")
info = load_match_info(MATCH)

def _find_gk(players, label):
    for p in players:
        if p.get("position") == "TW" and p.get("starting"):
            n = p.get("shirt_number")
            if isinstance(n, int) and n > 0:
                return n
    raise RuntimeError(f"No starting GK for {label}")

gk_jerseys = {0: _find_gk(info["away_players"], "away"),
              1: _find_gk(info["home_players"], "home")}

events = load_cached_events(MATCH)
shots = events[events["event_type"] == "shot"].copy()
shots["d"] = (shots["parquet_frame"] - GOAL_F).abs()
shot_team_id = shots.nsmallest(1, "d").iloc[0]["team_id"]
home_id = info.get("home_team_id", "")
attacking_team = 1 if shot_team_id == home_id else 0
attacking_right = True  # Bayern ataca a la derecha en este half

print(f"  attacking_team={attacking_team} ({info.get('home_team_name') if attacking_team==1 else info.get('away_team_name')})")
print(f"  GKs: home={gk_jerseys[1]}, away={gk_jerseys[0]}")

print(f"[{time.strftime('%H:%M:%S')}] Loading skeleton + dynamics...")
skel = load_cached_skeleton(MATCH)
ss = skel[skel["frame_number"].between(start_f - 5, end_f + 5)]
del skel
ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori)
del ss, ori

SMOOTH_W, POLY, MAX_SPEED = 13, 1, 12.0
raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
for col in ("vx", "vy"):
    dyn[col] = (dyn.groupby(["team", "jersey"], sort=False)[col]
                .transform(lambda s: pd.Series(
                    savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                    index=s.index)))

ball = load_cached_ball(MATCH)
vframes = sorted(f for f in dyn["frame_number"].unique() if start_f <= f <= end_f)
print(f"  {len(dyn):,} orientation rows, {len(vframes)} render frames")


# ────────────────────────────────────────────────────────────────────────
# 1) VISION video — sencillo, pa que arranque rapido
# ────────────────────────────────────────────────────────────────────────
def render_vision():
    print(f"\n[{time.strftime('%H:%M:%S')}] Rendering VISION video...")
    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)

    def render(i):
        ax.clear()
        f = vframes[i]
        fo = dyn[dyn["frame_number"] == f]
        bf = ball[ball["frame_number"] == f]
        bx = bf.iloc[0]["x"] if len(bf) > 0 else None
        by = bf.iloc[0]["y"] if len(bf) > 0 else None
        try:
            plot_vision_frame(
                fo, focus_team=1, focus_jersey=9,
                ball_x=bx, ball_y=by,
                att_team=1, smoothing=7.0,
                ax=ax,
            )
        except ValueError:
            ax.clear()
            ax.set_facecolor(BG)

    t0 = time.time()
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save("results/renders/vision_kane_goal5.mp4", fps=FPS, dpi=200,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)
    print(f"  vision OK ({time.time()-t0:.0f}s)")


# ────────────────────────────────────────────────────────────────────────
# 2) PPCF video
# ────────────────────────────────────────────────────────────────────────
def render_ppcf():
    print(f"\n[{time.strftime('%H:%M:%S')}] Rendering PPCF video...")
    params = ppcf_default_params()
    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)

    def render(i):
        ax.clear()
        f = vframes[i]
        fo = dyn[dyn["frame_number"] == f]
        bf = ball[ball["frame_number"] == f]
        bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else None
        by = float(bf.iloc[0]["y"]) if len(bf) > 0 else None
        ball_xy = (bx, by) if bx is not None else None
        ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
            fo, attacking_team, params=params, n_grid_x=100)
        plot_ppcf_frame(
            fo, attacking_team=attacking_team, ball_xy=ball_xy,
            gk_jerseys=gk_jerseys,
            ppcf_att=ppcf_att, ppcf_def=ppcf_def,
            ax=ax,
        )

    t0 = time.time()
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save("results/renders/ppcf_kane_goal5.mp4", fps=FPS, dpi=200,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)
    print(f"  ppcf OK ({time.time()-t0:.0f}s)")


# ────────────────────────────────────────────────────────────────────────
# 3) DOS video (mas pesado — possession + scanning memory)
# ────────────────────────────────────────────────────────────────────────
def render_dos():
    print(f"\n[{time.strftime('%H:%M:%S')}] Building possession + scanning memory...")
    timeline = build_possession_timeline(events, info, gap_fill_max_frames=300)
    in_w = timeline[(timeline["frame_end"] >= start_f) & (timeline["frame_start"] <= end_f)].sort_values("frame_start")
    if len(in_w):
        if timeline.loc[in_w.index[0], "frame_start"] > start_f:
            timeline.loc[in_w.index[0], "frame_start"] = start_f
        if timeline.loc[in_w.index[-1], "frame_end"] < end_f:
            timeline.loc[in_w.index[-1], "frame_end"] = end_f

    cfg = ScanningMemoryConfig(memory_window_s=2.5, tau_decay_s=1.2,
                                vision_smoothing=2.0, framerate=FPS)
    t0 = time.time()
    memories = compute_scanning_memory_sequence(dyn, timeline, (start_f, end_f), cfg)
    print(f"  scanning memory OK ({time.time()-t0:.0f}s)")

    N_GRID, N_DIRS = 100, 24
    EMA_ALPHA, SPATIAL_BLUR_SIGMA_M = 0.10, 1.5
    SHADOW_MAX_DIST_M = 35.0
    n_grid_y = int(round(N_GRID * PITCH_WIDTH / PITCH_LENGTH))
    dx, dy = PITCH_LENGTH/N_GRID, PITCH_WIDTH/n_grid_y
    XGRID = np.arange(N_GRID)*dx - PITCH_LENGTH/2 + dx/2
    YGRID = np.arange(n_grid_y)*dy - PITCH_WIDTH/2 + dy/2
    BLUR_CELLS = SPATIAL_BLUR_SIGMA_M / dx
    params = dos_default_params()

    state = {"vis": None, "sh": None, "owner": None}
    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)

    def render(i):
        ax.clear()
        f = vframes[i]
        fo = dyn[dyn["frame_number"] == f]
        bf = ball[ball["frame_number"] == f]
        bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else 0.0
        by = float(bf.iloc[0]["y"]) if len(bf) > 0 else 0.0
        ball_xy = (bx, by)
        dos_surf, _, _, _, _ = compute_dos_surface(
            fo, attacking_team, ball_xy, attacking_right,
            params=params, n_grid_x=N_GRID, n_directions=N_DIRS,
            vision_smoothing=1.0,
        )
        fm = memories.get(f)
        if fm is not None:
            mem = resample_memory_to_grid(fm.memory, XGRID, YGRID)
            owner = (fm.owner_team, fm.owner_jersey)
        else:
            mem = np.zeros_like(dos_surf, dtype=np.float32)
            owner = None
        dos_pos = np.clip(dos_surf, 0.0, None).astype(np.float32)
        gated = dos_pos * mem
        fwd = compute_forward_cone_mask(ball_xy, attacking_right, XGRID, YGRID, max_dist_m=SHADOW_MAX_DIST_M)
        shadow = dos_pos * (1.0 - mem) * fwd
        gated = gaussian_filter(gated, sigma=BLUR_CELLS, mode="constant", cval=0.0).astype(np.float32)
        shadow = gaussian_filter(shadow, sigma=BLUR_CELLS, mode="constant", cval=0.0).astype(np.float32)
        if owner != state["owner"] or state["vis"] is None or state["vis"].shape != gated.shape:
            ema_v = gated.copy(); ema_s = shadow.copy()
        else:
            ema_v = EMA_ALPHA*gated + (1.0-EMA_ALPHA)*state["vis"]
            ema_s = EMA_ALPHA*shadow + (1.0-EMA_ALPHA)*state["sh"]
        state["vis"]=ema_v; state["sh"]=ema_s; state["owner"]=owner
        plot_dos_frame(
            fo, attacking_team=attacking_team, ball_xy=ball_xy,
            attacking_right=attacking_right,
            dos_surface=ema_v,
            scanning_memory=np.ones_like(ema_v, dtype=np.float32),
            shadow_surface=ema_s,
            shadow_noise_floor=0.003, shadow_display_max=0.025, shadow_alpha_max=0.55,
            gk_jerseys=gk_jerseys,
            noise_floor=0.0005, display_max=0.015,
            ax=ax,
        )

    print(f"[{time.strftime('%H:%M:%S')}] Rendering DOS video...")
    t0 = time.time()
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save("results/renders/dos_kane_goal5.mp4", fps=FPS, dpi=200,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)
    print(f"  dos OK ({time.time()-t0:.0f}s)")


# Orden: vision (mas rapido) -> ppcf -> dos (mas pesado)
render_vision()
render_ppcf()
render_dos()
print(f"\n[{time.strftime('%H:%M:%S')}] ALL DONE")
