"""Render Kane goal 5 DOS video with the on-ball-FOV scanning gate.

Pipeline:
  1. Load skeleton + events for the rendering window.
  2. Build a frame-exact possession timeline from carries + passes
     (linked to receptions via play_id). This tells us, at every frame,
     which player's FOV the DOS should be gated by.
  3. Compute the scanning memory for the whole sequence ONCE: per frame,
     the on-ball player's full Bekkers FOV plus a 2.5s exponentially
     decayed memory of his own scanning. This is the cognitive layer
     that turns the DOS from a god-eye metric into an actionable one.
  4. For each render frame, compute the DOS surface (24 directions) and
     gate it by the scanning memory resampled to the DOS grid. Render
     with absolute thresholds (no per-frame renormalization), so the
     only visible animation is real model dynamics — zero flicker.

TRAMPA in this script: orientations are backfilled with 150 synthetic
historical frames per player so the memory looks fully populated from
the very first rendered frame. The honest fix is to bump
preprocess.PRE_WINDOW_FRAMES from 50 to ~150 and rerun preprocess.
"""

import sys; sys.path.insert(0, ".")
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball,
    load_cached_events, load_cached_metadata,
)
from src.dos import compute_dos_surface, default_params
from src.possession import build_possession_timeline
from src.scanning import (
    ScanningMemoryConfig,
    compute_scanning_memory_sequence,
    backfill_orientations,
    resample_memory_to_grid,
)
from src.viz.dos_plot import plot_dos_frame, BG

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218  # Kane goal 5 shot frame
FPS = 50
WINDOW_PRE = 20   # seconds before shot
WINDOW_POST = 15  # seconds after shot

start_f = GOAL_F - WINDOW_PRE * FPS
end_f = GOAL_F + WINDOW_POST * FPS

# --- Metadata + GK identification ---
info = load_match_info(MATCH)
metadata = load_cached_metadata(MATCH)


def _find_gk(players, label):
    for p in players:
        if p.get("position") == "TW" and p.get("starting"):
            n = p.get("shirt_number")
            if isinstance(n, int) and n > 0:
                return n
    raise RuntimeError(f"No starting GK found for {label}")


home_gk = _find_gk(info["home_players"], "home")
away_gk = _find_gk(info["away_players"], "away")
gk_jerseys = {0: away_gk, 1: home_gk}

# --- Attacking team from shot event ---
events = load_cached_events(MATCH)
shots = events[events["event_type"] == "shot"].copy()
shots["frame_diff"] = (shots["parquet_frame"] - GOAL_F).abs()
shot_team_id = shots.nsmallest(1, "frame_diff").iloc[0]["team_id"]
home_team_id = info.get("home_team_id", "")
attacking_team = 1 if shot_team_id == home_team_id else 0
attacking_right = True  # Bayern attacks right in this half

print(f"Video: {(end_f - start_f) / FPS:.0f}s @ {FPS}fps native")
print(f"  Attacking: team {attacking_team} ({info.get('home_team_name') if attacking_team == 1 else info.get('away_team_name')})")
print(f"  GKs: home={home_gk}, away={away_gk}")

# --- Load data ---
print("Loading skeleton...")
skel = load_cached_skeleton(MATCH)
ss = skel[skel["frame_number"].between(start_f - 5, end_f + 5)]
del skel
print(f"  {len(ss):,} skeleton rows")

print("Computing orientations + dynamics...")
ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori)
del ss, ori

# Smooth velocities: cap outliers + Savitzky-Golay filter per player
from scipy.signal import savgol_filter
SMOOTH_W, POLY = 13, 1
MAX_SPEED = 12.0
raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
for col in ("vx", "vy"):
    dyn[col] = (dyn.groupby(["team", "jersey"], sort=False)[col]
                .transform(lambda s: pd.Series(
                    savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                    index=s.index)))
print(f"  {len(dyn):,} orientation rows (velocity smoothed, w={SMOOTH_W})")

# TRAMPA: pad each player's history with 150 synthetic frames so the
# scanning memory is fully populated from frame 1. Replace by re-cache
# with PRE_WINDOW_FRAMES=150 for production-quality renders.
BACKFILL_FRAMES = 150
dyn = backfill_orientations(dyn, lookback_frames=BACKFILL_FRAMES)
print(f"  Backfilled +{BACKFILL_FRAMES} frames per player (TRAMPA)")

ball = load_cached_ball(MATCH)
vframes = sorted(f for f in dyn["frame_number"].unique()
                 if start_f <= f <= end_f)
print(f"  {len(vframes)} render frames")

# --- Possession timeline (data-driven from kpi_data) ---
print("Building possession timeline from carries + passes...")
timeline = build_possession_timeline(events, metadata)
in_window = timeline[
    (timeline["frame_end"] >= start_f) & (timeline["frame_start"] <= end_f)
]
print(f"  {len(in_window)} possession segments in window")
print(f"  Modes: {dict(in_window['mode'].value_counts())}")

# --- Scanning memory (FULL detail, 50Hz, real Bekkers vision per frame) ---
SCAN_CONFIG = ScanningMemoryConfig(
    memory_window_s=2.5,
    tau_decay_s=1.2,
    vision_smoothing=2.0,  # 2.0 = balance speed/quality (210x136 grid)
    framerate=FPS,
)
print(f"Computing scanning memory ({SCAN_CONFIG.memory_window_s}s window, "
      f"tau={SCAN_CONFIG.tau_decay_s}s, smoothing={SCAN_CONFIG.vision_smoothing})...")
t0 = time.time()
memories = compute_scanning_memory_sequence(
    dyn, timeline, (start_f, end_f), SCAN_CONFIG)
print(f"  {len(memories)} frame memories computed ({time.time()-t0:.0f}s)")

# --- DOS params ---
params = default_params()
N_GRID = 50          # DOS grid resolution
N_DIRS = 24          # 24 candidate directions (every 15 deg)
DOS_VISION_SM = 1.0  # vision smoothing INSIDE compute_dos_surface (cheap)

# Pre-compute DOS grid coords for memory resampling
from src.ppcf import PITCH_LENGTH, PITCH_WIDTH
import numpy as np
N_GRID_Y = int(round(N_GRID * PITCH_WIDTH / PITCH_LENGTH))
DX = PITCH_LENGTH / N_GRID
DY = PITCH_WIDTH / N_GRID_Y
XGRID = np.arange(N_GRID) * DX - PITCH_LENGTH / 2 + DX / 2
YGRID = np.arange(N_GRID_Y) * DY - PITCH_WIDTH / 2 + DY / 2

# --- Render ---
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

    dos_surf, _, best_dir, _, _ = compute_dos_surface(
        fo, attacking_team, ball_xy, attacking_right,
        params=params, n_grid_x=N_GRID, n_directions=N_DIRS,
        vision_smoothing=DOS_VISION_SM,
    )

    # Resample scanning memory onto the DOS grid (or zeros if unknown owner)
    fm = memories.get(f)
    if fm is not None:
        memory_on_dos = resample_memory_to_grid(fm.memory, XGRID, YGRID)
    else:
        memory_on_dos = np.zeros_like(dos_surf, dtype=np.float32)

    plot_dos_frame(
        fo,
        attacking_team=attacking_team,
        ball_xy=ball_xy,
        attacking_right=attacking_right,
        gk_jerseys=gk_jerseys,
        dos_surface=dos_surf,
        best_direction=best_dir,
        scanning_memory=memory_on_dos,
        absolute_threshold=0.003,
        display_max=0.025,
        ax=ax,
    )


print("Rendering DOS video (this will take a LONG while)...")
t0 = time.time()
ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
ani.save(
    "test/dos_kane_goal5.mp4",
    fps=FPS,
    dpi=200,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"],
)
plt.close()
dt = time.time() - t0
print(f"Saved: test/dos_kane_goal5.mp4 ({len(vframes)} frames, {dt:.0f}s, "
      f"{dt/len(vframes):.2f}s/frame)")
