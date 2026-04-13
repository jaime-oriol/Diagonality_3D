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
  4. For each render frame, compute the DOS surface (24 directions),
     gate it by the resampled scanning memory, blur it spatially with a
     ~1.5m gaussian, and apply a temporal EMA across frames (alpha=0.10,
     half-life ~140 ms) for visual readability. Reset on owner change.
  5. Render with smoothstep visibility curve over a fixed display range,
     spline36 interpolation. No flicker, no on/off cliffs.

Requires the cache to be generated with PRE_WINDOW_FRAMES >= 150 in
preprocess.py so the 2.5s scanning lookback has full coverage. Run
test/regenerate_caches.py if the cache is older.
"""

import sys; sys.path.insert(0, ".")
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball, load_cached_events,
)
from src.dos import compute_dos_surface, default_params
from src.possession import build_possession_timeline
from src.scanning import (
    ScanningMemoryConfig,
    compute_scanning_memory_sequence,
    resample_memory_to_grid,
)
from src.viz.dos_plot import plot_dos_frame, compute_forward_cone_mask, BG

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218  # Kane goal 5 shot frame
FPS = 50
WINDOW_PRE = 20   # seconds before shot
WINDOW_POST = 15  # seconds after shot

start_f = GOAL_F - WINDOW_PRE * FPS
end_f = GOAL_F + WINDOW_POST * FPS

# --- Match info + GK identification ---
info = load_match_info(MATCH)


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

ball = load_cached_ball(MATCH)
vframes = sorted(f for f in dyn["frame_number"].unique()
                 if start_f <= f <= end_f)
print(f"  {len(vframes)} render frames")

# --- Possession timeline (data-driven from kpi_data) ---
# gap_fill_max_frames=300 -> any inter-event gap up to 6s is filled by
# extending the previous segment forward (covers transition phases that
# kpi_data doesn't represent as discrete events).
print("Building possession timeline from carries + passes...")
timeline = build_possession_timeline(events, info, gap_fill_max_frames=300)

# Extend the first/last segments in the window to its boundaries so the
# scanning gate has coverage all the way through (otherwise the celebration
# tail and any leading silence show no DOS at all).
in_window = timeline[
    (timeline["frame_end"] >= start_f) & (timeline["frame_start"] <= end_f)
].sort_values("frame_start")
if len(in_window):
    first_idx = in_window.index[0]
    last_idx = in_window.index[-1]
    if timeline.loc[first_idx, "frame_start"] > start_f:
        timeline.loc[first_idx, "frame_start"] = start_f
    if timeline.loc[last_idx, "frame_end"] < end_f:
        timeline.loc[last_idx, "frame_end"] = end_f
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
N_GRID = 100         # DOS grid resolution (1.05m cells)
N_DIRS = 24          # 24 candidate directions (every 15 deg)
DOS_VISION_SM = 1.0  # vision smoothing INSIDE compute_dos_surface (cheap)
EMA_ALPHA = 0.10     # temporal EMA on the gated DOS (newer-frame weight).
                     # alpha=0.10 -> half-life ~7 frames = 140ms at 50fps,
                     # tuned for visual readability (slower than biology).
SPATIAL_BLUR_SIGMA_M = 1.5   # gaussian blur on gated DOS, sigma in meters
                             # (~1.4 cells at N_GRID=100). Smooths the
                             # cell-level discontinuities created by the
                             # discrete direction sampling without
                             # erasing the macroscopic blob structure.

# Shadow DOS: "unseen but ahead of ball" layer. Gold cells = diagonal
# opportunities in the player's offensive arc that he is NOT scanning.
SHADOW_MAX_DIST_M = 35.0       # realistic pass / carry reach
SHADOW_NOISE_FLOOR = 0.003     # 6x higher than visible -> only strong
                               # unseen opportunities light up (no noise)
SHADOW_DISPLAY_MAX = 0.025
SHADOW_ALPHA_MAX = 0.55        # secondary visual weight vs visible DOS

# Pre-compute DOS grid coords for memory resampling
from src.ppcf import PITCH_LENGTH, PITCH_WIDTH
import numpy as np
N_GRID_Y = int(round(N_GRID * PITCH_WIDTH / PITCH_LENGTH))
DX = PITCH_LENGTH / N_GRID
DY = PITCH_WIDTH / N_GRID_Y
XGRID = np.arange(N_GRID) * DX - PITCH_LENGTH / 2 + DX / 2
YGRID = np.arange(N_GRID_Y) * DY - PITCH_WIDTH / 2 + DY / 2
# Spatial blur sigma in CELLS (so it scales naturally with N_GRID).
BLUR_SIGMA_CELLS = SPATIAL_BLUR_SIGMA_M / DX

# --- Render ---
fig, ax = plt.subplots(figsize=(16, 10.4))
fig.set_facecolor(BG)

# EMA state for the gated DOS and the shadow DOS, plus the previous
# owner so we reset the temporal smoothing whenever possession switches
# (otherwise the new player's view would be contaminated by the previous
# owner's gating or by shadows that were relevant to the previous carrier).
_ema_state = {"visible": None, "shadow": None, "owner": None}


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
        vision_smoothing=DOS_VISION_SM,
    )

    # Resample scanning memory onto the DOS grid + identify owner
    fm = memories.get(f)
    if fm is not None:
        memory_on_dos = resample_memory_to_grid(fm.memory, XGRID, YGRID)
        owner = (fm.owner_team, fm.owner_jersey)
    else:
        memory_on_dos = np.zeros_like(dos_surf, dtype=np.float32)
        owner = None

    # Apply gate now (so the EMA operates on the final, comparable signal).
    # Visible layer: DOS the player sees or has scanned recently.
    dos_pos = np.clip(dos_surf, 0.0, None).astype(np.float32)
    gated_now = dos_pos * memory_on_dos

    # Shadow layer: DOS the player does NOT see BUT is ahead of the ball
    # and within realistic pass/carry range (no distraction in own half).
    # Same blur + EMA pipeline as visible, applied to a different signal.
    forward_mask = compute_forward_cone_mask(
        ball_xy, attacking_right, XGRID, YGRID,
        max_dist_m=SHADOW_MAX_DIST_M,
    )
    shadow_now = dos_pos * (1.0 - memory_on_dos) * forward_mask

    # Spatial smoothing: gaussian blur (sigma in meters) to soften the
    # cell-level discontinuities from direction discretization. Applied
    # independently to both layers.
    gated_now = gaussian_filter(gated_now, sigma=BLUR_SIGMA_CELLS,
                                mode="constant", cval=0.0).astype(np.float32)
    shadow_now = gaussian_filter(shadow_now, sigma=BLUR_SIGMA_CELLS,
                                 mode="constant", cval=0.0).astype(np.float32)

    # Temporal EMA per layer, hard reset on owner change.
    prev_v = _ema_state["visible"]
    prev_s = _ema_state["shadow"]
    if (owner != _ema_state["owner"]
            or prev_v is None or prev_v.shape != gated_now.shape):
        ema_vis = gated_now.copy()
        ema_sh = shadow_now.copy()
    else:
        ema_vis = EMA_ALPHA * gated_now + (1.0 - EMA_ALPHA) * prev_v
        ema_sh = EMA_ALPHA * shadow_now + (1.0 - EMA_ALPHA) * prev_s
    _ema_state["visible"] = ema_vis
    _ema_state["shadow"] = ema_sh
    _ema_state["owner"] = owner

    # Both surfaces are ALREADY gated/masked, blurred and EMA-smoothed.
    # Pass an all-ones scanning_memory so the renderer's smoothstep
    # operates directly on ema_vis; pass ema_sh as the shadow layer.
    plot_dos_frame(
        fo,
        attacking_team=attacking_team,
        ball_xy=ball_xy,
        attacking_right=attacking_right,
        dos_surface=ema_vis,
        scanning_memory=np.ones_like(ema_vis, dtype=np.float32),
        shadow_surface=ema_sh,
        shadow_noise_floor=SHADOW_NOISE_FLOOR,
        shadow_display_max=SHADOW_DISPLAY_MAX,
        shadow_alpha_max=SHADOW_ALPHA_MAX,
        gk_jerseys=gk_jerseys,
        noise_floor=0.0005,
        display_max=0.015,
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
