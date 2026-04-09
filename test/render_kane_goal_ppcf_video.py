"""Render Kane goal 5 PPCF video. Max quality: 50fps native, n_grid_x=100, dpi=200.

Same sequence as render_kane_goal.py (vision video) but with the Immediate
Orientation-Aware PPCF reach-field overlay instead of the vision model.
~35s of play (20s before + 15s after the shot frame).
"""
import sys; sys.path.insert(0, ".")
import pandas as pd
import matplotlib, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import load_cached_skeleton, load_cached_ball, load_cached_events
from src.ppcf import compute_ppcf_surfaces, default_params
from src.viz.ppcf_plot import plot_ppcf_frame, BG

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218  # Kane goal 5 shot frame
FPS = 50
WINDOW_PRE = 20   # seconds before shot
WINDOW_POST = 15  # seconds after shot

start_f = GOAL_F - WINDOW_PRE * FPS
end_f = GOAL_F + WINDOW_POST * FPS

# --- Metadata + GK identification ---
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
SMOOTH_W, POLY = 13, 1   # 13 frames @ 50Hz = 0.26s window
MAX_SPEED = 12.0          # m/s cap (physically impossible above this)
raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
for col in ("vx", "vy"):
    dyn[col] = (dyn.groupby(["team", "jersey"], sort=False)[col]
                .transform(lambda s: pd.Series(
                    savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                    index=s.index)))
print(f"  {len(dyn):,} orientation rows (velocity smoothed, w={SMOOTH_W})")

ball = load_cached_ball(MATCH)
vframes = sorted(f for f in dyn["frame_number"].unique() if start_f <= f <= end_f)
print(f"  {len(vframes)} render frames")

# --- Pre-compute PPCF params (shared across frames) ---
params = default_params()
N_GRID = 100

# --- Render ---
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

    # Pre-compute PPCF surfaces (separate from rendering for clarity)
    ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
        fo, attacking_team, params=params, n_grid_x=N_GRID,
    )

    plot_ppcf_frame(
        fo,
        attacking_team=attacking_team,
        ball_xy=ball_xy,
        gk_jerseys=gk_jerseys,
        ppcf_att=ppcf_att,
        ppcf_def=ppcf_def,
        ax=ax,
    )

print("Rendering (max quality — this will take a while)...")
t0 = time.time()
ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
ani.save(
    "test/ppcf_kane_goal5.mp4",
    fps=FPS,
    dpi=200,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"],
)
plt.close()
dt = time.time() - t0
print(f"Saved: test/ppcf_kane_goal5.mp4 ({len(vframes)} frames, {dt:.0f}s, {dt/len(vframes):.2f}s/frame)")
