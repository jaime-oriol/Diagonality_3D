"""One-off: re-render los posters estaticos del paper en LIGHT OPTA.

Genera:
  deliverable/figures/Vision.png  — frame estatico del Kane goal 5 (vision FOV)
  deliverable/figures/PPCF.png    — frame estatico del Kane goal 5 (PPCF reach-field)
  deliverable/figures/DOS.png     — frame estatico del Kane goal 5 (DOS + shadow)
  deliverable/figures/Michael_Olise_Bayern_Hamburg.png  — pass map Olise

Estos son los PNGs que el paper main.pdf y los slides exec embeben.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from src.loader import load_match_info, compute_attacking_right
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball, load_cached_events, load_cached_metadata,
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
from src.viz.passes_plot import plot_player_passes
from src.viz.common import BG

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218
FPS = 50

print("Loading skeleton + dynamics for Kane goal 5 frame...")
info = load_match_info(MATCH)
def _gk(pls, lbl):
    for p in pls:
        if p.get("position")=="TW" and p.get("starting"):
            n=p.get("shirt_number")
            if isinstance(n,int) and n>0: return n
    raise RuntimeError(lbl)
gk = {0: _gk(info["away_players"],"away"), 1: _gk(info["home_players"],"home")}

events = load_cached_events(MATCH)
shots = events[events["event_type"]=="shot"].copy()
shots["d"] = (shots["parquet_frame"]-GOAL_F).abs()
shot = shots.nsmallest(1,"d").iloc[0]
att = 1 if shot["team_id"]==info["home_team_id"] else 0
md = load_cached_metadata(MATCH); hgk_left_p2 = bool(md["home_gk_left"][2])
attacking_right = hgk_left_p2 if att==1 else (not hgk_left_p2)

skel = load_cached_skeleton(MATCH)
# Necesitamos ~3s antes pa el scanning memory del DOS
ss = skel[skel["frame_number"].between(GOAL_F - 200, GOAL_F + 5)]
del skel
ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori); del ss, ori
raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
dyn.loc[raw_speed > 12.0, ["vx","vy"]] = float("nan")
for col in ("vx","vy"):
    dyn[col] = (dyn.groupby(["team","jersey"], sort=False)[col]
                .transform(lambda s: pd.Series(savgol_filter(s.fillna(0).values, 13, 1), index=s.index)))

ball = load_cached_ball(MATCH)
fo = dyn[dyn["frame_number"] == GOAL_F]
bf = ball[ball["frame_number"] == GOAL_F]
bx, by = float(bf.iloc[0]["x"]), float(bf.iloc[0]["y"])

# ── 1. VISION poster ──────────────────────────────────────────────────
print("Rendering Vision poster...")
fig, ax = plt.subplots(figsize=(16, 10.4))
fig.set_facecolor(BG)
plot_vision_frame(
    fo, focus_team=1, focus_jersey=9, ball_x=bx, ball_y=by,
    att_team=1, smoothing=7.0, ax=ax,
    save_path="deliverable/figures/Vision.png",
)
plt.close(fig)
print("  Vision.png OK")

# ── 2. PPCF poster ────────────────────────────────────────────────────
print("Rendering PPCF poster...")
ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
    fo, att, params=ppcf_default_params(), n_grid_x=100)
fig, ax = plt.subplots(figsize=(16, 10.4))
fig.set_facecolor(BG)
plot_ppcf_frame(
    fo, attacking_team=att, ball_xy=(bx,by), gk_jerseys=gk,
    ppcf_att=ppcf_att, ppcf_def=ppcf_def, ax=ax,
    save_path="deliverable/figures/PPCF.png",
)
plt.close(fig)
print("  PPCF.png OK")

# ── 3. DOS poster (con shadow layer) ──────────────────────────────────
print("Rendering DOS poster (con scanning memory)...")
timeline = build_possession_timeline(events, info, gap_fill_max_frames=300)
in_w = timeline[(timeline["frame_end"] >= GOAL_F-200) & (timeline["frame_start"] <= GOAL_F)].sort_values("frame_start")
if len(in_w):
    if timeline.loc[in_w.index[0], "frame_start"] > GOAL_F-200:
        timeline.loc[in_w.index[0], "frame_start"] = GOAL_F-200
    if timeline.loc[in_w.index[-1], "frame_end"] < GOAL_F:
        timeline.loc[in_w.index[-1], "frame_end"] = GOAL_F
cfg = ScanningMemoryConfig(memory_window_s=2.5, tau_decay_s=1.2,
                            vision_smoothing=2.0, framerate=FPS)
memories = compute_scanning_memory_sequence(dyn, timeline, (GOAL_F-200, GOAL_F), cfg)

dos_surf, _, _, _, _ = compute_dos_surface(
    fo, att, (bx,by), attacking_right,
    params=dos_default_params(), n_grid_x=100, n_directions=24, vision_smoothing=1.0)
n_y = int(round(100 * PITCH_WIDTH / PITCH_LENGTH))
dx, dy_ = PITCH_LENGTH/100, PITCH_WIDTH/n_y
XG = np.arange(100)*dx - PITCH_LENGTH/2 + dx/2
YG = np.arange(n_y)*dy_ - PITCH_WIDTH/2 + dy_/2
fm = memories.get(GOAL_F)
mem = resample_memory_to_grid(fm.memory, XG, YG) if fm else np.zeros_like(dos_surf, dtype=np.float32)
dos_pos = np.clip(dos_surf, 0.0, None).astype(np.float32)
gated = dos_pos * mem
fwd = compute_forward_cone_mask((bx,by), attacking_right, XG, YG, max_dist_m=35.0)
shadow = dos_pos * (1.0 - mem) * fwd
BLUR = 1.5 / dx
gated = gaussian_filter(gated, sigma=BLUR, mode="constant", cval=0.0).astype(np.float32)
shadow = gaussian_filter(shadow, sigma=BLUR, mode="constant", cval=0.0).astype(np.float32)

fig, ax = plt.subplots(figsize=(16, 10.4))
fig.set_facecolor(BG)
plot_dos_frame(
    fo, attacking_team=att, ball_xy=(bx,by), attacking_right=attacking_right,
    dos_surface=gated, scanning_memory=np.ones_like(gated, dtype=np.float32),
    shadow_surface=shadow, shadow_noise_floor=0.003, shadow_display_max=0.025, shadow_alpha_max=0.55,
    gk_jerseys=gk, noise_floor=0.0005, display_max=0.015, ax=ax,
    save_path="deliverable/figures/DOS.png",
)
plt.close(fig)
print("  DOS.png OK")

# ── 4. Michael Olise pass map ─────────────────────────────────────────
print("Rendering Olise pass map...")
OLISE_ID = "DFL-OBJ-J01R3R"
def _cls(a):
    if pd.isna(a): return "unknown"
    a = abs(float(a))
    if a <= 22.5: return "forward"
    if a <= 67.5: return "diagonal"
    if a <= 112.5: return "sideways"
    return "backward"
hgk_left_p1 = bool(md["home_gk_left"][1])
p = events[(events["event_type"]=="pass") & (events["player_id"]==OLISE_ID)].copy().dropna(subset=["x","y","x_receiver","y_receiver"])
hid = info.get("home_team_id","")
b_int = 1 if info["home_team_id"]==hid else 0
flip = np.array([not compute_attacking_right(b_int, int(r["half"]), hgk_left_p1) for _, r in p.iterrows()])
for c in ("x","y","x_receiver","y_receiver"): p.loc[flip, c] = -p.loc[flip, c]
p["direction_class"] = p["play_angle"].apply(_cls)
p["successful"] = p["evaluation"].isin({"successfullyCompleted","successful"})
plot_player_passes(p, title="Michael Olise · Passes",
    subtitle=["Bundesliga 2025-26","Bayern Munich 5-0 Hamburger SV (13 September 2025)"],
    attacking_right=True, team_logo_path="figures/logos/bayern_munich.png",
    save_path="deliverable/figures/Michael_Olise_Bayern_Hamburg.png")
print("  Olise pass map OK")
print("\nAll posters done.")
