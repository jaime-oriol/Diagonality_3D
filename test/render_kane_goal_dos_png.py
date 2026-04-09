"""Render DOS on Kane goal 5 frame. Shows diagonal opportunity surface
with direction arrows overlaid on player positions."""
import sys; sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import load_cached_skeleton, load_cached_ball, load_cached_events
from src.dos import compute_dos_surface
from src.ppcf import default_params
from src.viz.dos_plot import plot_dos_frame

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218
PRE_PASS_F = 3584976  # pass frame ~5s before goal, players still spread out

# --- Match info + GK ---
info = load_match_info(MATCH)

def _find_gk(players):
    for p in players:
        if p.get("position") == "TW" and p.get("starting"):
            n = p.get("shirt_number")
            if isinstance(n, int) and n > 0:
                return n
    return 1

home_gk = _find_gk(info["home_players"])
away_gk = _find_gk(info["away_players"])
gk_jerseys = {0: away_gk, 1: home_gk}

# --- Attacking team ---
events = load_cached_events(MATCH)
shots = events[events["event_type"] == "shot"].copy()
shots["frame_diff"] = (shots["parquet_frame"] - GOAL_F).abs()
shot_team_id = shots.nsmallest(1, "frame_diff").iloc[0]["team_id"]
home_team_id = info.get("home_team_id", "")
attacking_team = 1 if shot_team_id == home_team_id else 0
attacking_right = True  # Bayern attacks right in this half

# --- Load skeleton for this frame ---
print("Loading skeleton...")
skel = load_cached_skeleton(MATCH)
ss = skel[skel["frame_number"].between(PRE_PASS_F - 2, PRE_PASS_F + 2)]
del skel

ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori)
del ss, ori

fo = dyn[dyn["frame_number"] == PRE_PASS_F]
print(f"  {len(fo)} players in frame {PRE_PASS_F}")

# --- Ball ---
ball = load_cached_ball(MATCH)
bf = ball[ball["frame_number"] == PRE_PASS_F]
bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else 0.0
by = float(bf.iloc[0]["y"]) if len(bf) > 0 else 0.0
ball_xy = (bx, by)

# --- Compute DOS ---
print("Computing DOS surface (full Bekkers vision per defender)...")
params = default_params()
dos, baseline, best_dir, xg, yg = compute_dos_surface(
    fo, attacking_team, ball_xy, attacking_right,
    params=params, n_grid_x=50, n_directions=12, vision_smoothing=3.0,
)
print(f"  DOS range: [{dos.min():.4f}, {dos.max():.4f}]")
print(f"  Baseline PPCF range: [{baseline.min():.4f}, {baseline.max():.4f}]")

# --- Render ---
print("Rendering...")
fig = plot_dos_frame(
    fo, attacking_team, ball_xy, attacking_right,
    gk_jerseys=gk_jerseys,
    dos_surface=dos, best_direction=best_dir,
    alpha_max=0.9,
)
fig.savefig("test/dos_kane_goal5.png", dpi=400, facecolor=fig.get_facecolor(),
            bbox_inches="tight")
print("Saved: test/dos_kane_goal5.png")
