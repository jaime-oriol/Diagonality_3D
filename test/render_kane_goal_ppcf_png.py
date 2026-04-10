"""Render Kane goal 5 PPCF test PNG. Single frame, max quality.

Uses the repo's canonical data flow:
    - load_cached_* for skeleton / ball / metadata (not raw parquet reads)
    - load_match_info for reliable GK identification (PlayingPosition="TW")
    - infer_attacking_team from the ball + skeleton (not hardcoded team id)
    - compute_orientations + add_dynamics from src.orientation
    - plot_ppcf_frame from src.viz.ppcf_plot (Opta-style aesthetic)
"""
import sys; sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball, load_cached_events,
)
from src.viz.ppcf_plot import plot_ppcf_frame

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218  # Kane goal 5 shot frame


def _find_gk(players: list, team_label: str) -> int:
    """Pull the starting GK's shirt number from a list of players (from
    load_match_info), using the official PlayingPosition code 'TW'
    (Torwart = goalkeeper). Raises if no GK found — no silent fallback.
    """
    for p in players:
        if p.get("position") == "TW" and p.get("starting"):
            n = p.get("shirt_number")
            if isinstance(n, int) and n > 0:
                return n
    raise RuntimeError(f"No starting GK (PlayingPosition='TW') found for {team_label}")


print("Loading match info...")
info = load_match_info(MATCH)

home_gk = _find_gk(info["home_players"], "home")
away_gk = _find_gk(info["away_players"], "away")
gk_jerseys = {0: away_gk, 1: home_gk}  # team 0 = away, 1 = home
print(f"  GK jerseys: home={home_gk} ({info.get('home_team_name', '?')}), "
      f"away={away_gk} ({info.get('away_team_name', '?')})")

# Attacking team: pull directly from the shot event at GOAL_F via team_id.
# Nearest-player inference fails at shot frames because a blocking defender
# can be closer to the ball than the shooter.
print("Resolving attacking team from shot event...")
events = load_cached_events(MATCH)
shots = events[events["event_type"] == "shot"].copy()
shots["frame_diff"] = (shots["parquet_frame"] - GOAL_F).abs()
shot_row = shots.nsmallest(1, "frame_diff").iloc[0]
shot_team_id = shot_row["team_id"]  # DFL-CLU-*** string
# Map DFL team_id to (0, 1) via match info
home_team_id = info.get("home_team_id", "")
away_team_id = info.get("away_team_id", "")
if shot_team_id == home_team_id:
    attacking_team = 1
elif shot_team_id == away_team_id:
    attacking_team = 0
else:
    raise RuntimeError(f"Shot team_id {shot_team_id} matches neither home nor away")
att_name = info["home_team_name"] if attacking_team == 1 else info["away_team_name"]
print(f"  Shot at frame {int(shot_row['parquet_frame'])} (delta {int(shot_row['frame_diff'])})")
print(f"  Attacking team: {attacking_team} ({att_name})")

print("Loading cached skeleton + ball...")
skel = load_cached_skeleton(MATCH)
ss = skel[skel["frame_number"].between(GOAL_F - 50, GOAL_F + 5)]
del skel
print(f"  {len(ss):,} skeleton rows in window")

ball = load_cached_ball(MATCH)

print("Computing orientations + dynamics...")
ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori)
del ss, ori

# Frame slice
frame = GOAL_F
fo = dyn[dyn["frame_number"] == frame]
bf = ball[ball["frame_number"] == frame]
bx = float(bf.iloc[0]["x"])
by = float(bf.iloc[0]["y"])
print(f"  Ball position: ({bx:.2f}, {by:.2f})")
print(f"  Players in frame: {len(fo)}")

print("Rendering PPCF frame (n_grid_x=100, window=2.5s)...")
fig = plot_ppcf_frame(
    fo,
    attacking_team=attacking_team,
    ball_xy=(bx, by),
    gk_jerseys=gk_jerseys,
    n_grid_x=100,
    alpha_max=1.0,
    title="",
    save_path="test/ppcf_kane_goal5.png",
)
print("Saved: test/ppcf_kane_goal5.png")
