"""Render Bekkers vision videos for the top events.

Bekkers vision render pipeline (vision_plot.plot_vision_frame with full
FOV + occlusion at smoothing=7), parameterized to render ANY event from
`outputs/tables/top_dos_events.json`. The focus player
is the event's on-ball player (passer / carrier / take-on winner),
resolved via `MatchInformations.xml`.

Per event, ~20 seconds of footage centered on the action frame.

Resume: skips any video already on disk above 100 KB.
"""

import sys
sys.path.insert(0, ".")

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import load_cached_skeleton, load_cached_ball
from src.viz.vision_plot import plot_vision_frame, BG


WINDOW_PRE_S = 12
WINDOW_POST_S = 8
FPS = 50
SMOOTHING = 7.0      # Bekkers vision grid: 105*7 x 68*7 = 735 x 476
TOP_N = 8

EVENT_LIST = Path("outputs/tables/top_dos_events.json")
OUT_DIR = Path("outputs/videos")


def _resolve_player_team_jersey(player_id: str, info: dict) -> tuple:
    """Map a DFL-OBJ-* player_id to (team_int, jersey_int) using
    match_info. team_int is 1=home, 0=away (matching skeleton).
    Returns (None, None) if not found."""
    for team_int, key in [(1, "home_players"), (0, "away_players")]:
        for p in info.get(key, []) or []:
            if p.get("person_id") == player_id:
                jersey = p.get("shirt_number")
                if isinstance(jersey, int) and jersey > 0:
                    return team_int, jersey
    return None, None


def _resolve_attacking_team_int(team_id: str, info: dict) -> int:
    """team_id (DFL-CLU-*) -> team_int (0/1)."""
    if team_id == info.get("home_team_id"):
        return 1
    if team_id == info.get("away_team_id"):
        return 0
    return -1


def _render_one(event: dict, out_path: Path):
    match = event["match"]
    event_frame = int(event["frame"])
    player_id = event.get("player_id", "")
    team_id = event.get("team_id", "") or ""

    info = load_match_info(match)
    focus_team, focus_jersey = _resolve_player_team_jersey(player_id, info)
    if focus_team is None:
        # Some take-ons have player_id missing; fall back to attacking_team's
        # most-events skeleton. We just skip — vision needs a focus player.
        print(f"  [WARN] player_id '{player_id}' not in roster — skipped")
        return False

    att_team_int = int(event.get("attacking_team", focus_team))

    start_f = event_frame - WINDOW_PRE_S * FPS
    end_f = event_frame + WINDOW_POST_S * FPS

    skel = load_cached_skeleton(match)
    ss = skel[skel["frame_number"].between(start_f - 5, end_f + 5)]
    del skel
    if len(ss) == 0:
        print(f"  [WARN] No skeleton in window — skipped")
        return False
    ori = compute_orientations(ss, smooth=True)
    dyn = add_dynamics(ori)
    del ss, ori

    ball = load_cached_ball(match)
    vframes = sorted(f for f in dyn["frame_number"].unique()
                     if start_f <= f <= end_f)
    if not vframes:
        print(f"  [WARN] No render frames — skipped")
        return False

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
                fo,
                focus_team=focus_team, focus_jersey=focus_jersey,
                ball_x=bx, ball_y=by,
                att_team=att_team_int,
                smoothing=SMOOTHING,
                ax=ax,
            )
        except ValueError as e:
            # Player off the pitch at this frame — skip silently
            ax.clear()
            ax.set_facecolor(BG)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save(
        str(out_path),
        fps=FPS, dpi=200,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"],
    )
    plt.close(fig)
    return True


def _video_filename(i: int, event: dict) -> str:
    pname = (event.get("player_name") or "unknown").replace(" ", "_").replace("/", "")
    et = event.get("event_type", "ev")
    f = event.get("frame", 0)
    return f"top{i+1:02d}_vision_{event['match']}_{pname}_{et}_{f}.mp4"


def _render_task(event_dict: dict, out_path_str: str) -> tuple:
    """Top-level wrapper picklable by ProcessPoolExecutor."""
    out = Path(out_path_str)
    if out.exists() and out.stat().st_size > 100_000:
        return ("skip", f"{out.name} ({out.stat().st_size/1024/1024:.1f} MB)")
    t0 = time.time()
    try:
        ok = _render_one(event_dict, out)
    except Exception as e:
        return ("fail", f"{out.name}: {e}")
    if ok:
        return ("done", f"{out.name} ({time.time()-t0:.0f}s)")
    return ("fail", f"{out.name}: render returned False")


def main():
    if not EVENT_LIST.exists():
        raise SystemExit(f"Missing {EVENT_LIST}. Run select_top_events.py first.")
    events = json.loads(EVENT_LIST.read_text())[:TOP_N]

    default_n = min(os.cpu_count() or 4, len(events))
    n_workers = int(os.environ.get("RENDER_WORKERS", str(default_n)))
    n_workers = max(1, min(n_workers, len(events)))
    print(f"Loaded {EVENT_LIST}: rendering top {len(events)} as VISION "
          f"videos, parallelism={n_workers}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(ev, str(OUT_DIR / _video_filename(i, ev))) for i, ev in enumerate(events)]

    n_skip = n_done = n_fail = 0
    t_total = time.time()
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(_render_task, ev, op): op for ev, op in tasks}
        for fut in as_completed(futures):
            status, msg = fut.result()
            if status == "done":
                n_done += 1; print(f"  + {msg}")
            elif status == "skip":
                n_skip += 1; print(f"  ⤳ SKIP {msg}")
            else:
                n_fail += 1; print(f"  [FAIL] {msg}")

    print(f"\nDONE: {n_done} new, {n_skip} skipped, {n_fail} failed "
          f"({(time.time()-t_total)/60:.1f} min wall, {n_workers} workers)")


if __name__ == "__main__":
    main()
