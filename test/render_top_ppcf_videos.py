"""Render PPCF reach-field videos for the top events.

Same pipeline as `render_kane_goal_ppcf_video.py` (compute_ppcf_surfaces
on n_grid_x=100, plot via ppcf_plot.plot_ppcf_frame) but parameterized
to render ANY event from `outputs/tables/top_dos_events.json`.

PPCF videos are TEAM-level (not player-focused) so no per-event focus
lookup is needed — only the attacking team and the GK jerseys.

Per event, ~20 seconds of footage. Resume: skips files above 100 KB.
"""

import sys
sys.path.insert(0, ".")

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import load_cached_skeleton, load_cached_ball
from src.ppcf import compute_ppcf_surfaces, default_params
from src.viz.ppcf_plot import plot_ppcf_frame, BG


WINDOW_PRE_S = 12
WINDOW_POST_S = 8
FPS = 50
N_GRID = 100
SMOOTH_W, POLY = 13, 1
MAX_SPEED = 12.0
TOP_N = 8

EVENT_LIST = Path("outputs/tables/top_dos_events.json")
OUT_DIR = Path("outputs/videos")


def _resolve_gks(info: dict) -> dict:
    def _find(players, label):
        for p in players:
            if p.get("position") == "TW" and p.get("starting"):
                n = p.get("shirt_number")
                if isinstance(n, int) and n > 0:
                    return n
        for p in players:
            if p.get("position") == "TW":
                n = p.get("shirt_number")
                if isinstance(n, int) and n > 0:
                    return n
        raise RuntimeError(f"No GK for {label}")
    return {0: _find(info["away_players"], "away"),
            1: _find(info["home_players"], "home")}


def _smooth_velocities(dyn: pd.DataFrame) -> pd.DataFrame:
    raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
    dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
    for col in ("vx", "vy"):
        dyn[col] = (dyn.groupby(["team", "jersey"], sort=False)[col]
                    .transform(lambda s: pd.Series(
                        savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                        index=s.index)))
    return dyn


def _render_one(event: dict, out_path: Path):
    match = event["match"]
    event_frame = int(event["frame"])
    attacking_team = int(event["attacking_team"])
    info = load_match_info(match)
    gk_jerseys = _resolve_gks(info)

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
    dyn = _smooth_velocities(dyn)

    ball = load_cached_ball(match)
    vframes = sorted(f for f in dyn["frame_number"].unique()
                     if start_f <= f <= end_f)
    if not vframes:
        print(f"  [WARN] No render frames — skipped")
        return False

    params = default_params()
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
            fo, attacking_team, params=params, n_grid_x=N_GRID)

        plot_ppcf_frame(
            fo, attacking_team=attacking_team, ball_xy=ball_xy,
            gk_jerseys=gk_jerseys,
            ppcf_att=ppcf_att, ppcf_def=ppcf_def,
            ax=ax,
        )

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
    return f"top{i+1:02d}_ppcf_{event['match']}_{pname}_{et}_{f}.mp4"


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
    print(f"Loaded {EVENT_LIST}: rendering top {len(events)} as PPCF "
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
