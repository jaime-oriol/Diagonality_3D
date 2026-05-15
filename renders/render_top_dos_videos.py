"""Render DOS videos for the top events selected by `select_top_events.py`.

DOS + scanning-gate render pipeline:
  - skeleton + ball window around event_frame
  - orientations + add_dynamics + Savitzky-Golay velocity smoothing
  - possession timeline (built from kpi_data)
  - scanning memory sequence (Bekkers FOV + 2.5s decayed memory)
  - DOS surface per frame (visible + shadow layers)
  - spatial gaussian blur + temporal EMA, hard reset on owner change
  - ffmpeg encode at the same crf=18, dpi=200, 50fps

Per-event window: WINDOW_PRE seconds before event, WINDOW_POST seconds
after — same as the Kane render. Output is named after the event so the
video filename itself carries the metadata for the storytelling slides.

Resume logic: any video that already exists on disk is skipped, so the
script is safe to re-run after a partial render.
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
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball, load_cached_events,
)
from src.dos import compute_dos_surface, default_params
from src.ppcf import PITCH_LENGTH, PITCH_WIDTH
from src.possession import build_possession_timeline
from src.scanning import (
    ScanningMemoryConfig,
    compute_scanning_memory_sequence,
    resample_memory_to_grid,
)
from src.viz.dos_plot import plot_dos_frame, compute_forward_cone_mask, BG


# ── Config (matches kane render so videos look consistent) ───────────────

WINDOW_PRE_S = 12          # seconds before event center
WINDOW_POST_S = 8          # seconds after
FPS = 50
N_GRID = 100
N_DIRS = 24
DOS_VISION_SM = 1.0
EMA_ALPHA = 0.10
SPATIAL_BLUR_SIGMA_M = 1.5
SHADOW_MAX_DIST_M = 35.0
SHADOW_NOISE_FLOOR = 0.003
SHADOW_DISPLAY_MAX = 0.025
SHADOW_ALPHA_MAX = 0.55
SMOOTH_W, POLY = 13, 1
MAX_SPEED = 12.0

EVENT_LIST = Path("outputs/tables/top_dos_events.json")
OUT_DIR = Path("outputs/videos")


# ── Per-event renderer ──────────────────────────────────────────────────

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
    attacking_right = bool(event["attacking_right"])

    start_f = event_frame - WINDOW_PRE_S * FPS
    end_f = event_frame + WINDOW_POST_S * FPS

    info = load_match_info(match)
    gk_jerseys = _resolve_gks(info)

    # Skeleton + dynamics
    skel = load_cached_skeleton(match)
    ss = skel[skel["frame_number"].between(start_f - 5, end_f + 5)]
    del skel
    if len(ss) == 0:
        print(f"  [WARN] No skeleton frames for {match} window {start_f}-{end_f} — skipped")
        return False
    ori = compute_orientations(ss, smooth=True)
    dyn = add_dynamics(ori)
    del ss, ori
    dyn = _smooth_velocities(dyn)

    ball = load_cached_ball(match)
    events_all = load_cached_events(match)

    vframes = sorted(f for f in dyn["frame_number"].unique()
                     if start_f <= f <= end_f)
    if not vframes:
        print(f"  [WARN] No render frames available — skipped")
        return False

    # Possession timeline
    timeline = build_possession_timeline(
        events_all, info, gap_fill_max_frames=300)

    # Extend window edges so gating is continuous
    in_w = timeline[
        (timeline["frame_end"] >= start_f) & (timeline["frame_start"] <= end_f)
    ].sort_values("frame_start")
    if len(in_w):
        first_idx = in_w.index[0]
        last_idx = in_w.index[-1]
        if timeline.loc[first_idx, "frame_start"] > start_f:
            timeline.loc[first_idx, "frame_start"] = start_f
        if timeline.loc[last_idx, "frame_end"] < end_f:
            timeline.loc[last_idx, "frame_end"] = end_f

    # Scanning memory
    cfg = ScanningMemoryConfig(
        memory_window_s=2.5, tau_decay_s=1.2,
        vision_smoothing=2.0, framerate=FPS,
    )
    memories = compute_scanning_memory_sequence(
        dyn, timeline, (start_f, end_f), cfg)

    # DOS grid coords
    n_grid_y = int(round(N_GRID * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / N_GRID
    dy = PITCH_WIDTH / n_grid_y
    XGRID = np.arange(N_GRID) * dx - PITCH_LENGTH / 2 + dx / 2
    YGRID = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2 + dy / 2
    BLUR_CELLS = SPATIAL_BLUR_SIGMA_M / dx
    params = default_params()

    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)
    state = {"vis": None, "sh": None, "owner": None}

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

        fm = memories.get(f)
        if fm is not None:
            mem_on_dos = resample_memory_to_grid(fm.memory, XGRID, YGRID)
            owner = (fm.owner_team, fm.owner_jersey)
        else:
            mem_on_dos = np.zeros_like(dos_surf, dtype=np.float32)
            owner = None

        dos_pos = np.clip(dos_surf, 0.0, None).astype(np.float32)
        gated = dos_pos * mem_on_dos
        forward_mask = compute_forward_cone_mask(
            ball_xy, attacking_right, XGRID, YGRID,
            max_dist_m=SHADOW_MAX_DIST_M)
        shadow = dos_pos * (1.0 - mem_on_dos) * forward_mask

        gated = gaussian_filter(gated, sigma=BLUR_CELLS,
                                mode="constant", cval=0.0).astype(np.float32)
        shadow = gaussian_filter(shadow, sigma=BLUR_CELLS,
                                 mode="constant", cval=0.0).astype(np.float32)

        if owner != state["owner"] or state["vis"] is None or state["vis"].shape != gated.shape:
            ema_v = gated.copy()
            ema_s = shadow.copy()
        else:
            ema_v = EMA_ALPHA * gated + (1.0 - EMA_ALPHA) * state["vis"]
            ema_s = EMA_ALPHA * shadow + (1.0 - EMA_ALPHA) * state["sh"]
        state["vis"] = ema_v; state["sh"] = ema_s; state["owner"] = owner

        plot_dos_frame(
            fo,
            attacking_team=attacking_team,
            ball_xy=ball_xy,
            attacking_right=attacking_right,
            dos_surface=ema_v,
            scanning_memory=np.ones_like(ema_v, dtype=np.float32),
            shadow_surface=ema_s,
            shadow_noise_floor=SHADOW_NOISE_FLOOR,
            shadow_display_max=SHADOW_DISPLAY_MAX,
            shadow_alpha_max=SHADOW_ALPHA_MAX,
            gk_jerseys=gk_jerseys,
            noise_floor=0.0005,
            display_max=0.015,
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
    """Filename encodes the rank and the event identity."""
    pname = (event.get("player_name") or "unknown").replace(" ", "_").replace("/", "")
    et = event.get("event_type", "ev")
    f = event.get("frame", 0)
    return f"top{i+1:02d}_{event['match']}_{pname}_{et}_{f}.mp4"


def _render_task(event_dict: dict, out_path_str: str) -> tuple:
    """Top-level wrapper picklable by ProcessPoolExecutor. Returns
    (status, message) where status in {'done', 'skip', 'fail'}."""
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
    events = json.loads(EVENT_LIST.read_text())

    # Parallelism: defaults to min(cpu_count, n_events). Each worker peaks
    # at ~4 GB RAM, so on m5.2xlarge (8 vCPU / 32 GB) all 8 cores fit
    # comfortably, on c5.9xlarge (36 vCPU / 72 GB) the cap is the event
    # count (10). Override with RENDER_WORKERS env if you need to throttle.
    default_n = min(os.cpu_count() or 4, len(events))
    n_workers = int(os.environ.get("RENDER_WORKERS", str(default_n)))
    n_workers = max(1, min(n_workers, len(events)))
    print(f"Loaded {EVENT_LIST}: {len(events)} events, parallelism={n_workers}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(ev, str(OUT_DIR / _video_filename(i, ev))) for i, ev in enumerate(events)]

    n_skip = 0; n_done = 0; n_fail = 0
    t_total = time.time()
    # spawn ctx so each worker boots a fresh matplotlib (Agg) instead of
    # forking the parent process — avoids inherited figure-manager state.
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
