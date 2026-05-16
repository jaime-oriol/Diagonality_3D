"""Render single PNG frames at the action moment for each highlighted
event in `outputs/tables/highlight_events.json`. Cheap alternative to
the video renders — for the slide deck and frame gallery.

Per event, two PNGs named {match}_{player}_{type}_{frame}_{layer}.png:
  outputs/frames/..._dos.png    DOS + shadow at event_frame
  outputs/frames/..._ppcf.png   PPCF reach-field at event_frame

Both use the same player markers + shoulder bars + head wedge style as
the video renders, so the gallery is visually consistent with the videos.

Resume: skips any output PNG already on disk.
"""

import sys
sys.path.insert(0, ".")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball, load_cached_events,
)
from src.dos import compute_dos_surface, default_params
from src.ppcf import compute_ppcf_surfaces, PITCH_LENGTH, PITCH_WIDTH
from src.possession import build_possession_timeline
from src.scanning import (
    ScanningMemoryConfig, compute_scanning_memory_sequence,
    resample_memory_to_grid,
)
from src.viz.dos_plot import plot_dos_frame, compute_forward_cone_mask, BG
from src.viz.ppcf_plot import plot_ppcf_frame


FPS = 50
N_GRID = 100
N_DIRS = 24
SHADOW_MAX_DIST_M = 35.0
SMOOTH_W, POLY = 13, 1
MAX_SPEED = 12.0
SCAN_PRE_S = 2.5      # need this much history for the scanning gate

GALLERY = Path("outputs/tables/highlight_events.json")
OUT_DIR = Path("outputs/frames")


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


def _render_dos_png(event: dict, out_path: Path,
                     dyn: pd.DataFrame, ball: pd.DataFrame,
                     events_all: pd.DataFrame, info: dict,
                     gk_jerseys: dict):
    """Single-frame DOS PNG with the scanning gate + shadow layer."""
    f = int(event["frame"])
    attacking_team = int(event["attacking_team"])
    attacking_right = bool(event["attacking_right"])

    fo = dyn[dyn["frame_number"] == f]
    if len(fo) < 6:
        print(f"  [WARN] Only {len(fo)} skeletons at frame {f} — skipped")
        return False
    bf = ball[ball["frame_number"] == f]
    bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else 0.0
    by = float(bf.iloc[0]["y"]) if len(bf) > 0 else 0.0
    ball_xy = (bx, by)

    # Scanning memory needs the segment context, so build a small
    # timeline and a tiny memory window covering [f - 2.5s, f].
    scan_start = f - int(SCAN_PRE_S * FPS)
    timeline = build_possession_timeline(events_all, info,
                                          gap_fill_max_frames=300)
    in_w = timeline[
        (timeline["frame_end"] >= scan_start) & (timeline["frame_start"] <= f)
    ].sort_values("frame_start")
    if len(in_w):
        first_idx = in_w.index[0]
        last_idx = in_w.index[-1]
        if timeline.loc[first_idx, "frame_start"] > scan_start:
            timeline.loc[first_idx, "frame_start"] = scan_start
        if timeline.loc[last_idx, "frame_end"] < f:
            timeline.loc[last_idx, "frame_end"] = f
    cfg = ScanningMemoryConfig(memory_window_s=2.5, tau_decay_s=1.2,
                                vision_smoothing=2.0, framerate=FPS)
    memories = compute_scanning_memory_sequence(
        dyn, timeline, (scan_start, f), cfg)

    dos_surf, _, _, _, _ = compute_dos_surface(
        fo, attacking_team, ball_xy, attacking_right,
        params=default_params(), n_grid_x=N_GRID, n_directions=N_DIRS,
        vision_smoothing=1.0,
    )

    n_grid_y = int(round(N_GRID * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / N_GRID
    dy = PITCH_WIDTH / n_grid_y
    XGRID = np.arange(N_GRID) * dx - PITCH_LENGTH / 2 + dx / 2
    YGRID = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2 + dy / 2
    BLUR_CELLS = 1.5 / dx

    fm = memories.get(f)
    if fm is not None:
        mem_on_dos = resample_memory_to_grid(fm.memory, XGRID, YGRID)
    else:
        mem_on_dos = np.zeros_like(dos_surf, dtype=np.float32)

    dos_pos = np.clip(dos_surf, 0.0, None).astype(np.float32)
    gated = dos_pos * mem_on_dos
    forward_mask = compute_forward_cone_mask(
        ball_xy, attacking_right, XGRID, YGRID, max_dist_m=SHADOW_MAX_DIST_M)
    shadow = dos_pos * (1.0 - mem_on_dos) * forward_mask

    gated = gaussian_filter(gated, sigma=BLUR_CELLS,
                            mode="constant", cval=0.0).astype(np.float32)
    shadow = gaussian_filter(shadow, sigma=BLUR_CELLS,
                             mode="constant", cval=0.0).astype(np.float32)

    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)
    plot_dos_frame(
        fo,
        attacking_team=attacking_team,
        ball_xy=ball_xy,
        attacking_right=attacking_right,
        dos_surface=gated,
        scanning_memory=np.ones_like(gated, dtype=np.float32),
        shadow_surface=shadow,
        gk_jerseys=gk_jerseys,
        noise_floor=0.0005, display_max=0.015,
        shadow_noise_floor=0.003, shadow_display_max=0.025,
        shadow_alpha_max=0.55,
        ax=ax,
        save_path=str(out_path),
    )
    plt.close(fig)
    return True


def _render_ppcf_png(event: dict, out_path: Path,
                      dyn: pd.DataFrame, ball: pd.DataFrame,
                      gk_jerseys: dict):
    """Single-frame PPCF reach-field PNG."""
    f = int(event["frame"])
    attacking_team = int(event["attacking_team"])

    fo = dyn[dyn["frame_number"] == f]
    if len(fo) < 6:
        return False
    bf = ball[ball["frame_number"] == f]
    bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else None
    by = float(bf.iloc[0]["y"]) if len(bf) > 0 else None
    ball_xy = (bx, by) if bx is not None else None

    ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
        fo, attacking_team, params=default_params(), n_grid_x=N_GRID)

    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)
    plot_ppcf_frame(
        fo, attacking_team=attacking_team, ball_xy=ball_xy,
        gk_jerseys=gk_jerseys, ppcf_att=ppcf_att, ppcf_def=ppcf_def,
        ax=ax, save_path=str(out_path),
    )
    plt.close(fig)
    return True


def _frame_filename(event: dict, layer: str) -> str:
    pname = (event.get("player_name") or "unk").replace(" ", "_").replace("/", "")
    et = event.get("event_type", "ev")
    f = event.get("frame", 0)
    return f"{event['match']}_{pname}_{et}_{f}_{layer}.png"


def main():
    if not GALLERY.exists():
        raise SystemExit(f"Missing {GALLERY}. Run select_top_events.py first.")
    events = json.loads(GALLERY.read_text())
    print(f"Loaded {GALLERY}: {len(events)} events")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by match so we load each match's data ONCE
    by_match: dict = {}
    for ev in events:
        by_match.setdefault(ev["match"], []).append(ev)

    n_done = 0; n_skip = 0; n_fail = 0
    t_total = time.time()

    for match, evs in by_match.items():
        print(f"\n[{match}] {len(evs)} events")
        # Determine the global frame window we need for this match
        frames = [int(e["frame"]) for e in evs]
        f_min = min(frames) - int(SCAN_PRE_S * FPS) - 25
        f_max = max(frames) + 25

        info = load_match_info(match)
        gk_jerseys = _resolve_gks(info)

        skel = load_cached_skeleton(match)
        ss = skel[skel["frame_number"].between(f_min, f_max)]
        del skel
        if len(ss) == 0:
            print(f"  [WARN] No skeleton for window — match skipped")
            continue
        ori = compute_orientations(ss, smooth=True)
        dyn = add_dynamics(ori)
        del ss, ori
        dyn = _smooth_velocities(dyn)
        ball = load_cached_ball(match)
        events_all = load_cached_events(match)

        for ev in evs:
            for layer, fn in [("dos", _render_dos_png),
                              ("ppcf", _render_ppcf_png)]:
                out = OUT_DIR / _frame_filename(ev, layer)
                if out.exists() and out.stat().st_size > 50_000:
                    n_skip += 1
                    continue
                t0 = time.time()
                try:
                    if layer == "dos":
                        ok = fn(ev, out, dyn, ball, events_all, info, gk_jerseys)
                    else:
                        ok = fn(ev, out, dyn, ball, gk_jerseys)
                except Exception as e:
                    print(f"  [ERROR] {out.name}: {e}")
                    n_fail += 1
                    continue
                if ok:
                    n_done += 1
                    print(f"  + {out.name} ({time.time()-t0:.1f}s)")
                else:
                    n_fail += 1

    print(f"\nDONE: {n_done} new, {n_skip} skipped, {n_fail} failed "
          f"({(time.time()-t_total)/60:.1f} min)")


if __name__ == "__main__":
    main()
