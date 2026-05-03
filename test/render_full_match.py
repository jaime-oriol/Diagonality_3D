"""Full-match video renderer with parallel segments + ffmpeg concat.

Renders an entire match (all cached frames) as one of three video styles:
  - dos     : DOS + shadow surface, gated by on-ball player's scanning memory
  - ppcf    : Team-level immediate orientation-aware PPCF reach-field
  - vision  : Bekkers vision of a SINGLE focus player (e.g. Kane, Olise)

Splits the match into N segments (default = cpu_count), renders each in a
worker process via ProcessPoolExecutor (spawn ctx, fresh matplotlib per
worker), then concatenates the segment .mp4s with ffmpeg's concat demuxer
without re-encoding (`-c copy`).

Usage:
  python3 test/render_full_match.py --match Bayern_Hamburg --type dos
  python3 test/render_full_match.py --match Bayern_Hamburg --type ppcf
  python3 test/render_full_match.py --match Bayern_Hamburg --type vision --focus-player DFL-OBJ-J00ZZ3
  python3 test/render_full_match.py --match Frankfurt_Bayern --type vision --focus-player DFL-OBJ-J01R3R

Output (configurable via --out-dir, default outputs/full_match/):
  outputs/full_match/{match}_{type}[_focus-{player}].mp4

Per-frame render times (observed): DOS ~1.67 s, PPCF ~1.34 s, Vision ~1.07 s.
With 36 workers on c5.9xlarge a 70-min match takes ~2-3 h wall per render type.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

sys.path.insert(0, ".")
from src.loader import load_match_info, compute_attacking_right
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_ball, load_cached_events, load_cached_metadata, CACHE_DIR,
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
from src.viz.vision_plot import plot_vision_frame


# ── Render config (matches kane goal renders for visual consistency) ────

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
SCAN_BUFFER_FRAMES = 150     # 3 s — covers scanning memory lookback at segment start
SMOOTH_W, POLY = 13, 1
MAX_SPEED = 12.0
VISION_SMOOTHING = 7.0       # Bekkers vision grid for single-player vision render


# ── Shared helpers ──────────────────────────────────────────────────────

def _read_skeleton_window(match: str, f_lo: int, f_hi: int) -> pd.DataFrame:
    path = CACHE_DIR / match / "skeleton.parquet"
    table = pq.read_table(path, filters=[
        ("frame_number", ">=", int(f_lo)),
        ("frame_number", "<=", int(f_hi)),
    ])
    return table.to_pandas()


def _smooth_velocities(dyn: pd.DataFrame) -> pd.DataFrame:
    raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
    dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
    for col in ("vx", "vy"):
        dyn[col] = (dyn.groupby(["team", "jersey"], sort=False)[col]
                    .transform(lambda s: pd.Series(
                        savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                        index=s.index)))
    return dyn


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


def _resolve_player_team_jersey(player_id: str, info: dict):
    for team_int, key in [(1, "home_players"), (0, "away_players")]:
        for p in info.get(key, []) or []:
            if p.get("person_id") == player_id:
                jersey = p.get("shirt_number")
                if isinstance(jersey, int) and jersey > 0:
                    return team_int, jersey
    return None, None


def _half_for_frame(f: int, phases: dict) -> int:
    """Return 1 or 2 depending on which phase frame f belongs to."""
    for h in (1, 2):
        if h in phases:
            s, e = phases[h]
            if s <= f <= e:
                return h
    # Fallback: use first phase
    return 1


def _all_match_frames(match: str) -> tuple:
    """Read skeleton parquet metadata to determine the full frame range
    cached for this match. Avoids loading the whole 90M-row file."""
    path = CACHE_DIR / match / "skeleton.parquet"
    pf = pq.ParquetFile(path)
    f_min = float("inf")
    f_max = float("-inf")
    for rg_idx in range(pf.metadata.num_row_groups):
        col = pf.metadata.row_group(rg_idx).column(0)  # frame_number
        if col.statistics is not None:
            f_min = min(f_min, col.statistics.min)
            f_max = max(f_max, col.statistics.max)
    return int(f_min), int(f_max)


# ── Per-segment renderers ───────────────────────────────────────────────

def _render_dos_segment(match: str, f_lo: int, f_hi: int, out_path: str) -> tuple:
    """Render the DOS+shadow video for a frame range to out_path."""
    info = load_match_info(match)
    gk_jerseys = _resolve_gks(info)
    metadata = load_cached_metadata(match)
    phases = metadata["phases"]
    home_gk_left_p1 = bool(metadata["home_gk_left"][1])

    # Skeleton + scanning lookback buffer
    ss = _read_skeleton_window(match, f_lo - SCAN_BUFFER_FRAMES, f_hi)
    if len(ss) == 0:
        return ("skip", out_path, 0)
    ori = compute_orientations(ss, smooth=True)
    dyn = add_dynamics(ori)
    del ss, ori
    dyn = _smooth_velocities(dyn)

    ball = load_cached_ball(match)
    events = load_cached_events(match)
    timeline = build_possession_timeline(events, info, gap_fill_max_frames=300)

    cfg = ScanningMemoryConfig(memory_window_s=2.5, tau_decay_s=1.2,
                                vision_smoothing=2.0, framerate=FPS)
    memories = compute_scanning_memory_sequence(dyn, timeline, (f_lo, f_hi), cfg)

    vframes = sorted(f for f in dyn["frame_number"].unique() if f_lo <= f <= f_hi)
    if not vframes:
        return ("skip", out_path, 0)

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

        fm = memories.get(f)
        if fm is None:
            # No on-ball owner — render an empty pitch with players only
            ax.set_facecolor(BG)
            return

        attacking_team = int(fm.owner_team)
        half = _half_for_frame(f, phases)
        attacking_right = compute_attacking_right(attacking_team, half, home_gk_left_p1)

        try:
            dos_surf, _, _, _, _ = compute_dos_surface(
                fo, attacking_team, ball_xy, attacking_right,
                params=params, n_grid_x=N_GRID, n_directions=N_DIRS,
                vision_smoothing=DOS_VISION_SM,
            )
        except Exception:
            return

        mem_on_dos = resample_memory_to_grid(fm.memory, XGRID, YGRID)
        owner = (fm.owner_team, fm.owner_jersey)

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

        if (owner != state["owner"]
                or state["vis"] is None
                or state["vis"].shape != gated.shape):
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
            noise_floor=0.0005, display_max=0.015,
            ax=ax,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save(out_path, fps=FPS, dpi=200,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)
    return ("done", out_path, len(vframes))


def _render_ppcf_segment(match: str, f_lo: int, f_hi: int, out_path: str) -> tuple:
    """Render the PPCF reach-field video for a frame range to out_path."""
    info = load_match_info(match)
    gk_jerseys = _resolve_gks(info)
    metadata = load_cached_metadata(match)
    phases = metadata["phases"]
    home_gk_left_p1 = bool(metadata["home_gk_left"][1])

    ss = _read_skeleton_window(match, f_lo, f_hi)
    if len(ss) == 0:
        return ("skip", out_path, 0)
    ori = compute_orientations(ss, smooth=True)
    dyn = add_dynamics(ori)
    del ss, ori
    dyn = _smooth_velocities(dyn)

    ball = load_cached_ball(match)
    events = load_cached_events(match)
    timeline = build_possession_timeline(events, info, gap_fill_max_frames=300)
    # Pre-compute per-frame attacking team from possession timeline
    tl_lookup = {}
    for _, seg in timeline.iterrows():
        for f in range(int(seg["frame_start"]), int(seg["frame_end"]) + 1):
            tl_lookup[f] = int(seg["team"])

    vframes = sorted(f for f in dyn["frame_number"].unique() if f_lo <= f <= f_hi)
    if not vframes:
        return ("skip", out_path, 0)

    params = default_params()
    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)

    def render(i):
        ax.clear()
        f = vframes[i]
        fo = dyn[dyn["frame_number"] == f]
        if len(fo) < 6:
            return
        bf = ball[ball["frame_number"] == f]
        bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else None
        by = float(bf.iloc[0]["y"]) if len(bf) > 0 else None
        ball_xy = (bx, by) if bx is not None else None

        # Use on-ball team as attacking team; fallback to home if no owner
        attacking_team = tl_lookup.get(f, 1)
        try:
            ppcf_att, ppcf_def, _, _ = compute_ppcf_surfaces(
                fo, attacking_team, params=params, n_grid_x=N_GRID)
        except Exception:
            return
        plot_ppcf_frame(
            fo, attacking_team=attacking_team, ball_xy=ball_xy,
            gk_jerseys=gk_jerseys,
            ppcf_att=ppcf_att, ppcf_def=ppcf_def,
            ax=ax,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save(out_path, fps=FPS, dpi=200,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)
    return ("done", out_path, len(vframes))


def _render_vision_segment(match: str, f_lo: int, f_hi: int, out_path: str,
                            focus_player: str) -> tuple:
    """Render the Bekkers vision video for a SINGLE focus player over a
    frame range. The player must be present in the skeleton at each frame
    rendered; frames where they are off-pitch are skipped."""
    info = load_match_info(match)
    metadata = load_cached_metadata(match)
    phases = metadata["phases"]
    home_gk_left_p1 = bool(metadata["home_gk_left"][1])

    focus_team, focus_jersey = _resolve_player_team_jersey(focus_player, info)
    if focus_team is None:
        return ("fail", out_path, 0)

    # The "attacking team" for direction purposes = the focus player's team
    # (so attacking_right is computed for THEIR side, used to label players)
    att_team_int = focus_team

    ss = _read_skeleton_window(match, f_lo, f_hi)
    if len(ss) == 0:
        return ("skip", out_path, 0)
    ori = compute_orientations(ss, smooth=True)
    dyn = add_dynamics(ori)
    del ss, ori

    ball = load_cached_ball(match)
    vframes = sorted(f for f in dyn["frame_number"].unique() if f_lo <= f <= f_hi)
    if not vframes:
        return ("skip", out_path, 0)

    fig, ax = plt.subplots(figsize=(16, 10.4))
    fig.set_facecolor(BG)

    def render(i):
        ax.clear()
        f = vframes[i]
        fo = dyn[dyn["frame_number"] == f]
        bf = ball[ball["frame_number"] == f]
        bx = bf.iloc[0]["x"] if len(bf) > 0 else None
        by = bf.iloc[0]["y"] if len(bf) > 0 else None

        # Skip silently if focus player is off the pitch this frame
        present = ((fo["team"] == focus_team) & (fo["jersey"] == focus_jersey)).any()
        if not present:
            ax.set_facecolor(BG)
            return

        try:
            plot_vision_frame(
                fo,
                focus_team=focus_team, focus_jersey=focus_jersey,
                ball_x=bx, ball_y=by,
                att_team=att_team_int,
                smoothing=VISION_SMOOTHING,
                ax=ax,
            )
        except ValueError:
            ax.set_facecolor(BG)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani = FuncAnimation(fig, render, frames=len(vframes), blit=False, interval=20)
    ani.save(out_path, fps=FPS, dpi=200,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
    plt.close(fig)
    return ("done", out_path, len(vframes))


# ── Worker dispatch ─────────────────────────────────────────────────────

def _segment_task(args: tuple) -> tuple:
    """Top-level wrapper for ProcessPoolExecutor."""
    match, render_type, focus_player, f_lo, f_hi, out_path = args
    out = Path(out_path)
    if out.exists() and out.stat().st_size > 100_000:
        return ("skip", out_path, 0)
    try:
        if render_type == "dos":
            return _render_dos_segment(match, f_lo, f_hi, out_path)
        if render_type == "ppcf":
            return _render_ppcf_segment(match, f_lo, f_hi, out_path)
        if render_type == "vision":
            return _render_vision_segment(match, f_lo, f_hi, out_path, focus_player)
        return ("fail", out_path, 0)
    except Exception as e:
        return ("fail", f"{out_path}: {e}", 0)


# ── ffmpeg concat ───────────────────────────────────────────────────────

def _concat_segments(segment_paths: list, final_out: Path):
    """Concatenate segments via ffmpeg's concat demuxer with -c copy
    (no re-encode, instant). All segments must share codec params, which
    they do because they were all produced with identical FuncAnimation
    + ffmpeg writer settings here."""
    final_out.parent.mkdir(parents=True, exist_ok=True)
    list_file = final_out.parent / f"{final_out.stem}_concat_list.txt"
    list_file.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in segment_paths))
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(final_out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    list_file.unlink()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True,
                    help="Match folder name (e.g. Bayern_Hamburg)")
    ap.add_argument("--type", required=True, choices=["dos", "ppcf", "vision"],
                    help="Render type")
    ap.add_argument("--focus-player", default=None,
                    help="DFL-OBJ-* player id (required for --type vision)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel workers (default = cpu_count)")
    ap.add_argument("--out-dir", default="outputs/full_match",
                    help="Output directory for the final video")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="Smoke test: only render the first N frames of the match")
    ap.add_argument("--seg-frames", type=int, default=1500,
                    help="Max frames per segment (caps ffmpeg memory; "
                         "ffmpeg buffers RGBA frames in pipe ~1MB/frame, "
                         "so 1500 → ~1.5 GB per worker). Default: 1500.")
    args = ap.parse_args()

    if args.type == "vision" and not args.focus_player:
        sys.exit("--focus-player is required for --type vision")

    n_workers = args.workers or (os.cpu_count() or 8)

    # Determine full frame range from cache
    f_min, f_max = _all_match_frames(args.match)
    if args.limit_frames:
        f_max = min(f_max, f_min + args.limit_frames - 1)
    total_frames = f_max - f_min + 1
    print(f"Match: {args.match}")
    print(f"Frame range: {f_min} - {f_max} ({total_frames:,} potential frames)")
    print(f"Render type: {args.type}" +
          (f" (focus: {args.focus_player})" if args.type == "vision" else ""))
    print(f"Workers: {n_workers}")

    # Split frame range into many small segments (cap by --seg-frames so
    # each ffmpeg subprocess stays bounded; otherwise ffmpeg buffers RGBA
    # frames in pipe and a 5000-frame segment grows to ~5 GB → OOM with
    # 72 workers). Workers consume the queue, so n_segments can far
    # exceed n_workers.
    seg_size = max(1, min(args.seg_frames, total_frames))
    segments = []
    s = f_min
    while s <= f_max:
        e = min(f_max, s + seg_size - 1)
        segments.append((s, e))
        s = e + 1
    print(f"Segments: {len(segments)} of ~{seg_size} frames each "
          f"(workers consume from queue)")

    # Output paths
    label = args.type
    if args.type == "vision":
        label = f"vision_focus_{args.focus_player}"
    final_out = Path(args.out_dir) / f"{args.match}_{label}.mp4"
    if final_out.exists() and final_out.stat().st_size > 1_000_000:
        print(f"Final output already exists: {final_out} — delete it to regenerate")
        return

    # Prepare segment temp paths
    tmp_dir = Path(mkdtemp(prefix=f"render_{args.match}_{args.type}_"))
    try:
        tasks = []
        for i, (lo, hi) in enumerate(segments):
            seg_path = tmp_dir / f"segment_{i:03d}.mp4"
            tasks.append((args.match, args.type, args.focus_player,
                          lo, hi, str(seg_path)))

        # Render in parallel
        print(f"\nRendering {len(tasks)} segments in parallel ({n_workers} workers)...")
        t0 = time.time()
        ctx = get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = {ex.submit(_segment_task, t): t for t in tasks}
            for fut in as_completed(futures):
                status, path, n = fut.result()
                results.append((status, path, n))
                tag = {"done": "+", "skip": "⤳", "fail": "[FAIL]"}.get(status, "?")
                print(f"  {tag} {Path(path).name}  {n} frames")
        dt = time.time() - t0
        n_ok = sum(1 for s, *_ in results if s == "done")
        n_skip = sum(1 for s, *_ in results if s == "skip")
        n_fail = sum(1 for s, *_ in results if s == "fail")
        print(f"\nSegments: {n_ok} done, {n_skip} skipped, {n_fail} failed "
              f"({dt/60:.1f} min wall, {n_workers} workers)")

        if n_fail > 0:
            print("Some segments failed — aborting concat to avoid corrupt output")
            return

        # Build concat list (only segments that produced output)
        seg_paths = sorted(p for p in tmp_dir.glob("segment_*.mp4"))
        if not seg_paths:
            print("No segments produced output — nothing to concat")
            return

        print(f"\nConcatenating {len(seg_paths)} segments → {final_out}")
        t0 = time.time()
        _concat_segments(seg_paths, final_out)
        sz_mb = final_out.stat().st_size / 1024 / 1024
        print(f"Done ({time.time()-t0:.0f}s). Final: {final_out} ({sz_mb:.1f} MB)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
