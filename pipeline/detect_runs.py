"""Detect off-ball runs across the 5 cached matches and score each one
with the orientation-aware DOS model.

Pipeline:
  1. Build the frame-exact possession timeline (carries + passes linked to
     receptions) so the on-ball owner is known at every frame.
  2. Stream the skeleton cache in frame chunks (pyarrow predicate pushdown,
     memory-safe on WSL), compute orientations + smoothed velocities.
  3. `src.runs.detect_runs` finds every off-ball attacking run in the
     chunk (sustained > 5 m/s for > 0.7 s, net displacement > 5 m).
  4. Each run is anchored at its start frame and scored with
     `dos.dos_for_event` using the runner's instantaneous skeleton
     velocity as the action direction and a virtual destination
     `pos + vel * 1 s` — the same convention used for take-ons in
     `validate_dos_outcomes.py`, so runs are directly comparable.

Run detection is intentionally a lightweight front-end (a real burst, not
a classified run taxonomy): commercial providers — SkillCorner Game
Intelligence, StatsPerform Opta Vision — already productise run detection
and classification. The contribution here is the orientation-aware DOS
score on top of the detected run.

Usage:
    python3 pipeline/detect_runs.py [--matches NAMES...] [--max-frame-span N]

Outputs:
    outputs/intermediate/runs_dos.csv
    outputs/intermediate/runs_dos_report.md
"""

import sys
sys.path.insert(0, ".")

import argparse
import gc
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.loader import MATCHES, compute_attacking_right, load_match_info
from src.preprocess import load_cached_events, load_cached_metadata
from src.orientation import compute_orientations, add_dynamics
from src.possession import build_possession_timeline
from src.runs import detect_runs, RUN_COLUMNS
from src.dos import dos_for_event, default_params
from src.skeleton_chunks import (
    _read_skeleton_window, _smooth_velocities,
    DEFAULT_MAX_FRAME_SPAN, DEFAULT_MAX_CHUNK_EVENTS,
)

VISION_SMOOTHING = 1.0          # same as validate_dos_outcomes.py
FRAME_PAD = 25                  # frames padded each side for vel smoothing
RUN_VIRTUAL_HORIZON_S = 1.0     # virtual-destination look-ahead (s)
RUN_MIN_SPEED = 0.6             # m/s: below this, fall back to forward axis
RUN_MIN_HORIZON_M = 3.0         # floor on virtual-destination distance (m)

OUT_CSV = Path("outputs/intermediate/runs_dos.csv")
OUT_REPORT = Path("outputs/intermediate/runs_dos_report.md")


# ── Helpers ──────────────────────────────────────────────────────────────

def _half_of_frame(metadata: dict, frame: int) -> int:
    """Return 1 or 2 from the metadata phase ranges (default 1)."""
    phases = metadata.get("phases", {})
    for h in (2, 1):
        rng = phases.get(h) or phases.get(str(h))
        if rng and int(rng[0]) <= frame <= int(rng[1]):
            return h
    return 1


def _split_into_chunks(frames_sorted: np.ndarray, max_frame_span: int,
                       max_events: int) -> List[tuple]:
    """Split a sorted frame array into (lo, hi) frame-range chunks bounded
    by both a frame span and a max number of anchor frames. Frame span is
    the real memory cap on WSL."""
    chunks = []
    n = len(frames_sorted)
    start = 0
    while start < n:
        f0 = int(frames_sorted[start])
        end = start + 1
        while (end < n and (end - start) < max_events
               and int(frames_sorted[end]) - f0 <= max_frame_span):
            end += 1
        chunks.append((f0, int(frames_sorted[end - 1])))
        start = end
    return chunks


def _evaluate_run(fo: pd.DataFrame, run, attacking_team: int,
                  attacking_right: bool, params: dict) -> Optional[dict]:
    """Single-frame DOS for one run, anchored at its start frame.

    The runner's instantaneous skeleton velocity is the action direction;
    the virtual destination is `pos + vel * horizon`. If the runner is
    essentially stationary at the anchor frame the direction falls back to
    the attacking goal axis (keeps the run scorable rather than dropped)."""
    if fo is None or len(fo) < 6:
        return None
    px, py = float(run.x_start), float(run.y_start)
    vx, vy = float(run.vx_start), float(run.vy_start)
    speed = float(np.hypot(vx, vy))

    if speed >= RUN_MIN_SPEED:
        horizon_m = max(RUN_VIRTUAL_HORIZON_S * speed, RUN_MIN_HORIZON_M)
        dir_x, dir_y = vx / speed, vy / speed
    else:
        forward_sign = 1.0 if attacking_right else -1.0
        dir_x, dir_y = forward_sign, 0.0
        horizon_m = RUN_MIN_HORIZON_M

    evd = {
        "event_type": "carry", "x": px, "y": py,
        "carry_end_x": px + horizon_m * dir_x,
        "carry_end_y": py + horizon_m * dir_y,
    }
    try:
        return dos_for_event(
            fo, evd, attacking_team, attacking_right,
            params=params, vision_smoothing=VISION_SMOOTHING,
        )
    except Exception:
        return None


# ── Per-match processing ─────────────────────────────────────────────────

def _process_match(match: str, max_frame_span: int,
                   max_events: int) -> pd.DataFrame:
    print(f"\n[{match}] building possession timeline...")
    events = load_cached_events(match)
    metadata = load_cached_metadata(match)
    home_gk_left_p1 = bool(metadata["home_gk_left"].get("1",
                           metadata["home_gk_left"].get(1)))
    match_info = load_match_info(match)
    timeline = build_possession_timeline(events, match_info,
                                         gap_fill_max_frames=50)
    print(f"  {len(timeline)} possession segments")
    if len(timeline) == 0:
        return pd.DataFrame()

    roster = metadata.get("roster", {})
    params = default_params()

    # Anchor frames = every cached event frame; the skeleton cache only
    # exists inside those windows, so this covers all detectable runs.
    anchors = events["parquet_frame"].dropna()
    anchors = np.sort(anchors[anchors > 0].astype(int).values)
    chunks = _split_into_chunks(anchors, max_frame_span, max_events)
    print(f"  {len(chunks)} skeleton chunk(s)")

    all_runs: List[dict] = []
    t0 = time.time()
    for c, (f0, f1) in enumerate(chunks):
        skel = _read_skeleton_window(match, f0 - FRAME_PAD, f1 + FRAME_PAD)
        if len(skel) == 0:
            continue
        ori = compute_orientations(skel, smooth=True)
        del skel
        dyn = _smooth_velocities(add_dynamics(ori))
        del ori

        runs = detect_runs(dyn, timeline)
        dyn_by_frame = {f: g for f, g in dyn.groupby("frame_number",
                                                     sort=False)}
        del dyn

        kept = 0
        for run in runs.itertuples(index=False):
            fo = dyn_by_frame.get(int(run.frame_start))
            attacking_team = int(run.owner_team)
            half = _half_of_frame(metadata, int(run.frame_start))
            attacking_right = compute_attacking_right(
                attacking_team, half, home_gk_left_p1)
            r = _evaluate_run(fo, run, attacking_team, attacking_right, params)
            if r is None:
                continue
            pkey = f"{int(run.team)}_{int(run.jersey)}"
            pinfo = roster.get(pkey, {})
            all_runs.append({
                "match": match,
                "team": int(run.team),
                "jersey": int(run.jersey),
                "player_name": pinfo.get("name", ""),
                "team_name": pinfo.get("team_name", ""),
                "owner_jersey": int(run.owner_jersey),
                "half": half,
                "frame_start": int(run.frame_start),
                "frame_end": int(run.frame_end),
                "duration_s": float(run.duration_s),
                "x_start": float(run.x_start), "y_start": float(run.y_start),
                "x_end": float(run.x_end), "y_end": float(run.y_end),
                "displacement_m": float(run.displacement_m),
                "peak_speed": float(run.peak_speed),
                "mean_speed": float(run.mean_speed),
                "run_dir": float(run.run_dir),
                "attacking_right": bool(attacking_right),
                "dos": r["dos"],
                "ppcf_baseline": r["ppcf_att_baseline"],
                "ppcf_with_awareness": r["ppcf_att_with_awareness"],
                "direction_class": r["direction_class"],
                "event_direction_rad": r["event_direction"],
                "awareness_mean": r["awareness_mean"],
                "detection_delay_mean": r["detection_delay_mean"],
            })
            kept += 1

        dyn_by_frame.clear()
        del dyn_by_frame, runs
        gc.collect()
        print(f"  chunk {c+1}/{len(chunks)}  frames {f0}-{f1}  "
              f"runs_kept={kept}  elapsed={time.time()-t0:.0f}s")

    df = pd.DataFrame(all_runs)
    if len(df) == 0:
        return df
    # Skeleton windows around neighbouring events overlap, so a run near a
    # chunk boundary can be detected twice — dedupe on its identity.
    df = df.drop_duplicates(
        subset=["match", "team", "jersey", "frame_start"]
    ).sort_values("frame_start").reset_index(drop=True)
    print(f"  {len(df):,} runs ({time.time()-t0:.0f}s)")
    return df


# ── Report ───────────────────────────────────────────────────────────────

def _write_report(df: pd.DataFrame, elapsed: float):
    lines = ["# Off-ball run detection + DOS", ""]
    lines.append(f"_Generated by `pipeline/detect_runs.py` in {elapsed:.0f}s._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Matches: **{df['match'].nunique()}**")
    lines.append(f"- Off-ball runs detected: **{len(df):,}**")
    lines.append("- Detector: attacking off-ball player, sustained "
                 ">5 m/s for >0.7 s, net displacement >5 m "
                 "(`src/runs.py`).")
    lines.append("- DOS: `dos_for_event` anchored at the run start frame, "
                 "instantaneous skeleton velocity as direction "
                 "(virtual destination `pos + vel*1s`).")
    lines.append("")

    lines.append("## Run physical profile")
    lines.append("")
    lines.append("```")
    lines.append(df[["duration_s", "displacement_m", "peak_speed",
                     "mean_speed"]].describe().to_string())
    lines.append("```")
    lines.append("")

    lines.append("## DOS distribution")
    lines.append("")
    lines.append("```")
    lines.append(df["dos"].describe().to_string())
    lines.append("```")
    lines.append(f"- DOS > 0: **{(df['dos']>0).mean()*100:.1f}%** of runs")
    lines.append("")

    lines.append("## Direction-class breakdown")
    lines.append("")
    rows = []
    for cls, g in df.groupby("direction_class"):
        rows.append({
            "direction_class": cls, "n": len(g),
            "dos_mean": g["dos"].mean(),
            "awareness_mean": g["awareness_mean"].mean(),
            "peak_speed_mean": g["peak_speed"].mean(),
        })
    tbl = pd.DataFrame(rows).set_index("direction_class")
    tbl = tbl.reindex(["forward", "diagonal", "sideways"]).dropna(how="all")
    lines.append("```")
    lines.append(tbl.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Per-match")
    lines.append("")
    rows = []
    for m, g in df.groupby("match"):
        rows.append({"match": m, "n_runs": len(g),
                     "dos_mean": g["dos"].mean(),
                     "diag_share": (g["direction_class"] == "diagonal").mean()})
    lines.append("```")
    lines.append(pd.DataFrame(rows).set_index("match").to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Top players by mean DOS on runs (min 15 runs)")
    lines.append("")
    g = df[df["player_name"] != ""].groupby(["player_name", "team_name"])
    agg = g.agg(n_runs=("dos", "size"), dos_mean=("dos", "mean"),
                diag_share=("direction_class",
                            lambda s: (s == "diagonal").mean()))
    agg = agg[agg["n_runs"] >= 15].sort_values("dos_mean", ascending=False)
    lines.append("```")
    lines.append(agg.head(15).to_string())
    lines.append("```")
    lines.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Report: {OUT_REPORT}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--max-frame-span", type=int, default=DEFAULT_MAX_FRAME_SPAN)
    ap.add_argument("--max-chunk-events", type=int,
                    default=DEFAULT_MAX_CHUNK_EVENTS)
    args = ap.parse_args()
    matches = args.matches if args.matches else list(MATCHES.keys())

    t0 = time.time()
    parts = []
    for m in matches:
        df_m = _process_match(m, args.max_frame_span, args.max_chunk_events)
        if len(df_m):
            parts.append(df_m)
        gc.collect()

    if not parts:
        print("No runs detected.")
        return
    df = pd.concat(parts, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    elapsed = time.time() - t0
    print(f"\nWrote {len(df):,} runs -> {OUT_CSV}")
    _write_report(df, elapsed)
    print(f"Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
