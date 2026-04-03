"""
preprocess — Extract and cache all event-linked data for a match.

Reads the heavy nested parquet ONCE, extracts skeleton snapshots for every
event (passes, carries, receptions, shots) plus windows for D-Def and
velocity computation. Saves flat parquets that load instantly.

This is the pipeline that runs once per match (locally for dev,
on SageMaker for all 5). After preprocessing, all downstream modules
(orientation, vision, theta, ddef, ppcf) work from the cache.

Output in cache/{match}/:
    events.parquet       — All passes, carries, receptions, shots unified
    possessions.parquet  — TeamPossession sequences with xG
    skeleton.parquet     — Skeleton keypoints at all needed frames
    ball.parquet         — Ball position/velocity at all needed frames
    metadata.json        — Match metadata (roster, phases, pitch)
"""

import json
import time
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .loader import (
    MATCHES, MATCH_DIR, PART_NAMES, ORIENTATION_PARTS,
    TEAM_HOME, TEAM_AWAY,
    load_metadata, load_passes, load_carries, load_receptions,
    synced_frame_to_parquet, _safe_float, _safe_int, _kpi_path,
)
import xml.etree.ElementTree as ET

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

FRAMERATE = 50

# D-Def window: 3s after event at full 50Hz (150 frames)
DDEF_WINDOW_FRAMES = int(3.0 * FRAMERATE)  # 150

# Pre-event window: 1s before at full 50Hz (50 frames)
PRE_WINDOW_FRAMES = int(1.0 * FRAMERATE)  # 50


# --- Shots loader ---------------------------------------------------------

def _load_shots_kpi(match: str, metadata: dict) -> pd.DataFrame:
    """Load shots from kpi_data XML with xG, position, SyncedFrameId."""
    tree = ET.parse(_kpi_path(match))
    root = tree.getroot()

    rows = []
    for event_elem in root.iter("Event"):
        shot = event_elem.find("ShotAtGoal")
        if shot is None:
            continue

        section = shot.get("InGameSection", "")
        half = 1 if "first" in section.lower() else 2

        synced = shot.get("SyncedFrameId")
        synced_ok = shot.get("SyncSuccessful") == "true"

        rows.append({
            "event_id": shot.get("EventId"),
            "event_type": "shot",
            "team_id": shot.get("TeamId"),
            "player_id": shot.get("PlayerId"),
            "half": half,
            "synced_frame_id": int(synced) if synced and synced_ok else -1,
            "x": _safe_float(shot.get("X-Position")),
            "y": _safe_float(shot.get("Y-Position")),
            "xg": _safe_float(shot.get("xG")),
            "angle_to_goal": _safe_float(shot.get("AngleToGoal")),
            "distance_to_goal": _safe_float(shot.get("DistanceToGoal")),
            "pressure_on_shot": _safe_float(shot.get("PressureOnPlayer")),
            "is_penalty": shot.get("IsPenalty") == "true",
            "is_free_kick": shot.get("IsFreeKick") == "true",
            "shot_result": shot.get("ShotResult", ""),
        })

    return pd.DataFrame(rows)


# --- TeamPossession loader ------------------------------------------------

def _load_possessions(match: str) -> pd.DataFrame:
    """Load TeamPossession sequences from kpi_data XML."""
    tree = ET.parse(_kpi_path(match))
    root = tree.getroot()

    rows = []
    for event_elem in root.iter("Event"):
        tp = event_elem.find("TeamPossession")
        if tp is None:
            continue

        # Collect child event IDs in this possession
        child_ids = [pe.get("EventId") for pe in tp.findall("PossessionEvent")]

        rows.append({
            "event_id": tp.get("EventId"),
            "team_id": tp.get("TeamId"),
            "vertical_gain_carries": _safe_float(tp.get("VerticalGainCarries")),
            "sum_xg_ind": _safe_float(tp.get("SumXGInd")),
            "sum_xg_con": _safe_float(tp.get("SumXGCon")),
            "half": 1 if "first" in tp.get("InGameSection", "").lower() else 2,
            "event_ids": child_ids,
            "n_events": len(child_ids),
        })

    return pd.DataFrame(rows)


# --- Events builder -------------------------------------------------------

def _build_events(match: str, metadata: dict) -> pd.DataFrame:
    """Load and unify ALL event types into one DataFrame."""

    # Passes
    passes = load_passes(match)
    passes["event_type"] = "pass"
    passes["parquet_frame"] = passes.apply(
        lambda r: synced_frame_to_parquet(r["synced_frame_id"], metadata, r["half"]),
        axis=1,
    )

    # Carries
    carries = load_carries(match)
    carries["event_type"] = "carry"
    carries["parquet_frame"] = carries.apply(
        lambda r: synced_frame_to_parquet(r["synced_frame_id"], metadata, r["half"]),
        axis=1,
    )
    carries["parquet_frame_end"] = carries.apply(
        lambda r: synced_frame_to_parquet(r["end_synced_frame_id"], metadata, r["half"])
        if r["end_synced_frame_id"] > 0 else -1,
        axis=1,
    )

    # Receptions
    recs = load_receptions(match)
    recs["event_type"] = "reception"
    recs["parquet_frame"] = recs.apply(
        lambda r: synced_frame_to_parquet(r["synced_frame_id"], metadata, r["half"]),
        axis=1,
    )

    # Shots
    shots = _load_shots_kpi(match, metadata)
    shots["parquet_frame"] = shots.apply(
        lambda r: synced_frame_to_parquet(r["synced_frame_id"], metadata, r["half"])
        if r["synced_frame_id"] > 0 else -1,
        axis=1,
    )

    # Combine all
    events = pd.concat([passes, carries, recs, shots], ignore_index=True, sort=False)
    events["match"] = match
    return events


# --- Frame collection -----------------------------------------------------

def _collect_frames(events: pd.DataFrame) -> np.ndarray:
    """Collect all unique parquet frame numbers needed for the cache.

    FULL 50Hz resolution everywhere — no subsampling.

    For passes/receptions/shots:
        -1s before (50 frames) + event frame + 3s after (150 frames)
        = 201 frames per event at 50Hz

    For carries:
        -1s before start + FULL carry at 50Hz + 3s after end
        = variable, up to ~1000 frames for a 16s carry
    """
    frame_set: Set[int] = set()

    for _, row in events.iterrows():
        base = int(row["parquet_frame"])
        if base <= 0:
            continue

        if row.get("event_type") == "carry":
            end = row.get("parquet_frame_end", -1)
            if isinstance(end, (int, np.integer)) and end > 0:
                end = int(end)
                # -1s before start ... carry duration ... +3s after end
                for f in range(base - PRE_WINDOW_FRAMES, end + DDEF_WINDOW_FRAMES + 1):
                    frame_set.add(f)
            else:
                # Fallback: -1s to +3s from start
                for f in range(base - PRE_WINDOW_FRAMES, base + DDEF_WINDOW_FRAMES + 1):
                    frame_set.add(f)
        else:
            # Passes, receptions, shots: -1s to +3s
            for f in range(base - PRE_WINDOW_FRAMES, base + DDEF_WINDOW_FRAMES + 1):
                frame_set.add(f)

    frames = np.array(sorted(f for f in frame_set if f > 0), dtype=np.int64)
    return frames


# --- Skeleton batch extraction (row-group scan) ----------------------------

def _extract_skeleton_batch(
    match: str,
    frames: np.ndarray,
    parts: set = ORIENTATION_PARTS,
) -> pd.DataFrame:
    """Extract skeleton data for many frames, scanning row groups once."""
    path = MATCH_DIR / match / f"{MATCHES[match]}.parquet"
    pf = pq.ParquetFile(path)

    frame_set = set(frames.tolist())
    all_rows = []

    n_groups = pf.metadata.num_row_groups
    for rg_idx in range(n_groups):
        table = pf.read_row_group(rg_idx)
        frame_col = table.column("frame_number").to_pylist()

        hits = frame_set & set(frame_col)
        if not hits:
            continue

        skel_col = table.column("skeletons").to_pylist()
        for i, fn in enumerate(frame_col):
            if fn not in hits:
                continue

            for skel in skel_col[i]:
                team = skel["team"]
                jersey = skel["jersey_number"]
                if team not in (TEAM_HOME, TEAM_AWAY) or jersey <= 0:
                    continue

                for part in skel["parts"]:
                    pid = part["name"]
                    if pid not in parts:
                        continue
                    all_rows.append((
                        fn, team, jersey, pid,
                        part["position_x"], part["position_y"], part["position_z"],
                    ))

        frame_set -= hits
        if not frame_set:
            break

    df = pd.DataFrame(all_rows, columns=[
        "frame_number", "team", "jersey", "part_id", "x", "y", "z",
    ])
    df["part"] = df["part_id"].map(PART_NAMES)
    df["team"] = df["team"].astype(np.int8)
    df["jersey"] = df["jersey"].astype(np.int8)
    df["part_id"] = df["part_id"].astype(np.int8)
    df["x"] = df["x"].astype(np.float32)
    df["y"] = df["y"].astype(np.float32)
    df["z"] = df["z"].astype(np.float32)

    return df.sort_values(["frame_number", "team", "jersey", "part_id"]).reset_index(drop=True)


def _extract_ball_batch(
    match: str,
    frames: np.ndarray,
) -> pd.DataFrame:
    """Extract ball data for many frames, scanning row groups once."""
    path = MATCH_DIR / match / f"{MATCHES[match]}.parquet"
    pf = pq.ParquetFile(path)

    frame_set = set(frames.tolist())
    all_rows = []

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        frame_col = table.column("frame_number").to_pylist()

        hits = frame_set & set(frame_col)
        if not hits:
            continue

        ball_col = table.column("ball").to_pylist()
        exists_col = table.column("ball_exists").to_pylist()

        for i, fn in enumerate(frame_col):
            if fn not in hits:
                continue
            ball = ball_col[i]
            exists = exists_col[i]
            if ball is not None and exists:
                all_rows.append((
                    fn,
                    ball["position_x"], ball["position_y"], ball["position_z"],
                    ball["velocity_x"], ball["velocity_y"], ball["velocity_z"],
                    True,
                ))
            else:
                all_rows.append((fn, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False))

        frame_set -= hits
        if not frame_set:
            break

    df = pd.DataFrame(all_rows, columns=[
        "frame_number", "x", "y", "z", "vx", "vy", "vz", "ball_exists",
    ])
    for c in ["x", "y", "z", "vx", "vy", "vz"]:
        df[c] = df[c].astype(np.float32)

    return df.sort_values("frame_number").reset_index(drop=True)


# --- Main pipeline --------------------------------------------------------

def preprocess_match(match: str, verbose: bool = True) -> Path:
    """Run the full preprocessing pipeline for one match.

    Reads the heavy parquet ONCE, extracts everything needed,
    saves to cache/{match}/.
    """
    cache_path = CACHE_DIR / match
    cache_path.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # 1. Metadata
    if verbose:
        print(f"[{match}] Loading metadata...")
    metadata = load_metadata(match)
    meta_json = {k: v for k, v in metadata.items() if k != "roster"}
    meta_json["roster"] = {f"{t}_{j}": v for (t, j), v in metadata["roster"].items()}
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2, default=str)

    # 2. Events (passes + carries + receptions + shots)
    if verbose:
        print(f"[{match}] Loading events...")
    t0 = time.time()
    events = _build_events(match, metadata)
    if verbose:
        for et in ["pass", "carry", "reception", "shot"]:
            n = (events["event_type"] == et).sum()
            print(f"  {et}: {n}")
        print(f"  Total: {len(events)} ({time.time()-t0:.1f}s)")
    events.to_parquet(cache_path / "events.parquet", index=False)

    # 3. Possessions
    if verbose:
        print(f"[{match}] Loading possessions...")
    poss = _load_possessions(match)
    if verbose:
        print(f"  {len(poss)} possessions, total xG_ind={poss['sum_xg_ind'].sum():.2f}")
    poss.to_parquet(cache_path / "possessions.parquet", index=False)

    # 4. Collect frames
    frames = _collect_frames(events)
    if verbose:
        print(f"[{match}] Frames to extract: {len(frames):,} unique")
        if len(frames) > 0:
            print(f"  Range: {frames[0]} - {frames[-1]}")

    # 5. Extract skeleton
    if verbose:
        print(f"[{match}] Extracting skeleton (scanning {280} row groups)...")
    t0 = time.time()
    skeleton = _extract_skeleton_batch(match, frames)
    dt_skel = time.time() - t0
    if verbose:
        mb = skeleton.memory_usage(deep=True).sum() / 1024 / 1024
        n_frames = skeleton["frame_number"].nunique()
        n_players = skeleton.groupby(["frame_number", "team", "jersey"]).ngroups // max(n_frames, 1) if n_frames > 0 else 0
        print(f"  {len(skeleton):,} rows, {n_frames:,} frames, ~{n_players} players/frame")
        print(f"  {mb:.1f} MB in memory ({dt_skel:.0f}s)")
    skeleton.to_parquet(cache_path / "skeleton.parquet", index=False)

    # 6. Extract ball
    if verbose:
        print(f"[{match}] Extracting ball...")
    t0 = time.time()
    ball = _extract_ball_batch(match, frames)
    if verbose:
        print(f"  {len(ball):,} rows ({time.time()-t0:.0f}s)")
    ball.to_parquet(cache_path / "ball.parquet", index=False)

    # 7. Validation
    if verbose:
        print(f"\n[{match}] Validating cache...")

    # Check all synced events have their frames in skeleton
    synced_events = events[events["parquet_frame"] > 0]
    event_frames = set(synced_events["parquet_frame"].unique())
    cached_frames = set(skeleton["frame_number"].unique())
    missing = event_frames - cached_frames
    if missing:
        print(f"  WARNING: {len(missing)} event frames missing from skeleton cache!")
    else:
        if verbose:
            print(f"  OK: all {len(event_frames)} event frames present in skeleton")

    # Check players per frame
    sample_frame = skeleton["frame_number"].iloc[len(skeleton)//2]
    n_in_frame = skeleton[skeleton["frame_number"] == sample_frame].groupby(["team", "jersey"]).ngroups
    if verbose:
        print(f"  OK: {n_in_frame} players in sample frame {sample_frame}")

    # 8. Summary
    dt_total = time.time() - t_total
    if verbose:
        sizes = {}
        for f in ["skeleton.parquet", "ball.parquet", "events.parquet", "possessions.parquet"]:
            sizes[f] = (cache_path / f).stat().st_size / 1024 / 1024
        print(f"\n[{match}] DONE in {dt_total:.0f}s")
        print(f"  Cache: {cache_path}")
        for f, mb in sizes.items():
            print(f"    {f:25s} {mb:6.1f} MB")
        print(f"    {'TOTAL':25s} {sum(sizes.values()):6.1f} MB")

    return cache_path


# --- Cache loaders (instant) ----------------------------------------------

def load_cached_skeleton(match: str) -> pd.DataFrame:
    """Load preprocessed skeleton from cache. Instant."""
    return pd.read_parquet(CACHE_DIR / match / "skeleton.parquet")

def load_cached_ball(match: str) -> pd.DataFrame:
    """Load preprocessed ball from cache. Instant."""
    return pd.read_parquet(CACHE_DIR / match / "ball.parquet")

def load_cached_events(match: str) -> pd.DataFrame:
    """Load preprocessed events from cache. Instant."""
    return pd.read_parquet(CACHE_DIR / match / "events.parquet")

def load_cached_possessions(match: str) -> pd.DataFrame:
    """Load preprocessed possessions from cache. Instant."""
    return pd.read_parquet(CACHE_DIR / match / "possessions.parquet")

def load_cached_metadata(match: str) -> dict:
    """Load preprocessed metadata from cache. Instant."""
    with open(CACHE_DIR / match / "metadata.json") as f:
        return json.load(f)

def is_cached(match: str) -> bool:
    """Check if a match has been preprocessed."""
    cache_path = CACHE_DIR / match
    return all(
        (cache_path / f).exists()
        for f in ["events.parquet", "skeleton.parquet", "ball.parquet",
                   "possessions.parquet", "metadata.json"]
    )


# --- CLI ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    match = sys.argv[1] if len(sys.argv) > 1 else "Bayern_Hamburg"
    preprocess_match(match)
