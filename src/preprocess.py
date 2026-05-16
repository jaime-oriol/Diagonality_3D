"""
preprocess — Extract and cache all event-linked data for a match.

Reads the heavy nested parquet ONCE, extracts skeleton snapshots for every
event (passes, carries, receptions, shots) plus windows for D-Def and
velocity computation. Saves flat parquets that load instantly.

This is the pipeline that runs once per match (locally for dev,
on EC2 for all 5). After preprocessing, all downstream modules
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
    MATCHES, MATCH_DIR,
    TEAM_HOME, TEAM_AWAY,
    load_metadata, load_passes, load_carries, load_receptions,
    synced_frame_to_parquet, safe_float, kpi_path,
)
import xml.etree.ElementTree as ET

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

FRAMERATE = 50

# D-Def window: 3s after event at full 50Hz (150 frames)
DDEF_WINDOW_FRAMES = int(3.0 * FRAMERATE)  # 150

# Pre-event window: 3s before at full 50Hz (150 frames). Sized to fully
# cover the 2.5s scanning memory lookback used by src/scanning.py, plus
# 0.5s of buffer. Bumping this requires regenerating the cache via
# pipeline/regenerate_caches.py.
PRE_WINDOW_FRAMES = int(3.0 * FRAMERATE)  # 150


# --- Shots loader ---------------------------------------------------------

def _load_shots_kpi(match: str) -> pd.DataFrame:
    """Load shots from kpi_data XML with xG, position, SyncedFrameId."""
    tree = ET.parse(kpi_path(match))
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
            "x": safe_float(shot.get("X-Position")),
            "y": safe_float(shot.get("Y-Position")),
            "xg": safe_float(shot.get("xG")),
            "angle_to_goal": safe_float(shot.get("AngleToGoal")),
            "distance_to_goal": safe_float(shot.get("DistanceToGoal")),
            "pressure_on_shot": safe_float(shot.get("PressureOnPlayer")),
            "is_penalty": shot.get("IsPenalty") == "true",
            "is_free_kick": shot.get("IsFreeKick") == "true",
            "shot_result": shot.get("ShotResult", ""),
        })

    return pd.DataFrame(rows)


# --- TeamPossession loader ------------------------------------------------

def _load_possessions(match: str) -> pd.DataFrame:
    """Load TeamPossession sequences from kpi_data XML."""
    tree = ET.parse(kpi_path(match))
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
            "vertical_gain_carries": safe_float(tp.get("VerticalGainCarries")),
            "sum_xg_ind": safe_float(tp.get("SumXGInd")),
            "sum_xg_con": safe_float(tp.get("SumXGCon")),
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
    shots = _load_shots_kpi(match)
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
        -3s before (150 frames) + event frame + 3s after (150 frames)
        = 301 frames per event at 50Hz

    For carries:
        -3s before start + FULL carry at 50Hz + 3s after end
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
                # -3s before start ... carry duration ... +3s after end
                for f in range(base - PRE_WINDOW_FRAMES, end + DDEF_WINDOW_FRAMES + 1):
                    frame_set.add(f)
            else:
                # Fallback: -3s to +3s from start
                for f in range(base - PRE_WINDOW_FRAMES, base + DDEF_WINDOW_FRAMES + 1):
                    frame_set.add(f)
        else:
            # Passes, receptions, shots: -3s to +3s
            for f in range(base - PRE_WINDOW_FRAMES, base + DDEF_WINDOW_FRAMES + 1):
                frame_set.add(f)

    frames = np.array(sorted(f for f in frame_set if f > 0), dtype=np.int64)
    return frames


# --- Skeleton batch extraction (row-group scan) ----------------------------

def _extract_skeleton_batch(
    match: str,
    frames: np.ndarray,
    output_path: Path,
    parts: set = None,  # None = all 21 parts
    verbose: bool = True,
) -> int:
    """Extract skeleton data for many frames, writing chunks to parquet.

    Scans row groups one at a time and writes each chunk to disk
    immediately, keeping memory usage flat (~200MB peak).

    Returns total row count.
    """
    import pyarrow as pa
    import gc

    path = MATCH_DIR / match / f"{MATCHES[match]}.parquet"
    pf = pq.ParquetFile(path)

    frame_set = set(frames.tolist())
    n_groups = pf.metadata.num_row_groups
    total_rows = 0
    writer = None

    schema = pa.schema([
        ("frame_number", pa.int32()),
        ("team", pa.int8()),
        ("jersey", pa.int8()),
        ("part_id", pa.int8()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
    ])

    for rg_idx in range(n_groups):
        table = pf.read_row_group(rg_idx)
        frame_col = table.column("frame_number").to_pylist()

        hits = frame_set & set(frame_col)
        if not hits:
            del table
            continue

        skel_col = table.column("skeletons").to_pylist()
        chunk_rows = []

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
                    if parts is not None and pid not in parts:
                        continue
                    chunk_rows.append((
                        fn, team, jersey, pid,
                        part["position_x"], part["position_y"], part["position_z"],
                    ))

        # Free the heavy parquet table immediately
        del table, skel_col, frame_col
        gc.collect()

        if chunk_rows:
            chunk_df = pd.DataFrame(chunk_rows, columns=[
                "frame_number", "team", "jersey", "part_id", "x", "y", "z",
            ])
            chunk_df["team"] = chunk_df["team"].astype(np.int8)
            chunk_df["jersey"] = chunk_df["jersey"].astype(np.int8)
            chunk_df["part_id"] = chunk_df["part_id"].astype(np.int8)
            chunk_df["x"] = chunk_df["x"].astype(np.float32)
            chunk_df["y"] = chunk_df["y"].astype(np.float32)
            chunk_df["z"] = chunk_df["z"].astype(np.float32)

            pa_table = pa.Table.from_pandas(chunk_df, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(pa_table)
            total_rows += len(chunk_df)

            del chunk_rows, chunk_df, pa_table
            gc.collect()

            if verbose and rg_idx % 20 == 0:
                print(f"    row_group {rg_idx}/{n_groups}, {total_rows:,} rows so far")

        frame_set -= hits
        if not frame_set:
            if verbose:
                print(f"    all frames found at row_group {rg_idx}/{n_groups}")
            break

    if writer is not None:
        writer.close()

    return total_rows


def _extract_ball_batch(
    match: str,
    frames: np.ndarray,
) -> pd.DataFrame:
    """Extract ball data for many frames. Ball data is small so kept in memory."""
    import gc

    path = MATCH_DIR / match / f"{MATCHES[match]}.parquet"
    pf = pq.ParquetFile(path)

    frame_set = set(frames.tolist())
    all_rows = []

    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx)
        frame_col = table.column("frame_number").to_pylist()

        hits = frame_set & set(frame_col)
        if not hits:
            del table
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

        del table, ball_col, exists_col
        gc.collect()

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

    # 5. Extract skeleton — ALL 21 keypoints, streamed to disk
    skel_path = cache_path / "skeleton.parquet"
    if verbose:
        print(f"[{match}] Extracting skeleton — all 21 keypoints (streaming to disk)...")
    t0 = time.time()
    skel_rows = _extract_skeleton_batch(match, frames, skel_path, parts=None, verbose=verbose)
    dt_skel = time.time() - t0
    if verbose:
        skel_mb = skel_path.stat().st_size / 1024 / 1024
        print(f"  {skel_rows:,} rows written, {skel_mb:.1f} MB on disk ({dt_skel:.0f}s)")

    # 6. Extract ball
    if verbose:
        print(f"[{match}] Extracting ball...")
    t0 = time.time()
    ball = _extract_ball_batch(match, frames)
    if verbose:
        print(f"  {len(ball):,} rows ({time.time()-t0:.0f}s)")
    ball.to_parquet(cache_path / "ball.parquet", index=False)

    # 7. Validation — read a small sample from cached parquet
    if verbose:
        print(f"\n[{match}] Validating cache...")

    # Quick check: read first row group only (avoids loading full file)
    sample = pq.ParquetFile(skel_path).read_row_group(0).to_pandas()
    sample_frame = sample["frame_number"].iloc[0]
    n_in_frame = sample[sample["frame_number"] == sample_frame].groupby(["team", "jersey"]).ngroups
    parts_in_frame = sample[sample["frame_number"] == sample_frame]["part_id"].nunique()
    if verbose:
        print(f"  Sample frame {sample_frame}: {n_in_frame} players, {parts_in_frame} parts/player")
        print(f"  Skeleton rows on disk: {skel_rows:,}")
    del sample

    # 8. Summary
    import gc
    gc.collect()
    dt_total = time.time() - t_total
    if verbose:
        sizes = {}
        for fname in ["skeleton.parquet", "ball.parquet", "events.parquet", "possessions.parquet"]:
            sizes[fname] = (cache_path / fname).stat().st_size / 1024 / 1024
        print(f"\n[{match}] DONE in {dt_total:.0f}s")
        print(f"  Cache: {cache_path}")
        for fname, mb in sizes.items():
            print(f"    {fname:25s} {mb:6.1f} MB")
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

def load_cached_metadata(match: str) -> dict:
    """Load preprocessed metadata from cache. Instant.

    Converts JSON string keys back to int for phases and home_gk_left
    so the result is compatible with synced_frame_to_parquet().
    """
    with open(CACHE_DIR / match / "metadata.json") as f:
        data = json.load(f)
    # JSON serializes dict keys as strings; convert back to int
    if "phases" in data:
        data["phases"] = {int(k): v for k, v in data["phases"].items()}
    if "home_gk_left" in data:
        data["home_gk_left"] = {int(k): v for k, v in data["home_gk_left"].items()}
    return data


# --- CLI ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    match = sys.argv[1] if len(sys.argv) > 1 else "Bayern_Hamburg"
    preprocess_match(match)
