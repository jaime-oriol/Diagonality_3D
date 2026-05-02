"""Enrich `dos_validation_raw_xt_ddef.csv` with theta orientation metrics
(`src/theta.py`) so we get the storytelling-ready numbers:

  - Defender disruption (axis 1):
      * mean_theta_shoulder_all (avg defender shoulder misalignment vs target)
      * pct_nearest_blind       (nearest defender has action in his blind half)
      * n_wrongfooted           (affected defenders > 60° misaligned)
      * pct_wrongfooted

  - Receiver advantage (axis 2, passes only):
      * receiver_open_angle     (receiver shoulder vs goal direction)
      * receiver_turn_needed    (receiver head vs goal direction)
      * n_teammates_in_fov      (teammates inside receiver's 120° cone)

These give the cifras citables in the narrative (Spielverlagerung +
Hamilton Manifesto) that don't fall out of the DOS / D-Def / xT
abstractions.

Memory model: identical to enrich_with_ddef.py — chunked by frame range
via pyarrow predicate pushdown. Peak ~1.6 GB per chunk.

Output: `test/dos_validation_raw_xt_ddef_theta.csv` (input + ~17 cols).
"""

import sys
sys.path.insert(0, ".")

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.theta import compute_pass_theta, compute_carry_theta
from src.loader import load_takeons
from src.preprocess import load_cached_events
from src.orientation import compute_orientations, add_dynamics

sys.path.insert(0, "test")
from validate_dos_outcomes import (        # noqa: E402
    _read_skeleton_window, _smooth_velocities,
    DEFAULT_MAX_FRAME_SPAN, DEFAULT_MAX_CHUNK_EVENTS,
)

RAW_IN = Path("test/dos_validation_raw_xt_ddef.csv")
OUT = Path("test/dos_validation_raw_xt_ddef_theta.csv")

FRAME_PAD = 25  # for velocity smoothing

# Theta keys we keep from the per-event compute_*_theta dict.
THETA_KEYS_PASS = [
    # Axis 1 — defender disruption
    "nearest_theta_head", "nearest_theta_shoulder", "nearest_theta_hip",
    "nearest_in_blind", "n_affected", "n_wrongfooted", "pct_wrongfooted",
    "mean_theta_affected", "mean_theta_head_all", "mean_theta_shoulder_all",
    "lane_theta_head", "lane_theta_flight",
    # Axis 2 — receiver advantage (passes only)
    "receiver_open_angle", "receiver_turn_needed", "n_teammates_in_fov",
]
THETA_KEYS_CARRY = [
    # Axis 1 only (carries / take-ons have no receiver)
    "nearest_theta_head", "nearest_theta_shoulder",
    "nearest_in_blind", "n_affected", "n_wrongfooted", "pct_wrongfooted",
    "mean_theta_affected", "mean_theta_head_all", "mean_theta_shoulder_all",
]

ALL_KEYS = sorted(set(THETA_KEYS_PASS) | set(THETA_KEYS_CARRY))


def _build_event_lookup(match: str) -> dict:
    """event_id -> dict with the original event fields needed by theta:
    x, y, x_receiver, y_receiver, x_end, y_end, play_angle, x_direction,
    y_direction, evaluation, xp, pressure_player, bypassed_defenders,
    back_line_break, distance, half. Take-ons are added with end coords
    set to NaN (carry-style theta path will skip them gracefully)."""
    ev = load_cached_events(match)
    out: dict = {}
    for r in ev[ev["event_type"].isin(("pass", "carry"))].itertuples(index=False):
        out[str(r.event_id)] = r._asdict()
    try:
        tk = load_takeons(match)
        for r in tk.itertuples(index=False):
            d = r._asdict()
            d["event_type"] = "takeon"
            d["x_end"] = float("nan"); d["y_end"] = float("nan")
            d["x_receiver"] = float("nan"); d["y_receiver"] = float("nan")
            out[str(r.event_id)] = d
    except FileNotFoundError:
        pass
    return out


def _empty_row(event_id: str) -> dict:
    return {"event_id": event_id, **{k: np.nan for k in ALL_KEYS}}


def _process_chunk(match: str, chunk: pd.DataFrame, lookup: dict) -> list:
    if len(chunk) == 0:
        return []
    f_min = int(chunk["frame"].min()) - FRAME_PAD
    f_max = int(chunk["frame"].max()) + FRAME_PAD

    skel = _read_skeleton_window(match, f_min, f_max)
    if len(skel) == 0:
        return []
    ori = compute_orientations(skel, smooth=True)
    del skel
    dyn = add_dynamics(ori)
    del ori
    dyn = _smooth_velocities(dyn)

    rows: list = []
    for r in chunk.itertuples(index=False):
        eid = str(r.event_id)
        meta = lookup.get(eid)
        if meta is None:
            rows.append(_empty_row(eid))
            continue

        # Build a Series compatible with compute_*_theta. parquet_frame
        # comes from the validation CSV (already correct, no need to
        # recompute from synced_frame_id).
        ev = pd.Series(meta)
        ev["parquet_frame"] = int(r.frame)
        ev["half"] = int(r.half)

        attacking_team = int(r.attacking_team)
        attacking_right = bool(r.attacking_right)

        try:
            if r.event_type == "pass":
                res = compute_pass_theta(ev, dyn, attacking_team, attacking_right)
            else:  # carry, takeon
                # Take-ons have no x_end/y_end: compute_carry_theta returns
                # an empty result (NaN) which is the honest behaviour.
                res = compute_carry_theta(ev, dyn, attacking_team, attacking_right)
        except Exception:
            rows.append(_empty_row(eid))
            continue

        rows.append({"event_id": eid,
                     **{k: res.get(k, np.nan) for k in ALL_KEYS}})

    del dyn
    gc.collect()
    return rows


def _split_chunks(events: pd.DataFrame, max_span: int, max_events: int) -> list:
    chunks = []
    n = len(events)
    if n == 0:
        return chunks
    start = 0
    frames = events["frame"].to_numpy()
    while start < n:
        f0 = int(frames[start])
        end = start + 1
        while (end < n
               and (end - start) < max_events
               and (int(frames[end]) - f0) <= max_span):
            end += 1
        chunks.append(events.iloc[start:end].copy())
        start = end
    return chunks


def _process_match(df: pd.DataFrame, match: str) -> list:
    sub = df[df["match"] == match].sort_values("frame").reset_index(drop=True)
    if len(sub) == 0:
        return []
    print(f"\n[{match}] theta enrichment: {len(sub)} events")
    lookup = _build_event_lookup(match)
    print(f"  event lookup: {len(lookup)} entries")

    chunks = _split_chunks(sub, DEFAULT_MAX_FRAME_SPAN, DEFAULT_MAX_CHUNK_EVENTS)
    print(f"  {len(chunks)} chunks")

    out_rows: list = []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        f_lo = int(chunk["frame"].min())
        f_hi = int(chunk["frame"].max())
        rows = _process_chunk(match, chunk, lookup)
        out_rows.extend(rows)
        print(f"  chunk {i+1}/{len(chunks)}  frames {f_lo}-{f_hi}  "
              f"events={len(chunk)}  rows_kept={len(rows)}  "
              f"elapsed={time.time()-t0:.0f}s")
        gc.collect()
    return out_rows


def main():
    if not RAW_IN.exists():
        raise SystemExit(f"Missing {RAW_IN}. Run enrich_with_ddef.py first.")
    df = pd.read_csv(RAW_IN)
    print(f"Loaded {RAW_IN}: {len(df):,} rows")

    all_rows: list = []
    for m in df["match"].unique():
        rows = _process_match(df, m)
        all_rows.extend(rows)
        gc.collect()

    if not all_rows:
        raise SystemExit("No theta rows produced.")

    th_df = pd.DataFrame(all_rows)
    print(f"\nTheta rows: {len(th_df):,}")

    # Coerce event_id to string on both sides so the merge is reliable
    df["event_id"] = df["event_id"].astype(str)
    th_df["event_id"] = th_df["event_id"].astype(str)

    merged = df.merge(th_df, on="event_id", how="left", suffixes=("", "_th"))
    print(f"Merged shape: {merged.shape}")

    # Sanity: how many events got a non-NaN theta?
    n_with = int(merged["nearest_theta_shoulder"].notna().sum())
    print(f"Events with theta: {n_with:,}/{len(merged):,} "
          f"({n_with / len(merged) * 100:.1f}%)")

    merged.to_csv(OUT, index=False)
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
