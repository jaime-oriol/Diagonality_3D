"""Enrich `dos_validation_raw_xt.csv` with D-Def (Goes et al. 2019) per
event so we can test H2 from the original proposal:

  H2: high-θ / diagonal actions disrupt PC1 (longitudinal) AND PC2
      (lateral) simultaneously, while forward / sideways actions disrupt
      only one axis.

Pipeline:
  1. Read raw_xt.csv (DOS + xT + outcomes per event, 6.9k rows).
  2. For each match, chunk by frame range (memory-safe for WSL).
  3. Per chunk: load skeleton (with +200-frame buffer for the 3s D-Def
     window), compute orientations + dynamics + smoothing, run
     `compute_event_ddef` and `compute_local_ddef` per event.
  4. Concatenate across all matches → run `_apply_pca` ACROSS the full
     dataset (Goes et al. methodology: z-score deltas, PCA, decompose
     into PC1=longitudinal, PC2=lateral, PC3=shape).
  5. Save `dos_validation_raw_xt_ddef.csv` (raw_xt + 14 state +
     14×len(windows) deltas + 6 local + 8 PCA = ~70 cols total).

Memory model: identical to validate_dos_outcomes.py — peak ~1.6 GB per
chunk, gc-collected between matches. Total runtime ~3-5 min on the 5
cached matches.

NO orientation / vision / DOS recomputation — uses raw_xt.csv as the
event spine. `frame` and `attacking_team` already set there.
"""

import sys
sys.path.insert(0, ".")

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ddef import (
    compute_event_ddef, compute_local_ddef,
    _apply_pca, _empty_ddef,
    DDEF_WINDOWS, STATE_NAMES,
)
from src.loader import load_match_info, MATCHES, compute_attacking_right
from src.preprocess import CACHE_DIR
from src.orientation import compute_orientations, add_dynamics

# Reuse memory-safe primitives from validate_dos_outcomes.py
sys.path.insert(0, "test")
from validate_dos_outcomes import (        # noqa: E402
    _read_skeleton_window, _smooth_velocities,
    DEFAULT_MAX_FRAME_SPAN, DEFAULT_MAX_CHUNK_EVENTS,
)

RAW_XT = Path("test/dos_validation_raw_xt.csv")
OUT = Path("test/dos_validation_raw_xt_ddef.csv")

# D-def needs frames at event_frame AND event_frame + 150 (3s window).
# Skeleton load buffer must cover that + a Savitzky-Golay margin.
FRAME_PAD_PRE = 25                          # for velocity smoothing
FRAME_PAD_POST = 200                        # 3s + margin


# ── GK identification per match ────────────────────────────────────────

def _gk_jersey_per_team(match: str) -> dict:
    """Return {team_int: gk_jersey} from match_info.xml (PlayingPosition='TW')."""
    info = load_match_info(match)
    out: dict = {}
    for team_int, key in [(1, "home_players"), (0, "away_players")]:
        for p in info.get(key, []) or []:
            if p.get("position") == "TW" and p.get("starting"):
                jersey = p.get("shirt_number")
                if isinstance(jersey, int) and jersey > 0:
                    out[team_int] = jersey
                    break
        # Fallback: any GK if no starting flag
        if team_int not in out:
            for p in info.get(key, []) or []:
                if p.get("position") == "TW":
                    jersey = p.get("shirt_number")
                    if isinstance(jersey, int) and jersey > 0:
                        out[team_int] = jersey
                        break
        # Last resort: jersey 1
        out.setdefault(team_int, 1)
    return out


# ── Chunking by frame span ─────────────────────────────────────────────

def _split_chunks(events_sorted: pd.DataFrame,
                  max_span: int, max_events: int) -> list:
    chunks = []
    start = 0
    n = len(events_sorted)
    if n == 0:
        return chunks
    frames = events_sorted["frame"].to_numpy()
    while start < n:
        f0 = int(frames[start])
        end = start + 1
        while (end < n
               and (end - start) < max_events
               and (int(frames[end]) - f0) <= max_span):
            end += 1
        chunks.append(events_sorted.iloc[start:end].copy())
        start = end
    return chunks


# ── Per-chunk D-Def computation ────────────────────────────────────────

def _process_chunk(match: str, chunk: pd.DataFrame,
                   gk_jerseys: dict) -> list:
    if len(chunk) == 0:
        return []
    f_min = int(chunk["frame"].min()) - FRAME_PAD_PRE
    f_max = int(chunk["frame"].max()) + FRAME_PAD_POST

    skel = _read_skeleton_window(match, f_min, f_max)
    if len(skel) == 0:
        return []
    ori = compute_orientations(skel, smooth=True)
    del skel
    dyn = add_dynamics(ori)
    del ori
    dyn = _smooth_velocities(dyn)
    # ddef.compute_event_ddef internally filters by frame_number, so
    # passing the full per-chunk dyn DataFrame is fine.

    rows: list = []
    for r in chunk.itertuples(index=False):
        attacking_team = int(r.attacking_team)
        defending_team = 1 - attacking_team
        attacking_right = bool(r.attacking_right)
        gk = int(gk_jerseys.get(defending_team, 1))

        # GLOBAL D-def (block centroid, lines, area, spread, inter-line
        # distances). Returns state_t0_* + delta_*_2s + delta_*_3s.
        ddef_d = compute_event_ddef(
            int(r.frame), dyn, defending_team,
            attacking_right, gk, DDEF_WINDOWS,
        )

        # LOCAL D-def (5 nearest defenders, Forcher 2024). Origin coords
        # for passes/take-ons; carry origin too — measures the local
        # disruption around where the action started.
        bx = float(r.x_origin) if pd.notna(r.x_origin) else float("nan")
        by = float(r.y_origin) if pd.notna(r.y_origin) else float("nan")
        if not np.isnan(bx) and not np.isnan(by):
            local = compute_local_ddef(
                int(r.frame), dyn, defending_team,
                bx, by, gk, window=3.0,
            )
        else:
            local = {
                "local_area_t0": np.nan, "local_area_tw": np.nan,
                "local_spread_t0": np.nan, "local_spread_tw": np.nan,
                "local_delta_area": np.nan, "local_delta_spread": np.nan,
            }

        merged = {**ddef_d, **local, "event_id": str(r.event_id)}
        rows.append(merged)

    del dyn
    gc.collect()
    return rows


# ── Per-match driver ───────────────────────────────────────────────────

def _process_match(df: pd.DataFrame, match: str) -> list:
    sub = df[df["match"] == match].sort_values("frame").reset_index(drop=True)
    if len(sub) == 0:
        return []
    print(f"\n[{match}] D-def enrichment: {len(sub)} events")
    gk_jerseys = _gk_jersey_per_team(match)
    print(f"  GK jerseys: {gk_jerseys}")
    chunks = _split_chunks(sub, DEFAULT_MAX_FRAME_SPAN, DEFAULT_MAX_CHUNK_EVENTS)
    print(f"  {len(chunks)} chunks (max_span={DEFAULT_MAX_FRAME_SPAN}, "
          f"max_events={DEFAULT_MAX_CHUNK_EVENTS})")

    out_rows: list = []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        f_lo = int(chunk["frame"].min())
        f_hi = int(chunk["frame"].max())
        rows = _process_chunk(match, chunk, gk_jerseys)
        out_rows.extend(rows)
        print(f"  chunk {i+1}/{len(chunks)}  frames {f_lo}-{f_hi}  "
              f"events={len(chunk)}  rows_kept={len(rows)}  "
              f"elapsed={time.time()-t0:.0f}s")
        gc.collect()
    return out_rows


# ── Main ───────────────────────────────────────────────────────────────

def main():
    if not RAW_XT.exists():
        raise SystemExit(f"Missing {RAW_XT}. Run enrich_with_xt.py first.")
    df = pd.read_csv(RAW_XT)
    print(f"Loaded {RAW_XT}: {len(df):,} rows")

    all_rows: list = []
    for m in df["match"].unique():
        rows = _process_match(df, m)
        all_rows.extend(rows)
        gc.collect()

    if not all_rows:
        raise SystemExit("No D-def rows produced.")

    ddef_df = pd.DataFrame(all_rows)
    print(f"\nD-def rows: {len(ddef_df):,}")

    # Coerce event_id to string on both sides — raw_xt.csv may parse it
    # as int64 if all values are numeric, but ddef rows are str.
    df["event_id"] = df["event_id"].astype(str)
    ddef_df["event_id"] = ddef_df["event_id"].astype(str)

    # Merge with the original raw_xt.csv on event_id (1-to-1)
    merged = df.merge(ddef_df, on="event_id", how="left", suffixes=("", "_dd"))
    print(f"Merged shape: {merged.shape}")

    # PCA across the FULL dataset (Goes et al. — z-score deltas, PCA on top)
    print("\nRunning Z-score + PCA across the full dataset...")
    merged = _apply_pca(merged, DDEF_WINDOWS)

    # Sanity: PCA explained variance ratios should sum to ~the total
    for w in DDEF_WINDOWS:
        ws = f"{w:.0f}s"
        ratio_cols = [f"pca_var_ratio_pc{i+1}_{ws}" for i in range(3)]
        if all(c in merged.columns for c in ratio_cols):
            ratios = [merged[c].iloc[0] for c in ratio_cols]
            print(f"  PCA {ws} explained variance ratios: "
                  f"PC1={ratios[0]:.3f} PC2={ratios[1]:.3f} PC3={ratios[2]:.3f}")
        else:
            n_valid = merged[[f"delta_{n}_{ws}" for n in STATE_NAMES]].notna().all(axis=1).sum()
            print(f"  PCA {ws} skipped (only {n_valid} fully-valid rows; needs ≥10)")

    merged.to_csv(OUT, index=False)
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
