"""Enrich `dos_validation_raw_fixed.csv` with xT-delta as the per-action
outcome metric.

Rationale: `parent_possession_xg` (the current outcome) credits every
event in a TeamPossession sequence with the same xG, regardless of how
many events sit between the action and the eventual shot. This dilutes
the signal for carries and take-ons (which often happen mid-buildup).

xT-delta is the field-standard alternative since 2019 (Singh 2018 +
Decroos socceraction). For each move action:

    success → xt_delta = xT(dest) - xT(origin)
    fail    → xt_delta = -xT(origin)
    take-on → xt_origin only (dribbler stays at the same position; the
              value preserved is the xT of the duel zone)

This script joins every event in raw_fixed.csv against:
  - `cache/{match}/events.parquet` for passes/carries (origin + dest)
  - `loader.load_takeons(match)` for take-ons (origin only)

It then computes `xt_origin`, `xt_dest`, `xt_delta` and writes the
enriched CSV at `outputs/intermediate/dos_validation_raw_xt.csv`. Stats are recomputed
by `pipeline/compute_stats_xt.py`.
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import load_takeons, synced_frame_to_parquet
from src.preprocess import load_cached_events, load_cached_metadata
from src.xt import xt_at_tracab, xt_delta as xt_delta_fn

# ── Paths ──────────────────────────────────────────────────────────────

RAW_FIXED = Path("outputs/intermediate/dos_validation_raw_fixed.csv")
RAW_XT_OUT = Path("outputs/intermediate/dos_validation_raw_xt.csv")
RAW_XT_OUT.parent.mkdir(parents=True, exist_ok=True)


# ── Per-match coordinate lookup ────────────────────────────────────────

def _build_coords_map(match: str) -> dict:
    """Return a dict event_id -> dict(x, y, x_dest, y_dest, event_type).

    Pulls origin and destination coords from the cached events.parquet for
    passes and carries, and from `load_takeons(match)` for successful
    take-ons (which are not in events.parquet)."""
    out: dict = {}

    # Passes + carries from cached events.parquet
    ev = load_cached_events(match)
    pas = ev[ev["event_type"] == "pass"]
    for r in pas.itertuples(index=False):
        if pd.isna(r.x) or pd.isna(r.y):
            continue
        out[str(r.event_id)] = {
            "x": float(r.x), "y": float(r.y),
            "x_dest": float(r.x_receiver) if pd.notna(r.x_receiver) else np.nan,
            "y_dest": float(r.y_receiver) if pd.notna(r.y_receiver) else np.nan,
            "event_type": "pass",
        }
    car = ev[ev["event_type"] == "carry"]
    for r in car.itertuples(index=False):
        if pd.isna(r.x) or pd.isna(r.y):
            continue
        out[str(r.event_id)] = {
            "x": float(r.x), "y": float(r.y),
            "x_dest": float(r.x_end) if pd.notna(r.x_end) else np.nan,
            "y_dest": float(r.y_end) if pd.notna(r.y_end) else np.nan,
            "event_type": "carry",
        }

    # Take-ons from XML (not in events.parquet)
    try:
        tk = load_takeons(match)
        if len(tk) > 0:
            for r in tk.itertuples(index=False):
                if pd.isna(r.x) or pd.isna(r.y):
                    continue
                out[str(r.event_id)] = {
                    "x": float(r.x), "y": float(r.y),
                    "x_dest": np.nan, "y_dest": np.nan,    # see take-on note above
                    "event_type": "takeon",
                }
    except FileNotFoundError:
        # Match without raw XML — take-ons aren't recoverable. They stay
        # without xT enrichment (xt_origin = NaN) which is honest.
        pass

    return out


# ── Enrichment ─────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add xt_origin, xt_dest, xt_delta columns to a copy of `df`.

    Convention:
      success ≡ evaluation in {"successfullyCompleted", "successful"}
                (take-ons are filtered to attacker-wins so always success)
      pass / carry  → xt_delta as in `xt_delta_fn`
      take-on       → xt_dest = NaN, xt_delta = 0 (preservation; no
                      displacement). The interpretable outcome variable
                      for take-ons is xt_origin (the xT of the duel zone).
    """
    df = df.copy()

    # Per-match coords lookup, built once per match
    matches = sorted(df["match"].dropna().unique())
    coords_by_match: dict = {m: _build_coords_map(m) for m in matches}

    n = len(df)
    x_o = np.full(n, np.nan); y_o = np.full(n, np.nan)
    x_d = np.full(n, np.nan); y_d = np.full(n, np.nan)
    found = np.zeros(n, dtype=bool)

    for i, row in enumerate(df.itertuples(index=False)):
        cmap = coords_by_match.get(row.match, {})
        meta = cmap.get(str(row.event_id))
        if meta is None:
            continue
        x_o[i] = meta["x"]; y_o[i] = meta["y"]
        x_d[i] = meta["x_dest"]; y_d[i] = meta["y_dest"]
        found[i] = True

    df["x_origin"] = x_o
    df["y_origin"] = y_o
    df["x_dest"] = x_d
    df["y_dest"] = y_d
    df["coords_resolved"] = found

    ar = df["attacking_right"].astype(bool).to_numpy()
    df["xt_origin"] = xt_at_tracab(x_o, y_o, ar)

    # Destination xT only meaningful when dest coords exist
    has_dest = ~np.isnan(x_d) & ~np.isnan(y_d)
    xt_d = np.full(n, np.nan)
    if has_dest.any():
        xt_d[has_dest] = xt_at_tracab(x_d[has_dest], y_d[has_dest], ar[has_dest])
    df["xt_dest"] = xt_d

    # Success flag (take-ons are always success by filter; passes use
    # evaluation; carries are physical movements so always success)
    is_pass = (df["event_type"] == "pass").to_numpy()
    is_carry = (df["event_type"] == "carry").to_numpy()
    is_takeon = (df["event_type"] == "takeon").to_numpy()
    succ = pd.Series([False] * n, index=df.index)
    pass_succ = df["evaluation"].isin({"successfullyCompleted", "successful"}).to_numpy()
    succ[is_pass] = pass_succ[is_pass]
    succ[is_carry] = True       # carries always reach end frame
    succ[is_takeon] = True      # take-ons filtered to winners only

    delta = np.full(n, np.nan)
    # Passes + carries: xt_delta_fn handles success vs fail
    pc_mask = (is_pass | is_carry) & has_dest
    if pc_mask.any():
        delta[pc_mask] = xt_delta_fn(
            x_o[pc_mask], y_o[pc_mask],
            x_d[pc_mask], y_d[pc_mask],
            ar[pc_mask],
            success=succ.to_numpy()[pc_mask],
        )
    # Failed pass without dest coords: -xt_origin
    pass_fail_no_dest = is_pass & ~has_dest & ~succ.to_numpy()
    delta[pass_fail_no_dest] = -df.loc[pass_fail_no_dest, "xt_origin"].to_numpy()

    # Take-ons: xt_delta = 0 (preservation). The interpretable outcome is
    # `xt_origin` itself (the xT of the duel zone).
    delta[is_takeon] = 0.0

    df["xt_delta"] = delta
    return df


def main():
    if not RAW_FIXED.exists():
        raise SystemExit(f"Missing {RAW_FIXED}. Run validate + fix first.")

    df = pd.read_csv(RAW_FIXED)
    print(f"Loading {RAW_FIXED}: {len(df):,} rows")

    out = enrich(df)
    print(f"Coords resolved: {out['coords_resolved'].sum():,}/{len(out):,} "
          f"({out['coords_resolved'].mean()*100:.1f}%)")
    by_t = out.groupby("event_type")["coords_resolved"].agg(["sum", "size"])
    print(by_t)

    # Sanity: xt_origin / xt_delta distributions
    for name, mask in [
        ("pass",  out["event_type"] == "pass"),
        ("carry", out["event_type"] == "carry"),
        ("takeon", out["event_type"] == "takeon"),
    ]:
        s = out.loc[mask, "xt_delta"].dropna()
        ox = out.loc[mask, "xt_origin"].dropna()
        if len(s) > 0:
            print(f"\n{name}: n={len(s)}, "
                  f"xt_origin mean={ox.mean():.4f}, "
                  f"xt_delta mean={s.mean():+.4f}, "
                  f"p50={s.median():+.4f}, p95={s.quantile(0.95):+.4f}")

    out.to_csv(RAW_XT_OUT, index=False)
    print(f"\nWrote: {RAW_XT_OUT}")


if __name__ == "__main__":
    main()
