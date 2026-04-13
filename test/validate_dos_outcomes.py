"""Empirical validation of DOS against real outcomes (5 cached matches).

Loads the skeleton in **frame chunks** via pyarrow predicate pushdown so
WSL never sees more than ~5M skeleton rows in memory at once. Computes
`dos_for_event` for every pass and carry, then runs:

  - Mann-Whitney U with rank-biserial effect size (DOS in events that
    led to chances vs the rest, line breaks vs the rest, completed vs
    failed passes).
  - Logistic regression for back_line_break and success ~ dos +
    distance + pressure + n_defenders_in_lane + n_defenders_goal_side.
  - Quintile chart with bootstrap 95% CIs on each rate.
  - Direction-class breakdown (forward / diagonal / sideways) on real
    pass directions.
  - Per-match consistency table (does the pattern hold in each game?).
  - Half-space vs center vs wing breakdown (Spielverlagerung's
    half-space hypothesis).
  - Awareness distribution and its link to line breaks (the cognitive
    layer of DOS made measurable).
  - Per-half pass vs carry split.

Usage:
    python3 test/validate_dos_outcomes.py [--sample N] [--matches NAMES...]
                                          [--chunk N]

`--sample N` randomly subsamples N events per match (default: all).
`--chunk N` controls the per-chunk frame batch (default 200 events).
Outputs:
    test/dos_validation_raw.csv
    test/dos_validation_quintiles.png
    test/dos_validation_report.md
"""

import sys
sys.path.insert(0, ".")

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu

from src.loader import MATCHES, compute_attacking_right, infer_attacking_team
from src.preprocess import load_cached_events, CACHE_DIR
from src.orientation import compute_orientations, add_dynamics
from src.dos import dos_for_event, default_params

# ── Config ───────────────────────────────────────────────────────────────

VISION_SMOOTHING = 1.0       # cheap; render uses 1.0 inside compute_dos_surface
SMOOTH_W, POLY = 13, 1       # Savitzky-Golay for vx, vy (matches render)
MAX_SPEED = 12.0
EVENT_TYPES = ("pass", "carry")
FRAME_PAD = 25               # frames before/after each chunk for vel smoothing
DEFAULT_MAX_FRAME_SPAN = 30000   # ~10 min of game @ 50fps; ~3-5M skeleton rows
DEFAULT_MAX_CHUNK_EVENTS = 250   # safety cap on events per chunk

# --- Carry multi-frame DOS ---------------------------------------------
CARRY_N_SAMPLES = 5          # frames sampled uniformly along [fs, fe]
CARRY_VIRTUAL_HORIZON_S = 1.0  # virtual-destination look-ahead in seconds
CARRY_MIN_SPEED = 0.8        # m/s: below this, carrier is "stationary",
                             # velocity direction is meaningless -> skip frame
CARRY_MIN_HORIZON_M = 3.0    # floor on virtual-destination distance (even
                             # for slow carries, evaluate DOS at 3m ahead)
MIN_CARRY_DURATION_FRAMES = 10   # carries shorter than this fall back to
                                 # start->end vector (not enough samples)
RNG = np.random.default_rng(42)

PITCH_X = 105.0
PITCH_Y = 68.0
HALF_SPACE_Y = (-PITCH_Y / 2 * 0.55, -PITCH_Y / 2 * 0.18,
                PITCH_Y / 2 * 0.18, PITCH_Y / 2 * 0.55)
# Spielverlagerung half-spaces: lateral strips between center and wings.
# We use |y| in (0.18*half_y, 0.55*half_y) as half-space, |y| < 0.18*half_y
# as center, |y| > 0.55*half_y as wing.


# ── Cache helpers ────────────────────────────────────────────────────────

def _load_metadata_home_gk_left_p1(match: str) -> bool:
    p = CACHE_DIR / match / "metadata.json"
    with open(p) as f:
        meta = json.load(f)
    val = meta["home_gk_left"].get("1", meta["home_gk_left"].get(1))
    return bool(val)


def _load_possessions_xg_map(match: str) -> dict:
    """event_id -> sum_xg_ind of the parent TeamPossession."""
    p = CACHE_DIR / match / "possessions.parquet"
    if not p.exists():
        return {}
    po = pd.read_parquet(p)
    out = {}
    for _, row in po.iterrows():
        xg = float(row["sum_xg_ind"])
        for eid in row["event_ids"]:
            out[str(eid)] = xg
    return out


def _read_skeleton_window(match: str, f_lo: int, f_hi: int) -> pd.DataFrame:
    """Read only skeleton rows in [f_lo, f_hi] using parquet predicate
    pushdown — bounds peak memory regardless of match size."""
    path = CACHE_DIR / match / "skeleton.parquet"
    table = pq.read_table(
        path,
        filters=[("frame_number", ">=", int(f_lo)),
                 ("frame_number", "<=", int(f_hi))],
    )
    return table.to_pandas()


def _smooth_velocities(dyn: pd.DataFrame) -> pd.DataFrame:
    raw_speed = (dyn["vx"]**2 + dyn["vy"]**2).pow(0.5)
    dyn.loc[raw_speed > MAX_SPEED, ["vx", "vy"]] = float("nan")
    for col in ("vx", "vy"):
        dyn[col] = (
            dyn.groupby(["team", "jersey"], sort=False)[col]
            .transform(lambda s: pd.Series(
                savgol_filter(s.fillna(0).values, SMOOTH_W, POLY),
                index=s.index))
        )
    return dyn


# ── Zone classification (Spielverlagerung half-spaces) ───────────────────

def _zone_of(x: float, y: float) -> str:
    """Return 'center', 'half_space', or 'wing' for a pitch coord."""
    ay = abs(y)
    if ay < (PITCH_Y / 2) * 0.18:
        return "center"
    if ay < (PITCH_Y / 2) * 0.55:
        return "half_space"
    return "wing"


# ── Per-match processing ────────────────────────────────────────────────

def _build_event_records(match: str, sample: Optional[int]) -> pd.DataFrame:
    """Load + filter cached events. The home/away mapping is NOT needed here:
    `attacking_team` is inferred per event from the skeleton (nearest player
    to the event position) inside `_process_chunk`. That keeps the script
    fully cache-driven and independent of the raw XML."""
    home_gk_left_p1 = _load_metadata_home_gk_left_p1(match)

    events = load_cached_events(match)
    ev = events[events["event_type"].isin(EVENT_TYPES)].copy()
    ev = ev[ev["parquet_frame"].notna() & (ev["parquet_frame"] > 0)]
    ev["parquet_frame"] = ev["parquet_frame"].astype(int)

    is_pass = ev["event_type"] == "pass"
    has_dest = (
        (is_pass & ev["x_receiver"].notna() & ev["y_receiver"].notna())
        | ((~is_pass) & ev["x_end"].notna() & ev["y_end"].notna())
    )
    ev = ev[has_dest].reset_index(drop=True)

    if sample is not None and len(ev) > sample:
        idx = np.sort(RNG.choice(len(ev), size=sample, replace=False))
        ev = ev.iloc[idx].reset_index(drop=True)

    ev = ev.sort_values("parquet_frame").reset_index(drop=True)
    ev["ctx_home_gk_left_p1"] = home_gk_left_p1
    ev["ctx_match"] = match
    return ev


def _locate_carrier(fo: pd.DataFrame, cx: float, cy: float,
                    max_dist: float = 3.0) -> Optional[tuple]:
    """Find (team, jersey) of the skeleton player closest to (cx, cy) at
    the start frame of a carry. Same idea as `infer_attacking_team` but
    returns the jersey too, so the caller can track that specific player
    through later frames of the carry trajectory."""
    if len(fo) == 0:
        return None
    d = np.hypot(fo["x"].values - cx, fo["y"].values - cy)
    i = int(np.argmin(d))
    if d[i] > max_dist:
        return None
    row = fo.iloc[i]
    return int(row["team"]), int(row["jersey"]), float(d[i])


def _evaluate_carry_multiframe(
    dyn_by_frame: dict,
    carry,               # event namedtuple
    attacking_team: int,
    attacking_right: bool,
    params: dict,
) -> Optional[dict]:
    """Multi-frame DOS evaluation for a carry.

    For each of N samples taken uniformly along [fs, fe], evaluate DOS
    using the **carrier's instantaneous velocity direction from the
    skeleton** as the event direction and a virtual destination
    `pos + vel * horizon_s` as the target cell. Aggregate with `max` —
    the peak diagonal moment of the carry trajectory. Falls back to the
    naive start→end vector if no frame has a usable velocity sample
    (very short / stationary carries)."""
    fs = int(carry.parquet_frame)
    fe_val = getattr(carry, "parquet_frame_end", None)
    fe = int(fe_val) if (fe_val is not None and pd.notna(fe_val)
                         and fe_val > fs) else fs
    duration = fe - fs
    fo0 = dyn_by_frame.get(fs)
    if fo0 is None or len(fo0) < 6:
        return None

    # Identify the carrier via skeleton proximity at the start frame.
    loc = _locate_carrier(fo0, float(carry.x), float(carry.y), max_dist=3.0)
    if loc is None:
        return None
    team, jersey, _ = loc

    # If the carry is too short to sample reliably, fall back.
    if duration < MIN_CARRY_DURATION_FRAMES:
        return None

    # Uniformly spaced frames across [fs, fe]
    sample_frames = np.linspace(fs, fe, CARRY_N_SAMPLES).astype(int)

    results = []
    for sf in sample_frames:
        fo = dyn_by_frame.get(int(sf))
        if fo is None:
            continue
        me = fo[(fo["team"] == team) & (fo["jersey"] == jersey)]
        if len(me) == 0:
            continue
        m = me.iloc[0]
        px, py = float(m["x"]), float(m["y"])
        vx = float(m["vx"]) if pd.notna(m["vx"]) else 0.0
        vy = float(m["vy"]) if pd.notna(m["vy"]) else 0.0
        speed = float(np.hypot(vx, vy))
        if speed < CARRY_MIN_SPEED:
            # Carrier stationary at this sample; no defined direction
            continue
        # Virtual destination: look ahead along the instantaneous velocity
        # direction. Use `max(horizon_s * speed, min_horizon_m)` so slow
        # carries still evaluate DOS at a reasonable lookahead.
        horizon_m = max(CARRY_VIRTUAL_HORIZON_S * speed, CARRY_MIN_HORIZON_M)
        dir_x, dir_y = vx / speed, vy / speed
        tx = px + horizon_m * dir_x
        ty = py + horizon_m * dir_y
        evd = {
            "event_type": "carry", "x": px, "y": py,
            "carry_end_x": tx, "carry_end_y": ty,
        }
        try:
            r = dos_for_event(
                fo, evd, attacking_team, attacking_right,
                params=params, vision_smoothing=VISION_SMOOTHING,
            )
        except Exception:
            continue
        results.append(r)

    if not results:
        return None

    # Aggregate: peak diagonal moment of the trajectory.
    best = max(results, key=lambda r: r["dos"])
    mean_dos = float(np.mean([r["dos"] for r in results]))
    mean_aw = float(np.mean([r["awareness_mean"] for r in results]))
    mean_delay = float(np.mean([r["detection_delay_mean"] for r in results]))
    return {
        "dos": best["dos"],
        "dos_mean": mean_dos,
        "ppcf_att_baseline": best["ppcf_att_baseline"],
        "ppcf_att_with_awareness": best["ppcf_att_with_awareness"],
        "direction_class": best["direction_class"],
        "event_direction": best["event_direction"],
        "awareness_mean": mean_aw,
        "detection_delay_mean": mean_delay,
        "n_samples_used": len(results),
    }


def _process_chunk(
    match: str,
    ev_chunk: pd.DataFrame,
    xg_map: dict,
    params: dict,
) -> List[dict]:
    """Compute DOS for one frame-bounded batch of events."""
    if len(ev_chunk) == 0:
        return []
    # Extend f_max by the longest carry end-frame in the chunk so the
    # skeleton load covers the full carry trajectory (needed for the
    # multi-frame sampling below).
    f_min = int(ev_chunk["parquet_frame"].min()) - FRAME_PAD
    carry_ends = ev_chunk["parquet_frame_end"].dropna()
    carry_end_max = int(carry_ends.max()) if len(carry_ends) else 0
    f_max = max(int(ev_chunk["parquet_frame"].max()), carry_end_max) + FRAME_PAD

    skel = _read_skeleton_window(match, f_min, f_max)
    if len(skel) == 0:
        return []

    ori = compute_orientations(skel, smooth=True)
    del skel
    dyn = add_dynamics(ori)
    del ori
    dyn = _smooth_velocities(dyn)

    dyn_by_frame = {f: g for f, g in dyn.groupby("frame_number", sort=False)}
    del dyn
    gc.collect()

    rows = []
    for e in ev_chunk.itertuples(index=False):
        f = int(e.parquet_frame)
        fo = dyn_by_frame.get(f)
        if fo is None or len(fo) < 6:
            continue

        # Determine attacking team via skeleton proximity (no XML needed).
        # `infer_attacking_team` looks for the player closest to (x, y) in
        # the orientations frame and returns its team integer (0/1, where
        # 1 = home in the skeleton convention). Falls back to None if no
        # player is within 3 m of the event position.
        attacking_team = infer_attacking_team(
            float(e.x), float(e.y), fo, f, max_dist=3.0)
        if attacking_team is None:
            continue
        attacking_right = compute_attacking_right(
            attacking_team, int(e.half), e.ctx_home_gk_left_p1)

        if e.event_type == "carry":
            # Multi-frame DOS: instantaneous velocity direction per sample
            # + virtual destination `pos + vel*horizon`. Aggregate with max.
            r = _evaluate_carry_multiframe(
                dyn_by_frame, e, attacking_team, attacking_right, params)
            if r is None:
                # Fallback: single-frame evaluation with start->end vector
                # (same as the old behaviour for very short carries).
                evd = {
                    "event_type": "carry", "x": e.x, "y": e.y,
                    "carry_end_x": e.x_end, "carry_end_y": e.y_end,
                }
                try:
                    r = dos_for_event(
                        fo, evd, attacking_team, attacking_right,
                        params=params, vision_smoothing=VISION_SMOOTHING,
                    )
                    r["n_samples_used"] = 0  # marker: fallback path
                    r["dos_mean"] = r["dos"]
                except Exception:
                    continue
            dest_x, dest_y = e.x_end, e.y_end
        else:
            evd = {
                "event_type": "pass", "x": e.x, "y": e.y,
                "x_receiver": e.x_receiver, "y_receiver": e.y_receiver,
            }
            try:
                r = dos_for_event(
                    fo, evd, attacking_team, attacking_right,
                    params=params, vision_smoothing=VISION_SMOOTHING,
                )
                r["n_samples_used"] = 1  # passes are single-frame by design
                r["dos_mean"] = r["dos"]
            except Exception:
                continue
            dest_x, dest_y = e.x_receiver, e.y_receiver

        if e.event_type == "carry":
            dist = float(np.hypot(e.x_end - e.x, e.y_end - e.y))
            pressure = float("nan")
        else:
            dist = float(e.distance) if pd.notna(e.distance) else float(
                np.hypot(e.x_receiver - e.x, e.y_receiver - e.y))
            pressure = float(e.pressure_player) if pd.notna(e.pressure_player) else float("nan")

        rows.append({
            "match": match,
            "event_id": str(e.event_id),
            "event_type": e.event_type,
            "half": int(e.half),
            "frame": f,
            "team_id": e.team_id,
            "attacking_team": attacking_team,
            "attacking_right": bool(attacking_right),
            "dos": r["dos"],
            "dos_mean": r.get("dos_mean", r["dos"]),
            "n_samples_used": r.get("n_samples_used", 1),
            "ppcf_baseline": r["ppcf_att_baseline"],
            "ppcf_with_awareness": r["ppcf_att_with_awareness"],
            "direction_class": r["direction_class"],
            "event_direction_rad": r["event_direction"],
            "awareness_mean": r["awareness_mean"],
            "detection_delay_mean": r["detection_delay_mean"],
            "distance": dist,
            "pressure_player": pressure,
            "n_def_lane": float(e.num_defenders_passing_lane) if pd.notna(e.num_defenders_passing_lane) else float("nan"),
            "n_def_goal": float(e.num_defenders_goal_side) if pd.notna(e.num_defenders_goal_side) else float("nan"),
            "evaluation": str(e.evaluation) if e.evaluation is not None else "",
            "back_line_break": bool(e.back_line_break) if e.back_line_break is not None else False,
            "bypassed_defenders": float(e.bypassed_defenders) if pd.notna(e.bypassed_defenders) else float("nan"),
            "through_ball": bool(e.through_ball) if e.through_ball is not None else False,
            "parent_possession_xg": xg_map.get(str(e.event_id), 0.0),
            "origin_zone": _zone_of(float(e.x), float(e.y)),
            "dest_zone": _zone_of(float(dest_x), float(dest_y)),
        })

    dyn_by_frame.clear()
    del dyn_by_frame
    gc.collect()
    return rows


def _split_into_chunks(
    ev: pd.DataFrame,
    max_frame_span: int,
    max_events: int,
) -> List[pd.DataFrame]:
    """Split a frame-sorted event DataFrame into chunks bounded by both
    a max number of events AND a max frame span. The frame-span bound is
    what protects WSL: peak skeleton memory scales with frame range, not
    with event count, so a chunk of 1 sparse event can still cost as much
    as a chunk of 200 dense ones if not bounded.
    """
    chunks = []
    if len(ev) == 0:
        return chunks
    start = 0
    n = len(ev)
    frames = ev["parquet_frame"].values
    while start < n:
        f0 = int(frames[start])
        end = start + 1
        while end < n and (end - start) < max_events and (
                int(frames[end]) - f0) <= max_frame_span:
            end += 1
        chunks.append(ev.iloc[start:end])
        start = end
    return chunks


def _process_match(match: str, sample: Optional[int],
                   max_frame_span: int, max_events: int) -> pd.DataFrame:
    print(f"\n[{match}] loading event index...")
    ev = _build_event_records(match, sample)
    print(f"  {len(ev):,} events queued")
    if len(ev) == 0:
        return pd.DataFrame()

    xg_map = _load_possessions_xg_map(match)
    params = default_params()

    chunks = _split_into_chunks(ev, max_frame_span, max_events)
    print(f"  {len(chunks)} chunk(s)  (max_frame_span={max_frame_span}, "
          f"max_events={max_events})")

    all_rows: List[dict] = []
    t0 = time.time()
    for c, chunk in enumerate(chunks):
        f_lo = int(chunk["parquet_frame"].min())
        f_hi = int(chunk["parquet_frame"].max())
        rows = _process_chunk(match, chunk, xg_map, params)
        all_rows.extend(rows)
        print(f"  chunk {c+1}/{len(chunks)}  frames {f_lo}-{f_hi} "
              f"(span={f_hi-f_lo})  events={len(chunk)}  "
              f"rows_kept={len(rows)}  elapsed={time.time()-t0:.0f}s")
        gc.collect()

    print(f"  {len(all_rows):,} rows total ({time.time()-t0:.0f}s)")
    return pd.DataFrame(all_rows)


# ── Statistics ──────────────────────────────────────────────────────────

def _mann_whitney(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    """One-sided MWU (alt='greater') of df.loc[mask, 'dos'] vs ~mask."""
    a = df.loc[mask, "dos"].dropna().values
    b = df.loc[~mask, "dos"].dropna().values
    if len(a) < 5 or len(b) < 5:
        return {"label": label, "n_pos": int(len(a)), "n_neg": int(len(b)),
                "u": float("nan"), "p": float("nan"),
                "median_pos": float("nan"), "median_neg": float("nan"),
                "mean_pos": float("nan"), "mean_neg": float("nan"),
                "rbc": float("nan")}
    u, p = mannwhitneyu(a, b, alternative="greater")
    # Rank-biserial correlation = 1 - 2*U2/(n1*n2). With alternative='greater'
    # scipy returns U for sample 'a', so:
    rbc = 1.0 - (2.0 * u) / (len(a) * len(b)) if (len(a) * len(b)) else float("nan")
    # Above is the formula for the *other* tail; flip sign so positive
    # rbc means "a > b" (matches our 'greater' alternative).
    rbc = -rbc
    return {
        "label": label,
        "n_pos": int(len(a)), "n_neg": int(len(b)),
        "u": float(u), "p": float(p),
        "median_pos": float(np.median(a)), "median_neg": float(np.median(b)),
        "mean_pos": float(np.mean(a)), "mean_neg": float(np.mean(b)),
        "rbc": float(rbc),
    }


def _logistic(df: pd.DataFrame, target: str, predictors: List[str]) -> dict:
    import statsmodels.api as sm
    sub = df.dropna(subset=["dos", target] + predictors).copy()
    if len(sub) < 50 or sub[target].nunique() < 2:
        return {"target": target, "n": int(len(sub)), "fit": False}
    X = sub[["dos"] + predictors].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = sub[target].astype(int)
    try:
        res = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
    except Exception as ex:
        return {"target": target, "n": int(len(sub)), "fit": False, "err": str(ex)}
    return {
        "target": target,
        "n": int(len(sub)),
        "fit": True,
        "coefs": dict(zip(X.columns, res.params.tolist())),
        "pvalues": dict(zip(X.columns, res.pvalues.tolist())),
        "pseudo_r2": float(res.prsquared),
        "llr_p": float(res.llr_pvalue),
    }


def _bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot=1000, alpha=0.05) -> tuple:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots[i] = stat(sample)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _quintiles(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["dos"]).copy()
    sub["dos_q"] = pd.qcut(sub["dos"], 5, labels=[
        "Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"], duplicates="drop")
    rows = []
    for q, g in sub.groupby("dos_q", observed=True):
        succ = (g["evaluation"] .isin({"successfullyCompleted", "successful"})).astype(float).values
        brk = g["back_line_break"].astype(float).values
        chc = (g["parent_possession_xg"] > 0).astype(float).values
        rows.append({
            "dos_q": str(q),
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "success_rate": float(succ.mean()) if len(succ) else float("nan"),
            "success_ci": _bootstrap_ci(succ),
            "line_break_rate": float(brk.mean()) if len(brk) else float("nan"),
            "line_break_ci": _bootstrap_ci(brk),
            "chance_rate": float(chc.mean()) if len(chc) else float("nan"),
            "chance_ci": _bootstrap_ci(chc),
            "xg_mean": float(g["parent_possession_xg"].mean()),
        })
    return pd.DataFrame(rows).set_index("dos_q")


def _direction_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, g in df.groupby("direction_class"):
        succ = (g["evaluation"] .isin({"successfullyCompleted", "successful"})).astype(float).values
        brk = g["back_line_break"].astype(float).values
        chc = (g["parent_possession_xg"] > 0).astype(float).values
        rows.append({
            "direction_class": cls,
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "success_rate": float(succ.mean()) if len(succ) else float("nan"),
            "line_break_rate": float(brk.mean()) if len(brk) else float("nan"),
            "chance_rate": float(chc.mean()) if len(chc) else float("nan"),
            "awareness_mean": float(g["awareness_mean"].mean()),
            "xg_mean": float(g["parent_possession_xg"].mean()),
        })
    out = pd.DataFrame(rows).set_index("direction_class")
    return out.reindex(["forward", "diagonal", "sideways"]).dropna(how="all")


def _zone_breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for z, g in df.groupby(col):
        rows.append({
            "zone": z, "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "dos_pos_rate": float((g["dos"] > 0).mean()),
            "line_break_rate": float(g["back_line_break"].mean()),
            "chance_rate": float((g["parent_possession_xg"] > 0).mean()),
            "awareness_mean": float(g["awareness_mean"].mean()),
        })
    return pd.DataFrame(rows).set_index("zone").reindex(["center", "half_space", "wing"]).dropna(how="all")


def _per_match(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("match"):
        rows.append({
            "match": m,
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "dos_pos_rate": float((g["dos"] > 0).mean()),
            "line_break_rate": float(g["back_line_break"].mean()),
            "chance_rate": float((g["parent_possession_xg"] > 0).mean()),
            "diag_share": float((g["direction_class"] == "diagonal").mean()),
            "fwd_share": float((g["direction_class"] == "forward").mean()),
        })
    return pd.DataFrame(rows).set_index("match")


def _bonferroni(p_values: List[float]) -> List[float]:
    n = sum(1 for p in p_values if p == p)
    return [min(1.0, p * n) if p == p else float("nan") for p in p_values]


# ── Plot ─────────────────────────────────────────────────────────────────

def _plot_quintiles(quintiles: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = list(quintiles.index)
    x = np.arange(len(labels))
    metrics = [
        ("success_rate", "success_ci", "Pass success rate", "tab:blue"),
        ("line_break_rate", "line_break_ci", "Back line break rate", "tab:orange"),
        ("chance_rate", "chance_ci", "Possession ended in shot", "tab:red"),
    ]
    for ax, (col, ci_col, title, color) in zip(axes, metrics):
        vals = quintiles[col].values
        cis = quintiles[ci_col].tolist()
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("rate")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(0.05, hi.max() * 1.15))
    fig.suptitle("DOS quintiles vs real outcomes (5 matches, 95% bootstrap CI)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────

def _fmt_p(p):
    if p != p:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _write_report(
    df: pd.DataFrame,
    mw_results: List[dict],
    log_results: List[dict],
    quintiles: pd.DataFrame,
    direction_table: pd.DataFrame,
    zone_origin_table: pd.DataFrame,
    zone_dest_table: pd.DataFrame,
    per_match: pd.DataFrame,
    out_path: Path,
    elapsed: float,
):
    lines = []
    lines.append("# DOS empirical validation report")
    lines.append("")
    lines.append(f"_Generated by `test/validate_dos_outcomes.py` in {elapsed:.0f}s._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Matches analysed: **{df['match'].nunique()}**")
    lines.append(f"- Events evaluated: **{len(df):,}** "
                 f"(passes={int((df['event_type']=='pass').sum())}, "
                 f"carries={int((df['event_type']=='carry').sum())})")
    lines.append(f"- DOS source: `dos_for_event` on the **real** event "
                 f"direction (vision smoothing={VISION_SMOOTHING}, default "
                 f"PPCF params).")
    lines.append("- Outcomes (NOT used as model inputs): `evaluation`, "
                 "`back_line_break`, `bypassed_defenders`, "
                 "`parent_possession.sum_xg_ind`.")
    lines.append("- Critical: the model has **no access to xG, xT, success "
                 "labels, or any value model**. Inputs are pure geometry + "
                 "vision + skeleton orientation.")
    lines.append("")

    lines.append("## DOS distribution")
    lines.append("")
    s = df["dos"].describe()
    lines.append("```")
    lines.append(s.to_string())
    lines.append("```")
    lines.append("")
    lines.append(f"- DOS > 0: **{(df['dos']>0).mean()*100:.1f}%** of events")
    lines.append(f"- DOS > 0.01: **{(df['dos']>0.01).mean()*100:.1f}%**")
    lines.append(f"- DOS > 0.05: **{(df['dos']>0.05).mean()*100:.1f}%**")
    lines.append("")

    lines.append("## H1 — Mann-Whitney U with effect size")
    lines.append("")
    lines.append("One-sided alternative: events with positive outcome have "
                 "**higher** DOS than the rest. `rbc` = rank-biserial "
                 "correlation (effect size; +1 = perfect, 0 = no effect, "
                 "−1 = inverse). `p_bonf` = Bonferroni-corrected p across "
                 "the H1 family.")
    lines.append("")
    p_raw = [r.get("p", float("nan")) for r in mw_results]
    p_bonf = _bonferroni(p_raw)
    lines.append("| Outcome | n+ | n− | median DOS (+) | median DOS (−) | mean DOS (+) | mean DOS (−) | rbc | U | p | p (bonf) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r, pb in zip(mw_results, p_bonf):
        lines.append(
            f"| {r['label']} | {r['n_pos']} | {r['n_neg']} | "
            f"{r['median_pos']:.4f} | {r['median_neg']:.4f} | "
            f"{r['mean_pos']:.4f} | {r['mean_neg']:.4f} | "
            f"{r['rbc']:+.3f} | {r['u']:.0f} | {_fmt_p(r['p'])} | {_fmt_p(pb)} |"
        )
    lines.append("")

    lines.append("## H2 — Logistic regression (passes only)")
    lines.append("")
    lines.append("`outcome ~ dos + distance + pressure + n_def_lane + "
                 "n_def_goal`. The DOS coefficient should remain "
                 "significant after controlling for the obvious confounders.")
    lines.append("")
    for r in log_results:
        lines.append(f"### Target: `{r['target']}`")
        lines.append("")
        if not r.get("fit"):
            lines.append(f"_Fit skipped (n={r['n']}{', '+r.get('err','') if r.get('err') else ''})._")
            lines.append("")
            continue
        lines.append(f"- N = {r['n']}")
        lines.append(f"- Pseudo R² = {r['pseudo_r2']:.4f}")
        lines.append(f"- LLR p = {_fmt_p(r['llr_p'])}")
        lines.append("")
        lines.append("| coef | value | p-value |")
        lines.append("|---|---:|---:|")
        for k in ("const", "dos", "distance", "pressure_player",
                  "n_def_lane", "n_def_goal"):
            if k in r["coefs"]:
                lines.append(f"| {k} | {r['coefs'][k]:+.4f} | "
                             f"{_fmt_p(r['pvalues'][k])} |")
        lines.append("")

    lines.append("## H3 — DOS quintiles vs outcomes")
    lines.append("")
    lines.append("If the model captures real signal, success and chance "
                 "rates should grow monotonically across DOS quintiles "
                 "(95% bootstrap CIs).")
    lines.append("")
    qstr = quintiles.copy()
    for c in ["success_ci", "line_break_ci", "chance_ci"]:
        qstr[c] = qstr[c].apply(lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]")
    lines.append("```")
    lines.append(qstr.to_string())
    lines.append("```")
    lines.append("")
    lines.append("![DOS quintiles vs outcomes](dos_validation_quintiles.png)")
    lines.append("")

    lines.append("## H4 — Direction class breakdown")
    lines.append("")
    lines.append("Direction class assigned by `dos_for_event` from the real "
                 "event vector vs forward axis. This is the most direct test "
                 "of Spielverlagerung's diagonality thesis: forward and "
                 "diagonal classes should differ in line breaks and chance "
                 "creation.")
    lines.append("")
    lines.append("```")
    lines.append(direction_table.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## H5 — Zone breakdown (Spielverlagerung half-spaces)")
    lines.append("")
    lines.append("Half-space defined as |y| in (0.18, 0.55) of the half "
                 "pitch width — the lateral strip between the central column "
                 "(|y|<0.18) and the wing (|y|>0.55).")
    lines.append("")
    lines.append("**Origin zone** (where the ball is when the event starts):")
    lines.append("")
    lines.append("```")
    lines.append(zone_origin_table.to_string())
    lines.append("```")
    lines.append("")
    lines.append("**Destination zone** (where the ball is delivered):")
    lines.append("")
    lines.append("```")
    lines.append(zone_dest_table.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Per-match consistency")
    lines.append("")
    lines.append("If DOS captures a real causal mechanism, the pattern "
                 "should hold across **every** match — not just on average.")
    lines.append("")
    lines.append("```")
    lines.append(per_match.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Awareness layer")
    lines.append("")
    lines.append("`awareness_mean` is the mean defender awareness toward the "
                 "real event direction. Lower awareness = defenders blinder "
                 "= bigger DOS gain.")
    lines.append("")
    aw_corr = df[["awareness_mean", "dos"]].corr().iloc[0, 1]
    lines.append(f"- Pearson(awareness, dos) = **{aw_corr:+.3f}**")
    lines.append(f"- Mean awareness on back-line-break events: "
                 f"{df.loc[df['back_line_break'], 'awareness_mean'].mean():.3f}")
    lines.append(f"- Mean awareness on the rest: "
                 f"{df.loc[~df['back_line_break'], 'awareness_mean'].mean():.3f}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\nReport written: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def run_validation(matches: List[str], sample: Optional[int],
                   max_frame_span: int, max_events: int) -> dict:
    t0 = time.time()
    parts = []
    for m in matches:
        df_m = _process_match(
            m, sample=sample,
            max_frame_span=max_frame_span,
            max_events=max_events,
        )
        if len(df_m):
            parts.append(df_m)
        gc.collect()

    if not parts:
        print("No data collected.")
        return {}

    df = pd.concat(parts, ignore_index=True)
    out_csv = Path("test/dos_validation_raw.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nRaw rows: {len(df):,} -> {out_csv}")

    df["chance"] = df["parent_possession_xg"] > 0
    df["success"] = df["evaluation"] .isin({"successfullyCompleted", "successful"})

    mw_results = [
        _mann_whitney(df, df["chance"], "led_to_chance (xg>0)"),
        _mann_whitney(df, df["back_line_break"], "back_line_break"),
        _mann_whitney(df[df["event_type"] == "pass"],
                      df.loc[df["event_type"] == "pass", "success"],
                      "successfully_completed (passes)"),
        _mann_whitney(df, df["through_ball"], "through_ball"),
    ]

    log_results = [
        _logistic(df, "back_line_break",
                  ["distance", "pressure_player", "n_def_lane", "n_def_goal"]),
        _logistic(df.assign(success=df["success"].astype(int)),
                  "success",
                  ["distance", "pressure_player", "n_def_lane", "n_def_goal"]),
        _logistic(df.assign(chance=df["chance"].astype(int)),
                  "chance",
                  ["distance", "pressure_player", "n_def_lane", "n_def_goal"]),
    ]

    quintiles = _quintiles(df)
    direction_table = _direction_breakdown(df)
    zone_origin = _zone_breakdown(df, "origin_zone")
    zone_dest = _zone_breakdown(df, "dest_zone")
    per_match = _per_match(df)

    print("\n=== Mann-Whitney U ===")
    for r in mw_results:
        print(f"  {r['label']:35s} p={_fmt_p(r['p']):>10s}  "
              f"rbc={r['rbc']:+.3f}  med {r['median_pos']:.4f} vs {r['median_neg']:.4f}")
    print("\n=== Quintiles ===")
    print(quintiles.to_string())
    print("\n=== Direction class ===")
    print(direction_table.to_string())

    _plot_quintiles(quintiles, Path("test/dos_validation_quintiles.png"))

    elapsed = time.time() - t0
    _write_report(
        df, mw_results, log_results, quintiles, direction_table,
        zone_origin, zone_dest, per_match,
        Path("test/dos_validation_report.md"),
        elapsed,
    )
    print(f"\nDone in {elapsed:.0f}s")
    return {
        "df": df, "mw": mw_results, "log": log_results,
        "quintiles": quintiles, "direction": direction_table,
        "zone_origin": zone_origin, "zone_dest": zone_dest,
        "per_match": per_match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="random subsample of events per match")
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--max-frame-span", type=int, default=DEFAULT_MAX_FRAME_SPAN,
                    help="max parquet-frame range per skeleton chunk; the "
                         "primary memory cap (default 30000 ~ 10 minutes "
                         "of game @ 50fps)")
    ap.add_argument("--max-chunk-events", type=int, default=DEFAULT_MAX_CHUNK_EVENTS,
                    help="hard cap on events per chunk (default 250)")
    args = ap.parse_args()
    matches = args.matches if args.matches else list(MATCHES.keys())
    run_validation(matches, args.sample, args.max_frame_span,
                   args.max_chunk_events)


if __name__ == "__main__":
    main()
