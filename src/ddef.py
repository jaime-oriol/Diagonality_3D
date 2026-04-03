"""
ddef — Defensive Disruption metric (Goes et al. 2019, extended).

Measures how much an action disrupts the defensive structure by comparing
the defensive state at t0 (action moment) vs t0+delta.

Implementation follows Goes et al. exactly:
  1. Assign defenders to 3 lines via K-Means on x-coordinate at t0
     (line labels are FIXED for both t0 and t0+delta — same players same line)
  2. Compute 10-variable state vector S at t0 and t0+delta
  3. Delta_S = S(t0+delta) - S(t0)
  4. Z-score standardize all deltas across events
  5. PCA -> PC1 (longitudinal), PC2 (lateral), PC3 (shape)
  6. D-Def = |PC1| + |PC2| + |PC3|

Extensions beyond Goes:
  - Local D-Def: 5 nearest defenders (Forcher et al. 2024)
  - Inter-line distances (Forcher et al. 2024)
  - Longitudinal/lateral stretch decomposition (Zhang et al. 2025)
  - Multiple windows (2s, 3s)
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FRAMERATE = 50
DDEF_WINDOWS = [2.0, 3.0]
N_LOCAL = 5

STATE_NAMES = [
    "cx", "cy",                     # team centroid
    "cx_def", "cy_def",             # defensive line centroid
    "cx_mid", "cy_mid",             # midfield line centroid
    "cx_att", "cy_att",             # attacking line centroid
    "area",                          # convex hull area
    "spread",                        # mean distance from centroid (stretch index)
    "spread_long",                   # longitudinal spread (x-axis)
    "spread_lat",                    # lateral spread (y-axis)
    "dist_def_mid",                  # inter-line: defensive to midfield
    "dist_mid_att",                  # inter-line: midfield to attacking
]


# --- Helpers ---------------------------------------------------------------

def _convex_hull_area(x: np.ndarray, y: np.ndarray) -> float:
    """Convex hull area of 2D points. Returns 0 if < 3 points."""
    if len(x) < 3:
        return 0.0
    try:
        return ConvexHull(np.column_stack([x, y])).volume
    except Exception:
        return 0.0


def _assign_lines(x_coords: np.ndarray, k: int = 3) -> np.ndarray:
    """Assign defenders to lines via K-Means on longitudinal coordinate.

    Returns labels sorted so 0=deepest (most negative x), 2=highest.
    Handles degenerate cases (all players at same x).
    """
    if len(x_coords) < k:
        return np.zeros(len(x_coords), dtype=int)
    # Degenerate: all players at nearly the same x
    if np.ptp(x_coords) < 0.5:
        return np.zeros(len(x_coords), dtype=int)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(x_coords.reshape(-1, 1))
    order = np.argsort([x_coords[labels == i].mean() for i in range(k)])
    return np.array([{old: new for new, old in enumerate(order)}[l] for l in labels])


# --- State vector ----------------------------------------------------------

def _compute_state(x: np.ndarray, y: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Compute 14-variable defensive state vector.

    Args:
        x, y: defender positions (normalized: defenders defend toward -X)
        lines: line labels per defender (0=def, 1=mid, 2=att), FIXED from t0
    """
    if len(x) < 3:
        return np.full(len(STATE_NAMES), np.nan)

    cx, cy = x.mean(), y.mean()

    # Line centroids (use line labels from t0 assignment)
    line_cx = np.array([x[lines == i].mean() if (lines == i).any() else cx for i in range(3)])
    line_cy = np.array([y[lines == i].mean() if (lines == i).any() else cy for i in range(3)])

    # Convex hull area
    area = _convex_hull_area(x, y)

    # Stretch index: overall, longitudinal, lateral
    spread = np.sqrt((x - cx)**2 + (y - cy)**2).mean()
    spread_long = np.abs(x - cx).mean()  # longitudinal component
    spread_lat = np.abs(y - cy).mean()   # lateral component

    # Inter-line distances (longitudinal)
    dist_def_mid = abs(line_cx[1] - line_cx[0])
    dist_mid_att = abs(line_cx[2] - line_cx[1])

    return np.array([
        cx, cy,
        line_cx[0], line_cy[0],
        line_cx[1], line_cy[1],
        line_cx[2], line_cy[2],
        area, spread, spread_long, spread_lat,
        dist_def_mid, dist_mid_att,
    ])


# --- Per-event D-Def -------------------------------------------------------

def compute_event_ddef(
    event_frame: int,
    orientations: pd.DataFrame,
    defending_team: int,
    attacking_right: bool,
    gk_jersey: int,
    windows: list = DDEF_WINDOWS,
) -> dict:
    """Compute D-Def for a single event.

    Line assignment is done ONCE at t0 and applied to all windows.
    Same defenders, same labels — delta measures actual movement, not re-clustering.
    """
    # Defenders at t0 (excluding GK)
    t0_ori = orientations[orientations["frame_number"] == event_frame]
    t0_def = t0_ori[(t0_ori["team"] == defending_team) & (t0_ori["jersey"] != gk_jersey)]

    if len(t0_def) < 3:
        return _empty_ddef(windows)

    # Normalize direction: defenders always toward -X
    x_t0 = t0_def["x"].values.copy()
    y_t0 = t0_def["y"].values.copy()
    if not attacking_right:
        x_t0 = -x_t0

    # Assign lines at t0 — these labels are FIXED for all windows
    lines_t0 = _assign_lines(x_t0)
    jerseys_t0 = t0_def["jersey"].values

    # State at t0
    s_t0 = _compute_state(x_t0, y_t0, lines_t0)

    result = {}
    for i, name in enumerate(STATE_NAMES):
        result[f"state_t0_{name}"] = s_t0[i]

    available_frames = set(orientations["frame_number"].unique())

    for w in windows:
        ws = f"{w:.0f}s"
        target_frame = event_frame + int(w * FRAMERATE)

        # Find closest available frame within 0.1s tolerance
        if target_frame not in available_frames:
            candidates = [f for f in available_frames if abs(f - target_frame) <= 5]
            if not candidates:
                for name in STATE_NAMES:
                    result[f"delta_{name}_{ws}"] = np.nan
                continue
            target_frame = min(candidates, key=lambda f: abs(f - target_frame))

        tw_ori = orientations[orientations["frame_number"] == target_frame]
        tw_def = tw_ori[(tw_ori["team"] == defending_team) & (tw_ori["jersey"] != gk_jersey)]

        # Match SAME players from t0 by jersey number
        tw_matched = tw_def[tw_def["jersey"].isin(jerseys_t0)]

        if len(tw_matched) < 3:
            for name in STATE_NAMES:
                result[f"delta_{name}_{ws}"] = np.nan
            continue

        # Build position arrays in same order as t0
        x_tw = np.full(len(jerseys_t0), np.nan)
        y_tw = np.full(len(jerseys_t0), np.nan)
        for j, jersey in enumerate(jerseys_t0):
            match = tw_matched[tw_matched["jersey"] == jersey]
            if len(match) > 0:
                x_tw[j] = match.iloc[0]["x"]
                y_tw[j] = match.iloc[0]["y"]

        if not attacking_right:
            x_tw = -x_tw

        # Drop players missing at t0+delta
        valid = ~np.isnan(x_tw)
        if valid.sum() < 3:
            for name in STATE_NAMES:
                result[f"delta_{name}_{ws}"] = np.nan
            continue

        # Use SAME line labels from t0 for matched players
        s_tw = _compute_state(x_tw[valid], y_tw[valid], lines_t0[valid])

        # Recompute t0 state for matched-only players (apples-to-apples)
        s_t0_matched = _compute_state(x_t0[valid], y_t0[valid], lines_t0[valid])

        delta = s_tw - s_t0_matched
        for i, name in enumerate(STATE_NAMES):
            result[f"delta_{name}_{ws}"] = delta[i]

    return result


# --- Local D-Def (Forcher et al. 2024) ------------------------------------

def compute_local_ddef(
    event_frame: int,
    orientations: pd.DataFrame,
    defending_team: int,
    ball_x: float,
    ball_y: float,
    attacking_right: bool,
    gk_jersey: int,
    window: float = 3.0,
    n_nearest: int = N_LOCAL,
) -> dict:
    """Local D-Def: area and spread of N nearest defenders to the ball."""
    t0_ori = orientations[orientations["frame_number"] == event_frame]
    t0_def = t0_ori[(t0_ori["team"] == defending_team) & (t0_ori["jersey"] != gk_jersey)]

    empty = {"local_area_t0": np.nan, "local_area_tw": np.nan,
             "local_spread_t0": np.nan, "local_spread_tw": np.nan,
             "local_delta_area": np.nan, "local_delta_spread": np.nan}

    if len(t0_def) < n_nearest:
        return empty

    # N nearest to ball at t0
    dists = np.sqrt((t0_def["x"].values - ball_x)**2 + (t0_def["y"].values - ball_y)**2)
    nearest_idx = np.argsort(dists)[:n_nearest]
    nearest_jerseys = t0_def.iloc[nearest_idx]["jersey"].values

    lx0 = t0_def.iloc[nearest_idx]["x"].values
    ly0 = t0_def.iloc[nearest_idx]["y"].values
    area_t0 = _convex_hull_area(lx0, ly0)
    cx0, cy0 = lx0.mean(), ly0.mean()
    spread_t0 = np.sqrt((lx0 - cx0)**2 + (ly0 - cy0)**2).mean()

    # At t0 + window: track SAME players
    target = event_frame + int(window * FRAMERATE)
    available = orientations["frame_number"].unique()
    if target not in available:
        candidates = [f for f in available if abs(f - target) <= 5]
        if not candidates:
            return {**empty, "local_area_t0": area_t0, "local_spread_t0": spread_t0}
        target = min(candidates, key=lambda f: abs(f - target))

    tw_ori = orientations[orientations["frame_number"] == target]
    tw_def = tw_ori[(tw_ori["team"] == defending_team)]
    tw_nearest = tw_def[tw_def["jersey"].isin(nearest_jerseys)]

    if len(tw_nearest) < 3:
        return {**empty, "local_area_t0": area_t0, "local_spread_t0": spread_t0}

    lxw = tw_nearest["x"].values
    lyw = tw_nearest["y"].values
    area_tw = _convex_hull_area(lxw, lyw)
    cxw, cyw = lxw.mean(), lyw.mean()
    spread_tw = np.sqrt((lxw - cxw)**2 + (lyw - cyw)**2).mean()

    return {
        "local_area_t0": area_t0, "local_area_tw": area_tw,
        "local_spread_t0": spread_t0, "local_spread_tw": spread_tw,
        "local_delta_area": area_tw - area_t0,
        "local_delta_spread": spread_tw - spread_t0,
    }


# --- Batch with PCA --------------------------------------------------------

def compute_all_ddef(
    theta_df: pd.DataFrame,
    orientations: pd.DataFrame,
    home_gk_left_p1: bool = False,
    home_gk_jersey: int = 1,
    away_gk_jersey: int = 1,
    windows: list = DDEF_WINDOWS,
) -> pd.DataFrame:
    """Compute D-Def for all events, then apply PCA decomposition.

    Args:
        theta_df: Output of compute_all_theta() — must have x, y, half columns
        orientations: Full orientation DataFrame
        home_gk_left_p1: GK side flag for direction normalization
        home_gk_jersey: Jersey number of home team GK
        away_gk_jersey: Jersey number of away team GK
    """
    results = []

    for _, row in theta_df.iterrows():
        frame = int(row.get("frame_number", -1))
        if frame <= 0:
            results.append(_empty_ddef(windows))
            continue

        # Infer attacking team from position match
        attacking_team = _infer_team(row, orientations, frame)
        if attacking_team is None:
            results.append(_empty_ddef(windows))
            continue

        defending_team = 1 - attacking_team
        half = int(row.get("half", 1))

        if half == 1:
            attacking_right = (attacking_team == 1) == home_gk_left_p1
        else:
            attacking_right = (attacking_team == 1) != home_gk_left_p1

        gk_jersey = away_gk_jersey if defending_team == 0 else home_gk_jersey

        # Global D-Def
        ddef = compute_event_ddef(frame, orientations, defending_team,
                                   attacking_right, gk_jersey, windows)

        # Local D-Def (use receiver position for passes, carrier position for carries)
        bx = row.get("x_receiver") if row.get("event_type") == "pass" else row.get("x")
        by = row.get("y_receiver") if row.get("event_type") == "pass" else row.get("y")
        if not pd.isna(bx) and not pd.isna(by):
            local = compute_local_ddef(frame, orientations, defending_team,
                                        bx, by, attacking_right, gk_jersey)
            ddef.update(local)
        else:
            ddef.update({"local_area_t0": np.nan, "local_area_tw": np.nan,
                        "local_spread_t0": np.nan, "local_spread_tw": np.nan,
                        "local_delta_area": np.nan, "local_delta_spread": np.nan})

        results.append(ddef)

    ddef_df = pd.DataFrame(results)
    merged = pd.concat([theta_df.reset_index(drop=True), ddef_df.reset_index(drop=True)], axis=1)
    merged = _apply_pca(merged, windows)
    return merged


def _apply_pca(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Z-score + PCA on delta vectors (Goes et al. methodology)."""
    for w in windows:
        ws = f"{w:.0f}s"
        delta_cols = [f"delta_{n}_{ws}" for n in STATE_NAMES]
        valid_mask = df[delta_cols].notna().all(axis=1)

        if valid_mask.sum() < 10:
            for i in range(3):
                df[f"pc{i+1}_{ws}"] = np.nan
            df[f"ddef_{ws}"] = np.nan
            continue

        X = df.loc[valid_mask, delta_cols].values
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(3, X_std.shape[1]))
        components = pca.fit_transform(X_std)

        for i in range(pca.n_components_):
            df.loc[valid_mask, f"pc{i+1}_{ws}"] = components[:, i]

        df.loc[valid_mask, f"ddef_{ws}"] = np.abs(components).sum(axis=1)
        df.attrs[f"pca_explained_variance_{ws}"] = pca.explained_variance_ratio_.tolist()

    return df


def _infer_team(row, orientations, frame):
    """Infer attacking team from event position."""
    ex, ey = row.get("x"), row.get("y")
    if pd.isna(ex) or pd.isna(ey):
        return None
    frame_ori = orientations[orientations["frame_number"] == frame]
    if len(frame_ori) == 0:
        return None
    dists = np.sqrt((frame_ori["x"].values - ex)**2 + (frame_ori["y"].values - ey)**2)
    if dists.min() < 3.0:
        return int(frame_ori.iloc[np.argmin(dists)]["team"])
    return None


def _empty_ddef(windows):
    result = {}
    for name in STATE_NAMES:
        result[f"state_t0_{name}"] = np.nan
    for w in windows:
        ws = f"{w:.0f}s"
        for name in STATE_NAMES:
            result[f"delta_{name}_{ws}"] = np.nan
    result.update({"local_area_t0": np.nan, "local_area_tw": np.nan,
                   "local_spread_t0": np.nan, "local_spread_tw": np.nan,
                   "local_delta_area": np.nan, "local_delta_spread": np.nan})
    return result


# --- Summary ---------------------------------------------------------------

def ddef_summary_by_direction(df: pd.DataFrame, window: str = "3s") -> pd.DataFrame:
    """Summarize D-Def by pass direction."""
    valid = df[df[f"ddef_{window}"].notna() & (df["direction"] != "unknown")]
    stats = valid.groupby("direction").agg(
        count=(f"ddef_{window}", "size"),
        mean_ddef=(f"ddef_{window}", "mean"),
        median_ddef=(f"ddef_{window}", "median"),
        mean_pc1_abs=(f"pc1_{window}", lambda x: x.abs().mean()),
        mean_pc2_abs=(f"pc2_{window}", lambda x: x.abs().mean()),
        mean_delta_spread_long=(f"delta_spread_long_{window}", "mean"),
        mean_delta_spread_lat=(f"delta_spread_lat_{window}", "mean"),
        mean_delta_dist_def_mid=(f"delta_dist_def_mid_{window}", "mean"),
        mean_delta_dist_mid_att=(f"delta_dist_mid_att_{window}", "mean"),
        mean_local_delta_area=("local_delta_area", "mean"),
        mean_local_delta_spread=("local_delta_spread", "mean"),
    ).reset_index()

    # PC1/PC2 balance ratio (computed separately to avoid fragile lambda)
    pc1 = stats["mean_pc1_abs"]
    pc2 = stats["mean_pc2_abs"]
    max_pc = np.maximum(pc1, pc2).replace(0, np.nan)
    stats["pc1_pc2_ratio"] = np.minimum(pc1, pc2) / max_pc

    return stats
