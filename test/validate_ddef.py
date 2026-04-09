"""Validate D-Def by direction: do diagonals disrupt BOTH PC1 (longitudinal)
AND PC2 (lateral) simultaneously, while forward only disrupts PC1 and
sideways only PC2? (Spielverlagerung core thesis)
"""
import sys; sys.path.insert(0, ".")
import gc
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from src.preprocess import load_cached_events, load_cached_skeleton
from src.orientation import compute_orientations, add_dynamics
from src.loader import load_match_info, compute_attacking_right, MATCHES
from src.ddef import compute_event_ddef, compute_local_ddef, STATE_NAMES


def classify_direction(pa):
    if pd.isna(pa): return "unknown"
    a = abs(float(pa))
    if a <= 22.5: return "forward"
    elif a <= 67.5: return "diagonal"
    elif a <= 112.5: return "sideways"
    return "backward"


def process_match(match, max_per_dir=50):
    print(f"\n{'='*60}")
    print(f"  MATCH: {match}")
    print(f"{'='*60}")

    events = load_cached_events(match)
    info = load_match_info(match)
    home_team_id = info.get("home_team_id", "")

    # GK jerseys
    def _find_gk(players):
        for p in players:
            if p.get("position") == "TW" and p.get("starting"):
                n = p.get("shirt_number")
                if isinstance(n, int) and n > 0: return n
        return 1
    home_gk = _find_gk(info["home_players"])
    away_gk = _find_gk(info["away_players"])

    actions = events[events["event_type"].isin(["pass", "carry"])].copy()
    actions = actions[actions["parquet_frame"] > 0].copy()
    actions["direction"] = actions["play_angle"].apply(classify_direction)
    actions = actions[actions["direction"] != "unknown"]
    actions["success"] = (actions["evaluation"] == "successfullyCompleted").astype(int)

    sample = actions.groupby("direction", group_keys=False).apply(
        lambda g: g.sample(min(max_per_dir, len(g)), random_state=42)
    ).reset_index(drop=True)
    print(f"  {len(sample)} events sampled")

    # Load full skeleton + orientations (need future frames for D-Def windows)
    print("  Loading skeleton + orientations...")
    skel = load_cached_skeleton(match)
    ori = compute_orientations(skel, smooth=True)
    dyn = add_dynamics(ori)
    del skel, ori
    gc.collect()
    print(f"  {len(dyn):,} orientation rows")

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        frame_id = int(row["parquet_frame"])
        team_id = row["team_id"]
        attacking_team = 1 if team_id == home_team_id else 0
        defending_team = 1 - attacking_team
        half = int(row.get("half", 1))
        attacking_right = compute_attacking_right(attacking_team, half, True)
        gk_jersey = away_gk if defending_team == 0 else home_gk

        try:
            ddef = compute_event_ddef(
                frame_id, dyn, defending_team, attacking_right, gk_jersey,
                windows=[2.0, 3.0])

            # Only keep events with valid state vectors
            if pd.isna(ddef.get("state_t0_cx", np.nan)):
                continue

            ddef["direction"] = row["direction"]
            ddef["success"] = int(row["success"])
            ddef["event_type"] = row["event_type"]
            ddef["progression"] = abs(
                float(row.get("x_receiver", row.get("x_end", row["x"]))) - float(row["x"]))
            results.append(ddef)
        except Exception:
            pass

        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(sample)} done")

    del dyn
    gc.collect()
    return results


# --- Run ---
all_results = []
for match in MATCHES:
    try:
        r = process_match(match, max_per_dir=50)
        all_results.extend(r)
        print(f"  -> {len(r)} events with D-Def")
    except Exception as e:
        print(f"  -> SKIP: {e}")

df = pd.DataFrame(all_results)

# Apply PCA manually (compute_all_ddef does this but we have raw deltas)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

for ws in ["2s", "3s"]:
    delta_cols = [f"delta_{n}_{ws}" for n in STATE_NAMES]
    valid = df[delta_cols].notna().all(axis=1)
    if valid.sum() < 10:
        print(f"  Not enough valid events for PCA ({ws})")
        continue
    X = df.loc[valid, delta_cols].values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(X_std)
    df.loc[valid, f"pc1_{ws}"] = pcs[:, 0]
    df.loc[valid, f"pc2_{ws}"] = pcs[:, 1]
    df.loc[valid, f"pc3_{ws}"] = pcs[:, 2]
    df.loc[valid, f"ddef_{ws}"] = np.abs(pcs[:, 0]) + np.abs(pcs[:, 1]) + np.abs(pcs[:, 2])
    print(f"\n  PCA {ws}: var explained = {pca.explained_variance_ratio_}")

print(f"\n\n{'#'*60}")
print(f"  TOTAL: {len(df)} events with D-Def across {df.get('match', pd.Series()).nunique()} matches")
print(f"{'#'*60}")

# ================================================================
# D-Def BY DIRECTION
# ================================================================
print("\n\n" + "=" * 60)
print("D-DEF BY DIRECTION (3s window)")
print("=" * 60)
ws = "3s"
for d in ["forward", "diagonal", "sideways", "backward"]:
    s = df[df["direction"] == d]
    if len(s) < 3: continue
    has_pc = f"pc1_{ws}" in s.columns and s[f"pc1_{ws}"].notna().sum() > 0
    if not has_pc: continue
    sv = s[s[f"ddef_{ws}"].notna()]
    print(f"\n  {d.upper()} (n={len(sv)}):")
    print(f"    D-Def mean:     {sv[f'ddef_{ws}'].mean():.3f}")
    print(f"    |PC1| (longit): {sv[f'pc1_{ws}'].abs().mean():.3f}")
    print(f"    |PC2| (lateral):{sv[f'pc2_{ws}'].abs().mean():.3f}")
    print(f"    |PC3| (shape):  {sv[f'pc3_{ws}'].abs().mean():.3f}")
    print(f"    Success:        {sv['success'].mean():.1%}")

# ================================================================
# THE SPIELVERLAGERUNG TEST: diagonals disrupt BOTH axes
# ================================================================
print("\n\n" + "=" * 60)
print("SPIELVERLAGERUNG TEST: DIAGONALS DISRUPT BOTH AXES?")
print("=" * 60)
if f"pc1_{ws}" in df.columns:
    for d in ["forward", "diagonal", "sideways"]:
        s = df[(df["direction"] == d) & df[f"pc1_{ws}"].notna()]
        if len(s) < 5: continue
        pc1 = s[f"pc1_{ws}"].abs().mean()
        pc2 = s[f"pc2_{ws}"].abs().mean()
        ratio = pc1 / (pc2 + 1e-9)
        print(f"  {d:>10}: |PC1|={pc1:.3f}, |PC2|={pc2:.3f}, ratio={ratio:.2f}")
    print("\n  Expected (Spielverlagerung):")
    print("    Forward:  high PC1, low PC2  (ratio >> 1)")
    print("    Diagonal: balanced PC1+PC2   (ratio ≈ 1)")
    print("    Sideways: low PC1, high PC2  (ratio << 1)")

# ================================================================
# D-DEF → SUCCESS
# ================================================================
print("\n\n" + "=" * 60)
print("D-DEF → SUCCESS?")
print("=" * 60)
if f"ddef_{ws}" in df.columns:
    valid = df[df[f"ddef_{ws}"].notna()]
    if len(valid) > 20:
        r, p = spearmanr(valid[f"ddef_{ws}"], valid["success"])
        print(f"  Spearman(D-Def, success): r={r:+.3f}, p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

    # By direction
    for d in ["forward", "diagonal"]:
        s = valid[valid["direction"] == d]
        if len(s) < 10: continue
        med = s[f"ddef_{ws}"].median()
        h = s[s[f"ddef_{ws}"] > med]
        l = s[s[f"ddef_{ws}"] <= med]
        print(f"\n  {d.upper()}:")
        print(f"    HIGH D-Def (n={len(h)}): success={h['success'].mean():.1%}")
        print(f"    LOW  D-Def (n={len(l)}): success={l['success'].mean():.1%}")

# ================================================================
# DIAGONAL DISRUPTION: PC1+PC2 vs FORWARD/SIDEWAYS
# ================================================================
print("\n\n" + "=" * 60)
print("MANN-WHITNEY: DIAGONAL PC1+PC2 vs OTHERS")
print("=" * 60)
if f"pc1_{ws}" in df.columns:
    diag = df[(df["direction"] == "diagonal") & df[f"pc1_{ws}"].notna()]
    fwd = df[(df["direction"] == "forward") & df[f"pc1_{ws}"].notna()]
    side = df[(df["direction"] == "sideways") & df[f"pc1_{ws}"].notna()]

    diag_both = diag[f"pc1_{ws}"].abs() + diag[f"pc2_{ws}"].abs()
    if len(fwd) > 5:
        fwd_both = fwd[f"pc1_{ws}"].abs() + fwd[f"pc2_{ws}"].abs()
        u, p = mannwhitneyu(diag_both, fwd_both, alternative="greater")
        print(f"  Diagonal vs Forward (|PC1|+|PC2|): U={u:.0f}, p={p:.4f} {'***' if p<0.01 else 'ns'}")
        print(f"    Diagonal: {diag_both.mean():.3f}")
        print(f"    Forward:  {fwd_both.mean():.3f}")

    if len(side) > 5:
        side_both = side[f"pc1_{ws}"].abs() + side[f"pc2_{ws}"].abs()
        u, p = mannwhitneyu(diag_both, side_both, alternative="greater")
        print(f"  Diagonal vs Sideways (|PC1|+|PC2|): U={u:.0f}, p={p:.4f} {'***' if p<0.01 else 'ns'}")
        print(f"    Diagonal: {diag_both.mean():.3f}")
        print(f"    Sideways: {side_both.mean():.3f}")

print("\n\nDone.")
