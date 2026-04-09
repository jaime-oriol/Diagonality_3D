"""Full pipeline validation: PPCF + Vision + DOS + Theta cross-analysis.

Validates every module against real event outcomes across all matches.
Tests the core thesis: diagonal actions exploit blind spots, and our
pipeline (orientation -> vision -> PPCF -> DOS) captures this mechanism.

Processes ALL events (passes + carries). Caches results per match to disk.
One match at a time in memory, freed before loading next.
"""
import sys; sys.path.insert(0, ".")
import gc
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from scipy.interpolate import RegularGridInterpolator

from src.preprocess import load_cached_events, load_cached_skeleton
from src.orientation import compute_orientations, add_dynamics
from src.loader import load_match_info, compute_attacking_right
from src.ppcf import ppcf_at_targets
from src.dos import dos_for_event
from src.vision import compute_player_vision

# Matches with working XML (Dortmund_Stuttgart excluded: no XML)
MATCHES = [
    "Bayern_Hamburg",
    "Frankfurt_Bayern",
    "Frankfurt_Union",
    "Union_Bayern",
]
CACHE_DIR = "cache/validation"
os.makedirs(CACHE_DIR, exist_ok=True)


def classify_direction(pa):
    if pd.isna(pa): return "unknown"
    a = abs(float(pa))
    if a <= 22.5: return "forward"
    elif a <= 67.5: return "diagonal"
    elif a <= 112.5: return "sideways"
    return "backward"


def process_match(match):
    """Process ALL events for one match. Cache to disk. Return DataFrame."""
    cache_path = f"{CACHE_DIR}/{match}_validation.parquet"
    if os.path.exists(cache_path):
        print(f"\n  {match}: loading from cache ({cache_path})")
        return pd.read_parquet(cache_path)

    print(f"\n{'='*60}")
    print(f"  MATCH: {match}")
    print(f"{'='*60}")

    events = load_cached_events(match)
    info = load_match_info(match)
    home_team_id = info.get("home_team_id", "")

    # home_gk_left per half from metadata (dict {"1": bool, "2": bool})
    import json
    meta = json.load(open(f"cache/{match}/metadata.json"))
    home_gk_left_map = meta.get("home_gk_left", {"1": True, "2": False})

    # PASSES ONLY — passing decision framework
    actions = events[(events["event_type"] == "pass") & (events["parquet_frame"] > 0)].copy()
    actions["direction"] = actions["play_angle"].apply(classify_direction)
    actions = actions[actions["direction"] != "unknown"]
    actions["success"] = (actions["evaluation"] == "successfullyCompleted").astype(int)
    print(f"  Total events: {len(actions)} ({actions['event_type'].value_counts().to_dict()})")
    print(f"  Directions: {actions['direction'].value_counts().to_dict()}")

    # Load skeleton
    print("  Loading skeleton...")
    skel = load_cached_skeleton(match)
    print(f"  Skeleton loaded ({len(skel)} rows)")

    results = []
    errors = 0
    for i, (_, row) in enumerate(actions.iterrows()):
        frame_id = int(row["parquet_frame"])
        if frame_id < 1:
            continue

        ss = skel[skel["frame_number"].between(frame_id - 2, frame_id + 2)]
        if len(ss) == 0:
            errors += 1
            continue

        ori = compute_orientations(ss, smooth=False)
        dyn = add_dynamics(ori)
        fo = dyn[dyn["frame_number"] == frame_id]
        if len(fo) < 4:
            errors += 1
            continue

        team_id = row["team_id"]
        attacking_team = 1 if team_id == home_team_id else 0
        defending_team = 1 - attacking_team
        half = int(row.get("half", 1))
        home_gk_left_p1 = home_gk_left_map.get(str(1), True)
        attacking_right = compute_attacking_right(attacking_team, half, home_gk_left_p1)

        ball_xy = (float(row["x"]), float(row["y"]))
        rx, ry = float(row["x_receiver"]), float(row["y_receiver"])
        if np.isnan(rx) or np.isnan(ry):
            errors += 1
            continue
        dest = np.array([[rx, ry]])

        event_dict = {
            "x": float(row["x"]), "y": float(row["y"]),
            "x_receiver": rx, "y_receiver": ry,
            "event_type": row["event_type"],
        }

        try:
            # --- DOS ---
            dos_result = dos_for_event(
                fo, event_dict, attacking_team, attacking_right,
                vision_smoothing=1.0,
            )

            # --- PPCF at destination (baseline, no detection delay) ---
            ppcf_att, ppcf_def = ppcf_at_targets(fo, dest, attacking_team)

            # --- Defender theta (shoulder angle vs pass direction) ---
            pass_dir = np.arctan2(ry - float(row["y"]), rx - float(row["x"]))
            defenders = fo[fo["team"] == defending_team]

            nearest_def_dist = 99.0
            nearest_def_theta = 0.0
            v_passer = 0.0
            v_receiver = 0.0
            mean_theta_all = 0.0
            n_blind = 0
            pct_blind = 0.0
            n_wrongfooted = 0
            pct_wrongfooted = 0.0

            if len(defenders) > 0:
                def_sa = defenders["shoulder_angle"].values
                theta_flight = np.abs(np.arctan2(
                    np.sin(pass_dir - def_sa), np.cos(pass_dir - def_sa)))

                # Nearest defender to destination
                def_pos = defenders[["x", "y"]].values
                dists_to_dest = np.sqrt((def_pos[:, 0] - rx)**2 + (def_pos[:, 1] - ry)**2)
                nearest_def_idx = np.argmin(dists_to_dest)
                nearest_def_dist = float(dists_to_dest[nearest_def_idx])
                nearest_def_theta = float(theta_flight[nearest_def_idx])

                mean_theta_all = float(theta_flight.mean())
                n_blind = int((theta_flight > np.pi / 2).sum())
                pct_blind = n_blind / len(defenders)
                n_wrongfooted = int((theta_flight > np.radians(60)).sum())
                pct_wrongfooted = n_wrongfooted / len(defenders)

                # Nearest defender vision of passer and receiver
                nd = defenders.iloc[nearest_def_idx]
                others = fo[~((fo["team"] == nd["team"]) & (fo["jersey"] == nd["jersey"]))]
                try:
                    vgrid = compute_player_vision(
                        float(nd["x"]), float(nd["y"]),
                        float(nd.get("head_angle", nd["shoulder_angle"])),
                        float(nd.get("speed", 0)),
                        others["x"].values.astype(float),
                        others["y"].values.astype(float),
                        others["shoulder_angle"].values.astype(float),
                        others["shoulder_width"].values.astype(float) if "shoulder_width" in others.columns else None,
                        smoothing=1.0,
                    )
                    gl, gw = int(105 * 1.0), int(68 * 1.0)
                    xc = np.linspace(-52.5, 52.5, gl)
                    yc = np.linspace(-34, 34, gw)
                    interp = RegularGridInterpolator(
                        (yc, xc), vgrid, method="linear",
                        bounds_error=False, fill_value=0.0)
                    v_passer = float(np.clip(interp([[float(row["y"]), float(row["x"])]])[0], 0, 1))
                    v_receiver = float(np.clip(interp([[ry, rx]])[0], 0, 1))
                except Exception:
                    pass

            # --- Compile ---
            results.append({
                "match": match,
                "event_type": row["event_type"],
                "direction": row["direction"],
                "success": int(row["success"]),
                "xp": float(row.get("xp", 0)) if not pd.isna(row.get("xp")) else 0.0,
                "play_angle": float(row.get("play_angle", 0)) if not pd.isna(row.get("play_angle")) else 0.0,
                "pass_length": float(np.sqrt((rx - float(row["x"]))**2 + (ry - float(row["y"]))**2)),
                "progression": float(rx - float(row["x"])) if attacking_right else float(float(row["x"]) - rx),
                "ball_x": float(row["x"]),
                "ball_y": float(row["y"]),
                # DOS
                "dos": float(dos_result["dos"]),
                "dos_direction_class": dos_result["direction_class"],
                "ppcf_att_baseline": float(dos_result["ppcf_att_baseline"]),
                "ppcf_att_aware": float(dos_result["ppcf_att_with_awareness"]),
                "awareness_mean": float(dos_result["awareness_mean"]),
                "detection_delay": float(dos_result["detection_delay_mean"]),
                # PPCF at destination
                "ppcf_att_dest": float(ppcf_att[0]),
                "ppcf_def_dest": float(ppcf_def[0]),
                # Defender orientation
                "nearest_def_dist": nearest_def_dist,
                "nearest_def_theta": nearest_def_theta,
                "mean_theta_all_def": mean_theta_all,
                "n_defenders_blind": n_blind,
                "pct_defenders_blind": pct_blind,
                "n_wrongfooted": n_wrongfooted,
                "pct_wrongfooted": pct_wrongfooted,
                # Vision
                "nearest_def_sees_passer": v_passer,
                "nearest_def_sees_receiver": v_receiver,
            })

        except Exception as e:
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(actions)} events done ({len(results)} ok, {errors} errors)")
            gc.collect()

    # Free skeleton ASAP
    del skel
    gc.collect()

    print(f"  DONE: {len(results)} events processed, {errors} errors")

    # Cache to disk
    match_df = pd.DataFrame(results)
    match_df.to_parquet(cache_path)
    print(f"  Cached to {cache_path}")

    return match_df


# ============================================================
#  MAIN: process all matches sequentially, cache each
# ============================================================
all_dfs = []
for match in MATCHES:
    try:
        match_df = process_match(match)
        all_dfs.append(match_df)
        print(f"  -> {len(match_df)} events from {match}")
    except Exception as e:
        print(f"  -> SKIP {match}: {e}")
    gc.collect()

df = pd.concat(all_dfs, ignore_index=True)
df.to_parquet(f"{CACHE_DIR}/all_validation.parquet")
print(f"\n\n{'#'*60}")
print(f"  TOTAL: {len(df)} events across {df['match'].nunique()} matches")
print(f"  Saved to {CACHE_DIR}/all_validation.parquet")
print(f"{'#'*60}")

# ================================================================
# ANALYSIS 1: Direction breakdown
# ================================================================
print("\n\n" + "=" * 60)
print("1. DIRECTION BREAKDOWN")
print("=" * 60)
for d in ["forward", "diagonal", "sideways", "backward"]:
    s = df[df["direction"] == d]
    if len(s) == 0: continue
    print(f"\n  {d.upper()} (n={len(s)}):")
    print(f"    Success rate:      {s['success'].mean():.1%}")
    print(f"    Mean xP:           {s['xp'].mean():.3f}")
    print(f"    Pass length:       {s['pass_length'].mean():.1f}m")
    print(f"    Progression:       {s['progression'].mean():.1f}m")
    print(f"    PPCF att at dest:  {s['ppcf_att_dest'].mean():.3f}")
    print(f"    PPCF def at dest:  {s['ppcf_def_dest'].mean():.3f}")
    print(f"    DOS:               {s['dos'].mean():.4f} (median={s['dos'].median():.4f})")
    print(f"    Awareness:         {s['awareness_mean'].mean():.3f}")
    print(f"    Detection delay:   {s['detection_delay'].mean():.4f}s")
    print(f"    Nearest def dist:  {s['nearest_def_dist'].mean():.1f}m")
    print(f"    Nearest def theta: {np.degrees(s['nearest_def_theta'].mean()):.1f} deg")
    print(f"    Mean theta all:    {np.degrees(s['mean_theta_all_def'].mean()):.1f} deg")
    print(f"    % def blind:       {s['pct_defenders_blind'].mean():.1%}")
    print(f"    % wrongfooted:     {s['pct_wrongfooted'].mean():.1%}")
    print(f"    Def sees passer:   {s['nearest_def_sees_passer'].mean():.3f}")
    print(f"    Def sees receiver: {s['nearest_def_sees_receiver'].mean():.3f}")

# ================================================================
# ANALYSIS 2: Theta by direction (Mann-Whitney tests)
# ================================================================
print("\n\n" + "=" * 60)
print("2. THETA BY DIRECTION")
print("=" * 60)
for metric, label in [("nearest_def_theta", "Nearest def theta"),
                       ("mean_theta_all_def", "Mean theta all defenders"),
                       ("pct_wrongfooted", "% wrongfooted")]:
    print(f"\n  {label}:")
    diag = df[df["direction"] == "diagonal"][metric].dropna()
    fwd = df[df["direction"] == "forward"][metric].dropna()
    side = df[df["direction"] == "sideways"][metric].dropna()
    for name, other in [("Forward", fwd), ("Sideways", side)]:
        if len(diag) > 5 and len(other) > 5:
            u, p = mannwhitneyu(diag, other, alternative="greater")
            print(f"    Diag vs {name}: U={u:.0f}, p={p:.4f} "
                  f"{'***' if p<0.01 else ('*' if p<0.05 else 'ns')} "
                  f"(diag={diag.mean():.3f}, {name.lower()}={other.mean():.3f})")

# ================================================================
# ANALYSIS 3: DOS by direction
# ================================================================
print("\n\n" + "=" * 60)
print("3. DOS BY DIRECTION")
print("=" * 60)
for d in ["forward", "diagonal", "sideways", "backward"]:
    s = df[df["direction"] == d]["dos"]
    if len(s) < 3: continue
    print(f"  {d:>10}: mean={s.mean():.4f}, median={s.median():.4f}, "
          f"std={s.std():.4f}, P25={s.quantile(0.25):.4f}, P75={s.quantile(0.75):.4f}")

# ================================================================
# ANALYSIS 4: High DOS -> higher success
# ================================================================
print("\n\n" + "=" * 60)
print("4. HIGH DOS -> HIGHER SUCCESS?")
print("=" * 60)
med = df["dos"].median()
hi = df[df["dos"] > med]
lo = df[df["dos"] <= med]
print(f"  Median DOS: {med:.4f}")
print(f"  HIGH DOS (n={len(hi)}): success={hi['success'].mean():.1%}")
print(f"  LOW  DOS (n={len(lo)}): success={lo['success'].mean():.1%}")
if len(hi) > 5 and len(lo) > 5:
    u, p = mannwhitneyu(hi["success"], lo["success"], alternative="greater")
    print(f"  Mann-Whitney U={u:.0f}, p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

# Per direction
for d in ["forward", "diagonal", "sideways", "backward"]:
    s = df[df["direction"] == d]
    if len(s) < 10: continue
    m = s["dos"].median()
    h, l = s[s["dos"] > m], s[s["dos"] <= m]
    if len(h) > 3 and len(l) > 3:
        print(f"\n  {d.upper()} (n={len(s)}):")
        print(f"    HIGH DOS (n={len(h)}): success={h['success'].mean():.1%}")
        print(f"    LOW  DOS (n={len(l)}): success={l['success'].mean():.1%}")
        u, p = mannwhitneyu(h["success"], l["success"], alternative="greater")
        print(f"    Mann-Whitney p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

# ================================================================
# ANALYSIS 5: Vision -> Success
# ================================================================
print("\n\n" + "=" * 60)
print("5. VISION PREDICTS OUTCOME?")
print("=" * 60)
succ = df[df["success"] == 1]
fail = df[df["success"] == 0]
if len(fail) > 5 and len(succ) > 5:
    for col, label in [("nearest_def_sees_passer", "Def sees passer"),
                        ("nearest_def_sees_receiver", "Def sees receiver"),
                        ("awareness_mean", "Awareness mean")]:
        s_mean = succ[col].mean()
        f_mean = fail[col].mean()
        u, p = mannwhitneyu(fail[col], succ[col], alternative="greater")
        print(f"  {label:>25}: success={s_mean:.3f}, fail={f_mean:.3f}, "
              f"MW p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

# ================================================================
# ANALYSIS 6: PPCF at destination -> Success
# ================================================================
print("\n\n" + "=" * 60)
print("6. PPCF AT DESTINATION -> SUCCESS?")
print("=" * 60)
if len(succ) > 5 and len(fail) > 5:
    print(f"  Successful (n={len(succ)}): PPCF_att={succ['ppcf_att_dest'].mean():.3f}, PPCF_def={succ['ppcf_def_dest'].mean():.3f}")
    print(f"  Failed     (n={len(fail)}): PPCF_att={fail['ppcf_att_dest'].mean():.3f}, PPCF_def={fail['ppcf_def_dest'].mean():.3f}")
    r, p = spearmanr(df["ppcf_att_dest"], df["success"])
    print(f"  Spearman(PPCF_att, success): r={r:.3f}, p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")
    r, p = spearmanr(df["ppcf_def_dest"], df["success"])
    print(f"  Spearman(PPCF_def, success): r={r:.3f}, p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

# ================================================================
# ANALYSIS 7: Full correlation matrix
# ================================================================
print("\n\n" + "=" * 60)
print("7. KEY CORRELATIONS (Spearman)")
print("=" * 60)
pairs = [
    ("dos", "success", "DOS -> Success"),
    ("ppcf_att_dest", "success", "PPCF att dest -> Success"),
    ("ppcf_att_aware", "success", "PPCF att aware -> Success"),
    ("nearest_def_theta", "success", "Nearest def theta -> Success"),
    ("mean_theta_all_def", "success", "Mean theta all -> Success"),
    ("pct_defenders_blind", "success", "% blind -> Success"),
    ("pct_wrongfooted", "success", "% wrongfooted -> Success"),
    ("awareness_mean", "success", "Awareness -> Success (expect -)"),
    ("detection_delay", "success", "Detection delay -> Success"),
    ("nearest_def_sees_passer", "success", "Def sees passer -> Success (expect -)"),
    ("nearest_def_sees_receiver", "success", "Def sees receiver -> Success (expect -)"),
    ("nearest_def_dist", "success", "Nearest def dist -> Success"),
    ("xp", "success", "xP -> Success"),
    ("detection_delay", "dos", "Detection delay -> DOS"),
    ("nearest_def_theta", "dos", "Nearest def theta -> DOS"),
    ("pct_wrongfooted", "dos", "% wrongfooted -> DOS"),
    ("progression", "dos", "Progression -> DOS"),
    ("awareness_mean", "dos", "Awareness -> DOS (expect -)"),
]
for x, y, label in pairs:
    valid = df[[x, y]].dropna()
    if len(valid) > 10:
        r, p = spearmanr(valid[x], valid[y])
        sig = "***" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  {label:>45}: r={r:+.3f}, p={p:.4f} {sig}")

# ================================================================
# ANALYSIS 8: The money stat
# ================================================================
print("\n\n" + "=" * 60)
print("8. THE MONEY STAT: DIAGONAL + BLIND SPOT + HIGH DOS")
print("=" * 60)
diag_df = df[df["direction"] == "diagonal"]
if len(diag_df) > 10:
    # Split by defender blindness
    blind_med = diag_df["pct_defenders_blind"].median()
    blind_high = diag_df[diag_df["pct_defenders_blind"] > blind_med]
    blind_low = diag_df[diag_df["pct_defenders_blind"] <= blind_med]
    print(f"  Diagonal + HIGH blind % (>{blind_med:.1%}): success={blind_high['success'].mean():.1%} (n={len(blind_high)})")
    print(f"  Diagonal + LOW  blind %:                  success={blind_low['success'].mean():.1%} (n={len(blind_low)})")
    if len(blind_high) > 5 and len(blind_low) > 5:
        u, p = mannwhitneyu(blind_high["success"], blind_low["success"], alternative="greater")
        print(f"  Mann-Whitney p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

    # Triple filter
    dos_med = diag_df["dos"].median()
    dos_high = diag_df["dos"] > dos_med
    blind_high_mask = diag_df["pct_defenders_blind"] > blind_med
    triple = diag_df[dos_high & blind_high_mask]
    rest = diag_df[~(dos_high & blind_high_mask)]
    print(f"\n  TRIPLE (diagonal + high DOS + high blind):")
    print(f"    Success: {triple['success'].mean():.1%} (n={len(triple)})")
    print(f"    Rest:    {rest['success'].mean():.1%} (n={len(rest)})")
    if len(triple) > 0 and len(rest) > 0:
        print(f"    PPCF att: {triple['ppcf_att_dest'].mean():.3f} vs {rest['ppcf_att_dest'].mean():.3f}")
        print(f"    Awareness: {triple['awareness_mean'].mean():.3f} vs {rest['awareness_mean'].mean():.3f}")
        print(f"    Det. delay: {triple['detection_delay'].mean():.4f}s vs {rest['detection_delay'].mean():.4f}s")
    if len(triple) > 5 and len(rest) > 5:
        u, p = mannwhitneyu(triple["success"], rest["success"], alternative="greater")
        print(f"    Mann-Whitney p={p:.4f} {'***' if p<0.01 else ('*' if p<0.05 else 'ns')}")

# ================================================================
# ANALYSIS 9: Per-match consistency
# ================================================================
print("\n\n" + "=" * 60)
print("9. PER-MATCH CONSISTENCY")
print("=" * 60)
for match in df["match"].unique():
    m = df[df["match"] == match]
    diag_m = m[m["direction"] == "diagonal"]
    fwd_m = m[m["direction"] == "forward"]
    print(f"\n  {match} (n={len(m)}):")
    print(f"    Overall success: {m['success'].mean():.1%}")
    if len(diag_m) > 0:
        print(f"    Diagonal: success={diag_m['success'].mean():.1%}, "
              f"DOS={diag_m['dos'].mean():.4f}, n={len(diag_m)}")
    if len(fwd_m) > 0:
        print(f"    Forward:  success={fwd_m['success'].mean():.1%}, "
              f"DOS={fwd_m['dos'].mean():.4f}, n={len(fwd_m)}")

print("\n\nDone. Results cached in cache/validation/")
