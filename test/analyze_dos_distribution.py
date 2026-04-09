"""Analyze DOS value distribution across all matches.

Processes ONE match at a time, prints partial results, frees memory.
Goal: find an informed threshold to separate signal from noise in rendering.
"""
import sys; sys.path.insert(0, ".")
import gc
import numpy as np
import pandas as pd
from src.preprocess import load_cached_skeleton, load_cached_events
from src.orientation import compute_orientations, add_dynamics
from src.dos import compute_dos_surface
from src.loader import load_match_info, compute_attacking_right

MATCHES = [
    "Bayern_Hamburg",
    "Dortmund_Stuttgart",
    "Frankfurt_Bayern",
    "Frankfurt_Union",
    "Union_Bayern",
]
EVENTS_PER_DIR = 5   # 5 per direction x 4 = 20 per match, 100 total
N_GRID = 50
N_DIRS = 8

# Lightweight collectors (just numbers, no big arrays)
all_surface_stats = []
all_cell_values = []
all_spatial = []


def classify_play_angle(row):
    pa = row.get("play_angle")
    if pd.isna(pa):
        dx = row.get("x_end", row.get("x_receiver", row["x"])) - row["x"]
        dy = row.get("y_end", row.get("y_receiver", row["y"])) - row["y"]
        pa = np.degrees(np.arctan2(dy, dx))
    a = abs(float(pa))
    if a <= 22.5: return "forward"
    elif a <= 67.5: return "diagonal"
    elif a <= 112.5: return "sideways"
    return "backward"


def print_match_summary(match_name, stats_list):
    """Print partial stats for this match."""
    if not stats_list:
        print(f"  No events processed for {match_name}")
        return
    df = pd.DataFrame(stats_list)
    print(f"\n  --- {match_name} summary ({len(df)} events) ---")
    print(f"  Max DOS:  mean={df['dos_max'].mean():.4f}, "
          f"median={df['dos_max'].median():.4f}, "
          f"max={df['dos_max'].max():.4f}")
    print(f"  Positive cells: {df['pct_positive'].mean():.1%} of grid on average")
    for d in ["forward", "diagonal", "sideways", "backward"]:
        sub = df[df["direction"] == d]
        if len(sub) == 0: continue
        print(f"    {d:>10}: n={len(sub)}, dos_max_mean={sub['dos_max'].mean():.4f}, "
              f"success={sub['success'].mean():.0%}")


# ============================================================
#  MAIN LOOP: one match at a time
# ============================================================
for match_idx, match_name in enumerate(MATCHES):
    print(f"\n{'='*60}")
    print(f"  [{match_idx+1}/{len(MATCHES)}] Loading {match_name}...")
    print(f"{'='*60}")

    try:
        events = load_cached_events(match_name)
        info = load_match_info(match_name)
        home_team_id = info.get("home_team_id", "")
    except Exception as e:
        print(f"  ERROR loading events/info: {e}")
        continue

    # Filter & classify
    actions = events[events["event_type"].isin(["pass", "carry"])].copy()
    actions = actions[actions["parquet_frame"] > 0].copy()
    actions["success"] = (actions["evaluation"] == "successfullyCompleted").astype(int)
    actions["direction"] = actions.apply(classify_play_angle, axis=1)

    # Sample
    sample = actions.groupby("direction").apply(
        lambda g: g.sample(min(EVENTS_PER_DIR, len(g)), random_state=42)
    ).reset_index(drop=True)
    print(f"  {len(sample)} events sampled")

    # Load skeleton (the heavy part)
    print(f"  Loading skeleton...")
    skel = load_cached_skeleton(match_name)
    print(f"  Skeleton loaded. Processing events...")

    match_stats = []
    for i, (_, row) in enumerate(sample.iterrows()):
        frame_id = int(row["parquet_frame"])
        if frame_id < 1:
            continue
        ss = skel[skel["frame_number"].between(frame_id - 2, frame_id + 2)]
        if len(ss) == 0:
            print(f"    [skip] frame {frame_id}: no skeleton data")
            continue

        ori = compute_orientations(ss, smooth=False)
        dyn = add_dynamics(ori)
        fo = dyn[dyn["frame_number"] == frame_id]
        if len(fo) < 4:
            print(f"    [skip] frame {frame_id}: only {len(fo)} players")
            continue

        team_id = row["team_id"]
        attacking_team = 1 if team_id == home_team_id else 0
        half = int(row.get("half", 1))
        attacking_right = compute_attacking_right(attacking_team, half, True)
        ball_xy = np.array([float(row["x"]), float(row["y"])])

        att_mask = fo["team"].values == attacking_team
        att_pos = fo[att_mask][["x", "y"]].values

        try:
            dos_surface, _, _, grid_x, grid_y = compute_dos_surface(
                fo, attacking_team, ball_xy, attacking_right,
                n_grid_x=N_GRID, n_directions=N_DIRS,
                vision_smoothing=1.0,
            )
        except Exception as e:
            print(f"    [skip] frame {frame_id}: DOS error: {e}")
            continue

        # Surface stats
        dos_pos = dos_surface[dos_surface > 0]
        dos_all = dos_surface.ravel()

        stats = {
            "match": match_name,
            "direction": row["direction"],
            "success": int(row["success"]),
            "event_type": row["event_type"],
            "ball_x": ball_xy[0], "ball_y": ball_xy[1],
            "dos_max": float(dos_surface.max()),
            "dos_mean_pos": float(dos_pos.mean()) if len(dos_pos) > 0 else 0,
            "dos_median_pos": float(np.median(dos_pos)) if len(dos_pos) > 0 else 0,
            "dos_p90": float(np.percentile(dos_pos, 90)) if len(dos_pos) > 0 else 0,
            "dos_p95": float(np.percentile(dos_pos, 95)) if len(dos_pos) > 0 else 0,
            "n_positive_cells": int(len(dos_pos)),
            "n_total_cells": int(dos_all.size),
            "pct_positive": float(len(dos_pos) / dos_all.size),
        }
        match_stats.append(stats)
        all_surface_stats.append(stats)

        # Spatial: high-DOS cells (top 30% of frame)
        gx, gy = np.meshgrid(grid_x, grid_y)
        thr = dos_surface.max() * 0.3
        if thr > 0:
            hm = dos_surface >= thr
            hx, hy = gx[hm], gy[hm]
            dist_ball = np.sqrt((hx - ball_xy[0])**2 + (hy - ball_xy[1])**2)
            if len(att_pos) > 0:
                dist_att = np.min(
                    np.sqrt((hx[:, None] - att_pos[:, 0])**2 +
                            (hy[:, None] - att_pos[:, 1])**2), axis=1)
            else:
                dist_att = np.full(len(hx), 99.0)
            fwd = (hx - ball_xy[0]) if attacking_right else (ball_xy[0] - hx)
            for db, da, f in zip(dist_ball, dist_att, fwd):
                all_spatial.append((float(db), float(da), float(f)))

        # Subsample cell values (50 per frame max)
        if len(dos_pos) > 0:
            idx = np.random.choice(len(dos_pos), min(50, len(dos_pos)), replace=False)
            all_cell_values.extend(dos_pos[idx].tolist())

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(sample)} events done")

    # Print partial results for this match
    print_match_summary(match_name, match_stats)

    # FREE MEMORY before next match
    del skel, sample, actions, events, match_stats
    gc.collect()
    print(f"  Memory freed.\n")


# ============================================================
#  GLOBAL REPORT
# ============================================================
print("\n" + "=" * 60)
print("  OVERALL DOS DISTRIBUTION (all matches)")
print("=" * 60)

df = pd.DataFrame(all_surface_stats)
print(f"\nTotal events: {len(df)} across {df['match'].nunique()} matches")

print(f"\n--- Frame max DOS distribution ---")
for label, q in [("P10", 0.10), ("P25", 0.25), ("Median", 0.50),
                  ("P75", 0.75), ("P90", 0.90), ("P95", 0.95)]:
    print(f"  {label:>6}: {df['dos_max'].quantile(q):.4f}")
print(f"  {'Max':>6}: {df['dos_max'].max():.4f}")

print(f"\n--- Positive cell values (pooled from all frames) ---")
vals = np.array(all_cell_values)
print(f"  N samples: {len(vals)}")
for label, q in [("P25", 25), ("Median", 50), ("P75", 75),
                  ("P90", 90), ("P95", 95), ("P99", 99)]:
    print(f"  {label:>6}: {np.percentile(vals, q):.5f}")
print(f"  {'Max':>6}: {vals.max():.5f}")

print(f"\n--- By direction ---")
for d in ["forward", "diagonal", "sideways", "backward"]:
    sub = df[df["direction"] == d]
    if len(sub) == 0: continue
    print(f"  {d.upper():>10} (n={len(sub):>2}): "
          f"dos_max_mean={sub['dos_max'].mean():.4f}, "
          f"pct_pos={sub['pct_positive'].mean():.1%}, "
          f"success={sub['success'].mean():.0%}")

print(f"\n--- Spatial: where are high-DOS cells? ---")
sp = np.array(all_spatial)  # columns: dist_ball, dist_att, fwd_of_ball
if len(sp) > 0:
    print(f"  N high-DOS cells: {len(sp)}")
    print(f"  Dist to ball:    mean={sp[:,0].mean():.1f}m, median={np.median(sp[:,0]):.1f}m")
    print(f"  Dist to att:     mean={sp[:,1].mean():.1f}m, median={np.median(sp[:,1]):.1f}m")
    print(f"  Fwd of ball:     mean={sp[:,2].mean():.1f}m, median={np.median(sp[:,2]):.1f}m")
    print(f"  % behind ball:   {(sp[:,2] < 0).mean():.1%}")
    print(f"  % far from att (>15m): {(sp[:,1] > 15).mean():.1%}")
    print(f"  % far from att (>10m): {(sp[:,1] > 10).mean():.1%}")

print("\nDone.")
