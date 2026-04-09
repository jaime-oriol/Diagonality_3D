"""Validate the diagonality thesis with DOS numbers.

For each pass/carry, compute:
  1. Direction classification (forward/diagonal/sideways/backward)
  2. DOS at destination (diagonal gain vs forward)
  3. Cross with success, xP, progression

Then show: do diagonals with high DOS succeed more?
Do they disrupt both defensive axes?
"""
import sys; sys.path.insert(0, ".")
import gc
import numpy as np
import pandas as pd
from src.preprocess import load_cached_skeleton, load_cached_events
from src.orientation import compute_orientations, add_dynamics
from src.dos import dos_for_event, classify_attack_angle
from src.loader import load_match_info, infer_attacking_team, compute_attacking_right

MATCH = "Bayern_Hamburg"

# --- Load events ---
events = load_cached_events(MATCH)
passes = events[events["event_type"] == "pass"].copy()
carries = events[events["event_type"] == "carry"].copy()
all_actions = pd.concat([passes, carries], ignore_index=True)
all_actions = all_actions[all_actions["parquet_frame"] > 0].copy()
print(f"Total actions: {len(all_actions)} ({len(passes)} passes, {len(carries)} carries)")

# --- Classify directions from play_angle ---
def _classify_play_angle(row):
    pa = row.get("play_angle")
    if pd.isna(pa):
        # For carries, compute from vector
        dx = row.get("x_end", row.get("x_receiver", row["x"])) - row["x"]
        dy = row.get("y_end", row.get("y_receiver", row["y"])) - row["y"]
        pa = np.degrees(np.arctan2(dy, dx))
    a = abs(float(pa))
    if a <= 22.5:
        return "forward"
    elif a <= 67.5:
        return "diagonal"
    elif a <= 112.5:
        return "sideways"
    return "backward"

all_actions["direction"] = all_actions.apply(_classify_play_angle, axis=1)
print("\nDirection distribution:")
print(all_actions["direction"].value_counts())

# --- Success rate by direction (baseline, no DOS needed) ---
all_actions["success"] = (all_actions["evaluation"] == "successfullyCompleted").astype(int)
print("\n=== SUCCESS RATE BY DIRECTION (no DOS, just events) ===")
for d in ["forward", "diagonal", "sideways", "backward"]:
    subset = all_actions[all_actions["direction"] == d]
    if len(subset) == 0:
        continue
    rate = subset["success"].mean()
    xp_mean = subset["xp"].mean() if "xp" in subset.columns else 0
    prog = 0
    for _, r in subset.iterrows():
        rx = r.get("x_receiver", r.get("x_end", r["x"]))
        if not pd.isna(rx):
            prog += abs(rx - r["x"])
    prog /= max(len(subset), 1)
    print(f"  {d:>10}: n={len(subset):>4}, success={rate:.1%}, "
          f"mean_xP={xp_mean:.3f}, mean_progression={prog:.1f}m")

# --- Compute DOS per event (sample for speed) ---
print("\n=== COMPUTING DOS PER EVENT ===")
print("Loading skeleton (this takes a moment)...")
skel = load_cached_skeleton(MATCH)

# Get match info for attacking team
info = load_match_info(MATCH)
home_team_id = info.get("home_team_id", "")

# Process a sample of events (100 from each direction for speed)
sample = all_actions.groupby("direction").apply(
    lambda g: g.sample(min(60, len(g)), random_state=42)
).reset_index(drop=True)
print(f"Processing {len(sample)} events (sampled)...")

results = []
for i, (_, row) in enumerate(sample.iterrows()):
    frame_id = int(row["parquet_frame"])

    # Extract skeleton for this frame
    ss = skel[skel["frame_number"].between(frame_id - 2, frame_id + 2)]
    if len(ss) == 0:
        continue

    ori = compute_orientations(ss, smooth=False)
    dyn = add_dynamics(ori)
    fo = dyn[dyn["frame_number"] == frame_id]
    if len(fo) < 4:
        continue

    # Determine attacking team
    team_id = row["team_id"]
    attacking_team = 1 if team_id == home_team_id else 0
    half = int(row.get("half", 1))

    # Determine attacking direction
    home_gk_left = True  # default, could be refined
    attacking_right = compute_attacking_right(attacking_team, half, home_gk_left)

    ball_xy = (float(row["x"]), float(row["y"]))

    event_dict = {
        "x": float(row["x"]),
        "y": float(row["y"]),
        "x_receiver": float(row.get("x_receiver", row.get("x_end", row["x"]))),
        "y_receiver": float(row.get("y_receiver", row.get("y_end", row["y"]))),
        "event_type": row["event_type"],
    }

    try:
        dos_result = dos_for_event(
            fo, event_dict, attacking_team, attacking_right,
            vision_smoothing=1.0,  # lower smoothing for speed
        )
        dos_result["direction"] = row["direction"]
        dos_result["success"] = int(row["success"])
        dos_result["xp"] = float(row.get("xp", 0))
        dos_result["event_type"] = row["event_type"]
        results.append(dos_result)
    except Exception as e:
        pass

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(sample)} done")
        gc.collect()

del skel
gc.collect()

df = pd.DataFrame(results)
print(f"\n{len(df)} events computed successfully")

# --- Analysis ---
print("\n" + "=" * 60)
print("=== DOS BY DIRECTION ===")
print("=" * 60)
for d in ["forward", "diagonal", "sideways", "backward"]:
    subset = df[df["direction"] == d]
    if len(subset) == 0:
        continue
    print(f"\n  {d.upper()} (n={len(subset)}):")
    print(f"    DOS mean:        {subset['dos'].mean():.4f}")
    print(f"    DOS median:      {subset['dos'].median():.4f}")
    print(f"    Awareness mean:  {subset['awareness_mean'].mean():.3f}")
    print(f"    Det. delay mean: {subset['detection_delay_mean'].mean():.4f}s")
    print(f"    Success rate:    {subset['success'].mean():.1%}")

print("\n" + "=" * 60)
print("=== SUCCESS RATE: HIGH DOS vs LOW DOS ===")
print("=" * 60)
if len(df) > 10:
    median_dos = df["dos"].median()
    high = df[df["dos"] > median_dos]
    low = df[df["dos"] <= median_dos]
    print(f"  DOS threshold (median): {median_dos:.4f}")
    print(f"  HIGH DOS (n={len(high)}): success={high['success'].mean():.1%}")
    print(f"  LOW  DOS (n={len(low)}):  success={low['success'].mean():.1%}")

    # By direction
    for d in ["forward", "diagonal"]:
        sub = df[df["direction"] == d]
        if len(sub) < 5:
            continue
        med = sub["dos"].median()
        h = sub[sub["dos"] > med]
        l = sub[sub["dos"] <= med]
        print(f"\n  {d.upper()}:")
        print(f"    HIGH DOS (n={len(h)}): success={h['success'].mean():.1%}")
        print(f"    LOW  DOS (n={len(l)}):  success={l['success'].mean():.1%}")

print("\n" + "=" * 60)
print("=== PPCF BASELINE vs WITH AWARENESS ===")
print("=" * 60)
for d in ["forward", "diagonal", "sideways"]:
    subset = df[df["direction"] == d]
    if len(subset) == 0:
        continue
    base = subset["ppcf_att_baseline"].mean()
    aware = subset["ppcf_att_with_awareness"].mean()
    print(f"  {d:>10}: baseline={base:.3f}, with_awareness={aware:.3f}, "
          f"gain={aware - base:.4f}")

print("\nDone.")
