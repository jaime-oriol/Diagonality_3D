"""Diagnostic script for the DOS scanning gate.

Runs the full pipeline on a SINGLE frame near the Kane goal and prints
exhaustive stats at every stage so we can localize where the gate
breaks down. Run with:

    python3 test/debug_dos_gate.py
"""
import sys; sys.path.insert(0, ".")
import numpy as np

from src.loader import load_match_info
from src.orientation import compute_orientations, add_dynamics
from src.preprocess import (
    load_cached_skeleton, load_cached_ball,
    load_cached_events, load_cached_metadata,
)
from src.dos import compute_dos_surface, default_params
from src.possession import build_possession_timeline, get_owner_at_frame
from src.scanning import (
    ScanningMemoryConfig, compute_scanning_memory_sequence,
    backfill_orientations, resample_memory_to_grid,
)
from src.ppcf import PITCH_LENGTH, PITCH_WIDTH

MATCH = "Bayern_Hamburg"
GOAL_F = 3585218
FPS = 50
# Probe several frames around the goal to see if any of them work
PROBE_FRAMES = [GOAL_F - 500, GOAL_F - 250, GOAL_F - 50, GOAL_F, GOAL_F + 50]
WINDOW_PRE = 20
WINDOW_POST = 15
start_f = GOAL_F - WINDOW_PRE * FPS
end_f = GOAL_F + WINDOW_POST * FPS

print(f"=== render window: [{start_f}, {end_f}] ({(end_f-start_f)/FPS:.0f}s) ===\n")

# --- Metadata + match info ---
print("[1] Loading metadata + events...")
info = load_match_info(MATCH)
metadata = load_cached_metadata(MATCH)
events = load_cached_events(MATCH)
print(f"  events total: {len(events)}")
print(f"  modes: {dict(events['event_type'].value_counts())}")
in_window = events[
    (events["parquet_frame"] >= start_f) & (events["parquet_frame"] <= end_f)
]
print(f"  events in render window: {len(in_window)}")
print(f"  in-window modes: {dict(in_window['event_type'].value_counts())}")

# --- Possession timeline ---
print("\n[2] Building possession timeline...")
tl = build_possession_timeline(events, info, gap_fill_max_frames=300)
in_w_pre = tl[(tl["frame_end"] >= start_f) & (tl["frame_start"] <= end_f)].sort_values("frame_start")
if len(in_w_pre):
    fi, li = in_w_pre.index[0], in_w_pre.index[-1]
    if tl.loc[fi, "frame_start"] > start_f:
        tl.loc[fi, "frame_start"] = start_f
    if tl.loc[li, "frame_end"] < end_f:
        tl.loc[li, "frame_end"] = end_f
print(f"  total segments: {len(tl)}")
in_w = tl[(tl["frame_end"] >= start_f) & (tl["frame_start"] <= end_f)]
print(f"  segments overlapping render window: {len(in_w)}")
if len(in_w):
    print(f"  modes in window: {dict(in_w['mode'].value_counts())}")
    print("  first 5 segments in window:")
    for _, seg in in_w.head(5).iterrows():
        print(f"    [{seg['frame_start']}, {seg['frame_end']}] team={seg['team']} "
              f"jersey={seg['jersey']} mode={seg['mode']} ({seg['frame_end']-seg['frame_start']+1} frames)")

# Owner lookup at probe frames
print("\n[3] Owner at probe frames:")
for pf in PROBE_FRAMES:
    o = get_owner_at_frame(tl, pf)
    if o:
        print(f"  frame {pf}: team={o['team']} jersey={o['jersey']} mode={o['mode']}")
    else:
        print(f"  frame {pf}: NO OWNER")

# --- Skeleton + dyn ---
print("\n[4] Loading skeleton + computing dynamics...")
skel = load_cached_skeleton(MATCH)
ss = skel[skel["frame_number"].between(start_f - 5, end_f + 5)]
del skel
ori = compute_orientations(ss, smooth=True)
dyn = add_dynamics(ori)
del ss, ori

# Check a few frames
for pf in PROBE_FRAMES:
    fd = dyn[dyn["frame_number"] == pf]
    print(f"  frame {pf}: {len(fd)} player rows")

# --- Backfill ---
print("\n[5] Applying backfill (lookback=150)...")
dyn_filled = backfill_orientations(dyn, lookback_frames=150)
print(f"  before backfill: {len(dyn)} rows, after: {len(dyn_filled)} rows")
print(f"  unique players before: {dyn[['team','jersey']].drop_duplicates().shape[0]}")
print(f"  unique players after:  {dyn_filled[['team','jersey']].drop_duplicates().shape[0]}")
print(f"  earliest frame before: {dyn['frame_number'].min()}")
print(f"  earliest frame after:  {dyn_filled['frame_number'].min()}")

# --- Scanning memory ---
print("\n[6] Computing scanning memory for probe frames only...")
SCAN_CONFIG = ScanningMemoryConfig(
    memory_window_s=2.5, tau_decay_s=1.2,
    vision_smoothing=2.0, framerate=FPS,
)
# Compute only for the probe range to be fast
probe_min = min(PROBE_FRAMES) - 5
probe_max = max(PROBE_FRAMES) + 5
memories = compute_scanning_memory_sequence(
    dyn_filled, tl, (probe_min, probe_max), SCAN_CONFIG)
print(f"  frames in result dict: {len(memories)}")

for pf in PROBE_FRAMES:
    fm = memories.get(pf)
    if fm is None:
        print(f"  frame {pf}: NO MEMORY")
        continue
    print(f"  frame {pf}: owner={fm.owner_jersey} mode={fm.mode}")
    print(f"    memory.shape={fm.memory.shape}, max={fm.memory.max():.4f}, "
          f"sum>0.1: {(fm.memory > 0.1).sum()} cells")
    print(f"    fov_now.shape={fm.fov_now.shape}, max={fm.fov_now.max():.4f}, "
          f"sum>0.1: {(fm.fov_now > 0.1).sum()} cells")

# --- DOS ---
print("\n[7] Computing DOS surfaces at probe frames...")
attacking_team = 1  # Bayern home in Bayern_Hamburg
attacking_right = True
params = default_params()

ball = load_cached_ball(MATCH)
N_GRID = 50
N_GRID_Y = int(round(N_GRID * PITCH_WIDTH / PITCH_LENGTH))
DX = PITCH_LENGTH / N_GRID
DY = PITCH_WIDTH / N_GRID_Y
XGRID = np.arange(N_GRID) * DX - PITCH_LENGTH / 2 + DX / 2
YGRID = np.arange(N_GRID_Y) * DY - PITCH_WIDTH / 2 + DY / 2

for pf in PROBE_FRAMES:
    fd = dyn_filled[dyn_filled["frame_number"] == pf]
    if len(fd) == 0:
        print(f"  frame {pf}: no dyn data, skip")
        continue
    bf = ball[ball["frame_number"] == pf]
    bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else 0.0
    by = float(bf.iloc[0]["y"]) if len(bf) > 0 else 0.0
    dos_surf, _, _, _, _ = compute_dos_surface(
        fd, attacking_team, (bx, by), attacking_right,
        params=params, n_grid_x=N_GRID, n_directions=24, vision_smoothing=1.0,
    )
    print(f"  frame {pf}: dos.shape={dos_surf.shape}, "
          f"min={dos_surf.min():.5f}, max={dos_surf.max():.5f}, "
          f"mean+={np.clip(dos_surf,0,None).mean():.5f}, "
          f"cells>0.003: {(dos_surf > 0.003).sum()}")

# --- Gate ---
print("\n[8] Applying gate (memory * dos)...")
for pf in PROBE_FRAMES:
    fd = dyn_filled[dyn_filled["frame_number"] == pf]
    if len(fd) == 0:
        continue
    bf = ball[ball["frame_number"] == pf]
    bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else 0.0
    by = float(bf.iloc[0]["y"]) if len(bf) > 0 else 0.0
    dos_surf, _, _, _, _ = compute_dos_surface(
        fd, attacking_team, (bx, by), attacking_right,
        params=params, n_grid_x=N_GRID, n_directions=24, vision_smoothing=1.0,
    )
    dos_pos = np.clip(dos_surf, 0.0, None)

    fm = memories.get(pf)
    if fm is None:
        print(f"  frame {pf}: NO MEMORY -> gate=zeros, painted cells=0")
        continue
    memory_on_dos = resample_memory_to_grid(fm.memory, XGRID, YGRID)
    print(f"  frame {pf}: resampled memory shape={memory_on_dos.shape}, "
          f"max={memory_on_dos.max():.4f}, "
          f"cells>0.1: {(memory_on_dos > 0.1).sum()}")

    gated = dos_pos * memory_on_dos
    print(f"    gated dos: max={gated.max():.5f}, "
          f"cells>0.003: {(gated > 0.003).sum()}/{gated.size}")
    print(f"    cells > 0.001: {(gated > 0.001).sum()}, "
          f"cells > 0.0005: {(gated > 0.0005).sum()}")
    print(f"    cells > 0.0001: {(gated > 0.0001).sum()}")
