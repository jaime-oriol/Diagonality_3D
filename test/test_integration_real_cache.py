"""
Integration tests against the REAL cached data for Bayern_Hamburg.

Synthetic-only fixtures cannot catch mismatches between data sources
(e.g., metadata.json TRACAB integer player IDs vs kpi_data DFL-OBJ-*
string PersonIds). These tests load real cache files and assert that
the full possession + scanning + DOS gate pipeline produces non-empty
output.

Skipped if the cache directory is missing.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pytest

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "Bayern_Hamburg"

if not CACHE_DIR.exists():
    pytest.skip(
        f"Real cache not found at {CACHE_DIR} — skipping integration tests",
        allow_module_level=True,
    )

from src.loader import load_match_info
from src.preprocess import (
    load_cached_events, load_cached_metadata,
    load_cached_skeleton, load_cached_ball,
)
from src.orientation import compute_orientations, add_dynamics
from src.possession import (
    build_possession_timeline, get_owner_at_frame,
    _build_player_lookup,
)
from src.scanning import (
    ScanningMemoryConfig, compute_scanning_memory_sequence,
    backfill_orientations, resample_memory_to_grid,
)
from src.dos import compute_dos_surface, default_params
from src.ppcf import PITCH_LENGTH, PITCH_WIDTH


MATCH = "Bayern_Hamburg"
GOAL_F = 3585218  # Kane goal 5 shot frame


# Module-level cache so we don't reload for every test
_loaded = {}


def _load():
    if not _loaded:
        _loaded["info"] = load_match_info(MATCH)
        _loaded["events"] = load_cached_events(MATCH)
        _loaded["metadata"] = load_cached_metadata(MATCH)
    return _loaded


# ── Roster lookup ──────────────────────────────────────────────────────

def test_player_lookup_resolves_real_event_ids():
    """The DFL-OBJ-* player_ids from kpi_data must resolve via match_info."""
    d = _load()
    lookup = _build_player_lookup(d["info"])
    assert len(lookup) > 0, "match_info produced empty lookup"

    events = d["events"]
    sample_pids = events["player_id"].dropna().unique()[:50]
    resolved = sum(1 for pid in sample_pids if pid in lookup)
    assert resolved == len(sample_pids), (
        f"Only {resolved}/{len(sample_pids)} event player_ids resolved. "
        f"This means the lookup is built from the wrong source."
    )


# ── Possession timeline ────────────────────────────────────────────────

def test_timeline_non_empty_for_full_match():
    d = _load()
    tl = build_possession_timeline(d["events"], d["info"])
    # Match has ~1138 passes + 338 carries -> expect 1000+ segments
    assert len(tl) > 500, f"Expected >500 segments, got {len(tl)}"


def test_timeline_covers_kane_goal_window():
    d = _load()
    tl = build_possession_timeline(d["events"], d["info"])
    start_f = GOAL_F - 20 * 50
    end_f = GOAL_F + 15 * 50
    in_window = tl[
        (tl["frame_end"] >= start_f) & (tl["frame_start"] <= end_f)
    ]
    assert len(in_window) >= 5, (
        f"Expected at least 5 segments in 35s Kane goal window, got {len(in_window)}"
    )
    # Modes should include carries and pass_flights
    modes = set(in_window["mode"].unique())
    assert "carry" in modes
    assert "pass_flight" in modes


def test_kane_is_on_ball_near_goal():
    d = _load()
    tl = build_possession_timeline(d["events"], d["info"])
    # Kane (jersey 9, team 1) should be on-ball within a few seconds of the shot
    found_kane = False
    for delta in range(-200, 1):
        owner = get_owner_at_frame(tl, GOAL_F + delta)
        if owner and owner["team"] == 1 and owner["jersey"] == 9:
            found_kane = True
            break
    assert found_kane, "Kane (team=1, jersey=9) never appears as on-ball near the goal"


def test_gap_fill_increases_coverage():
    """gap_fill should fill all between-segment gaps in the active play
    region; trailing gaps after the last event (e.g. post-goal celebration)
    are real periods of zero possession and stay uncovered."""
    d = _load()
    tl_no_fill = build_possession_timeline(d["events"], d["info"])
    tl_fill = build_possession_timeline(
        d["events"], d["info"], gap_fill_max_frames=300)

    # Active play: from the first to the last segment in the window
    start_f = GOAL_F - 20 * 50
    end_f = GOAL_F + 15 * 50

    def _active_coverage(timeline):
        in_w = timeline[
            (timeline["frame_end"] >= start_f) & (timeline["frame_start"] <= end_f)
        ].sort_values("frame_start")
        if len(in_w) == 0:
            return 0, 0
        active_start = max(int(in_w.iloc[0]["frame_start"]), start_f)
        active_end = min(int(in_w.iloc[-1]["frame_end"]), end_f)
        total = 0
        for _, seg in in_w.iterrows():
            fs = max(int(seg["frame_start"]), start_f)
            fe = min(int(seg["frame_end"]), end_f)
            total += fe - fs + 1
        return total, active_end - active_start + 1

    cov_no, _ = _active_coverage(tl_no_fill)
    cov_fill, active_window = _active_coverage(tl_fill)
    assert cov_fill > cov_no, (
        f"gap_fill must strictly increase coverage (no_fill={cov_no}, fill={cov_fill})"
    )
    # Within the active play region, coverage after fill must be near 100%
    assert cov_fill / active_window > 0.95, (
        f"After gap_fill the active-play coverage should be >95%, "
        f"got {cov_fill/active_window:.1%} ({cov_fill}/{active_window})"
    )


# ── Scanning memory + DOS gate ─────────────────────────────────────────

@pytest.fixture(scope="module")
def kane_frame_setup():
    d = _load()
    skel = load_cached_skeleton(MATCH)
    # Use a small window around the goal frame to keep test fast
    probe = GOAL_F - 100  # 2s before the shot, Kane has the ball
    ss = skel[skel["frame_number"].between(probe - 200, probe + 5)]
    del skel
    ori = compute_orientations(ss, smooth=True)
    dyn = add_dynamics(ori)
    del ss, ori
    dyn = backfill_orientations(dyn, lookback_frames=150)

    tl = build_possession_timeline(d["events"], d["info"], gap_fill_max_frames=250)
    config = ScanningMemoryConfig(
        memory_window_s=2.5, tau_decay_s=1.2,
        vision_smoothing=2.0, framerate=50)
    memories = compute_scanning_memory_sequence(dyn, tl, (probe, probe), config)
    return {
        "info": d["info"],
        "dyn": dyn,
        "ball": load_cached_ball(MATCH),
        "tl": tl,
        "memories": memories,
        "probe": probe,
    }


def test_scanning_memory_non_empty_at_kane_carry(kane_frame_setup):
    s = kane_frame_setup
    fm = s["memories"].get(s["probe"])
    assert fm is not None, "Scanning memory missing for the probe frame"
    assert fm.memory.max() > 0.1, (
        f"Memory peak too low: {fm.memory.max():.4f} (expected > 0.1)"
    )
    assert (fm.memory > 0.05).sum() > 50, (
        f"Memory too sparse: {(fm.memory > 0.05).sum()} cells > 0.05"
    )


def test_dos_gate_paints_cells_at_kane_carry(kane_frame_setup):
    s = kane_frame_setup
    fd = s["dyn"][s["dyn"]["frame_number"] == s["probe"]]
    assert len(fd) > 0
    bf = s["ball"][s["ball"]["frame_number"] == s["probe"]]
    bx = float(bf.iloc[0]["x"]) if len(bf) > 0 else 0.0
    by = float(bf.iloc[0]["y"]) if len(bf) > 0 else 0.0

    dos_surf, _, _, _, _ = compute_dos_surface(
        fd, attacking_team=1, ball_xy=(bx, by), attacking_right=True,
        params=default_params(), n_grid_x=50, n_directions=24,
        vision_smoothing=1.0,
    )
    dos_pos = np.clip(dos_surf, 0.0, None)
    assert dos_pos.max() > 0.005, f"DOS too small: max={dos_pos.max()}"

    fm = s["memories"][s["probe"]]
    n_grid = 50
    n_grid_y = int(round(n_grid * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / n_grid
    dy = PITCH_WIDTH / n_grid_y
    xgrid = np.arange(n_grid) * dx - PITCH_LENGTH / 2 + dx / 2
    ygrid = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2 + dy / 2

    memory_on_dos = resample_memory_to_grid(fm.memory, xgrid, ygrid)
    assert memory_on_dos.shape == dos_pos.shape, (
        f"Resampled memory shape {memory_on_dos.shape} doesn't match dos {dos_pos.shape}"
    )
    assert memory_on_dos.max() > 0.1, (
        f"Resampled memory max too low: {memory_on_dos.max()}"
    )

    gated = dos_pos * memory_on_dos
    assert gated.max() > 0.001, f"Gated DOS max too small: {gated.max()}"
    # With proper thresholds (0.0008), at least a few cells should pass
    n_painted = (gated > 0.0008).sum()
    assert n_painted >= 10, (
        f"Only {n_painted} cells survive the gate at threshold 0.0008. "
        f"This means the visualization will be nearly empty."
    )
