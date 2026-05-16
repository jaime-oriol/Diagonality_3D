"""
Tests for src/runs.py — off-ball run detection.

Covers:
  - _contiguous_segments: gaps from the event-windowed cache
  - _threshold_intervals: min length, gap tolerance, trimming
  - detect_runs: a real sprint is detected; a jog is not; the on-ball
    owner's burst is excluded; a defender's burst is excluded; a fast but
    low-displacement burst is excluded; empty inputs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.runs import (
    detect_runs, _contiguous_segments, _threshold_intervals,
    RUN_COLUMNS, FRAMERATE,
)
from src.possession import TIMELINE_COLUMNS


# --- interval helpers -----------------------------------------------------

def test_contiguous_segments():
    assert _contiguous_segments(np.array([])) == []
    assert _contiguous_segments(np.array([5])) == [(0, 0)]
    assert _contiguous_segments(np.array([1, 2, 3, 7, 8])) == [(0, 2), (3, 4)]
    assert _contiguous_segments(np.array([1, 3, 5])) == [(0, 0), (1, 1), (2, 2)]


def test_threshold_intervals_basic():
    above = np.array([False, True, True, True, False])
    assert _threshold_intervals(above, min_len=3, gap_tol=0) == [(1, 3)]
    assert _threshold_intervals(above, min_len=4, gap_tol=0) == []


def test_threshold_intervals_gap_tolerance():
    # one-frame dip in the middle of an otherwise sustained burst
    above = np.array([True, True, False, True, True])
    assert _threshold_intervals(above, min_len=4, gap_tol=1) == [(0, 4)]
    # with no tolerance the dip splits it into two short intervals
    assert _threshold_intervals(above, min_len=4, gap_tol=0) == []
    assert _threshold_intervals(above, min_len=2, gap_tol=0) == [(0, 1), (3, 4)]


def test_threshold_intervals_trims_to_true():
    above = np.array([False, False, True, True, True, True, False])
    # interval starts/ends on True, not on the leading/trailing False
    assert _threshold_intervals(above, min_len=1, gap_tol=0) == [(2, 5)]


# --- detect_runs ----------------------------------------------------------

FRAMES = np.arange(1000, 1101)  # 101 contiguous frames @ 50 Hz = 2.0 s


def _player(team, jersey, vx, vy, x0=0.0, y0=0.0, x=None, y=None):
    """Build one player's dynamics rows over FRAMES."""
    n = len(FRAMES)
    vx = np.full(n, vx, dtype=float) if np.isscalar(vx) else np.asarray(vx, float)
    vy = np.full(n, vy, dtype=float) if np.isscalar(vy) else np.asarray(vy, float)
    if x is None:
        x = x0 + np.cumsum(vx) / FRAMERATE
    if y is None:
        y = y0 + np.cumsum(vy) / FRAMERATE
    return pd.DataFrame({
        "frame_number": FRAMES, "team": team, "jersey": jersey,
        "x": x, "y": y, "vx": vx, "vy": vy,
    })


def _timeline(owner_team, owner_jersey, fs=1000, fe=1100):
    """Single-segment possession timeline: one owner over [fs, fe]."""
    row = {
        "frame_start": fs, "frame_end": fe,
        "team": owner_team, "jersey": owner_jersey,
        "player_id": "P", "mode": "carry", "source_event_id": "E",
    }
    return pd.DataFrame([row], columns=TIMELINE_COLUMNS)


def test_detects_a_real_sprint():
    runner = _player(team=1, jersey=11, vx=6.0, vy=0.0)
    dyn = runner
    timeline = _timeline(owner_team=1, owner_jersey=7)
    runs = detect_runs(dyn, timeline)
    assert len(runs) == 1
    r = runs.iloc[0]
    assert r["team"] == 1 and r["jersey"] == 11
    assert r["frame_start"] == 1000
    assert r["duration_s"] >= 0.7
    assert r["displacement_m"] > 5.0
    assert r["owner_jersey"] == 7
    assert list(runs.columns) == RUN_COLUMNS


def test_excludes_jog_below_speed_threshold():
    jog = _player(team=1, jersey=9, vx=3.0, vy=0.0)  # 3 m/s < 5 m/s
    runs = detect_runs(jog, _timeline(1, 7))
    assert len(runs) == 0


def test_excludes_on_ball_owner_burst():
    # the owner himself sprinting is a carry, not an off-ball run
    owner_burst = _player(team=1, jersey=7, vx=6.0, vy=0.0)
    runs = detect_runs(owner_burst, _timeline(owner_team=1, owner_jersey=7))
    assert len(runs) == 0


def test_excludes_defender_burst():
    defender = _player(team=0, jersey=4, vx=6.0, vy=0.0)
    runs = detect_runs(defender, _timeline(owner_team=1, owner_jersey=7))
    assert len(runs) == 0


def test_excludes_fast_but_low_displacement():
    # fast (speed 6 m/s) but pinned in place -> net displacement ~0
    n = len(FRAMES)
    spinning = _player(team=1, jersey=13, vx=6.0, vy=0.0,
                       x=np.zeros(n), y=np.zeros(n))
    runs = detect_runs(spinning, _timeline(1, 7))
    assert len(runs) == 0


def test_excludes_short_burst():
    # 20 frames above threshold (0.4 s) then stop -> below min duration
    vx = np.concatenate([np.full(20, 6.0), np.zeros(len(FRAMES) - 20)])
    short = _player(team=1, jersey=12, vx=vx, vy=0.0)
    runs = detect_runs(short, _timeline(1, 7))
    assert len(runs) == 0


def test_full_scene_keeps_only_the_off_ball_attacker():
    dyn = pd.concat([
        _player(team=1, jersey=11, vx=6.0, vy=0.0),   # off-ball attacker -> run
        _player(team=1, jersey=7, vx=6.0, vy=0.0),    # owner -> excluded
        _player(team=0, jersey=4, vx=6.0, vy=0.0),    # defender -> excluded
        _player(team=1, jersey=9, vx=3.0, vy=0.0),    # jog -> excluded
    ], ignore_index=True)
    runs = detect_runs(dyn, _timeline(owner_team=1, owner_jersey=7))
    assert len(runs) == 1
    assert runs.iloc[0]["jersey"] == 11


def test_empty_inputs():
    empty = pd.DataFrame(columns=["frame_number", "team", "jersey",
                                  "x", "y", "vx", "vy"])
    assert len(detect_runs(empty, _timeline(1, 7))) == 0
    runner = _player(team=1, jersey=11, vx=6.0, vy=0.0)
    assert len(detect_runs(runner, pd.DataFrame(columns=TIMELINE_COLUMNS))) == 0
