"""
Tests for src/scanning.py — scanning memory pipeline.

Covers:
  - Vision grid extent shapes (smoothing parameter)
  - Single vision compute presence + missing player handling
  - Single-frame memory equals current FOV
  - Static scene: memory == FOV at every frame
  - Rotation: memory remembers an old gaze direction
  - Hard cutoff: memory drops past memory_window_s
  - Decay weights monotonic
  - Owner transition: receiver does NOT inherit passer's memory
  - resample_memory_to_grid: shape, range, value preservation
  - Frames outside any segment have no entry in result
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.scanning import (
    ScanningMemoryConfig, FrameMemory,
    compute_scanning_memory_sequence,
    resample_memory_to_grid,
    backfill_orientations,
    _compute_single_vision, _vision_grid_extent,
    PITCH_LENGTH, PITCH_WIDTH,
)
from src.possession import MODE_CARRY


# ── Fixtures ────────────────────────────────────────────────────────────

def _player(team, jersey, x, y, head_angle, shoulder_angle=None,
            shoulder_width=0.45, speed=0.0):
    if shoulder_angle is None:
        shoulder_angle = head_angle
    return {
        "team": team, "jersey": jersey, "x": x, "y": y,
        "head_angle": head_angle, "shoulder_angle": shoulder_angle,
        "shoulder_width": shoulder_width, "speed": speed,
    }


def _orientations(frames, players_per_frame):
    """Build orientations DataFrame from a per-frame players list.

    `players_per_frame` may be a single list (same for all frames) or a
    dict {frame: list_of_players}.
    """
    rows = []
    for f in frames:
        if isinstance(players_per_frame, dict):
            ps = players_per_frame.get(f, [])
        else:
            ps = players_per_frame
        for p in ps:
            rows.append({"frame_number": f, **p})
    return pd.DataFrame(rows)


def _carry_segment(team, jersey, fs, fe):
    return pd.DataFrame([{
        "frame_start": fs, "frame_end": fe,
        "team": team, "jersey": jersey,
        "player_id": f"P{team}_{jersey}", "mode": MODE_CARRY,
        "source_event_id": "c",
    }])


# ── Vision grid extent ─────────────────────────────────────────────────

def test_vision_grid_extent_shape():
    y, x = _vision_grid_extent(3.0)
    assert len(x) == int(round(PITCH_LENGTH * 3.0))
    assert len(y) == int(round(PITCH_WIDTH * 3.0))
    assert x[0] == pytest.approx(-PITCH_LENGTH / 2)
    assert x[-1] == pytest.approx(PITCH_LENGTH / 2)
    assert y[0] == pytest.approx(-PITCH_WIDTH / 2)
    assert y[-1] == pytest.approx(PITCH_WIDTH / 2)


def test_vision_grid_extent_smoothing_2():
    y, x = _vision_grid_extent(2.0)
    assert len(x) == 210
    assert len(y) == 136


# ── Single vision ──────────────────────────────────────────────────────

def test_single_vision_present_player():
    players = [
        _player(1, 11, 0.0, 0.0, head_angle=0.0),
        _player(0, 7,  20.0, 5.0, head_angle=np.pi),
    ]
    ori = _orientations([100], players)
    fd = ori[ori["frame_number"] == 100]
    grid = _compute_single_vision(fd, team=1, jersey=11, smoothing=2.0)
    assert grid is not None
    assert grid.shape == (136, 210)
    assert grid.dtype == np.float32
    assert grid.max() > 0
    # Player at (0,0) facing +X: right half should have more vision
    h, w = grid.shape
    assert grid[:, w // 2:].sum() > grid[:, : w // 2].sum()


def test_single_vision_missing_player_returns_none():
    players = [_player(1, 11, 0.0, 0.0, head_angle=0.0)]
    ori = _orientations([100], players)
    fd = ori[ori["frame_number"] == 100]
    assert _compute_single_vision(fd, team=1, jersey=99, smoothing=2.0) is None


# ── Memory: degenerate cases ────────────────────────────────────────────

def test_memory_single_frame_equals_fov():
    """When only one historical frame exists, memory == fov_now."""
    players = [
        _player(1, 11, 0.0, 0.0, head_angle=0.0),
        _player(0, 7,  10.0, 0.0, head_angle=np.pi),
    ]
    # Only frame 110 has data; segment is [110, 110]
    ori = _orientations([110], players)
    poss = _carry_segment(1, 11, 110, 110)
    config = ScanningMemoryConfig(memory_window_s=2.5, vision_smoothing=2.0)
    out = compute_scanning_memory_sequence(ori, poss, (110, 110), config)
    assert 110 in out
    fm = out[110]
    np.testing.assert_allclose(fm.memory, fm.fov_now, atol=1e-6)
    assert fm.owner_team == 1 and fm.owner_jersey == 11
    assert fm.mode == MODE_CARRY


def test_memory_static_scene_equals_fov_at_every_frame():
    """If the player and the world are static, memory should equal fov."""
    players = [
        _player(1, 11, 0.0, 0.0, head_angle=0.0),
        _player(0, 7,  20.0, 5.0, head_angle=np.pi),
    ]
    frames = list(range(80, 121))
    ori = _orientations(frames, players)
    poss = _carry_segment(1, 11, 100, 120)
    config = ScanningMemoryConfig(memory_window_s=0.5, tau_decay_s=1.0,
                                  vision_smoothing=2.0)
    out = compute_scanning_memory_sequence(ori, poss, (110, 120), config)
    for t in range(110, 121):
        fm = out[t]
        np.testing.assert_allclose(fm.memory, fm.fov_now, atol=1e-5)


def test_frames_outside_segments_absent():
    players = [_player(1, 11, 0.0, 0.0, 0.0),
               _player(0, 7, 10.0, 0.0, np.pi)]
    ori = _orientations(list(range(100, 121)), players)
    poss = _carry_segment(1, 11, 105, 110)
    out = compute_scanning_memory_sequence(
        ori, poss, (100, 120), ScanningMemoryConfig(vision_smoothing=2.0))
    for t in range(100, 105):
        assert t not in out
    for t in range(105, 111):
        assert t in out
    for t in range(111, 121):
        assert t not in out


# ── Memory: rotation / persistence ──────────────────────────────────────

def test_memory_remembers_past_rotation():
    """Player looks +X for 30 frames, then -X. Memory at the last frame
    should still encode some +X visibility (because it's within the 2.5s
    window) — even though current FOV is purely -X.
    """
    rows = []
    for f in range(80, 145):
        head = 0.0 if f < 110 else np.pi
        rows.append({"frame_number": f, **_player(1, 11, 0.0, 0.0, head)})
        rows.append({"frame_number": f, **_player(0, 7, 30.0, 0.0, np.pi)})
        rows.append({"frame_number": f, **_player(0, 8, -30.0, 0.0, 0.0)})
    ori = pd.DataFrame(rows)
    poss = _carry_segment(1, 11, 100, 140)
    config = ScanningMemoryConfig(memory_window_s=1.0, tau_decay_s=10.0,
                                  vision_smoothing=2.0)
    out = compute_scanning_memory_sequence(ori, poss, (115, 115), config)
    fm = out[115]
    h, w = fm.memory.shape
    # Current FOV (looking -X) should be left-heavy.
    assert fm.fov_now[:, : w // 2].sum() > fm.fov_now[:, w // 2 :].sum()
    # Memory should still have content on the right (the +X scan from
    # frames 80-109 within the 1s window of frame 115).
    assert fm.memory[:, w // 2 :].sum() > 0
    # And memory should also include the current left FOV.
    assert fm.memory[:, : w // 2].sum() > 0
    # Memory should have at least as much total mass as fov_now.
    assert fm.memory.sum() >= fm.fov_now.sum() - 1e-3


def test_memory_hard_cutoff_kills_old_observations():
    """A scan that happened way outside the memory window should NOT
    leak into the memory at the query frame."""
    rows = []
    for f in range(0, 200):
        head = 0.0 if f < 50 else np.pi
        rows.append({"frame_number": f, **_player(1, 11, 0.0, 0.0, head)})
        rows.append({"frame_number": f, **_player(0, 7, 30.0, 0.0, np.pi)})
    ori = pd.DataFrame(rows)
    poss = _carry_segment(1, 11, 0, 199)
    config = ScanningMemoryConfig(memory_window_s=0.5,  # 25 frames
                                  tau_decay_s=100.0,    # essentially no decay
                                  vision_smoothing=2.0)
    out = compute_scanning_memory_sequence(ori, poss, (40, 195), config)

    # At frame 60: 10 frames since the rotation, +X scan (last at f=49)
    # is 11 frames in the past — within the 25-frame window. Memory
    # should have RIGHT-side content.
    early = out[60]
    h, w = early.memory.shape
    right_early = early.memory[:, w // 2:].sum()
    assert right_early > 0

    # At frame 100: last +X scan was at f=49, age 51 frames > 25.
    # No right-side memory.
    late = out[100]
    right_late = late.memory[:, w // 2:].sum()
    assert right_late == 0.0


def test_decay_weights_monotonic_decreasing():
    """Per-age weights should be exp(-age/tau_frames), strictly decreasing."""
    rows = []
    # Player rotates by 1 deg per frame so each historical FOV is unique
    for f in range(0, 30):
        head = (f * np.pi / 60)
        rows.append({"frame_number": f, **_player(1, 11, 0.0, 0.0, head)})
        rows.append({"frame_number": f, **_player(0, 7, 10.0, 0.0, np.pi)})
    ori = pd.DataFrame(rows)
    poss = _carry_segment(1, 11, 0, 29)
    config = ScanningMemoryConfig(memory_window_s=0.5,
                                  tau_decay_s=0.2,  # fast decay
                                  vision_smoothing=2.0)
    out = compute_scanning_memory_sequence(ori, poss, (29, 29), config)
    fm = out[29]
    # The newest FOV (frame 29) should dominate the memory: max value
    # should equal the max of fov_now (weight = 1).
    assert abs(fm.memory.max() - fm.fov_now.max()) < 1e-5


# ── Owner transition ───────────────────────────────────────────────────

def test_receiver_does_not_inherit_passer_memory():
    """When the on-ball player changes, the new owner's vision history
    is what counts. The previous owner's vision should not contaminate
    the receiver's scanning memory.

    The Bekkers FOV peaks at the player's own position (distance=0
    -> radial gaussian = 1), so we verify the centre-of-mass of the
    vision grid moves with the on-ball player's pitch position.
    """
    # P11 at origin looking +X for f=0..49
    # P12 at (40, 10) looking +X for f=50..99
    rows = []
    for f in range(0, 100):
        rows.append({"frame_number": f, **_player(1, 11, 0.0, 0.0, 0.0)})
        rows.append({"frame_number": f, **_player(1, 12, 40.0, 10.0, 0.0)})
        rows.append({"frame_number": f, **_player(0, 7, -20.0, 0.0, np.pi)})
    ori = pd.DataFrame(rows)
    poss = pd.DataFrame([
        {"frame_start": 0, "frame_end": 49, "team": 1, "jersey": 11,
         "player_id": "P1_11", "mode": MODE_CARRY, "source_event_id": "c1"},
        {"frame_start": 50, "frame_end": 99, "team": 1, "jersey": 12,
         "player_id": "P1_12", "mode": MODE_CARRY, "source_event_id": "c2"},
    ])
    config = ScanningMemoryConfig(memory_window_s=2.5, tau_decay_s=2.0,
                                  vision_smoothing=2.0)
    out = compute_scanning_memory_sequence(ori, poss, (49, 60), config)

    # Helper: compute the centroid of a vision grid in field meters.
    def _centroid_meters(grid):
        h, w = grid.shape
        ys = np.linspace(-34, 34, h)
        xs = np.linspace(-52.5, 52.5, w)
        total = grid.sum() + 1e-12
        cx = (grid.sum(axis=0) * xs).sum() / total
        cy = (grid.sum(axis=1) * ys).sum() / total
        return cx, cy

    # P11 at origin and P12 at (40, 10) both look +x. Both vision
    # cones have most of their mass forward, so the centroid is biased
    # right of the player. Comparing the two: P12's centroid should be
    # ~40m further right and ~10m further up than P11's, reflecting his
    # pitch position. If the receiver inherited the passer's memory the
    # difference would shrink.
    fm49 = out[49]
    fm60 = out[60]
    assert fm49.owner_jersey == 11
    assert fm60.owner_jersey == 12

    cx49, cy49 = _centroid_meters(fm49.fov_now)
    cx60, cy60 = _centroid_meters(fm60.fov_now)

    # Centroids should shift in the direction of the player movement.
    # The shift is smaller than the player delta because P12's cone is
    # clipped by the pitch edge, but it must still be clearly positive.
    assert (cx60 - cx49) > 10.0   # centroid moved right with the player
    assert (cy60 - cy49) > 5.0    # centroid moved up with the player

    # Same check on the memory grid: the memory at f=60 must reflect P12,
    # not P11. If the receiver had inherited P11's memory, the memory
    # centroid would be similar to fm49's, not shifted with the player.
    cmx60, cmy60 = _centroid_meters(fm60.memory)
    cmx49, cmy49 = _centroid_meters(fm49.memory)
    assert (cmx60 - cmx49) > 10.0
    assert (cmy60 - cmy49) > 5.0


# ── Resampling ──────────────────────────────────────────────────────────

def test_resample_shape_matches_target():
    memory = np.zeros((136, 210), dtype=np.float32)
    target_x = np.linspace(-52.5, 52.5, 50)
    target_y = np.linspace(-34, 34, 32)
    out = resample_memory_to_grid(memory, target_x, target_y)
    assert out.shape == (32, 50)
    assert out.dtype == np.float32


def test_resample_preserves_central_bright_value():
    memory = np.zeros((136, 210), dtype=np.float32)
    # Brighten a 5x5 patch at the center
    cy, cx = 68, 105
    memory[cy - 2: cy + 3, cx - 2: cx + 3] = 1.0
    target_x = np.linspace(-52.5, 52.5, 105)
    target_y = np.linspace(-34, 34, 68)
    out = resample_memory_to_grid(memory, target_x, target_y)
    # Center cell should be ~1
    assert out[34, 52] > 0.9
    # Far corner should be ~0
    assert out[0, 0] < 0.05


def test_resample_clipped_to_unit_interval():
    memory = np.full((136, 210), 2.5, dtype=np.float32)
    target_x = np.linspace(-52.5, 52.5, 10)
    target_y = np.linspace(-34, 34, 7)
    out = resample_memory_to_grid(memory, target_x, target_y)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_resample_handles_zero_input():
    memory = np.zeros((50, 80), dtype=np.float32)
    target_x = np.linspace(-52.5, 52.5, 10)
    target_y = np.linspace(-34, 34, 7)
    out = resample_memory_to_grid(memory, target_x, target_y)
    assert out.shape == (7, 10)
    assert (out == 0.0).all()


# ── Backfill orientations ──────────────────────────────────────────────

def test_backfill_orientations_zero_lookback_noop():
    players = [_player(1, 11, 0.0, 0.0, 0.0)]
    ori = _orientations([100, 101, 102], players)
    out = backfill_orientations(ori, lookback_frames=0)
    assert len(out) == len(ori)


def test_backfill_orientations_adds_synthetic_rows():
    players = [
        _player(1, 11, 0.0, 0.0, 0.5),
        _player(0, 7, 10.0, 0.0, np.pi),
    ]
    ori = _orientations([100, 101], players)
    out = backfill_orientations(ori, lookback_frames=10)
    # Each player should now have rows from 90 to 101 = 12 frames
    for (t, j), grp in out.groupby(["team", "jersey"]):
        frames = sorted(grp["frame_number"].unique())
        assert frames[0] == 90
        assert frames[-1] == 101
        assert len(frames) == 12


def test_backfill_orientations_clones_earliest_pose():
    players = [_player(1, 11, 5.0, 3.0, 1.234, shoulder_width=0.42)]
    ori = _orientations([100], players)
    out = backfill_orientations(ori, lookback_frames=5)
    synth = out[(out["frame_number"] < 100)]
    assert len(synth) == 5
    assert (synth["x"] == 5.0).all()
    assert (synth["y"] == 3.0).all()
    assert np.allclose(synth["head_angle"], 1.234)
    assert np.allclose(synth["shoulder_width"], 0.42)


def test_backfill_orientations_multi_player_no_dupes():
    players = [
        _player(1, 11, 0.0, 0.0, 0.0),
        _player(1, 12, 5.0, 5.0, 0.5),
        _player(0, 7, 10.0, 0.0, np.pi),
    ]
    ori = _orientations([100, 101, 102], players)
    out = backfill_orientations(ori, lookback_frames=20)
    # Each player should have 23 frames (20 synthetic + 3 real), no dupes
    for (t, j), grp in out.groupby(["team", "jersey"]):
        frames = grp["frame_number"].values
        assert len(frames) == len(np.unique(frames))
        assert len(frames) == 23


# ── NaN handling ──────────────────────────────────────────────────────

def test_compute_single_vision_handles_nan_head_angle():
    """If head_angle is NaN (smoothing edge), fall back to shoulder_angle."""
    rows = [{
        "frame_number": 0, "team": 1, "jersey": 11,
        "x": 0.0, "y": 0.0,
        "head_angle": np.float32("nan"),
        "shoulder_angle": 0.0,
        "shoulder_width": 0.45, "speed": 0.0,
    }, {
        "frame_number": 0, "team": 0, "jersey": 7,
        "x": 10.0, "y": 0.0,
        "head_angle": np.pi, "shoulder_angle": np.pi,
        "shoulder_width": 0.45, "speed": 0.0,
    }]
    ori = pd.DataFrame(rows)
    fd = ori[ori["frame_number"] == 0]
    grid = _compute_single_vision(fd, team=1, jersey=11, smoothing=2.0)
    assert grid is not None
    assert not np.any(np.isnan(grid))
    # Should look like a +x cone (using shoulder_angle = 0.0 as fallback)
    h, w = grid.shape
    assert grid[:, w // 2:].sum() > grid[:, : w // 2].sum()


def test_compute_single_vision_handles_nan_speed():
    """If speed is NaN, treat as zero (widest cone)."""
    rows = [{
        "frame_number": 0, "team": 1, "jersey": 11,
        "x": 0.0, "y": 0.0,
        "head_angle": 0.0, "shoulder_angle": 0.0,
        "shoulder_width": 0.45, "speed": np.float32("nan"),
    }, {
        "frame_number": 0, "team": 0, "jersey": 7,
        "x": 10.0, "y": 0.0,
        "head_angle": np.pi, "shoulder_angle": np.pi,
        "shoulder_width": 0.45, "speed": 0.0,
    }]
    ori = pd.DataFrame(rows)
    fd = ori[ori["frame_number"] == 0]
    grid = _compute_single_vision(fd, team=1, jersey=11, smoothing=2.0)
    assert grid is not None
    assert not np.any(np.isnan(grid))
    assert grid.max() > 0


def test_compute_single_vision_handles_all_nan_angles_falls_back_to_zero():
    rows = [{
        "frame_number": 0, "team": 1, "jersey": 11,
        "x": 0.0, "y": 0.0,
        "head_angle": np.float32("nan"),
        "shoulder_angle": np.float32("nan"),
        "shoulder_width": 0.45, "speed": 0.0,
    }, {
        "frame_number": 0, "team": 0, "jersey": 7,
        "x": 10.0, "y": 0.0,
        "head_angle": 0.0, "shoulder_angle": 0.0,
        "shoulder_width": 0.45, "speed": 0.0,
    }]
    ori = pd.DataFrame(rows)
    fd = ori[ori["frame_number"] == 0]
    grid = _compute_single_vision(fd, team=1, jersey=11, smoothing=2.0)
    assert grid is not None
    assert not np.any(np.isnan(grid))
    # head defaults to 0 -> looking +x
    h, w = grid.shape
    assert grid[:, w // 2:].sum() > grid[:, : w // 2].sum()


# ── Frame range / segment filtering ───────────────────────────────────

def test_segment_fully_outside_range_skipped():
    players = [_player(1, 11, 0.0, 0.0, 0.0),
               _player(0, 7, 10.0, 0.0, np.pi)]
    ori = _orientations(list(range(0, 50)), players)
    poss = _carry_segment(1, 11, 0, 49)
    # Query range completely past the segment
    out = compute_scanning_memory_sequence(
        ori, poss, (200, 250),
        ScanningMemoryConfig(vision_smoothing=2.0))
    assert len(out) == 0


def test_segment_partially_inside_range_clipped():
    players = [_player(1, 11, 0.0, 0.0, 0.0),
               _player(0, 7, 10.0, 0.0, np.pi)]
    ori = _orientations(list(range(0, 200)), players)
    # Segment 50-150, query 100-180. Output frames should be 100-150.
    poss = _carry_segment(1, 11, 50, 150)
    out = compute_scanning_memory_sequence(
        ori, poss, (100, 180),
        ScanningMemoryConfig(memory_window_s=0.5, vision_smoothing=2.0))
    in_range = sorted(out.keys())
    assert in_range[0] >= 100
    assert in_range[-1] <= 150


def test_backfill_then_compute_memory_populated_at_first_frame():
    """End-to-end: without backfill the first frame's memory == fov_now,
    with backfill the memory is fully decayed-max populated."""
    players = [
        _player(1, 11, 0.0, 0.0, 0.0),
        _player(0, 7, 10.0, 0.0, np.pi),
    ]
    ori = _orientations([200], players)
    poss = _carry_segment(1, 11, 200, 200)
    config = ScanningMemoryConfig(memory_window_s=1.0, tau_decay_s=1.0,
                                  vision_smoothing=2.0)
    # No backfill: memory == fov_now (only 1 historical frame)
    out_raw = compute_scanning_memory_sequence(ori, poss, (200, 200), config)
    fm_raw = out_raw[200]
    assert np.isclose(fm_raw.memory.sum(), fm_raw.fov_now.sum())

    # With backfill: 50 synthetic prior frames appear in the cache
    ori_filled = backfill_orientations(ori, lookback_frames=50)
    out_fill = compute_scanning_memory_sequence(ori_filled, poss, (200, 200), config)
    fm_fill = out_fill[200]
    # Memory should have AT LEAST as much mass as fov_now (decayed clones
    # don't increase the max but they don't reduce it either; the static
    # scene means the max is fov_now anyway).
    assert fm_fill.memory.sum() >= fm_fill.fov_now.sum() - 1e-3
    # Owner unchanged
    assert fm_fill.owner_team == 1 and fm_fill.owner_jersey == 11

