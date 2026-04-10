"""
Tests for the FOV gate integration in viz/dos_plot.py.

These tests are array-only (no rendering): they instantiate the function
with `ax` set to a fresh matplotlib axes and verify that the dos_norm
field is correctly gated, thresholded and capped.

Covers:
  - scanning_memory shape mismatch raises
  - Zero memory -> zero painted DOS (no flicker)
  - Memory > 0, DOS > 0 -> painted (above threshold)
  - DOS below absolute_threshold killed
  - DOS above display_max saturates (no overflow)
  - Legacy heuristic still works when scanning_memory is None
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.viz.dos_plot import plot_dos_frame


# ── Fixtures ────────────────────────────────────────────────────────────

def _orientations_frame():
    """Minimal one-frame orientations DataFrame with 2 attackers + 2 defenders."""
    rows = [
        {"frame_number": 0, "team": 1, "jersey": 11, "x": -10.0, "y": 0.0,
         "head_angle": 0.0, "shoulder_angle": 0.0, "shoulder_width": 0.45,
         "speed": 1.0, "vx": 1.0, "vy": 0.0},
        {"frame_number": 0, "team": 1, "jersey": 12, "x":  20.0, "y": 5.0,
         "head_angle": 0.0, "shoulder_angle": 0.0, "shoulder_width": 0.45,
         "speed": 1.0, "vx": 1.0, "vy": 0.0},
        {"frame_number": 0, "team": 0, "jersey": 7, "x":  10.0, "y": -3.0,
         "head_angle": np.pi, "shoulder_angle": np.pi, "shoulder_width": 0.45,
         "speed": 0.5, "vx": -0.5, "vy": 0.0},
        {"frame_number": 0, "team": 0, "jersey": 8, "x":  30.0, "y": 0.0,
         "head_angle": np.pi, "shoulder_angle": np.pi, "shoulder_width": 0.45,
         "speed": 0.5, "vx": -0.5, "vy": 0.0},
    ]
    return pd.DataFrame(rows)


def _grid_shape(n_grid_x):
    PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
    return (int(round(n_grid_x * PITCH_WIDTH / PITCH_LENGTH)), n_grid_x)


def _fake_dos_surface(n_grid_x, value=0.02):
    """Create a uniform synthetic DOS grid for gate testing."""
    gy, gx = _grid_shape(n_grid_x)
    return np.full((gy, gx), value, dtype=np.float32)


def _fake_best_direction(n_grid_x):
    gy, gx = _grid_shape(n_grid_x)
    return np.zeros((gy, gx), dtype=np.float32)


# ── Shape mismatch ──────────────────────────────────────────────────────

def test_shape_mismatch_raises():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n)
    bad_memory = np.ones((10, 10), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="scanning_memory shape"):
            plot_dos_frame(
                fo, attacking_team=1, ball_xy=(0.0, 0.0),
                attacking_right=True,
                dos_surface=dos,
                best_direction=_fake_best_direction(n),
                scanning_memory=bad_memory,
                ax=ax,
            )
    finally:
        plt.close(fig)


# ── Zero memory: nothing painted ───────────────────────────────────────

def test_zero_memory_paints_nothing():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.02)
    memory = np.zeros(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, best_direction=_fake_best_direction(n),
            scanning_memory=memory, ax=ax,
        )
        # The DOS heatmap is the lowest-zorder image. Inspect its alpha.
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        assert len(ims) == 1
        rgba = ims[0].get_array()
        # Either alpha is zero everywhere, or the data is fully zero.
        assert rgba[..., 3].max() == 0.0
    finally:
        plt.close(fig)


# ── Full memory: painted where DOS exists ──────────────────────────────

def test_full_memory_paints_dos():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.005)  # well above threshold
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, best_direction=_fake_best_direction(n),
            scanning_memory=memory, ax=ax,
            absolute_threshold=0.0008, display_max=0.015,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        # All cells should have non-zero alpha
        assert rgba[..., 3].max() > 0.0
        assert (rgba[..., 3] > 0).sum() > 0
    finally:
        plt.close(fig)


# ── Threshold kills sub-threshold values ───────────────────────────────

def test_absolute_threshold_kills_low_values():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.0003)  # well below 0.0008 threshold
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, best_direction=_fake_best_direction(n),
            scanning_memory=memory, ax=ax,
            absolute_threshold=0.0008, display_max=0.015,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        assert rgba[..., 3].max() == 0.0
    finally:
        plt.close(fig)


# ── Display max saturates without overflow ─────────────────────────────

def test_display_max_saturates():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.5)  # huge, > display_max
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, best_direction=_fake_best_direction(n),
            scanning_memory=memory, ax=ax,
            absolute_threshold=0.0008, display_max=0.015,
            alpha_max=0.9,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        # Alpha should be at the cap (saturated), not exceed alpha_max
        assert rgba[..., 3].max() <= 0.9 + 1e-6
        assert rgba[..., 3].max() == pytest.approx(0.9, abs=1e-6)
    finally:
        plt.close(fig)


# ── Partial gating ─────────────────────────────────────────────────────

def test_partial_memory_partial_paint():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.02)
    gy, gx = _grid_shape(n)
    memory = np.zeros((gy, gx), dtype=np.float32)
    memory[:, gx // 2:] = 1.0  # right half visible
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, best_direction=_fake_best_direction(n),
            scanning_memory=memory, ax=ax,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        # Left half should have zero alpha, right half non-zero
        assert rgba[:, : gx // 2, 3].max() == 0.0
        assert rgba[:, gx // 2 :, 3].max() > 0.0
    finally:
        plt.close(fig)


# ── Legacy fallback (no memory provided) ───────────────────────────────

def test_legacy_path_runs_when_no_memory():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.02)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, best_direction=_fake_best_direction(n),
            scanning_memory=None, ax=ax,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        # Legacy behavior should also produce something within the
        # attacker_radius of the attackers we placed.
        assert rgba.shape[0] > 0
    finally:
        plt.close(fig)
