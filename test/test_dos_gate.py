"""
Tests for the FOV gate integration in viz/dos_plot.py.

These tests are array-only (no rendering): they instantiate the function
with `ax` set to a fresh matplotlib axes and verify that the dos_norm
field is correctly gated and mapped via the smoothstep visibility curve.

Covers:
  - scanning_memory shape mismatch raises
  - Zero memory -> zero painted DOS (no flicker)
  - Full memory + above-floor DOS -> painted
  - DOS below noise_floor smoothly fades to 0
  - Smoothstep midpoint -> intermediate alpha
  - DOS above display_max saturates to alpha_max (no overflow)
  - Partial memory -> partial paint
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
            dos_surface=dos, scanning_memory=memory, ax=ax,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        assert len(ims) == 1
        rgba = ims[0].get_array()
        assert rgba[..., 3].max() == 0.0
    finally:
        plt.close(fig)


# ── Full memory: painted where DOS exists ──────────────────────────────

def test_full_memory_paints_dos():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.005)  # mid-range DOS
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory, ax=ax,
            noise_floor=0.0005, display_max=0.015,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        assert rgba[..., 3].max() > 0.0
        assert (rgba[..., 3] > 0).sum() > 0
    finally:
        plt.close(fig)


# ── Smoothstep visibility curve ────────────────────────────────────────

def test_noise_floor_smoothstep_kills_low_values():
    """Cells with gated DOS at or below noise_floor must fade to 0.

    The smoothstep is C^1 continuous, so a value strictly equal to
    noise_floor produces visibility 0 (no on/off cliff).
    """
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.0003)  # < noise_floor
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory, ax=ax,
            noise_floor=0.0005, display_max=0.015,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        assert rgba[..., 3].max() == 0.0
    finally:
        plt.close(fig)


def test_smoothstep_intermediate_value_partially_visible():
    """Values between noise_floor and display_max produce partial alpha."""
    fo = _orientations_frame()
    n = 50
    # Middle of [0.0005, 0.015] -> ~0.0078, smoothstep midpoint = 0.5
    dos = _fake_dos_surface(n, value=0.00775)
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory, ax=ax,
            noise_floor=0.0005, display_max=0.015, alpha_max=0.9,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        # Smoothstep at t=0.5 gives 0.5; alpha = 0.5 * 0.9 = 0.45
        assert 0.35 < rgba[..., 3].max() < 0.55
    finally:
        plt.close(fig)


def test_display_max_saturates():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.5)  # huge, >> display_max
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory, ax=ax,
            noise_floor=0.0005, display_max=0.015, alpha_max=0.9,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
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
            dos_surface=dos, scanning_memory=memory, ax=ax,
        )
        ims = [im for im in ax.get_images() if im.get_zorder() == 1]
        rgba = ims[0].get_array()
        # Left half should have zero alpha, right half non-zero
        assert rgba[:, : gx // 2, 3].max() == 0.0
        assert rgba[:, gx // 2 :, 3].max() > 0.0
    finally:
        plt.close(fig)
