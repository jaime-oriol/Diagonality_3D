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

from src.viz.dos_plot import plot_dos_frame, compute_forward_cone_mask


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

# ── Forward-cone mask helper ────────────────────────────────────────────

def test_forward_cone_mask_attacking_right():
    xgrid = np.linspace(-52.5, 52.5, 100)
    ygrid = np.linspace(-34, 34, 65)
    mask = compute_forward_cone_mask((0.0, 0.0), True, xgrid, ygrid, max_dist_m=20.0)
    # Everything behind the ball (x<0) masked out
    for i in range(len(ygrid)):
        assert mask[i, xgrid < 0].max() == 0.0
    # Cell at (5, 0): ahead and within range
    ix = int(np.argmin(np.abs(xgrid - 5.0)))
    iy = int(np.argmin(np.abs(ygrid - 0.0)))
    assert mask[iy, ix] == 1.0
    # Cell at (25, 0): ahead but outside 20m range -> masked
    ix = int(np.argmin(np.abs(xgrid - 25.0)))
    assert mask[iy, ix] == 0.0


def test_forward_cone_mask_attacking_left():
    xgrid = np.linspace(-52.5, 52.5, 100)
    ygrid = np.linspace(-34, 34, 65)
    mask = compute_forward_cone_mask((0.0, 0.0), False, xgrid, ygrid, max_dist_m=20.0)
    for i in range(len(ygrid)):
        assert mask[i, xgrid > 0].max() == 0.0
    ix = int(np.argmin(np.abs(xgrid - (-5.0))))
    iy = int(np.argmin(np.abs(ygrid - 0.0)))
    assert mask[iy, ix] == 1.0


# ── Shadow layer ────────────────────────────────────────────────────────

def test_shadow_layer_shape_mismatch_raises():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.01)
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    bad_shadow = np.ones((10, 10), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="shadow_surface shape"):
            plot_dos_frame(
                fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
                dos_surface=dos, scanning_memory=memory,
                shadow_surface=bad_shadow, ax=ax,
            )
    finally:
        plt.close(fig)


def test_shadow_layer_not_rendered_when_none():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.01)
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory,
            shadow_surface=None, ax=ax,
        )
        # Only the visible DOS imshow (zorder=1) — no shadow (zorder=0).
        ims = ax.get_images()
        zs = sorted(im.get_zorder() for im in ims)
        assert 0 not in zs
        assert 1 in zs
    finally:
        plt.close(fig)


def test_shadow_layer_painted_when_active():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.001)          # visible DOS
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    shadow = _fake_dos_surface(n, value=0.02)         # above shadow floor
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory,
            shadow_surface=shadow, ax=ax,
            shadow_noise_floor=0.003, shadow_display_max=0.025,
            shadow_alpha_max=0.55,
        )
        shadow_ims = [im for im in ax.get_images() if im.get_zorder() == 0]
        assert len(shadow_ims) == 1
        rgba = shadow_ims[0].get_array()
        assert rgba[..., 3].max() > 0.0
    finally:
        plt.close(fig)


def test_shadow_layer_below_floor_invisible():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.001)
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    shadow = _fake_dos_surface(n, value=0.001)        # well below shadow floor
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory,
            shadow_surface=shadow, ax=ax,
            shadow_noise_floor=0.003, shadow_display_max=0.025,
        )
        shadow_ims = [im for im in ax.get_images() if im.get_zorder() == 0]
        assert shadow_ims[0].get_array()[..., 3].max() == 0.0
    finally:
        plt.close(fig)


def test_shadow_layer_full_contract_integration():
    """End-to-end: given DOS + memory + ball, the rendered shadow alpha is
    nonzero ONLY where ALL four invariants hold:
      (1) ahead of the ball in attacking direction
      (2) within shadow_max_dist_m of the ball
      (3) scanning memory < 1 (player doesn't see the cell)
      (4) effective shadow DOS >= shadow_noise_floor

    This is the contract that guarantees "no DOS de mierda que distraiga".
    """
    fo = _orientations_frame()
    n = 50
    PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
    gy, gx = _grid_shape(n)
    dx = PITCH_LENGTH / n
    dy = PITCH_WIDTH / gy
    xgrid = np.arange(n) * dx - PITCH_LENGTH / 2 + dx / 2
    ygrid = np.arange(gy) * dy - PITCH_WIDTH / 2 + dy / 2

    # Scenario: uniform strong DOS across the pitch; memory covers only
    # the right half (ahead of the ball when attacking_right=True); ball
    # at origin. Expected:
    #   - Visible DOS paints ONLY the right half (memory present).
    #   - Shadow paints ONLY the FORWARD-RIGHT cone minus memory=inside.
    #     Since memory=1 on the right side, shadow must be zero EVERYWHERE.
    # Then we flip memory to the LEFT half and verify the shadow fills
    # the forward-right cone exactly.
    dos = np.full((gy, gx), 0.020, dtype=np.float32)
    ball_xy = (0.0, 0.0)
    max_dist = 20.0

    # Scenario A: memory covers the forward cone -> shadow must be empty
    memory_A = np.zeros((gy, gx), dtype=np.float32)
    memory_A[:, gx // 2:] = 1.0  # right half seen
    mask_A = compute_forward_cone_mask(ball_xy, True, xgrid, ygrid, max_dist)
    shadow_A = dos * (1.0 - memory_A) * mask_A
    assert shadow_A.max() == 0.0  # no unseen cells ahead

    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=ball_xy, attacking_right=True,
            dos_surface=dos, scanning_memory=memory_A,
            shadow_surface=shadow_A, ax=ax,
            shadow_noise_floor=0.003, shadow_display_max=0.025,
        )
        sh_rgba = [im for im in ax.get_images() if im.get_zorder() == 0][0].get_array()
        assert sh_rgba[..., 3].max() == 0.0
    finally:
        plt.close(fig)

    # Scenario B: memory covers the LEFT half (behind ball when attacking
    # right). Shadow should paint the forward-right cone EXCLUSIVELY.
    memory_B = np.zeros((gy, gx), dtype=np.float32)
    memory_B[:, : gx // 2] = 1.0  # left half seen (behind ball)
    mask_B = compute_forward_cone_mask(ball_xy, True, xgrid, ygrid, max_dist)
    shadow_B = dos * (1.0 - memory_B) * mask_B

    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=ball_xy, attacking_right=True,
            dos_surface=dos, scanning_memory=memory_B,
            shadow_surface=shadow_B, ax=ax,
            shadow_noise_floor=0.003, shadow_display_max=0.025,
            shadow_alpha_max=0.55,
        )
        sh_rgba = [im for im in ax.get_images() if im.get_zorder() == 0][0].get_array()
        alpha = sh_rgba[..., 3]
        # BEHIND the ball (x < 0 in forward mask) -> strictly zero alpha
        # xgrid cells with xgrid[j] < 0 are behind; imshow extent matches
        # the grid we computed, so column j in the RGBA maps to xgrid[j].
        behind_cols = np.where(xgrid < 0)[0]
        assert alpha[:, behind_cols].max() == 0.0
        # AHEAD of ball AND within range -> should light up
        ahead_cols = np.where((xgrid > 0) & (xgrid < max_dist))[0]
        assert alpha[:, ahead_cols].max() > 0.0
        # Far AHEAD but beyond max_dist -> zero (range cap works)
        far_cols = np.where(xgrid > max_dist)[0]
        if len(far_cols) > 0:
            assert alpha[:, far_cols].max() == 0.0
    finally:
        plt.close(fig)


def test_shadow_layer_saturates_at_display_max():
    fo = _orientations_frame()
    n = 50
    dos = _fake_dos_surface(n, value=0.001)
    memory = np.ones(_grid_shape(n), dtype=np.float32)
    shadow = _fake_dos_surface(n, value=1.0)           # huge, clamps to max
    fig, ax = plt.subplots()
    try:
        plot_dos_frame(
            fo, attacking_team=1, ball_xy=(0.0, 0.0), attacking_right=True,
            dos_surface=dos, scanning_memory=memory,
            shadow_surface=shadow, ax=ax,
            shadow_noise_floor=0.003, shadow_display_max=0.025,
            shadow_alpha_max=0.55,
        )
        shadow_ims = [im for im in ax.get_images() if im.get_zorder() == 0]
        assert shadow_ims[0].get_array()[..., 3].max() == pytest.approx(0.55, abs=1e-6)
    finally:
        plt.close(fig)


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
