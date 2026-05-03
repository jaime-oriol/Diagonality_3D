"""Unit tests for src/xt.py — Karun Singh xT lookup.

Covers every public surface:
  - load_grid (file presence, shape)
  - tracab_to_xt (scalar, vector, attacking_right True/False)
  - xt_at_pitch (scalar, vector, bounds clipping, NaN propagation, shape)
  - xt_at_tracab (delegation)
  - xt_delta (success path, fail path, mixed success array, NaN)
  - Cross-validation against socceraction's published values
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from src.xt import (
    PITCH_LENGTH, PITCH_WIDTH, GRID_NX, GRID_NY,
    load_grid, tracab_to_xt,
    xt_at_pitch, xt_at_tracab, xt_delta,
)


# ── Grid loader ────────────────────────────────────────────────────────

def test_load_grid_shape_and_range():
    g = load_grid()
    assert g.shape == (GRID_NY, GRID_NX)
    # Karun Singh's published values: defensive zone ~0.006, attacking centre ~0.26
    assert 0.005 < g.min() < 0.01
    assert 0.20 < g.max() < 0.30
    # Symmetric in y: row 0 ≈ row 7 (sidelines), row 3 ≈ row 4 (centre)
    np.testing.assert_allclose(g[0], g[7], rtol=0.02)
    np.testing.assert_allclose(g[3], g[4], rtol=0.02)


def test_load_grid_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="xT grid not found"):
        load_grid(tmp_path / "no_such_file.npy")


# ── Coordinate conversion ─────────────────────────────────────────────

def test_tracab_to_xt_attacking_right_centre():
    x_p, y_p = tracab_to_xt(0.0, 0.0, True)
    assert float(x_p) == pytest.approx(52.5)
    assert float(y_p) == pytest.approx(34.0)


def test_tracab_to_xt_attacking_right_extreme():
    # +x in TRACAB is the rival goal when attacking right
    x_p, y_p = tracab_to_xt(52.5, 34.0, True)
    assert float(x_p) == pytest.approx(105.0)
    assert float(y_p) == pytest.approx(68.0)


def test_tracab_to_xt_attacking_left_flips():
    # Attacking left: rival goal is at -x. Should flip to xT pitch +x.
    x_p, y_p = tracab_to_xt(-52.5, -34.0, False)
    assert float(x_p) == pytest.approx(105.0)
    assert float(y_p) == pytest.approx(68.0)


def test_tracab_to_xt_vectorized():
    xs = np.array([0.0, 30.0, -30.0])
    ys = np.array([0.0, 10.0, -5.0])
    ar = np.array([True, True, False])
    x_p, y_p = tracab_to_xt(xs, ys, ar)
    assert x_p.shape == (3,)
    assert y_p.shape == (3,)
    # Attack right: x + 52.5; attack left: -x + 52.5
    np.testing.assert_allclose(x_p, [52.5, 82.5, 82.5])
    np.testing.assert_allclose(y_p, [34.0, 44.0, 39.0])


# ── xt_at_pitch ────────────────────────────────────────────────────────

def test_xt_at_pitch_centre_value():
    # Centre of pitch should be a low-ish but non-trivial value
    v = xt_at_pitch(52.5, 34.0)
    assert 0.005 < v < 0.05


def test_xt_at_pitch_near_goal_high():
    # Near rival goal centre: very high value
    v = xt_at_pitch(104.0, 34.0)
    assert v > 0.20


def test_xt_at_pitch_own_box_low():
    # Own goal area: very low
    v = xt_at_pitch(2.0, 34.0)
    assert v < 0.02


def test_xt_at_pitch_vector_input():
    out = xt_at_pitch(np.array([52.5, 100.0]), np.array([34.0, 34.0]))
    assert out.shape == (2,)
    assert out[1] > out[0]   # closer to rival goal → higher xT


def test_xt_at_pitch_clipping_handles_out_of_range():
    # Out-of-pitch inputs are clipped, NOT raised
    v = xt_at_pitch(150.0, 100.0)
    assert np.isfinite(v)
    v_neg = xt_at_pitch(-50.0, -50.0)
    assert np.isfinite(v_neg)


def test_xt_at_pitch_nan_propagates():
    out = xt_at_pitch(np.array([52.5, np.nan]), np.array([34.0, 34.0]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


def test_xt_at_pitch_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must have the same shape"):
        xt_at_pitch(np.array([1.0, 2.0]), np.array([3.0]))


# ── xt_at_tracab ───────────────────────────────────────────────────────

def test_xt_at_tracab_attacking_right_matches_pitch():
    # Same point, both calls, same answer
    v_pitch = xt_at_pitch(95.0, 34.0)
    v_tracab = xt_at_tracab(95.0 - 52.5, 34.0 - 34.0, True)
    np.testing.assert_allclose(v_tracab, v_pitch, rtol=1e-6)


def test_xt_at_tracab_attacking_left_mirrors():
    # Same physical position, attacking opposite direction → same xT
    # (because we flip into the standard frame).
    v_right = xt_at_tracab(30.0, 10.0, True)
    v_left = xt_at_tracab(-30.0, -10.0, False)
    np.testing.assert_allclose(v_right, v_left, rtol=1e-6)


# ── xt_delta ────────────────────────────────────────────────────────────

def test_xt_delta_progressive_pass_positive():
    # Progressive pass from own half to rival box, attacking right
    d = xt_delta(
        x_origin=-20.0, y_origin=0.0,
        x_dest=40.0, y_dest=10.0,
        attacking_right=True, success=True,
    )
    assert d.item() > 0.05


def test_xt_delta_backward_pass_negative():
    d = xt_delta(
        x_origin=40.0, y_origin=0.0,
        x_dest=-10.0, y_dest=0.0,
        attacking_right=True, success=True,
    )
    assert d.item() < -0.05


def test_xt_delta_unsuccessful_returns_minus_origin_xt():
    # Failed pass: xt_delta = -xT(origin), regardless of dest
    origin_xt = xt_at_tracab(40.0, 10.0, True).item()
    d = xt_delta(40.0, 10.0, 50.0, 0.0, True, success=False)
    np.testing.assert_allclose(d.item(), -origin_xt, rtol=1e-6)


def test_xt_delta_default_success():
    # success=None → assumed success (full Δ)
    d = xt_delta(0.0, 0.0, 30.0, 0.0, True)
    d_explicit = xt_delta(0.0, 0.0, 30.0, 0.0, True, success=True)
    np.testing.assert_allclose(d, d_explicit)


def test_xt_delta_mixed_success_vector():
    n = 4
    xo = np.full(n, 0.0); yo = np.full(n, 0.0)
    xd = np.full(n, 30.0); yd = np.full(n, 0.0)
    ar = np.full(n, True)
    succ = np.array([True, False, True, False])
    d = xt_delta(xo, yo, xd, yd, ar, success=succ)
    assert d.shape == (n,)
    # Where success=True the value > 0 (progressive); where False, it's -xT(origin) < 0
    assert d[0] > 0
    assert d[1] < 0
    assert d[2] > 0
    assert d[3] < 0
    np.testing.assert_allclose(d[0], d[2])     # both successful, same coords
    np.testing.assert_allclose(d[1], d[3])     # both failed, same origin


def test_xt_delta_nan_inputs_give_nan():
    d = xt_delta(np.nan, 0.0, 30.0, 0.0, True, success=True)
    assert np.isnan(d.item())


# ── Symmetry: attacking right vs left should give same Δ for same physical motion ──

def test_xt_delta_symmetric_to_attacking_direction():
    # A→B with attacking right
    d_right = xt_delta(-20.0, 5.0, 30.0, 5.0, True, success=True)
    # Same physical motion but mirrored, attacking left (positions flipped)
    d_left = xt_delta(20.0, -5.0, -30.0, -5.0, False, success=True)
    np.testing.assert_allclose(d_right, d_left, rtol=1e-6)


# ── Cross-check against the publicly observed grid values ──────────────

def test_grid_central_attacking_third_high():
    # The centre of the attacking final third (~ x=90, y=34) should be one
    # of the highest-value cells in the grid.
    g = load_grid()
    # Cells near rival goal (right side)
    central_attacking_zone = g[3:5, 9:12]
    others = np.concatenate([g[:3].ravel(), g[5:].ravel(), g[3:5, :9].ravel()])
    assert central_attacking_zone.mean() > others.mean() * 1.5
