"""
xt — Karun Singh (2018) Expected Threat lookup.

We use the published 12x8 xT grid (bundled offline at src/data/xt_singh_12x8.npy
to avoid network dependency at run time). xT is a Markov-chain valuation
that assigns a per-cell value xT(x, y) ∈ [0, 1] interpreted as the expected
goal probability from an attacking team in possession at that cell. It is
the de-facto standard outcome metric in the football analytics literature
since 2019 because:

  - It is **per-action**: ΔxT = xT(end) - xT(start) values each event
    independently. No dilution across an entire possession.
  - It works **uniformly** for passes / carries / take-ons (any action
    that moves the ball or preserves it at a position).
  - It is **transparent** (no black-box xgboost) and uses NO information
    from skeleton / vision / orientation, so it cannot leak with our
    DOS metric: DOS and xT-delta are mathematically independent variables.

Coordinate convention:
  - Internal grid: 8 rows (y) × 12 cols (x). Pitch 105 m × 68 m. Attacking
    direction is +x (toward x = 105). Cell centers at (col*8.75 + 4.375,
    row*8.5 + 4.25).
  - Project / TRACAB convention: x ∈ [-52.5, 52.5], y ∈ [-34, 34], origin
    at the center circle. Attacking direction depends on team and half.
  - `xt_at_tracab(x, y, attacking_right)` handles both conventions: when
    attacking_right=False it flips x AND y so the lookup always uses the
    "attacking-right" frame the grid expects.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ── Constants ──────────────────────────────────────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GRID_NX = 12
GRID_NY = 8
CELL_W = PITCH_LENGTH / GRID_NX     # 8.75 m
CELL_H = PITCH_WIDTH / GRID_NY      # 8.50 m

_GRID_PATH = Path(__file__).resolve().parent / "data" / "xt_singh_12x8.npy"

# Lazy-loaded singletons (load grid only on first call)
_GRID: Optional[np.ndarray] = None
_INTERP: Optional[RegularGridInterpolator] = None


# ── Loader ─────────────────────────────────────────────────────────────

def load_grid(path: Optional[Path] = None) -> np.ndarray:
    """Load the bundled Karun Singh 12x8 xT grid as a (8, 12) np.array.

    The file is generated once via `socceraction.xthreat.load_model(URL)`
    and saved to `src/data/xt_singh_12x8.npy`. Range observed: 0.006 to 0.257.
    """
    p = Path(path) if path is not None else _GRID_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"xT grid not found at {p}. Regenerate via: "
            "`from socceraction.xthreat import load_model; "
            "import numpy as np; "
            "xt = load_model('https://karun.in/blog/data/open_xt_12x8_v1.json'); "
            f"np.save({p!r}, xt.xT)`")
    grid = np.load(p)
    if grid.shape != (GRID_NY, GRID_NX):
        raise ValueError(
            f"xT grid shape {grid.shape} != expected ({GRID_NY}, {GRID_NX})")
    return grid


def _ensure_interpolator() -> RegularGridInterpolator:
    global _GRID, _INTERP
    if _INTERP is None:
        _GRID = load_grid()
        # Cell centers in pitch coords. The grid is regular so a bilinear
        # interpolator is exact at the cell centers and smooth elsewhere.
        y_centers = (np.arange(GRID_NY) + 0.5) * CELL_H   # [4.25, 12.75, ..., 63.75]
        x_centers = (np.arange(GRID_NX) + 0.5) * CELL_W   # [4.375, 13.125, ..., 100.625]
        _INTERP = RegularGridInterpolator(
            (y_centers, x_centers), _GRID,
            method="linear", bounds_error=False, fill_value=None,
        )
    return _INTERP


# ── Coordinate conversion ──────────────────────────────────────────────

def tracab_to_xt(
    x_tracab: Union[float, Sequence[float], np.ndarray],
    y_tracab: Union[float, Sequence[float], np.ndarray],
    attacking_right: Union[bool, Sequence[bool], np.ndarray],
) -> tuple:
    """Convert TRACAB centered coords to xT pitch coords (origin
    bottom-left, attacking +x). When `attacking_right` is False, x and y
    are flipped so the result is always in the attacking-right frame.

    Args:
        x_tracab, y_tracab: scalar or array-like in TRACAB meters
            (x ∈ [-52.5, 52.5], y ∈ [-34, 34]).
        attacking_right: scalar bool or boolean array, broadcastable.

    Returns:
        (x_pitch, y_pitch) as np.ndarray in [0, 105] × [0, 68].
    """
    x = np.asarray(x_tracab, dtype=np.float64)
    y = np.asarray(y_tracab, dtype=np.float64)
    ar = np.asarray(attacking_right, dtype=bool)
    sign = np.where(ar, 1.0, -1.0)
    x_pitch = sign * x + PITCH_LENGTH / 2.0
    y_pitch = sign * y + PITCH_WIDTH / 2.0
    return x_pitch, y_pitch


# ── Lookups ────────────────────────────────────────────────────────────

def xt_at_pitch(
    x_pitch: Union[float, Sequence[float], np.ndarray],
    y_pitch: Union[float, Sequence[float], np.ndarray],
) -> np.ndarray:
    """xT lookup at pitch coords (x ∈ [0, 105], y ∈ [0, 68]). Bilinear
    interpolation of the 12×8 grid; clipped to a 0.5 m margin to avoid
    out-of-grid extrapolation. Returns shape matching the input."""
    interp = _ensure_interpolator()
    x = np.atleast_1d(np.asarray(x_pitch, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y_pitch, dtype=np.float64))
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    xc = np.clip(x, 0.5, PITCH_LENGTH - 0.5)
    yc = np.clip(y, 0.5, PITCH_WIDTH - 0.5)
    pts = np.column_stack([yc.ravel(), xc.ravel()])
    out = interp(pts).reshape(x.shape)
    # NaN propagation: NaN inputs → NaN output (RGI returns nan implicitly,
    # but make it deterministic).
    nan_mask = np.isnan(x) | np.isnan(y)
    out = np.where(nan_mask, np.nan, out)
    return out if x.shape != () else float(out)


def xt_at_tracab(
    x_tracab: Union[float, Sequence[float], np.ndarray],
    y_tracab: Union[float, Sequence[float], np.ndarray],
    attacking_right: Union[bool, Sequence[bool], np.ndarray],
) -> np.ndarray:
    """xT lookup directly from TRACAB centered coords with attacking_right
    flag. Returns the same shape as the input arrays."""
    x_p, y_p = tracab_to_xt(x_tracab, y_tracab, attacking_right)
    return xt_at_pitch(x_p, y_p)


# ── Action-level value ─────────────────────────────────────────────────

def xt_delta(
    x_origin: Union[Sequence[float], np.ndarray],
    y_origin: Union[Sequence[float], np.ndarray],
    x_dest: Union[Sequence[float], np.ndarray],
    y_dest: Union[Sequence[float], np.ndarray],
    attacking_right: Union[bool, Sequence[bool], np.ndarray],
    success: Optional[Union[bool, Sequence[bool], np.ndarray]] = None,
) -> np.ndarray:
    """Compute Δ-xT for a batch of move actions (passes / carries).

    Convention used by the standard xT literature (Singh 2018; Decroos
    socceraction):
        success=True  → xt_delta = xT(dest)   - xT(origin)
        success=False → xt_delta = -xT(origin)   (lost the ball)
        success=None  → assume success

    Args:
        x/y_origin: origin position(s) in TRACAB centered meters.
        x/y_dest: destination position(s) in TRACAB centered meters.
        attacking_right: bool or array, broadcastable to the same shape.
        success: optional bool or array. If None, treated as all-success.

    Returns:
        np.ndarray of shape matching the inputs, with Δ-xT per action.
        NaN where any input coord is NaN.
    """
    xt_o = xt_at_tracab(x_origin, y_origin, attacking_right)
    xt_d = xt_at_tracab(x_dest, y_dest, attacking_right)
    if success is None:
        return xt_d - xt_o
    succ = np.asarray(success, dtype=bool)
    delta_success = xt_d - xt_o
    delta_fail = -xt_o
    return np.where(succ, delta_success, delta_fail)


__all__ = [
    "PITCH_LENGTH", "PITCH_WIDTH", "GRID_NX", "GRID_NY", "CELL_W", "CELL_H",
    "load_grid", "tracab_to_xt", "xt_at_pitch", "xt_at_tracab", "xt_delta",
]
