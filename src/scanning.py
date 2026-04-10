"""
scanning — On-ball player visual scanning memory.

For a rendering sequence, precomputes the scanning memory grid at every
frame: the current on-ball player's field-of-view (Bekkers vision model)
combined with an exponentially-decayed memory of their own vision over
the last `memory_window_s` seconds (hard cutoff).

The scanning memory is computed at the full vision grid resolution
(pitch_length * smoothing by pitch_width * smoothing pixels) using the
complete Bekkers FOV + occlusion model with REAL per-occluder shoulder
widths from 3D skeleton keypoints. No subsampling, no interpolation
shortcuts — meant to run on GPU for production.

Cognitively, this represents: "what the current on-ball player has seen
in the last ~2.5 seconds, weighted by recency". It is the natural gate
for DOS (and any other action-relevant metric) because decisions the
player cannot perceive are not actionable.

Formally, for owner P at frame t:
    M(x, y, t) = max over tau in [0, memory_window_s] of:
                 V_P(x, y, t - tau) * exp(-tau / tau_decay_s)

where V_P is player P's FULL vision grid at the historical frame.

Key properties:
  - Owner transitions (pass -> receiver) do NOT inherit the passer's
    memory. The receiver's own scanning history is used, because that
    is what the receiver perceives.
  - Hard window cutoff: frames older than memory_window_s contribute
    exactly zero (not just exponentially small).
  - Per-segment caching: each possession segment precomputes its owner's
    vision once for [segment_start - window, segment_end], then the
    memory is a cheap sliding-window max over the cache.

Reference: Spielverlagerung (2025) — scanning behavior; Bekkers (2026)
— vision model. Memory window and decay calibrated to typical elite
player scanning rhythm (~2-3 head turns per 2-3 seconds).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .vision import compute_player_vision


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FRAMERATE = 50


# ── Config ──────────────────────────────────────────────────────────────

@dataclass
class ScanningMemoryConfig:
    """Parameters for the scanning memory computation.

    memory_window_s: hard cutoff (seconds). Frames older than this are
        not considered at all.
    tau_decay_s: exponential half-life for the decay weight. Smaller =
        faster forgetting.
    vision_smoothing: grid resolution multiplier for the Bekkers model
        (same meaning as in vision.py). 3.0 = 315x204 grid.
    framerate: sampling rate of the skeleton data. Default 50 Hz.
    """
    memory_window_s: float = 2.5
    tau_decay_s: float = 1.2
    vision_smoothing: float = 3.0
    framerate: int = FRAMERATE


@dataclass
class FrameMemory:
    """Scanning memory state for one frame."""
    memory: np.ndarray    # (gy, gx) decayed max over buffer, in [0, 1]
    fov_now: np.ndarray   # (gy, gx) current-frame FOV only, in [0, 1]
    owner_team: int
    owner_jersey: int
    mode: str


# ── Helpers ─────────────────────────────────────────────────────────────

def _vision_grid_extent(smoothing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Axis coordinate arrays (meters) for a vision grid at `smoothing`.

    Matches the convention used inside vision.py: grid covers the full
    pitch [-52.5, 52.5] x [-34, 34] with `pitch_length*smoothing` cells
    in x and `pitch_width*smoothing` cells in y.
    """
    gy = int(round(PITCH_WIDTH * smoothing))
    gx = int(round(PITCH_LENGTH * smoothing))
    y_coords = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, gy)
    x_coords = np.linspace(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, gx)
    return y_coords, x_coords


def _compute_single_vision(
    frame_df: pd.DataFrame,
    team: int,
    jersey: int,
    smoothing: float,
) -> Optional[np.ndarray]:
    """Compute the full Bekkers vision grid for one player at one frame.

    Returns None if the player is not present in `frame_df`.
    """
    row = frame_df[(frame_df["team"] == team) & (frame_df["jersey"] == jersey)]
    if len(row) == 0:
        return None
    p = row.iloc[0]
    others = frame_df[
        ~((frame_df["team"] == team) & (frame_df["jersey"] == jersey))
    ]
    ox = others["x"].values.astype(float)
    oy = others["y"].values.astype(float)
    osa = (others["shoulder_angle"].values.astype(float)
           if "shoulder_angle" in others.columns
           else np.zeros(len(others)))
    osw = (others["shoulder_width"].values.astype(float)
           if "shoulder_width" in others.columns else None)
    head = p.get("head_angle", p.get("shoulder_angle", 0.0))
    if head is None or pd.isna(head):
        head = p.get("shoulder_angle", 0.0)
    if head is None or pd.isna(head):
        head = 0.0
    speed = p.get("speed", 0.0)
    if speed is None or pd.isna(speed):
        speed = 0.0
    return compute_player_vision(
        float(p["x"]), float(p["y"]),
        float(head), float(speed),
        ox, oy, osa, osw,
        smoothing=smoothing,
    ).astype(np.float32)


# ── Main API ────────────────────────────────────────────────────────────

def compute_scanning_memory_sequence(
    orientations: pd.DataFrame,
    possession: pd.DataFrame,
    frame_range: Tuple[int, int],
    config: Optional[ScanningMemoryConfig] = None,
) -> Dict[int, FrameMemory]:
    """Compute scanning memory for every frame in `[frame_min, frame_max]`.

    For each frame t in the range, computes:
        memory(t) = max over tau in [0, memory_window_s] of:
                    V_owner(t)(x, y, t - tau) * exp(-tau / tau_decay_s)

    where owner(t) is the on-ball player at frame t (from `possession`),
    and V_P(s) is player P's full Bekkers vision grid at frame s.

    Visions are precomputed per-segment: for each possession segment
    with owner P on [fs, fe], P's vision is computed on
    [fs - mem_window, fe] exactly once and cached. The memory grid at
    each frame is then a max of `(cache[t - age] * weight[age])` over
    all valid ages in [0, mem_window].

    Args:
        orientations: Orientations+dynamics DataFrame (must contain
            frame_number, team, jersey, x, y, head_angle, shoulder_angle,
            speed, and ideally shoulder_width).
        possession: Timeline DataFrame from possession.build_possession_timeline.
        frame_range: (min_frame, max_frame) inclusive range to compute.
        config: Scanning memory configuration. Default if None.

    Returns:
        Dict mapping frame_number -> FrameMemory. Frames outside any
        possession segment (unknown owner) are simply absent from the
        dict and should render as fully masked (no DOS visible).
    """
    if config is None:
        config = ScanningMemoryConfig()

    mem_window = int(round(config.memory_window_s * config.framerate))
    tau_frames = config.tau_decay_s * config.framerate
    decay_weights = np.exp(
        -np.arange(mem_window + 1, dtype=np.float32) / tau_frames
    )

    f_min, f_max = frame_range
    y_coords, x_coords = _vision_grid_extent(config.vision_smoothing)
    gy, gx = len(y_coords), len(x_coords)

    result: Dict[int, FrameMemory] = {}

    # Segments that intersect the requested range.
    segments = possession[
        (possession["frame_end"] >= f_min) & (possession["frame_start"] <= f_max)
    ]

    # Index orientations by frame for fast per-frame slicing.
    ori_by_frame = dict(tuple(orientations.groupby("frame_number", sort=False)))

    for _, seg in segments.iterrows():
        team = int(seg["team"])
        jersey = int(seg["jersey"])
        mode = str(seg["mode"])

        fs = max(int(seg["frame_start"]), f_min)
        fe = min(int(seg["frame_end"]), f_max)
        if fs > fe:
            continue

        # Pre-compute the owner's vision for [fs - mem_window, fe] once.
        hist_start = fs - mem_window
        hist_end = fe
        cached_frames = []
        cached_grids = []
        for t in range(hist_start, hist_end + 1):
            fd = ori_by_frame.get(t)
            if fd is None or len(fd) == 0:
                continue
            v = _compute_single_vision(fd, team, jersey, config.vision_smoothing)
            if v is None:
                continue
            cached_frames.append(t)
            cached_grids.append(v)

        if not cached_grids:
            continue

        # Build a frame-indexed map for O(1) access.
        cache: Dict[int, np.ndarray] = dict(zip(cached_frames, cached_grids))

        # Memory per query frame in [fs, fe].
        for t in range(fs, fe + 1):
            fov_now = cache.get(t, np.zeros((gy, gx), dtype=np.float32))
            # Gather visions in the [t - mem_window, t] window.
            window_grids = []
            window_weights = []
            for age in range(mem_window + 1):
                tf = t - age
                g = cache.get(tf)
                if g is None:
                    continue
                window_grids.append(g)
                window_weights.append(decay_weights[age])
            if not window_grids:
                memory = np.zeros((gy, gx), dtype=np.float32)
            else:
                stack = np.stack(window_grids, axis=0)
                w = np.asarray(window_weights, dtype=np.float32)[:, None, None]
                memory = (stack * w).max(axis=0)
            result[t] = FrameMemory(
                memory=memory,
                fov_now=fov_now,
                owner_team=team,
                owner_jersey=jersey,
                mode=mode,
            )

    return result


# ── Resampling to a DOS grid ────────────────────────────────────────────

def resample_memory_to_grid(
    memory: np.ndarray,
    target_xgrid: np.ndarray,
    target_ygrid: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample a scanning memory grid onto a target grid.

    Input `memory` is assumed to span the full pitch at whatever
    resolution it was computed. Output is sized `(len(target_ygrid),
    len(target_xgrid))` and clipped to [0, 1].
    """
    gy, gx = memory.shape
    y_coords = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, gy)
    x_coords = np.linspace(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, gx)
    interp = RegularGridInterpolator(
        (y_coords, x_coords), memory,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    yy, xx = np.meshgrid(target_ygrid, target_xgrid, indexing="ij")
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    out = interp(pts).reshape(yy.shape)
    return np.clip(out.astype(np.float32), 0.0, 1.0)
