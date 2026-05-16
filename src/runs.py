"""
runs — Off-ball run detection from skeleton dynamics.

An off-ball run is a sustained acceleration by an attacking player who is
NOT the on-ball owner — the tactically relevant movement that creates a
passing option (Spielverlagerung, "Diagonality in the movement").

Detection is deliberately lightweight and aligned with the public notion
used by commercial providers (SkillCorner Game Intelligence, StatsPerform
Opta Vision): a run is a contiguous burst of high speed with real
displacement. We do NOT classify run types — the contribution of this
project is feeding the detected run into the orientation-aware DOS model,
not re-deriving a run taxonomy.

A run is detected when an attacking off-ball player:
  - sustains speed > RUN_SPEED_MS for >= RUN_MIN_DURATION_S, and
  - covers a net displacement > RUN_MIN_DISPLACEMENT_M.

A brief dip below the speed threshold (<= gap_tolerance_frames) does not
break a run — elite runners modulate pace mid-run.

Input:  dynamics DataFrame (frame_number, team, jersey, x, y, vx, vy), as
        produced by orientation.add_dynamics, plus a possession timeline
        from possession.build_possession_timeline.
Output: one row per run, anchored at its start frame.

All coordinates in METERS, angles in RADIANS.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from .possession import get_owner_at_frame

FRAMERATE = 50

# --- Detection thresholds -------------------------------------------------
# Aligned with the generic off-ball-run notion used by SkillCorner / Opta
# Vision: a real burst, not a jog. Tunable via detect_runs() arguments.
RUN_SPEED_MS = 5.0            # sustained speed threshold (m/s)
RUN_MIN_DURATION_S = 0.7      # minimum sustained duration (s)
RUN_MIN_DISPLACEMENT_M = 5.0  # minimum net displacement (m)
GAP_TOLERANCE_FRAMES = 3      # brief sub-threshold dips that do not split a run

RUN_COLUMNS = [
    "team", "jersey", "frame_start", "frame_end", "n_frames", "duration_s",
    "x_start", "y_start", "x_end", "y_end", "displacement_m",
    "vx_start", "vy_start", "peak_speed", "mean_speed", "run_dir",
    "owner_team", "owner_jersey",
]


# --- Interval helpers -----------------------------------------------------

def _contiguous_segments(frames: np.ndarray) -> List[Tuple[int, int]]:
    """Split a sorted frame-number array into runs of consecutive frames.

    The skeleton cache is event-windowed, so a single player's frame
    series has gaps between windows. Within a window frames step by 1.
    Returns a list of (lo, hi) INDEX slices (inclusive) into `frames`.
    """
    n = len(frames)
    if n == 0:
        return []
    segments = []
    lo = 0
    for i in range(1, n):
        if frames[i] != frames[i - 1] + 1:
            segments.append((lo, i - 1))
            lo = i
    segments.append((lo, n - 1))
    return segments


def _threshold_intervals(
    above: np.ndarray, min_len: int, gap_tol: int,
) -> List[Tuple[int, int]]:
    """Maximal intervals where `above` is True, merging internal gaps of
    up to `gap_tol` consecutive False frames.

    Returns (start, end) INDEX pairs (inclusive), trimmed so each interval
    starts and ends on a True frame, keeping only intervals whose frame
    span is at least `min_len`.
    """
    idx = np.flatnonzero(above)
    if len(idx) == 0:
        return []
    intervals = []
    group_start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k - prev > gap_tol + 1:
            intervals.append((group_start, prev))
            group_start = k
        prev = k
    intervals.append((group_start, prev))
    return [(s, e) for s, e in intervals if (e - s + 1) >= min_len]


# --- Main detector --------------------------------------------------------

def detect_runs(
    dyn: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    speed_thresh: float = RUN_SPEED_MS,
    min_duration_s: float = RUN_MIN_DURATION_S,
    min_displacement_m: float = RUN_MIN_DISPLACEMENT_M,
    gap_tolerance_frames: int = GAP_TOLERANCE_FRAMES,
    framerate: int = FRAMERATE,
) -> pd.DataFrame:
    """Detect off-ball attacking runs in a dynamics DataFrame.

    Args:
        dyn: Per-frame per-player dynamics with columns frame_number,
            team, jersey, x, y, vx, vy (output of orientation.add_dynamics).
        timeline: Possession timeline from build_possession_timeline.
            Used to (a) identify the attacking team frame by frame and
            (b) exclude the on-ball owner (whose burst is a carry, not an
            off-ball run).
        speed_thresh, min_duration_s, min_displacement_m,
        gap_tolerance_frames: Detection parameters (see module docstring).

    Returns:
        DataFrame with one row per run (columns = RUN_COLUMNS), anchored
        at the run's start frame. A run is kept only if, at its start
        frame, an on-ball owner exists who is a team-mate of the runner
        (attacking) and a different player (off-ball).
    """
    if len(dyn) == 0 or len(timeline) == 0:
        return pd.DataFrame(columns=RUN_COLUMNS)

    needed = {"frame_number", "team", "jersey", "x", "y", "vx", "vy"}
    missing = needed - set(dyn.columns)
    if missing:
        raise ValueError(f"dyn missing columns: {sorted(missing)}")

    min_frames = int(round(min_duration_s * framerate))
    records: List[dict] = []

    for (team, jersey), g in dyn.groupby(["team", "jersey"], sort=False):
        g = g.sort_values("frame_number")
        frames = g["frame_number"].values.astype(np.int64)
        xs = g["x"].values.astype(np.float64)
        ys = g["y"].values.astype(np.float64)
        vx = g["vx"].values.astype(np.float64)
        vy = g["vy"].values.astype(np.float64)
        speed = np.hypot(vx, vy)

        for lo, hi in _contiguous_segments(frames):
            seg_above = speed[lo:hi + 1] > speed_thresh
            for s, e in _threshold_intervals(
                seg_above, min_frames, gap_tolerance_frames
            ):
                i, j = lo + s, lo + e  # absolute indices into g
                disp = float(np.hypot(xs[j] - xs[i], ys[j] - ys[i]))
                if disp < min_displacement_m:
                    continue

                f_start = int(frames[i])
                owner = get_owner_at_frame(timeline, f_start)
                if owner is None:
                    continue
                # Attacking team-mate, but not the ball carrier himself.
                if owner["team"] != int(team):
                    continue
                if owner["jersey"] == int(jersey):
                    continue

                records.append({
                    "team": int(team),
                    "jersey": int(jersey),
                    "frame_start": f_start,
                    "frame_end": int(frames[j]),
                    "n_frames": int(j - i + 1),
                    "duration_s": float((frames[j] - frames[i]) / framerate),
                    "x_start": xs[i], "y_start": ys[i],
                    "x_end": xs[j], "y_end": ys[j],
                    "displacement_m": disp,
                    "vx_start": vx[i], "vy_start": vy[i],
                    "peak_speed": float(speed[i:j + 1].max()),
                    "mean_speed": float(speed[i:j + 1].mean()),
                    "run_dir": float(np.arctan2(ys[j] - ys[i], xs[j] - xs[i])),
                    "owner_team": int(owner["team"]),
                    "owner_jersey": int(owner["jersey"]),
                })

    if not records:
        return pd.DataFrame(columns=RUN_COLUMNS)
    return pd.DataFrame(records, columns=RUN_COLUMNS).sort_values(
        "frame_start").reset_index(drop=True)
