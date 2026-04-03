"""
orientation — Compute body orientations from skeleton keypoints.

Extracts head yaw, shoulder facing, hip facing, torso lean, movement
direction, body openness, and angular velocities from the 21-keypoint
TRACAB skeleton data.

All angles in RADIANS, measured counter-clockwise from the positive X-axis.

Input:  cached skeleton DataFrame (from preprocess.py)
Output: player-level DataFrame with orientations per frame
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .loader import PART_NAMES

# Part IDs
_NOSE = 2
_L_EAR = 1
_R_EAR = 3
_L_SHOULDER = 4
_R_SHOULDER = 6
_NECK = 5
_L_HIP = 11
_R_HIP = 13
_PELVIS = 12
_L_ANKLE = 16
_R_ANKLE = 17

# Parts needed for orientation computation
ORIENT_PART_IDS = {_NOSE, _L_EAR, _R_EAR, _L_SHOULDER, _R_SHOULDER,
                   _NECK, _L_HIP, _R_HIP, _PELVIS, _L_ANKLE, _R_ANKLE}


# --- Angle math (vectorized) ---------------------------------------------

def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angular difference (a - b), wrapped to [-pi, pi]."""
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unsigned angular distance between two angles, in [0, pi]."""
    return np.abs(angle_diff(a, b))


def _perp_angle(lx, ly, rx, ry):
    """Facing direction: perpendicular to left->right axis (rotated +90 deg)."""
    dx = rx - lx
    dy = ry - ly
    return np.arctan2(dx, -dy)



# --- Pivot skeleton to wide format ----------------------------------------

def _pivot_skeleton(skeleton: pd.DataFrame) -> pd.DataFrame:
    """Pivot from long (1 row per part) to wide (1 row per player per frame)."""
    df = skeleton[skeleton["part_id"].isin(ORIENT_PART_IDS)].copy()
    df["part"] = df["part_id"].map(PART_NAMES)

    wide = df.pivot_table(
        index=["frame_number", "team", "jersey"],
        columns="part",
        values=["x", "y", "z"],
        aggfunc="first",
    )
    wide.columns = [f"{part}_{coord}" for coord, part in wide.columns]
    return wide.reset_index()


# --- Smoothing ------------------------------------------------------------

def _smooth_angle_series(angles: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth an angle series using Savitzky-Golay on sin/cos components.

    Angles are circular, so we decompose into sin/cos, smooth each,
    and reconstruct. This avoids wrap-around artifacts.
    """
    if len(angles) < window:
        return angles
    sin_smooth = savgol_filter(np.sin(angles), window, polyorder=2, mode="nearest")
    cos_smooth = savgol_filter(np.cos(angles), window, polyorder=2, mode="nearest")
    return np.arctan2(sin_smooth, cos_smooth)


def _smooth_scalar_series(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth a scalar series using Savitzky-Golay."""
    if len(values) < window:
        return values
    return savgol_filter(values, window, polyorder=2, mode="nearest")


# --- Main computation -----------------------------------------------------

def compute_orientations(skeleton: pd.DataFrame, smooth: bool = True) -> pd.DataFrame:
    """Compute all body orientations from skeleton data.

    Args:
        skeleton: Cached skeleton DataFrame (frame_number, team, jersey, part_id, x, y, z)
        smooth: Apply Savitzky-Golay smoothing to angles and positions (recommended)

    Returns:
        DataFrame with one row per player per frame:
            frame_number, team, jersey,
            x, y,                — player position (pelvis, meters)
            head_angle,          — head yaw: nose direction from ear midpoint (radians)
            head_angle_perp,     — head yaw: perpendicular to ear-ear axis (radians)
            shoulder_angle,      — shoulder facing: perp to shoulder axis (radians)
            hip_angle,           — hip facing: perp to hip axis (radians)
            shoulder_width,      — distance between shoulders (meters)
            neck_z,              — neck height (meters, posture indicator)
            torso_lean,          — forward lean angle (radians, 0=upright, >0=leaning forward)
            stance_width,        — distance between ankles (meters, wide=bracing for COD)
    """
    wide = _pivot_skeleton(skeleton)

    # --- Position (pelvis) ---
    wide["x"] = wide["pelvis_x"].astype(np.float64)
    wide["y"] = wide["pelvis_y"].astype(np.float64)

    # --- Head angle: nose -> ear midpoint (primary) ---
    mid_ear_x = (wide["l_ear_x"] + wide["r_ear_x"]) / 2
    mid_ear_y = (wide["l_ear_y"] + wide["r_ear_y"]) / 2
    wide["head_angle"] = np.arctan2(
        wide["nose_y"] - mid_ear_y,
        wide["nose_x"] - mid_ear_x,
    ).values

    # --- Head angle: perpendicular to ear axis (Bekkers style) ---
    wide["head_angle_perp"] = _perp_angle(
        wide["l_ear_x"].values, wide["l_ear_y"].values,
        wide["r_ear_x"].values, wide["r_ear_y"].values,
    )

    # --- Shoulder facing ---
    wide["shoulder_angle"] = _perp_angle(
        wide["l_shoulder_x"].values, wide["l_shoulder_y"].values,
        wide["r_shoulder_x"].values, wide["r_shoulder_y"].values,
    )

    # --- Hip facing ---
    wide["hip_angle"] = _perp_angle(
        wide["l_hip_x"].values, wide["l_hip_y"].values,
        wide["r_hip_x"].values, wide["r_hip_y"].values,
    )

    # --- Shoulder width ---
    wide["shoulder_width"] = np.sqrt(
        (wide["r_shoulder_x"] - wide["l_shoulder_x"]) ** 2 +
        (wide["r_shoulder_y"] - wide["l_shoulder_y"]) ** 2
    ).values

    # --- Neck height (posture indicator) ---
    # .values strips MultiIndex column metadata from pivot; needed for flat assignment
    wide["neck_z"] = wide["neck_z"].values

    # --- Torso lean: angle of neck-pelvis vector relative to vertical ---
    # 0 = perfectly upright, positive = leaning forward (neck ahead of pelvis)
    neck_pelvis_horiz = np.sqrt(
        (wide["neck_x"] - wide["pelvis_x"]) ** 2 +
        (wide["neck_y"] - wide["pelvis_y"]) ** 2
    )
    neck_pelvis_vert = wide["neck_z"] - wide["pelvis_z"]
    wide["torso_lean"] = np.arctan2(neck_pelvis_horiz, neck_pelvis_vert).values

    # --- Stance width (ankle-to-ankle distance) ---
    wide["stance_width"] = np.sqrt(
        (wide["r_ankle_x"] - wide["l_ankle_x"]) ** 2 +
        (wide["r_ankle_y"] - wide["l_ankle_y"]) ** 2
    ).values

    # --- Select output columns ---
    cols = ["frame_number", "team", "jersey", "x", "y",
            "head_angle", "head_angle_perp", "shoulder_angle", "hip_angle",
            "shoulder_width", "neck_z", "torso_lean", "stance_width"]
    result = wide[cols].copy()

    # --- Smoothing (per player, preserves real peaks) ---
    if smooth:
        angle_cols = ["head_angle", "head_angle_perp", "shoulder_angle", "hip_angle"]
        scalar_cols = ["x", "y", "torso_lean", "stance_width"]

        # Sort by frame within each player to ensure temporal order for smoothing
        result = result.sort_values(["team", "jersey", "frame_number"])
        for _, grp in result.groupby(["team", "jersey"], sort=False):
            idx = grp.index
            if len(idx) < 7:
                continue
            for col in angle_cols:
                result.loc[idx, col] = _smooth_angle_series(grp[col].values)
            for col in scalar_cols:
                result.loc[idx, col] = _smooth_scalar_series(grp[col].values)

    # --- Optimize dtypes ---
    result["team"] = result["team"].astype(np.int8)
    result["jersey"] = result["jersey"].astype(np.int8)
    for col in cols[3:]:
        result[col] = result[col].astype(np.float32)

    return result.sort_values(["frame_number", "team", "jersey"]).reset_index(drop=True)


# --- Speed and derived metrics --------------------------------------------

def add_dynamics(orientations: pd.DataFrame, framerate: int = 50) -> pd.DataFrame:
    """Add velocity, movement direction, body openness, and angular velocities.

    Args:
        orientations: Output of compute_orientations()
        framerate: Skeleton framerate (default 50Hz)

    Returns:
        Same DataFrame with added columns:
            vx, vy, speed,           — velocity (m/s) from pelvis central diff
            move_angle,              — movement direction (radians, from velocity vector)
            head_body_offset,        — |head_angle - shoulder_angle| in [0, pi]
            shoulder_hip_offset,     — |shoulder_angle - hip_angle| in [0, pi]
            body_move_offset,        — |shoulder_angle - move_angle| in [0, pi]
                                       high = backpedaling/side-shuffle
            head_angular_vel,        — head rotation speed (rad/s, unsigned)
            shoulder_angular_vel,    — shoulder rotation speed (rad/s, unsigned)
    """
    df = orientations.sort_values(["team", "jersey", "frame_number"]).copy()
    dt = 1.0 / framerate

    grouped = df.groupby(["team", "jersey"], sort=False)

    # Velocity from position (smoothed central difference via np.gradient)
    df["vx"] = grouped["x"].transform(lambda s: np.gradient(s.values, dt)).astype(np.float32)
    df["vy"] = grouped["y"].transform(lambda s: np.gradient(s.values, dt)).astype(np.float32)
    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2).astype(np.float32)

    # Movement direction (from velocity vector)
    df["move_angle"] = np.arctan2(df["vy"], df["vx"]).astype(np.float32)

    # Body openness metrics
    df["head_body_offset"] = angle_between(
        df["head_angle"].values, df["shoulder_angle"].values
    ).astype(np.float32)

    df["shoulder_hip_offset"] = angle_between(
        df["shoulder_angle"].values, df["hip_angle"].values
    ).astype(np.float32)

    # Movement vs body orientation (high = moving backwards/sideways relative to body)
    df["body_move_offset"] = angle_between(
        df["shoulder_angle"].values, df["move_angle"].values
    ).astype(np.float32)

    # Angular velocities (how fast head/shoulders rotate)
    def _angular_vel(angle_series):
        d = angle_diff(angle_series.values[1:], angle_series.values[:-1])
        vel = np.abs(np.concatenate([[0], d])) * framerate
        return vel

    df["head_angular_vel"] = grouped["head_angle"].transform(_angular_vel).astype(np.float32)
    df["shoulder_angular_vel"] = grouped["shoulder_angle"].transform(_angular_vel).astype(np.float32)

    return df.sort_values(["frame_number", "team", "jersey"]).reset_index(drop=True)
