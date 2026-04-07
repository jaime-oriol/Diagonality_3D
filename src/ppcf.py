"""
ppcf — Immediate Orientation-Aware PPCF via Reach Fields.

Per-player anisotropic Gaussian influence model derived directly from
biomechanics and real 3D skeleton orientation. Each player's blob is
shaped by the distance they can physically cover in the immediate
window, which in turn depends on the orientation-aware reaction delay
applied to their REAL shoulder angle from the skeleton keypoints.

For each player i and target cell x:
    theta_i(x) = |angle(shoulder_angle_i, direction(pos_i -> x))|
    delay_i(x) = rt(theta_i(x)) + cod(theta_i(x))
                  rt(θ) = base + gain * sigmoid((deg(θ) - c) / w)   # Vater 2024
                  cod(θ) = A * sin^2(θ / 2)                         # Dos'Santos 2018
    drift_i(x) = pos_i + vel_i * delay_i(x)                         # where player is when ready
    reach_i(x) = max(0, W - delay_i(x)) * vmax                      # remaining travel budget
    factor_i(x) = reach_i(x) / max_reach                             # [0, 1] orientation factor
    sigma_i(x) = sigma_base * (0.5 + 0.5 * factor_i(x))             # base ±50% modulation
    infl_i(x)  = exp(-||x - drift_i(x)||^2 / (2 * sigma_i(x)^2))    # in [0, 1]

Per-team control (aggregation):
    I_att(x) = sum_i infl_i(x) over attackers
    I_def(x) = sum_i infl_i(x) over defenders
    ppcf_att(x) = (1 - exp(-(I_att + I_def))) * I_att / (I_att + I_def)
    ppcf_def(x) = (1 - exp(-(I_att + I_def))) * I_def / (I_att + I_def)

The blob of each player is naturally anisotropic: stretched in the
direction they face (theta=0 -> low delay -> big reach -> big sigma)
and compressed in their blind spot (theta=pi -> big delay -> low reach ->
sigma compressed to 50% of base). A defender with the ball in his back has a
hole in his control field on that side — the diagonality thesis made
visible on a per-player basis.

Coordinate convention: TRACAB meters, centered at (0, 0).
All angles in RADIANS.

References:
  - Vater, C. (2024) Reaction time vs visual eccentricity. Sci Rep.
  - Dos'Santos, T. (2018) Change-of-direction deficit.       Sports Med.
  - Bekkers, J. (2026) SSAC Wide Open Gazes ("imminent" pitch control).
  - Spielverlagerung (2025) Tactical Theory: Diagonality.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


# ── Default parameters ────────────────────────────────────────────────────

def default_params(immediate_window: float = 2.5) -> dict:
    """Default parameters for the reach-field PPCF.

    Args:
        immediate_window: Integration horizon in seconds. Default 2.5.
            Calibrated so a well-oriented player (theta=0) has reach
            ~9m (sigma = sigma_base), while blind-spot directions
            (theta=pi) still have ~3m reach (sigma compressed to ~50%
            of sigma_base). Orientation modulates the SHAPE, not existence.
    """
    return {
        # Physical reach bounds
        "max_player_speed": 5.0,                # m/s
        "max_int_time": float(immediate_window),  # immediate window (s)

        # Orientation-aware reaction time (Vater 2024)
        # rt(theta) = rt_base + rt_gain * sigmoid((deg(theta) - c) / w)
        "rt_base": 0.7,
        "rt_gain": 0.4,
        "rt_theta_c_deg": 60.0,
        "rt_theta_w_deg": 15.0,

        # Change-of-direction deficit (Dos'Santos 2018)
        # cod(theta) = cod_amplitude * sin^2(theta / 2)
        "cod_amplitude": 0.8,

        # Blob geometry
        # sigma_base: base radius (meters) of each player's blob. 6.5m
        # produces blobs comparable to Bekkers' c_in=0.5 imminent PC.
        "blob_sigma_base": 6.5,
        # orientation_mod: how much orientation modulates the base radius.
        # 0.5 = ±50% — blind spot compresses to 50% of base, facing
        # direction stays at 100%. Gives ~3x influence ratio at 8m.
        "blob_orientation_mod": 0.5,
    }


# ── Biomechanical delay functions ─────────────────────────────────────────

def reaction_time(theta: np.ndarray, params: dict) -> np.ndarray:
    """Vater (2024) reaction time as a function of visual eccentricity.

    theta: unsigned angle (radians) between the player's facing and the
           direction to the stimulus. 0 = facing stimulus, pi = full blind.

    Returns seconds, preserving input shape.
    """
    theta_deg = np.degrees(theta)
    c = params["rt_theta_c_deg"]
    w = params["rt_theta_w_deg"]
    z = np.clip((theta_deg - c) / w, -50, 50)  # prevent overflow at extremes
    sig = 1.0 / (1.0 + np.exp(-z))
    return params["rt_base"] + params["rt_gain"] * sig


def cod_penalty(theta: np.ndarray, params: dict) -> np.ndarray:
    """Dos'Santos (2018) change-of-direction biomechanical delay.

    Extra seconds on top of reaction time to rotate hips and transition
    from current posture into motion in the target direction. Shaped as
    sin^2(theta / 2): 0 at theta=0, amplitude at theta=pi.
    """
    return params["cod_amplitude"] * np.sin(theta / 2.0) ** 2


def reaction_delay(theta: np.ndarray, params: dict) -> np.ndarray:
    """Total biomechanical delay = reaction_time + cod_penalty (seconds)."""
    return reaction_time(theta, params) + cod_penalty(theta, params)


# ── Geometry ──────────────────────────────────────────────────────────────

def _theta_player_to_targets(
    pos: np.ndarray,
    facing: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Unsigned angle (radians) between each player's facing and the
    direction from that player to each target cell.

    Shapes:
        pos:     (P, 2)
        facing:  (P,)
        targets: (N, 2)
    Returns: (P, N) in [0, pi].
    """
    if len(pos) == 0:
        return np.zeros((0, len(targets)))
    dx = targets[None, :, 0] - pos[:, None, 0]
    dy = targets[None, :, 1] - pos[:, None, 1]
    dir_to_target = np.arctan2(dy, dx)
    diff = dir_to_target - facing[:, None]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(diff)


# ── Per-player reach-field influence ──────────────────────────────────────

def _player_influence(
    pos: np.ndarray,
    vel: np.ndarray,
    facing: np.ndarray,
    targets: np.ndarray,
    params: dict,
    extra_delay: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-player anisotropic Gaussian influence over all target cells.

    For each (player, cell) pair:
        theta     = angle between player's shoulder facing and (player -> cell)
        delay     = rt(theta) + cod(theta) + extra_delay (if provided)
        drift     = pos + vel * delay             # where player is when ready
        reach     = max(0, W - delay) * vmax      # remaining travel budget
        sigma     = sigma_base * ((1-mod) + mod * reach/max_reach)  # base ±50%
        influence = exp(-dist(drift, cell)^2 / (2 * sigma^2))

    Args:
        extra_delay: Optional (P, N) or (P, 1) array of additional seconds
            added to each player's reaction delay. Used by dos.py to inject
            detection delays based on defender awareness. Default None = 0.

    Returns (P, N) influences in [0, 1] (peak = 1 at each player's drifted
    position). Shape (0, N) if P == 0.
    """
    P = len(pos)
    N = len(targets)
    if P == 0:
        return np.zeros((0, N))

    W = params["max_int_time"]
    vmax = params["max_player_speed"]

    # Theta per (player, cell) — drives the delay anisotropy
    theta = _theta_player_to_targets(pos, facing, targets)  # (P, N)

    # Orientation-aware reaction + COD delay per (player, cell)
    delay = reaction_delay(theta, params)  # (P, N)
    if extra_delay is not None:
        delay = delay + extra_delay

    # Drift during delay: each player continues current velocity until
    # they have reoriented and can start moving toward the target.
    drift_x = pos[:, 0:1] + vel[:, 0:1] * delay  # (P, N)
    drift_y = pos[:, 1:2] + vel[:, 1:2] * delay

    # Distance from drifted position to target
    dx = targets[None, :, 0] - drift_x
    dy = targets[None, :, 1] - drift_y
    dist = np.sqrt(dx * dx + dy * dy)

    # Reach budget (physical meters the player can cover after reacting)
    reach = np.maximum(0.0, W - delay) * vmax
    max_reach = (W - params["rt_base"]) * vmax  # best-case reach (theta=0)

    # Sigma: base radius modulated by orientation.
    # Every player has a substantial blob (sigma_base ~7m, like a normal
    # PPCF). Orientation modulates the shape by ±50%: facing forward
    # stretches to sigma_base, blind spot compresses to 50% of sigma_base.
    # This produces Bekkers-like blobs where the orientation DEFORMS the
    # shape rather than eliminating the back.
    sigma_base = params["blob_sigma_base"]
    orientation_mod = params["blob_orientation_mod"]
    orientation_factor = np.clip(reach / (max_reach + 1e-9), 0.0, 1.0)
    sigma = sigma_base * ((1 - orientation_mod) + orientation_mod * orientation_factor)

    # Anisotropic Gaussian
    return np.exp(-(dist * dist) / (2.0 * sigma * sigma))


# ── Frame extraction ──────────────────────────────────────────────────────

def _extract_team_arrays(
    frame_df: pd.DataFrame,
    team_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (pos, vel, facing) arrays for one team from a single-frame
    orientations DataFrame.

    Expected columns: team, x, y, shoulder_angle, [vx, vy].
    """
    df = frame_df[frame_df["team"] == team_id]
    if len(df) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    pos = df[["x", "y"]].values.astype(np.float64)

    if "vx" in df.columns and "vy" in df.columns:
        vx = df["vx"].fillna(0.0).values.astype(np.float64)
        vy = df["vy"].fillna(0.0).values.astype(np.float64)
        vel = np.column_stack([vx, vy])
    else:
        vel = np.zeros_like(pos)

    if "shoulder_angle" in df.columns:
        facing = df["shoulder_angle"].fillna(0.0).values.astype(np.float64)
    else:
        facing = np.zeros(len(df))

    return pos, vel, facing


# ── PPCF aggregation (per-team normalized shares) ─────────────────────────

def _aggregate_team_shares(
    I_att: np.ndarray,
    I_def: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw influence sums into ppcf shares.

    Interpretation:
        resolved    = 1 - exp(-(I_att + I_def))        # [0, 1]
        ppcf_att    = resolved * I_att / (I_att + I_def)
        ppcf_def    = resolved * I_def / (I_att + I_def)

    Properties:
        0 <= ppcf_att, ppcf_def
        ppcf_att + ppcf_def = resolved <= 1
        (1 - resolved) is the "unresolved" mass: cells no player reaches.
    """
    total = I_att + I_def
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_att = np.where(total > 1e-12, I_att / total, 0.0)
        frac_def = np.where(total > 1e-12, I_def / total, 0.0)
    resolved = 1.0 - np.exp(-total)
    return resolved * frac_att, resolved * frac_def


# ── Public API ────────────────────────────────────────────────────────────

def compute_ppcf_surfaces(
    orientations_frame: pd.DataFrame,
    attacking_team: int,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Immediate Orientation-Aware PPCF reach-field surfaces.

    Args:
        orientations_frame: Single-frame slice of an orientations DataFrame
            (as produced by orientation.compute_orientations + add_dynamics).
            Must contain columns: team, x, y, shoulder_angle. Velocity
            columns (vx, vy) are optional but strongly recommended.
        attacking_team: Team ID of the attacking team (0 or 1).
        params: Parameters from default_params(). Default if None.
        n_grid_x: Grid resolution in the longitudinal direction. Default 50.

    Returns:
        (ppcf_att, ppcf_def, xgrid, ygrid):
            ppcf_att, ppcf_def: (n_grid_y, n_grid_x) arrays in [0, 1]
                representing each team's share of resolved control. Their
                sum is `resolved` = fraction of the cell where any player
                has influence; 1 - sum is unresolved (nobody reaches).
            xgrid, ygrid: (n_grid_x,), (n_grid_y,) cell centers in meters.
    """
    if params is None:
        params = default_params()

    required = {"team", "x", "y", "shoulder_angle"}
    missing = required - set(orientations_frame.columns)
    if missing:
        raise ValueError(f"orientations_frame missing columns: {sorted(missing)}")

    # Grid (centered meters)
    n_grid_y = int(round(n_grid_x * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / n_grid_x
    dy = PITCH_WIDTH / n_grid_y
    xgrid = np.arange(n_grid_x) * dx - PITCH_LENGTH / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([xx.ravel(), yy.ravel()])

    defending_team = 1 - attacking_team
    att_pos, att_vel, att_facing = _extract_team_arrays(orientations_frame, attacking_team)
    def_pos, def_vel, def_facing = _extract_team_arrays(orientations_frame, defending_team)

    att_infl = _player_influence(att_pos, att_vel, att_facing, targets, params)
    def_infl = _player_influence(def_pos, def_vel, def_facing, targets, params)

    I_att_flat = att_infl.sum(axis=0) if len(att_infl) else np.zeros(len(targets))
    I_def_flat = def_infl.sum(axis=0) if len(def_infl) else np.zeros(len(targets))

    ppcf_att_flat, ppcf_def_flat = _aggregate_team_shares(I_att_flat, I_def_flat)

    return (
        ppcf_att_flat.reshape(n_grid_y, n_grid_x),
        ppcf_def_flat.reshape(n_grid_y, n_grid_x),
        xgrid,
        ygrid,
    )


def ppcf_at_targets(
    orientations_frame: pd.DataFrame,
    targets_xy: np.ndarray,
    attacking_team: int,
    params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """PPCF shares at specific target positions (point-wise API).

    Faster than compute_ppcf_surfaces for a handful of targets (e.g. pass
    options, receiver positions, DOS sampling at pass destinations).

    Args:
        orientations_frame: Single-frame orientations slice.
        targets_xy: (N, 2) positions in meters. Accepts 1D (2,) as a
            shortcut for a single point.
        attacking_team: Team ID (0 or 1).
        params: Default if None.

    Returns:
        (ppcf_att, ppcf_def) each shape (N,) in [0, 1]. See
        compute_ppcf_surfaces for semantics.
    """
    if params is None:
        params = default_params()

    targets = np.asarray(targets_xy, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[None, :]

    defending_team = 1 - attacking_team
    att_pos, att_vel, att_facing = _extract_team_arrays(orientations_frame, attacking_team)
    def_pos, def_vel, def_facing = _extract_team_arrays(orientations_frame, defending_team)

    att_infl = _player_influence(att_pos, att_vel, att_facing, targets, params)
    def_infl = _player_influence(def_pos, def_vel, def_facing, targets, params)

    I_att = att_infl.sum(axis=0) if len(att_infl) else np.zeros(len(targets))
    I_def = def_infl.sum(axis=0) if len(def_infl) else np.zeros(len(targets))

    return _aggregate_team_shares(I_att, I_def)
