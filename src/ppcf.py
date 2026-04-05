"""
ppcf - Immediate Orientation-Aware Pitch Control.

Extends Spearman (2018) PPCF (vectorized port, LaurieOnTracking-style) with:
  1. Immediate window: integration truncated to ~1s from NOW (not asymptotic).
     Equivalent in spirit to Bekkers' c_in ("imminent control"): who dominates
     the cell in the very short term, not at infinity.
  2. Orientation-aware reaction time per (player, target cell):
        rt(theta) = rt_base + rt_gain * sigmoid((deg(theta) - c) / w)
     From Vater (2024): reaction time grows monotonically with visual
     eccentricity up to ~90 degrees. Calibrated as 0.7s at theta=0, 1.1s at
     theta>=180.
  3. Change-of-direction penalty per (player, target cell):
        cod(theta) = A * sin^2(theta/2)
     From Dos'Santos (2018): biomechanical delay to turn hips before running
     in a new direction. ~0 at theta=0, ~0.4s at theta=90, ~0.8s at theta=180.
  4. Total reaction delay per (player, target) = rt(theta) + cod(theta).
     Applied symmetrically to both teams — theta is the angle between the
     player's shoulder facing (from 3D skeleton) and the direction from the
     player to the target cell.

Core algorithm is vectorized numpy: all grid targets evaluated in one pass.
Mathematically identical to per-target Euler integration (Spearman Eq 2-4),
extended so that each (player, target) pair has its own reaction delay.

Coordinate convention: TRACAB meters, centered at (0, 0).
Grid: 105 x 68 m, configurable resolution.
All angles in RADIANS.

Reference: Spearman (2018) "Beyond Expected Goals"
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


# --- Default parameters ---------------------------------------------------

def default_params(immediate_window: float = 1.0) -> dict:
    """Return default Immediate Orientation-Aware PPCF parameters.

    Args:
        immediate_window: Integration horizon in seconds (default 1.0).
            Controls how "immediate" the pitch control is. 1s is calibrated
            to the upper bound of orientation-induced reaction delay
            (Vater RT + Dos'Santos COD at theta=180 ~= 1.9s, so within 1s
            a defender with ball in blind spot barely reacts).

    Returns:
        Dict with all model parameters. Spearman core params (max_player_speed,
        tti_sigma, lambdas, ball speed) are identical to the Opta Forum
        reference implementation. Orientation params (rt_*, cod_*) are new.
    """
    params = {
        # Spearman 2018 core
        "max_player_speed": 5.0,           # m/s
        "tti_sigma": 0.45,                 # arrival-time uncertainty (s)
        "kappa_def": 1.0,                  # defending advantage factor
        "lambda_att": 4.3,                 # attacking ball-control rate
        "average_ball_speed": 15.0,        # m/s
        "int_dt": 0.04,                    # integration timestep (s)
        "max_int_time": immediate_window,  # IMMEDIATE horizon (s)
        "model_converge_tol": 0.01,        # convergence at sum PPCF > 0.99
        "time_to_control_veto": 3,         # ignore players with p < 10^-veto

        # GK boost: Spearman's asymptotic model uses 3.0 (GK uses hands, so
        # once they reach the ball they grab it instantly). That's an
        # asymptotic trick. In IMMEDIATE mode the GK has the same legs and
        # the same reaction time as a field player; the 3x boost massively
        # and unphysically inflates their territory. Default to 1.0 (no
        # boost) for immediate; override if you want asymptotic behavior.
        "lambda_gk_boost": 1.0,

        # Orientation-aware reaction time (Vater 2024)
        "rt_base": 0.7,                    # baseline RT (s)
        "rt_gain": 0.4,                    # max additional RT at theta=pi (s)
        "rt_theta_c_deg": 60.0,            # sigmoid center (degrees)
        "rt_theta_w_deg": 15.0,            # sigmoid width (degrees)

        # Change-of-direction penalty (Dos'Santos 2018)
        "cod_amplitude": 0.8,              # COD penalty at theta=pi (s)
    }
    params["lambda_def"] = params["lambda_att"] * params["kappa_def"]
    params["lambda_gk"] = params["lambda_def"] * params["lambda_gk_boost"]

    # Spearman dominance thresholds (used for shortcut veto)
    veto = params["time_to_control_veto"]
    sigma_term = np.sqrt(3) * params["tti_sigma"] / np.pi
    params["time_to_control_att"] = (
        veto * np.log(10) * (sigma_term + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        veto * np.log(10) * (sigma_term + 1 / params["lambda_def"])
    )
    return params


# --- Orientation-aware delay functions ------------------------------------

def reaction_time(theta: np.ndarray, params: dict) -> np.ndarray:
    """Vater-style reaction time as a function of visual eccentricity.

    theta: unsigned angle (radians) between player facing and direction to
           target. theta=0 means facing directly at target.

    Returns RT in seconds. Shape preserved from input.
    """
    theta_deg = np.degrees(theta)
    c = params["rt_theta_c_deg"]
    w = params["rt_theta_w_deg"]
    # Clip sigmoid argument to prevent overflow at extreme angles
    z = np.clip((theta_deg - c) / w, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-z))
    return params["rt_base"] + params["rt_gain"] * sig


def cod_penalty(theta: np.ndarray, params: dict) -> np.ndarray:
    """Dos'Santos-style change-of-direction biomechanical penalty.

    theta: unsigned angle (radians). 0 means no turn needed, pi means 180 deg.

    Returns additional delay (s) on top of reaction time, due to
    hip/shoulder rotation and lateral-to-linear acceleration transition.
    """
    return params["cod_amplitude"] * np.sin(theta / 2.0) ** 2


def reaction_delay(theta: np.ndarray, params: dict) -> np.ndarray:
    """Total per-(player,target) delay = RT + COD penalty."""
    return reaction_time(theta, params) + cod_penalty(theta, params)


# --- Geometry helpers -----------------------------------------------------

def _theta_player_to_targets(
    pos: np.ndarray,
    facing: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Angle between each player's facing and direction to each target.

    Args:
        pos: (P, 2) player positions in meters.
        facing: (P,) shoulder facing angles in radians.
        targets: (N, 2) target positions in meters.

    Returns:
        (P, N) array of unsigned angles in [0, pi].
    """
    if len(pos) == 0:
        return np.zeros((0, len(targets)))
    dx = targets[None, :, 0] - pos[:, None, 0]
    dy = targets[None, :, 1] - pos[:, None, 1]
    dir_to_target = np.arctan2(dy, dx)
    diff = dir_to_target - facing[:, None]
    diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(diff_wrapped)


def _player_tti(
    pos: np.ndarray,
    vel: np.ndarray,
    facing: np.ndarray,
    targets: np.ndarray,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-(player, target) time-to-intercept with orientation-aware delay.

    Returns:
        tti: (P, N) time in seconds from t=0 for each player to reach each
             target cell, accounting for reaction + COD delay + travel.
        delay: (P, N) the (rt + cod) component only (for debug/inspection).
    """
    if len(pos) == 0:
        return np.zeros((0, len(targets))), np.zeros((0, len(targets)))
    vmax = params["max_player_speed"]
    theta = _theta_player_to_targets(pos, facing, targets)       # (P, N)
    delay = reaction_delay(theta, params)                        # (P, N)
    # Position after reaction delay (player assumed to continue current vel
    # during the delay; no teleportation, same assumption as Spearman).
    r_pos = pos[:, None, :] + vel[:, None, :] * delay[:, :, None]  # (P, N, 2)
    diff = targets[None, :, :] - r_pos                           # (P, N, 2)
    dist = np.linalg.norm(diff, axis=2)                          # (P, N)
    tti = delay + dist / vmax                                    # (P, N)
    return tti, delay


# --- Core vectorized PPCF -------------------------------------------------

def _ppcf_vectorized_surfaces(
    targets: np.ndarray,
    att_pos: np.ndarray, att_vel: np.ndarray, att_gk: np.ndarray, att_facing: np.ndarray,
    def_pos: np.ndarray, def_vel: np.ndarray, def_gk: np.ndarray, def_facing: np.ndarray,
    ball_pos: Optional[np.ndarray],
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Immediate Orientation-Aware PPCF, returning BOTH surfaces.

    Returns (ppcf_att, ppcf_def), each shape (N,). The key property of the
    immediate model is that ppcf_att + ppcf_def can be < 1: the remainder
    is UNRESOLVED mass (nobody dominates the cell within the window). The
    viz uses resolved = ppcf_att + ppcf_def as alpha, so unresolved cells
    fade to the pitch background — which is exactly the semantic difference
    versus the asymptotic Opta/Spearman model where every cell is resolved.

    Integration variable: t in [0, max_int_time], where t=0 is NOW (the
    current frame). A player can only contribute to control after their
    per-target TTI, AND after the ball has arrived at the target.
    """
    N = len(targets)
    Pa, Pd = len(att_pos), len(def_pos)

    # Degenerate: both teams empty -> nothing resolves
    if Pa == 0 and Pd == 0:
        return np.zeros(N), np.zeros(N)

    vmax = params["max_player_speed"]
    sigma = params["tti_sigma"]
    lam_a = params["lambda_att"]
    lam_d = params["lambda_def"]
    lam_gk = params["lambda_gk"]
    dt = params["int_dt"]
    max_int = params["max_int_time"]
    tol = params["model_converge_tol"]
    tc_a = params["time_to_control_att"]
    tc_d = params["time_to_control_def"]
    sig_c = np.pi / np.sqrt(3.0) / sigma

    # Ball travel time to each target (s)
    if ball_pos is not None and not np.any(np.isnan(ball_pos)):
        ball_tt = np.linalg.norm(targets - ball_pos, axis=1) / params["average_ball_speed"]
    else:
        ball_tt = np.zeros(N)

    # Per-(player, target) time-to-intercept (orientation-aware)
    att_tti, _ = _player_tti(att_pos, att_vel, att_facing, targets, params)
    def_tti, _ = _player_tti(def_pos, def_vel, def_facing, targets, params)

    att_min = att_tti.min(axis=0) if Pa else np.full(N, np.inf)
    def_min = def_tti.min(axis=0) if Pd else np.full(N, np.inf)

    # Pre-allocate output surfaces
    res_att = np.zeros(N)
    res_def = np.zeros(N)

    # Spearman dominance shortcut (Opta-style). Uses BALL-RELATIVE TTI:
    # tc_a and tc_d are thresholds for the tail of the sigmoid. Cells where
    # one team arrives much earlier than the other, relative to ball
    # arrival, are assigned directly without running Euler.
    done_mask = np.zeros(N, dtype=bool)
    def_dom = (att_min - np.maximum(ball_tt, def_min)) >= tc_d
    att_dom = (def_min - np.maximum(ball_tt, att_min)) >= tc_a
    res_def[def_dom] = 1.0
    done_mask |= def_dom
    pure_att = att_dom & ~def_dom
    res_att[pure_att] = 1.0
    done_mask |= pure_att

    # One-team-empty degenerate cases
    if Pa == 0:
        can = ~done_mask
        res_def[can] = 1.0
        return res_att, res_def
    if Pd == 0:
        can = ~done_mask
        res_att[can] = 1.0
        return res_att, res_def

    eidx = np.where(~done_mask)[0]
    if len(eidx) == 0:
        return res_att, res_def

    # --- Euler integration (BALL-RELATIVE time) ---
    # Integration variable tau = t - ball_tt: "time elapsed since ball
    # arrival at each target". Window is [0, max_int] measured from ball
    # arrival. This is the Opta/Spearman formulation; our only change is
    # the short max_int + per-(player, target) orientation-aware TTI in the
    # player delays. Orientation snapshot frozen at t=0 (current frame),
    # used as a "what-if" capacity map over the whole pitch.
    M = len(eidx)
    e_btt = ball_tt[eidx]                     # (M,)
    e_att = att_tti[:, eidx]                  # (Pa, M)
    e_def = def_tti[:, eidx]                  # (Pd, M)

    # TTI relative to ball arrival: if negative, player already there.
    att_tti_rel = e_att - e_btt[None, :]      # (Pa, M)
    def_tti_rel = e_def - e_btt[None, :]      # (Pd, M)

    # Active player masks: skip players much slower than team best
    a_act = (e_att - att_min[eidx]) < tc_a    # (Pa, M)
    d_act = (e_def - def_min[eidx]) < tc_d    # (Pd, M)

    d_lam = np.where(def_gk, lam_gk, lam_d)[:, None]  # (Pd, 1)

    pa_cum = np.zeros((Pa, M))
    pd_cum = np.zeros((Pd, M))
    tot_a = np.zeros(M)
    tot_d = np.zeros(M)
    conv = np.zeros(M, dtype=bool)

    for tau in np.arange(0.0, max_int, dt):
        if conv.all():
            break
        live = ~conv
        rem = 1.0 - tot_a - tot_d

        with np.errstate(over="ignore"):
            p_a = 1.0 / (1.0 + np.exp(-sig_c * (tau - att_tti_rel)))
            p_d = 1.0 / (1.0 + np.exp(-sig_c * (tau - def_tti_rel)))

        pa_cum += (rem * p_a * lam_a * a_act * live) * dt
        pd_cum += (rem * p_d * d_lam * d_act * live) * dt
        tot_a = pa_cum.sum(axis=0)
        tot_d = pd_cum.sum(axis=0)
        conv = (tot_a + tot_d) > (1.0 - tol)

    res_att[eidx] = tot_a
    res_def[eidx] = tot_d
    return res_att, res_def


def _ppcf_vectorized(
    targets: np.ndarray,
    att_pos: np.ndarray, att_vel: np.ndarray, att_gk: np.ndarray, att_facing: np.ndarray,
    def_pos: np.ndarray, def_vel: np.ndarray, def_gk: np.ndarray, def_facing: np.ndarray,
    ball_pos: Optional[np.ndarray],
    params: dict,
) -> np.ndarray:
    """Backward-compat wrapper returning only the attacker surface.

    Kept for the existing test suite. New callers should prefer
    _ppcf_vectorized_surfaces() which returns both att and def, enabling
    correct "unresolved cell" handling in visualization (alpha from
    resolved mass).
    """
    att, _ = _ppcf_vectorized_surfaces(
        targets,
        att_pos, att_vel, att_gk, att_facing,
        def_pos, def_vel, def_gk, def_facing,
        ball_pos, params,
    )
    return att


# --- Frame extraction -----------------------------------------------------

def _extract_team_arrays(
    frame_df: pd.DataFrame,
    team_id: int,
    gk_jerseys: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull (pos, vel, is_gk, facing) arrays for one team from a frame."""
    df = frame_df[frame_df["team"] == team_id]
    if len(df) == 0:
        return (np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros(0, dtype=bool), np.zeros(0))

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

    gk_j = gk_jerseys.get(team_id, 1)
    is_gk = (df["jersey"].values == gk_j)

    return pos, vel, is_gk, facing


# --- Public API -----------------------------------------------------------

def compute_ppcf(
    orientations_frame: pd.DataFrame,
    attacking_team: int,
    ball_xy: Tuple[float, float],
    gk_jerseys: Optional[Dict[int, int]] = None,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Immediate Orientation-Aware PPCF surface for one frame.

    Args:
        orientations_frame: Single-frame slice of an orientations DataFrame
            (as produced by orientation.compute_orientations + add_dynamics).
            Must contain columns: team, jersey, x, y, shoulder_angle.
            Velocity columns (vx, vy) optional but strongly recommended.
        attacking_team: Team ID of the attacking team (0 or 1).
        ball_xy: (x, y) ball position in meters, centered pitch.
        gk_jerseys: {team_id: gk_jersey_number}. Default {0:1, 1:1}.
        params: Model parameters from default_params(). Default if None.
        n_grid_x: Grid resolution in the longitudinal direction. Default 50.

    Returns:
        (ppcf_att, xgrid, ygrid):
            ppcf_att: (n_grid_y, n_grid_x) array in [0, 1].
            xgrid: (n_grid_x,) longitudinal cell centers in meters.
            ygrid: (n_grid_y,) lateral cell centers in meters.
    """
    if params is None:
        params = default_params()
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    required = {"team", "jersey", "x", "y", "shoulder_angle"}
    missing = required - set(orientations_frame.columns)
    if missing:
        raise ValueError(f"orientations_frame missing columns: {missing}")

    # Grid (centered meters)
    n_grid_y = int(round(n_grid_x * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / n_grid_x
    dy = PITCH_WIDTH / n_grid_y
    xgrid = np.arange(n_grid_x) * dx - PITCH_LENGTH / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([xx.ravel(), yy.ravel()])

    defending_team = 1 - attacking_team
    att = _extract_team_arrays(orientations_frame, attacking_team, gk_jerseys)
    def_ = _extract_team_arrays(orientations_frame, defending_team, gk_jerseys)

    ppcf_flat = _ppcf_vectorized(
        targets,
        att[0], att[1], att[2], att[3],
        def_[0], def_[1], def_[2], def_[3],
        np.asarray(ball_xy, dtype=np.float64),
        params,
    )
    return ppcf_flat.reshape(n_grid_y, n_grid_x), xgrid, ygrid


def compute_ppcf_surfaces(
    orientations_frame: pd.DataFrame,
    attacking_team: int,
    ball_xy: Tuple[float, float],
    gk_jerseys: Optional[Dict[int, int]] = None,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute BOTH Immediate PPCF surfaces (attacker and defender) for one
    frame.

    Unlike compute_ppcf(), this returns the two raw probability mass
    surfaces separately so that downstream viz can distinguish
    'defender dominates' (ppcf_def ~ 1) from 'nobody resolves control in
    the window' (ppcf_att + ppcf_def ~ 0). The sum ppcf_att + ppcf_def
    is the 'resolved mass' in [0, 1]; 1 - (att + def) is unresolved
    (attacker neither wins nor loses in the immediate horizon).

    Returns:
        (ppcf_att, ppcf_def, xgrid, ygrid), each grid shape (n_grid_y, n_grid_x).
    """
    if params is None:
        params = default_params()
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    required = {"team", "jersey", "x", "y", "shoulder_angle"}
    missing = required - set(orientations_frame.columns)
    if missing:
        raise ValueError(f"orientations_frame missing columns: {missing}")

    n_grid_y = int(round(n_grid_x * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / n_grid_x
    dy = PITCH_WIDTH / n_grid_y
    xgrid = np.arange(n_grid_x) * dx - PITCH_LENGTH / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([xx.ravel(), yy.ravel()])

    defending_team = 1 - attacking_team
    att = _extract_team_arrays(orientations_frame, attacking_team, gk_jerseys)
    def_ = _extract_team_arrays(orientations_frame, defending_team, gk_jerseys)

    att_flat, def_flat = _ppcf_vectorized_surfaces(
        targets,
        att[0], att[1], att[2], att[3],
        def_[0], def_[1], def_[2], def_[3],
        np.asarray(ball_xy, dtype=np.float64),
        params,
    )
    return (
        att_flat.reshape(n_grid_y, n_grid_x),
        def_flat.reshape(n_grid_y, n_grid_x),
        xgrid, ygrid,
    )


def ppcf_at_targets(
    orientations_frame: pd.DataFrame,
    targets_xy: np.ndarray,
    attacking_team: int,
    ball_xy: Tuple[float, float],
    gk_jerseys: Optional[Dict[int, int]] = None,
    params: Optional[dict] = None,
) -> np.ndarray:
    """Compute PPCF_att at specific target positions (point-wise).

    Much faster than compute_ppcf for a handful of targets (e.g. pass
    options, receiver positions, DOS sampling).

    Args:
        targets_xy: (N, 2) array of positions in meters.

    Returns:
        (N,) PPCF_att in [0, 1].
    """
    if params is None:
        params = default_params()
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    targets = np.asarray(targets_xy, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[None, :]

    defending_team = 1 - attacking_team
    att = _extract_team_arrays(orientations_frame, attacking_team, gk_jerseys)
    def_ = _extract_team_arrays(orientations_frame, defending_team, gk_jerseys)

    return _ppcf_vectorized(
        targets,
        att[0], att[1], att[2], att[3],
        def_[0], def_[1], def_[2], def_[3],
        np.asarray(ball_xy, dtype=np.float64),
        params,
    )
