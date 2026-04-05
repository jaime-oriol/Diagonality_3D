"""
ppcf — Immediate Orientation-Aware Pitch Control Function.

Extends Spearman (2018) PPCF with real per-player body orientation from
3D skeleton data. First implementation that modulates reaction time by
measured biomechanics (head, shoulder, hip angles) instead of proxying
orientation from the velocity vector.

Immediate variant follows Bekkers (SSAC 2026) PC^c_in: scales effective
max_player_speed by c_in to capture "reachable in a very short time".
Default c_in=0.5 (Bekkers).

Five mechanisms modulate reaction time per player and per target cell:

  1. theta penalty (sigmoid on target-shoulder angle)
     Vater (2024 Sci Rep): reaction time grows monotonically with visual
     eccentricity, up to ~0.4s at 90 deg. Sigmoid with midpoint at 60 deg
     captures the transition between "seen" and "blind".

  2. twist penalty (shoulder-hip offset at t0)
     Spielverlagerung (2025): turning hips to respect a diagonal runner
     adds 0.12-0.18s vs straight chase. Applied as linear ramp above 20 deg
     twist, saturating at 60 deg.

  3. backpedal penalty (body_move_offset > 90 deg AND speed > 2 m/s)
     Dos'Santos (2018 Sports Med): 180 deg COD at moderate speed costs
     0.3-0.5s vs straight sprint. Applied only when backpedaling.

  4. scanning bonus (head_angular_vel > 200 deg/s)
     Vater (2024) + McGuckian (2018): active head scanning shortens
     reaction by gathering peripheral info early. Negative penalty.

  5. head-ahead bonus (head on target, shoulder lagging)
     Eye-head coordination (Fang 2015): if head already points at target
     but body doesn't, the perceptual part of reaction is already done.

All mechanisms apply to both defenders and attackers — the receiver's
orientation matters as much as the defender's (a receiver with body already
facing the action reacts faster to the ball arrival).

Core integration math is Spearman 2018 Eq 2-4, vectorized over targets.
With all modulation flags off and c_in=1.0, output is bit-identical to
Spearman's original PPCF (see test_ppcf_scalar_matches_spearman).

Ball travel time uses real 3D ball velocity from the parquet when the
ball is in motion; falls back to the constant average_ball_speed (15 m/s)
only for static balls.

Ballistic reaction: during the full (base + extra) reaction time, the
player continues at current velocity (momentum). After that, he races
to the target at vmax_eff. Position after reaction is computed per-target
because reaction time is per-target. This is physically correct and is
what Spearman assumes implicitly with his scalar rt.

Grid matches the vision grid resolution (smoothing_multiplier from
src/vision.py) so that PPCF ⊙ V_i is element-wise without resample.
Default smoothing=5 → 525x340 cells (0.2 m per cell).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# --- Parameters -----------------------------------------------------------

@dataclass
class ParamsOAPPCF:
    """Full parameter set for Immediate Orientation-Aware PPCF.

    Defaults are calibrated from literature. No parameter is fitted — all
    values are documented with their source so the model is fully auditable.
    """

    # --- Spearman (2018) base ------------------------------------------
    max_player_speed: float = 5.0           # m/s (Spearman)
    reaction_time_base: float = 0.7         # s irreducible (Vater 2024)
    tti_sigma: float = 0.45                 # TTI uncertainty (Spearman)
    lambda_att: float = 4.3                 # ball control rate attacker
    kappa_def: float = 1.0                  # defending advantage
    lambda_gk_mult: float = 3.0             # GK bonus
    average_ball_speed: float = 15.0        # m/s fallback
    int_dt: float = 0.02                    # 50 Hz native (doubles Opta's 0.04)
    max_int_time: float = 10.0              # s integration cap
    model_converge_tol: float = 0.005       # tighter than Opta 0.01
    time_to_control_veto: int = 3           # Spearman shortcut threshold

    # --- Immediate scaling (Bekkers PC^c_in) ---------------------------
    c_in: float = 0.5                       # imminent default (Bekkers)

    # --- Orientation modulation toggles --------------------------------
    modulate_defenders: bool = True
    modulate_attackers: bool = True
    use_ballistic_reaction: bool = True     # extend pos by vel during reaction

    # --- Theta penalty (target angle vs shoulder) -----------------------
    dt_theta_max: float = 0.40              # s at full blindside (Vater 2024)
    dt_theta_midpoint_deg: float = 60.0     # sigmoid inflection
    dt_theta_slope_deg: float = 15.0        # sigmoid slope

    # --- Torso twist penalty (shoulder-hip at t0) -----------------------
    dt_twist_max: float = 0.18              # s (Spielverlagerung 2025)
    twist_threshold_deg: float = 20.0
    twist_saturation_deg: float = 60.0

    # --- Backpedal penalty (body_move_offset > 90 deg AND moving) -------
    dt_backpedal_max: float = 0.30          # s (Dos'Santos 2018 90deg COD)
    backpedal_speed_threshold: float = 2.0  # m/s
    backpedal_angle_threshold_deg: float = 90.0

    # --- Scanning bonus (head_angular_vel) ------------------------------
    dt_scanning_max: float = 0.10           # s (Vater, McGuckian)
    scanning_angvel_threshold_degps: float = 200.0
    scanning_angvel_saturation_degps: float = 500.0

    # --- Head-ahead bonus (head on target, shoulder lagging) ------------
    dt_head_ahead_max: float = 0.08         # s (Fang 2015 eye-head coordination)
    head_ahead_threshold_deg: float = 30.0

    # --- Hard clips -----------------------------------------------------
    dt_extra_max: float = 0.80              # s biomech ceiling
    dt_extra_min: float = -0.15             # s bonus floor

    # Derived (computed in __post_init__)
    lambda_def: float = field(init=False)
    lambda_gk: float = field(init=False)
    time_to_control_att: float = field(init=False)
    time_to_control_def: float = field(init=False)

    def __post_init__(self):
        self.lambda_def = self.lambda_att * self.kappa_def
        self.lambda_gk = self.lambda_def * self.lambda_gk_mult
        sigma_term = np.sqrt(3) * self.tti_sigma / np.pi
        self.time_to_control_att = (
            self.time_to_control_veto * np.log(10)
            * (sigma_term + 1 / self.lambda_att)
        )
        self.time_to_control_def = (
            self.time_to_control_veto * np.log(10)
            * (sigma_term + 1 / self.lambda_def)
        )


def params_standard() -> ParamsOAPPCF:
    """Spearman-equivalent baseline: no orientation, no immediate scaling."""
    return ParamsOAPPCF(
        c_in=1.0,
        modulate_defenders=False,
        modulate_attackers=False,
        use_ballistic_reaction=False,
        int_dt=0.04,
        model_converge_tol=0.01,
    )


def params_immediate() -> ParamsOAPPCF:
    """Bekkers imminent, no orientation (pure replication)."""
    return ParamsOAPPCF(
        c_in=0.5,
        modulate_defenders=False,
        modulate_attackers=False,
    )


def params_oa() -> ParamsOAPPCF:
    """Full Immediate Orientation-Aware (the real deliverable)."""
    return ParamsOAPPCF()


# --- Reaction time modulation ---------------------------------------------

def _sigmoid_penalty(
    theta_rad: np.ndarray,
    dt_max: float,
    midpoint_deg: float,
    slope_deg: float,
) -> np.ndarray:
    """Smooth sigmoid penalty (seconds) as function of angular mismatch.

    theta in [0, pi]; output in [~0, dt_max]. Midpoint = 50% of dt_max.
    """
    theta_deg = np.degrees(theta_rad)
    return dt_max / (1.0 + np.exp(-(theta_deg - midpoint_deg) / slope_deg))


def _twist_penalty(twist_rad: np.ndarray, params: ParamsOAPPCF) -> np.ndarray:
    """Linear ramp from threshold to saturation, clipped."""
    deg = np.degrees(np.abs(twist_rad))
    span = params.twist_saturation_deg - params.twist_threshold_deg
    frac = np.clip((deg - params.twist_threshold_deg) / (span + 1e-9), 0, 1)
    return params.dt_twist_max * frac


def _backpedal_penalty(
    body_move_offset_rad: np.ndarray,
    speed: np.ndarray,
    params: ParamsOAPPCF,
) -> np.ndarray:
    """Active only if (speed > threshold) AND (body_move_offset > 90 deg)."""
    deg = np.degrees(np.abs(body_move_offset_rad))
    active = (
        (speed > params.backpedal_speed_threshold)
        & (deg > params.backpedal_angle_threshold_deg)
    )
    frac = np.clip((deg - 90.0) / 90.0, 0, 1)
    return np.where(active, params.dt_backpedal_max * frac, 0.0)


def _scanning_bonus(
    head_angvel_radps: np.ndarray, params: ParamsOAPPCF,
) -> np.ndarray:
    """Negative penalty: fast head rotation shortens reaction."""
    deg = np.degrees(np.abs(head_angvel_radps))
    span = (
        params.scanning_angvel_saturation_degps
        - params.scanning_angvel_threshold_degps
    )
    frac = np.clip(
        (deg - params.scanning_angvel_threshold_degps) / (span + 1e-9), 0, 1,
    )
    return -params.dt_scanning_max * frac


def _compute_dt_extra(
    pos: np.ndarray,                 # (P, 2)
    shoulder: np.ndarray,            # (P,)
    head: np.ndarray,                # (P,)
    twist: np.ndarray,               # (P,) shoulder_hip_offset
    body_move_offset: np.ndarray,    # (P,)
    speed: np.ndarray,               # (P,)
    head_angvel: np.ndarray,         # (P,)
    targets: np.ndarray,             # (N, 2)
    params: ParamsOAPPCF,
) -> np.ndarray:
    """Per-player per-target additional reaction time (can be negative).

    Output shape (P, N). Components (bonuses already signed negative):
        theta_penalty(shoulder vs target)       >= 0
      + twist_penalty(shoulder_hip_offset)      >= 0
      + backpedal_penalty(body_move_offset)     >= 0
      + scanning_bonus(head_angular_vel)        <= 0
      + head_ahead_bonus(head vs target)        <= 0
    Clipped to [dt_extra_min, dt_extra_max].
    """
    P, N = len(pos), len(targets)
    if P == 0:
        return np.zeros((0, N))

    # Angle from each player to each target
    dx = targets[None, :, 0] - pos[:, 0:1]
    dy = targets[None, :, 1] - pos[:, 1:2]
    target_angle = np.arctan2(dy, dx)  # (P, N)

    # --- 1. theta shoulder penalty ---
    diff_s = target_angle - shoulder[:, None]
    theta_s = np.abs(np.arctan2(np.sin(diff_s), np.cos(diff_s)))
    dt_theta = _sigmoid_penalty(
        theta_s, params.dt_theta_max,
        params.dt_theta_midpoint_deg, params.dt_theta_slope_deg,
    )

    # --- 2. theta head (for head-ahead bonus) ---
    diff_h = target_angle - head[:, None]
    theta_h = np.abs(np.arctan2(np.sin(diff_h), np.cos(diff_h)))

    # --- 5. head-ahead bonus (head sees target, body lagging) ---
    head_ahead_mask = (
        (theta_h < np.radians(params.head_ahead_threshold_deg))
        & (theta_s > np.radians(params.head_ahead_threshold_deg))
    )
    dt_head_ahead = -params.dt_head_ahead_max * head_ahead_mask.astype(float)

    # --- 3. twist (per-player, broadcast to (P, N)) ---
    dt_twist = _twist_penalty(twist, params)[:, None]

    # --- 4. backpedal (per-player) ---
    dt_back = _backpedal_penalty(body_move_offset, speed, params)[:, None]

    # --- 5. scanning bonus (per-player) ---
    dt_scan = _scanning_bonus(head_angvel, params)[:, None]

    dt_total = dt_theta + dt_twist + dt_back + dt_scan + dt_head_ahead
    return np.clip(dt_total, params.dt_extra_min, params.dt_extra_max)


# --- PPCF core (vectorized Spearman + per-player per-target rt) -----------

def _ppcf_core(
    targets: np.ndarray,              # (N, 2)
    att_pos: np.ndarray, att_vel: np.ndarray, att_gk: np.ndarray,
    att_dt_extra: np.ndarray,         # (Pa, N); zeros when modulation off
    def_pos: np.ndarray, def_vel: np.ndarray, def_gk: np.ndarray,
    def_dt_extra: np.ndarray,         # (Pd, N); zeros when modulation off
    ball_pos: Optional[np.ndarray],   # (2,) or None
    ball_vel: Optional[np.ndarray],   # (3,) or None
    params: ParamsOAPPCF,
) -> np.ndarray:
    """Core vectorized PPCF with per-player per-target reaction time.

    With zero dt_extra, c_in=1.0, and use_ballistic_reaction=False, this
    matches Spearman (2018) exactly.
    """
    N = len(targets)
    Pa, Pd = len(att_pos), len(def_pos)
    # Short-circuit: no players at all -> PPCF is 0 everywhere by convention
    if Pa == 0 and Pd == 0:
        return np.zeros(N)
    vmax = params.max_player_speed * params.c_in
    rt = params.reaction_time_base
    sigma = params.tti_sigma
    lam_a = params.lambda_att
    lam_d = params.lambda_def
    lam_gk = params.lambda_gk
    dt = params.int_dt
    max_int = params.max_int_time
    tol = params.model_converge_tol
    tc_a = params.time_to_control_att
    tc_d = params.time_to_control_def
    sig_c = np.pi / np.sqrt(3.0) / sigma

    # --- Ball travel time per target ---
    # If the ball is measurably in motion, use its real speed from the
    # parquet (3D skeleton gives us the true ball speed). Otherwise fall
    # back to the hypothetical-pass constant (Spearman average 15 m/s).
    if ball_pos is not None and not np.any(np.isnan(ball_pos)):
        ball_dist = np.linalg.norm(targets - ball_pos, axis=1)
        ball_speed_used = params.average_ball_speed
        if ball_vel is not None and not np.any(np.isnan(ball_vel)):
            bs = float(np.linalg.norm(np.asarray(ball_vel).ravel()[:2]))
            if bs > 1.0:
                ball_speed_used = bs
        ball_tt = ball_dist / ball_speed_used
    else:
        ball_tt = np.zeros(N)

    # --- Attacker TTI with per-(Pa, N) reaction ---
    if Pa > 0:
        rt_att = rt + att_dt_extra                     # (Pa, N)
        if params.use_ballistic_reaction:
            pos_after = (
                att_pos[:, None, :] + att_vel[:, None, :] * rt_att[:, :, None]
            )                                           # (Pa, N, 2)
            dist = np.linalg.norm(targets[None, :, :] - pos_after, axis=2)
        else:
            att_r = att_pos + att_vel * rt
            dist = np.linalg.norm(targets[None] - att_r[:, None], axis=2)
        att_tti = rt_att + dist / vmax                  # (Pa, N)
    else:
        att_tti = np.full((0, N), np.inf)

    # --- Defender TTI ---
    if Pd > 0:
        rt_def = rt + def_dt_extra
        if params.use_ballistic_reaction:
            pos_after = (
                def_pos[:, None, :] + def_vel[:, None, :] * rt_def[:, :, None]
            )
            dist = np.linalg.norm(targets[None, :, :] - pos_after, axis=2)
        else:
            def_r = def_pos + def_vel * rt
            dist = np.linalg.norm(targets[None] - def_r[:, None], axis=2)
        def_tti = rt_def + dist / vmax
    else:
        def_tti = np.full((0, N), np.inf)

    att_min = att_tti.min(axis=0) if Pa else np.full(N, np.inf)
    def_min = def_tti.min(axis=0) if Pd else np.full(N, np.inf)

    # --- Veto shortcuts (Spearman Eq 5) ---
    result = np.full(N, np.nan)
    def_dom = (att_min - np.maximum(ball_tt, def_min)) >= tc_d
    att_dom = (def_min - np.maximum(ball_tt, att_min)) >= tc_a
    result[def_dom] = 0.0
    result[att_dom & ~def_dom] = 1.0

    eidx = np.where(np.isnan(result))[0]
    if len(eidx) == 0:
        return result

    # --- Euler integration for contested cells ---
    # Guard: with no players at all, nothing to integrate.
    if Pa == 0 and Pd == 0:
        result[eidx] = 0.0
        return result

    M = len(eidx)
    e_btt = ball_tt[eidx]
    e_att = att_tti[:, eidx]
    e_def = def_tti[:, eidx]

    a_act = (e_att - att_min[eidx]) < tc_a
    d_act = (e_def - def_min[eidx]) < tc_d

    d_lam = np.where(def_gk, lam_gk, lam_d)[:, None] if Pd else np.zeros((0, M))

    att_tti_rel = e_att - e_btt[None, :]
    def_tti_rel = e_def - e_btt[None, :]

    pa_cum = np.zeros((Pa, M))
    pd_cum = np.zeros((Pd, M))
    tot_a = np.zeros(M)
    tot_d = np.zeros(M)
    conv = np.zeros(M, dtype=bool)

    # np.arange for deterministic float step (avoids accumulated error)
    for tau in np.arange(0.0, max_int, dt):
        if conv.all():
            break
        live = ~conv
        rem = 1.0 - tot_a - tot_d
        with np.errstate(over='ignore'):
            p_a = 1.0 / (1.0 + np.exp(-sig_c * (tau - att_tti_rel)))
            p_d = 1.0 / (1.0 + np.exp(-sig_c * (tau - def_tti_rel)))
        if Pa:
            pa_cum += (rem * p_a * lam_a * a_act * live) * dt
            tot_a = pa_cum.sum(axis=0)
        if Pd:
            pd_cum += (rem * p_d * d_lam * d_act * live) * dt
            tot_d = pd_cum.sum(axis=0)
        conv = (tot_a + tot_d) > (1.0 - tol)

    result[eidx] = tot_a
    return result


# --- Team extraction ------------------------------------------------------

def _extract_team(
    df: pd.DataFrame, team_id: int, gk_jersey: int,
    max_speed: float = 12.0,
) -> Dict[str, np.ndarray]:
    """Pack a team's per-frame orientation data into numpy arrays.

    Velocity magnitudes are capped at max_speed (physical human ceiling)
    to protect against noisy per-frame gradient estimates from the
    skeleton smoother.
    """
    sub = df[df["team"] == team_id]
    n = len(sub)
    if n == 0:
        empty = np.zeros(0)
        return {
            "pos": np.zeros((0, 2)), "vel": np.zeros((0, 2)),
            "gk": np.zeros(0, dtype=bool),
            "shoulder": empty, "head": empty, "twist": empty,
            "body_move_offset": empty, "speed": empty, "head_angvel": empty,
        }

    pos = sub[["x", "y"]].values.astype(float)
    if "vx" in sub.columns and "vy" in sub.columns:
        vel = sub[["vx", "vy"]].fillna(0.0).values.astype(float)
        # Clip per-player velocity magnitude to physical ceiling
        mag = np.linalg.norm(vel, axis=1)
        over = mag > max_speed
        if over.any():
            scale = np.where(over, max_speed / np.maximum(mag, 1e-9), 1.0)
            vel = vel * scale[:, None]
    else:
        vel = np.zeros_like(pos)
    gk = (sub["jersey"].values == gk_jersey)

    def _col(name):
        if name in sub.columns:
            return np.nan_to_num(sub[name].values.astype(float), nan=0.0)
        return np.zeros(n)

    return {
        "pos": pos, "vel": vel, "gk": gk,
        "shoulder": _col("shoulder_angle"),
        "head": _col("head_angle"),
        "twist": _col("shoulder_hip_offset"),
        "body_move_offset": _col("body_move_offset"),
        "speed": _col("speed"),
        "head_angvel": _col("head_angular_vel"),
    }


# --- Grid utilities -------------------------------------------------------

def make_grid(
    pitch_length: float = 105.0, pitch_width: float = 68.0,
    smoothing: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build centered grid matching vision.py smoothing convention.

    Default smoothing=5 -> 525x340 cells (0.2 m per cell). This matches
    vision.compute_player_vision(smoothing=5) exactly so PPCF can be
    multiplied element-wise with the vision grid without resample.

    Returns (targets (N,2), xgrid, ygrid).
    """
    nx = int(round(pitch_length * smoothing))
    ny = int(round(pitch_width * smoothing))
    xgrid = np.linspace(-pitch_length / 2, pitch_length / 2, nx)
    ygrid = np.linspace(-pitch_width / 2, pitch_width / 2, ny)
    xx, yy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    return targets, xgrid, ygrid


# --- Public API -----------------------------------------------------------

def compute_ppcf_surface(
    orientations_frame: pd.DataFrame,
    att_team: int,
    ball_xy: Optional[np.ndarray] = None,
    ball_vel: Optional[np.ndarray] = None,
    gk_jerseys: Optional[Dict[int, int]] = None,
    params: Optional[ParamsOAPPCF] = None,
    smoothing: float = 5.0,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Immediate Orientation-Aware PPCF surface for one frame.

    Args:
        orientations_frame: Rows from add_dynamics() for a single frame.
            Must contain at minimum x, y, team, jersey, shoulder_angle,
            head_angle, shoulder_hip_offset, body_move_offset, speed,
            head_angular_vel, vx, vy.
        att_team: 0 or 1, team ID of the attacking side.
        ball_xy: Ball position (x, y) in meters. Can be None (used as 0,0).
        ball_vel: Ball velocity (vx, vy, vz) in m/s for real ball_tt.
        gk_jerseys: {team_id: jersey} dict. Default {0:1, 1:1}.
        params: PPCF params. Default params_oa() (full orientation-aware).
        smoothing: Grid multiplier matching vision.py (default 5).

    Returns:
        (ppcf_grid shape (ny, nx), xgrid (nx,), ygrid (ny,))
        ppcf_grid values are fraction of control by the attacking team.
    """
    if params is None:
        params = params_oa()
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    targets, xgrid, ygrid = make_grid(pitch_length, pitch_width, smoothing)

    def_team = 1 - att_team
    att = _extract_team(orientations_frame, att_team, gk_jerseys.get(att_team, 1))
    dfn = _extract_team(orientations_frame, def_team, gk_jerseys.get(def_team, 1))

    if params.modulate_attackers:
        att_dt_extra = _compute_dt_extra(
            att["pos"], att["shoulder"], att["head"], att["twist"],
            att["body_move_offset"], att["speed"], att["head_angvel"],
            targets, params,
        )
    else:
        att_dt_extra = np.zeros((len(att["pos"]), len(targets)))

    if params.modulate_defenders:
        def_dt_extra = _compute_dt_extra(
            dfn["pos"], dfn["shoulder"], dfn["head"], dfn["twist"],
            dfn["body_move_offset"], dfn["speed"], dfn["head_angvel"],
            targets, params,
        )
    else:
        def_dt_extra = np.zeros((len(dfn["pos"]), len(targets)))

    ball_pos_arr = (
        np.asarray(ball_xy, dtype=float) if ball_xy is not None else None
    )
    ball_vel_arr = (
        np.asarray(ball_vel, dtype=float) if ball_vel is not None else None
    )

    result = _ppcf_core(
        targets,
        att["pos"], att["vel"], att["gk"], att_dt_extra,
        dfn["pos"], dfn["vel"], dfn["gk"], def_dt_extra,
        ball_pos_arr, ball_vel_arr, params,
    )

    ny, nx = len(ygrid), len(xgrid)
    return result.reshape(ny, nx), xgrid, ygrid


def compute_ppcf_at_points(
    orientations_frame: pd.DataFrame,
    att_team: int,
    targets: np.ndarray,
    ball_xy: Optional[np.ndarray] = None,
    ball_vel: Optional[np.ndarray] = None,
    gk_jerseys: Optional[Dict[int, int]] = None,
    params: Optional[ParamsOAPPCF] = None,
) -> np.ndarray:
    """PPCF at a sparse set of target points (no full grid).

    Useful for: evaluating PPCF along a ball trajectory, at pass options,
    or at specific pitch locations without the 525x340 grid cost.

    Returns shape (N,) in [0, 1].
    """
    if params is None:
        params = params_oa()
    if gk_jerseys is None:
        gk_jerseys = {0: 1, 1: 1}

    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"targets must be shape (N, 2), got {targets.shape}")

    def_team = 1 - att_team
    att = _extract_team(orientations_frame, att_team, gk_jerseys.get(att_team, 1))
    dfn = _extract_team(orientations_frame, def_team, gk_jerseys.get(def_team, 1))

    if params.modulate_attackers:
        att_dt = _compute_dt_extra(
            att["pos"], att["shoulder"], att["head"], att["twist"],
            att["body_move_offset"], att["speed"], att["head_angvel"],
            targets, params,
        )
    else:
        att_dt = np.zeros((len(att["pos"]), len(targets)))

    if params.modulate_defenders:
        def_dt = _compute_dt_extra(
            dfn["pos"], dfn["shoulder"], dfn["head"], dfn["twist"],
            dfn["body_move_offset"], dfn["speed"], dfn["head_angvel"],
            targets, params,
        )
    else:
        def_dt = np.zeros((len(dfn["pos"]), len(targets)))

    bp = np.asarray(ball_xy, dtype=float) if ball_xy is not None else None
    bv = np.asarray(ball_vel, dtype=float) if ball_vel is not None else None

    return _ppcf_core(
        targets,
        att["pos"], att["vel"], att["gk"], att_dt,
        dfn["pos"], dfn["vel"], dfn["gk"], def_dt,
        bp, bv, params,
    )
