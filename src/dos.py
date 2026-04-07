"""
dos — Diagonal Opportunity Surfaces.

Quantifies how much additional pitch control the attacking team gains by
exploiting diagonal actions (passes, carries, runs) vs orthogonal ones.

For each cell (x, y) on the pitch, DOS measures:
    DOS(x, y) = ppcf_att(best_diagonal_direction) - ppcf_att(forward_direction)

The mechanism: when a diagonal action occurs, defenders who cannot SEE the
threat (attacker or ball flight) suffer extra detection delay on top of
their normal biomechanical delay (Vater RT + Dos'Santos COD). This shrinks
their reach-field blob, opening gaps the attacker can exploit.

Detection delay depends on two independent vision signals per defender:
    1. V_i(attacker_xy) — can the defender see the attacker? (vision.py, full
       FOV + occlusion model from real 3D skeleton)
    2. V_i(passer_xy) — can the defender see where the ball comes from?

Awareness combines both signals dynamically:
    awareness_i = f(V_attacker, V_passer, attacker_speed, attacker_direction)

Extra detection delay:
    detection_delay = rt(theta_to_threat) * (1 - awareness)

This feeds into ppcf._player_influence via the extra_delay parameter,
producing an awareness-modified PPCF that quantifies the diagonal gain.

Direction classification (DFL PlayAngle convention):
    forward:  |angle_from_forward| < 22.5°
    diagonal: 22.5° <= |angle_from_forward| < 67.5°
    sideways: 67.5° <= |angle_from_forward| < 90°

Coordinate convention: TRACAB meters, centered at (0, 0).
All angles in RADIANS.

References:
  - Spielverlagerung (2025) Tactical Theory: Diagonality
  - Vater (2024) Reaction time vs visual eccentricity
  - Bekkers (2026) Wide Open Gazes (vision model used for awareness)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from scipy.spatial import cKDTree

from .vision import compute_player_vision
from .ppcf import (
    default_params, reaction_time,
    _player_influence, _extract_team_arrays, _aggregate_team_shares,
    PITCH_LENGTH, PITCH_WIDTH,
)


# ── Direction classification ─────────────────────────────────────────────

FORWARD_HALF = np.radians(22.5)
DIAGONAL_HALF = np.radians(67.5)


def classify_attack_angle(angle: float) -> str:
    """Classify attack angle relative to forward axis.

    Args:
        angle: Absolute angle from forward direction in radians [0, pi].

    Returns:
        'forward', 'diagonal', or 'sideways'.
    """
    a = abs(angle)
    if a < FORWARD_HALF:
        return "forward"
    elif a < DIAGONAL_HALF:
        return "diagonal"
    return "sideways"


# ── Vision grid → point-wise lookup ──────────────────────────────────────

def _build_vision_grids(
    frame_df: pd.DataFrame,
    team_id: int,
    smoothing: float = 3.0,
) -> list:
    """Compute full Bekkers vision grid per defender. Returns list of
    (interpolator, pos_xy) tuples for fast point queries.

    Each interpolator maps (x_meters, y_meters) → V in [0, 1], using the
    complete FOV + occlusion model from vision.py with real skeleton widths.
    """
    team_df = frame_df[frame_df["team"] == team_id].reset_index(drop=True)
    if len(team_df) == 0:
        return []

    # All other players (both teams) as potential occluders
    all_players = frame_df.reset_index(drop=True)

    grids = []
    for _, player in team_df.iterrows():
        px, py = float(player["x"]), float(player["y"])
        head = float(player.get("head_angle", player.get("shoulder_angle", 0.0)))
        speed = float(player.get("speed", 0.0))

        # Occluders: everyone except this player
        others = all_players[
            ~((all_players["team"] == player["team"]) &
              (all_players["jersey"] == player["jersey"]))
        ]
        ox = others["x"].values.astype(float)
        oy = others["y"].values.astype(float)
        o_shoulder = others["shoulder_angle"].values.astype(float) if "shoulder_angle" in others.columns else np.zeros(len(others))
        o_sw = others["shoulder_width"].values.astype(float) if "shoulder_width" in others.columns else None

        # Full Bekkers vision grid (grid_width x grid_length pixels)
        vision_grid = compute_player_vision(
            px, py, head, speed,
            ox, oy, o_shoulder, o_sw,
            smoothing=smoothing,
        )

        # Build interpolator mapping real meters → vision value
        grid_len = int(PITCH_LENGTH * smoothing)
        grid_wid = int(PITCH_WIDTH * smoothing)
        x_coords = np.linspace(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, grid_len)
        y_coords = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, grid_wid)

        interp = RegularGridInterpolator(
            (y_coords, x_coords), vision_grid,
            method="linear", bounds_error=False, fill_value=0.0,
        )
        grids.append((interp, np.array([px, py])))

    return grids


def _query_vision(grids: list, points_xy: np.ndarray) -> np.ndarray:
    """Query precomputed vision grids at specific points.

    Args:
        grids: Output of _build_vision_grids. List of (interpolator, pos_xy).
        points_xy: (M, 2) array of query points in meters.

    Returns:
        (P, M) array of vision values V_i(point_j) in [0, 1].
    """
    if len(grids) == 0 or len(points_xy) == 0:
        return np.zeros((len(grids), len(points_xy)))

    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]

    # RegularGridInterpolator expects (y, x) order
    query = np.column_stack([pts[:, 1], pts[:, 0]])

    V = np.zeros((len(grids), len(pts)))
    for i, (interp, _) in enumerate(grids):
        V[i] = interp(query)

    return np.clip(V, 0.0, 1.0)


# ── Awareness model ──────────────────────────────────────────────────────

def _compute_awareness(
    V_attacker: np.ndarray,
    V_passer: np.ndarray,
    attacker_speed: np.ndarray = None,
    attacker_angle_from_fwd: np.ndarray = None,
) -> np.ndarray:
    """Combine vision signals into defender awareness score [0, 1].

    Awareness quantifies how much information the defender has about the
    threat. Two independent vision signals, weighted dynamically:

    - V_attacker dominates: if you can't see the threat, you can't defend it,
      regardless of whether you see the passer.
    - V_passer modulates: seeing the passer helps anticipate the delivery,
      but only if you also know WHERE the threat is.
    - Attacker dynamics: fast/diagonal attackers are harder to track even
      when briefly visible (Spielverlagerung: diagonal runners "arrive sooner
      than predicted").

    Args:
        V_attacker: (...) vision toward the attacker [0, 1]. Any shape.
        V_passer: (...) vision toward the passer [0, 1]. Broadcastable.
        attacker_speed: (...) m/s of the nearest attacker. None = 0.
        attacker_angle_from_fwd: (...) absolute angle of attacker's movement
            direction from forward axis (radians). None = 0 (no penalty).

    Returns:
        (...) awareness in [0, 1], same shape as V_attacker.
    """
    w_att, w_pass = 0.7, 0.3
    base_awareness = w_att * V_attacker + w_pass * V_passer

    # Speed penalty: faster attacker = harder to track even when visible
    # At 8 m/s sprint, tracking penalty ~20%; at 0 m/s, no penalty
    if attacker_speed is not None:
        speed_penalty = 1.0 - 0.025 * np.clip(attacker_speed, 0.0, 8.0)
    else:
        speed_penalty = 1.0

    # Diagonal penalty: human brain misjudges time-to-collision at oblique
    # angles (Spielverlagerung). Diagonal runner "arrives a step sooner."
    # Continuous: max 15% penalty at 45°, 0% at 0° or 90°.
    if attacker_angle_from_fwd is not None:
        # Peak penalty at 45° (pi/4), zero at 0° and 90°
        diag_factor = np.sin(2.0 * np.clip(attacker_angle_from_fwd, 0.0, np.pi / 2))
        diag_penalty = 1.0 - 0.15 * diag_factor
    else:
        diag_penalty = 1.0

    return np.clip(base_awareness * speed_penalty * diag_penalty, 0.0, 1.0)


# ── Nearest attacker dynamics ────────────────────────────────────────────

def _nearest_attacker_dynamics(
    att_pos: np.ndarray,
    att_vel: np.ndarray,
    targets: np.ndarray,
    forward_angle: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each target cell, get speed and direction of the nearest attacker.

    Returns:
        nearest_idx: (N,) index into att_pos
        speeds: (N,) speed of nearest attacker in m/s
        angles_from_fwd: (N,) absolute movement angle from forward axis [0, pi]
    """
    if len(att_pos) == 0:
        N = len(targets)
        return np.zeros(N, dtype=int), np.zeros(N), np.zeros(N)

    tree = cKDTree(att_pos)
    _, nearest_idx = tree.query(targets)

    # Speed of nearest attacker
    speeds = np.sqrt(att_vel[nearest_idx, 0]**2 + att_vel[nearest_idx, 1]**2)

    # Movement direction of nearest attacker relative to forward
    move_angle = np.arctan2(att_vel[nearest_idx, 1], att_vel[nearest_idx, 0])
    angles_from_fwd = np.abs(np.arctan2(
        np.sin(move_angle - forward_angle),
        np.cos(move_angle - forward_angle)))

    return nearest_idx, speeds, angles_from_fwd


# ── Detection delay ─────────────────────────────────────────────────────

def _detection_delay(
    theta_to_threat: np.ndarray,
    awareness: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Extra detection delay (seconds) due to low awareness.

    When awareness is 1: no extra delay (defender knows what's happening).
    When awareness is 0: full extra RT based on theta to threat.
    In between: linear interpolation.

    Args:
        theta_to_threat: (P,) angle from defender shoulder to threat [0, pi].
        awareness: (P,) awareness in [0, 1].
        params: PPCF params (uses rt_base, rt_gain, rt_theta_c_deg, rt_theta_w_deg).

    Returns:
        (P,) extra delay in seconds, shape broadcastable to (P, 1) for
        use as extra_delay in _player_influence.
    """
    rt_threat = reaction_time(theta_to_threat, params)
    # Extra delay = RT for the threat direction, scaled by how blind the defender is.
    # Subtract rt_base to avoid double-counting the base reaction already in PPCF.
    extra_rt = rt_threat - params["rt_base"]
    return np.maximum(0.0, extra_rt * (1.0 - awareness))


# ── DOS computation ─────────────────────────────────────────────────────

def compute_dos_surface(
    orientations_frame: pd.DataFrame,
    attacking_team: int,
    ball_xy: Tuple[float, float],
    attacking_right: bool,
    params: Optional[dict] = None,
    n_grid_x: int = 50,
    n_directions: int = 12,
    vision_smoothing: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Diagonal Opportunity Surface for a single frame.

    For each cell on the pitch, computes how much more pitch control the
    attacking team gains from the best diagonal delivery vs a forward one.

    DOS(x,y) = max(ppcf_att over diagonal directions) - ppcf_att(forward)

    Args:
        orientations_frame: Single-frame orientations slice with columns:
            team, jersey, x, y, shoulder_angle, [head_angle, speed, vx, vy,
            shoulder_width]. More columns = richer vision model.
        attacking_team: Team ID (0 or 1).
        ball_xy: Current ball position (x, y) in meters.
        attacking_right: True if attacking team attacks toward +x.
        params: PPCF params. Default if None.
        n_grid_x: Grid resolution. Default 50.
        n_directions: Number of candidate delivery directions (evenly
            spaced over [0, 2*pi)). Default 12 = every 30 degrees.
        vision_smoothing: Smoothing multiplier for vision.py grids.

    Returns:
        (dos, ppcf_baseline, best_direction, xgrid, ygrid):
            dos: (n_grid_y, n_grid_x) diagonal gain in [-1, 1].
                Positive = diagonal delivery gains more control.
            ppcf_baseline: (n_grid_y, n_grid_x) baseline attacker PPCF
                (forward direction, no extra detection delay).
            best_direction: (n_grid_y, n_grid_x) angle of the best diagonal
                direction per cell (radians, absolute).
            xgrid, ygrid: Cell center coordinates.
    """
    if params is None:
        params = default_params()

    defending_team = 1 - attacking_team
    forward_angle = 0.0 if attacking_right else np.pi

    # ── Build grid ──────────────────────────────────────────────────
    n_grid_y = int(round(n_grid_x * PITCH_WIDTH / PITCH_LENGTH))
    dx = PITCH_LENGTH / n_grid_x
    dy = PITCH_WIDTH / n_grid_y
    xgrid = np.arange(n_grid_x) * dx - PITCH_LENGTH / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    N = len(targets)

    # ── Extract team arrays ─────────────────────────────────────────
    att_pos, att_vel, att_facing = _extract_team_arrays(
        orientations_frame, attacking_team)
    def_pos, def_vel, def_facing = _extract_team_arrays(
        orientations_frame, defending_team)
    P_def = len(def_pos)

    # ── Precompute defender vision grids (full Bekkers, once per frame) ──
    def_vision_grids = _build_vision_grids(
        orientations_frame, defending_team, smoothing=vision_smoothing)

    # Defender vision of passer (constant across directions)
    ball = np.array(ball_xy, dtype=float).reshape(1, 2)
    V_passer = _query_vision(def_vision_grids, ball).ravel()  # (P_def,)

    # Defender vision of each REAL attacker (not grid cells).
    # V_att_real[d, a] = can defender d see attacker a?
    P_att = len(att_pos)
    if P_att > 0 and P_def > 0:
        V_att_real = _query_vision(def_vision_grids, att_pos)  # (P_def, P_att)
        # Map each grid cell to its nearest real attacker
        nearest_idx, att_speeds, att_angles = _nearest_attacker_dynamics(
            att_pos, att_vel, targets, forward_angle)
        V_att_per_cell = V_att_real[:, nearest_idx]  # (P_def, N)
    else:
        V_att_per_cell = np.zeros((P_def, max(N, 1)))
        att_speeds = np.zeros(N)
        att_angles = np.zeros(N)

    # ── Baseline PPCF (forward direction, no extra delay) ───────────
    att_infl_base = _player_influence(att_pos, att_vel, att_facing, targets, params)
    def_infl_base = _player_influence(def_pos, def_vel, def_facing, targets, params)
    I_att_base = att_infl_base.sum(axis=0) if len(att_infl_base) else np.zeros(N)
    I_def_base = def_infl_base.sum(axis=0) if len(def_infl_base) else np.zeros(N)
    ppcf_att_base, _ = _aggregate_team_shares(I_att_base, I_def_base)

    # ── Candidate directions ────────────────────────────────────────
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    best_diag_ppcf = np.full(N, -1.0)
    best_diag_dir = np.zeros(N)
    best_fwd_ppcf = np.full(N, -1.0)

    for D in angles:
        angle_from_fwd = np.abs(np.arctan2(
            np.sin(D - forward_angle), np.cos(D - forward_angle)))
        direction_class = classify_attack_angle(angle_from_fwd)

        if direction_class == "sideways":
            continue

        if P_def == 0:
            extra = None
        else:
            theta_to_threat = np.abs(np.arctan2(
                np.sin(D - def_facing), np.cos(D - def_facing)))  # (P_def,)

            # Awareness: vectorized over (P_def, N), uses REAL attacker
            # visibility + speed + movement direction of nearest attacker
            awareness = _compute_awareness(
                V_att_per_cell,                          # (P_def, N)
                V_passer[:, None] * np.ones((1, N)),     # (P_def, N)
                attacker_speed=att_speeds[None, :] * np.ones((P_def, 1)),
                attacker_angle_from_fwd=att_angles[None, :] * np.ones((P_def, 1)),
            )

            extra = _detection_delay(
                theta_to_threat[:, None] * np.ones((1, N)),
                awareness,
                params,
            )

        # Compute PPCF with modified defender delays
        def_infl_mod = _player_influence(
            def_pos, def_vel, def_facing, targets, params,
            extra_delay=extra,
        )
        I_def_mod = def_infl_mod.sum(axis=0) if len(def_infl_mod) else np.zeros(N)
        ppcf_att_D, _ = _aggregate_team_shares(I_att_base, I_def_mod)

        # Track best per class
        if direction_class == "diagonal":
            improved = ppcf_att_D > best_diag_ppcf
            best_diag_ppcf[improved] = ppcf_att_D[improved]
            best_diag_dir[improved] = D
        elif direction_class == "forward":
            improved = ppcf_att_D > best_fwd_ppcf
            best_fwd_ppcf[improved] = ppcf_att_D[improved]

    # Fallback: if no forward direction was sampled, use baseline
    no_fwd = best_fwd_ppcf < 0
    best_fwd_ppcf[no_fwd] = ppcf_att_base[no_fwd]

    # If no diagonal was sampled (shouldn't happen with n_dirs >= 8)
    no_diag = best_diag_ppcf < 0
    best_diag_ppcf[no_diag] = ppcf_att_base[no_diag]

    dos = best_diag_ppcf - best_fwd_ppcf

    shape = (n_grid_y, n_grid_x)
    return (
        dos.reshape(shape),
        ppcf_att_base.reshape(shape),
        best_diag_dir.reshape(shape),
        xgrid,
        ygrid,
    )


def dos_at_targets(
    orientations_frame: pd.DataFrame,
    targets_xy: np.ndarray,
    attacking_team: int,
    ball_xy: Tuple[float, float],
    attacking_right: bool,
    params: Optional[dict] = None,
    n_directions: int = 12,
    vision_smoothing: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DOS at specific target positions (point-wise API for event analysis).

    Args:
        orientations_frame: Single-frame orientations slice.
        targets_xy: (M, 2) positions in meters.
        attacking_team: Team ID (0 or 1).
        ball_xy: Ball position.
        attacking_right: True if attacking toward +x.
        params: PPCF params.
        n_directions: Candidate directions.
        vision_smoothing: Vision grid resolution.

    Returns:
        (dos, ppcf_baseline, best_direction) each shape (M,).
    """
    if params is None:
        params = default_params()

    targets = np.asarray(targets_xy, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    M = len(targets)

    defending_team = 1 - attacking_team
    forward_angle = 0.0 if attacking_right else np.pi

    att_pos, att_vel, att_facing = _extract_team_arrays(
        orientations_frame, attacking_team)
    def_pos, def_vel, def_facing = _extract_team_arrays(
        orientations_frame, defending_team)
    P_def = len(def_pos)

    # Vision grids (full Bekkers)
    def_vision_grids = _build_vision_grids(
        orientations_frame, defending_team, smoothing=vision_smoothing)
    ball = np.array(ball_xy, dtype=float).reshape(1, 2)
    V_passer = _query_vision(def_vision_grids, ball).ravel()

    # Vision of REAL attackers, mapped to targets via nearest attacker
    P_att = len(att_pos)
    if P_att > 0 and P_def > 0:
        V_att_real = _query_vision(def_vision_grids, att_pos)
        nearest_idx, att_speeds, att_angles = _nearest_attacker_dynamics(
            att_pos, att_vel, targets, forward_angle)
        V_att_per_cell = V_att_real[:, nearest_idx]  # (P_def, M)
    else:
        V_att_per_cell = np.zeros((P_def, M))
        att_speeds = np.zeros(M)
        att_angles = np.zeros(M)

    # Baseline
    att_infl = _player_influence(att_pos, att_vel, att_facing, targets, params)
    def_infl = _player_influence(def_pos, def_vel, def_facing, targets, params)
    I_att = att_infl.sum(axis=0) if len(att_infl) else np.zeros(M)
    I_def = def_infl.sum(axis=0) if len(def_infl) else np.zeros(M)
    ppcf_base, _ = _aggregate_team_shares(I_att, I_def)

    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    best_diag = np.full(M, -1.0)
    best_dir = np.zeros(M)
    best_fwd = np.full(M, -1.0)

    for D in angles:
        angle_from_fwd = np.abs(np.arctan2(
            np.sin(D - forward_angle), np.cos(D - forward_angle)))
        dc = classify_attack_angle(angle_from_fwd)
        if dc == "sideways":
            continue

        if P_def == 0:
            extra = None
        else:
            theta_tt = np.abs(np.arctan2(
                np.sin(D - def_facing), np.cos(D - def_facing)))
            awareness = _compute_awareness(
                V_att_per_cell,
                V_passer[:, None] * np.ones((1, M)),
                attacker_speed=att_speeds[None, :] * np.ones((P_def, 1)),
                attacker_angle_from_fwd=att_angles[None, :] * np.ones((P_def, 1)),
            )
            extra = _detection_delay(
                theta_tt[:, None] * np.ones((1, M)), awareness, params)

        def_infl_mod = _player_influence(
            def_pos, def_vel, def_facing, targets, params, extra_delay=extra)
        I_def_mod = def_infl_mod.sum(axis=0) if len(def_infl_mod) else np.zeros(M)
        ppcf_D, _ = _aggregate_team_shares(I_att, I_def_mod)

        if dc == "diagonal":
            imp = ppcf_D > best_diag
            best_diag[imp] = ppcf_D[imp]
            best_dir[imp] = D
        elif dc == "forward":
            imp = ppcf_D > best_fwd
            best_fwd[imp] = ppcf_D[imp]

    best_fwd[best_fwd < 0] = ppcf_base[best_fwd < 0]
    best_diag[best_diag < 0] = ppcf_base[best_diag < 0]

    return best_diag - best_fwd, ppcf_base, best_dir


def dos_for_event(
    orientations_frame: pd.DataFrame,
    event: dict,
    attacking_team: int,
    attacking_right: bool,
    params: Optional[dict] = None,
    vision_smoothing: float = 3.0,
) -> dict:
    """Compute DOS for a specific pass or carry event.

    Uses the actual event direction (PlayAngle or carry vector) instead of
    sampling multiple candidates. Returns the DOS at the destination plus
    the baseline PPCF, enabling direct comparison.

    Args:
        orientations_frame: Frame at event moment.
        event: Dict with keys: x, y (origin), x_receiver/y_receiver (for
            passes) or carry_end_x/carry_end_y (for carries), event_type.
        attacking_team, attacking_right: Context.
        params, vision_smoothing: Model parameters.

    Returns:
        Dict with: dos, ppcf_att_baseline, ppcf_att_diagonal, direction_class,
        event_direction, awareness_mean, detection_delay_mean.
    """
    if params is None:
        params = default_params()

    etype = event.get("event_type", "pass")
    ox, oy = float(event["x"]), float(event["y"])

    if etype == "carry":
        dx_e = float(event.get("carry_end_x", event.get("x_receiver", ox)))
        dy_e = float(event.get("carry_end_y", event.get("y_receiver", oy)))
    else:
        dx_e = float(event.get("x_receiver", ox))
        dy_e = float(event.get("y_receiver", oy))

    event_dir = np.arctan2(dy_e - oy, dx_e - ox)
    forward_angle = 0.0 if attacking_right else np.pi
    angle_from_fwd = np.abs(np.arctan2(
        np.sin(event_dir - forward_angle),
        np.cos(event_dir - forward_angle)))
    direction_class = classify_attack_angle(angle_from_fwd)

    dest = np.array([[dx_e, dy_e]])
    ball_xy = (ox, oy)

    defending_team = 1 - attacking_team
    att_pos, att_vel, att_facing = _extract_team_arrays(
        orientations_frame, attacking_team)
    def_pos, def_vel, def_facing = _extract_team_arrays(
        orientations_frame, defending_team)
    P_def = len(def_pos)

    # Vision: query at REAL attacker nearest to destination
    def_vision_grids = _build_vision_grids(
        orientations_frame, defending_team, smoothing=vision_smoothing)
    ball = np.array(ball_xy, dtype=float).reshape(1, 2)
    V_passer = _query_vision(def_vision_grids, ball).ravel()

    P_att = len(att_pos)
    if P_att > 0 and P_def > 0:
        V_att_real = _query_vision(def_vision_grids, att_pos)
        nearest_idx, att_spd, att_ang = _nearest_attacker_dynamics(
            att_pos, att_vel, dest, forward_angle)
        V_att_at_dest = V_att_real[:, nearest_idx].ravel()  # (P_def,)
        att_spd_scalar = float(att_spd[0])
        att_ang_scalar = float(att_ang[0])
    else:
        V_att_at_dest = np.zeros(P_def)
        att_spd_scalar = 0.0
        att_ang_scalar = 0.0

    # Baseline PPCF at destination
    att_infl = _player_influence(att_pos, att_vel, att_facing, dest, params)
    def_infl = _player_influence(def_pos, def_vel, def_facing, dest, params)
    I_att = att_infl.sum(axis=0) if len(att_infl) else np.zeros(1)
    I_def_base = def_infl.sum(axis=0) if len(def_infl) else np.zeros(1)
    ppcf_base, _ = _aggregate_team_shares(I_att, I_def_base)

    # Modified PPCF with detection delay for this event's direction
    if P_def > 0:
        theta_tt = np.abs(np.arctan2(
            np.sin(event_dir - def_facing), np.cos(event_dir - def_facing)))
        awareness = _compute_awareness(
            V_att_at_dest, V_passer,
            attacker_speed=att_spd_scalar * np.ones(P_def),
            attacker_angle_from_fwd=att_ang_scalar * np.ones(P_def),
        )
        det_delay = _detection_delay(theta_tt, awareness, params)
        extra = det_delay[:, None]  # (P_def, 1) for single target
    else:
        awareness = np.array([])
        det_delay = np.array([])
        extra = None

    def_infl_mod = _player_influence(
        def_pos, def_vel, def_facing, dest, params, extra_delay=extra)
    I_def_mod = def_infl_mod.sum(axis=0) if len(def_infl_mod) else np.zeros(1)
    ppcf_diag, _ = _aggregate_team_shares(I_att, I_def_mod)

    return {
        "dos": float(ppcf_diag[0] - ppcf_base[0]),
        "ppcf_att_baseline": float(ppcf_base[0]),
        "ppcf_att_with_awareness": float(ppcf_diag[0]),
        "direction_class": direction_class,
        "event_direction": float(event_dir),
        "awareness_mean": float(awareness.mean()) if len(awareness) else 0.0,
        "detection_delay_mean": float(det_delay.mean()) if len(det_delay) else 0.0,
    }
