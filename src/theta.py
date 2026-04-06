"""
theta — Compute orientation-threat metrics for passes and carries.

Theta measures the angular relationship between body orientation and action
direction. But diagonality is NOT about "higher theta = more dangerous."
It's about the TRADEOFF: diagonal actions combine the progression of a
vertical with the safety of a horizontal, WHILE exploiting the defender's
orientation limitations and giving the receiver a better body position.

We compute metrics on THREE axes:

1. DEFENDER DISRUPTION: how misaligned are defenders relative to the action?
   - theta_target: is the defender looking at the receiving zone?
   - theta_flight: is the defender aware of the ball trajectory?
   - n_wrongfooted: how many defenders must reorient?
   - bypassed_defenders, back_line_break (from kpi_data)

2. RECEIVER ADVANTAGE: how well-positioned is the receiver to continue?
   - receiver_open_angle: is the receiver facing toward goal?
   - receiver_turn_needed: how much must they rotate to progress?
   - n_teammates_in_fov: how many options does the receiver SEE?
   (Manifesto: "diagonal reception = maximum multiplicity of options")

3. RISK-REWARD TRADEOFF: what progression does this action achieve per unit of risk?
   - progression: meters gained toward goal
   - xp: expected pass probability (risk)
   - success: actual outcome
   (Spielverlagerung: "progress with less risk, control with the chance to react")

The thesis: diagonal actions dominate on axis 3 (best tradeoff),
and axes 1+2 explain WHY (mechanism).
"""

import numpy as np
import pandas as pd

from .loader import infer_attacking_team, compute_attacking_right
from .orientation import angle_between

# --- DFL direction classification (catalogue p.28) ------------------------

def classify_direction(play_angle: pd.Series) -> pd.Series:
    """Classify pass direction using DFL official angle ranges.

    PlayAngle: 0 = attacking direction, clockwise, degrees [-180, 180].
    Ranges are symmetric (left/right diagonal both count as diagonal).
    """
    abs_angle = play_angle.abs()
    result = pd.Series("unknown", index=play_angle.index)
    result[abs_angle <= 22.5] = "forward"
    result[(abs_angle > 22.5) & (abs_angle <= 67.5)] = "diagonal"
    result[(abs_angle > 67.5) & (abs_angle <= 112.5)] = "sideways"
    result[abs_angle > 112.5] = "backward"
    result[play_angle.isna()] = "unknown"
    return result


# --- Geometry helpers -----------------------------------------------------

def _point_to_line_distance(px, py, x1, y1, x2, y2):
    """Perpendicular distance from points to line segment."""
    dx, dy = x2 - x1, y2 - y1
    line_len_sq = dx * dx + dy * dy
    if line_len_sq < 1e-10:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / line_len_sq, 0, 1)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def _goal_direction(x, y, attacking_right: bool = True):
    """Angle from position toward the center of the attacking goal."""
    goal_x = 52.5 if attacking_right else -52.5
    goal_y = 0.0
    return np.arctan2(goal_y - y, goal_x - x)


# --- Per-event computation ------------------------------------------------

def compute_pass_theta(
    event: pd.Series,
    orientations: pd.DataFrame,
    attacking_team: int,
    attacking_right: bool = True,
) -> dict:
    """Compute all theta metrics for a single pass.

    Returns dict with defender disruption, receiver advantage,
    and risk-reward tradeoff metrics.
    """
    frame = int(event["parquet_frame"])
    frame_ori = orientations[orientations["frame_number"] == frame]
    if len(frame_ori) == 0:
        return _empty_result(event)

    defending_team = 1 - attacking_team
    defenders = frame_ori[frame_ori["team"] == defending_team]
    attackers = frame_ori[frame_ori["team"] == attacking_team]

    if len(defenders) == 0:
        return _empty_result(event)

    px, py = event.get("x"), event.get("y")
    rx, ry = event.get("x_receiver"), event.get("y_receiver")
    if pd.isna(px) or pd.isna(py) or pd.isna(rx) or pd.isna(ry):
        return _empty_result(event)

    # Pass vector
    pass_dx = event.get("x_direction")
    pass_dy = event.get("y_direction")
    if pd.isna(pass_dx) or pd.isna(pass_dy):
        pass_dx, pass_dy = rx - px, ry - py
    pass_dir = np.arctan2(pass_dy, pass_dx)
    pass_length = np.sqrt(pass_dx**2 + pass_dy**2)

    # Direction classification
    play_angle = event.get("play_angle")
    direction = classify_direction(pd.Series([play_angle])).iloc[0] if not pd.isna(play_angle) else "unknown"

    # Progression toward goal (meters gained in x toward attacking goal)
    if attacking_right:
        progression = rx - px  # positive = toward goal
    else:
        progression = px - rx

    d_x = defenders["x"].values
    d_y = defenders["y"].values

    # ================================================================
    # AXIS 1: DEFENDER DISRUPTION
    # ================================================================

    # Target theta: is the defender looking at the receiving zone?
    target_angle = np.arctan2(ry - d_y, rx - d_x)
    theta_head_target = angle_between(defenders["head_angle"].values, target_angle)
    theta_shoulder_target = angle_between(defenders["shoulder_angle"].values, target_angle)

    # Flight theta: is the defender aware of the ball trajectory direction?
    theta_head_flight = angle_between(defenders["head_angle"].values, pass_dir)
    theta_shoulder_flight = angle_between(defenders["shoulder_angle"].values, pass_dir)

    # Distances
    dist_to_receiver = np.sqrt((d_x - rx)**2 + (d_y - ry)**2)
    dist_to_lane = _point_to_line_distance(d_x, d_y, px, py, rx, ry)

    nearest_idx = np.argmin(dist_to_receiver)
    lane_idx = np.argmin(dist_to_lane)

    # Affected defenders: within 10m of passing lane, between passer and receiver
    lane_dx, lane_dy = rx - px, ry - py
    lane_len = np.sqrt(lane_dx**2 + lane_dy**2)
    if lane_len > 0.1:
        t_proj = ((d_x - px) * lane_dx + (d_y - py) * lane_dy) / (lane_len**2)
        affected = (dist_to_lane < 10.0) & (t_proj > 0.1) & (t_proj < 1.5)
    else:
        affected = np.zeros(len(defenders), dtype=bool)
    n_affected = affected.sum()

    # Wrongfooted: affected defenders with shoulder misalignment > 60 deg
    if n_affected > 0:
        n_wrongfooted = int((theta_shoulder_flight[affected] > np.radians(60)).sum())
        pct_wrongfooted = n_wrongfooted / n_affected
        mean_theta_affected = float(theta_shoulder_flight[affected].mean())
    else:
        n_wrongfooted = 0
        pct_wrongfooted = np.nan
        mean_theta_affected = np.nan

    # In blind half (theta > 90 deg)
    nearest_in_blind = bool(theta_head_target[nearest_idx] > np.pi / 2)

    # ================================================================
    # AXIS 2: RECEIVER ADVANTAGE
    # ================================================================
    receiver_metrics = _compute_receiver_advantage(
        rx, ry, attackers, attacking_right
    )

    # ================================================================
    # AXIS 3: RISK-REWARD
    # ================================================================
    xp = event.get("xp")
    success = 1.0 if event.get("evaluation") == "successfullyCompleted" else 0.0

    # ================================================================
    # BUILD RESULT
    # ================================================================
    result = {
        "event_id": event.get("event_id"),
        "event_type": "pass",
        "frame_number": frame,
        "half": event.get("half", 1),
        "team_id": event.get("team_id"),
        "player_id": event.get("player_id"),
        "x": px,
        "y": py,
        "x_receiver": rx,
        "y_receiver": ry,
        "direction": direction,
        "play_angle": play_angle,
        "pass_length": pass_length,

        # Axis 1: Defender disruption
        "nearest_dist": dist_to_receiver[nearest_idx],
        "nearest_theta_head": theta_head_target[nearest_idx],
        "nearest_theta_shoulder": theta_shoulder_target[nearest_idx],
        "nearest_theta_hip": angle_between(
            defenders["hip_angle"].values, target_angle)[nearest_idx],
        "nearest_in_blind": nearest_in_blind,
        "nearest_jersey": int(defenders.iloc[nearest_idx]["jersey"]),
        "lane_theta_head": theta_head_target[lane_idx],
        "lane_theta_flight": theta_head_flight[lane_idx],
        "n_affected": n_affected,
        "n_wrongfooted": n_wrongfooted,
        "pct_wrongfooted": pct_wrongfooted,
        "mean_theta_affected": mean_theta_affected,
        "mean_theta_head_all": float(theta_head_target.mean()),
        "mean_theta_shoulder_all": float(theta_shoulder_target.mean()),

        # Axis 2: Receiver advantage
        **receiver_metrics,

        # Axis 3: Risk-reward
        "progression": progression,
        "xp": xp,
        "success": success,
        "evaluation": event.get("evaluation"),
        "pressure_player": event.get("pressure_player"),
        "pressure_receiver": event.get("pressure_receiver"),
        "bypassed_defenders": event.get("bypassed_defenders"),
        "back_line_break": event.get("back_line_break"),
    }
    return result


def _compute_receiver_advantage(
    rx: float, ry: float,
    attackers: pd.DataFrame,
    attacking_right: bool,
) -> dict:
    """Compute receiver advantage metrics.

    Measures how well-positioned the receiver is to continue the attack
    after receiving the ball. Diagonal receptions should give:
    - Lower turn_needed (body already oriented to progress)
    - Higher n_teammates_in_fov (more passing options visible)
    """
    # Find the receiver in the orientation data (closest attacker to rx, ry)
    if len(attackers) == 0:
        return {"receiver_open_angle": np.nan, "receiver_turn_needed": np.nan,
                "n_teammates_in_fov": np.nan}

    dists = np.sqrt((attackers["x"].values - rx)**2 + (attackers["y"].values - ry)**2)
    rec_idx = np.argmin(dists)

    if dists[rec_idx] > 5.0:  # no attacker close to receiver position
        return {"receiver_open_angle": np.nan, "receiver_turn_needed": np.nan,
                "n_teammates_in_fov": np.nan}

    rec = attackers.iloc[rec_idx]
    rec_shoulder = rec["shoulder_angle"]
    rec_head = rec["head_angle"]

    # Direction toward goal from receiver position
    goal_dir = _goal_direction(rx, ry, attacking_right)

    # Open angle: how close is receiver's body orientation to goal direction?
    # Low = already facing goal = ready to progress
    receiver_open_angle = float(angle_between(
        np.array([rec_shoulder]), np.array([goal_dir])
    )[0])

    # Turn needed: how much must the receiver rotate to face goal?
    receiver_turn_needed = float(angle_between(
        np.array([rec_head]), np.array([goal_dir])
    )[0])

    # Teammates in FOV: count teammates within 120 deg cone of receiver's head direction
    teammates = attackers.drop(attackers.index[rec_idx])
    if len(teammates) == 0:
        n_in_fov = 0
    else:
        angle_to_teammate = np.arctan2(
            teammates["y"].values - ry,
            teammates["x"].values - rx,
        )
        angular_dist = angle_between(
            np.full(len(teammates), rec_head),
            angle_to_teammate,
        )
        # 120 deg binocular vision = 60 deg half-angle
        n_in_fov = int((angular_dist < np.radians(60)).sum())

    return {
        "receiver_open_angle": receiver_open_angle,
        "receiver_turn_needed": receiver_turn_needed,
        "n_teammates_in_fov": n_in_fov,
    }


def compute_carry_theta(
    event: pd.Series,
    orientations: pd.DataFrame,
    attacking_team: int,
    attacking_right: bool = True,
) -> dict:
    """Compute theta metrics for a carry event (at carry START)."""
    frame = int(event["parquet_frame"])
    frame_ori = orientations[orientations["frame_number"] == frame]
    if len(frame_ori) == 0:
        return _empty_result(event)

    defending_team = 1 - attacking_team
    defenders = frame_ori[frame_ori["team"] == defending_team]
    if len(defenders) == 0:
        return _empty_result(event)

    cx, cy = event.get("x"), event.get("y")
    cx_end, cy_end = event.get("x_end"), event.get("y_end")
    if pd.isna(cx) or pd.isna(cy) or pd.isna(cx_end) or pd.isna(cy_end):
        return _empty_result(event)

    carry_dx, carry_dy = cx_end - cx, cy_end - cy
    carry_dir = np.arctan2(carry_dy, carry_dx)
    carry_length = np.sqrt(carry_dx**2 + carry_dy**2)

    if attacking_right:
        progression = cx_end - cx
    else:
        progression = cx - cx_end

    # Direction classification relative to attacking direction
    # Convert carry_dir to attacking-relative angle
    # If attacking right: 0 rad = forward. If attacking left: pi rad = forward.
    if attacking_right:
        relative_angle = carry_dir  # 0 = right = forward
    else:
        relative_angle = carry_dir - np.pi  # shift so -X = forward
        # Wrap to [-pi, pi]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
    abs_deg = abs(np.degrees(relative_angle))
    direction = ("forward" if abs_deg <= 22.5 else
                 "diagonal" if abs_deg <= 67.5 else
                 "sideways" if abs_deg <= 112.5 else "backward")

    d_x = defenders["x"].values
    d_y = defenders["y"].values

    # Theta: are defenders looking at the carrier?
    target_angle = np.arctan2(cy - d_y, cx - d_x)
    theta_head = angle_between(defenders["head_angle"].values, target_angle)
    theta_shoulder = angle_between(defenders["shoulder_angle"].values, target_angle)

    # Flight theta: are defenders aligned with carry direction?
    theta_flight = angle_between(defenders["shoulder_angle"].values, carry_dir)

    dist_to_carrier = np.sqrt((d_x - cx)**2 + (d_y - cy)**2)
    nearest_idx = np.argmin(dist_to_carrier)

    near_mask = dist_to_carrier < 10.0
    n_near = near_mask.sum()

    return {
        "event_id": event.get("event_id"),
        "event_type": "carry",
        "frame_number": frame,
        "half": event.get("half", 1),
        "team_id": event.get("team_id"),
        "player_id": event.get("player_id"),
        "x": cx,
        "y": cy,
        "x_receiver": cx_end,
        "y_receiver": cy_end,
        "direction": direction,
        "carry_length": carry_length,

        # Axis 1
        "nearest_dist": dist_to_carrier[nearest_idx],
        "nearest_theta_head": theta_head[nearest_idx],
        "nearest_theta_shoulder": theta_shoulder[nearest_idx],
        "nearest_in_blind": bool(theta_head[nearest_idx] > np.pi / 2),
        "nearest_jersey": int(defenders.iloc[nearest_idx]["jersey"]),
        "mean_theta_affected": float(theta_flight[near_mask].mean()) if n_near > 0 else np.nan,
        "n_affected": int(n_near),
        "n_wrongfooted": int((theta_flight[near_mask] > np.radians(60)).sum()) if n_near > 0 else 0,
        "pct_wrongfooted": float((theta_flight[near_mask] > np.radians(60)).mean()) if n_near > 0 else np.nan,
        "mean_theta_head_all": float(theta_head.mean()),
        "mean_theta_shoulder_all": float(theta_shoulder.mean()),

        # Axis 2 (no receiver for carries, but we have the carrier's orientation)
        "receiver_open_angle": np.nan,
        "receiver_turn_needed": np.nan,
        "n_teammates_in_fov": np.nan,

        # Axis 3
        "progression": progression,
        "xp": np.nan,
        "success": np.nan,
        "evaluation": event.get("evaluation", ""),
        "pressure_player": np.nan,
        "pressure_receiver": np.nan,
        "bypassed_defenders": np.nan,
        "back_line_break": np.nan,
        "pass_length": np.nan,
        "play_angle": np.nan,
        "lane_theta_head": np.nan,
        "lane_theta_flight": np.nan,
        "nearest_theta_hip": np.nan,
    }


# --- Batch computation ----------------------------------------------------

def compute_all_theta(
    events: pd.DataFrame,
    orientations: pd.DataFrame,
    home_gk_left_p1: bool = False,
) -> pd.DataFrame:
    """Compute theta for all passes and carries in a match.

    Args:
        events: Cached events DataFrame
        orientations: Output of add_dynamics()
        home_gk_left_p1: Whether home GK is on the left in phase 1.
                         Used to determine attacking direction.
    """
    results = []

    for _, event in events.iterrows():
        etype = event.get("event_type")
        if etype not in ("pass", "carry"):
            continue

        frame = int(event.get("parquet_frame", -1))
        if frame <= 0:
            continue

        # Determine attacking team
        attacking_team = infer_attacking_team(
            event.get("x"), event.get("y"), orientations, frame)
        if attacking_team is None:
            continue

        # Determine attacking direction
        half = event.get("half", 1)
        attacking_right = compute_attacking_right(attacking_team, half, home_gk_left_p1)

        if etype == "pass":
            result = compute_pass_theta(event, orientations, attacking_team, attacking_right)
        elif etype == "carry":
            result = compute_carry_theta(event, orientations, attacking_team, attacking_right)
        results.append(result)

    return pd.DataFrame(results)


# --- Helpers --------------------------------------------------------------

def _empty_result(event):
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "frame_number": int(event.get("parquet_frame", -1)),
        "half": event.get("half", 1),
        "team_id": event.get("team_id"),
        "player_id": event.get("player_id"),
        "x": event.get("x"),
        "y": event.get("y"),
        "x_receiver": event.get("x_receiver"),
        "y_receiver": event.get("y_receiver"),
        "direction": "unknown",
        "carry_length": np.nan,
        "nearest_theta_head": np.nan, "nearest_theta_shoulder": np.nan,
        "nearest_theta_hip": np.nan, "nearest_in_blind": np.nan,
        "mean_theta_head_all": np.nan, "mean_theta_shoulder_all": np.nan,
        "mean_theta_affected": np.nan, "n_affected": 0,
        "n_wrongfooted": 0, "pct_wrongfooted": np.nan,
        "receiver_open_angle": np.nan, "receiver_turn_needed": np.nan,
        "n_teammates_in_fov": np.nan,
        "progression": np.nan, "xp": np.nan, "success": np.nan,
        "evaluation": event.get("evaluation"),
        "pass_length": np.nan, "play_angle": np.nan,
        "nearest_dist": np.nan, "nearest_jersey": np.nan,
        "lane_theta_head": np.nan, "lane_theta_flight": np.nan,
        "pressure_player": np.nan, "pressure_receiver": np.nan,
        "bypassed_defenders": np.nan, "back_line_break": np.nan,
    }


# --- Summary statistics ---------------------------------------------------

def theta_summary_by_direction(theta_df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats grouped by direction — the main analysis table.

    Shows the tradeoff: diagonal should dominate on progression/xp ratio
    and receiver advantage, while explaining mechanism via defender theta.
    """
    df = theta_df[theta_df["direction"] != "unknown"].copy()
    df["success"] = df["success"].fillna(0)

    grouped = df.groupby("direction")
    stats = grouped.agg(
        count=("event_type", "size"),

        # Axis 1: Defender disruption
        mean_nearest_theta_head=("nearest_theta_head", "mean"),
        mean_nearest_theta_shoulder=("nearest_theta_shoulder", "mean"),
        pct_nearest_blind=("nearest_in_blind", "mean"),
        mean_n_wrongfooted=("n_wrongfooted", "mean"),
        mean_pct_wrongfooted=("pct_wrongfooted", "mean"),

        # Axis 2: Receiver advantage
        mean_receiver_open=("receiver_open_angle", "mean"),
        mean_receiver_turn=("receiver_turn_needed", "mean"),
        mean_fov_teammates=("n_teammates_in_fov", "mean"),

        # Axis 3: Risk-reward
        mean_progression=("progression", "mean"),
        mean_xp=("xp", "mean"),
        success_rate=("success", "mean"),
        mean_pass_length=("pass_length", "mean"),
    ).reset_index()

    # Derived: progression per unit of risk
    stats["progression_per_risk"] = stats["mean_progression"] / (1 - stats["mean_xp"] + 0.01)

    # Convert angles to degrees
    for col in stats.columns:
        if "theta" in col or "open" in col or "turn" in col:
            stats[col + "_deg"] = np.degrees(stats[col])

    return stats
