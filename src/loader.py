"""
loader — Unified data loading for DFL/TRACAB hackathon data.

Loads 3 data types from 5 Bundesliga 2025-26 matches:
  - Skeleton parquet (3D keypoints, 50Hz)
  - AdvancedEvents / kpi_data XML (passes, carries, receptions, shots)
  - Metadata JSON (rosters, phases, pitch dimensions)

All coordinates in METERS, origin at center circle.
TF15 doc says "cm" but actual parquet values are METERS.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --- Paths ----------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hackathon"
MATCH_DIR = DATA_DIR / "Match_Data"

# Match folders and parquet prefixes
MATCHES = {
    "Bayern_Hamburg":      "FCB-HSV",
    "Dortmund_Stuttgart":  "BVB-VFB",
    "Frankfurt_Bayern":    "SGE-FCB",
    "Frankfurt_Union":     "SGE-FCU",
    "Union_Bayern":        "FCU-FCB",
}

# Skeleton part ID -> name (TF15 doc, verified empirically)
PART_NAMES = {
    1: "l_ear", 2: "nose", 3: "r_ear",
    4: "l_shoulder", 5: "neck", 6: "r_shoulder",
    7: "l_elbow", 8: "r_elbow",
    9: "l_wrist", 10: "r_wrist",
    11: "l_hip", 12: "pelvis", 13: "r_hip",
    14: "l_knee", 15: "r_knee",
    16: "l_ankle", 17: "r_ankle",
    18: "l_heel", 19: "l_toe",
    20: "r_heel", 21: "r_toe",
}

# Skeleton team encoding
TEAM_HOME = 1
TEAM_AWAY = 0
TEAM_REF = 3

# Parts we need for orientation (subset for efficiency)
ORIENTATION_PARTS = {1, 2, 3, 4, 5, 6, 11, 12, 13}


# --- Metadata -------------------------------------------------------------

def _find_metadata_json(match: str) -> Path:
    """Find the metadata JSON file inside the match subfolder."""
    match_dir = MATCH_DIR / match
    jsons = list(match_dir.glob("*/*.json"))
    if len(jsons) != 1:
        raise FileNotFoundError(f"Expected 1 metadata JSON in {match_dir}, found {len(jsons)}")
    return jsons[0]


def load_metadata(match: str) -> Dict:
    """Load match metadata JSON.

    Returns dict with keys: game_id, framerate, pitch (m), phases (frame ranges),
    home_gk_left (per half), and player rosters with jersey->identity mapping.
    """
    path = _find_metadata_json(match)
    with open(path) as f:
        raw = json.load(f)

    # Pitch dimensions: JSON has cm, convert to meters
    pitch_x = raw["PitchLongSide"] / 100  # 105.0 m
    pitch_y = raw["PitchShortSide"] / 100  # 68.0 m

    # Phase boundaries (50Hz frame numbers)
    phases = {}
    for i in range(1, 6):
        s = raw.get(f"Phase{i}StartFrame", 0)
        e = raw.get(f"Phase{i}EndFrame", 0)
        if s > 0 and e > 0:
            phases[i] = (s, e)

    # GK side per half
    home_gk_left = {}
    for i in range(1, 6):
        val = raw.get(f"Phase{i}HomeGKLeft")
        if val is not None:
            home_gk_left[i] = val

    # Player rosters: jersey_number + team -> player info
    roster = {}
    for team_key, team_id in [("HomeTeam", TEAM_HOME), ("AwayTeam", TEAM_AWAY)]:
        team = raw[team_key]
        for p in team["Players"]:
            roster[(team_id, p["JerseyNo"])] = {
                "player_id": p["PlayerID"],
                "name": f"{p['FirstName'].split()[0]} {p['LastName']}",
                "full_name": f"{p['FirstName']} {p['LastName']}",
                "position": p.get("StartingPosition", ""),
                "start_frame": p["StartFrameCount"],
                "end_frame": p["EndFrameCount"],
                "team_name": team["LongName"],
                "team_short": team["ShortName"],
            }

    return {
        "game_id": raw["GameID"],
        "kickoff": raw.get("Kickoff", ""),
        "framerate": raw["FrameRate"],
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "phases": phases,
        "home_gk_left": home_gk_left,
        "roster": roster,
    }


# --- Skeleton parquet -----------------------------------------------------

def _parquet_path(match: str) -> Path:
    """Return path to the skeleton parquet file for a match."""
    prefix = MATCHES[match]
    return MATCH_DIR / match / f"{prefix}.parquet"


def load_skeleton_frames(
    match: str,
    frame_start: int,
    frame_end: int,
    parts: Optional[set] = ORIENTATION_PARTS,
) -> pd.DataFrame:
    """Load skeleton data for a range of frames.

    Extracts player positions and keypoints from the nested parquet
    structure into a flat DataFrame. Filters out referees and
    unidentified persons (team=-1).

    Args:
        match: Match folder name (e.g. "Bayern_Hamburg").
        frame_start: First frame number (inclusive).
        frame_end: Last frame number (inclusive).
        parts: Set of part IDs to extract. Default: ORIENTATION_PARTS (9 key parts).
               Pass set(PART_NAMES.keys()) for all 21 parts.

    Returns:
        DataFrame with columns:
            frame_number, team, jersey, part, x, y, z
        Where part is the part name string (e.g. "nose", "pelvis").
        Coordinates in METERS.
    """

    path = _parquet_path(match)
    table = pq.read_table(path, filters=[
        ("frame_number", ">=", frame_start),
        ("frame_number", "<=", frame_end),
    ])

    rows = []
    for i in range(len(table)):
        frame_num = table.column("frame_number")[i].as_py()
        skeletons = table.column("skeletons")[i].as_py()

        for skel in skeletons:
            team = skel["team"]
            jersey = skel["jersey_number"]

            # Skip referees, unidentified, and invalid
            if team not in (TEAM_HOME, TEAM_AWAY) or jersey <= 0:
                continue

            for part in skel["parts"]:
                pid = part["name"]
                if pid not in parts:
                    continue
                rows.append((
                    frame_num, team, jersey,
                    PART_NAMES[pid],
                    part["position_x"],
                    part["position_y"],
                    part["position_z"],
                ))

    df = pd.DataFrame(rows, columns=[
        "frame_number", "team", "jersey", "part", "x", "y", "z",
    ])
    df["team"] = df["team"].astype(np.int8)
    df["jersey"] = df["jersey"].astype(np.int8)
    return df


def load_ball_frames(
    match: str,
    frame_start: int,
    frame_end: int,
) -> pd.DataFrame:
    """Load ball position and velocity for a range of frames.

    Returns:
        DataFrame with columns:
            frame_number, x, y, z, vx, vy, vz, ball_exists
        Coordinates in METERS, velocity in M/S.
    """
    path = _parquet_path(match)
    table = pq.read_table(path, filters=[
        ("frame_number", ">=", frame_start),
        ("frame_number", "<=", frame_end),
    ])

    rows = []
    for i in range(len(table)):
        frame_num = table.column("frame_number")[i].as_py()
        ball_exists = table.column("ball_exists")[i].as_py()
        ball = table.column("ball")[i].as_py()

        if ball is not None and ball_exists:
            rows.append((
                frame_num,
                ball["position_x"], ball["position_y"], ball["position_z"],
                ball["velocity_x"], ball["velocity_y"], ball["velocity_z"],
                True,
            ))
        else:
            rows.append((frame_num, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False))

    return pd.DataFrame(rows, columns=[
        "frame_number", "x", "y", "z", "vx", "vy", "vz", "ball_exists",
    ])


def load_skeleton_at_frame(
    match: str,
    frame: int,
    parts: Optional[set] = ORIENTATION_PARTS,
) -> pd.DataFrame:
    """Convenience: load a single frame of skeleton data."""
    return load_skeleton_frames(match, frame, frame, parts)


# --- Frame mapping: Positions 25Hz (SyncedFrameId) <-> Skeleton 50Hz ------

def synced_frame_to_parquet(
    synced_frame_id: int,
    metadata: Dict,
    half: int = 1,
) -> int:
    """Convert a SyncedFrameId (25Hz, from kpi_data) to a parquet frame_number (50Hz).

    Formula (verified empirically):
        parquet_frame = Phase{half}StartFrame_JSON + 1 + (N - N_start) * 2
    where N_start = 10001 for 1st half, 100001 for 2nd half (DFL catalogue p.96).
    """
    phase_start = metadata["phases"][half][0]
    n_start = 10001 if half == 1 else 100001
    return phase_start + 1 + (synced_frame_id - n_start) * 2


def parquet_to_synced_frame(
    parquet_frame: int,
    metadata: Dict,
    half: int = 1,
) -> int:
    """Convert a parquet frame_number (50Hz) to a SyncedFrameId (25Hz)."""
    phase_start = metadata["phases"][half][0]
    n_start = 10001 if half == 1 else 100001
    return n_start + (parquet_frame - phase_start - 1) // 2


def frame_to_seconds(
    frame: int,
    metadata: Dict,
    half: int = 1,
) -> float:
    """Convert a parquet frame to seconds elapsed in the half."""
    phase_start = metadata["phases"][half][0]
    return (frame - phase_start) / metadata["framerate"]


# --- AdvancedEvents (kpi_data XML) ----------------------------------------

def _kpi_path(match: str) -> Path:
    """Return path to the kpi_data XML file."""
    return MATCH_DIR / match / f"kpi_data_{match}.xml"


def load_passes(match: str) -> pd.DataFrame:
    """Load all pass events from kpi_data XML.

    Returns DataFrame with one row per pass, columns:
        event_id, team_id, player_id, receiver_id, evaluation,
        synced_frame_id, game_time, half,
        x, y, x_receiver, y_receiver,
        play_angle, distance, max_height,
        xp, pressure_player, pressure_receiver,
        dist_closest_defender, num_defenders_goal_side,
        num_defenders_passing_lane, bypassed_defenders,
        back_line_break, through_ball, defensive_state,
        play_num_in_possession,
        x_player_speed, y_player_speed,
    Coordinates in METERS.
    """
    tree = ET.parse(_kpi_path(match))
    root = tree.getroot()

    rows = []
    for event_elem in root.iter("Event"):
        play = event_elem.find("Play")
        if play is None or play.get("IsPass") != "true":
            continue

        # Determine half from InGameSection
        section = play.get("InGameSection", "")
        half = 1 if "first" in section.lower() else 2

        # Skip unsynchronized events
        synced = play.get("SyncedFrameId")
        if synced is None or play.get("SyncSuccessful") != "true":
            continue

        rows.append({
            "event_id": play.get("EventId"),
            "team_id": play.get("TeamId"),
            "player_id": play.get("PlayerId"),
            "receiver_id": play.get("ReceiverId", ""),
            "evaluation": play.get("Evaluation", ""),
            "synced_frame_id": int(synced),
            "game_time": play.get("GameTime", ""),
            "half": half,
            "x": _safe_float(play.get("X-Position")),
            "y": _safe_float(play.get("Y-Position")),
            "x_receiver": _safe_float(play.get("X-PositionReceiver")),
            "y_receiver": _safe_float(play.get("Y-PositionReceiver")),
            "play_angle": _safe_float(play.get("PlayAngle")),
            "distance": _safe_float(play.get("Distance")),
            "max_height": _safe_float(play.get("MaxHeight")),
            "xp": _safe_float(play.get("xP")),
            "pressure_player": _safe_float(play.get("PressureOnPlayer")),
            "pressure_receiver": _safe_float(play.get("PressureOnReceiver")),
            "dist_closest_defender": _safe_float(play.get("DistanceClosestDefenderToPlayer")),
            "num_defenders_goal_side": _safe_int(play.get("NumDefendersGoalSide")),
            "num_defenders_passing_lane": _safe_int(play.get("NumDefendersPassingLane")),
            "bypassed_defenders": _safe_int(play.get("ByPassedDefenders")),
            "back_line_break": play.get("BackLineBreak") == "true",
            "through_ball": play.get("ThroughBall") == "true",
            "defensive_state": play.get("DefensiveState", ""),
            "play_num_in_possession": _safe_int(play.get("PlayNumInPossession")),
            "x_player_speed": _safe_float(play.get("X-PlayerSpeed")),
            "y_player_speed": _safe_float(play.get("Y-PlayerSpeed")),
            "x_direction": _safe_float(play.get("X-Direction")),
            "y_direction": _safe_float(play.get("Y-Direction")),
        })

    return pd.DataFrame(rows)


def load_carries(match: str) -> pd.DataFrame:
    """Load all carry events from kpi_data XML.

    Returns DataFrame with columns:
        event_id, team_id, player_id, half,
        synced_frame_id, end_synced_frame_id,
        x, y, x_end, y_end, distance,
        defensive_state_start, defensive_state_end,
    """
    tree = ET.parse(_kpi_path(match))
    root = tree.getroot()

    rows = []
    for event_elem in root.iter("Event"):
        carry = event_elem.find("Carry")
        if carry is None:
            continue

        synced = carry.get("SyncedFrameId")
        if synced is None or carry.get("SyncSuccessful") != "true":
            continue

        section = carry.get("InGameSection", "")
        half = 1 if "first" in section.lower() else 2

        rows.append({
            "event_id": carry.get("EventId"),
            "team_id": carry.get("TeamId"),
            "player_id": carry.get("PlayerId"),
            "half": half,
            "synced_frame_id": int(synced),
            "end_synced_frame_id": _safe_int(carry.get("EndSyncedFrameId")),
            "x": _safe_float(carry.get("X-Position")),
            "y": _safe_float(carry.get("Y-Position")),
            "x_end": _safe_float(carry.get("X-EndPosition")),
            "y_end": _safe_float(carry.get("Y-EndPosition")),
            "distance": _safe_float(carry.get("Distance")),
            "defensive_state_start": carry.get("DefensiveStateStart", ""),
            "defensive_state_end": carry.get("DefensiveStateEnd", ""),
        })

    return pd.DataFrame(rows)


def load_receptions(match: str) -> pd.DataFrame:
    """Load all reception events from kpi_data XML."""
    tree = ET.parse(_kpi_path(match))
    root = tree.getroot()

    rows = []
    for event_elem in root.iter("Event"):
        rec = event_elem.find("Reception")
        if rec is None:
            continue

        synced = rec.get("SyncedFrameId")
        if synced is None or rec.get("SyncSuccessful") != "true":
            continue

        section = rec.get("InGameSection", "")
        half = 1 if "first" in section.lower() else 2

        rows.append({
            "event_id": rec.get("EventId"),
            "play_id": rec.get("PlayId", ""),
            "team_id": rec.get("TeamId"),
            "player_id": rec.get("PlayerId"),
            "half": half,
            "synced_frame_id": int(synced),
            "x": _safe_float(rec.get("X-Position")),
            "y": _safe_float(rec.get("Y-Position")),
            "is_interception": rec.get("IsInterception") == "true",
            "pressure_receiver": _safe_float(rec.get("PressureOnReceiver")),
            "x_receiver_speed": _safe_float(rec.get("X-ReceiverSpeed")),
            "y_receiver_speed": _safe_float(rec.get("Y-ReceiverSpeed")),
            "defensive_state": rec.get("DefensiveState", ""),
        })

    return pd.DataFrame(rows)


def load_shots(match: str) -> pd.DataFrame:
    """Load all shot events from kpi_data XML (ShotAtGoal in Events XML)."""
    tree = ET.parse(MATCH_DIR / match / f"Events_{match}.xml")
    root = tree.getroot()

    rows = []
    for shot in root.iter("ShotAtGoal"):
        event = shot.getparent() if hasattr(shot, "getparent") else None
        # Get the parent Event element
        event_id = shot.get("EventId", "")

        # Check for goal (SuccessfulShot child)
        is_goal = shot.find("SuccessfulShot") is not None

        rows.append({
            "event_id": event_id,
            "team_id": shot.get("Team", ""),
            "player_id": shot.get("Player", ""),
            "x": _safe_float(shot.get("X-Position")),
            "y": _safe_float(shot.get("Y-Position")),
            "xg": _safe_float(shot.get("xG")),
            "is_goal": is_goal,
        })

    return pd.DataFrame(rows)


# --- MatchInformations XML (formations, positions) ------------------------

def load_match_info(match: str) -> Dict:
    """Load MatchInformations XML for lineup, formation, and player details.

    Returns dict with:
        home_formation, away_formation, home_players, away_players,
        stadium, pitch_x, pitch_y, result
    """
    path = MATCH_DIR / match / f"MatchInformations_{match}.xml"
    tree = ET.parse(path)
    root = tree.getroot()

    info = {}
    general = root.find(".//General")
    if general is not None:
        info["result"] = general.get("Result", "")
        info["match_id"] = general.get("MatchId", "")
        info["match_title"] = general.get("MatchTitle", "")

    env = root.find(".//Environment")
    if env is not None:
        info["stadium"] = env.get("StadiumName", "")
        info["pitch_x"] = _safe_float(env.get("PitchX"))
        info["pitch_y"] = _safe_float(env.get("PitchY"))

    # Teams
    for team_elem in root.iter("Team"):
        role = team_elem.get("Role", "")
        prefix = "home" if role == "home" else "away"
        info[f"{prefix}_formation"] = team_elem.get("LineUp", "")
        info[f"{prefix}_team_id"] = team_elem.get("TeamId", "")
        info[f"{prefix}_team_name"] = team_elem.get("TeamName", "")

        players = []
        for p in team_elem.iter("Player"):
            players.append({
                "person_id": p.get("PersonId", ""),
                "shirt_number": _safe_int(p.get("ShirtNumber")),
                "name": p.get("Shortname", ""),
                "starting": p.get("Starting") == "true",
                "position": p.get("PlayingPosition", ""),
            })
        info[f"{prefix}_players"] = players

    return info


# --- Helpers --------------------------------------------------------------

def _safe_float(val) -> float:
    """Convert string to float, return NaN if None or invalid."""
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _safe_int(val) -> int:
    """Convert string to int, return -1 if None or invalid."""
    if val is None:
        return -1
    try:
        return int(val)
    except (ValueError, TypeError):
        return -1


def get_match_list() -> List[str]:
    """Return list of available match folder names."""
    return list(MATCHES.keys())


def get_match_summary() -> pd.DataFrame:
    """Return summary DataFrame of all matches with metadata."""
    rows = []
    for match in MATCHES:
        meta = load_metadata(match)
        p1 = meta["phases"].get(1, (0, 0))
        p2 = meta["phases"].get(2, (0, 0))
        p1_dur = (p1[1] - p1[0]) / meta["framerate"]
        p2_dur = (p2[1] - p2[0]) / meta["framerate"]

        pf = pq.ParquetFile(_parquet_path(match))

        rows.append({
            "match": match,
            "game_id": meta["game_id"],
            "kickoff": meta["kickoff"],
            "framerate": meta["framerate"],
            "pitch": f"{meta['pitch_x']:.0f}x{meta['pitch_y']:.1f}m",
            "total_frames": pf.metadata.num_rows,
            "p1_duration_min": p1_dur / 60,
            "p2_duration_min": p2_dur / 60,
        })
    return pd.DataFrame(rows)
