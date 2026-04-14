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
from typing import Dict, Optional

import numpy as np
import pandas as pd

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


# --- AdvancedEvents (kpi_data XML) ----------------------------------------

def kpi_path(match: str) -> Path:
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
        x_direction, y_direction,
    Coordinates in METERS.
    """
    tree = ET.parse(kpi_path(match))
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
            "x": safe_float(play.get("X-Position")),
            "y": safe_float(play.get("Y-Position")),
            "x_receiver": safe_float(play.get("X-PositionReceiver")),
            "y_receiver": safe_float(play.get("Y-PositionReceiver")),
            "play_angle": safe_float(play.get("PlayAngle")),
            "distance": safe_float(play.get("Distance")),
            "max_height": safe_float(play.get("MaxHeight")),
            "xp": safe_float(play.get("xP")),
            "pressure_player": safe_float(play.get("PressureOnPlayer")),
            "pressure_receiver": safe_float(play.get("PressureOnReceiver")),
            "dist_closest_defender": safe_float(play.get("DistanceClosestDefenderToPlayer")),
            "num_defenders_goal_side": safe_int(play.get("NumDefendersGoalSide")),
            "num_defenders_passing_lane": safe_int(play.get("NumDefendersPassingLane")),
            "bypassed_defenders": safe_int(play.get("ByPassedDefenders")),
            "back_line_break": play.get("BackLineBreak") == "true",
            "through_ball": play.get("ThroughBall") == "true",
            "defensive_state": play.get("DefensiveState", ""),
            "play_num_in_possession": safe_int(play.get("PlayNumInPossession")),
            "x_player_speed": safe_float(play.get("X-PlayerSpeed")),
            "y_player_speed": safe_float(play.get("Y-PlayerSpeed")),
            "x_direction": safe_float(play.get("X-Direction")),
            "y_direction": safe_float(play.get("Y-Direction")),
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
    tree = ET.parse(kpi_path(match))
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
            "end_synced_frame_id": safe_int(carry.get("EndSyncedFrameId")),
            "x": safe_float(carry.get("X-Position")),
            "y": safe_float(carry.get("Y-Position")),
            "x_end": safe_float(carry.get("X-EndPosition")),
            "y_end": safe_float(carry.get("Y-EndPosition")),
            "distance": safe_float(carry.get("Distance")),
            "defensive_state_start": carry.get("DefensiveStateStart", ""),
            "defensive_state_end": carry.get("DefensiveStateEnd", ""),
        })

    return pd.DataFrame(rows)


def load_receptions(match: str) -> pd.DataFrame:
    """Load all reception events from kpi_data XML."""
    tree = ET.parse(kpi_path(match))
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
            "x": safe_float(rec.get("X-Position")),
            "y": safe_float(rec.get("Y-Position")),
            "is_interception": rec.get("IsInterception") == "true",
            "pressure_receiver": safe_float(rec.get("PressureOnReceiver")),
            "x_receiver_speed": safe_float(rec.get("X-ReceiverSpeed")),
            "y_receiver_speed": safe_float(rec.get("Y-ReceiverSpeed")),
            "defensive_state": rec.get("DefensiveState", ""),
        })

    return pd.DataFrame(rows)


# --- Take-ons (TacklingGame, attacker wins) -------------------------------

# WinnerResult values that count as a "real" successful take-on:
#   - dribbledAround:        attacker explicitly dribbled around defender
#   - ballControlRetained:   attacker kept ball through a contest
#   - ballcontactSucceeded:  attacker won the contact and kept ball
# Excluded WinnerResult values (still WinnerRole=withBallControl but NOT
# the SV "diagonal touch" we care about):
#   - layoff:                attacker received and passed back, not a regate
#   - fouled:                defender fouled the attacker (not skill)
TAKEON_OK_RESULTS = frozenset({
    "dribbledAround",
    "ballControlRetained",
    "ballcontactSucceeded",
})


def load_takeons(match: str) -> pd.DataFrame:
    """Load successful take-ons (1v1 duels won by the attacker WITH ball).

    Source:
      - Events_*.xml provides the rich attributes (WinnerRole, WinnerResult,
        DribblingType, DribbleEvaluation, DribblingSide).
      - kpi_data_*.xml provides the sync info (SyncedFrameId, X-Position,
        Y-Position) that the Events XML lacks.

    Both files share `EventId` so we join one-to-one. We keep only
    take-ons whose attacker (WinnerPlayer) kept ball control AND the
    duel resolved as a real take-on (see TAKEON_OK_RESULTS).

    Returns a DataFrame with one row per take-on and columns:
        event_id, team_id (winner), player_id (winner), opponent_id (loser),
        half, synced_frame_id, x, y,
        winner_result, dribbling_type, dribbling_side, dribble_evaluation,
        type, possession_change, foul_won (=loser_result fouled).
    Coordinates in METERS (TRACAB convention, centered).
    """
    events_path = MATCH_DIR / match / f"Events_{match}.xml"
    kpi = kpi_path(match)

    # Pass 1: rich attributes from Events XML keyed by event_id
    rich = {}
    for event_elem in ET.parse(events_path).getroot().iter("Event"):
        tg = event_elem.find("TacklingGame")
        if tg is None:
            continue
        if tg.get("WinnerRole") != "withBallControl":
            continue
        wres = tg.get("WinnerResult", "")
        if wres not in TAKEON_OK_RESULTS:
            continue
        eid = tg.get("EventId") or event_elem.get("EventId")
        if eid is None:
            continue
        rich[eid] = {
            "winner_id": tg.get("Winner", ""),
            "winner_team": tg.get("WinnerTeam", ""),
            "loser_id": tg.get("Loser", ""),
            "loser_team": tg.get("LoserTeam", ""),
            "type": tg.get("Type", ""),
            "winner_result": wres,
            "dribbling_type": tg.get("DribblingType", ""),
            "dribbling_side": tg.get("DribblingSide", ""),
            "dribble_evaluation": tg.get("DribbleEvaluation", ""),
            "possession_change": tg.get("PossessionChange") == "true",
        }

    # Pass 2: sync + position from kpi_data, joined on EventId
    rows = []
    for event_elem in ET.parse(kpi).getroot().iter("Event"):
        tg = event_elem.find("TacklingGame")
        if tg is None:
            continue
        eid = tg.get("EventId") or event_elem.get("EventId")
        if eid is None or eid not in rich:
            continue
        if tg.get("SyncSuccessful") != "true":
            continue
        synced = tg.get("SyncedFrameId")
        if synced is None:
            continue
        section = tg.get("InGameSection", "")
        half = 1 if "first" in section.lower() else 2

        meta = rich[eid]
        rows.append({
            "event_id": eid,
            "team_id": meta["winner_team"],
            "player_id": meta["winner_id"],
            "opponent_id": meta["loser_id"],
            "opponent_team": meta["loser_team"],
            "half": half,
            "synced_frame_id": int(synced),
            "x": safe_float(tg.get("X-Position")),
            "y": safe_float(tg.get("Y-Position")),
            "type": meta["type"],
            "winner_result": meta["winner_result"],
            "dribbling_type": meta["dribbling_type"],
            "dribbling_side": meta["dribbling_side"],
            "dribble_evaluation": meta["dribble_evaluation"],
            "possession_change": meta["possession_change"],
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
        info["pitch_x"] = safe_float(env.get("PitchX"))
        info["pitch_y"] = safe_float(env.get("PitchY"))

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
                "shirt_number": safe_int(p.get("ShirtNumber")),
                "name": p.get("Shortname", ""),
                "starting": p.get("Starting") == "true",
                "position": p.get("PlayingPosition", ""),
            })
        info[f"{prefix}_players"] = players

    return info


# --- Helpers --------------------------------------------------------------

def safe_float(val) -> float:
    """Convert string to float, return NaN if None or invalid."""
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def safe_int(val) -> int:
    """Convert string to int, return -1 if None or invalid."""
    if val is None:
        return -1
    try:
        return int(val)
    except (ValueError, TypeError):
        return -1


# --- Shared game logic (used by theta.py, ddef.py) -----------------------

def infer_attacking_team(
    event_x: float,
    event_y: float,
    orientations: pd.DataFrame,
    frame: int,
    max_dist: float = 3.0,
) -> Optional[int]:
    """Match event position to skeleton to find the attacking team.

    Returns team (0 or 1) of the nearest player to (event_x, event_y)
    at the given frame, or None if no match within max_dist.
    """
    if pd.isna(event_x) or pd.isna(event_y):
        return None
    frame_ori = orientations[orientations["frame_number"] == frame]
    if len(frame_ori) == 0:
        return None
    dists = np.sqrt((frame_ori["x"].values - event_x)**2 + (frame_ori["y"].values - event_y)**2)
    if dists.min() < max_dist:
        return int(frame_ori.iloc[np.argmin(dists)]["team"])
    return None


def compute_attacking_right(attacking_team: int, half: int, home_gk_left_p1: bool) -> bool:
    """Determine if the attacking team attacks toward +X (right).

    home_gk_left_p1=False -> home GK on RIGHT in half 1 -> home attacks LEFT.
    Teams swap sides at halftime.
    """
    if half == 1:
        return (attacking_team == 1) == home_gk_left_p1
    else:
        return (attacking_team == 1) != home_gk_left_p1


