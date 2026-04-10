"""
possession — Frame-exact on-ball possession timeline from kpi_data events.

Builds a non-overlapping, priority-resolved list of possession segments
from carries, passes, and receptions. Used by the DOS FOV gate to know
which player's vision matters at every frame.

Priority (high -> low, higher wins on overlap):
  3. Carry       — explicit ranges from kpi_data carry events.
  2. Reception   — post-reception hold (fills gap until next carry).
  1. Pass flight — from pass release to matching reception, passer owns.

Pass -> reception matching is data-driven via the `play_id` field in
kpi_data reception events (each reception stores the event_id of the
originating pass). No nearest-ball heuristic is used.

Event flow (typical):
    [carry by A]  pass release A  -> ball flight (A owns FOV)  -> reception by B
                                                                 |
                                                                 v
                                                          [carry by B]

Carries use the explicit SyncedFrameId ranges. Pass flights fill the
gap between release and reception with the passer (who read the play).
At the reception frame the FOV jumps instantly to the receiver; the
"reception hold" segment covers any gap before the next explicit carry.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


MODE_CARRY = "carry"
MODE_PASS_FLIGHT = "pass_flight"
MODE_RECEPTION = "reception"

_PRIORITY = {
    MODE_PASS_FLIGHT: 1,
    MODE_RECEPTION: 2,
    MODE_CARRY: 3,
}

TIMELINE_COLUMNS = [
    "frame_start", "frame_end",
    "team", "jersey",
    "player_id", "mode", "source_event_id",
]


# ── Roster lookup ───────────────────────────────────────────────────────

def _build_player_lookup(metadata: dict) -> Dict[str, Tuple[int, int]]:
    """Build `player_id (DFL-OBJ-*) -> (team, jersey)` from metadata roster.

    The metadata roster dict uses tuple keys in-memory (from loader) or
    string "team_jersey" keys after JSON round-trip (cached metadata).
    Both are supported.
    """
    lookup: Dict[str, Tuple[int, int]] = {}
    for key, info in metadata.get("roster", {}).items():
        if isinstance(key, tuple):
            team, jersey = key
        else:
            t_str, j_str = str(key).split("_")
            team, jersey = int(t_str), int(j_str)
        pid = info.get("player_id")
        if pid:
            lookup[pid] = (int(team), int(jersey))
    return lookup


# ── Timeline builder ────────────────────────────────────────────────────

def _coerce_frame(val) -> Optional[int]:
    """Convert a possibly-NaN frame value to int, or None if missing.

    Frame 0 is a valid frame number; only NaN/None/negative are rejected.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = int(val)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None


def build_possession_timeline(
    events: pd.DataFrame,
    metadata: dict,
    pass_flight_fallback_frames: int = 25,
    reception_hold_frames: int = 10,
) -> pd.DataFrame:
    """Build non-overlapping possession segments from kpi_data events.

    Args:
        events: Unified events DataFrame (from `load_cached_events`).
            Must contain columns: event_type, parquet_frame, player_id,
            and for carries `parquet_frame_end`; for receptions `play_id`.
        metadata: Match metadata dict (provides roster for player_id ->
            jersey lookup).
        pass_flight_fallback_frames: Duration to assume when a pass has
            no linked reception (ball goes out, etc.). Default 25 = 0.5s.
        reception_hold_frames: How long to hold receiver's FOV after a
            reception if no immediate carry covers it. Default 10 = 0.2s.

    Returns:
        DataFrame with columns `[frame_start, frame_end, team, jersey,
        player_id, mode, source_event_id]`, sorted by frame_start, with
        strictly non-overlapping rows.
    """
    lookup = _build_player_lookup(metadata)

    raw_segments = []

    # ── Carries (highest priority, explicit ranges) ──────────────────
    carries = events[events["event_type"] == "carry"]
    for _, c in carries.iterrows():
        pid = c.get("player_id")
        if pid not in lookup:
            continue
        fs = _coerce_frame(c.get("parquet_frame"))
        fe = _coerce_frame(c.get("parquet_frame_end"))
        if fs is None or fe is None or fe < fs:
            continue
        team, jersey = lookup[pid]
        raw_segments.append({
            "frame_start": fs, "frame_end": fe,
            "team": team, "jersey": jersey,
            "player_id": pid, "mode": MODE_CARRY,
            "source_event_id": c.get("event_id", ""),
            "priority": _PRIORITY[MODE_CARRY],
        })

    # ── Receptions keyed by originating play_id ──────────────────────
    receptions = events[events["event_type"] == "reception"]
    rec_by_play: Dict[str, dict] = {}
    if "play_id" in receptions.columns:
        for row in receptions.to_dict("records"):
            pid_link = row.get("play_id")
            if isinstance(pid_link, str) and pid_link:
                rec_by_play[pid_link] = row

    # ── Passes -> pass_flight + reception segments ───────────────────
    passes = events[events["event_type"] == "pass"]
    for _, p in passes.iterrows():
        passer = p.get("player_id")
        if passer not in lookup:
            continue
        release = _coerce_frame(p.get("parquet_frame"))
        if release is None:
            continue
        team, jersey = lookup[passer]
        pass_evid = p.get("event_id", "")

        rec_row = rec_by_play.get(pass_evid)
        if rec_row is not None:
            recep_frame = _coerce_frame(rec_row.get("parquet_frame"))
        else:
            recep_frame = None

        if recep_frame is None or recep_frame <= release:
            recep_frame = release + pass_flight_fallback_frames
            rec_row = None

        # Pass flight: [release, recep_frame - 1] owned by passer.
        if recep_frame - 1 >= release:
            raw_segments.append({
                "frame_start": release, "frame_end": recep_frame - 1,
                "team": team, "jersey": jersey,
                "player_id": passer, "mode": MODE_PASS_FLIGHT,
                "source_event_id": pass_evid,
                "priority": _PRIORITY[MODE_PASS_FLIGHT],
            })

        # Reception hold: [recep_frame, recep_frame + hold] owned by
        # receiver. Gets overridden by any carry that starts in-range.
        if rec_row is not None:
            rid = rec_row.get("player_id")
            if rid in lookup:
                r_team, r_jersey = lookup[rid]
                raw_segments.append({
                    "frame_start": recep_frame,
                    "frame_end": recep_frame + reception_hold_frames,
                    "team": r_team, "jersey": r_jersey,
                    "player_id": rid, "mode": MODE_RECEPTION,
                    "source_event_id": rec_row.get("event_id", ""),
                    "priority": _PRIORITY[MODE_RECEPTION],
                })

    if not raw_segments:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)

    # ── Paint into a frame-indexed array, highest priority wins ──────
    fmin = min(s["frame_start"] for s in raw_segments)
    fmax = max(s["frame_end"] for s in raw_segments)
    n = fmax - fmin + 1

    painted_idx = np.full(n, -1, dtype=np.int32)
    painted_pri = np.full(n, -1, dtype=np.int8)

    for idx, s in enumerate(raw_segments):
        lo = s["frame_start"] - fmin
        hi = s["frame_end"] - fmin
        seg = slice(lo, hi + 1)
        mask = painted_pri[seg] < s["priority"]
        if mask.any():
            painted_idx[seg] = np.where(mask, idx, painted_idx[seg])
            painted_pri[seg] = np.where(mask, s["priority"], painted_pri[seg])

    # ── Collapse back to non-overlapping segments ────────────────────
    result = []
    i = 0
    while i < n:
        cur = int(painted_idx[i])
        if cur < 0:
            i += 1
            continue
        j = i + 1
        while j < n and int(painted_idx[j]) == cur:
            j += 1
        src = raw_segments[cur]
        result.append({
            "frame_start": i + fmin,
            "frame_end": j - 1 + fmin,
            "team": src["team"],
            "jersey": src["jersey"],
            "player_id": src["player_id"],
            "mode": src["mode"],
            "source_event_id": src["source_event_id"],
        })
        i = j

    return pd.DataFrame(result, columns=TIMELINE_COLUMNS)


# ── Frame lookup ────────────────────────────────────────────────────────

def get_owner_at_frame(
    timeline: pd.DataFrame,
    frame: int,
) -> Optional[dict]:
    """Return the on-ball owner at `frame`, or None if unknown.

    Timeline rows are non-overlapping so at most one matches.
    """
    if len(timeline) == 0:
        return None
    hit = timeline[
        (timeline["frame_start"] <= frame) & (timeline["frame_end"] >= frame)
    ]
    if len(hit) == 0:
        return None
    row = hit.iloc[0]
    return {
        "team": int(row["team"]),
        "jersey": int(row["jersey"]),
        "player_id": row["player_id"],
        "mode": row["mode"],
        "source_event_id": row["source_event_id"],
    }
