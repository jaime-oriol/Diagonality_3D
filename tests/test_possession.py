"""
Tests for src/possession.py — data-driven possession timeline.

Covers:
  - Empty events
  - Carries (single, multiple, NaN end frame, unknown player_id)
  - Passes with linked reception, without reception (fallback), with
    receiver outside roster
  - Carry-vs-pass-flight priority resolution
  - Reception hold filling the post-reception gap until next carry
  - get_owner_at_frame: in-range, out-of-range, empty
  - Roster lookup tuple keys vs string keys (cached metadata)
  - Frame ordering, no overlaps invariant
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.possession import (
    build_possession_timeline,
    get_owner_at_frame,
    _build_player_lookup,
    MODE_CARRY, MODE_PASS_FLIGHT, MODE_RECEPTION,
    TIMELINE_COLUMNS,
)


# ── Fixtures ────────────────────────────────────────────────────────────

def _metadata():
    """Match info shape (mirrors loader.load_match_info output).

    The kpi_data events identify players by DFL-OBJ-* PersonId; the
    bridge to (team, jersey) lives in MatchInformations.xml, exposed
    via load_match_info as home_players / away_players lists with
    person_id and shirt_number.
    """
    return {
        "home_players": [
            {"person_id": "DFL-HOME-11", "shirt_number": 11, "name": "HomeP11"},
            {"person_id": "DFL-HOME-12", "shirt_number": 12, "name": "HomeP12"},
        ],
        "away_players": [
            {"person_id": "DFL-AWAY-7", "shirt_number": 7, "name": "AwayP7"},
            {"person_id": "DFL-AWAY-8", "shirt_number": 8, "name": "AwayP8"},
        ],
    }


def _events_df(rows):
    """Build an events DataFrame ensuring all expected columns exist."""
    df = pd.DataFrame(rows)
    needed = ["event_type", "parquet_frame", "parquet_frame_end",
              "player_id", "event_id", "play_id"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ── Roster lookup ───────────────────────────────────────────────────────

def test_lookup_from_match_info_basic():
    lk = _build_player_lookup(_metadata())
    assert lk["DFL-HOME-11"] == (1, 11)
    assert lk["DFL-HOME-12"] == (1, 12)
    assert lk["DFL-AWAY-7"] == (0, 7)
    assert lk["DFL-AWAY-8"] == (0, 8)
    assert len(lk) == 4


def test_lookup_skips_players_without_person_id():
    info = {
        "home_players": [
            {"person_id": "", "shirt_number": 11},
            {"person_id": "DFL-HOME-12", "shirt_number": 12},
        ],
        "away_players": [],
    }
    lk = _build_player_lookup(info)
    assert lk == {"DFL-HOME-12": (1, 12)}


def test_lookup_skips_players_with_invalid_jersey():
    info = {
        "home_players": [
            {"person_id": "DFL-X", "shirt_number": -1},
            {"person_id": "DFL-Y", "shirt_number": None},
            {"person_id": "DFL-Z", "shirt_number": 9},
        ],
        "away_players": [],
    }
    lk = _build_player_lookup(info)
    assert lk == {"DFL-Z": (1, 9)}


def test_lookup_handles_missing_lists():
    lk = _build_player_lookup({})
    assert lk == {}
    lk = _build_player_lookup({"home_players": None, "away_players": None})
    assert lk == {}


# ── Empty / degenerate cases ────────────────────────────────────────────

def test_empty_events_returns_empty_timeline():
    ev = pd.DataFrame(columns=["event_type", "parquet_frame"])
    tl = build_possession_timeline(ev, _metadata())
    assert list(tl.columns) == TIMELINE_COLUMNS
    assert len(tl) == 0


def test_unknown_player_carry_dropped():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1100, "player_id": "UNKNOWN", "event_id": "c"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert len(tl) == 0


def test_carry_with_nan_end_frame_dropped():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": np.nan, "player_id": "DFL-HOME-11",
         "event_id": "c"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert len(tl) == 0


def test_carry_inverted_range_dropped():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1100,
         "parquet_frame_end": 1000, "player_id": "DFL-HOME-11",
         "event_id": "c"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert len(tl) == 0


# ── Carries ─────────────────────────────────────────────────────────────

def test_single_carry():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1100, "player_id": "DFL-HOME-11",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert len(tl) == 1
    r = tl.iloc[0]
    assert (r["frame_start"], r["frame_end"]) == (1000, 1100)
    assert r["team"] == 1 and r["jersey"] == 11
    assert r["mode"] == MODE_CARRY
    assert r["source_event_id"] == "c1"


def test_two_disjoint_carries():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1050, "player_id": "DFL-HOME-11",
         "event_id": "c1"},
        {"event_type": "carry", "parquet_frame": 1200,
         "parquet_frame_end": 1300, "player_id": "DFL-HOME-12",
         "event_id": "c2"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert len(tl) == 2
    assert tl.iloc[0]["frame_end"] < tl.iloc[1]["frame_start"]
    assert tl.iloc[0]["jersey"] == 11
    assert tl.iloc[1]["jersey"] == 12


# ── Passes with reception ───────────────────────────────────────────────

def test_pass_linked_to_reception_via_play_id():
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1040,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
    ])
    tl = build_possession_timeline(ev, _metadata(), reception_hold_frames=10)
    assert len(tl) == 2
    flight, recep = tl.iloc[0], tl.iloc[1]
    assert flight["mode"] == MODE_PASS_FLIGHT
    assert (flight["frame_start"], flight["frame_end"]) == (1000, 1039)
    assert flight["jersey"] == 11
    assert recep["mode"] == MODE_RECEPTION
    assert (recep["frame_start"], recep["frame_end"]) == (1040, 1050)
    assert recep["jersey"] == 12


def test_pass_without_reception_falls_back():
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
    ])
    tl = build_possession_timeline(ev, _metadata(),
                                   pass_flight_fallback_frames=30)
    assert len(tl) == 1
    r = tl.iloc[0]
    assert r["mode"] == MODE_PASS_FLIGHT
    assert (r["frame_start"], r["frame_end"]) == (1000, 1029)
    assert r["jersey"] == 11


def test_reception_with_unknown_receiver_skipped_but_flight_kept():
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1030,
         "play_id": "p1", "player_id": "UNKNOWN", "event_id": "r1"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    # Pass flight still uses reception's frame as the cap
    flight = tl[tl["mode"] == MODE_PASS_FLIGHT]
    assert len(flight) == 1
    assert (flight.iloc[0]["frame_start"], flight.iloc[0]["frame_end"]) == (1000, 1029)
    # Reception not added because receiver unknown
    assert (tl["mode"] == MODE_RECEPTION).sum() == 0


def test_pass_with_recep_at_same_frame_uses_fallback():
    # If reception happened at the same frame as the release (data
    # glitch), the function falls back to the default flight length.
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1000,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
    ])
    tl = build_possession_timeline(ev, _metadata(),
                                   pass_flight_fallback_frames=20)
    flight = tl[tl["mode"] == MODE_PASS_FLIGHT]
    assert len(flight) == 1
    assert flight.iloc[0]["frame_end"] == 1019
    # The reception was discarded (because it was treated as fallback)
    assert (tl["mode"] == MODE_RECEPTION).sum() == 0


# ── Priority resolution ────────────────────────────────────────────────

def test_carry_overrides_pass_flight_overlap():
    # Pass flight 1000-1099, carry by another player 1050-1080.
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1100,
         "play_id": "p1", "player_id": "DFL-AWAY-7", "event_id": "r1"},
        {"event_type": "carry", "parquet_frame": 1050,
         "parquet_frame_end": 1080, "player_id": "DFL-HOME-12",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata(), reception_hold_frames=5)
    # Three pass-flight pieces? No: carry splits it into [1000-1049] and [1081-1099]
    pass_segs = tl[tl["mode"] == MODE_PASS_FLIGHT].sort_values("frame_start")
    assert len(pass_segs) == 2
    assert (pass_segs.iloc[0]["frame_start"], pass_segs.iloc[0]["frame_end"]) == (1000, 1049)
    assert (pass_segs.iloc[1]["frame_start"], pass_segs.iloc[1]["frame_end"]) == (1081, 1099)
    assert all(pass_segs["jersey"] == 11)
    carry_segs = tl[tl["mode"] == MODE_CARRY]
    assert len(carry_segs) == 1
    assert (carry_segs.iloc[0]["frame_start"], carry_segs.iloc[0]["frame_end"]) == (1050, 1080)
    assert carry_segs.iloc[0]["jersey"] == 12
    rec_segs = tl[tl["mode"] == MODE_RECEPTION]
    assert len(rec_segs) == 1
    assert rec_segs.iloc[0]["frame_start"] == 1100


def test_reception_hold_overridden_by_immediate_carry():
    # Reception hold [1100-1109], but a carry by the receiver starts at 1100.
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1100,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
        {"event_type": "carry", "parquet_frame": 1100,
         "parquet_frame_end": 1200, "player_id": "DFL-HOME-12",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata(), reception_hold_frames=10)
    # The carry overrides the reception hold entirely (priority 3 > 2).
    assert (tl["mode"] == MODE_RECEPTION).sum() == 0
    carry = tl[tl["mode"] == MODE_CARRY].iloc[0]
    assert (carry["frame_start"], carry["frame_end"]) == (1100, 1200)


def test_reception_hold_partially_overridden_by_late_carry():
    # Reception 1100, hold to 1109. Carry by receiver starts at 1105.
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 1000,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1100,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
        {"event_type": "carry", "parquet_frame": 1105,
         "parquet_frame_end": 1200, "player_id": "DFL-HOME-12",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata(), reception_hold_frames=10)
    rec = tl[tl["mode"] == MODE_RECEPTION]
    assert len(rec) == 1
    assert (rec.iloc[0]["frame_start"], rec.iloc[0]["frame_end"]) == (1100, 1104)
    carry = tl[tl["mode"] == MODE_CARRY].iloc[0]
    assert (carry["frame_start"], carry["frame_end"]) == (1105, 1200)


# ── Invariants ──────────────────────────────────────────────────────────

def test_timeline_sorted_and_non_overlapping():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1099, "player_id": "DFL-HOME-11",
         "event_id": "c1"},
        {"event_type": "pass",  "parquet_frame": 1100,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 1140,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
        {"event_type": "carry", "parquet_frame": 1141,
         "parquet_frame_end": 1300, "player_id": "DFL-HOME-12",
         "event_id": "c2"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    starts = tl["frame_start"].values
    ends = tl["frame_end"].values
    # Strictly sorted starts
    assert (np.diff(starts) > 0).all()
    # No overlaps: each frame_end < next frame_start
    assert (ends[:-1] < starts[1:]).all()


def test_timeline_columns_present_even_when_empty():
    ev = pd.DataFrame(columns=["event_type"])
    tl = build_possession_timeline(ev, _metadata())
    assert list(tl.columns) == TIMELINE_COLUMNS


# ── get_owner_at_frame ──────────────────────────────────────────────────

def test_get_owner_in_carry():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1100, "player_id": "DFL-HOME-11",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    o = get_owner_at_frame(tl, 1050)
    assert o is not None
    assert o["team"] == 1 and o["jersey"] == 11
    assert o["mode"] == MODE_CARRY
    assert o["player_id"] == "DFL-HOME-11"


def test_get_owner_outside_range_returns_none():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1100, "player_id": "DFL-HOME-11",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert get_owner_at_frame(tl, 999) is None
    assert get_owner_at_frame(tl, 1101) is None


def test_get_owner_empty_timeline():
    tl = pd.DataFrame(columns=TIMELINE_COLUMNS)
    assert get_owner_at_frame(tl, 1000) is None


def test_get_owner_at_segment_boundary_inclusive():
    ev = _events_df([
        {"event_type": "carry", "parquet_frame": 1000,
         "parquet_frame_end": 1100, "player_id": "DFL-HOME-11",
         "event_id": "c1"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    assert get_owner_at_frame(tl, 1000) is not None
    assert get_owner_at_frame(tl, 1100) is not None


# ── Full integration: realistic build-up sequence ──────────────────────

def test_full_buildup_sequence_three_passes_two_carries():
    """Realistic 5-event sequence: carry -> pass -> carry -> pass -> carry.
    Verifies possession handover is correct end-to-end without gaps.
    """
    ev = _events_df([
        # P11 carries from frame 0 to 99
        {"event_type": "carry", "parquet_frame": 0, "parquet_frame_end": 99,
         "player_id": "DFL-HOME-11", "event_id": "c1"},
        # P11 passes at frame 100, P12 receives at frame 130
        {"event_type": "pass", "parquet_frame": 100,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 130,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
        # P12 carries 130-199
        {"event_type": "carry", "parquet_frame": 130,
         "parquet_frame_end": 199, "player_id": "DFL-HOME-12",
         "event_id": "c2"},
        # P12 passes at 200, P11 receives at 240 (one-two)
        {"event_type": "pass", "parquet_frame": 200,
         "player_id": "DFL-HOME-12", "event_id": "p2"},
        {"event_type": "reception", "parquet_frame": 240,
         "play_id": "p2", "player_id": "DFL-HOME-11", "event_id": "r2"},
        # P11 carries 240-300
        {"event_type": "carry", "parquet_frame": 240,
         "parquet_frame_end": 300, "player_id": "DFL-HOME-11",
         "event_id": "c3"},
    ])
    tl = build_possession_timeline(ev, _metadata())

    # Spot-check ownership at every key frame
    assert get_owner_at_frame(tl, 50)["jersey"] == 11   # carry 1
    assert get_owner_at_frame(tl, 100)["mode"] == MODE_PASS_FLIGHT
    assert get_owner_at_frame(tl, 100)["jersey"] == 11  # passer in flight
    assert get_owner_at_frame(tl, 129)["jersey"] == 11  # last pass-flight frame
    assert get_owner_at_frame(tl, 130)["jersey"] == 12  # receiver carries
    assert get_owner_at_frame(tl, 165)["jersey"] == 12
    assert get_owner_at_frame(tl, 200)["mode"] == MODE_PASS_FLIGHT
    assert get_owner_at_frame(tl, 200)["jersey"] == 12
    assert get_owner_at_frame(tl, 240)["jersey"] == 11
    assert get_owner_at_frame(tl, 300)["jersey"] == 11

    # Whole range [0, 300] should have an owner at every frame (no gaps)
    for f in range(0, 301):
        assert get_owner_at_frame(tl, f) is not None


def test_two_passes_back_to_back_no_overlap():
    """If a pass starts the same frame the previous reception ends,
    the timeline must remain non-overlapping."""
    ev = _events_df([
        {"event_type": "pass", "parquet_frame": 100,
         "player_id": "DFL-HOME-11", "event_id": "p1"},
        {"event_type": "reception", "parquet_frame": 120,
         "play_id": "p1", "player_id": "DFL-HOME-12", "event_id": "r1"},
        {"event_type": "pass", "parquet_frame": 121,
         "player_id": "DFL-HOME-12", "event_id": "p2"},
        {"event_type": "reception", "parquet_frame": 140,
         "play_id": "p2", "player_id": "DFL-HOME-11", "event_id": "r2"},
    ])
    tl = build_possession_timeline(ev, _metadata())
    starts = tl["frame_start"].values
    ends = tl["frame_end"].values
    assert (np.diff(starts) > 0).all()
    assert (ends[:-1] < starts[1:]).all()
