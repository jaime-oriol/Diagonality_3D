"""Enrich `dos_validation_raw_xt_ddef_theta.csv` with the player + team
identity fields needed for the storytelling rankings.

The validation CSV stores `team_id` (DFL-CLU-*) but NOT `player_id`,
because validate_dos_outcomes.py loads events by frame and infers the
attacking team from skeleton proximity (no player identity needed for
the DOS computation). For the rankings/narrative we need:

  - player_id      (DFL-OBJ-*)   from events.parquet / load_takeons
  - player_name    (Shortname)   from MatchInformations.xml
  - player_position (TW/IVR/...) from MatchInformations.xml
  - team_name      (LongName)    from MatchInformations.xml

Output: `test/dos_validation_full.csv` (input + 5 columns).

This is the canonical input for `aggregate_rankings.py`,
`select_top_events.py`, and the parameterized renders.
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import load_match_info, load_takeons
from src.preprocess import load_cached_events

RAW_IN = Path("test/dos_validation_raw_xt_ddef_theta.csv")
OUT = Path("test/dos_validation_full.csv")


def _build_event_to_player(match: str) -> dict:
    """event_id -> player_id (DFL-OBJ-*) for passes, carries, take-ons."""
    out: dict = {}
    ev = load_cached_events(match)
    for r in ev[ev["event_type"].isin(("pass", "carry"))].itertuples(index=False):
        if pd.notna(r.player_id):
            out[str(r.event_id)] = str(r.player_id)
    try:
        tk = load_takeons(match)
        for r in tk.itertuples(index=False):
            if pd.notna(r.player_id):
                out[str(r.event_id)] = str(r.player_id)
    except FileNotFoundError:
        pass
    return out


def _build_player_meta(match: str) -> dict:
    """player_id -> {name, position, team_name, team_id}."""
    info = load_match_info(match)
    home_team_name = info.get("home_team_name", "")
    away_team_name = info.get("away_team_name", "")
    home_team_id = info.get("home_team_id", "")
    away_team_id = info.get("away_team_id", "")
    out: dict = {}
    for key, tname, tid in [("home_players", home_team_name, home_team_id),
                            ("away_players", away_team_name, away_team_id)]:
        for p in info.get(key, []) or []:
            pid = p.get("person_id")
            if not pid:
                continue
            out[pid] = {
                "player_name": p.get("name", ""),
                "player_position": p.get("position", ""),
                "team_name": tname,
                "team_id": tid,
            }
    return out


def _build_team_name_map(match: str) -> dict:
    """team_id (DFL-CLU-*) -> team_name."""
    info = load_match_info(match)
    return {
        info.get("home_team_id", ""): info.get("home_team_name", ""),
        info.get("away_team_id", ""): info.get("away_team_name", ""),
    }


def main():
    if not RAW_IN.exists():
        raise SystemExit(f"Missing {RAW_IN}. Run enrich_with_theta.py first.")
    df = pd.read_csv(RAW_IN)
    print(f"Loaded {RAW_IN}: {len(df):,} rows")

    matches = sorted(df["match"].dropna().unique())
    eid_to_pid: dict = {}
    pid_meta: dict = {}
    team_to_name: dict = {}
    for m in matches:
        e2p = _build_event_to_player(m)
        eid_to_pid.update(e2p)
        pid_meta.update(_build_player_meta(m))
        team_to_name.update(_build_team_name_map(m))
        print(f"  [{m}] events->player: {len(e2p)}, players: {len(_build_player_meta(m))}")

    df["event_id"] = df["event_id"].astype(str)
    df["player_id"] = df["event_id"].map(eid_to_pid)

    df["player_name"] = df["player_id"].map(
        lambda p: pid_meta.get(p, {}).get("player_name", "") if pd.notna(p) else "")
    df["player_position"] = df["player_id"].map(
        lambda p: pid_meta.get(p, {}).get("player_position", "") if pd.notna(p) else "")
    df["team_name"] = df["team_id"].map(team_to_name).fillna("")

    n_pid = int(df["player_id"].notna().sum())
    n_pname = int((df["player_name"].fillna("") != "").sum())
    n_tname = int((df["team_name"].fillna("") != "").sum())
    print(f"\nResolved player_id: {n_pid:,}/{len(df):,} ({n_pid/len(df)*100:.1f}%)")
    print(f"Resolved player_name: {n_pname:,}/{len(df):,}")
    print(f"Resolved team_name: {n_tname:,}/{len(df):,}")

    df.to_csv(OUT, index=False)
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
