"""Render per-player pass maps for the storytelling deck.

For every (player, match) combination with at least `MIN_PASSES_PER_MATCH`
passes, render the project-style pass map (`src/viz/passes_plot.py`)
containing direction-class colored arrows + Opta-style stats footer.

Player selection: top `TOP_N_PLAYERS` of the `players_by_dos` ranking,
filtered to attacking positions (any non-GK / non-CB) so we focus on
players whose diagonality is the storytelling angle.

Output naming:
  outputs/frames/pass_maps/{PlayerName}_{Match}.png

Resume: skips files that already exist on disk.
"""

import sys
sys.path.insert(0, ".")

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import (
    MATCHES, load_match_info, compute_attacking_right,
)
from src.preprocess import load_cached_events, load_cached_metadata
from src.viz.passes_plot import plot_player_passes


RAW = Path("outputs/intermediate/dos_validation_full.csv")
RANK = Path("outputs/tables/players_by_dos.csv")
OUT_DIR = Path("outputs/frames/pass_maps")

TOP_N_PLAYERS = 8
MIN_PASSES_PER_MATCH = 12          # don't render maps with too few passes
MAX_MATCHES_PER_PLAYER = 3          # cap renders per player

# Positions we EXCLUDE — pass maps for GKs / IVR aren't the story.
EXCLUDE_POSITIONS = {"TW"}


def _classify_dfl(angle_deg: float) -> str:
    """DFL official PlayAngle bucketing (catalogue p.28)."""
    if pd.isna(angle_deg):
        return "unknown"
    a = abs(float(angle_deg))
    if a <= 22.5:    return "forward"
    if a <= 67.5:    return "diagonal"
    if a <= 112.5:   return "sideways"
    return "backward"


# Explicit team_name → logo filename mapping. The team_name we see in the
# enriched CSV comes straight from MatchInformations.xml LongName, so we
# whitelist exactly those values. Logos are sourced from the
# luukhopman/football-logos GitHub repo (Bundesliga folder), all 139x181
# RGBA PNGs with transparent background — drop-in compatible with the
# top-right project logo placement in src/viz/passes_plot.py.
TEAM_LOGO_FILES = {
    "FC Bayern München":   "bayern_munich.png",
    "Bayern Munich":       "bayern_munich.png",     # alt spelling, just in case
    "Hamburger SV":        "hamburger_sv.png",
    "Borussia Dortmund":   "borussia_dortmund.png",
    "VfB Stuttgart":       "vfb_stuttgart.png",
    "Eintracht Frankfurt": "eintracht_frankfurt.png",
    "1. FC Union Berlin":  "1fc_union_berlin.png",
}


def _team_logo_path(team_name: str) -> str:
    """Map a team name to its logo PNG (if present)."""
    if not team_name:
        return ""
    fname = TEAM_LOGO_FILES.get(team_name.strip())
    if fname:
        p = Path("figures/logos") / fname
        if p.exists():
            return str(p)
    return ""  # plot_player_passes handles empty paths gracefully


def _build_player_passes(match: str, player_id: str) -> pd.DataFrame:
    """Coordinate normalisation: flip x and y in halves where the player
    attacked LEFT so the rendered map always reads as if the player
    attacked RIGHT."""
    info = load_match_info(match)
    home_team_id = info.get("home_team_id", "")
    home_gk_left_p1 = bool(load_cached_metadata(match)["home_gk_left"][1])

    events = load_cached_events(match)
    p = events[(events["event_type"] == "pass")
               & (events["player_id"] == player_id)].copy()
    p = p.dropna(subset=["x", "y", "x_receiver", "y_receiver"])
    if len(p) == 0:
        return p

    # Determine the player's team_int from any of his event rows.
    team_int = 1 if str(p.iloc[0]["team_id"]) == home_team_id else 0

    flips = []
    for _, row in p.iterrows():
        ar = compute_attacking_right(team_int, int(row["half"]), home_gk_left_p1)
        flips.append(not ar)
    flip_mask = np.asarray(flips)
    for col in ("x", "y", "x_receiver", "y_receiver"):
        p.loc[flip_mask, col] = -p.loc[flip_mask, col]

    p["direction_class"] = p["play_angle"].apply(_classify_dfl)
    p["successful"] = p["evaluation"].isin({"successfullyCompleted", "successful"})
    return p[["x", "y", "x_receiver", "y_receiver",
              "direction_class", "successful", "play_angle", "evaluation"]]


def _select_targets(df: pd.DataFrame, ranking: pd.DataFrame) -> list:
    """For each top player, pick up to MAX_MATCHES_PER_PLAYER of their
    matches with the highest event count. Returns list of dicts."""
    selected = []
    if len(ranking) == 0:
        return selected
    top = ranking[~ranking["player_position"].isin(EXCLUDE_POSITIONS)].head(TOP_N_PLAYERS)

    for _, prow in top.iterrows():
        pid = prow["player_id"]
        pname = prow["player_name"]
        sub = df[df["player_id"] == pid]
        if len(sub) == 0:
            continue
        # Per-match pass count
        per_match = (sub[sub["event_type"] == "pass"]
                     .groupby("match").size()
                     .reset_index(name="n_passes")
                     .sort_values("n_passes", ascending=False))
        per_match = per_match[per_match["n_passes"] >= MIN_PASSES_PER_MATCH]
        per_match = per_match.head(MAX_MATCHES_PER_PLAYER)
        for _, mrow in per_match.iterrows():
            selected.append({
                "player_id": str(pid),
                "player_name": str(pname),
                "team_name": str(prow["team_name"]),
                "match": str(mrow["match"]),
                "n_passes": int(mrow["n_passes"]),
            })
    return selected


# Storytelling-friendly short team names for the subtitle line.
# DFL LongName is verbose ("FC Bayern München", "1. FC Union Berlin"); we
# anglicize / shorten them for the Opta-style subtitle.
NICE_TEAM_NAME = {
    "FC Bayern München":   "Bayern Munich",
    "Hamburger SV":        "Hamburger SV",
    "Borussia Dortmund":   "Borussia Dortmund",
    "VfB Stuttgart":       "VfB Stuttgart",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "1. FC Union Berlin":  "Union Berlin",
}


def _match_subtitle(match: str, info: dict) -> list:
    """Build the 2-line subtitle in the Opta style:
        ['Bundesliga 2025-26',
         '{home} {hg}-{ag} {away} ({DD Month YYYY})']

    Reads home/away names from MatchInformations.xml LongName (mapped via
    NICE_TEAM_NAME for storytelling), score from Result ('h:a' format),
    date from cache metadata.json Kickoff ('YYYY-MM-DD HH:MM:SS')."""
    from datetime import datetime
    from src.preprocess import load_cached_metadata

    home = NICE_TEAM_NAME.get(info.get("home_team_name", ""), info.get("home_team_name", ""))
    away = NICE_TEAM_NAME.get(info.get("away_team_name", ""), info.get("away_team_name", ""))
    result = (info.get("result", "") or "").replace(":", "-")

    date_str = ""
    try:
        kickoff = load_cached_metadata(match).get("kickoff", "")
        if kickoff:
            dt = datetime.fromisoformat(kickoff.replace(" ", "T")[:19])
            date_str = dt.strftime("%-d %B %Y")  # e.g. "13 September 2025"
    except Exception:
        pass

    if home and away and result:
        line2 = f"{home} {result} {away}"
        if date_str:
            line2 += f" ({date_str})"
    else:
        line2 = match.replace("_", " vs ")

    return ["Bundesliga 2025-26", line2]


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Run enrich_full_metadata.py first.")
    if not RANK.exists():
        raise SystemExit(f"Missing {RANK}. Run aggregate_rankings.py first.")

    df = pd.read_csv(RAW)
    ranking = pd.read_csv(RANK)
    print(f"Loaded full data: {len(df):,} rows, ranking: {len(ranking)} players")

    targets = _select_targets(df, ranking)
    print(f"Targets selected: {len(targets)}")
    for t in targets:
        print(f"  {t['player_name']:25s} {t['team_name']:18s} "
              f"{t['match']:25s} ({t['n_passes']} passes)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_done = 0; n_skip = 0; n_fail = 0
    t_total = time.time()

    for tg in targets:
        pname_safe = tg["player_name"].replace(" ", "_").replace("/", "")
        out = OUT_DIR / f"{pname_safe}_{tg['match']}.png"
        if out.exists() and out.stat().st_size > 30_000:
            print(f"  SKIP {out.name}")
            n_skip += 1
            continue

        try:
            passes = _build_player_passes(tg["match"], tg["player_id"])
        except Exception as e:
            print(f"  [ERROR build] {tg['player_name']} @ {tg['match']}: {e}")
            n_fail += 1
            continue

        if len(passes) < MIN_PASSES_PER_MATCH:
            print(f"  [SKIP] {tg['player_name']} @ {tg['match']}: {len(passes)} passes")
            n_fail += 1
            continue

        info = load_match_info(tg["match"])
        try:
            plot_player_passes(
                passes,
                title=f"{tg['player_name']} Passes",
                subtitle=_match_subtitle(tg["match"], info),
                attacking_right=True,
                team_logo_path=_team_logo_path(tg["team_name"]) or None,
                project_logo_path="figures/Logo.png" if Path("figures/Logo.png").exists() else None,
                save_path=str(out),
            )
        except Exception as e:
            print(f"  [ERROR plot] {out.name}: {e}")
            n_fail += 1
            continue

        n_done += 1
        print(f"  + {out.name}  ({len(passes)} passes)")

    print(f"\nDONE: {n_done} new, {n_skip} skipped, {n_fail} failed "
          f"({(time.time()-t_total)/60:.1f} min)")


if __name__ == "__main__":
    main()
