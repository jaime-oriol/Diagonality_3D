"""Render the real Olise passes figure for one Bayern match.

Pipeline (no DOS needed — just the events.parquet):
  1. Load cached events for the match.
  2. Filter to passes by Olise (DFL-OBJ-J01R3R).
  3. Normalise coords so Bayern always attacks → (flip x,y in halves where
     Bayern attacks left, then force attacking_right=True in the renderer).
  4. Classify each pass as forward / diagonal / sideways / backward via
     the DFL PlayAngle convention.
  5. Pass to plot_player_passes.

Output: test/passes_olise_<MATCH>.png
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.loader import load_match_info, compute_attacking_right
from src.preprocess import load_cached_events, load_cached_metadata
from src.viz.passes_plot import plot_player_passes

# ── Config ───────────────────────────────────────────────────────────────
MATCH = "Bayern_Hamburg"
OLISE_ID = "DFL-OBJ-J01R3R"          # Michael Olise (Bayern, jersey 17, ORM)
SUBTITLE = [
    "Bundesliga 2025-26",
    "Bayern Munich 5-0 Hamburger SV (13 September 2025)",
]


def _classify_dfl(angle_deg: float) -> str:
    """DFL official PlayAngle bucketing (catalogue p.28).
    Angles are degrees relative to attacking direction, range [-180, 180]."""
    if pd.isna(angle_deg):
        return "unknown"
    a = abs(float(angle_deg))
    if a <= 22.5:    return "forward"
    if a <= 67.5:    return "diagonal"
    if a <= 112.5:   return "sideways"
    return "backward"


def _build_olise_passes(match: str, player_id: str) -> pd.DataFrame:
    info = load_match_info(match)
    home_team_id = info.get("home_team_id", "")
    home_gk_left_p1 = bool(load_cached_metadata(match)["home_gk_left"][1])

    events = load_cached_events(match)
    p = events[(events["event_type"] == "pass")
               & (events["player_id"] == player_id)].copy()
    p = p.dropna(subset=["x", "y", "x_receiver", "y_receiver"])

    # Coord normalisation: every pass should display as if Bayern were
    # attacking RIGHT. Compute attacking_right per half, then flip x,y for
    # passes where Bayern attacked LEFT in that half.
    bayern_team_int = 1 if info["home_team_id"] == home_team_id else 0
    flip_mask = []
    for _, row in p.iterrows():
        ar = compute_attacking_right(bayern_team_int, int(row["half"]),
                                     home_gk_left_p1)
        flip_mask.append(not ar)
    flip_mask = np.array(flip_mask)
    for col in ("x", "y", "x_receiver", "y_receiver"):
        p.loc[flip_mask, col] = -p.loc[flip_mask, col]

    p["direction_class"] = p["play_angle"].apply(_classify_dfl)
    p["successful"] = p["evaluation"].isin({"successfullyCompleted", "successful"})

    return p[["x", "y", "x_receiver", "y_receiver",
              "direction_class", "successful", "play_angle", "evaluation"]]


def main():
    passes = _build_olise_passes(MATCH, OLISE_ID)
    print(f"Olise passes ({MATCH}): {len(passes)}")
    print(passes["direction_class"].value_counts().to_dict())
    print(f"Accuracy: {passes['successful'].mean()*100:.1f}%")

    out = f"test/passes_olise_{MATCH}.png"
    plot_player_passes(
        passes,
        title="Michael Olise Passes",
        subtitle=SUBTITLE,
        attacking_right=True,
        team_logo_path="figures/logos/bayern.png",
        project_logo_path="figures/Logo_vizs.png",
        save_path=out,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
