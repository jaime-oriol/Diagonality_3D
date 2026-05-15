"""Storytelling rankings + summaries from `dos_validation_full.csv`.

Outputs (under `outputs/tables/`):

  players_by_dos.csv          Top players by mean DOS (overall + per type)
  players_by_xt.csv           Top players by mean xT-delta gained
  players_by_ddef.csv         Top players by mean D-Def disruption
  players_by_wrongfooting.csv Top players by mean n_wrongfooted (theta)
  players_by_diag_share.csv   Top players by % diagonal of their actions
  players_by_volume.csv       Top players by event volume (n_events)
  teams_summary.csv           Per team: DOS, xT, D-Def, diag share, n events
  matches_summary.csv         Per match: counts, means, top scorer
  matches_team_dir_breakdown.csv  Per (match, team, direction) breakdown
  direction_breakdown.csv     Global per-direction-class summary
  events_per_player_match.csv Tall table: (player, match, n_events, mean_dos)

Filter: players with a minimum of `MIN_EVENTS_PLAYER` events (default 30)
to keep rankings statistically meaningful.

This is the single source of truth for the narrative document numbers.
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd


RAW = Path("outputs/intermediate/dos_validation_full.csv")
OUT_DIR = Path("outputs/tables")

MIN_EVENTS_PLAYER = 30   # min n events to qualify a player for rankings
TOP_N = 30               # rows kept per ranking table


def _agg_metrics(g: pd.DataFrame) -> dict:
    """Aggregate the canonical metrics over a group of events.

    Includes the AWARENESS COGNITIVE LAYER (awareness_mean,
    detection_delay_mean) — the mechanism that links DOS to real
    impact. These are the most narrative-friendly numbers."""
    succ = g["evaluation"].isin({"successfullyCompleted", "successful"})
    return {
        "n_events": int(len(g)),
        "n_passes": int((g["event_type"] == "pass").sum()),
        "n_carries": int((g["event_type"] == "carry").sum()),
        "n_takeons": int((g["event_type"] == "takeon").sum()),
        # DOS distribution
        "dos_mean": float(g["dos"].mean()),
        "dos_median": float(g["dos"].median()),
        "dos_pos_rate": float((g["dos"] > 0).mean()),
        "dos_p95": float(g["dos"].quantile(0.95)),
        # Cognitive layer (the WHY behind DOS)
        "awareness_mean": float(g["awareness_mean"].mean()) if "awareness_mean" in g.columns else float("nan"),
        "detection_delay_mean_s": float(g["detection_delay_mean"].mean()) if "detection_delay_mean" in g.columns else float("nan"),
        # xT outcome
        "xt_delta_mean": float(g["xt_delta"].mean()) if "xt_delta" in g.columns else float("nan"),
        "xt_delta_sum": float(g["xt_delta"].sum()) if "xt_delta" in g.columns else float("nan"),
        "xt_origin_mean": float(g["xt_origin"].mean()) if "xt_origin" in g.columns else float("nan"),
        # D-Def disruption
        "ddef_3s_mean": float(g["ddef_3s"].mean()) if "ddef_3s" in g.columns else float("nan"),
        "biaxial_mean": float(g["biaxial"].mean()) if "biaxial" in g.columns else float("nan"),
        # Theta orientation
        "n_wrongfooted_mean": float(g["n_wrongfooted"].mean()) if "n_wrongfooted" in g.columns else float("nan"),
        "pct_nearest_blind": float(g["nearest_in_blind"].mean()) if "nearest_in_blind" in g.columns else float("nan"),
        # Direction profile
        "diag_share": float((g["direction_class"] == "diagonal").mean()),
        "fwd_share": float((g["direction_class"] == "forward").mean()),
        "side_share": float((g["direction_class"] == "sideways").mean()),
        # Outcomes
        "success_rate": float(succ.mean()),
        "back_line_break_rate": float(g["back_line_break"].mean()) if "back_line_break" in g.columns else float("nan"),
    }


def _bi_axial_balance(df: pd.DataFrame, w: str = "3s") -> pd.Series:
    p1 = df[f"pc1_{w}"].abs() if f"pc1_{w}" in df.columns else pd.Series(np.nan, index=df.index)
    p2 = df[f"pc2_{w}"].abs() if f"pc2_{w}" in df.columns else pd.Series(np.nan, index=df.index)
    mn = np.minimum(p1, p2)
    mx = np.maximum(p1, p2)
    return mn / mx.where(mx > 1e-9, np.nan)


def _player_table(df: pd.DataFrame, sort_by: str, ascending: bool = False,
                   min_events: int = MIN_EVENTS_PLAYER) -> pd.DataFrame:
    """Group by player_id, aggregate metrics, sort by `sort_by`."""
    rows = []
    for pid, g in df.groupby("player_id"):
        if pd.isna(pid) or len(g) < min_events:
            continue
        # Player metadata uses the FIRST non-empty value across rows
        meta_row = g.iloc[0]
        rows.append({
            "player_id": str(pid),
            "player_name": str(meta_row.get("player_name", "")),
            "player_position": str(meta_row.get("player_position", "")),
            "team_name": str(meta_row.get("team_name", "")),
            **_agg_metrics(g),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return out.head(TOP_N)


def _team_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tname, g in df.groupby("team_name"):
        if pd.isna(tname) or not tname:
            continue
        rows.append({"team_name": str(tname), **_agg_metrics(g)})
    return pd.DataFrame(rows).sort_values("dos_mean", ascending=False).reset_index(drop=True)


def _match_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("match"):
        # Top scorer per match (highest mean DOS among qualifying players)
        top_player = ""
        top_dos = float("nan")
        per_player = []
        for pid, gp in g.groupby("player_id"):
            if pd.isna(pid) or len(gp) < 10:
                continue
            per_player.append({
                "player_name": str(gp.iloc[0].get("player_name", "")),
                "dos_mean": float(gp["dos"].mean()),
                "n": int(len(gp)),
            })
        if per_player:
            best = max(per_player, key=lambda r: r["dos_mean"])
            top_player = best["player_name"]
            top_dos = best["dos_mean"]
        rows.append({
            "match": str(m),
            **_agg_metrics(g),
            "top_dos_player": top_player,
            "top_dos_value": top_dos,
        })
    return pd.DataFrame(rows).sort_values("match").reset_index(drop=True)


def _match_team_dir_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team, direction_class): n events, mean DOS, mean xT, success."""
    rows = []
    for (m, t, d), g in df.groupby(["match", "team_name", "direction_class"]):
        if not isinstance(t, str) or not t:
            continue
        succ = g["evaluation"].isin({"successfullyCompleted", "successful"})
        rows.append({
            "match": str(m), "team_name": str(t), "direction_class": str(d),
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "xt_delta_mean": float(g["xt_delta"].mean()) if "xt_delta" in g.columns else float("nan"),
            "success_rate": float(succ.mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["match", "team_name", "direction_class"]).reset_index(drop=True)


def _direction_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, g in df.groupby("direction_class"):
        succ = g["evaluation"].isin({"successfullyCompleted", "successful"})
        rows.append({
            "direction_class": str(cls),
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "dos_pos_rate": float((g["dos"] > 0).mean()),
            "awareness_mean": float(g["awareness_mean"].mean()) if "awareness_mean" in g.columns else float("nan"),
            "detection_delay_mean_s": float(g["detection_delay_mean"].mean()) if "detection_delay_mean" in g.columns else float("nan"),
            "xt_delta_mean": float(g["xt_delta"].mean()) if "xt_delta" in g.columns else float("nan"),
            "ddef_3s_mean": float(g["ddef_3s"].mean()) if "ddef_3s" in g.columns else float("nan"),
            "biaxial_mean": float(g["biaxial"].mean()) if "biaxial" in g.columns else float("nan"),
            "success_rate": float(succ.mean()),
            "back_line_break_rate": float(g["back_line_break"].mean()) if "back_line_break" in g.columns else float("nan"),
            "pct_nearest_blind": float(g["nearest_in_blind"].mean()) if "nearest_in_blind" in g.columns else float("nan"),
            "n_wrongfooted_mean": float(g["n_wrongfooted"].mean()) if "n_wrongfooted" in g.columns else float("nan"),
        })
    return pd.DataFrame(rows).reindex(
        pd.RangeIndex(len(rows))).sort_values("direction_class").reset_index(drop=True)


def _zones_breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per zone (center / half_space / wing) summary. `col` is either
    'origin_zone' or 'dest_zone'. The half-space hypothesis is THE
    Spielverlagerung claim — this CSV is the empirical answer."""
    if col not in df.columns:
        return pd.DataFrame()
    rows = []
    for z, g in df.groupby(col):
        if pd.isna(z) or not z:
            continue
        succ = g["evaluation"].isin({"successfullyCompleted", "successful"})
        rows.append({
            "zone": str(z),
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "dos_pos_rate": float((g["dos"] > 0).mean()),
            "awareness_mean": float(g["awareness_mean"].mean()) if "awareness_mean" in g.columns else float("nan"),
            "xt_delta_mean": float(g["xt_delta"].mean()) if "xt_delta" in g.columns else float("nan"),
            "ddef_3s_mean": float(g["ddef_3s"].mean()) if "ddef_3s" in g.columns else float("nan"),
            "diag_share": float((g["direction_class"] == "diagonal").mean()),
            "success_rate": float(succ.mean()),
            "back_line_break_rate": float(g["back_line_break"].mean()) if "back_line_break" in g.columns else float("nan"),
        })
    return pd.DataFrame(rows).set_index("zone").reindex(
        ["center", "half_space", "wing"]).dropna(how="all").reset_index()


def _zones_x_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Tall table: (origin_zone, direction_class) -> metrics. Lets the
    narrative quote 'diagonals from the half-space' specifically — the
    SV signature combination."""
    if "origin_zone" not in df.columns:
        return pd.DataFrame()
    rows = []
    for (z, d), g in df.groupby(["origin_zone", "direction_class"]):
        if pd.isna(z) or not z or pd.isna(d):
            continue
        succ = g["evaluation"].isin({"successfullyCompleted", "successful"})
        rows.append({
            "origin_zone": str(z),
            "direction_class": str(d),
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "xt_delta_mean": float(g["xt_delta"].mean()) if "xt_delta" in g.columns else float("nan"),
            "awareness_mean": float(g["awareness_mean"].mean()) if "awareness_mean" in g.columns else float("nan"),
            "success_rate": float(succ.mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["origin_zone", "direction_class"]).reset_index(drop=True)


def _top_events_full_vector(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Top-N events by DOS with the FULL metric vector for narrative
    deep dives. The writer can pick any of these and have every number
    needed for a paragraph at hand."""
    cols = [c for c in [
        "match", "event_id", "event_type", "frame", "half",
        "team_name", "player_name", "player_position",
        "direction_class", "origin_zone", "dest_zone",
        "x_origin", "y_origin", "x_dest", "y_dest",
        "dos", "xt_delta", "xt_origin",
        "ddef_3s", "biaxial",
        "pc1_3s", "pc2_3s", "pc3_3s",
        "awareness_mean", "detection_delay_mean",
        "n_wrongfooted", "pct_wrongfooted", "nearest_in_blind",
        "mean_theta_shoulder_all", "mean_theta_head_all",
        "receiver_open_angle", "receiver_turn_needed", "n_teammates_in_fov",
        "evaluation", "back_line_break", "bypassed_defenders", "through_ball",
        "distance", "pressure_player",
    ] if c in df.columns]
    return df.nlargest(n, "dos")[cols].reset_index(drop=True)


def _events_per_player_match(df: pd.DataFrame) -> pd.DataFrame:
    """Tall table: one row per (player, match) with counts + mean DOS."""
    rows = []
    for (pid, m), g in df.groupby(["player_id", "match"]):
        if pd.isna(pid):
            continue
        rows.append({
            "player_id": str(pid),
            "player_name": str(g.iloc[0].get("player_name", "")),
            "team_name": str(g.iloc[0].get("team_name", "")),
            "match": str(m),
            "n_events": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "diag_share": float((g["direction_class"] == "diagonal").mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["player_name", "match"]).reset_index(drop=True)


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Run enrich_full_metadata.py first.")
    df = pd.read_csv(RAW)
    print(f"Loaded {RAW}: {len(df):,} rows")

    # Bi-axial balance (used by some rankings); compute once if missing.
    if "biaxial" not in df.columns:
        df["biaxial"] = _bi_axial_balance(df, "3s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "players_by_dos":           _player_table(df, "dos_mean"),
        "players_by_xt":            _player_table(df, "xt_delta_sum"),
        "players_by_ddef":          _player_table(df, "ddef_3s_mean"),
        "players_by_wrongfooting":  _player_table(df, "n_wrongfooted_mean"),
        "players_by_awareness_low": _player_table(df, "awareness_mean", ascending=True),
        "players_by_diag_share":    _player_table(df, "diag_share"),
        "players_by_volume":        _player_table(df, "n_events", min_events=10),
        "teams_summary":            _team_table(df),
        "matches_summary":          _match_table(df),
        "matches_team_dir_breakdown": _match_team_dir_breakdown(df),
        "direction_breakdown":      _direction_breakdown(df),
        "zones_origin_breakdown":   _zones_breakdown(df, "origin_zone"),
        "zones_dest_breakdown":     _zones_breakdown(df, "dest_zone"),
        "zones_x_direction":        _zones_x_direction(df),
        "events_per_player_match":  _events_per_player_match(df),
        "top_events_full_vector":   _top_events_full_vector(df, n=50),
    }

    for name, t in tables.items():
        path = OUT_DIR / f"{name}.csv"
        if isinstance(t, pd.DataFrame) and len(t):
            t.to_csv(path, index=False)
            print(f"  {name:30s} {len(t):>4d} rows -> {path}")
        else:
            print(f"  {name:30s}    (empty, not written)")

    # Print a quick textual top-5 to console for sanity
    print("\n=== TOP 5 BY MEAN DOS (min 30 events) ===")
    p = tables["players_by_dos"].head(5)
    if len(p):
        print(p[["player_name", "team_name", "n_events", "dos_mean", "diag_share"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
