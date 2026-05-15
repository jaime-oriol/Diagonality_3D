"""Pick the top-N events for video / frame rendering, save to JSON.

Used by:
  - renders/render_top_dos_videos.py
  - renders/render_top_event_frames.py

Selection strategy: a composite score that combines DOS magnitude with
real-world impact (xT-delta and back_line_break) so we don't pick events
that have high DOS but no observable consequence. Three lists are saved:

  top_dos_events.json        — top N by DOS (any event_type)
  top_dos_carries.json       — top N by DOS among carries
  top_dos_passes.json        — top N by DOS among passes
  top_xt_events.json         — top N by xT-delta (any event_type)
  top_ddef_events.json       — top N by D-Def 3s
  highlight_events.json      — ~30 events curated for the FRAME gallery
                              (mix of best-by-DOS + best-by-xT + diagonals
                              + back_line_break + take-ons)

Each JSON is a list of dicts with the fields the renderers need:
  match, event_id, event_type, frame, half,
  attacking_team, attacking_right,
  player_id, player_name, team_name, dos, xt_delta, ddef_3s.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW = Path("outputs/intermediate/dos_validation_full.csv")
OUT_DIR = Path("outputs/tables")

# Tuneables
TOP_N_VIDEO = 10            # videos are expensive — cap aggressively
TOP_N_FRAME = 30            # frames are cheap; render more for the gallery
SCORE_DOS_W = 1.0
SCORE_XT_W = 5.0            # xt_delta has typical magnitude ~0.01 - boost it
SCORE_BLB_W = 0.05          # back_line_break flag bonus


def _composite_score(df: pd.DataFrame) -> pd.Series:
    """DOS + alpha * xt_delta + beta * back_line_break. Honest tradeoff:
    pure DOS picks events that are mechanistically interesting; xt and
    back_line_break add real-world impact so the rendered videos look
    like things a viewer would care about, not just abstract heatmaps."""
    dos = df["dos"].fillna(0.0)
    xt = df.get("xt_delta", pd.Series(0.0, index=df.index)).fillna(0.0)
    blb = df.get("back_line_break", pd.Series(False, index=df.index)).fillna(False).astype(float)
    return SCORE_DOS_W * dos + SCORE_XT_W * xt + SCORE_BLB_W * blb


def _row_to_record(r: pd.Series) -> dict:
    return {
        "match": str(r.get("match", "")),
        "event_id": str(r.get("event_id", "")),
        "event_type": str(r.get("event_type", "")),
        "frame": int(r["frame"]) if pd.notna(r.get("frame")) else -1,
        "half": int(r["half"]) if pd.notna(r.get("half")) else 1,
        "attacking_team": int(r["attacking_team"]) if pd.notna(r.get("attacking_team")) else -1,
        "attacking_right": bool(r.get("attacking_right", False)),
        "player_id": str(r["player_id"]) if pd.notna(r.get("player_id")) else "",
        "player_name": str(r.get("player_name", "")),
        "team_name": str(r.get("team_name", "")),
        "direction_class": str(r.get("direction_class", "")),
        "dos": float(r["dos"]) if pd.notna(r.get("dos")) else float("nan"),
        "xt_delta": float(r["xt_delta"]) if pd.notna(r.get("xt_delta")) else float("nan"),
        "ddef_3s": float(r["ddef_3s"]) if pd.notna(r.get("ddef_3s")) else float("nan"),
        "back_line_break": bool(r.get("back_line_break", False)),
    }


def _save(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, default=str))
    print(f"  {path.name:30s} {len(records):>3d} events -> {path}")


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Run enrich_full_metadata.py first.")
    df = pd.read_csv(RAW)
    print(f"Loaded {RAW}: {len(df):,} rows")

    # Ensure a few helpers exist
    df["score"] = _composite_score(df)

    # Filter: only events with a valid frame and player_id
    df = df.dropna(subset=["frame"]).copy()
    df["frame"] = df["frame"].astype(int)

    # ── Top N by DOS (composite score) ────────────────────────────────
    top_all = df.nlargest(TOP_N_VIDEO, "score")
    top_carries = df[df["event_type"] == "carry"].nlargest(TOP_N_VIDEO, "score")
    top_passes = df[df["event_type"] == "pass"].nlargest(TOP_N_VIDEO, "score")
    top_xt = df.nlargest(TOP_N_VIDEO, "xt_delta") if "xt_delta" in df.columns else df.head(0)
    top_ddef = df.nlargest(TOP_N_VIDEO, "ddef_3s") if "ddef_3s" in df.columns else df.head(0)

    _save([_row_to_record(r) for _, r in top_all.iterrows()],
          OUT_DIR / "top_dos_events.json")
    _save([_row_to_record(r) for _, r in top_carries.iterrows()],
          OUT_DIR / "top_dos_carries.json")
    _save([_row_to_record(r) for _, r in top_passes.iterrows()],
          OUT_DIR / "top_dos_passes.json")
    _save([_row_to_record(r) for _, r in top_xt.iterrows()],
          OUT_DIR / "top_xt_events.json")
    _save([_row_to_record(r) for _, r in top_ddef.iterrows()],
          OUT_DIR / "top_ddef_events.json")

    # ── Highlight gallery (mix of top-by-criteria + edge cases) ───────
    # We want diversity in the FRAME gallery: not just the top-N by DOS,
    # but also a few from each interesting axis (best diagonal, best
    # carry, best pass, line-break, take-on, every match represented).
    seen_ids = set()
    gallery_rows = []

    def _take(df_sub, n, label):
        """Add up to n rows from df_sub that haven't been added yet."""
        cnt = 0
        for _, r in df_sub.iterrows():
            eid = str(r["event_id"])
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            d = _row_to_record(r)
            d["highlight_reason"] = label
            gallery_rows.append(d)
            cnt += 1
            if cnt >= n:
                break

    _take(df.nlargest(8, "score"), 8, "top_score")
    _take(df[df["event_type"] == "carry"].nlargest(5, "score"), 5, "top_carry")
    _take(df[df["event_type"] == "takeon"].nlargest(4, "score"), 4, "top_takeon")
    _take(df[df["direction_class"] == "diagonal"].nlargest(5, "dos"), 5, "top_diagonal")
    if "back_line_break" in df.columns:
        _take(df[df["back_line_break"]].nlargest(4, "score"), 4, "back_line_break")
    if "xt_delta" in df.columns:
        _take(df.nlargest(4, "xt_delta"), 4, "top_xt")

    # Ensure at least 1-2 per match for diversity
    for m, g in df.groupby("match"):
        already = sum(1 for r in gallery_rows if r["match"] == m)
        needed = max(0, 2 - already)
        if needed > 0:
            _take(g.nlargest(needed, "score"), needed, f"per_match_{m}")

    # Cap at TOP_N_FRAME
    gallery_rows = gallery_rows[:TOP_N_FRAME]
    _save(gallery_rows, OUT_DIR / "highlight_events.json")

    print(f"\nGallery composition ({len(gallery_rows)} events):")
    reasons = pd.Series([r["highlight_reason"] for r in gallery_rows]).value_counts()
    print(reasons.to_string())


if __name__ == "__main__":
    main()
