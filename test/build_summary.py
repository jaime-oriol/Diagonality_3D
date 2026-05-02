"""Build `outputs/SUMMARY.md` — single-page index of every artefact the
AWS pipeline produced. Designed to be read top-to-bottom for the
storytelling document drafting + slide deck preparation.

Sections:
  1. Headline cifras (10-12 quotable numbers extracted from the reports)
  2. Per-direction-class breakdown
  3. Top players (DOS, xT, D-Def, wrongfooting, diagonal share)
  4. Top teams
  5. Per-match summary
  6. Top events for video gallery
  7. Direct links to every report, table, video, frame.

This file is the single entrypoint for the narrative writer.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_MD = Path("outputs/SUMMARY.md")
TABLES_DIR = Path("outputs/tables")
REPORTS_DIR = Path("outputs/reports")
VIDEOS_DIR = Path("outputs/videos")
FRAMES_DIR = Path("outputs/frames")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _list_files(folder: Path, pattern: str = "*") -> list:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern))


def _fmt_pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x*100:.1f}%"


def _fmt_4(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.4f}"


def _build_headlines(direction_breakdown: pd.DataFrame,
                      teams: pd.DataFrame,
                      players_dos: pd.DataFrame,
                      zones_origin: pd.DataFrame,
                      zones_x_direction: pd.DataFrame) -> list:
    """Top 10-15 quotable cifras for the narrative document."""
    lines = []

    # ── Direction-class cifras (the SV thesis) ─────────────────────
    if len(direction_breakdown):
        d = direction_breakdown.set_index("direction_class")
        if "diagonal" in d.index and "forward" in d.index:
            lines.append(
                f"- **Diagonal pos-DOS rate**: {_fmt_pct(d.loc['diagonal','dos_pos_rate'])} "
                f"vs forward {_fmt_pct(d.loc['forward','dos_pos_rate'])} "
                f"vs sideways {_fmt_pct(d.loc['sideways','dos_pos_rate']) if 'sideways' in d.index else 'n/a'}.")
            if "xt_delta_mean" in d.columns:
                xt_side = d.loc["sideways", "xt_delta_mean"] if "sideways" in d.index else float("nan")
                lines.append(
                    f"- **Mean xT-delta by direction**: diagonal {_fmt_4(d.loc['diagonal','xt_delta_mean'])}, "
                    f"forward {_fmt_4(d.loc['forward','xt_delta_mean'])}, "
                    f"sideways {_fmt_4(xt_side)}.")
            if "biaxial_mean" in d.columns:
                lines.append(
                    f"- **Bi-axial disruption (D-Def PC1+PC2 balance)**: "
                    f"diagonal {_fmt_4(d.loc['diagonal','biaxial_mean'])}, "
                    f"forward {_fmt_4(d.loc['forward','biaxial_mean'])} — "
                    f"diagonals break BOTH axes simultaneously.")
            if "awareness_mean" in d.columns:
                lines.append(
                    f"- **Defender awareness (cognitive layer)**: "
                    f"diagonals find defenders {_fmt_4(d.loc['diagonal','awareness_mean'])} "
                    f"vs forwards {_fmt_4(d.loc['forward','awareness_mean'])} "
                    f"(lower awareness = more blindness exploited).")
            if "pct_nearest_blind" in d.columns:
                lines.append(
                    f"- **Nearest defender in his blind half**: "
                    f"diagonals {_fmt_pct(d.loc['diagonal','pct_nearest_blind'])} "
                    f"vs forward {_fmt_pct(d.loc['forward','pct_nearest_blind'])}.")
            if "n_wrongfooted_mean" in d.columns:
                lines.append(
                    f"- **Wrongfooted defenders per action** (>60° misaligned): "
                    f"diagonals {_fmt_4(d.loc['diagonal','n_wrongfooted_mean'])} "
                    f"vs forwards {_fmt_4(d.loc['forward','n_wrongfooted_mean'])}.")

    # ── Half-space cifras (Spielverlagerung's signature claim) ─────
    if len(zones_origin):
        z = zones_origin.set_index("zone")
        if "half_space" in z.index and "wing" in z.index:
            lines.append(
                f"- **Half-space vs wing origin** (SV thesis): "
                f"DOS {_fmt_4(z.loc['half_space','dos_mean'])} from half-space "
                f"vs {_fmt_4(z.loc['wing','dos_mean'])} from wing; "
                f"xT-delta {_fmt_4(z.loc['half_space','xt_delta_mean'])} vs "
                f"{_fmt_4(z.loc['wing','xt_delta_mean'])}.")
    if len(zones_x_direction):
        # The SV signature: diagonal-from-half-space
        diag_hs = zones_x_direction[
            (zones_x_direction["origin_zone"] == "half_space")
            & (zones_x_direction["direction_class"] == "diagonal")]
        diag_wing = zones_x_direction[
            (zones_x_direction["origin_zone"] == "wing")
            & (zones_x_direction["direction_class"] == "diagonal")]
        if len(diag_hs) and len(diag_wing):
            lines.append(
                f"- **Diagonal from half-space vs from wing**: "
                f"DOS {_fmt_4(diag_hs.iloc[0]['dos_mean'])} vs "
                f"{_fmt_4(diag_wing.iloc[0]['dos_mean'])} "
                f"(n={int(diag_hs.iloc[0]['n']):,} vs {int(diag_wing.iloc[0]['n']):,}).")

    # ── Top players + teams ─────────────────────────────────────────
    if len(teams):
        best = teams.iloc[0]
        lines.append(
            f"- **Top team by mean DOS**: {best['team_name']} "
            f"(mean DOS {_fmt_4(best['dos_mean'])}, n={int(best['n_events']):,} events, "
            f"diagonal share {_fmt_pct(best['diag_share'])}).")
    if len(players_dos):
        top3 = players_dos.head(3)
        names = ", ".join(
            f"{r['player_name']} ({r['team_name']}, {_fmt_4(r['dos_mean'])})"
            for _, r in top3.iterrows())
        lines.append(f"- **Top 3 players by mean DOS** (min 30 events): {names}.")
    return lines


def _md_table(df: pd.DataFrame, cols: list, max_rows: int = 10) -> str:
    if len(df) == 0:
        return "_(empty)_"
    sub = df.head(max_rows)[cols]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in sub.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section_header(title: str, anchor: str = None) -> str:
    if anchor:
        return f"\n## {title}  <a id='{anchor}'></a>\n"
    return f"\n## {title}\n"


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    # ── Load every artefact we need ─────────────────────────────────
    direction = _read_csv(TABLES_DIR / "direction_breakdown.csv")
    teams = _read_csv(TABLES_DIR / "teams_summary.csv")
    matches = _read_csv(TABLES_DIR / "matches_summary.csv")
    pl_dos = _read_csv(TABLES_DIR / "players_by_dos.csv")
    pl_xt = _read_csv(TABLES_DIR / "players_by_xt.csv")
    pl_ddef = _read_csv(TABLES_DIR / "players_by_ddef.csv")
    pl_wf = _read_csv(TABLES_DIR / "players_by_wrongfooting.csv")
    pl_aw = _read_csv(TABLES_DIR / "players_by_awareness_low.csv")
    pl_diag = _read_csv(TABLES_DIR / "players_by_diag_share.csv")
    pl_vol = _read_csv(TABLES_DIR / "players_by_volume.csv")
    zones_origin = _read_csv(TABLES_DIR / "zones_origin_breakdown.csv")
    zones_dest = _read_csv(TABLES_DIR / "zones_dest_breakdown.csv")
    zones_xdir = _read_csv(TABLES_DIR / "zones_x_direction.csv")
    top_full = _read_csv(TABLES_DIR / "top_events_full_vector.csv")
    top_dos_ev = _read_json(TABLES_DIR / "top_dos_events.json")
    top_xt_ev = _read_json(TABLES_DIR / "top_xt_events.json")
    top_ddef_ev = _read_json(TABLES_DIR / "top_ddef_events.json")
    gallery_ev = _read_json(TABLES_DIR / "highlight_events.json")

    videos = _list_files(VIDEOS_DIR, "*.mp4")
    frames_dos = _list_files(FRAMES_DIR, "*_dos.png")
    frames_ppcf = _list_files(FRAMES_DIR, "*_ppcf.png")
    pass_maps = _list_files(FRAMES_DIR / "pass_maps", "*.png")

    headlines = _build_headlines(direction, teams, pl_dos, zones_origin, zones_xdir)

    # ── Build the markdown ──────────────────────────────────────────
    md = []
    md.append("# Diagonality 3D — AWS pipeline summary\n")
    md.append("Single-page index of every artefact produced by "
              "`scripts/aws_pipeline.py`. All numbers reproducible "
              "from the cached event-level CSV `dos_validation_full.csv`.\n")

    md.append("\n## Headline cifras (storytelling)\n")
    if headlines:
        md.extend(headlines)
    else:
        md.append("_(no data)_")

    md.append(_section_header("Direction-class breakdown"))
    cols = [c for c in ["direction_class", "n", "dos_mean", "dos_pos_rate",
                        "awareness_mean", "xt_delta_mean", "ddef_3s_mean",
                        "biaxial_mean", "n_wrongfooted_mean", "pct_nearest_blind",
                        "success_rate", "back_line_break_rate"] if c in direction.columns]
    md.append(_md_table(direction, cols, max_rows=10))

    md.append(_section_header("Zone breakdown (Spielverlagerung half-spaces)"))
    md.append("Origin zone — where the action begins:")
    cols_z = [c for c in ["zone", "n", "dos_mean", "awareness_mean",
                          "xt_delta_mean", "diag_share", "success_rate",
                          "back_line_break_rate"] if c in zones_origin.columns]
    md.append(_md_table(zones_origin, cols_z, max_rows=5))
    md.append("\nDestination zone:")
    cols_z = [c for c in ["zone", "n", "dos_mean", "awareness_mean",
                          "xt_delta_mean", "diag_share", "success_rate",
                          "back_line_break_rate"] if c in zones_dest.columns]
    md.append(_md_table(zones_dest, cols_z, max_rows=5))
    md.append("\nZone × direction interaction (the SV signature combination):")
    cols_zd = [c for c in ["origin_zone", "direction_class", "n",
                            "dos_mean", "xt_delta_mean", "awareness_mean",
                            "success_rate"] if c in zones_xdir.columns]
    md.append(_md_table(zones_xdir, cols_zd, max_rows=20))

    md.append(_section_header("Top players"))
    if len(pl_dos):
        md.append("### By mean DOS (min 30 events)")
        cols = [c for c in ["player_name", "team_name", "player_position",
                            "n_events", "dos_mean", "diag_share",
                            "xt_delta_mean", "ddef_3s_mean"] if c in pl_dos.columns]
        md.append(_md_table(pl_dos, cols, max_rows=15))
    if len(pl_xt):
        md.append("\n### By total xT-delta gained")
        cols = [c for c in ["player_name", "team_name",
                            "n_events", "xt_delta_sum", "xt_delta_mean",
                            "dos_mean", "diag_share"] if c in pl_xt.columns]
        md.append(_md_table(pl_xt, cols, max_rows=10))
    if len(pl_ddef):
        md.append("\n### By mean D-Def 3s disruption caused")
        cols = [c for c in ["player_name", "team_name",
                            "n_events", "ddef_3s_mean", "biaxial_mean",
                            "dos_mean"] if c in pl_ddef.columns]
        md.append(_md_table(pl_ddef, cols, max_rows=10))
    if len(pl_wf):
        md.append("\n### By mean wrongfooted defenders (theta)")
        cols = [c for c in ["player_name", "team_name",
                            "n_events", "n_wrongfooted_mean",
                            "pct_nearest_blind", "dos_mean"] if c in pl_wf.columns]
        md.append(_md_table(pl_wf, cols, max_rows=10))
    if len(pl_aw):
        md.append("\n### By LOWEST defender awareness exploited (cognitive layer)")
        cols = [c for c in ["player_name", "team_name",
                            "n_events", "awareness_mean", "detection_delay_mean_s",
                            "dos_mean"] if c in pl_aw.columns]
        md.append(_md_table(pl_aw, cols, max_rows=10))
    if len(pl_diag):
        md.append("\n### By diagonal share of all actions")
        cols = [c for c in ["player_name", "team_name",
                            "n_events", "diag_share", "fwd_share",
                            "dos_mean"] if c in pl_diag.columns]
        md.append(_md_table(pl_diag, cols, max_rows=10))

    md.append(_section_header("Teams summary"))
    if len(teams):
        cols = [c for c in ["team_name", "n_events", "dos_mean",
                            "xt_delta_mean", "ddef_3s_mean", "biaxial_mean",
                            "diag_share", "success_rate"] if c in teams.columns]
        md.append(_md_table(teams, cols, max_rows=20))

    md.append(_section_header("Match summary"))
    if len(matches):
        cols = [c for c in ["match", "n_events", "dos_mean", "diag_share",
                            "top_dos_player", "top_dos_value"] if c in matches.columns]
        md.append(_md_table(matches, cols, max_rows=20))

    md.append(_section_header("Top events selected for video"))
    md.append("Composite score = `DOS + 5·xT_delta + 0.05·back_line_break` "
              "— picks events that combine mechanistic interest (DOS) with "
              "real impact (xT-delta and direct back-line breaks). Pure DOS-"
              "only ranking is in `outputs/tables/top_events_full_vector.csv`.\n")
    md.append(f"_(see `outputs/tables/top_dos_events.json` for full metadata, "
              f"{len(top_dos_ev)} entries)_\n")
    if top_dos_ev:
        md.append("| # | match | player | type | frame | DOS | xT-delta | back-line break |")
        md.append("|---:|---|---|---|---:|---:|---:|---|")
        for i, ev in enumerate(top_dos_ev[:15]):
            md.append(
                f"| {i+1} | {ev.get('match','')} | {ev.get('player_name','')} | "
                f"{ev.get('event_type','')} | {ev.get('frame','')} | "
                f"{ev.get('dos', float('nan')):.4f} | "
                f"{ev.get('xt_delta', float('nan')):.4f} | "
                f"{'YES' if ev.get('back_line_break') else 'no'} |"
            )

    md.append(_section_header("Files index (every artefact)"))
    md.append("\n### Reports")
    for r in [REPORTS_DIR / f for f in
              ("report_xt.md", "report_ddef.md", "report_theta.md")]:
        if r.exists():
            md.append(f"- [{r.name}]({r.relative_to(OUT_MD.parent)})")

    md.append("\n### Tables (CSV)")
    for f in _list_files(TABLES_DIR, "*.csv"):
        md.append(f"- [{f.name}]({f.relative_to(OUT_MD.parent)})")

    md.append(f"\n### Videos ({len(videos)})")
    for v in videos:
        size_mb = v.stat().st_size / 1024 / 1024
        md.append(f"- [{v.name}]({v.relative_to(OUT_MD.parent)}) ({size_mb:.1f} MB)")

    md.append(f"\n### Frame galleries")
    md.append(f"- DOS frames: {len(frames_dos)} files in `outputs/frames/`")
    md.append(f"- PPCF frames: {len(frames_ppcf)} files in `outputs/frames/`")
    md.append(f"- Pass maps: {len(pass_maps)} files in `outputs/frames/pass_maps/`")
    if pass_maps:
        md.append("\n#### Pass maps rendered")
        for f in pass_maps:
            md.append(f"- [{f.name}]({f.relative_to(OUT_MD.parent)})")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote: {OUT_MD}")
    print(f"Total length: {len(md)} lines")


if __name__ == "__main__":
    main()
