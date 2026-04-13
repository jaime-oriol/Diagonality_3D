"""Fix the carry → possession.sum_xg_ind link in `test/dos_validation_raw.csv`.

Why this exists:
  The DFL kpi_data XML `TeamPossession > PossessionEvent` lists only the
  pass and shot events inside each possession — carries and receptions are
  never referenced. When `validate_dos_outcomes.py` built the xg map via
  `event_id -> sum_xg_ind`, every carry silently got 0.0, which depressed
  `chance_rate` in the carry subset and inflated the sample size of the
  "no-chance" pool in Mann-Whitney tests.

Fix:
  For every event whose id is NOT already in the xg map, look up its
  parent possession by:
     (a) matching on team_id (both are DFL-CLU-*)
     (b) requiring the event's parquet_frame to fall inside the possession's
         frame range, which we reconstruct from the events.parquet rows
         whose ids DO appear in the possession's event_ids list.

Produces:
  test/dos_validation_raw_fixed.csv      (same rows + corrected xg column)
  test/dos_validation_report_fixed.md    (full report recomputed on fixed data)
  test/dos_validation_quintiles_fixed.png
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd

from src.loader import MATCHES
from src.preprocess import CACHE_DIR

RAW = Path("test/dos_validation_raw.csv")
OUT_RAW = Path("test/dos_validation_raw_fixed.csv")
OUT_REPORT = Path("test/dos_validation_report_fixed.md")
OUT_PNG = Path("test/dos_validation_quintiles_fixed.png")


def _build_possession_intervals(match: str) -> pd.DataFrame:
    """For `match`, compute per-possession (frame_lo, frame_hi, team_id,
    sum_xg_ind) using the frames of the events referenced inside it."""
    pos = pd.read_parquet(CACHE_DIR / match / "possessions.parquet")
    ev = pd.read_parquet(CACHE_DIR / match / "events.parquet")
    ev_frame = dict(zip(ev["event_id"].astype(str),
                        ev["parquet_frame"].fillna(-1).astype(int)))
    rows = []
    for _, r in pos.iterrows():
        ids = [str(e) for e in r["event_ids"]]
        frames = [ev_frame[i] for i in ids if i in ev_frame and ev_frame[i] > 0]
        if not frames:
            continue
        rows.append({
            "team_id": r["team_id"],
            "frame_lo": int(min(frames)),
            "frame_hi": int(max(frames)),
            "sum_xg_ind": float(r["sum_xg_ind"]),
        })
    return pd.DataFrame(rows).sort_values(["team_id", "frame_lo"]).reset_index(drop=True)


def _fix_match(df: pd.DataFrame, match: str) -> pd.DataFrame:
    """In-place-ish fix for one match subset. Returns the subset with
    parent_possession_xg recomputed for rows where it was 0.0."""
    sub = df[df["match"] == match].copy()
    if len(sub) == 0:
        return sub
    intervals = _build_possession_intervals(match)
    if len(intervals) == 0:
        return sub

    # Build a per-team IntervalIndex for fast containment lookup
    per_team = {}
    for team, g in intervals.groupby("team_id"):
        g = g.reset_index(drop=True)
        per_team[team] = (
            pd.IntervalIndex.from_arrays(g["frame_lo"], g["frame_hi"],
                                         closed="both"),
            g["sum_xg_ind"].values,
        )

    fixed = sub["parent_possession_xg"].values.copy()
    n_fixed = 0
    for i, row in enumerate(sub.itertuples(index=False)):
        # Only touch rows whose current xg is 0 (the broken-link cases)
        if fixed[i] > 0:
            continue
        team = row.team_id
        if team not in per_team:
            continue
        idx_iv, xgs = per_team[team]
        hits = idx_iv.contains(int(row.frame))
        if hits.any():
            # Take the possession with the largest overlap: we just pick the
            # first match (they're sorted by frame_lo and non-overlapping
            # within a team).
            j = int(np.argmax(hits))
            fixed[i] = float(xgs[j])
            if fixed[i] != row.parent_possession_xg:
                n_fixed += 1

    sub["parent_possession_xg"] = fixed
    print(f"  [{match}] rows fixed: {n_fixed} / {len(sub)}")
    return sub


def _recompute_stats(df: pd.DataFrame) -> str:
    """Recompute all stats on the fixed dataframe and render a markdown report."""
    from scipy.stats import mannwhitneyu, pearsonr, spearmanr
    import statsmodels.api as sm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["chance"] = df["parent_possession_xg"] > 0
    df["success"] = df["evaluation"] == "successfullyCompleted"

    def mw(mask: pd.Series, label: str):
        a = df.loc[mask, "dos"].dropna().values
        b = df.loc[~mask, "dos"].dropna().values
        if len(a) < 5 or len(b) < 5:
            return {"label": label, "n_pos": len(a), "n_neg": len(b),
                    "p": float("nan"), "rbc": float("nan"),
                    "m_pos": float("nan"), "m_neg": float("nan")}
        u, p = mannwhitneyu(a, b, alternative="greater")
        rbc = (2.0 * u) / (len(a) * len(b)) - 1.0
        return {"label": label, "n_pos": len(a), "n_neg": len(b),
                "p": float(p), "rbc": float(rbc),
                "m_pos": float(np.mean(a)), "m_neg": float(np.mean(b)),
                "med_pos": float(np.median(a)), "med_neg": float(np.median(b))}

    mw_results = [
        mw(df["chance"], "led_to_chance (xg>0) — ALL events"),
        mw(df["back_line_break"], "back_line_break — ALL events"),
        mw(df[df["event_type"] == "pass"]["chance"].reindex(df.index, fill_value=False),
           "led_to_chance — passes only"),
    ]
    # Passes-only, carries-only, split
    pases = df[df["event_type"] == "pass"]
    carries = df[df["event_type"] == "carry"]
    p_mw = mannwhitneyu(pases.loc[pases["chance"], "dos"],
                        pases.loc[~pases["chance"], "dos"],
                        alternative="greater")
    c_mw = mannwhitneyu(carries.loc[carries["chance"], "dos"],
                        carries.loc[~carries["chance"], "dos"],
                        alternative="greater") if carries["chance"].sum() >= 5 else (0.0, float("nan"))
    mw_results.append({"label": "passes only — led_to_chance",
                       "n_pos": int(pases["chance"].sum()),
                       "n_neg": int((~pases["chance"]).sum()),
                       "p": float(p_mw.pvalue), "rbc": float("nan"),
                       "m_pos": float(pases.loc[pases["chance"], "dos"].mean()),
                       "m_neg": float(pases.loc[~pases["chance"], "dos"].mean()),
                       "med_pos": float(pases.loc[pases["chance"], "dos"].median()),
                       "med_neg": float(pases.loc[~pases["chance"], "dos"].median())})
    mw_results.append({"label": "carries only — led_to_chance",
                       "n_pos": int(carries["chance"].sum()),
                       "n_neg": int((~carries["chance"]).sum()),
                       "p": float(c_mw[1] if hasattr(c_mw, 'pvalue') else c_mw.pvalue),
                       "rbc": float("nan"),
                       "m_pos": float(carries.loc[carries["chance"], "dos"].mean()) if carries["chance"].any() else float("nan"),
                       "m_neg": float(carries.loc[~carries["chance"], "dos"].mean()) if (~carries["chance"]).any() else float("nan"),
                       "med_pos": float(carries.loc[carries["chance"], "dos"].median()) if carries["chance"].any() else float("nan"),
                       "med_neg": float(carries.loc[~carries["chance"], "dos"].median()) if (~carries["chance"]).any() else float("nan")})

    # xG correlations
    m = ~(df["dos"].isna() | df["parent_possession_xg"].isna())
    pr, pp = pearsonr(df.loc[m, "dos"], df.loc[m, "parent_possession_xg"])
    sr, sp = spearmanr(df.loc[m, "dos"], df.loc[m, "parent_possession_xg"])

    # Quintiles
    df["dos_q"] = pd.qcut(df["dos"], 5,
                          labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                          duplicates="drop")
    q = df.groupby("dos_q", observed=True).agg(
        n=("dos", "size"),
        dos_mean=("dos", "mean"),
        success_rate=("evaluation", lambda s: float((s == "successfullyCompleted").mean())),
        line_break_rate=("back_line_break", "mean"),
        chance_rate=("parent_possession_xg", lambda s: float((s > 0).mean())),
        xg_mean=("parent_possession_xg", "mean"),
    )

    # Plot quintiles
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title, color) in zip(axes, [
        ("success_rate", "Pass success rate", "tab:blue"),
        ("line_break_rate", "Back line break rate", "tab:orange"),
        ("chance_rate", "Possession ended in shot (fixed)", "tab:red"),
    ]):
        vals = q[col].values
        ax.bar(range(len(vals)), vals, color=color, alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(list(q.index), rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("DOS quintiles vs outcomes — fixed carry→possession link",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Direction class
    dc = df.groupby("direction_class").agg(
        n=("dos", "size"),
        dos_mean=("dos", "mean"),
        success_rate=("evaluation", lambda s: float((s == "successfullyCompleted").mean())),
        line_break_rate=("back_line_break", "mean"),
        chance_rate=("parent_possession_xg", lambda s: float((s > 0).mean())),
        awareness_mean=("awareness_mean", "mean"),
    ).reindex(["forward", "diagonal", "sideways"])

    # Logistic on passes only (back_line_break, success, chance) with the fix
    def fit_logit(target):
        sub = df[df["event_type"] == "pass"].dropna(
            subset=["dos", target, "distance", "pressure_player",
                    "n_def_lane", "n_def_goal"]).copy()
        if len(sub) < 50 or sub[target].nunique() < 2:
            return None
        X = sm.add_constant(sub[["dos", "distance", "pressure_player",
                                 "n_def_lane", "n_def_goal"]].astype(float))
        y = sub[target].astype(int)
        try:
            return sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
        except Exception:
            return None

    logit_back = fit_logit("back_line_break")
    logit_chance = fit_logit("chance")

    # ── Build markdown ──
    lines = []
    lines.append("# DOS empirical validation — FIXED carry↔possession link")
    lines.append("")
    lines.append("_Recomputed from `dos_validation_raw.csv` after fixing the "
                 "DFL carry/possession linkage via frame-range containment. "
                 "No DOS values were recomputed; only `parent_possession_xg` "
                 "was corrected for carry rows (the XML omits carries from "
                 "`TeamPossession > PossessionEvent`, so the naive id join "
                 "returned 0.0 for every carry)._")
    lines.append("")
    lines.append(f"- Events evaluated: **{len(df):,}** "
                 f"(passes={int((df['event_type']=='pass').sum())}, "
                 f"carries={int((df['event_type']=='carry').sum())})")
    lines.append(f"- Events with `parent_possession_xg > 0` after fix: "
                 f"**{int((df['parent_possession_xg']>0).sum()):,}** "
                 f"(was {718} before; carries now properly linked).")
    lines.append("")

    lines.append("## Mann-Whitney U (fixed)")
    lines.append("")
    lines.append("| Outcome | n+ | n− | mean (+) | mean (−) | median (+) | median (−) | rbc | p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in mw_results:
        lines.append(f"| {r['label']} | {r['n_pos']} | {r['n_neg']} | "
                     f"{r.get('m_pos', float('nan')):.4f} | "
                     f"{r.get('m_neg', float('nan')):.4f} | "
                     f"{r.get('med_pos', float('nan')):.4f} | "
                     f"{r.get('med_neg', float('nan')):.4f} | "
                     f"{r.get('rbc', float('nan')):+.3f} | "
                     f"{r['p']:.2e} |")
    lines.append("")

    lines.append("## Correlation DOS ↔ parent xG (continuous)")
    lines.append("")
    lines.append(f"- Pearson  r = **{pr:+.4f}**, p = {pp:.2e}")
    lines.append(f"- Spearman ρ = **{sr:+.4f}**, p = {sp:.2e}")
    lines.append(f"- N = {int(m.sum())}")
    lines.append("")

    lines.append("## Quintiles (fixed)")
    lines.append("")
    lines.append("```")
    lines.append(q.to_string())
    lines.append("```")
    lines.append("")
    lines.append("![quintiles](dos_validation_quintiles_fixed.png)")
    lines.append("")

    lines.append("## Direction class (fixed)")
    lines.append("")
    lines.append("```")
    lines.append(dc.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Logistic regression (passes only, fixed xG for parent)")
    lines.append("")
    for name, fit in [("back_line_break", logit_back), ("chance", logit_chance)]:
        lines.append(f"### Target: `{name}`")
        if fit is None:
            lines.append("_fit skipped_")
            continue
        lines.append(f"- n = {int(fit.nobs)}, Pseudo R² = {fit.prsquared:.4f}, "
                     f"LLR p = {fit.llr_pvalue:.2e}")
        lines.append("")
        lines.append("| coef | value | p |")
        lines.append("|---|---:|---:|")
        for k in ["const", "dos", "distance", "pressure_player",
                  "n_def_lane", "n_def_goal"]:
            if k in fit.params.index:
                lines.append(f"| {k} | {fit.params[k]:+.4f} | "
                             f"{fit.pvalues[k]:.2e} |")
        lines.append("")

    return "\n".join(lines)


def main():
    print(f"Loading raw: {RAW}")
    df = pd.read_csv(RAW)
    print(f"  {len(df):,} rows, {(df['parent_possession_xg']>0).sum()} with xg>0")

    parts = []
    for m in df["match"].unique():
        parts.append(_fix_match(df, m))
    fixed = pd.concat(parts, ignore_index=True)
    print(f"After fix: {(fixed['parent_possession_xg']>0).sum()} rows with xg>0")

    fixed.to_csv(OUT_RAW, index=False)
    print(f"Wrote: {OUT_RAW}")

    report = _recompute_stats(fixed)
    OUT_REPORT.write_text(report)
    print(f"Wrote: {OUT_REPORT}")
    print(f"Wrote: {OUT_PNG}")


if __name__ == "__main__":
    main()
