"""Recompute the empirical-validation statistics with xT-delta as the
outcome metric, on top of `dos_validation_raw_xt.csv`.

This replaces the suffix `_fixed` (parent_possession_xg-based) with an
xT-based pipeline. No DOS values are recomputed — only the OUTCOME
variable in the Mann-Whitney U / logistic regression / quintile chart.

Per event-type the outcome is:
  - pass:   xt_delta  (xT(dest) − xT(origin); fail → −xT(origin))
  - carry:  xt_delta
  - takeon: xt_origin (xT of the duel zone — value preserved)

Outputs:
  outputs/intermediate/dos_validation_quintiles_xt.png
  outputs/intermediate/dos_validation_report_xt.md
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import statsmodels.api as sm


RAW_XT = Path("outputs/intermediate/dos_validation_raw_xt.csv")
OUT_PNG = Path("outputs/intermediate/dos_validation_quintiles_xt.png")
OUT_MD = Path("outputs/intermediate/dos_validation_report_xt.md")
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def _outcome(df: pd.DataFrame) -> pd.Series:
    """Per event-type outcome:
      pass / carry → xt_delta
      takeon       → xt_origin   (no displacement; value preserved)
    """
    out = df["xt_delta"].copy()
    is_takeon = df["event_type"] == "takeon"
    out.loc[is_takeon] = df.loc[is_takeon, "xt_origin"]
    return out


def _mwu(df: pd.DataFrame, mask: pd.Series, label: str, key: str = "dos") -> dict:
    """One-sided Mann-Whitney U: is `key` higher when mask=True than when False?"""
    a = df.loc[mask, key].dropna().to_numpy()
    b = df.loc[~mask, key].dropna().to_numpy()
    if len(a) < 5 or len(b) < 5:
        return {"label": label, "n_pos": int(len(a)), "n_neg": int(len(b)),
                "p": float("nan"), "rbc": float("nan"),
                "med_pos": float("nan"), "med_neg": float("nan")}
    u, p = mannwhitneyu(a, b, alternative="greater")
    rbc = (2.0 * u) / (len(a) * len(b)) - 1.0
    return {
        "label": label, "n_pos": int(len(a)), "n_neg": int(len(b)),
        "u": float(u), "p": float(p), "rbc": float(rbc),
        "med_pos": float(np.median(a)), "med_neg": float(np.median(b)),
        "mean_pos": float(np.mean(a)), "mean_neg": float(np.mean(b)),
    }


def _spearman(df: pd.DataFrame, label: str, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    if len(sub) < 30:
        return {"label": label, "n": len(sub), "rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr(sub[x], sub[y])
    pr, pp = pearsonr(sub[x], sub[y])
    return {"label": label, "n": int(len(sub)),
            "rho": float(rho), "p_spearman": float(p),
            "r": float(pr), "p_pearson": float(pp)}


def _logistic(df: pd.DataFrame, target: str, label: str) -> dict:
    sub = df.dropna(subset=["dos", target, "distance",
                            "pressure_player", "n_def_lane", "n_def_goal"]).copy()
    if len(sub) < 50 or sub[target].nunique() < 2:
        return {"label": label, "n": int(len(sub)), "fit": False}
    X = sm.add_constant(sub[["dos", "distance", "pressure_player",
                              "n_def_lane", "n_def_goal"]].astype(float))
    y = sub[target].astype(int)
    try:
        res = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
    except Exception as ex:
        return {"label": label, "n": int(len(sub)), "fit": False, "err": str(ex)}
    return {
        "label": label, "n": int(len(sub)), "fit": True,
        "coefs": dict(zip(X.columns, res.params.tolist())),
        "pvalues": dict(zip(X.columns, res.pvalues.tolist())),
        "pseudo_r2": float(res.prsquared),
        "llr_p": float(res.llr_pvalue),
    }


def _quintiles(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """DOS quintiles with the new xT-based outcome rates."""
    sub = df.dropna(subset=["dos", outcome_col]).copy()
    sub["dos_q"] = pd.qcut(sub["dos"], 5, labels=[
        "Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"], duplicates="drop")
    rows = []
    for q, g in sub.groupby("dos_q", observed=True):
        out_pos = (g[outcome_col] > 0).astype(float).values
        rows.append({
            "dos_q": str(q),
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "outcome_mean": float(g[outcome_col].mean()),
            "outcome_pos_rate": float(out_pos.mean()) if len(out_pos) else float("nan"),
            "outcome_p95": float(g[outcome_col].quantile(0.95)),
        })
    return pd.DataFrame(rows).set_index("dos_q")


def _direction_breakdown(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    rows = []
    for cls, g in df.groupby("direction_class"):
        rows.append({
            "direction_class": cls,
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "outcome_mean": float(g[outcome_col].mean()),
            "outcome_pos_rate": float((g[outcome_col] > 0).mean()),
            "awareness_mean": float(g["awareness_mean"].mean()),
        })
    return pd.DataFrame(rows).set_index("direction_class").reindex(
        ["forward", "diagonal", "sideways"]).dropna(how="all")


def _per_event_type(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    rows = []
    for et, g in df.groupby("event_type"):
        rows.append({
            "event_type": et,
            "n": int(len(g)),
            "dos_mean": float(g["dos"].mean()),
            "xt_origin_mean": float(g["xt_origin"].mean()),
            "outcome_mean": float(g[outcome_col].mean()),
            "outcome_pos_rate": float((g[outcome_col] > 0).mean()),
            "diag_share": float((g["direction_class"] == "diagonal").mean()),
        })
    return pd.DataFrame(rows).set_index("event_type")


def _fmt_p(p):
    if p != p:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _plot_quintiles(quintiles: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = list(quintiles.index)
    x = np.arange(len(labels))
    metrics = [
        ("outcome_mean", "Mean xT-based outcome", "tab:blue"),
        ("outcome_pos_rate", "Outcome > 0 rate", "tab:orange"),
        ("dos_mean", "DOS mean (sanity)", "tab:gray"),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        ax.bar(x, quintiles[col].values, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("DOS quintiles vs xT-based outcomes (5 matches)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    df: pd.DataFrame, outcome_col: str,
    mw_results: list, log_results: list, sp_results: list,
    quintiles: pd.DataFrame, direction_table: pd.DataFrame,
    per_et: pd.DataFrame, out_path: Path,
):
    lines = []
    lines.append("# DOS empirical validation — xT-DELTA outcome")
    lines.append("")
    lines.append("Outcome metric replaced: **`parent_possession_xg`** → "
                 "**`xt_delta`** (Karun Singh 2018 12×8 grid). "
                 "Take-ons use `xt_origin` (no displacement, value preserved).")
    lines.append("")
    lines.append("- Events: " + ", ".join(
        [f"{k}={int(per_et.loc[k,'n'])}" for k in per_et.index]))
    lines.append(f"- Total rows with valid outcome: {df.dropna(subset=[outcome_col]).shape[0]:,}")
    lines.append("")

    lines.append("## Mann-Whitney U: DOS higher when outcome > 0?")
    lines.append("")
    lines.append("One-sided alternative: events with positive xT-based "
                 "outcome have **higher DOS** than those with non-positive.")
    lines.append("")
    lines.append("| Subset | n+ | n− | mean DOS (+) | mean DOS (−) | rbc | U | p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in mw_results:
        if "u" not in r:
            lines.append(f"| {r['label']} | {r['n_pos']} | {r['n_neg']} | "
                         f"n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {r['label']} | {r['n_pos']} | {r['n_neg']} | "
            f"{r['mean_pos']:.4f} | {r['mean_neg']:.4f} | "
            f"{r['rbc']:+.3f} | {r['u']:.0f} | {_fmt_p(r['p'])} |"
        )
    lines.append("")

    lines.append("## Continuous correlation DOS ↔ outcome")
    lines.append("")
    lines.append("| Subset | n | Spearman ρ | p (spearman) | Pearson r | p (pearson) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in sp_results:
        if "rho" not in r or r["rho"] != r["rho"]:
            lines.append(f"| {r['label']} | {r.get('n','-')} | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {r['label']} | {r['n']} | {r['rho']:+.4f} | "
                         f"{_fmt_p(r['p_spearman'])} | {r['r']:+.4f} | "
                         f"{_fmt_p(r['p_pearson'])} |")
    lines.append("")

    lines.append("## Logistic regression: P(outcome > 0) ~ dos + controls")
    lines.append("")
    lines.append("Controls: distance, pressure_player, n_def_lane, n_def_goal.")
    lines.append("If DOS coef stays significant after adjustment → DOS captures "
                 "something beyond geometry/pressure/defender count.")
    lines.append("")
    for r in log_results:
        lines.append(f"### Subset: `{r['label']}`")
        lines.append("")
        if not r.get("fit"):
            lines.append(f"_Fit skipped (n={r['n']}{', err='+r.get('err','') if r.get('err') else ''})._")
            lines.append("")
            continue
        lines.append(f"- N={r['n']}, Pseudo R²={r['pseudo_r2']:.4f}, "
                     f"LLR p={_fmt_p(r['llr_p'])}")
        lines.append("")
        lines.append("| coef | value | p |")
        lines.append("|---|---:|---:|")
        for k in ["const", "dos", "distance", "pressure_player",
                  "n_def_lane", "n_def_goal"]:
            if k in r["coefs"]:
                lines.append(f"| {k} | {r['coefs'][k]:+.4f} | "
                             f"{_fmt_p(r['pvalues'][k])} |")
        lines.append("")

    lines.append("## DOS quintiles (xT outcome)")
    lines.append("")
    lines.append("```")
    lines.append(quintiles.to_string())
    lines.append("```")
    lines.append("")
    lines.append("![quintiles](dos_validation_quintiles_xt.png)")
    lines.append("")

    lines.append("## Direction class breakdown (xT outcome)")
    lines.append("")
    lines.append("```")
    lines.append(direction_table.to_string())
    lines.append("```")
    lines.append("")

    lines.append("## Per event-type breakdown")
    lines.append("")
    lines.append("```")
    lines.append(per_et.to_string())
    lines.append("```")

    out_path.write_text("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────────

def main():
    if not RAW_XT.exists():
        raise SystemExit(f"Missing {RAW_XT}. Run `enrich_with_xt.py` first.")
    df = pd.read_csv(RAW_XT)
    print(f"Loaded {RAW_XT}: {len(df):,} rows")

    # Per-event-type outcome (already injected via _outcome helper)
    df["outcome"] = _outcome(df)
    # Binary "positive outcome" for MWU/logit:
    #   - passes/carries: outcome > 0  (xT was gained)
    #   - takeons:        outcome > median(takeon outcomes)   (TAKE-ON is in
    #     a higher-than-median xT zone — the value-preservation interpretation)
    is_takeon = df["event_type"] == "takeon"
    pos = (df["outcome"] > 0)
    if is_takeon.any():
        med_takeon = float(df.loc[is_takeon, "outcome"].median())
        pos.loc[is_takeon] = df.loc[is_takeon, "outcome"] > med_takeon
    df["outcome_pos"] = pos

    # ── Mann-Whitney U ────
    mw_results = [_mwu(df, df["outcome_pos"], "ALL events")]
    for et, label in [("pass", "passes only"),
                      ("carry", "carries only"),
                      ("takeon", "takeons only")]:
        sub = df[df["event_type"] == et]
        mw_results.append(_mwu(sub, sub["outcome_pos"], label))

    # ── Spearman / Pearson on continuous DOS ↔ outcome ────
    sp_results = []
    for label, sub in [
        ("ALL events",   df),
        ("passes only",  df[df["event_type"] == "pass"]),
        ("carries only", df[df["event_type"] == "carry"]),
        ("takeons only", df[df["event_type"] == "takeon"]),
    ]:
        sp_results.append(_spearman(sub, label, x="dos", y="outcome"))

    # ── Logistic ────
    log_results = []
    for label, sub in [
        ("ALL events",   df),
        ("passes only",  df[df["event_type"] == "pass"]),
        ("carries only", df[df["event_type"] == "carry"]),
    ]:
        ssub = sub.assign(outcome_bin=(sub["outcome"] > 0).astype(int))
        log_results.append(_logistic(ssub, "outcome_bin", label))

    # ── Tables ────
    quintiles = _quintiles(df, "outcome")
    direction_table = _direction_breakdown(df, "outcome")
    per_et = _per_event_type(df, "outcome")

    print("\n=== Mann-Whitney U (xT outcome) ===")
    for r in mw_results:
        print(f"  {r['label']:25s} p={_fmt_p(r.get('p', float('nan'))):>10s} "
              f"rbc={r.get('rbc', float('nan')):+.3f}  "
              f"n+={r['n_pos']:>5d} n−={r['n_neg']:>5d}")

    print("\n=== Spearman ρ (DOS, outcome) ===")
    for r in sp_results:
        if r.get("rho") != r.get("rho"):
            print(f"  {r['label']:25s} n={r.get('n','-')} (insufficient)")
        else:
            print(f"  {r['label']:25s} n={r['n']:>5d}  ρ={r['rho']:+.4f}  "
                  f"p={_fmt_p(r['p_spearman']):>10s}")

    print("\n=== Quintiles ===")
    print(quintiles.to_string())
    print("\n=== Direction class ===")
    print(direction_table.to_string())
    print("\n=== Per-event-type ===")
    print(per_et.to_string())

    _plot_quintiles(quintiles, OUT_PNG)
    _write_report(df, "outcome", mw_results, log_results, sp_results,
                  quintiles, direction_table, per_et, OUT_MD)
    print(f"\nReport: {OUT_MD}")
    print(f"Plot:   {OUT_PNG}")


if __name__ == "__main__":
    main()
