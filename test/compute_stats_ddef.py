"""Statistical validation of H2 from the project proposal:

  H2: high-θ / diagonal actions disrupt PC1 (longitudinal) AND PC2
      (lateral) simultaneously — i.e., they break the defensive structure
      on BOTH axes at once. Forward and sideways actions disrupt only ONE
      axis (PC1 or PC2 respectively).

This is Spielverlagerung's claim "a diagonal pass breaks both the
horizontal AND the vertical line at once" turned into a quantifiable
test using Goes et al. 2019 D-Def with the Forcher 2024 local extension.

Inputs: `test/dos_validation_raw_xt_ddef.csv` produced by
`test/enrich_with_ddef.py`.

Outputs:
  - test/dos_validation_report_ddef.md
  - test/dos_validation_ddef_axes.png  (per-class |PC1| vs |PC2| scatter
    with marginal means + bi-axial isolines)
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, kruskal

import statsmodels.api as sm


RAW_DDEF = Path("test/dos_validation_raw_xt_ddef.csv")
OUT_PNG = Path("test/dos_validation_ddef_axes.png")
OUT_MD = Path("test/dos_validation_report_ddef.md")

# Window suffix used by enrich_with_ddef
W = "3s"


# ── Helper stats ───────────────────────────────────────────────────────

def _fmt_p(p):
    if p != p:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _mwu(a: np.ndarray, b: np.ndarray, alt: str = "greater") -> tuple:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), len(a), len(b)
    u, p = mannwhitneyu(a, b, alternative=alt)
    rbc = (2.0 * u) / (len(a) * len(b)) - 1.0
    return float(u), float(p), float(rbc), len(a), len(b)


# ── Bi-axial disruption metric ─────────────────────────────────────────

def _bi_axial_balance(df: pd.DataFrame, w: str = W) -> pd.Series:
    """Per event, compute the bi-axial balance ratio:
        balance = min(|PC1|, |PC2|) / max(|PC1|, |PC2|)
    1.0 = perfectly bi-axial (PC1 == PC2 in absolute disruption)
    0.0 = purely uni-axial (only one of the two components moves)
    """
    p1 = df[f"pc1_{w}"].abs()
    p2 = df[f"pc2_{w}"].abs()
    mn = np.minimum(p1, p2)
    mx = np.maximum(p1, p2)
    return mn / mx.where(mx > 1e-9, np.nan)


# ── Tables per direction class ─────────────────────────────────────────

def _direction_table(df: pd.DataFrame, w: str = W) -> pd.DataFrame:
    """Per-class means of |PC1|, |PC2|, |PC3| + two balance interpretations:
       - per_class_balance: min(mean|PC1|, mean|PC2|) / max(mean|PC1|, mean|PC2|)
         Captures whether the CLASS as a whole disrupts both axes equally.
       - per_event_balance_mean: average over per-event ratios. Captures
         whether INDIVIDUAL events tend to be balanced.
    """
    rows = []
    for cls, g in df.groupby("direction_class"):
        m1 = float(g[f"pc1_{w}"].abs().mean())
        m2 = float(g[f"pc2_{w}"].abs().mean())
        per_class_bal = min(m1, m2) / max(m1, m2) if max(m1, m2) > 0 else float("nan")
        rows.append({
            "direction_class": cls,
            "n": int(len(g)),
            "abs_pc1": m1,
            "abs_pc2": m2,
            "abs_pc3": float(g[f"pc3_{w}"].abs().mean()),
            "ddef": float(g[f"ddef_{w}"].mean()),
            "per_class_balance": per_class_bal,
            "per_event_balance_mean": float(g["biaxial"].mean()),
            "local_delta_area": float(g["local_delta_area"].mean()),
            "local_delta_spread": float(g["local_delta_spread"].mean()),
        })
    return pd.DataFrame(rows).set_index("direction_class").reindex(
        ["forward", "diagonal", "sideways"]).dropna(how="all")


def _per_event_type(df: pd.DataFrame, w: str = W) -> pd.DataFrame:
    rows = []
    for et, g in df.groupby("event_type"):
        rows.append({
            "event_type": et,
            "n": int(len(g)),
            "abs_pc1": float(g[f"pc1_{w}"].abs().mean()),
            "abs_pc2": float(g[f"pc2_{w}"].abs().mean()),
            "ddef": float(g[f"ddef_{w}"].mean()),
            "biaxial_mean": float(g["biaxial"].mean()),
        })
    return pd.DataFrame(rows).set_index("event_type")


# ── DOS ↔ D-Def cross-correlation ──────────────────────────────────────

def _dos_correlations(df: pd.DataFrame, w: str = W) -> pd.DataFrame:
    rows = []
    targets = [f"pc1_{w}", f"pc2_{w}", f"pc3_{w}", f"ddef_{w}",
               "local_delta_area", "local_delta_spread", "biaxial"]
    for t in targets:
        sub = df[["dos", t]].dropna()
        if len(sub) < 30:
            rows.append({"target": t, "n": len(sub),
                         "rho": float("nan"), "p": float("nan")})
            continue
        # Use absolute value for PC components (sign is arbitrary)
        x = sub["dos"].to_numpy()
        y = sub[t].abs().to_numpy() if t.startswith("pc") else sub[t].to_numpy()
        rho, p = spearmanr(x, y)
        rows.append({"target": t, "n": int(len(sub)),
                     "rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows)


# ── Logistic: P(bi-axial high disruption) ~ DOS + controls ────────────

def _logistic_biaxial(df: pd.DataFrame, w: str = W) -> dict:
    """Binary outcome: balance > median(balance). Tests if DOS predicts
    balanced (truly bi-axial) disruption after controlling for confounders."""
    sub = df.dropna(subset=["dos", "biaxial", "distance",
                            "pressure_player", "n_def_lane", "n_def_goal"]).copy()
    if len(sub) < 50:
        return {"fit": False, "n": len(sub)}
    threshold = float(sub["biaxial"].median())
    sub["high_balance"] = (sub["biaxial"] > threshold).astype(int)
    if sub["high_balance"].nunique() < 2:
        return {"fit": False, "n": len(sub)}
    X = sm.add_constant(sub[["dos", "distance", "pressure_player",
                              "n_def_lane", "n_def_goal"]].astype(float))
    y = sub["high_balance"]
    try:
        res = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
    except Exception as ex:
        return {"fit": False, "n": len(sub), "err": str(ex)}
    return {
        "fit": True, "n": int(len(sub)), "threshold": threshold,
        "coefs": dict(zip(X.columns, res.params.tolist())),
        "pvalues": dict(zip(X.columns, res.pvalues.tolist())),
        "pseudo_r2": float(res.prsquared),
        "llr_p": float(res.llr_pvalue),
    }


# ── H2 axis-by-axis MWU ────────────────────────────────────────────────

def _h2_mwu_diagonal_vs_orthogonal(df: pd.DataFrame, w: str = W) -> pd.DataFrame:
    """The exact H2 test: diagonal events have HIGHER balance than
    forward and sideways (combined orthogonal class)."""
    diag = df[df["direction_class"] == "diagonal"]
    fwd = df[df["direction_class"] == "forward"]
    side = df[df["direction_class"] == "sideways"]
    ortho = pd.concat([fwd, side])

    rows = []
    for label, target in [
        ("diagonal vs forward",          (diag, fwd, "biaxial")),
        ("diagonal vs sideways",         (diag, side, "biaxial")),
        ("diagonal vs ortho (fwd+side)", (diag, ortho, "biaxial")),
        ("diagonal vs forward [PC1+PC2]",(diag, fwd, "_combined_")),
        ("diagonal vs forward [PC1]",    (diag, fwd, f"pc1_{w}")),
        ("diagonal vs forward [PC2]",    (diag, fwd, f"pc2_{w}")),
        ("diagonal vs sideways [PC1]",   (diag, side, f"pc1_{w}")),
        ("diagonal vs sideways [PC2]",   (diag, side, f"pc2_{w}")),
    ]:
        a, b, t = target
        if t == "_combined_":
            xa = (a[f"pc1_{w}"].abs() + a[f"pc2_{w}"].abs()).to_numpy()
            xb = (b[f"pc1_{w}"].abs() + b[f"pc2_{w}"].abs()).to_numpy()
        elif t.startswith("pc"):
            xa = a[t].abs().to_numpy()
            xb = b[t].abs().to_numpy()
        else:
            xa = a[t].to_numpy()
            xb = b[t].to_numpy()
        u, p, rbc, na, nb = _mwu(xa, xb, alt="greater")
        rows.append({
            "test": label,
            "n_diag": na, "n_other": nb,
            "median_diag": float(np.nanmedian(xa)) if len(xa) else float("nan"),
            "median_other": float(np.nanmedian(xb)) if len(xb) else float("nan"),
            "rbc": rbc,
            "p": p,
        })
    return pd.DataFrame(rows)


# ── Plot ───────────────────────────────────────────────────────────────

def _plot_axes(df: pd.DataFrame, out_path: Path, w: str = W):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Left: scatter |PC1| vs |PC2| colored by direction class
    ax = axes[0]
    colors = {"forward": "#00BFFF", "diagonal": "#7CFC00",
              "sideways": "#FFFFFF", "backward": "#FF1493"}
    for cls, g in df.groupby("direction_class"):
        if cls in colors:
            ax.scatter(g[f"pc1_{w}"].abs(), g[f"pc2_{w}"].abs(),
                       s=8, alpha=0.35, color=colors[cls], label=cls)
    # Mean per class
    for cls, g in df.groupby("direction_class"):
        if cls in colors:
            mx = g[f"pc1_{w}"].abs().mean()
            my = g[f"pc2_{w}"].abs().mean()
            ax.scatter([mx], [my], s=180, color=colors[cls],
                       edgecolor="black", linewidth=2, zorder=10)
    # Diagonal line y=x indicating bi-axial balance
    lim = max(df[f"pc1_{w}"].abs().quantile(0.99),
              df[f"pc2_{w}"].abs().quantile(0.99))
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4)
    ax.set_xlabel("|PC1| — longitudinal disruption")
    ax.set_ylabel("|PC2| — lateral disruption")
    ax.set_title("Per-event D-Def axes (filled circles = class mean)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(loc="upper right", facecolor="white")
    ax.grid(alpha=0.3)
    ax.set_facecolor("#313332")

    # Right: bi-axial balance distribution per class
    ax = axes[1]
    for cls in ("forward", "diagonal", "sideways"):
        g = df[df["direction_class"] == cls]
        if len(g) == 0:
            continue
        ax.hist(g["biaxial"].dropna(), bins=30, alpha=0.5,
                color=colors[cls], label=f"{cls} (n={len(g)})")
    ax.set_xlabel("Bi-axial balance = min(|PC1|, |PC2|) / max(...)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of bi-axial balance by direction class\n"
                 "(higher = both axes disrupted simultaneously)")
    ax.set_facecolor("#313332")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.set_facecolor("#313332")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    if not RAW_DDEF.exists():
        raise SystemExit(f"Missing {RAW_DDEF}. Run enrich_with_ddef.py first.")
    df = pd.read_csv(RAW_DDEF)
    print(f"Loaded {RAW_DDEF}: {len(df):,} rows")

    # Bi-axial balance
    df["biaxial"] = _bi_axial_balance(df, W)
    df["abs_pc1"] = df[f"pc1_{W}"].abs()
    df["abs_pc2"] = df[f"pc2_{W}"].abs()

    # Tables
    dir_table = _direction_table(df, W)
    per_et = _per_event_type(df, W)
    h2 = _h2_mwu_diagonal_vs_orthogonal(df, W)
    corr = _dos_correlations(df, W)
    logit_ba = _logistic_biaxial(df, W)

    # Print to console
    print("\n=== Direction class — D-Def axes ===")
    print(dir_table.round(4).to_string())
    print("\n=== Per event-type ===")
    print(per_et.round(4).to_string())
    print("\n=== H2: diagonal disrupts bi-axial vs orthogonal? ===")
    print(h2.round(4).to_string(index=False))
    print("\n=== DOS ↔ D-Def correlations (Spearman ρ) ===")
    print(corr.round(4).to_string(index=False))
    print("\n=== Logistic: P(high biaxial balance) ~ DOS + controls ===")
    if logit_ba.get("fit"):
        print(f"  N={logit_ba['n']}, threshold={logit_ba['threshold']:.4f}, "
              f"Pseudo R²={logit_ba['pseudo_r2']:.4f}, LLR p={_fmt_p(logit_ba['llr_p'])}")
        for k in ["const", "dos", "distance", "pressure_player",
                  "n_def_lane", "n_def_goal"]:
            if k in logit_ba["coefs"]:
                print(f"  {k:18s} {logit_ba['coefs'][k]:+.4f}  p={_fmt_p(logit_ba['pvalues'][k])}")
    else:
        print("  fit skipped")

    # Plot
    _plot_axes(df, OUT_PNG, W)
    print(f"\nPlot:   {OUT_PNG}")

    # Markdown report
    lines = []
    lines.append("# D-Def empirical validation — Stage 3 / H2")
    lines.append("")
    lines.append("Goes et al. (2019) D-Def with Forcher (2024) local extension. "
                 "Decomposed via PCA across the full dataset:")
    lines.append("  - **PC1** = longitudinal disruption (vertical stretch).")
    lines.append("  - **PC2** = lateral disruption (horizontal stretch).")
    lines.append("  - **PC3** = shape disruption (deformation).")
    lines.append("")
    lines.append(f"- Events with valid PCA: {df[f'pc1_{W}'].notna().sum():,}")
    lines.append(f"- Window: {W}")
    lines.append("")
    lines.append("## H2 — Diagonal actions disrupt BOTH axes simultaneously")
    lines.append("")
    lines.append("Spielverlagerung's claim quantified: bi-axial balance "
                 "ratio = `min(|PC1|, |PC2|) / max(|PC1|, |PC2|)`. "
                 "**1.0 = perfectly bi-axial; 0.0 = uni-axial.**")
    lines.append("")
    lines.append("```")
    lines.append(dir_table.round(4).to_string())
    lines.append("```")
    lines.append("")
    lines.append("![axes](dos_validation_ddef_axes.png)")
    lines.append("")
    lines.append("### Mann-Whitney U: diagonal vs orthogonal balance")
    lines.append("")
    lines.append("| Test | n_diag | n_other | median_diag | median_other | rbc | p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in h2.iterrows():
        lines.append(
            f"| {r['test']} | {int(r['n_diag'])} | {int(r['n_other'])} | "
            f"{r['median_diag']:.4f} | {r['median_other']:.4f} | "
            f"{r['rbc']:+.3f} | {_fmt_p(r['p'])} |"
        )
    lines.append("")

    lines.append("## DOS ↔ D-Def — does our DOS predict actual disruption?")
    lines.append("")
    lines.append("Spearman ρ between DOS (input mechanism: defender visual "
                 "blind-spot exploitation) and D-Def quantities (output: real "
                 "structural disruption observed in the next 3s).")
    lines.append("")
    lines.append("| Target | N | Spearman ρ | p |")
    lines.append("|---|---:|---:|---:|")
    for _, r in corr.iterrows():
        lines.append(f"| {r['target']} | {int(r['n']) if r['n']==r['n'] else 'n/a'} | "
                     f"{r['rho']:+.4f} | {_fmt_p(r['p'])} |")
    lines.append("")

    lines.append("## Logistic — does DOS predict bi-axial high-balance?")
    lines.append("")
    if logit_ba.get("fit"):
        lines.append(f"- N = {logit_ba['n']}, threshold = "
                     f"{logit_ba['threshold']:.4f} (median balance)")
        lines.append(f"- Pseudo R² = {logit_ba['pseudo_r2']:.4f}, "
                     f"LLR p = {_fmt_p(logit_ba['llr_p'])}")
        lines.append("")
        lines.append("| coef | value | p |")
        lines.append("|---|---:|---:|")
        for k in ["const", "dos", "distance", "pressure_player",
                  "n_def_lane", "n_def_goal"]:
            if k in logit_ba["coefs"]:
                lines.append(f"| {k} | {logit_ba['coefs'][k]:+.4f} | "
                             f"{_fmt_p(logit_ba['pvalues'][k])} |")
        lines.append("")
    else:
        lines.append("_logit fit skipped_")
        lines.append("")

    lines.append("## Per event-type")
    lines.append("")
    lines.append("```")
    lines.append(per_et.round(4).to_string())
    lines.append("```")

    OUT_MD.write_text("\n".join(lines))
    print(f"Report: {OUT_MD}")


if __name__ == "__main__":
    main()
