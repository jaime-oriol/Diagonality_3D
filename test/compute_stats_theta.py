"""Storytelling-ready statistics from `dos_validation_raw_xt_ddef_theta.csv`.

Produces the cifras citables for the narrative document:

  - "1 in 3 diagonal passes catch the nearest defender in his blind half"
    → pct_nearest_blind per direction class

  - "Diagonals leave defenders X° more misaligned on average"
    → mean_theta_shoulder_all per direction class

  - "Diagonal receiver only needs to turn X° to face goal"
    → mean receiver_open_angle per direction class

  - "Diagonal receivers see X more teammates inside their FOV"
    → mean n_teammates_in_fov per direction class

  - "Diagonals find 2x more wrongfooted defenders"
    → mean_n_wrongfooted per direction class

Plus Mann-Whitney U tests so each cifra has a p-value attached for
defense against a sceptical reviewer.

Output:
  test/dos_validation_report_theta.md
  test/dos_validation_theta_axes.png
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


RAW = Path("test/dos_validation_raw_xt_ddef_theta.csv")
OUT_PNG = Path("test/dos_validation_theta_axes.png")
OUT_MD = Path("test/dos_validation_report_theta.md")


def _fmt_p(p):
    if p != p:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _mwu(a: np.ndarray, b: np.ndarray, alt: str = "greater") -> tuple:
    """One-sided MWU + rank-biserial effect size. Returns (u, p, rbc, na, nb)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan"), len(a), len(b)
    u, p = mannwhitneyu(a, b, alternative=alt)
    rbc = (2.0 * u) / (len(a) * len(b)) - 1.0
    return float(u), float(p), float(rbc), len(a), len(b)


# ── Direction class table (the primary storytelling table) ────────────

def _direction_table(df: pd.DataFrame, etype: str) -> pd.DataFrame:
    """Per-direction-class summary for one event_type.

    Angles are reported in DEGREES (more readable than radians for the
    narrative). Receiver metrics only make sense for passes.
    """
    sub = df[df["event_type"] == etype].copy()
    rows = []
    for cls in ("forward", "diagonal", "sideways"):
        g = sub[sub["direction_class"] == cls]
        if len(g) == 0:
            continue
        row = {
            "direction_class": cls,
            "n": int(len(g)),
            # Axis 1 — defender disruption
            "mean_theta_shoulder_deg": float(np.degrees(g["mean_theta_shoulder_all"].mean())),
            "mean_theta_head_deg": float(np.degrees(g["mean_theta_head_all"].mean())),
            "pct_nearest_blind": float(g["nearest_in_blind"].mean()),
            "mean_n_wrongfooted": float(g["n_wrongfooted"].mean()),
            "mean_pct_wrongfooted": float(g["pct_wrongfooted"].mean()),
            "mean_theta_affected_deg": float(np.degrees(g["mean_theta_affected"].mean())),
        }
        if etype == "pass":
            # Axis 2 — receiver advantage (passes only)
            row.update({
                "receiver_open_deg": float(np.degrees(g["receiver_open_angle"].mean())),
                "receiver_turn_deg": float(np.degrees(g["receiver_turn_needed"].mean())),
                "n_teammates_in_fov": float(g["n_teammates_in_fov"].mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("direction_class")


# ── Mann-Whitney U: diagonal vs forward (per metric) ────────────────────

def _mwu_diag_vs_fwd(df: pd.DataFrame, etype: str) -> pd.DataFrame:
    """For each theta metric, test if diagonal events differ from forward
    events. Two-sided alternative — the direction depends on the metric
    (e.g. higher mean_theta = more disruption, but lower receiver_turn =
    more advantage, so we report rbc with sign so the reader can see)."""
    sub = df[df["event_type"] == etype]
    diag = sub[sub["direction_class"] == "diagonal"]
    fwd = sub[sub["direction_class"] == "forward"]
    if len(diag) < 5 or len(fwd) < 5:
        return pd.DataFrame()

    metrics = [
        ("mean_theta_shoulder_all", "Defender shoulder misalignment", "greater"),
        ("mean_theta_head_all", "Defender head misalignment", "greater"),
        ("nearest_in_blind", "Nearest defender in blind half", "greater"),
        ("n_wrongfooted", "Wrongfooted defenders", "greater"),
        ("pct_wrongfooted", "% wrongfooted of affected", "greater"),
    ]
    if etype == "pass":
        metrics += [
            ("receiver_open_angle", "Receiver body→goal angle", "less"),
            ("receiver_turn_needed", "Receiver head→goal angle", "less"),
            ("n_teammates_in_fov", "Teammates in receiver FOV", "greater"),
        ]

    rows = []
    for col, label, alt in metrics:
        u, p, rbc, na, nb = _mwu(diag[col].to_numpy(), fwd[col].to_numpy(), alt=alt)
        rows.append({
            "metric": label, "alt": alt, "n_diag": na, "n_fwd": nb,
            "median_diag": float(np.nanmedian(diag[col])),
            "median_fwd": float(np.nanmedian(fwd[col])),
            "rbc": rbc, "p": p,
        })
    return pd.DataFrame(rows)


# ── Plot: receiver advantage per direction class ───────────────────────

def _plot_axes(df: pd.DataFrame, out_path: Path):
    pas = df[df["event_type"] == "pass"]
    if len(pas) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"forward": "#00BFFF", "diagonal": "#7CFC00", "sideways": "#FFFFFF"}
    classes = ["forward", "diagonal", "sideways"]

    # 1: defender shoulder misalignment distribution
    ax = axes[0]
    for cls in classes:
        g = pas[pas["direction_class"] == cls]
        if len(g) == 0:
            continue
        ax.hist(np.degrees(g["mean_theta_shoulder_all"].dropna()),
                bins=30, alpha=0.55, color=colors[cls], label=f"{cls} (n={len(g)})")
    ax.set_xlabel("Mean defender shoulder misalignment (deg)")
    ax.set_ylabel("# passes")
    ax.set_title("Axis 1: Defender disruption")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_facecolor("#313332")

    # 2: receiver_turn_needed distribution
    ax = axes[1]
    for cls in classes:
        g = pas[pas["direction_class"] == cls]
        if len(g) == 0:
            continue
        ax.hist(np.degrees(g["receiver_turn_needed"].dropna()),
                bins=30, alpha=0.55, color=colors[cls], label=f"{cls} (n={len(g)})")
    ax.set_xlabel("Receiver turn-to-goal angle (deg)")
    ax.set_ylabel("# passes")
    ax.set_title("Axis 2: Receiver advantage (lower = better)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_facecolor("#313332")

    # 3: n_teammates_in_fov distribution
    ax = axes[2]
    for cls in classes:
        g = pas[pas["direction_class"] == cls]
        if len(g) == 0:
            continue
        ax.hist(g["n_teammates_in_fov"].dropna(),
                bins=range(0, 12), alpha=0.55, color=colors[cls],
                label=f"{cls} (n={len(g)})")
    ax.set_xlabel("Teammates inside receiver's 120° FOV")
    ax.set_ylabel("# passes")
    ax.set_title("Axis 2b: Receiver options")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_facecolor("#313332")

    fig.set_facecolor("#313332")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Report writer ────────────────────────────────────────────────────────

def _write_report(
    df: pd.DataFrame,
    pass_table: pd.DataFrame,
    carry_table: pd.DataFrame,
    pass_mwu: pd.DataFrame,
    carry_mwu: pd.DataFrame,
    out_path: Path,
):
    lines = []
    lines.append("# Theta orientation metrics — storytelling cifras")
    lines.append("")
    lines.append("Per-direction-class numbers from `src/theta.py` "
                 "(defender disruption + receiver advantage). Designed for "
                 "direct citation in the narrative document.")
    lines.append("")
    lines.append(f"- Events with theta computed: "
                 f"{int(df['nearest_theta_shoulder'].notna().sum()):,}/{len(df):,}")
    lines.append("")

    lines.append("## Passes — per-direction summary")
    lines.append("")
    lines.append("Angles in degrees. `pct_nearest_blind` ∈ [0, 1]. "
                 "`receiver_turn_deg` is the angle the receiver still needs "
                 "to rotate to face goal — lower = more pre-oriented. "
                 "`n_teammates_in_fov` counts teammates inside the receiver's "
                 "120° binocular cone at reception.")
    lines.append("")
    lines.append("```")
    lines.append(pass_table.round(2).to_string())
    lines.append("```")
    lines.append("")

    lines.append("![theta axes](dos_validation_theta_axes.png)")
    lines.append("")

    lines.append("### Passes — Mann-Whitney U: diagonal vs forward")
    lines.append("")
    lines.append("`alt='greater'` = diagonal has higher; `alt='less'` = "
                 "diagonal has lower. `rbc` = rank-biserial correlation "
                 "(effect size; +1 = perfect, 0 = none, signed in the "
                 "direction of the alternative).")
    lines.append("")
    if len(pass_mwu):
        lines.append("| Metric | alt | n diag | n fwd | median diag | median fwd | rbc | p |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in pass_mwu.iterrows():
            mdiag = (np.degrees(r["median_diag"])
                     if "angle" in r["metric"].lower() or "misalignment" in r["metric"].lower()
                     else r["median_diag"])
            mfwd = (np.degrees(r["median_fwd"])
                    if "angle" in r["metric"].lower() or "misalignment" in r["metric"].lower()
                    else r["median_fwd"])
            lines.append(
                f"| {r['metric']} | {r['alt']} | {int(r['n_diag'])} | "
                f"{int(r['n_fwd'])} | {mdiag:.2f} | {mfwd:.2f} | "
                f"{r['rbc']:+.3f} | {_fmt_p(r['p'])} |"
            )
    lines.append("")

    lines.append("## Carries — per-direction summary (defender side only)")
    lines.append("")
    lines.append("Carries don't have a 'receiver' — only axis 1 metrics apply.")
    lines.append("")
    lines.append("```")
    lines.append(carry_table.round(2).to_string())
    lines.append("```")
    lines.append("")

    lines.append("### Carries — Mann-Whitney U: diagonal vs forward")
    lines.append("")
    if len(carry_mwu):
        lines.append("| Metric | alt | n diag | n fwd | median diag | median fwd | rbc | p |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in carry_mwu.iterrows():
            mdiag = (np.degrees(r["median_diag"])
                     if "angle" in r["metric"].lower() or "misalignment" in r["metric"].lower()
                     else r["median_diag"])
            mfwd = (np.degrees(r["median_fwd"])
                    if "angle" in r["metric"].lower() or "misalignment" in r["metric"].lower()
                    else r["median_fwd"])
            lines.append(
                f"| {r['metric']} | {r['alt']} | {int(r['n_diag'])} | "
                f"{int(r['n_fwd'])} | {mdiag:.2f} | {mfwd:.2f} | "
                f"{r['rbc']:+.3f} | {_fmt_p(r['p'])} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Run enrich_with_theta.py first.")
    df = pd.read_csv(RAW)
    print(f"Loaded {RAW}: {len(df):,} rows")

    pass_table = _direction_table(df, "pass")
    carry_table = _direction_table(df, "carry")
    pass_mwu = _mwu_diag_vs_fwd(df, "pass")
    carry_mwu = _mwu_diag_vs_fwd(df, "carry")

    print("\n=== Passes — per-direction summary ===")
    print(pass_table.round(2).to_string())
    print("\n=== Carries — per-direction summary ===")
    print(carry_table.round(2).to_string())
    print("\n=== Passes — diagonal vs forward MWU ===")
    print(pass_mwu.round(4).to_string(index=False) if len(pass_mwu) else "  (insufficient)")
    print("\n=== Carries — diagonal vs forward MWU ===")
    print(carry_mwu.round(4).to_string(index=False) if len(carry_mwu) else "  (insufficient)")

    _plot_axes(df, OUT_PNG)
    _write_report(df, pass_table, carry_table, pass_mwu, carry_mwu, OUT_MD)
    print(f"\nReport: {OUT_MD}")
    print(f"Plot:   {OUT_PNG}")


if __name__ == "__main__":
    main()
