"""
make_figures.py - Brand-style analytical figures for the "Diagonality"
deliverable.

Reads the validated event table (dos_validation_full.csv, 6,923 events)
and renders the analytical figures in the project's visual identity:
dark #313332 background, the project's vivid palette and a single
standardized 3-colour scheme for the direction classes (forward, diagonal,
sideways) reused across every figure — defined once in src/viz/common.py.

Output -> deliverable/figures/
Run:  source ~/anaconda3/bin/activate ritmo && python3 deliverable/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz.common import BG, WHITE, DIRECTION_COLORS, DOS_CMAP, PKW  # noqa: E402

CSV  = ROOT / "aws_results" / "intermediate" / "csvs" / "dos_validation_full.csv"
OUT  = ROOT / "deliverable" / "figures"
LOGO = ROOT / "figures" / "logos" / "Logo.png"

SUCCESS = {"successfullyCompleted", "successful"}
CLASSES = ["forward", "diagonal", "sideways"]
CLABEL  = {"forward": "Forward", "diagonal": "Diagonal", "sideways": "Sideways"}

# Standardized 3-colour scheme — forward / diagonal / sideways — used in
# EVERY figure, identical to src/viz/common.py (and the player pass maps):
#   diagonal = lawn green (the SV signature), forward = cyan, sideways = white.
COL  = {c: DIRECTION_COLORS[c] for c in CLASSES}
GRID = "#4a4c4b"
PE_S = [pe.withStroke(linewidth=2.6, foreground=BG)]   # text on busy areas

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 11.5})


# --- shared helpers -------------------------------------------------------

def _logo(fig, x=0.992, y=0.012, target_px=46):
    """Discreet brand mark, bottom-right of the figure."""
    if not LOGO.exists():
        return
    try:
        img = plt.imread(str(LOGO))
        zoom = target_px / float(img.shape[0])
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y),
                            xycoords="figure fraction", frameon=False,
                            box_alignment=(1.0, 0.0))
        ab.set_clip_on(False)
        fig.add_artist(ab)
    except Exception:
        pass


def _style(ax, ygrid=True):
    """Dark journal axes: drop top/right spines, optional light y-grid."""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def _save(fig, name):
    # No logo stamp on the analytical charts — it collides with axes content;
    # the brand identity here is the dark palette. The logo lives on the
    # full-pitch renders (Vision / PPCF / DOS) and the player pass maps.
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote deliverable/figures/{name}")


def _load():
    df = pd.read_csv(CSV)
    df = df[df["direction_class"].isin(CLASSES)].copy()
    # retention = ball kept: completed pass, or any carry / won take-on
    is_pass = df["event_type"] == "pass"
    pass_ok = df["evaluation"].isin(SUCCESS)
    df["retained"] = np.where(is_pass, pass_ok, True)
    df["xt_gain"] = (df["xt_delta"] > 0).astype(float)
    return df


# --- Figure: the progression-safety trade-off -----------------------------

def fig_tradeoff(df):
    rows = {}
    for c in CLASSES:
        g = df[df["direction_class"] == c]
        rows[c] = dict(n=len(g),
                       prog=g["xt_delta"].mean() * 1000.0,   # milli-xT
                       ret=g["retained"].mean() * 100.0,
                       gain=g["xt_gain"].mean() * 100.0)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.set_facecolor(BG)
    _style(ax, ygrid=False)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    xs = [rows[c]["prog"] for c in CLASSES]
    ax.set_xlim(min(xs) - 1.4, max(xs) + 1.4)
    ax.set_ylim(60, 100)                       # retention is a %: never > 100
    ax.axvline(0, color="#7a7c7b", linewidth=1.0, linestyle=(0, (3, 3)))

    side = {"sideways": -1, "diagonal": +1, "forward": -1}
    for c in CLASSES:
        x, y, r = rows[c]["prog"], rows[c]["ret"], rows[c]
        ax.scatter(x, y, s=950, color=COL[c], edgecolor=WHITE,
                   linewidth=1.4, alpha=0.95, zorder=5)
        s = side[c]
        va = "bottom" if s > 0 else "top"
        ax.annotate(CLABEL[c], (x, y), (0, s * 32), textcoords="offset points",
                    ha="center", va=va, color=COL[c], fontsize=13,
                    fontweight="bold", path_effects=PE_S)
        ax.annotate(f"{r['gain']:.0f}% positive xT   .   n = {r['n']:,}",
                    (x, y), (0, s * 18), textcoords="offset points",
                    ha="center", va=va, color="#c8c8c8", fontsize=9.5)

    ax.set_xlabel("Progression  —  mean expected-threat gain (milli-xT per action)")
    ax.set_ylabel("Safety  —  possession retained (%)")
    ax.set_title("The progression-safety trade-off across 6,923 on-ball actions",
                 fontweight="bold", pad=12)
    _save(fig, "Tradeoff.png")


# --- Figure: DOS predicts expected-threat gain (quintiles) ----------------

def fig_quintiles(df):
    d = df.dropna(subset=["dos", "xt_delta"]).copy()
    d["q"] = pd.qcut(d["dos"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    rate, lo, hi = [], [], []
    for q in [f"Q{i}" for i in range(1, 6)]:
        g = d[d["q"] == q]
        k = int(g["xt_gain"].sum()); n = len(g)
        ci = binomtest(k, n).proportion_ci(0.95)
        rate.append(k / n * 100); lo.append(ci.low * 100); hi.append(ci.high * 100)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.set_facecolor(BG)
    _style(ax)
    x = np.arange(5)
    cols = DOS_CMAP(np.linspace(0.20, 0.95, 5))
    ax.bar(x, rate, color=cols, edgecolor=WHITE, linewidth=1.0, width=0.64,
           zorder=3)
    ax.errorbar(x, rate, yerr=[np.array(rate) - lo, np.array(hi) - rate],
                fmt="none", ecolor=WHITE, capsize=4, linewidth=1.1, zorder=4)
    for i, rt in enumerate(rate):
        ax.text(i, hi[i] + 2.6, f"{rt:.0f}%", ha="center", color=WHITE,
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nlowest DOS", "Q2", "Q3", "Q4", "Q5\nhighest DOS"])
    ax.set_ylabel("Actions that gain expected threat (xT > 0, %)")
    ax.set_ylim(0, max(hi) + 12)
    ax.set_title("Positive expected-threat rate rises with DOS",
                 fontweight="bold", pad=12)
    _save(fig, "Quintiles.png")


# --- Figure: defensive disruption, D-Def PC1/PC2 --------------------------

def fig_ddef(df):
    d = df.dropna(subset=["pc1_3s", "pc2_3s"])
    p1 = {c: d[d["direction_class"] == c]["pc1_3s"].abs().mean() for c in CLASSES}
    p2 = {c: d[d["direction_class"] == c]["pc2_3s"].abs().mean() for c in CLASSES}

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.set_facecolor(BG)
    _style(ax)
    x = np.arange(len(CLASSES))
    w = 0.36
    b1 = ax.bar(x - w / 2, [p1[c] for c in CLASSES], w, color="#6a7c8c",
                edgecolor=WHITE, linewidth=1.0,
                label="PC1 — longitudinal disruption")
    b2 = ax.bar(x + w / 2, [p2[c] for c in CLASSES], w,
                color=[COL[c] for c in CLASSES], edgecolor=WHITE,
                linewidth=1.0, label="PC2 — lateral disruption")
    top = max([p1[c] for c in CLASSES] + [p2[c] for c in CLASSES])
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.025,
                    f"{b.get_height():.2f}", ha="center", color=WHITE,
                    fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CLABEL[c] for c in CLASSES])
    ax.set_ylabel("Mean |PCA component| of defensive disruption (3 s)")
    ax.set_ylim(0, top * 1.34)               # headroom for labels + legend
    leg = ax.legend(loc="upper center", ncol=2, columnspacing=1.6,
                    facecolor=BG, edgecolor=GRID, labelcolor=WHITE)
    leg.get_frame().set_alpha(0.0)
    ax.set_title("Diagonal actions specialise in lateral disruption (PC2)",
                 fontweight="bold", pad=12)
    _save(fig, "DDef.png")


# --- Figure: receiver advantage -------------------------------------------

def fig_receiver(df):
    d = df[df["event_type"] == "pass"].copy()
    turn = {c: np.degrees(d[d["direction_class"] == c]["receiver_turn_needed"]
                          .dropna()).mean() for c in CLASSES}
    fov = {c: d[d["direction_class"] == c]["n_teammates_in_fov"].dropna().mean()
           for c in CLASSES}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.set_facecolor(BG)
    x = np.arange(len(CLASSES))
    cols = [COL[c] for c in CLASSES]

    ax = axes[0]; _style(ax)
    ax.bar(x, [turn[c] for c in CLASSES], 0.58, color=cols,
           edgecolor=WHITE, linewidth=1.0)
    for i, c in enumerate(CLASSES):
        ax.text(i, turn[c] + 5, f"{turn[c]:.0f}°", ha="center",
                color=WHITE, fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES])
    ax.set_ylabel("Rotation to face goal at reception (degrees)")
    ax.set_ylim(0, max(turn.values()) * 1.26)
    ax.set_title("Receiver body orientation", fontweight="bold", pad=10)

    ax = axes[1]; _style(ax)
    ax.bar(x, [fov[c] for c in CLASSES], 0.58, color=cols,
           edgecolor=WHITE, linewidth=1.0)
    for i, c in enumerate(CLASSES):
        ax.text(i, fov[c] + max(fov.values()) * 0.035, f"{fov[c]:.1f}",
                ha="center", color=WHITE, fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES])
    ax.set_ylabel("Team-mates inside the receiver's 120° FOV")
    ax.set_ylim(0, max(fov.values()) * 1.24)
    ax.set_title("Receiver passing options", fontweight="bold", pad=10)

    fig.tight_layout(w_pad=3.0)
    _save(fig, "Receiver.png")


# --- Figure: aggregate DOS zone map ---------------------------------------

def fig_dos_map(df):
    from mplsoccer import Pitch
    d = df.dropna(subset=["dos", "x_origin", "y_origin"]).copy()
    # normalise so every attack runs left -> right
    ar = d["attacking_right"].astype(str).str.lower().eq("true")
    xs = np.where(ar, d["x_origin"], -d["x_origin"])
    ys = np.where(ar, d["y_origin"], -d["y_origin"])
    dos_milli = d["dos"].values * 1000.0      # milli-DOS -> readable labels

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    fig.set_facecolor(BG)
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)

    # coarse 6x4 zones: each cell is a real tactical area, big enough to read
    stat = pitch.bin_statistic(xs, ys, dos_milli, statistic="mean",
                               bins=(6, 4))
    pcm = pitch.heatmap(stat, ax=ax, cmap=DOS_CMAP, edgecolors=BG,
                        linewidth=2.0, alpha=0.95, zorder=2)
    pitch.label_heatmap(stat, str_format="{:.0f}", ax=ax, color=WHITE,
                        fontsize=11, fontweight="bold", ha="center",
                        va="center", zorder=5, path_effects=PE_S)

    cb = fig.colorbar(pcm, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("Mean DOS  (x10^-3)", color=WHITE)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.get_yticklabels(), color=WHITE)
    cb.outline.set_edgecolor(GRID)
    ax.set_title("Mean Diagonal Opportunity Surface by pitch zone  "
                 "(attacking left to right)", color=WHITE, fontweight="bold",
                 pad=12)
    _save(fig, "DOS_Map.png")


def main():
    print(f"Loading {CSV.name} ...")
    df = _load()
    print(f"  {len(df):,} events ({', '.join(CLASSES)})")
    fig_tradeoff(df)
    fig_quintiles(df)
    fig_ddef(df)
    fig_receiver(df)
    fig_dos_map(df)
    print("Done.")


if __name__ == "__main__":
    main()
