"""
make_figures.py - Paper-quality, brand-styled analytical figures for
"Diagonality: The Best of Both Worlds".

Reads the validated event table (dos_validation_full.csv, 6,923 events)
and renders the analytical figures in the project's visual identity
(dark #313332 background, project palette, logo) with publication-grade
English axis labels and titles.

Output -> paper/figures/
Run:  source ~/anaconda3/bin/activate ritmo && python3 paper/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz.common import BG, WHITE, DIRECTION_COLORS, DOS_CMAP, PKW  # noqa: E402

CSV  = ROOT / "aws_results" / "intermediate" / "csvs" / "dos_validation_full.csv"
OUT  = ROOT / "paper" / "figures"
LOGO = ROOT / "figures" / "logos" / "Logo.png"

SUCCESS = {"successfullyCompleted", "successful"}
CLASSES = ["forward", "diagonal", "sideways"]
CLABEL  = {"forward": "Forward", "diagonal": "Diagonal", "sideways": "Sideways"}
# Project palette (src/viz/common.py): diagonal = lawn green (SV signature)
CCOLOR  = {c: DIRECTION_COLORS[c] for c in CLASSES}

plt.rcParams.update({"font.size": 12.5, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.8})


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


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color="#4a4c4b", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=WHITE)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    _logo(fig)
    fig.savefig(OUT / name, dpi=240, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}")


def _load():
    df = pd.read_csv(CSV)
    df = df[df["direction_class"].isin(CLASSES)].copy()
    # retention = ball kept: completed pass, or any carry / won take-on
    is_pass = df["event_type"] == "pass"
    pass_ok = df["evaluation"].isin(SUCCESS)
    df["retained"] = np.where(is_pass, pass_ok, True)
    df["xt_gain"] = (df["xt_delta"] > 0).astype(float)
    return df


# --- Figure 1: the framework (4-stage schematic) --------------------------

def fig_framework():
    fig, ax = plt.subplots(figsize=(13.0, 3.6))
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    ax.axis("off")

    stages = [
        ("3D Skeleton", "21 keypoints / player\n@ 50 Hz", "#2f6f8f"),
        ("Orientation\n& Vision", "head/shoulder/hip\n+ Bekkers FOV", "#2f6f8f"),
        ("Orientation-Aware\nPitch Control", "anisotropic\nreach fields", "#2f6f8f"),
        ("Diagonal Opportunity\nSurface (DOS)", "diagonal vs\northogonal control", "#7CFC00"),
    ]
    bw, gap = 20.0, 6.4
    x0 = (100 - (len(stages) * bw + (len(stages) - 1) * gap)) / 2
    for i, (title, sub, col) in enumerate(stages):
        x = x0 + i * (bw + gap)
        dark = (col == "#7CFC00")
        ax.add_patch(FancyBboxPatch(
            (x, 9), bw, 13, boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=col, edgecolor=WHITE, linewidth=1.2,
            alpha=0.95 if dark else 0.55))
        tcol = "#0d1117" if dark else WHITE
        ax.text(x + bw / 2, 17.6, title, ha="center", va="center",
                color=tcol, fontsize=12.5, fontweight="bold")
        ax.text(x + bw / 2, 12.6, sub, ha="center", va="center",
                color=tcol, fontsize=9.5, alpha=0.9)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + bw + 0.6, 15.5), (x + bw + gap - 0.6, 15.5),
                arrowstyle="-|>", mutation_scale=22,
                color=WHITE, linewidth=2.0))
    # cognitive layer band under DOS
    xd = x0 + 3 * (bw + gap)
    ax.add_patch(FancyBboxPatch(
        (xd - 1, 3.2), bw + 2, 4.0, boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="none", edgecolor="#ffb300", linewidth=1.3, linestyle="--"))
    ax.text(xd + bw / 2, 5.2, "Scanning gate  -  on-ball FOV + 2.5 s memory",
            ha="center", va="center", color="#ffb300", fontsize=9.5,
            fontstyle="italic")
    ax.text(50, 27.0, "A four-stage orientation-aware framework",
            ha="center", va="center", color=WHITE, fontsize=15,
            fontweight="bold")
    _save(fig, "Framework.png")


# --- Figure 2: the progression-safety trade-off (HERO) --------------------

def fig_tradeoff(df):
    rows = []
    for c in CLASSES:
        g = df[df["direction_class"] == c]
        rows.append(dict(c=c, n=len(g),
                         prog=g["xt_delta"].mean() * 1000.0,   # milli-xT
                         ret=g["retained"].mean() * 100.0,
                         gain=g["xt_gain"].mean() * 100.0))
    r = {x["c"]: x for x in rows}

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    fig.set_facecolor(BG)
    _style(ax)

    xs = [r[c]["prog"] for c in CLASSES]
    ys = [r[c]["ret"] for c in CLASSES]
    xpad = (max(xs) - min(xs)) * 0.28 + 0.4
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - 12, max(ys) + 12)

    # "best of both worlds" quadrant: forward progression AND high retention
    ax.axvline(0, color="#6a6c6b", linewidth=1.0, linestyle=":")
    ax.axhspan(np.mean(ys), ax.get_ylim()[1], xmin=0.5, xmax=1.0,
               color="#7CFC00", alpha=0.06)

    for c in CLASSES:
        x, y = r[c]["prog"], r[c]["ret"]
        ax.scatter(x, y, s=2400, color=CCOLOR[c], edgecolor=WHITE,
                   linewidth=1.8, alpha=0.92, zorder=5)
        ax.text(x, y, CLABEL[c], ha="center", va="center",
                color="#0d1117" if c != "sideways" else "#0d1117",
                fontsize=12, fontweight="bold", zorder=6)
        ax.annotate(f"xT>0 in {r[c]['gain']:.0f}% of actions   (n={r[c]['n']:,})",
                    (x, y), (0, -52), textcoords="offset points",
                    ha="center", color=WHITE, fontsize=9.5, alpha=0.9)

    ax.set_xlabel("Progression  -  mean expected-threat gain  (milli-xT per action)",
                  color=WHITE)
    ax.set_ylabel("Safety  -  possession retained  (%)", color=WHITE)
    ax.set_title("Diagonal play is the optimum of the\nprogression-safety trade-off",
                 color=WHITE, fontsize=15, fontweight="bold", pad=14)
    ax.text(0.985, 0.04, '"the best of both worlds"', transform=ax.transAxes,
            ha="right", color="#7CFC00", fontsize=10.5, fontstyle="italic")
    _save(fig, "Tradeoff.png")


# --- Figure 3: DOS predicts expected-threat gain (quintiles) --------------

def fig_quintiles(df):
    d = df.dropna(subset=["dos", "xt_delta"]).copy()
    d["q"] = pd.qcut(d["dos"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    rate, lo, hi, dosm = [], [], [], []
    for q in [f"Q{i}" for i in range(1, 6)]:
        g = d[d["q"] == q]
        k = int(g["xt_gain"].sum()); n = len(g)
        ci = binomtest(k, n).proportion_ci(0.95)
        rate.append(k / n * 100); lo.append(ci.low * 100); hi.append(ci.high * 100)
        dosm.append(g["dos"].mean())

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.set_facecolor(BG)
    _style(ax)
    x = np.arange(5)
    cols = DOS_CMAP(np.linspace(0.15, 0.95, 5))
    ax.bar(x, rate, color=cols, edgecolor=WHITE, linewidth=1.0, width=0.66,
           zorder=3)
    ax.errorbar(x, rate, yerr=[np.array(rate) - lo, np.array(hi) - rate],
                fmt="none", ecolor=WHITE, capsize=4, linewidth=1.2, zorder=4)
    for i, (rt, dm) in enumerate(zip(rate, dosm)):
        ax.text(i, rt + 3.2, f"{rt:.0f}%", ha="center", color=WHITE,
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nlowest DOS", "Q2", "Q3", "Q4", "Q5\nhighest DOS"],
                       color=WHITE)
    ax.set_ylabel("Actions that gain expected threat  (xT > 0,  %)", color=WHITE)
    ax.set_ylim(0, max(hi) + 12)
    ax.set_title("DOS predicts expected-threat gain\n"
                 "(the model sees no xG, xT or outcome labels)",
                 color=WHITE, fontsize=14, fontweight="bold", pad=12)
    _save(fig, "Quintiles.png")


# --- Figure 4: defensive disruption, D-Def PC1/PC2 ------------------------

def fig_ddef(df):
    d = df.dropna(subset=["pc1_3s", "pc2_3s"])
    p1 = {c: d[d["direction_class"] == c]["pc1_3s"].abs().mean() for c in CLASSES}
    p2 = {c: d[d["direction_class"] == c]["pc2_3s"].abs().mean() for c in CLASSES}

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.set_facecolor(BG)
    _style(ax)
    x = np.arange(len(CLASSES))
    w = 0.36
    b1 = ax.bar(x - w / 2, [p1[c] for c in CLASSES], w, color="#5a6f8f",
                edgecolor=WHITE, linewidth=1.0, label="PC1  -  longitudinal disruption")
    b2 = ax.bar(x + w / 2, [p2[c] for c in CLASSES], w,
                color=[CCOLOR[c] for c in CLASSES], edgecolor=WHITE,
                linewidth=1.0, label="PC2  -  lateral disruption")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                    f"{b.get_height():.2f}", ha="center", color=WHITE,
                    fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CLABEL[c] for c in CLASSES], color=WHITE)
    ax.set_ylabel("Mean |PCA component| of defensive disruption (3 s)", color=WHITE)
    ax.set_title("Diagonal actions specialise in lateral\ndefensive disruption (PC2)",
                 color=WHITE, fontsize=14, fontweight="bold", pad=12)
    leg = ax.legend(facecolor="#3c3e3d", edgecolor="#888888", labelcolor=WHITE,
                    fontsize=10, loc="upper center", ncol=2)
    _save(fig, "DDef.png")


# --- Figure 5: receiver advantage -----------------------------------------

def fig_receiver(df):
    d = df[(df["event_type"] == "pass")].copy()
    turn = {c: np.degrees(d[d["direction_class"] == c]["receiver_turn_needed"]
                          .dropna()).mean() for c in CLASSES}
    fov = {c: d[d["direction_class"] == c]["n_teammates_in_fov"].dropna().mean()
           for c in CLASSES}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    fig.set_facecolor(BG)
    x = np.arange(len(CLASSES))
    cols = [CCOLOR[c] for c in CLASSES]

    ax = axes[0]; _style(ax)
    ax.bar(x, [turn[c] for c in CLASSES], 0.6, color=cols, edgecolor=WHITE,
           linewidth=1.0)
    for i, c in enumerate(CLASSES):
        ax.text(i, turn[c] + 3, f"{turn[c]:.0f}°", ha="center",
                color=WHITE, fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES], color=WHITE)
    ax.set_ylabel("Rotation to face goal at reception  (degrees)", color=WHITE)
    ax.set_title("Receiver body orientation\n(lower = already facing goal)",
                 color=WHITE, fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, max(turn.values()) + 22)

    ax = axes[1]; _style(ax)
    ax.bar(x, [fov[c] for c in CLASSES], 0.6, color=cols, edgecolor=WHITE,
           linewidth=1.0)
    for i, c in enumerate(CLASSES):
        ax.text(i, fov[c] + 0.16, f"{fov[c]:.1f}", ha="center", color=WHITE,
                fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES], color=WHITE)
    ax.set_ylabel("Team-mates inside the receiver's 120° FOV", color=WHITE)
    ax.set_title("Receiver passing options\n(higher = more multiplicity)",
                 color=WHITE, fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, max(fov.values()) + 1.4)

    fig.suptitle("The diagonal receiver: closer to goal-facing than a vertical "
                 "pass, with more options in view",
                 color=WHITE, fontsize=14, fontweight="bold", y=1.04)
    _save(fig, "Receiver.png")


# --- Figure 6: aggregate DOS map (the coaching deliverable) ---------------

def fig_dos_map(df):
    from mplsoccer import Pitch
    d = df.dropna(subset=["dos", "x_origin", "y_origin"]).copy()
    # normalise so every attack runs left -> right
    ar = d["attacking_right"].astype(str).str.lower().eq("true")
    xs = np.where(ar, d["x_origin"], -d["x_origin"])
    ys = np.where(ar, d["y_origin"], -d["y_origin"])

    fig, ax = plt.subplots(figsize=(10.4, 7.0))
    fig.set_facecolor(BG)
    pitch = Pitch(**PKW)
    pitch.draw(ax=ax)
    stat = pitch.bin_statistic(xs, ys, d["dos"].values, statistic="mean",
                               bins=(12, 8))
    pcm = pitch.heatmap(stat, ax=ax, cmap=DOS_CMAP, edgecolors=BG, alpha=0.92)
    cb = fig.colorbar(pcm, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("Mean DOS", color=WHITE)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.get_yticklabels(), color=WHITE)
    ax.set_title("Where diagonal play pays off: mean Diagonal Opportunity "
                 "Surface\nby pitch zone  (6,923 actions, attacking left "
                 "to right)",
                 color=WHITE, fontsize=13.5, fontweight="bold", pad=12)
    ax.annotate("attacking direction", (0, -35.5), ha="center", color=WHITE,
                fontsize=9.5, alpha=0.7,
                arrowprops=None)
    _save(fig, "DOS_Map.png")


def main():
    print(f"Loading {CSV.name} ...")
    df = _load()
    print(f"  {len(df):,} events ({', '.join(CLASSES)})")
    fig_framework()
    fig_tradeoff(df)
    fig_quintiles(df)
    fig_ddef(df)
    fig_receiver(df)
    fig_dos_map(df)
    print("Done.")


if __name__ == "__main__":
    main()
