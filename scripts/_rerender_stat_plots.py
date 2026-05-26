"""One-off: regenerar las 5 stat PNGs en `results/reports/` con la nueva
identidad LIGHT OPTA. Reusa los _plot_* de los tres scripts del pipeline
(que acabo de patchear pa importar src.viz.common) y los alimenta con
results/datasets/dos_validation_full.csv (versionado, contiene TODOS los
columns finales tras la cadena de enrichment).

Salida:
  results/reports/dos_validation_quintiles_xt.png
  results/reports/dos_validation_quintiles.png       (= alias raw)
  results/reports/dos_validation_quintiles_fixed.png (= alias raw_fixed)
  results/reports/dos_validation_ddef_axes.png
  results/reports/dos_validation_theta_axes.png
"""
import sys; sys.path.insert(0, ".")
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

CSV = Path("results/datasets/dos_validation_full.csv")
OUT = Path("results/reports"); OUT.mkdir(parents=True, exist_ok=True)

print(f"Loading {CSV}...")
df = pd.read_csv(CSV)
print(f"  {len(df):,} events, {len(df.columns)} columns")


# ── xT quintiles ──────────────────────────────────────────────────────
print("Computing xT-based outcome quintiles...")
df_xt = df.dropna(subset=["dos"]).copy()
df_xt["outcome"] = df_xt["xt_delta"].fillna(0.0)
df_xt["q"] = pd.qcut(df_xt["dos"], 5, labels=[f"Q{i} ({l})" for i, l in zip(
    range(1, 6), ["lowest", "low", "mid", "high", "highest"])])
quintiles = df_xt.groupby("q", observed=False).agg(
    outcome_mean=("outcome", "mean"),
    outcome_pos_rate=("outcome", lambda s: (s > 0).mean()),
    dos_mean=("dos", "mean"),
)

from pipeline.compute_stats_xt import _plot_quintiles
_plot_quintiles(quintiles, OUT / "dos_validation_quintiles_xt.png")
# Aliases que el codigo viejo creaba como hard links — replico via copy
shutil.copy(OUT / "dos_validation_quintiles_xt.png", OUT / "dos_validation_quintiles.png")
shutil.copy(OUT / "dos_validation_quintiles_xt.png", OUT / "dos_validation_quintiles_fixed.png")
print(f"  + dos_validation_quintiles_xt.png (+ 2 alias)")


# ── D-Def axes ────────────────────────────────────────────────────────
print("Plotting D-Def axes (PC1 vs PC2)...")
W = "3s"
df_dd = df.copy()
# Si las cols de biaxial no existen, las computo
if "biaxial" not in df_dd.columns:
    pc1, pc2 = df_dd[f"pc1_{W}"].abs(), df_dd[f"pc2_{W}"].abs()
    df_dd["biaxial"] = np.where(np.maximum(pc1, pc2) > 0,
                                 np.minimum(pc1, pc2) / np.maximum(pc1, pc2), 0.0)
from pipeline.compute_stats_ddef import _plot_axes as _plot_ddef_axes
_plot_ddef_axes(df_dd, OUT / "dos_validation_ddef_axes.png")
print(f"  + dos_validation_ddef_axes.png")


# ── Theta axes ────────────────────────────────────────────────────────
print("Plotting theta axes (defender/receiver per direction class)...")
from pipeline.compute_stats_theta import _plot_axes as _plot_theta_axes
_plot_theta_axes(df, OUT / "dos_validation_theta_axes.png")
print(f"  + dos_validation_theta_axes.png")

print("\nAll stat plots regenerated.")
