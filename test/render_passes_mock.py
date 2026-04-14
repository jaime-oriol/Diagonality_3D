"""Render a mock passes-figure to validate the visual design before
wiring it up to real cached events. Edit the synthetic passes below to
explore extreme cases (tons of diagonals, only failed passes, etc.)."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.viz.passes_plot import plot_player_passes


# ── Synthetic passes ────────────────────────────────────────────────────
# Inspired by the Opta Messi figure: ~40-50 passes from a right-half-space
# attacker, dominated by diagonals into the box, a handful failed.
rng = np.random.default_rng(42)
n_diag = 18
n_fwd = 9
n_side = 12
n_back = 5

def _rand_pass(origin_x_range, origin_y_range, dx_range, dy_range, n):
    rows = []
    for _ in range(n):
        x = rng.uniform(*origin_x_range)
        y = rng.uniform(*origin_y_range)
        dx = rng.uniform(*dx_range)
        dy = rng.uniform(*dy_range)
        rows.append((x, y, x + dx, y + dy))
    return rows

# Diagonals from right half-space attacking the box (gold)
diag = _rand_pass(
    origin_x_range=(0, 35), origin_y_range=(-20, 25),
    dx_range=(8, 22), dy_range=(-18, 18), n=n_diag,
)
# Forward verticals (cyan)
fwd = _rand_pass(
    origin_x_range=(-10, 30), origin_y_range=(-25, 25),
    dx_range=(8, 25), dy_range=(-3, 3), n=n_fwd,
)
# Sideways recycling (white)
side = _rand_pass(
    origin_x_range=(-20, 30), origin_y_range=(-30, 30),
    dx_range=(-3, 3), dy_range=(-15, 15), n=n_side,
)
# Backward construction (purple)
back = _rand_pass(
    origin_x_range=(-5, 35), origin_y_range=(-25, 25),
    dx_range=(-22, -8), dy_range=(-15, 15), n=n_back,
)

passes = pd.DataFrame(
    diag + fwd + side + back,
    columns=["x", "y", "x_receiver", "y_receiver"],
)
passes["direction_class"] = (
    ["diagonal"] * n_diag + ["forward"] * n_fwd
    + ["sideways"] * n_side + ["backward"] * n_back
)
# Some failures sprinkled across categories (mostly on diagonals — they
# are the riskier ones in real life).
n = len(passes)
fail_mask = rng.random(n) < np.where(
    passes["direction_class"].values == "diagonal", 0.18, 0.06)
passes["successful"] = ~fail_mask

print(f"Mock passes: {len(passes)} total, "
      f"{int(passes['successful'].sum())} OK, "
      f"{int((~passes['successful']).sum())} failed")
print(passes["direction_class"].value_counts().to_dict())

fig = plot_player_passes(
    passes,
    title="Michael Olise — Passes",
    subtitle="Bayern Munich 5-0 Hamburger SV · Bundesliga 2025-26 · 13 Sep 2025",
    attacking_right=True,
    team_logo_path="figures/logos/bayern.png",
    project_logo_path="figures/Logo_vizs.png",
    save_path="test/passes_mock.png",
)
print("Saved: test/passes_mock.png")
