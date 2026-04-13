"""Shared style and constants for viz modules.

Pitch: 105x68 m, origin at center (0, 0). Coordinates in METERS.
Colormap and color palette inherited from Opta Forum 2026 style.

Only constants and style defaults live here. Helper drawing functions are
intentionally inlined inside each viz module (vision_plot, ppcf_plot)
because the rendering logic is specific to each overlay.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Style ────────────────────────────────────────────────────────────────

BG = "#313332"
WHITE = "white"
FONT = "DejaVu Sans"

plt.style.use("default")
plt.rcParams.update({
    "font.family": FONT, "font.size": 10,
    "figure.dpi": 100, "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.facecolor": BG, "figure.facecolor": BG,
    "text.color": WHITE, "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
})

# ── Team / element colors ────────────────────────────────────────────────

ATT = "deepskyblue"
ATT_LIGHT = "lightskyblue"       # head wedge (attacker gaze)
DEF = "tomato"
DEF_LIGHT = "lightsalmon"        # head wedge (defender gaze)
GK = "black"
BALL = WHITE

# ── Colormaps ────────────────────────────────────────────────────────────

# Pitch control: defender red → neutral gray → attacker blue
PPCF_CMAP = LinearSegmentedColormap.from_list(
    "ppcf", ["#8B0000", "#777777", "#004D98"]
)

# Diagonal Opportunity Surface: transparent → cyan → magenta → white
# "visible" opportunities — inside the on-ball player's FOV + scanned memory.
DOS_CMAP = LinearSegmentedColormap.from_list(
    "dos", ["#0d1117", "#00d4ff", "#ff00ff", "#ffffff"]
)

# Shadow DOS: transparent → amber → gold → white.
# "unseen" opportunities — DOS is high, but the on-ball player does NOT see
# that cell AND it is ahead of the ball inside realistic pass/carry range.
# Visually warm so it reads as "latent warning" next to the cold primary DOS
# layer — if the player still delivers there it is a Hamilton shadowpass.
SHADOW_CMAP = LinearSegmentedColormap.from_list(
    "dos_shadow", ["#0d1117", "#5a3500", "#ffb300", "#ffea00", "#ffffff"]
)

# ── Pitch dimensions (TRACAB, meters, centered) ──────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Keyword dict for mplsoccer Pitch() — centered meters, secondspectrum type
PKW = dict(
    pitch_type="secondspectrum",
    pitch_length=PITCH_LENGTH,
    pitch_width=PITCH_WIDTH,
    pitch_color=BG,
    line_color=WHITE,
    line_zorder=2,
    linewidth=1,
)

