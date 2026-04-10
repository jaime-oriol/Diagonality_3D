# Seeing the Unseen: How Diagonal Actions Exploit Defenders' Blind Spots

**AWS World Sports Innovation Cup 2026 — Challenge 2: Unlock the Power of 3D Football Data**

A framework that uses 3D skeleton data to compute the real body orientation of every defender, measure the angle between that orientation and the direction of incoming actions (theta), and demonstrate that diagonal actions maximize theta — producing greater defensive disruption, slower reactions, and higher action success. The framework outputs an Orientation-Aware Pitch Control surface and Diagonal Opportunity Maps that coaching staffs can use to identify where and when to attack a specific opponent diagonally.

---

## Research Question

Do diagonal actions — passes, carries and off-ball runs — systematically exploit defenders' visual blind spots more than orthogonal ones, and does this orientation mismatch explain their greater defensive disruption?

---

## Repository Structure

```
Diagonality_3D/
├── docs/                        # Research and data documentation (git-ignored)
│   ├── propuesta_final.md       # Full project proposal
│   ├── hackathon_data.md        # Complete data inventory (verified)
│   ├── diagonality.md           # Spielverlagerung tactical theory
│   ├── vision_de_paul.md        # Bekkers SSAC 2026 vision model paper
│   ├── the_diagonalist_manifesto.md  # Hamilton manifesto
│   └── prompts.md               # Reusable prompts and notes
│
├── src/                         # Python modules
│   ├── loader.py                # XML events, metadata, frame mapping
│   ├── preprocess.py            # Per-match cache extraction
│   ├── orientation.py           # Head/shoulder/hip orientation + velocity
│   ├── vision.py                # Vision model (adapted Bekkers)
│   ├── theta.py                 # Theta per event (passes, carries)
│   ├── ddef.py                  # D-Def: defensive disruption (Goes et al.)
│   ├── ppcf.py                  # Immediate Orientation-Aware PPCF (reach fields)
│   ├── dos.py                   # Diagonal Opportunity Surfaces
│   ├── possession.py            # Frame-exact possession timeline
│   ├── scanning.py              # On-ball player FOV + 2.5s scanning memory
│   └── viz/                     # Visualization package
│       ├── common.py            # Shared style constants + colormaps
│       ├── vision_plot.py       # Vision map renderer
│       ├── ppcf_plot.py         # PPCF reach-field renderer
│       └── dos_plot.py          # DOS heatmap renderer (FOV gate)
│
├── cache/                       # Git-ignored (preprocessed per match)
├── data/                        # Git-ignored (~20GB hackathon data)
├── references/                  # Git-ignored (Bekkers code)
├── test/                        # Git-ignored (render scripts + sample outputs)
└── figures/                     # Pre-rendered outputs
```

---

## How It Works

The analysis runs as a four-stage pipeline:

**Stage 1 — Defender Orientation and Vision Model.** From 3D skeleton keypoints (head, shoulders, hips), compute the real body orientation of every defender at 50Hz. Build a probabilistic field-of-view map per defender using an adapted Bekkers (SSAC 2026) vision model with real head angles and shoulder-based occlusions.

**Stage 2 — Theta Computation and Diagonal Classification.** For every action (pass, carry, off-ball run), compute theta: the angle between the nearest defender's body orientation and the direction of the incoming threat. Classify actions as forward, diagonal, sideways, or backward per DFL's official angle ranges. Test whether diagonal actions have significantly higher theta.

**Stage 3 — Defensive Disruption with Orientation.** Reimplement D-Def (Goes et al. 2019) decomposed into longitudinal (PC1) and lateral (PC2) disruption. Cross with theta to show that diagonal actions (high theta) disrupt BOTH axes simultaneously, while orthogonal actions only disrupt one.

**Stage 4 — Immediate Orientation-Aware PPCF and Diagonal Opportunity Surfaces.** Each player is modelled as an anisotropic Gaussian reach field centred on themselves, with sigma derived from the orientation-aware biomechanical delay: Vater (2024) reaction time + Dos'Santos (2018) change-of-direction deficit, applied to the real shoulder angle from the 3D skeleton. A defender with the ball in their blind spot literally has a hole in their reach field, and diagonals exploit it. From this, compute Diagonal Opportunity Surfaces showing where on the pitch a diagonal action gains the most control advantage.

**Scanning gate (cognitive layer).** A frame-exact possession timeline (carries + passes linked to receptions via play_id) tells us, at every frame, which player is on-ball. The on-ball player's full Bekkers vision plus a 2.5 s exponentially-decayed scanning memory is used to gate the DOS surface: only cells the player can SEE or has scanned recently are painted. The lookback grows linearly from zero at the moment a new player becomes on-ball, so the receiver never inherits the passer's pre-pass scanning context. The renderer applies a 1.5 m gaussian blur and a temporal EMA across frames (alpha=0.10, ~140 ms half-life) for visual readability, then maps the result onto a fixed display range via a smoothstep visibility curve and spline36 interpolation — no flicker, no on/off cliffs. Requires the cache to be generated with `PRE_WINDOW_FRAMES >= 150` (3 s) so the lookback is fully covered; run `test/regenerate_caches.py` to (re)build it.

---

## Data

5 Bundesliga 2025-26 matches with TRACAB GEN5/GEN6 3D skeleton data (21 keypoints per player at 50Hz) + DFL enriched events (xP, xG, pressure, PlayAngle) + positional tracking (25Hz).

See [docs/hackathon_data.md](docs/hackathon_data.md) for full data inventory.

---

## References

- Spielverlagerung (2025) — Tactical Theory: Diagonality
- Bekkers (SSAC 2026) — Wide Open Gazes: vision model + imminent pitch control
- Spearman (2018) — PPCF: Probabilistic Pitch Control Function
- Fernández & Bornn (2018) — Wide Open Spaces: influence-field pitch control
- Goes et al. (2019) — D-Def: defensive disruption metric
- Forcher et al. (2021) — D-Def validation: diagonal passes in successful attacks
- Vater (2024) — Reaction time grows with visual eccentricity
- Dos'Santos (2018) — Change-of-direction deficit quantification
