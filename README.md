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
├── docs/                        # Research and data documentation
│   ├── propuesta_final.md       # Full project proposal
│   ├── hackathon_data.md        # Complete data inventory (verified)
│   ├── diagonality.md           # Spielverlagerung tactical theory
│   └── vision_de_paul.md        # Bekkers SSAC 2026 vision model paper
│
├── src/                         # Python modules
│   ├── loader.py                # Parquet skeleton + XML events unified loader
│   ├── orientation.py           # Head/shoulder/hip orientation from keypoints
│   ├── vision.py                # Vision model (adapted Bekkers)
│   ├── theta.py                 # Theta computation per event (passes, carries, runs)
│   ├── ddef.py                  # D-Def: defensive disruption (Goes et al.)
│   ├── ppcf.py                  # Orientation-Aware PPCF (pendiente)
│   ├── dos.py                   # Diagonal Opportunity Surfaces (pendiente)
│   └── viz/                     # Visualization package
│
├── notebooks/
│   └── main.ipynb               # Main deliverable — full analysis
│
├── data/                        # Git-ignored (~20GB hackathon data)
├── references/                  # Git-ignored (Bekkers code)
└── figures/                     # Pre-rendered outputs
```

---

## How It Works

The analysis runs as a four-stage pipeline:

**Stage 1 — Defender Orientation and Vision Model.** From 3D skeleton keypoints (head, shoulders, hips), compute the real body orientation of every defender at 50Hz. Build a probabilistic field-of-view map per defender using an adapted Bekkers (SSAC 2026) vision model with real head angles and shoulder-based occlusions.

**Stage 2 — Theta Computation and Diagonal Classification.** For every action (pass, carry, off-ball run), compute theta: the angle between the nearest defender's body orientation and the direction of the incoming threat. Classify actions as forward, diagonal, sideways, or backward per DFL's official angle ranges. Test whether diagonal actions have significantly higher theta.

**Stage 3 — Defensive Disruption with Orientation.** Reimplement D-Def (Goes et al. 2019) decomposed into longitudinal (PC1) and lateral (PC2) disruption. Cross with theta to show that diagonal actions (high theta) disrupt BOTH axes simultaneously, while orthogonal actions only disrupt one.

**Stage 4 — Orientation-Aware Pitch Control and Diagonal Opportunity Surfaces.** Extend Spearman's (2018) PPCF by modulating defender reaction time as a function of theta. Compute Diagonal Opportunity Surfaces showing where on the pitch a diagonal action gains the most control advantage.

---

## Data

5 Bundesliga 2025-26 matches with TRACAB GEN5/GEN6 3D skeleton data (21 keypoints per player at 50Hz) + DFL enriched events (xP, xG, pressure, PlayAngle) + positional tracking (25Hz).

See [docs/hackathon_data.md](docs/hackathon_data.md) for full data inventory.

---

## References

- Spielverlagerung (2025) — Tactical Theory: Diagonality
- Bekkers (SSAC 2026) — Wide Open Gazes: vision model with pose data
- Spearman (2018) — PPCF: Probabilistic Pitch Control Function
- Goes et al. (2019) — D-Def: defensive disruption metric
- Forcher et al. (2021) — D-Def validation: diagonal passes in successful attacks
- Vater (2024) — Reaction time grows with visual eccentricity
