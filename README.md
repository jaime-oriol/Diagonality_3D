# Diagonality: The Best of Both Worlds

**AWS World Sports Innovation Cup 2026 — Challenge 2: Unlock the Power of 3D Football Data**

A framework that uses TRACAB 3D skeleton data to measure the real body
orientation of every player, validates Spielverlagerung's tactical theory
of *diagonality* against the data, and turns it into the **Diagonal
Opportunity Surface (DOS)** — a real-time, orientation-aware pitch map of
where a diagonal pass, carry, take-on or off-ball run breaks a defender
who cannot see it.

The full write-up is the deliverable PDF, built from [`deliverable/`](deliverable/):
**Diagonality: The Best of Both Worlds**.

---

## Research question

Do diagonal actions systematically exploit defenders' visual blind spots
more than orthogonal ones, and can that mechanism be turned into a
real-time, coach-facing tool?

The framework operates in four orientation-aware stages — a per-defender
vision model, an Orientation-Aware Pitch Control Function, the DOS
surface, and a cognitive scanning gate — and is validated against 6,923
on-ball actions and 2,354 off-ball runs from five Bundesliga 2025-26
matches.

---

## Repository structure

```
Diagonality_3D/
├── README.md
├── CLAUDE.md                     # Development notes
├── requirements.txt              # Pinned dependencies (numpy 1.26.4, ...)
│
├── src/                          # Source modules (pure numpy, no frameworks)
│   ├── loader.py                 # XML events / metadata / take-ons
│   ├── preprocess.py             # Per-match cache extraction
│   ├── orientation.py            # Head/shoulder/hip orientation + velocity
│   ├── vision.py                 # Bekkers vision model (FOV + occlusion)
│   ├── ppcf.py                   # Orientation-Aware PPCF (reach fields)
│   ├── dos.py                    # Diagonal Opportunity Surface
│   ├── possession.py             # Frame-exact possession timeline
│   ├── scanning.py               # On-ball FOV + 2.5 s scanning memory
│   ├── runs.py                   # Off-ball run detection
│   ├── theta.py                  # Defender + receiver orientation metrics
│   ├── ddef.py                   # D-Def defensive disruption + PCA
│   ├── xt.py                     # Karun Singh xT lookup
│   ├── skeleton_chunks.py        # Memory-safe skeleton-cache reader
│   └── viz/                      # Renderers (DOS, PPCF, Vision, passes)
│
├── pipeline/                     # Metric chain + stats + aggregation
│   ├── validate_dos_outcomes.py  # DOS over every on-ball action
│   ├── detect_runs.py            # Off-ball run detection + DOS
│   ├── enrich_with_*.py          # xT / D-Def / theta / metadata
│   ├── compute_stats_*.py        # Statistical reports
│   └── aggregate_rankings.py ... # Rankings, selection, summary, caches
├── renders/                      # MP4 / PNG render scripts
├── tests/                        # Pytest unit + integration suite
│
├── deliverable/                  # The submission write-up (LaTeX)
│   ├── main.tex / references.bib / OUTLINE.md
│   ├── make_figures.py           # Analytical figure generator
│   └── figures/                  # Figures used in the PDF
│
├── results/                      # Versioned run results (only cache/ ignored)
│   ├── datasets/ tables/ reports/ renders/ SUMMARY.md
│   └── cache/                    # Git-ignored (>100 MB per file)
│
├── figures/                      # Tracked assets (logos, hero videos)
│
├── data/                         # Git-ignored — hackathon data NOT uploaded
├── cache/                        # Git-ignored — preprocessed per match
└── outputs/                      # Git-ignored — local run artefacts
```

**Per the challenge rules, no hackathon data is uploaded to this
repository.** `data/`, `cache/` and `outputs/` are git-ignored; inside
`results/` only `cache/` (the per-match skeleton caches, >100 MB per file)
is git-ignored. All versioned artefacts are reproducible from the steps
below.

---

## How it works

**Stage 1 — Vision.** From 3D skeleton keypoints (head, shoulders, hips),
compute the real orientation of every player at 50 Hz and a per-defender
probabilistic field of view, adapted from Bekkers (SSAC 2026): a 120°
cone with speed-dependent decay and torso occlusion from real shoulder
widths.

**Stage 2 — Orientation-Aware Pitch Control.** Each player is an
anisotropic Gaussian reach field whose width follows an orientation-aware
biomechanical delay — Vater (2024) reaction time + Dos'Santos (2018)
change-of-direction penalty applied to the real shoulder angle. A defender
with the threat in their blind spot has a hole in their control field.

**Stage 3 — Diagonal Opportunity Surface.** For every cell, DOS is the
extra attacker control bought by the best diagonal delivery over the best
orthogonal one. Defenders who cannot see the threat suffer an extra
detection delay that shrinks their reach. DOS scores passes, carries,
take-ons and off-ball runs with one action-independent routine.

**Stage 4 — Cognitive scanning gate.** A frame-exact possession timeline
identifies the on-ball player; DOS is gated by their field of view plus a
2.5 s exponentially-decayed scanning memory, so the surface only shows
what the player can actually perceive and act on.

The empirical validation is causally fair: the DOS model receives no xG,
no xT and no outcome labels — only geometry, the vision model and skeleton
orientation — so it is cross-checked against outcomes it never saw.

---

## Setup

```bash
conda create -n diag python=3.10
conda activate diag
pip install -r requirements.txt
```

Tested on Python 3.10. Dependencies are pinned in
[`requirements.txt`](requirements.txt) (numpy 1.26.4, pandas 2.3.3,
scipy 1.15.2, scikit-learn 1.7.2, statsmodels 0.14.5, matplotlib 3.10.8,
mplsoccer 1.6.1, pyarrow 23.0.0, Pillow 12.0.0).

## Reproducing the results

The five-match TRACAB dataset is not redistributed. With the hackathon
data placed under `data/hackathon/`, the pipeline runs from the repo root
in this order:

```bash
# 1. Cache — extract event-linked skeleton windows, per match
python3 -m src.preprocess <MATCH>

# 2. Metric chain — DOS over every action, then enrichment
python3 pipeline/validate_dos_outcomes.py
python3 pipeline/fix_carry_possession_link.py
python3 pipeline/enrich_with_xt.py
python3 pipeline/enrich_with_ddef.py
python3 pipeline/enrich_with_theta.py
python3 pipeline/enrich_full_metadata.py

# 3. Off-ball runs — detection + DOS
python3 pipeline/detect_runs.py

# 4. Statistics + aggregation
python3 pipeline/compute_stats_xt.py
python3 pipeline/compute_stats_ddef.py
python3 pipeline/compute_stats_theta.py
python3 pipeline/aggregate_rankings.py
python3 pipeline/select_top_events.py
python3 pipeline/build_summary.py

# 5. Renders (MP4 / PNG, parallelised)
python3 renders/render_top_dos_videos.py        # + ppcf / vision / frames

# 6. Deliverable figures
python3 deliverable/make_figures.py
```

The run already produced these artefacts; they are versioned under
[`results/`](results/). Memory-safe throughout (chunked pyarrow
predicate-pushdown reads).

## Tests

```bash
python3 -m pytest tests/ -q
```

The integration test auto-skips when no local cache is present.

## Building the deliverable PDF

```bash
cd deliverable && tectonic main.tex      # or compile main.tex on Overleaf
```

---

## Key references

- Spielverlagerung (2025) — Tactical Theory: Diagonality
- Hamilton (2024) — The Diagonalist Manifesto
- Bekkers (SSAC 2026) — Wide Open Gazes: vision model
- Spearman (2017, 2018) — pitch control / Beyond Expected Goals
- Fernández & Bornn (2018) — Wide Open Spaces
- Goes et al. (2019), Forcher et al. (2021, 2024) — D-Def defensive disruption
- Vater (2024) — reaction time and visual eccentricity
- Dos'Santos et al. (2018) — change-of-direction biomechanics
- Singh (2018) — Expected Threat

Full, verified bibliography in [`deliverable/references.bib`](deliverable/references.bib).
