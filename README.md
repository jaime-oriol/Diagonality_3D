# Diagonality: The Best of Both Worlds

**AWS World Sports Innovation Cup 2026 — Challenge 2: Unlock the Power of 3D Football Data**

A framework that uses TRACAB 3D skeleton data to measure the real body
orientation of every player, validates Spielverlagerung's tactical theory
of *diagonality* against the data, and turns it into the **Diagonal
Opportunity Surface (DOS)** — a real-time, orientation-aware pitch map of
where a diagonal pass, carry, take-on or off-ball run breaks a defender
who cannot see it.

The full write-up is the compiled paper
[`deliverable/main.pdf`](deliverable/main.pdf), built from the LaTeX
source [`main.tex`](deliverable/main.tex) (a readable Markdown render is
in [`main.md`](deliverable/main.md)). The PR/FAQ and the executive
summary are in [`submission/`](submission/).

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
├── requirements.txt              # Pinned dependencies (numpy 1.26.4, ...)
├── .gitignore
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
│   ├── data/xt_singh_12x8.npy    # Bundled Karun Singh xT grid
│   └── viz/                      # Renderers (DOS, PPCF, Vision, passes)
│
├── pipeline/                     # Metric chain + stats + aggregation
│   ├── validate_dos_outcomes.py  # DOS over every on-ball action
│   ├── fix_carry_possession_link.py
│   ├── enrich_with_xt.py / enrich_with_ddef.py / enrich_with_theta.py
│   ├── enrich_full_metadata.py
│   ├── detect_runs.py            # Off-ball run detection + DOS
│   ├── compute_stats_xt.py / compute_stats_ddef.py / compute_stats_theta.py
│   ├── aggregate_rankings.py / select_top_events.py / build_summary.py
│   └── regenerate_caches.py      # Cache rebuild driver
├── renders/                      # MP4 / PNG render scripts
├── tests/                        # Pytest unit + integration suite
│
├── deliverable/                  # The submission write-up
│   ├── main.pdf                  # Compiled paper (the deliverable)
│   ├── main.tex                  # LaTeX source of the paper
│   ├── main.md                   # Readable Markdown render of the paper
│   ├── references.bib            # Bibliography
│   ├── prfaq.tex                 # PR/FAQ (LaTeX source)
│   ├── OUTLINE.md                # Build plan
│   ├── make_figures.py           # Analytical figure generator
│   ├── make_exec_summary.py      # Executive-summary slide generator
│   └── figures/                  # Figures used in the PDF
│
├── submission/                   # Submission artefacts
│   ├── executive_summary.pdf     # 5-slide executive summary
│   ├── executive_summary.pptx    # ... with the hero videos embedded
│   ├── prfaq.pdf                 # Compiled PR/FAQ
│   └── github_link.txt           # Link to this repository
│
├── results/                      # Versioned run results (only cache/ ignored)
│   ├── datasets/                 # Final enriched CSVs
│   ├── tables/                   # Rankings CSV + top-event JSON
│   ├── reports/                  # Statistical reports (.md) + plots (.png)
│   ├── renders/                  # Hero MP4 renders + pass-map PNG
│   └── cache/                    # Git-ignored (>100 MB per file)
│
├── figures/                      # Tracked assets
│   ├── logos/                    # Project + team logos
│   └── videos/                   # Hero MP4s (Vision / PPCF / DOS)
│
├── data/                         # Git-ignored — hackathon data NOT uploaded
└── outputs/                      # Git-ignored — local run artefacts
```

**Per the challenge rules, no hackathon data is uploaded to this
repository.** `data/` and `outputs/` are git-ignored; inside `results/`
only `cache/` (the per-match skeleton caches, >100 MB per file) is
git-ignored. The compiled deliverable `deliverable/main.pdf` is tracked
in the repository, alongside its LaTeX source and the Markdown render
`deliverable/main.md`. All versioned artefacts are reproducible from the
steps below.

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
mplsoccer 1.6.1, pyarrow 23.0.0, Pillow 12.0.0; `pytest` for the test
suite and `python-pptx` for the executive-summary slides).

---

## Reproducing the results

The five-match TRACAB dataset is **not redistributed** (challenge rules).
With the hackathon data placed under `data/hackathon/`, the full pipeline
runs from the repository root, in this order:

```bash
# 1. Cache — extract event-linked skeleton windows, per match
python3 -m src.preprocess <MATCH>          # one match, or:
python3 pipeline/regenerate_caches.py      # all five matches

# 2. Metric chain — DOS over every action, then enrichment
python3 pipeline/validate_dos_outcomes.py
python3 pipeline/fix_carry_possession_link.py
python3 pipeline/enrich_with_xt.py
python3 pipeline/enrich_with_ddef.py
python3 pipeline/enrich_with_theta.py
python3 pipeline/enrich_full_metadata.py

# 3. Off-ball runs — detection + DOS
python3 pipeline/detect_runs.py

# 4. Statistics, aggregation and the narrative index
python3 pipeline/compute_stats_xt.py
python3 pipeline/compute_stats_ddef.py
python3 pipeline/compute_stats_theta.py
python3 pipeline/aggregate_rankings.py
python3 pipeline/select_top_events.py
python3 pipeline/build_summary.py

# 5. Renders (MP4 / PNG, parallelised via ProcessPoolExecutor)
python3 renders/render_top_dos_videos.py
python3 renders/render_top_ppcf_videos.py
python3 renders/render_top_vision_videos.py
python3 renders/render_top_event_frames.py
python3 renders/render_top_player_passes.py
python3 renders/render_full_match.py --match Bayern_Hamburg --type dos

# 6. Deliverable figures + executive summary
python3 deliverable/make_figures.py
python3 deliverable/make_exec_summary.py
```

Each metric-chain step reads the CSV the previous step wrote (intermediate
CSVs land in `outputs/intermediate/`). The whole chain is memory-safe:
chunked pyarrow predicate-pushdown reads keep peak memory near 1.6 GB, so
it runs on an 8 GB machine. The renders are embarrassingly parallel and
were produced on AWS EC2 (`c5.9xlarge` / `c5.18xlarge`).

The run already executed; its artefacts are versioned under
[`results/`](results/) (datasets, tables, reports, renders).

## Tests

```bash
python3 -m pytest tests/ -q
```

The integration test auto-skips when no local cache is present.

## Building the deliverable PDF

```bash
cd deliverable && tectonic main.tex      # or compile main.tex on Overleaf
```

The PR/FAQ compiles the same way (`tectonic prfaq.tex`).

---

## Submission notes

- **No hackathon data is uploaded.** `data/` is git-ignored; the
  repository ships only source code, the bundled public xT grid, the
  versioned `results/` artefacts and the deliverable.
- **Private repository.** If this repository is kept private, GitHub user
  **`MoellerO`** is invited as a collaborator, per the challenge
  instructions.
- The submission artefacts (executive summary, PR/FAQ, repository link)
  are collected in [`submission/`](submission/).

---

## Key references

- Spielverlagerung (2025) — Tactical Theory: Diagonality
- Hamilton (2025) — The Diagonalist Manifesto
- Bekkers (SSAC 2026) — Wide Open Gazes: vision model
- Spearman (2017, 2018) — pitch control / Beyond Expected Goals
- Fernández & Bornn (2018) — Wide Open Spaces
- Goes et al. (2019), Forcher et al. (2021, 2024) — D-Def defensive disruption
- Vater (2024) — reaction time and visual eccentricity
- Dos'Santos et al. (2018) — change-of-direction biomechanics
- Singh (2018) — Expected Threat

Full, verified bibliography in [`deliverable/references.bib`](deliverable/references.bib).
