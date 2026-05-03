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
├── README.md
├── requirements.txt                     # Pinned versions for reproducibility
│
├── src/                                 # Numpy-pure source modules
│   ├── loader.py                        # XML events / metadata / take-ons
│   ├── preprocess.py                    # Per-match cache extraction
│   ├── orientation.py                   # Head/shoulder/hip + velocity smoothing
│   ├── vision.py                        # Bekkers vision (FOV + per-occluder occlusion)
│   ├── theta.py                         # Theta per event (defender + receiver axes)
│   ├── ddef.py                          # D-Def + Forcher local + PCA
│   ├── ppcf.py                          # Immediate Orientation-Aware PPCF reach fields
│   ├── dos.py                           # Diagonal Opportunity Surfaces (3 APIs)
│   ├── possession.py                    # Frame-exact possession timeline
│   ├── scanning.py                      # On-ball FOV + 2.5s scanning memory
│   ├── xt.py                            # Karun Singh xT lookup (vectorized)
│   ├── data/xt_singh_12x8.npy           # Bundled xT grid
│   └── viz/                             # Renderers (DOS, PPCF, Vision, Passes)
│
├── scripts/                             # AWS execution
│   ├── aws_setup.sh                     # EC2 bootstrap (ffmpeg + miniconda + deps)
│   └── aws_pipeline.py                  # Master orchestrator (28 steps + resume)
│
├── test/                                # Pipeline scripts + unit tests (.py only)
│   # Metric chain
│   ├── validate_dos_outcomes.py         # DOS over every event (~6.7k)
│   ├── fix_carry_possession_link.py     # Carry↔possession link fix
│   ├── enrich_with_xt.py                # xt_origin / xt_dest / xt_delta
│   ├── enrich_with_ddef.py              # 14-var state vector + PCA
│   ├── enrich_with_theta.py             # Defender + receiver theta
│   ├── enrich_full_metadata.py          # player_id / player_name / team_name
│   # Stats + aggregations
│   ├── compute_stats_{xt,ddef,theta}.py # MD reports + plots
│   ├── aggregate_rankings.py            # 16 storytelling tables
│   ├── select_top_events.py             # Top-N event picker for renders
│   ├── build_summary.py                 # outputs/SUMMARY.md
│   # Renders (parallelized where possible)
│   ├── render_kane_goal*.py             # Kane goal: vision + PPCF + DOS
│   ├── render_olise_passes.py           # Olise pass map
│   ├── render_top_dos_videos.py         # Top-10 DOS videos
│   ├── render_top_vision_videos.py      # Top-8 vision videos
│   ├── render_top_ppcf_videos.py        # Top-8 PPCF videos
│   ├── render_top_event_frames.py       # ~60 PNG gallery
│   ├── render_top_player_passes.py      # ~20 player pass maps
│   └── test_*.py                        # 136 tests across 7 files
│
├── figures/                             # Tracked logos (Logo.png + logos/)
├── cache/                               # Git-ignored (preprocessed per match)
├── data/                                # Git-ignored (raw hackathon ~20 GB)
├── outputs/                             # Git-ignored (pipeline run artefacts)
│   ├── tables/                          # Rankings CSV
│   ├── frames/                          # PNG gallery + pass maps
│   ├── videos/                          # MP4 renders
│   ├── reports/                         # Stats reports + plots
│   ├── SUMMARY.md                       # Master index
│   └── .pipeline_state.json             # Resume state
└── docs/                                # Git-ignored (research + narrative)
```

All generated artefacts live in `outputs/`. The `test/` folder holds **only Python source**.

---

## How It Works

The analysis runs as a four-stage pipeline:

**Stage 1 — Defender Orientation and Vision Model.** From 3D skeleton keypoints (head, shoulders, hips), compute the real body orientation of every defender at 50Hz. Build a probabilistic field-of-view map per defender using an adapted Bekkers (SSAC 2026) vision model with real head angles and shoulder-based occlusions.

**Stage 2 — Theta Computation and Diagonal Classification.** For every action (pass, carry, off-ball run), compute theta: the angle between the nearest defender's body orientation and the direction of the incoming threat. Classify actions as forward, diagonal, sideways, or backward per DFL's official angle ranges. Test whether diagonal actions have significantly higher theta.

**Stage 3 — Defensive Disruption with Orientation.** Reimplement D-Def (Goes et al. 2019) decomposed into longitudinal (PC1) and lateral (PC2) disruption. Cross with theta to show that diagonal actions (high theta) disrupt BOTH axes simultaneously, while orthogonal actions only disrupt one.

**Stage 4 — Immediate Orientation-Aware PPCF and Diagonal Opportunity Surfaces.** Each player is modelled as an anisotropic Gaussian reach field centred on themselves, with sigma derived from the orientation-aware biomechanical delay: Vater (2024) reaction time + Dos'Santos (2018) change-of-direction deficit, applied to the real shoulder angle from the 3D skeleton. A defender with the ball in their blind spot literally has a hole in their reach field, and diagonals exploit it. From this, compute Diagonal Opportunity Surfaces showing where on the pitch a diagonal action gains the most control advantage.

**Scanning gate (cognitive layer).** A frame-exact possession timeline (carries + passes linked to receptions via play_id) tells us, at every frame, which player is on-ball. The on-ball player's full Bekkers vision plus a 2.5 s exponentially-decayed scanning memory is used to gate the DOS surface: only cells the player can SEE or has scanned recently are painted. The lookback grows linearly from zero at the moment a new player becomes on-ball, so the receiver never inherits the passer's pre-pass scanning context. The renderer applies a 1.5 m gaussian blur and a temporal EMA across frames (alpha=0.10, ~140 ms half-life) for visual readability, then maps the result onto a fixed display range via a smoothstep visibility curve and spline36 interpolation — no flicker, no on/off cliffs. Requires the cache to be generated with `PRE_WINDOW_FRAMES >= 150` (3 s) so the lookback is fully covered; run `test/regenerate_caches.py` to (re)build it.

**Empirical validation chain.** Every per-event metric is computed by a separate enrichment script in `test/`, chained via the master orchestrator `scripts/aws_pipeline.py`:

1. **DOS** — evaluates the Diagonal Opportunity Surface for every pass, carry and take-on of the 5 cached matches (~6.7k events). The model receives **no xG, no xT, no success labels** as inputs — only geometry + skeleton orientation + the Bekkers vision model. Carries use a multi-frame DOS path (5 uniformly-spaced samples, instantaneous skeleton velocity direction, virtual destination `pos + vel * 1s`, aggregated with max). Take-ons use the WinnerPlayer's instantaneous velocity at the duel frame.
2. **xT-delta** — Karun Singh (2018) per-action value `ΔxT = xT(end) - xT(origin)`. Causal-fair outcome that doesn't dilute carry/take-on signal the way `parent_possession_xg` does.
3. **D-Def + PCA** — Goes (2019) defensive disruption with Forcher (2024) local extension. 14-var state vector at t0 + at t0+3s, Z-score + PCA cross-dataset → PC1 (longitudinal), PC2 (lateral), PC3 (shape), bi-axial balance.
4. **Theta orientation** — defender disruption (`mean_theta_shoulder`, `nearest_in_blind`, `n_wrongfooted`) + receiver advantage (`receiver_open_angle`, `receiver_turn_needed`, `n_teammates_in_fov`).
5. **Player + team metadata** — joins event_id with the official roster (`MatchInformations.xml`) so rankings can be aggregated per player and per team.

Memory-safe throughout (pyarrow predicate-pushdown chunks, ~1.6 GB peak per chunk). All intermediate CSVs live in `test/` (gitignored); final reports + tables + visualisations land in `outputs/`.

---

## Running the full pipeline

The master orchestrator `scripts/aws_pipeline.py` runs every stage end-to-end with full resume support (`outputs/.pipeline_state.json`):

```bash
# On AWS EC2 (after scripts/aws_setup.sh has bootstrapped conda + ffmpeg + deps):
python3 scripts/aws_pipeline.py --list                # Show plan + per-step state
python3 scripts/aws_pipeline.py                       # Run / resume
python3 scripts/aws_pipeline.py --from STEP_NAME      # Restart from a step
python3 scripts/aws_pipeline.py --only STEP1 STEP2    # Run only specific steps
python3 scripts/aws_pipeline.py --force               # Rerun everything
```

The orchestrator runs 28 steps in 7 stages: cache regeneration × 5 matches → metric chain (validate → fix → xt → ddef → theta) → stats reports → metadata + rankings → video renders (Kane goal × 3 + top-10 DOS + top-8 vision + top-8 PPCF, parallelized) → frame gallery + pass maps → master `SUMMARY.md`. End-to-end wall time on an AWS `c5.9xlarge` (36 vCPU, 72 GB RAM): ~3-5 hours.

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
