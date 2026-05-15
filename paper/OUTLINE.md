# Paper build plan — Diagonality: The Best of Both Worlds

**Deliverable:** `docs/Challenge_3D_Football_Data.pdf`
**Source:** `paper/` (LaTeX, compile on Overleaf with pdfLaTeX + BibTeX)
**Event / deadline:** AWS World Sports Innovation Cup 2026 — Challenge 2. Submission 17 May 2026.

---

## Locked decisions

- **Title:** *Diagonality: The Best of Both Worlds* — Quantifying Football's
  Progression–Safety Trade-off with 3D Skeleton Tracking and Orientation-Aware
  Pitch Control.
- **Thesis (the axis):** the diagonal is the Pareto optimum of the
  progression↔safety trade-off. Forward = progression pole (high xT, 44%
  retention). Sideways = safety pole (78% retention, negative xT). Diagonal =
  the optimum (xT ≈ forward, retention solid, 82% positive-xT rate — highest).
- **Off-ball runs:** NOT in the submission. Framed as flagship future work
  (the framework is action-type-agnostic; only the run-detection front-end is
  missing). The paper does not depend on them.
- **Language:** English. **Toolchain:** LaTeX → Overleaf. **Videos:**
  referenced (not embedded) to the GitHub repo.
- **Repo:** https://github.com/jaime-oriol/Diagonality_3D

---

## Structure (main.tex)

1. Introduction — the two coaching dogmas as one axis; the diagonal optimum; orientation was unmeasurable until 3D skeleton; contributions.
2. Related Work — pitch control → vision → D-Def → action value → tactical theory → biomechanics. The unfilled intersection.
3. Data — 5 Bundesliga matches, TRACAB 21-keypoint 3D skeleton @ 50 Hz, kpi_data events, 6,923 actions.
4. Methods — (4.1) orientation, (4.2) vision model, (4.3) OA-PPCF, (4.4) DOS, (4.5) scanning gate, (4.6) metric chain + validation design.
5. Results — (5.1) the trade-off [HERO], (5.2) DOS predicts value, (5.3) mechanism I: defender, (5.4) mechanism II: receiver, (5.5) case studies.
6. Applications — match prep, post-match, scouting, broadcast.
7. Limitations & Future Work — 5 matches; off-ball runs; OA-PPCF AUC; DOS↔D-Def.
8. Conclusion.
9. References.

## Figures

| # | File (paper/figures/) | Source | Video ref |
|---|---|---|---|
| 1 Vision | impl_vision.png | extracted from kane_vision.mp4 | aws_results/videos/kane_vision.mp4 |
| 2 PPCF | impl_ppcf.png | aws_results/frames/kane_ppcf.png | aws_results/videos/kane_ppcf.mp4 |
| 3 DOS | impl_dos.png | extracted from kane_dos.mp4 | aws_results/videos/kane_dos.mp4 |
| 4 Trade-off | TODO — Pareto-plane scatter (build from dos_validation_full.csv or TikZ) | — | — |
| 5 Quintiles | results_quintiles.png | reports/dos_validation_quintiles_xt.png | — |
| 6 D-Def | results_ddef.png | reports/dos_validation_ddef_axes.png | — |
| 7 Receiver | results_theta.png | reports/dos_validation_theta_axes.png | — |
| 8 Olise | olise_passes.png | frames/pass_maps/passes_olise_Bayern_Hamburg.png | — |

Spares copied: impl_dos_kane.png, impl_dos_carry.png, impl_ppcf_kane.png.

## Numbers cheat-sheet (from aws_results/reports)

- Trade-off: forward xTΔ +0.0028 / ret 44% / xT>0 74% (n=1137); sideways −0.0012 / 78% / 32% (n=3514); **diagonal +0.0027 / 63% / 82% (n=2272)**.
- DOS↔xT: MWU all p=1.16e-25 (rbc +0.145); carries p=1.22e-36 (rbc +0.402). Logistic dos coef +10.86, p=2.6e-5 (controls: distance, pressure, defender counts). Quintiles xT>0 rate: 46→51→58→60→63%.
- D-Def: PC1+PC2 combined median diagonal 2.91 vs forward 2.62 (p=2e-4); PC2 diagonal vs forward p=7e-12.
- Receiver (passes): turn-to-goal forward 144.5° / diagonal 108.5° / sideways 59.9° (diag<fwd p=1.6e-59); team-mates in FOV forward 4.95 / diagonal 6.03 / sideways 5.12 (diag>fwd p=6.7e-11).
- DOS distribution: mean 0.0117, 91.7% of actions > 0.
- Top players by DOS: J. Leweling (Stuttgart) 0.0169, Tiago Tomás 0.0162, Luis Díaz 0.0159. Top team: VfB Stuttgart 0.0129.

## Build order

1. [done] Scaffold: paper/, figures copied, main.tex skeleton, references.bib.
2. Methods prose (4.1–4.6) — most stable, write next.
3. Results prose (5.1–5.5) — numbers locked above.
4. Introduction (contributions) + Related Work.
5. Data detail, Applications, Limitations, Conclusion.
6. Figure 4 (trade-off Pareto plane) — generate.
7. Verify references.bib `%% VERIFY` entries (Goes/Forcher exact citations).
8. Compile on Overleaf → export → docs/Challenge_3D_Football_Data.pdf.
