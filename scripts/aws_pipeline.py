"""Master orchestrator for the Diagonality 3D AWS pipeline.

Single entrypoint that runs every stage end-to-end with full resume
support. Designed for the AWS Innovation Sandbox m5.2xlarge instance
($0.504/h, 32 GB RAM, 8 vCPU).

Usage:
    python3 scripts/aws_pipeline.py                # resume (default)
    python3 scripts/aws_pipeline.py --force        # rerun everything
    python3 scripts/aws_pipeline.py --from STEP    # restart from STEP
    python3 scripts/aws_pipeline.py --only STEP1 STEP2  # run only these
    python3 scripts/aws_pipeline.py --dry-run      # print plan, do nothing

Resume logic:
  Each step declares an `outputs[]` list (file paths it produces). On
  start-up the orchestrator checks each output's existence + size and
  skips steps whose outputs are all present and non-empty. A step's
  state is also persisted in `outputs/.pipeline_state.json` for
  bookkeeping (timestamps, durations, return codes).

Pipeline (20 stages):

  STAGE 0 — DATA
    cache_<match>            preprocess.py per match (5 matches)
  STAGE 1 — METRICS
    validate_dos             test/validate_dos_outcomes.py
    fix_carry                test/fix_carry_possession_link.py
    enrich_xt                test/enrich_with_xt.py
    enrich_ddef              test/enrich_with_ddef.py
    enrich_theta             test/enrich_with_theta.py
  STAGE 2 — STATS REPORTS
    stats_xt                 test/compute_stats_xt.py
    stats_ddef               test/compute_stats_ddef.py
    stats_theta              test/compute_stats_theta.py
  STAGE 3 — METADATA + RANKINGS
    enrich_meta              test/enrich_full_metadata.py
    rankings                 test/aggregate_rankings.py
    select_top               test/select_top_events.py
  STAGE 4 — VIDEO RENDERS
    render_kane_vision       test/render_kane_goal.py
    render_kane_ppcf_video   test/render_kane_goal_ppcf_video.py
    render_kane_dos_video    test/render_kane_goal_dos_video.py
    render_top_dos_videos    test/render_top_dos_videos.py
  STAGE 5 — FRAME RENDERS
    render_kane_ppcf_png     test/render_kane_goal_ppcf_png.py
    render_top_event_frames  test/render_top_event_frames.py
    render_olise_passes      test/render_olise_passes.py
    render_player_passes     test/render_top_player_passes.py
  STAGE 6 — SUMMARY
    build_summary            test/build_summary.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / "outputs" / ".pipeline_state.json"
LOG_FILE = REPO_ROOT / "outputs" / "log.txt"


# ── Step definition ────────────────────────────────────────────────────

@dataclass
class Step:
    name: str
    command: List[str]                  # e.g. ["python3", "test/validate_dos_outcomes.py"]
    outputs: List[Path]                 # file paths to verify completion
    description: str = ""
    skip_if_done: bool = True

    def is_done(self) -> bool:
        """All declared outputs must exist and be non-empty."""
        if not self.outputs:
            return False
        for p in self.outputs:
            if not p.exists():
                return False
            try:
                if p.is_file() and p.stat().st_size == 0:
                    return False
            except OSError:
                return False
        return True

    def run(self, log_fh) -> int:
        """Execute the step's command and tee its output. Returns rc."""
        cmd_str = " ".join(self.command)
        msg = f"\n{'='*72}\n[{self.name}] {self.description}\n  $ {cmd_str}\n{'='*72}\n"
        print(msg, end="")
        log_fh.write(msg); log_fh.flush()
        proc = subprocess.Popen(
            self.command, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_fh.write(line)
        proc.wait()
        return proc.returncode


# ── State persistence ──────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"version": 1, "steps": {}}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ── Pipeline definition ────────────────────────────────────────────────

MATCHES = [
    "Bayern_Hamburg", "Dortmund_Stuttgart",
    "Frankfurt_Bayern", "Frankfurt_Union", "Union_Bayern",
]


def _cache_outputs(match: str) -> List[Path]:
    base = REPO_ROOT / "cache" / match
    return [base / f for f in
            ("skeleton.parquet", "ball.parquet",
             "events.parquet", "possessions.parquet", "metadata.json")]


def build_pipeline() -> List[Step]:
    test_csv = REPO_ROOT / "test"
    out_root = REPO_ROOT / "outputs"
    tables = out_root / "tables"

    steps: List[Step] = []

    # ── STAGE 0: cache (one step per match) ───────────────────────
    for m in MATCHES:
        steps.append(Step(
            name=f"cache_{m}",
            command=["python3", "-m", "src.preprocess", m],
            outputs=_cache_outputs(m),
            description=f"Preprocess + cache match {m}",
        ))

    # ── STAGE 1: metrics chain ────────────────────────────────────
    steps.append(Step(
        name="validate_dos",
        command=["python3", "test/validate_dos_outcomes.py"],
        outputs=[test_csv / "dos_validation_raw.csv",
                 test_csv / "dos_validation_quintiles.png",
                 test_csv / "dos_validation_report.md"],
        description="Compute DOS for every event across 5 matches (chunked, ~25 min)",
    ))
    steps.append(Step(
        name="fix_carry",
        command=["python3", "test/fix_carry_possession_link.py"],
        outputs=[test_csv / "dos_validation_raw_fixed.csv",
                 test_csv / "dos_validation_report_fixed.md",
                 test_csv / "dos_validation_quintiles_fixed.png"],
        description="Fix carry/possession xG link (post-process)",
    ))
    steps.append(Step(
        name="enrich_xt",
        command=["python3", "test/enrich_with_xt.py"],
        outputs=[test_csv / "dos_validation_raw_xt.csv"],
        description="Enrich with xT-delta (Karun Singh 2018)",
    ))
    steps.append(Step(
        name="enrich_ddef",
        command=["python3", "test/enrich_with_ddef.py"],
        outputs=[test_csv / "dos_validation_raw_xt_ddef.csv"],
        description="Enrich with D-Def Goes 2019 + Forcher 2024 + PCA",
    ))
    steps.append(Step(
        name="enrich_theta",
        command=["python3", "test/enrich_with_theta.py"],
        outputs=[test_csv / "dos_validation_raw_xt_ddef_theta.csv"],
        description="Enrich with theta orientation metrics (defender + receiver)",
    ))

    # ── STAGE 2: stats reports ────────────────────────────────────
    steps.append(Step(
        name="stats_xt",
        command=["python3", "test/compute_stats_xt.py"],
        outputs=[test_csv / "dos_validation_report_xt.md",
                 test_csv / "dos_validation_quintiles_xt.png"],
        description="xT-based outcome statistics report",
    ))
    steps.append(Step(
        name="stats_ddef",
        command=["python3", "test/compute_stats_ddef.py"],
        outputs=[test_csv / "dos_validation_report_ddef.md",
                 test_csv / "dos_validation_ddef_axes.png"],
        description="D-Def H2 statistics report (bi-axial disruption)",
    ))
    steps.append(Step(
        name="stats_theta",
        command=["python3", "test/compute_stats_theta.py"],
        outputs=[test_csv / "dos_validation_report_theta.md",
                 test_csv / "dos_validation_theta_axes.png"],
        description="Theta orientation report (storytelling cifras)",
    ))

    # ── STAGE 3: metadata + rankings + selection ──────────────────
    steps.append(Step(
        name="enrich_meta",
        command=["python3", "test/enrich_full_metadata.py"],
        outputs=[test_csv / "dos_validation_full.csv"],
        description="Add player_id / player_name / team_name to event CSV",
    ))
    steps.append(Step(
        name="rankings",
        command=["python3", "test/aggregate_rankings.py"],
        outputs=[
            tables / "players_by_dos.csv",
            tables / "players_by_xt.csv",
            tables / "players_by_ddef.csv",
            tables / "players_by_wrongfooting.csv",
            tables / "players_by_awareness_low.csv",
            tables / "players_by_diag_share.csv",
            tables / "players_by_volume.csv",
            tables / "teams_summary.csv",
            tables / "matches_summary.csv",
            tables / "matches_team_dir_breakdown.csv",
            tables / "direction_breakdown.csv",
            tables / "zones_origin_breakdown.csv",
            tables / "zones_dest_breakdown.csv",
            tables / "zones_x_direction.csv",
            tables / "events_per_player_match.csv",
            tables / "top_events_full_vector.csv",
        ],
        description="Build per-player / per-team / per-match / per-zone ranking tables (incl. awareness + halfspace)",
    ))
    steps.append(Step(
        name="select_top",
        command=["python3", "test/select_top_events.py"],
        outputs=[
            tables / "top_dos_events.json",
            tables / "top_dos_carries.json",
            tables / "top_dos_passes.json",
            tables / "top_xt_events.json",
            tables / "top_ddef_events.json",
            tables / "highlight_events.json",
        ],
        description="Pick top-N events for video / frame rendering",
    ))

    # ── STAGE 4: video renders ────────────────────────────────────
    # The Kane goal renders write into test/, but we ALSO copy them
    # into outputs/videos/ for the final deliverable. We rely on the
    # render scripts to write the test/* path; the copy is a separate
    # step so it's resumable.
    steps.append(Step(
        name="render_kane_vision",
        command=["python3", "test/render_kane_goal.py"],
        outputs=[test_csv / "vision_kane_goal5.mp4"],
        description="Render Kane goal vision video (50fps, dpi=200)",
    ))
    steps.append(Step(
        name="render_kane_ppcf_video",
        command=["python3", "test/render_kane_goal_ppcf_video.py"],
        outputs=[test_csv / "ppcf_kane_goal5.mp4"],
        description="Render Kane goal PPCF reach-field video",
    ))
    steps.append(Step(
        name="render_kane_dos_video",
        command=["python3", "test/render_kane_goal_dos_video.py"],
        outputs=[test_csv / "dos_kane_goal5.mp4"],
        description="Render Kane goal DOS + shadow video",
    ))
    # Top DOS events: cannot be a single output check (10 separate
    # files). The render script handles per-file resume internally.
    # We add it as a step that ALWAYS runs but is fast on full skip.
    steps.append(Step(
        name="render_top_dos_videos",
        command=["python3", "test/render_top_dos_videos.py"],
        outputs=[],   # internal resume; always invoke
        description="Render DOS videos for top-N selected events",
        skip_if_done=False,
    ))

    # ── STAGE 5: frame renders ────────────────────────────────────
    steps.append(Step(
        name="render_kane_ppcf_png",
        command=["python3", "test/render_kane_goal_ppcf_png.py"],
        outputs=[test_csv / "ppcf_kane_goal5.png"],
        description="Render Kane goal PPCF static frame",
    ))
    steps.append(Step(
        name="render_top_event_frames",
        command=["python3", "test/render_top_event_frames.py"],
        outputs=[],
        description="Render top-N event PNGs (DOS + PPCF gallery)",
        skip_if_done=False,
    ))
    steps.append(Step(
        name="render_olise_passes",
        command=["python3", "test/render_olise_passes.py"],
        outputs=[test_csv / "passes_olise_Bayern_Hamburg.png"],
        description="Render Olise pass map (Bayern_Hamburg)",
    ))
    steps.append(Step(
        name="render_player_passes",
        command=["python3", "test/render_top_player_passes.py"],
        outputs=[],
        description="Render top-N player pass maps across matches",
        skip_if_done=False,
    ))

    # ── STAGE 6: summary ──────────────────────────────────────────
    steps.append(Step(
        name="build_summary",
        command=["python3", "test/build_summary.py"],
        outputs=[REPO_ROOT / "outputs" / "SUMMARY.md"],
        description="Build outputs/SUMMARY.md (master index)",
    ))

    # ── STAGE 7: copy media into outputs/videos/ ──────────────────
    # Done as final post-step so the SUMMARY links resolve.
    steps.append(Step(
        name="collect_media",
        command=["bash", "-c",
                 "mkdir -p outputs/videos outputs/frames && "
                 "cp -n test/vision_kane_goal5.mp4 outputs/videos/kane_vision.mp4 2>/dev/null || true; "
                 "cp -n test/ppcf_kane_goal5.mp4 outputs/videos/kane_ppcf.mp4 2>/dev/null || true; "
                 "cp -n test/dos_kane_goal5.mp4 outputs/videos/kane_dos.mp4 2>/dev/null || true; "
                 "cp -n test/ppcf_kane_goal5.png outputs/frames/kane_ppcf.png 2>/dev/null || true; "
                 "cp -n test/passes_olise_*.png outputs/frames/pass_maps/ 2>/dev/null || true; "
                 "cp -n test/dos_validation_report*.md outputs/reports/ 2>/dev/null || true; "
                 "cp -n test/dos_validation_*.png outputs/reports/ 2>/dev/null || true; "
                 "echo 'Collected media into outputs/'"],
        outputs=[],
        description="Copy renders + reports into outputs/ tree",
        skip_if_done=False,
    ))

    return steps


# ── Driver ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Rerun every step regardless of cached outputs")
    ap.add_argument("--from", dest="from_step", default=None,
                    help="Restart from the given step name (skip earlier steps "
                         "but re-run from this one onwards)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only the listed steps (in pipeline order)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and what would be skipped, do nothing")
    ap.add_argument("--list", action="store_true",
                    help="List all steps and their cached state, exit")
    args = ap.parse_args()

    pipeline = build_pipeline()
    state = _load_state()

    if args.list:
        print(f"{'STEP':35s} {'STATUS':10s} OUTPUTS")
        for s in pipeline:
            status = "DONE" if s.is_done() else "PENDING"
            n_out = len(s.outputs) if s.outputs else "internal"
            print(f"  {s.name:35s} {status:10s} {n_out}")
        return

    # Determine which steps to actually run
    to_run = []
    started_running = (args.from_step is None)
    only = set(args.only) if args.only else None

    for s in pipeline:
        if only is not None:
            if s.name in only:
                to_run.append(s)
            continue

        if not started_running:
            if s.name == args.from_step:
                started_running = True
                to_run.append(s)
            continue

        if args.force:
            to_run.append(s)
            continue

        if s.skip_if_done and s.is_done():
            print(f"  SKIP {s.name} (outputs exist)")
            continue

        to_run.append(s)

    print(f"\nPlanning to run {len(to_run)} of {len(pipeline)} steps:\n")
    for s in to_run:
        print(f"  ▶ {s.name:35s} — {s.description}")

    if args.dry_run:
        print("\n--dry-run — exiting without execution")
        return

    # Execute
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a", buffering=1)
    log_fh.write(f"\n\n=== pipeline run @ {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")

    started = time.time()
    n_ok = 0; n_fail = 0
    for s in to_run:
        t0 = time.time()
        rc = s.run(log_fh)
        dt = time.time() - t0
        state["steps"][s.name] = {
            "rc": rc, "duration_s": round(dt, 1),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_state(state)
        if rc == 0:
            n_ok += 1
            log_fh.write(f"[{s.name}] OK in {dt:.1f}s\n")
        else:
            n_fail += 1
            log_fh.write(f"[{s.name}] FAILED rc={rc} in {dt:.1f}s\n")
            print(f"\n!! [{s.name}] FAILED with rc={rc} after {dt:.1f}s")
            print("   Stopping pipeline. Re-run to resume from here.")
            log_fh.close()
            sys.exit(rc)

    elapsed = time.time() - started
    print(f"\n{'='*72}")
    print(f"  PIPELINE COMPLETE — {n_ok}/{len(to_run)} steps OK ({elapsed/60:.1f} min)")
    print(f"  Log: {LOG_FILE}")
    print(f"  State: {STATE_FILE}")
    print(f"  Summary: {REPO_ROOT / 'outputs' / 'SUMMARY.md'}")
    print(f"{'='*72}")
    log_fh.close()


if __name__ == "__main__":
    main()
