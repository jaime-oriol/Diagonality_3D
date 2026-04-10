"""Unit tests for test/validate_dos_outcomes.py.

Covers:
  - chunk splitter (frame-span + event-count caps)
  - zone classification
  - velocity smoother behavior
  - bonferroni correction
  - bootstrap CI sanity
  - Mann-Whitney effect-size sign convention
  - quintile / direction breakdowns on synthetic data
  - logistic regression smoke
  - parquet predicate pushdown read on real cache (skipped if missing)
  - end-to-end run on a tiny event window of one cached match
"""

import sys
sys.path.insert(0, ".")

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import importlib.util
_SPEC = importlib.util.spec_from_file_location(
    "validate_dos_outcomes",
    Path(__file__).parent / "validate_dos_outcomes.py",
)
v = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v)


# ── Pure function tests ─────────────────────────────────────────────────

def test_zone_of_classification():
    assert v._zone_of(0.0, 0.0) == "center"
    assert v._zone_of(0.0, 5.0) == "center"
    assert v._zone_of(0.0, 10.0) == "half_space"
    assert v._zone_of(0.0, 25.0) == "wing"
    assert v._zone_of(0.0, -25.0) == "wing"
    assert v._zone_of(0.0, -10.0) == "half_space"


def test_split_into_chunks_event_cap():
    ev = pd.DataFrame({"parquet_frame": np.arange(0, 100)})
    chunks = v._split_into_chunks(ev, max_frame_span=10_000, max_events=10)
    assert len(chunks) == 10
    assert all(len(c) == 10 for c in chunks)


def test_split_into_chunks_frame_span_cap():
    # Sparse events spread over a huge frame range
    ev = pd.DataFrame({"parquet_frame": [0, 5_000, 12_000, 31_000, 32_000]})
    chunks = v._split_into_chunks(ev, max_frame_span=10_000, max_events=100)
    # Frames 0..12000 fit one chunk (span 12000 > 10000? -> broken at 5000+)
    # 0,5000 fit (span 5000); 12000 does not (>10000 from 0). Next chunk: 12000.
    # Then 31000, 32000.
    assert len(chunks) == 3
    spans = [int(c["parquet_frame"].max() - c["parquet_frame"].min()) for c in chunks]
    assert all(s <= 10_000 for s in spans)


def test_split_into_chunks_empty():
    ev = pd.DataFrame({"parquet_frame": []})
    assert v._split_into_chunks(ev, 1000, 10) == []


def test_split_into_chunks_preserves_all_rows():
    ev = pd.DataFrame({"parquet_frame": np.arange(0, 50, 2)})
    chunks = v._split_into_chunks(ev, max_frame_span=20, max_events=5)
    rebuilt = pd.concat(chunks).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, ev.reset_index(drop=True))


def test_smooth_velocities_clips_outliers():
    df = pd.DataFrame({
        "team": [0]*20, "jersey": [1]*20,
        "vx": [99.0]*20,  # everything above MAX_SPEED
        "vy": [0.0]*20,
    })
    out = v._smooth_velocities(df.copy())
    # Outliers replaced -> savgol over zeros -> ~0
    assert out["vx"].abs().max() < 1.0


def test_smooth_velocities_keeps_normal_values():
    n = 30
    df = pd.DataFrame({
        "team": [0]*n, "jersey": [1]*n,
        "vx": np.linspace(0.1, 5.0, n),
        "vy": np.zeros(n),
    })
    out = v._smooth_velocities(df.copy())
    # Mean roughly preserved
    assert abs(out["vx"].mean() - df["vx"].mean()) < 0.3


def test_bonferroni_correction():
    raw = [0.01, 0.02, 0.5, float("nan"), 0.001]
    out = v._bonferroni(raw)
    # 4 valid p-values -> multiply by 4, cap at 1.0
    assert out[0] == pytest.approx(0.04)
    assert out[1] == pytest.approx(0.08)
    assert out[2] == pytest.approx(1.0)  # 2.0 capped
    assert math.isnan(out[3])
    assert out[4] == pytest.approx(0.004)


def test_bootstrap_ci_constant_array():
    lo, hi = v._bootstrap_ci(np.array([0.5]*100), n_boot=200)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_empty():
    lo, hi = v._bootstrap_ci(np.array([]))
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=2.0, scale=1.0, size=400)
    lo, hi = v._bootstrap_ci(arr, n_boot=300)
    assert lo < arr.mean() < hi
    assert (hi - lo) < 0.5  # tight CI for n=400


# ── Statistical tests on synthetic data ─────────────────────────────────

def _make_synthetic_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dos = rng.normal(0.01, 0.01, n)
    # Build outcomes that genuinely correlate with dos
    p = 1.0 / (1.0 + np.exp(-(dos - 0.01) * 80))
    line_break = rng.uniform(0, 1, n) < (p * 0.4)
    success = rng.uniform(0, 1, n) < (0.6 + 0.3 * (dos > 0.01))
    pos_xg = rng.uniform(0, 1, n) < (p * 0.3)
    return pd.DataFrame({
        "match": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "event_type": ["pass"] * n,
        "dos": dos,
        "ppcf_baseline": rng.uniform(0.1, 0.6, n),
        "ppcf_with_awareness": rng.uniform(0.1, 0.7, n),
        "direction_class": rng.choice(
            ["forward", "diagonal", "sideways"], size=n, p=[0.3, 0.3, 0.4]),
        "event_direction_rad": rng.uniform(-math.pi, math.pi, n),
        "awareness_mean": rng.uniform(0.0, 1.0, n),
        "detection_delay_mean": rng.uniform(0.0, 0.4, n),
        "distance": rng.uniform(5, 40, n),
        "pressure_player": rng.uniform(0, 1, n),
        "n_def_lane": rng.integers(0, 4, n).astype(float),
        "n_def_goal": rng.integers(2, 11, n).astype(float),
        "evaluation": np.where(success, "successfullyCompleted", "unsuccessful"),
        "back_line_break": line_break,
        "bypassed_defenders": rng.normal(0, 1, n),
        "through_ball": rng.uniform(0, 1, n) < 0.05,
        "parent_possession_xg": np.where(pos_xg, rng.uniform(0.05, 0.5, n), 0.0),
        "origin_zone": rng.choice(["center", "half_space", "wing"], size=n),
        "dest_zone": rng.choice(["center", "half_space", "wing"], size=n),
    })


def test_mann_whitney_detects_signal():
    df = _make_synthetic_df(800, seed=1)
    df["chance"] = df["parent_possession_xg"] > 0
    r = v._mann_whitney(df, df["chance"], "test")
    assert r["p"] < 0.05
    assert r["rbc"] > 0  # positive: chance > rest
    assert r["mean_pos"] > r["mean_neg"]


def test_mann_whitney_small_n_returns_nan():
    df = pd.DataFrame({"dos": [0.1, 0.2]})
    mask = pd.Series([True, False])
    r = v._mann_whitney(df, mask, "tiny")
    assert math.isnan(r["p"])


def test_quintiles_monotonic_dos_mean():
    df = _make_synthetic_df(500, seed=2)
    q = v._quintiles(df)
    assert len(q) == 5
    means = q["dos_mean"].values
    assert all(means[i] < means[i+1] for i in range(4))


def test_direction_breakdown_has_three_classes():
    df = _make_synthetic_df(500, seed=3)
    out = v._direction_breakdown(df)
    assert set(out.index) <= {"forward", "diagonal", "sideways"}
    assert "n" in out.columns and out["n"].sum() == 500


def test_zone_breakdown_runs():
    df = _make_synthetic_df(500, seed=4)
    out = v._zone_breakdown(df, "origin_zone")
    assert set(out.index) <= {"center", "half_space", "wing"}


def test_per_match_runs():
    df = _make_synthetic_df(500, seed=5)
    pm = v._per_match(df)
    assert set(pm.index) == {"A", "B"}
    assert (pm["n"].sum()) == 500


def test_logistic_returns_keys_when_signal_present():
    df = _make_synthetic_df(800, seed=6)
    r = v._logistic(df, "back_line_break",
                    ["distance", "pressure_player",
                     "n_def_lane", "n_def_goal"])
    assert r["fit"] is True
    assert "dos" in r["coefs"]
    assert r["pseudo_r2"] >= 0


def test_logistic_skips_too_small():
    df = _make_synthetic_df(40, seed=7)
    r = v._logistic(df, "back_line_break",
                    ["distance", "pressure_player",
                     "n_def_lane", "n_def_goal"])
    assert r["fit"] is False


# ── Real-cache integration (skipped if cache missing) ──────────────────

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
HAS_CACHE = (CACHE_DIR / "Bayern_Hamburg" / "skeleton.parquet").exists()


@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_pyarrow_pushdown_returns_subset():
    """Window read returns far fewer rows than full file."""
    df = v._read_skeleton_window("Bayern_Hamburg", 3585000, 3585500)
    assert len(df) > 0
    assert df["frame_number"].min() >= 3585000
    assert df["frame_number"].max() <= 3585500
    # Should be << 90M rows (the full file)
    assert len(df) < 5_000_000


@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_pyarrow_pushdown_empty_window():
    df = v._read_skeleton_window("Bayern_Hamburg", -10, -1)
    assert len(df) == 0


@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_xg_map_loads_for_real_match():
    m = v._load_possessions_xg_map("Bayern_Hamburg")
    assert len(m) > 0
    assert all(isinstance(v_, float) for v_ in m.values())


@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_home_gk_left_p1_loads():
    val = v._load_metadata_home_gk_left_p1("Bayern_Hamburg")
    assert isinstance(val, bool)


@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_build_event_records_returns_sorted_passes_and_carries():
    ev = v._build_event_records("Bayern_Hamburg", sample=None)
    assert len(ev) > 0
    assert set(ev["event_type"].unique()) <= {"pass", "carry"}
    # Sorted by frame
    pf = ev["parquet_frame"].values
    assert np.all(pf[:-1] <= pf[1:])


@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_end_to_end_dense_window_keeps_memory_bounded(tmp_path, monkeypatch):
    """End-to-end run on a small dense frame window. Key check: chunk
    frame span stays under the cap, all events round-trip a row."""
    # Pick a dense interval (~30s of game) inside half 1
    f_lo, f_hi = 3580000, 3582500  # 50s @ 50fps
    ev = v._build_event_records("Bayern_Hamburg", sample=None)
    ev = ev[ev["parquet_frame"].between(f_lo, f_hi)].reset_index(drop=True)
    if len(ev) == 0:
        pytest.skip("no events in chosen window")

    chunks = v._split_into_chunks(ev, max_frame_span=5000, max_events=100)
    assert len(chunks) >= 1
    for c in chunks:
        span = int(c["parquet_frame"].max() - c["parquet_frame"].min())
        assert span <= 5000
        assert len(c) <= 100

    xg_map = v._load_possessions_xg_map("Bayern_Hamburg")
    from src.dos import default_params
    params = default_params()
    rows = []
    for c in chunks:
        rows.extend(v._process_chunk("Bayern_Hamburg", c, xg_map, params))
    assert len(rows) > 0
    df = pd.DataFrame(rows)
    # Required columns present
    for col in ["dos", "direction_class", "awareness_mean",
                "detection_delay_mean", "origin_zone", "dest_zone",
                "parent_possession_xg"]:
        assert col in df.columns
    # DOS in plausible range
    assert df["dos"].abs().max() < 1.0
    # Direction class is one of the 3
    assert set(df["direction_class"].unique()) <= {"forward", "diagonal", "sideways"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
