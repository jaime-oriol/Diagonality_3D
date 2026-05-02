"""Unit tests for the D-Def validation pipeline (test/enrich_with_ddef.py
+ test/compute_stats_ddef.py).

Covers:
  - GK jersey lookup per match (real cache)
  - Bi-axial balance computation (per-event and per-class)
  - Direction-class breakdown shapes / values
  - DOS-↔-DDef correlation function
  - Logistic helper handles small-N gracefully
  - End-to-end with a synthetic D-Def-enriched dataframe
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "test")

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Compute the import path
import importlib.util as _iu
spec_e = _iu.spec_from_file_location(
    "enrich_with_ddef", Path("test/enrich_with_ddef.py"))
enr = _iu.module_from_spec(spec_e); spec_e.loader.exec_module(enr)
spec_s = _iu.spec_from_file_location(
    "compute_stats_ddef", Path("test/compute_stats_ddef.py"))
sts = _iu.module_from_spec(spec_s); spec_s.loader.exec_module(sts)


CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
HAS_CACHE = (CACHE_DIR / "Bayern_Hamburg" / "skeleton.parquet").exists()


# ── GK jersey resolution ───────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CACHE, reason="cache not present")
def test_gk_jersey_resolves_for_all_5_matches():
    from src.loader import MATCHES
    for m in MATCHES:
        gks = enr._gk_jersey_per_team(m)
        assert 0 in gks and 1 in gks, f"{m}: missing GK side"
        assert isinstance(gks[0], int) and gks[0] > 0
        assert isinstance(gks[1], int) and gks[1] > 0


# ── Bi-axial balance ──────────────────────────────────────────────────

def test_biaxial_balance_perfect():
    df = pd.DataFrame({"pc1_3s": [2.0], "pc2_3s": [2.0]})
    b = sts._bi_axial_balance(df, "3s")
    assert float(b.iloc[0]) == pytest.approx(1.0)


def test_biaxial_balance_uni_axial():
    df = pd.DataFrame({"pc1_3s": [5.0], "pc2_3s": [0.5]})
    b = sts._bi_axial_balance(df, "3s")
    assert float(b.iloc[0]) == pytest.approx(0.1)


def test_biaxial_balance_zero_in_max_returns_nan():
    df = pd.DataFrame({"pc1_3s": [0.0], "pc2_3s": [0.0]})
    b = sts._bi_axial_balance(df, "3s")
    assert np.isnan(float(b.iloc[0]))


# ── Direction-class table ─────────────────────────────────────────────

def _synth_ddef_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    classes = rng.choice(["forward", "diagonal", "sideways"], size=n)
    df = pd.DataFrame({
        "direction_class": classes,
        "pc1_3s": rng.normal(0, 1.5, n),
        "pc2_3s": rng.normal(0, 1.0, n),
        "pc3_3s": rng.normal(0, 0.7, n),
        "ddef_3s": np.abs(rng.normal(0, 5, n)),
        "local_delta_area": rng.normal(0, 10, n),
        "local_delta_spread": rng.normal(0, 0.5, n),
        "dos": rng.normal(0.01, 0.01, n),
        "distance": rng.uniform(5, 30, n),
        "pressure_player": rng.uniform(0, 1, n),
        "n_def_lane": rng.integers(0, 4, n).astype(float),
        "n_def_goal": rng.integers(2, 11, n).astype(float),
        "event_type": "pass",
    })
    df["biaxial"] = sts._bi_axial_balance(df, "3s")
    return df


def test_direction_table_has_expected_columns_and_classes():
    df = _synth_ddef_df()
    tbl = sts._direction_table(df, "3s")
    assert set(tbl.index) <= {"forward", "diagonal", "sideways"}
    expected = {"n", "abs_pc1", "abs_pc2", "abs_pc3", "ddef",
                "per_class_balance", "per_event_balance_mean",
                "local_delta_area", "local_delta_spread"}
    assert expected <= set(tbl.columns)
    # per_class_balance bounded in [0, 1]
    assert (tbl["per_class_balance"] >= 0).all()
    assert (tbl["per_class_balance"] <= 1).all()


def test_per_event_type_breakdown_shape():
    df = _synth_ddef_df()
    pet = sts._per_event_type(df, "3s")
    assert "biaxial_mean" in pet.columns
    assert int(pet["n"].sum()) == 300


# ── DOS ↔ D-Def correlations ──────────────────────────────────────────

def test_dos_correlations_returns_targets():
    df = _synth_ddef_df()
    corr = sts._dos_correlations(df, "3s")
    targets = set(corr["target"].tolist())
    assert "pc1_3s" in targets and "pc2_3s" in targets
    assert "ddef_3s" in targets and "biaxial" in targets
    # rho in [-1, 1] for valid rows
    valid = corr.dropna(subset=["rho"])
    assert (valid["rho"].abs() <= 1.0).all()


# ── H2 MWU ────────────────────────────────────────────────────────────

def test_h2_mwu_table_shape():
    df = _synth_ddef_df(seed=123)
    out = sts._h2_mwu_diagonal_vs_orthogonal(df, "3s")
    assert "test" in out.columns
    assert "rbc" in out.columns and "p" in out.columns
    assert len(out) >= 5


# ── Logistic ──────────────────────────────────────────────────────────

def test_logistic_biaxial_runs_or_skips_gracefully():
    df = _synth_ddef_df(n=400, seed=2)
    res = sts._logistic_biaxial(df, "3s")
    assert "fit" in res
    if res["fit"]:
        assert "dos" in res["coefs"]
        assert "pseudo_r2" in res


def test_logistic_biaxial_skips_for_small_n():
    df = _synth_ddef_df(n=20, seed=3)
    res = sts._logistic_biaxial(df, "3s")
    assert res["fit"] is False


# ── Integration with real ddef-enriched CSV ───────────────────────────

@pytest.mark.skipif(
    not Path("test/dos_validation_raw_xt_ddef.csv").exists(),
    reason="ddef-enriched CSV not yet generated")
def test_integration_real_ddef_csv():
    df = pd.read_csv("test/dos_validation_raw_xt_ddef.csv")
    assert "pc1_3s" in df.columns and "pc2_3s" in df.columns
    assert df["pc1_3s"].notna().sum() > 1000
    df["biaxial"] = sts._bi_axial_balance(df, "3s")
    tbl = sts._direction_table(df, "3s")
    assert set(tbl.index) >= {"forward", "diagonal", "sideways"}
    # The full dataset must have a SUM of n equal to all valid events
    assert int(tbl["n"].sum()) > 4000
