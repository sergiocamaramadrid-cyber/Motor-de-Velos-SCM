"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Validates:
  (i)   All 6 standard artefacts are written.
  (ii)  Physical constraint d ≥ 0 is satisfied in every fold.
  (iii) 0 ≤ p_empirical ≤ 1.
  (iv)  groupkfold_metrics.csv column contract.
  (v)   gal_results.csv column contract.
  (vi)  coeffs_by_fold.csv column contract.
  (vii) permutation_summary.json required keys.

Uses a small 20-galaxy synthetic dataset so the test completes quickly
(n_splits=3, n_perm=10).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_scm import main, run_audit, _fit_full

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_GALAXIES_TEST = 20
N_SPLITS_TEST = 3
N_PERM_TEST = 10


@pytest.fixture(scope="module")
def audit_csv(tmp_path_factory) -> Path:
    """Build a small synthetic sparc_global.csv."""
    root = tmp_path_factory.mktemp("audit_data")
    rng = np.random.default_rng(7)
    n = N_GALAXIES_TEST
    df = pd.DataFrame({
        "galaxy_id": [f"G{i:03d}" for i in range(n)],
        "logM": rng.uniform(9.0, 12.0, n),
        "log_gbar": rng.uniform(-12.0, -9.0, n),
        "log_j": rng.uniform(2.0, 5.0, n),
        "v_obs": rng.uniform(1.8, 2.5, n),
    })
    csv_path = root / "sparc_global.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def audit_results(audit_csv, tmp_path_factory):
    """Run main() once and return (out_dir, results_dict) for all tests."""
    out_dir = tmp_path_factory.mktemp("audit_out")
    results = main([
        "--csv", str(audit_csv),
        "--out", str(out_dir),
        "--n-splits", str(N_SPLITS_TEST),
        "--n-perm", str(N_PERM_TEST),
        "--seed", "0",
    ])
    return out_dir, results


# ---------------------------------------------------------------------------
# Artefact existence (6 standard artefacts)
# ---------------------------------------------------------------------------

EXPECTED_ARTEFACTS = [
    "groupkfold_metrics.csv",
    "gal_results.csv",
    "permutation_runs.csv",
    "coeffs_by_fold.csv",
    "permutation_summary.json",
    "audit_summary.txt",
]


@pytest.mark.parametrize("fname", EXPECTED_ARTEFACTS)
def test_artefact_exists(audit_results, fname):
    """All 6 standard artefacts must be created by main()."""
    out_dir, _ = audit_results
    assert (out_dir / fname).exists(), f"{fname} was not written"


# ---------------------------------------------------------------------------
# Physical constraint: d ≥ 0 in every fold
# ---------------------------------------------------------------------------

def test_d_non_negative_in_all_folds(audit_results):
    """coeffs_by_fold.csv must have d ≥ 0 in every row."""
    out_dir, _ = audit_results
    coeff_df = pd.read_csv(out_dir / "coeffs_by_fold.csv")
    assert "d" in coeff_df.columns, "coeffs_by_fold.csv missing column 'd'"
    assert (coeff_df["d"] >= 0).all(), (
        f"Physical constraint d ≥ 0 violated: {coeff_df['d'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Permutation test: 0 ≤ p_empirical ≤ 1
# ---------------------------------------------------------------------------

def test_p_empirical_in_unit_interval(audit_results):
    """permutation_summary.json must have 0 ≤ p_empirical ≤ 1."""
    out_dir, results = audit_results
    with open(out_dir / "permutation_summary.json") as f:
        summary = json.load(f)
    p = summary["p_empirical"]
    assert 0.0 <= p <= 1.0, f"p_empirical={p} outside [0, 1]"


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

def test_groupkfold_metrics_columns(audit_results):
    """groupkfold_metrics.csv must have the required columns."""
    out_dir, _ = audit_results
    df = pd.read_csv(out_dir / "groupkfold_metrics.csv")
    required = {"fold", "n_train", "n_test", "rmse_btfr", "rmse_no_hinge", "rmse_full"}
    missing = required - set(df.columns)
    assert not missing, f"groupkfold_metrics.csv missing columns: {missing}"


def test_gal_results_columns(audit_results):
    """gal_results.csv must have the required columns."""
    out_dir, _ = audit_results
    df = pd.read_csv(out_dir / "gal_results.csv")
    required = {"galaxy_id", "fold", "rmse_btfr", "rmse_no_hinge", "rmse_full"}
    missing = required - set(df.columns)
    assert not missing, f"gal_results.csv missing columns: {missing}"


def test_coeffs_by_fold_columns(audit_results):
    """coeffs_by_fold.csv must have the required columns."""
    out_dir, _ = audit_results
    df = pd.read_csv(out_dir / "coeffs_by_fold.csv")
    required = {"fold", "beta", "C", "a", "b", "d", "logg0"}
    missing = required - set(df.columns)
    assert not missing, f"coeffs_by_fold.csv missing columns: {missing}"


def test_permutation_runs_columns(audit_results):
    """permutation_runs.csv must have 'perm_rmse' and 'run' columns."""
    out_dir, _ = audit_results
    df = pd.read_csv(out_dir / "permutation_runs.csv")
    assert "perm_rmse" in df.columns, "permutation_runs.csv missing 'perm_rmse'"
    assert len(df) == N_PERM_TEST, (
        f"Expected {N_PERM_TEST} permutation rows, got {len(df)}"
    )


def test_permutation_summary_keys(audit_results):
    """permutation_summary.json must contain the required keys."""
    out_dir, _ = audit_results
    with open(out_dir / "permutation_summary.json") as f:
        summary = json.load(f)
    required_keys = {
        "real_mean_rmse", "perm_mean_rmse", "p_empirical",
        "n_perm", "n_galaxies",
    }
    missing = required_keys - set(summary.keys())
    assert not missing, f"permutation_summary.json missing keys: {missing}"


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------

def test_fold_count(audit_results):
    """groupkfold_metrics.csv must have exactly n_splits rows."""
    out_dir, _ = audit_results
    df = pd.read_csv(out_dir / "groupkfold_metrics.csv")
    assert len(df) == N_SPLITS_TEST, (
        f"Expected {N_SPLITS_TEST} fold rows, got {len(df)}"
    )


def test_gal_results_row_count(audit_results):
    """gal_results.csv must have N_GALAXIES_TEST rows (one per galaxy)."""
    out_dir, _ = audit_results
    df = pd.read_csv(out_dir / "gal_results.csv")
    assert len(df) == N_GALAXIES_TEST, (
        f"Expected {N_GALAXIES_TEST} galaxy rows, got {len(df)}"
    )


# ---------------------------------------------------------------------------
# run_audit unit-level tests
# ---------------------------------------------------------------------------

def test_run_audit_returns_required_keys(audit_csv):
    """run_audit() result dict must contain the expected keys."""
    df = pd.read_csv(audit_csv)
    results = run_audit(df, n_splits=N_SPLITS_TEST, n_perm=5, seed=1)
    required = {
        "fold_records", "gal_records", "perm_rmses", "coeff_records",
        "wilcoxon_stat", "wilcoxon_p", "real_mean_rmse", "p_empirical",
    }
    missing = required - set(results.keys())
    assert not missing, f"run_audit result missing keys: {missing}"


def test_fit_full_d_constraint():
    """_fit_full must return d ≥ 0 even with adversarial data."""
    rng = np.random.default_rng(42)
    n = 15
    logM = rng.uniform(9, 12, n)
    log_gbar = rng.uniform(-12, -9, n)
    y = rng.uniform(1.5, 2.5, n)
    params = _fit_full(logM, log_gbar, y)
    d = params[4]
    assert d >= 0.0, f"d={d} < 0 after _fit_full — physical constraint violated"


# ---------------------------------------------------------------------------
# Auto-detection of --csv input
# ---------------------------------------------------------------------------

def test_autodetect_uses_default_csv(audit_csv, tmp_path, monkeypatch):
    """main() must auto-detect the default CSV and print an informational message."""
    out_dir = tmp_path / "audit_auto"
    # Point the CWD to a tmp dir that contains the default relative path
    data_dir = tmp_path / "results" / "audit"
    data_dir.mkdir(parents=True)
    import shutil
    shutil.copy(audit_csv, data_dir / "sparc_global.csv")
    monkeypatch.chdir(tmp_path)

    results = main([
        # No --csv argument
        "--out", str(out_dir),
        "--n-splits", str(N_SPLITS_TEST),
        "--n-perm", str(N_PERM_TEST),
        "--seed", "0",
    ])
    assert results is not None
    assert (out_dir / "audit_summary.txt").exists()


def test_autodetect_raises_when_no_default(tmp_path, monkeypatch):
    """main() must raise ValueError when --csv is omitted and default doesn't exist."""
    monkeypatch.chdir(tmp_path)  # empty dir — no sparc_global.csv here
    with pytest.raises(ValueError, match="results/audit/sparc_global.csv"):
        main(["--out", str(tmp_path / "out")])
