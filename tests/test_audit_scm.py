"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Validates the audit pipeline with a small synthetic dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from audit_scm import (
    _design_matrix,
    _ols_fit,
    kfold_cv,
    permutation_test,
    main,
    FEATURES,
    TARGET,
    GALAXY_COL,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def smoke_csv(tmp_path):
    """Tiny synthetic global CSV matching the problem-statement structure."""
    rng = np.random.default_rng(0)
    gals = [f"G{i}" for i in range(6)]
    rows = []
    for g in gals:
        logM = rng.normal(10, 0.4)
        for _ in range(3):
            log_gbar = rng.normal(-10.8, 0.3)
            log_j = rng.normal(1.5, 0.2)
            y = 0.32 * logM + (-0.05) * log_gbar + 0.04 * log_j + rng.normal(0, 0.03)
            rows.append(
                {"galaxy_id": g, "logM": logM, "log_gbar": log_gbar, "log_j": log_j, "v_obs": y}
            )
    df = pd.DataFrame(rows)
    p = tmp_path / "_smoke_sparc_global.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# _design_matrix
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    def test_shape(self):
        df = pd.DataFrame({"logM": [1.0, 2.0], "log_gbar": [-11.0, -10.5], "log_j": [1.5, 1.6]})
        X = _design_matrix(df)
        assert X.shape == (2, 4)

    def test_intercept_column(self):
        df = pd.DataFrame({"logM": [1.0], "log_gbar": [-11.0], "log_j": [1.5]})
        X = _design_matrix(df)
        np.testing.assert_array_equal(X[:, 0], [1.0])


# ---------------------------------------------------------------------------
# _ols_fit
# ---------------------------------------------------------------------------


class TestOlsFit:
    def test_perfect_fit(self):
        X = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
        y = 2.0 * np.arange(5, dtype=float) + 1.0
        coef, y_pred, r2, rmse = _ols_fit(X, y)
        assert r2 == pytest.approx(1.0, abs=1e-9)
        assert rmse == pytest.approx(0.0, abs=1e-9)

    def test_coef_shape(self):
        X = np.column_stack([np.ones(10), np.random.default_rng(0).normal(size=(10, 3))])
        y = np.random.default_rng(1).normal(size=10)
        coef, _, _, _ = _ols_fit(X, y)
        assert coef.shape == (4,)

    def test_r2_bounded(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
        y = rng.normal(size=20)
        _, _, r2, _ = _ols_fit(X, y)
        assert r2 <= 1.0


# ---------------------------------------------------------------------------
# kfold_cv
# ---------------------------------------------------------------------------


class TestKfoldCv:
    def test_returns_k_rows(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        cv = kfold_cv(df, k=3, seed=0)
        assert len(cv) == 3

    def test_columns(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        cv = kfold_cv(df, k=3, seed=0)
        for col in ("fold", "n_val", "rmse", "r2"):
            assert col in cv.columns

    def test_n_val_sums_to_n(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        cv = kfold_cv(df, k=3, seed=0)
        assert cv["n_val"].sum() == len(df)

    def test_rmse_non_negative(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        cv = kfold_cv(df, k=3, seed=0)
        assert (cv["rmse"] >= 0).all()


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_returns_expected_keys(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        result = permutation_test(df, n_perms=10, seed=0)
        for key in ("r2_true", "r2_perm_mean", "r2_perm_std", "p_value", "n_permutations"):
            assert key in result

    def test_n_permutations_recorded(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        result = permutation_test(df, n_perms=15, seed=0)
        assert result["n_permutations"] == 15

    def test_p_value_range(self, smoke_csv):
        df = pd.read_csv(smoke_csv)
        result = permutation_test(df, n_perms=20, seed=0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_true_r2_high_on_signal(self, smoke_csv):
        """Synthetic signal should give high R² that beats most permutations."""
        df = pd.read_csv(smoke_csv)
        result = permutation_test(df, n_perms=50, seed=0)
        assert result["r2_true"] > result["r2_perm_mean"]


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_output_files_exist(self, smoke_csv, tmp_path):
        out = tmp_path / "audit_out"
        main([
            "--input", str(smoke_csv),
            "--outdir", str(out),
            "--seed", "123",
            "--kfold", "3",
            "--permutations", "10",
        ])
        assert (out / "audit_coefs.csv").exists()
        assert (out / "audit_kfold.csv").exists()
        assert (out / "audit_permutations.csv").exists()
        assert (out / "audit.log").exists()

    def test_coefs_csv_columns(self, smoke_csv, tmp_path):
        out = tmp_path / "audit_cols"
        main([
            "--input", str(smoke_csv),
            "--outdir", str(out),
            "--seed", "0",
            "--kfold", "3",
            "--permutations", "5",
        ])
        df = pd.read_csv(out / "audit_coefs.csv")
        assert "term" in df.columns
        assert "coef" in df.columns
        terms = df["term"].tolist()
        assert "intercept" in terms
        for feat in FEATURES:
            assert feat in terms

    def test_kfold_csv_fold_count(self, smoke_csv, tmp_path):
        out = tmp_path / "audit_folds"
        main([
            "--input", str(smoke_csv),
            "--outdir", str(out),
            "--seed", "0",
            "--kfold", "3",
            "--permutations", "5",
        ])
        df = pd.read_csv(out / "audit_kfold.csv")
        assert len(df) == 3

    def test_missing_input_raises(self, tmp_path):
        out = tmp_path / "audit_err"
        with pytest.raises(FileNotFoundError):
            main([
                "--input", str(tmp_path / "no_such_file.csv"),
                "--outdir", str(out),
            ])

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(bad_csv, index=False)
        out = tmp_path / "audit_bad"
        with pytest.raises(ValueError):
            main([
                "--input", str(bad_csv),
                "--outdir", str(out),
            ])
