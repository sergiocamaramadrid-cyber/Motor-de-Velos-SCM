"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Uses synthetic audit_features.csv data so no real SPARC data is required.
Validates that all required artefacts are produced and satisfy the
checklist constraints from the problem statement:

    - groupkfold_metrics.csv   exists
    - groupkfold_per_galaxy.csv exists
    - coeffs_by_fold.csv        exists
    - permutation_summary.json  exists and 0 <= p_empirical <= 1
    - master_coeffs.json        exists and d >= 0
    - audit_summary.json        exists
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_scm import (
    _ols,
    _r2,
    _rmse,
    _group_kfold_splits,
    run_groupkfold,
    aggregate_per_galaxy,
    permutation_test,
    master_coefficients,
    plot_residual_vs_hinge_oos,
    run_audit,
    main,
    _FEATURES,
    _TARGET,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_audit_features(n_galaxies: int = 10, n_pts_per: int = 15,
                          seed: int = 0) -> pd.DataFrame:
    """Create a synthetic audit_features.csv DataFrame.

    Parameters
    ----------
    n_galaxies : int
        Number of distinct galaxy groups.
    n_pts_per : int
        Number of radial points per galaxy.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    n = n_galaxies * n_pts_per
    galaxy_labels = np.repeat([f"G{i:03d}" for i in range(n_galaxies)], n_pts_per)
    log_r = rng.uniform(-0.5, 1.5, n)
    log_gbar = rng.uniform(-12.5, -8.5, n)
    log_gobs = 0.5 * log_gbar + rng.normal(0, 0.05, n) - 5.0  # toy RAR
    logM = log_gbar + 2.0 * log_r
    log_j = 0.5 * log_gobs + 1.5 * log_r
    a0_log = np.log10(1.2e-10)
    hinge = np.maximum(0.0, a0_log - log_gbar)
    residual_dex = log_gobs - log_gbar
    return pd.DataFrame({
        "galaxy": galaxy_labels,
        "logM": logM,
        "log_gbar": log_gbar,
        "log_j": log_j,
        "hinge": hinge,
        "residual_dex": residual_dex,
    })


@pytest.fixture()
def audit_csv(tmp_path: Path) -> Path:
    """Write a synthetic audit_features.csv and return its path."""
    df = _make_audit_features(n_galaxies=10, n_pts_per=15)
    csv = tmp_path / "audit_features.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.fixture()
def audit_df() -> pd.DataFrame:
    return _make_audit_features(n_galaxies=10, n_pts_per=15)


# ---------------------------------------------------------------------------
# _ols
# ---------------------------------------------------------------------------

class TestOls:
    def test_perfect_fit(self):
        """OLS should recover exact coefficients for noiseless data."""
        rng = np.random.default_rng(7)
        n = 50
        X = rng.normal(size=(n, 3))
        true_coeffs = np.array([2.0, -1.0, 0.5])
        intercept = 3.0
        y = X @ true_coeffs + intercept
        coeffs, ic = _ols(X, y)
        np.testing.assert_allclose(coeffs, true_coeffs, atol=1e-8)
        np.testing.assert_allclose(ic, intercept, atol=1e-8)

    def test_returns_two_items(self):
        X = np.ones((10, 2))
        y = np.zeros(10)
        result = _ols(X, y)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _r2 / _rmse
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _r2(y, y) == pytest.approx(1.0)

    def test_r2_constant_pred(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.full_like(y, y.mean())
        assert _r2(y, pred) == pytest.approx(0.0)

    def test_rmse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _rmse(y, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _group_kfold_splits
# ---------------------------------------------------------------------------

class TestGroupKFoldSplits:
    def test_all_points_covered(self):
        groups = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        all_test_idx = []
        for _, test_idx in _group_kfold_splits(groups, n_folds=3, seed=0):
            all_test_idx.extend(test_idx.tolist())
        assert sorted(all_test_idx) == list(range(30))

    def test_no_train_test_overlap(self):
        groups = np.array(["G1"] * 5 + ["G2"] * 5 + ["G3"] * 5)
        for train_idx, test_idx in _group_kfold_splits(groups, n_folds=3, seed=0):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_group_not_split(self):
        """No galaxy should appear in both train and test for the same fold."""
        groups = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
        for train_idx, test_idx in _group_kfold_splits(groups, n_folds=4, seed=0):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups & test_groups == set()


# ---------------------------------------------------------------------------
# run_groupkfold
# ---------------------------------------------------------------------------

class TestRunGroupKFold:
    def test_returns_three_items(self, audit_df):
        result = run_groupkfold(audit_df, n_folds=3, seed=0)
        assert len(result) == 3

    def test_fold_metrics_keys(self, audit_df):
        fold_metrics, _, _ = run_groupkfold(audit_df, n_folds=3, seed=0)
        for m in fold_metrics:
            for key in ("fold", "n_train", "n_test", "r2_oos", "rmse_oos"):
                assert key in m, f"key '{key}' missing from fold_metrics"

    def test_per_point_rows_keys(self, audit_df):
        _, per_point_rows, _ = run_groupkfold(audit_df, n_folds=3, seed=0)
        assert len(per_point_rows) > 0
        for key in ("galaxy", "hinge", "residual_dex", "residual_dex_pred",
                    "residual_dex_oos", "fold"):
            assert key in per_point_rows[0], f"key '{key}' missing from per_point_rows"

    def test_all_points_in_oos(self, audit_df):
        """Every point should appear in exactly one fold's test set."""
        _, per_point_rows, _ = run_groupkfold(audit_df, n_folds=3, seed=0)
        assert len(per_point_rows) == len(audit_df)

    def test_coeffs_by_fold_keys(self, audit_df):
        _, _, fold_coeffs = run_groupkfold(audit_df, n_folds=3, seed=0)
        for row in fold_coeffs:
            assert "fold" in row and "intercept" in row
            for feat in _FEATURES:
                assert feat in row

    def test_n_folds_clipped_to_n_galaxies(self):
        """With only 3 galaxies, requesting 10 folds should not crash."""
        df = _make_audit_features(n_galaxies=3, n_pts_per=5)
        fold_metrics, per_point_rows, fold_coeffs = run_groupkfold(
            df, n_folds=10, seed=0
        )
        assert len(fold_metrics) <= 3


# ---------------------------------------------------------------------------
# aggregate_per_galaxy
# ---------------------------------------------------------------------------

class TestAggregatePerGalaxy:
    def test_one_row_per_galaxy(self, audit_df):
        _, per_point_rows, _ = run_groupkfold(audit_df, n_folds=3, seed=0)
        agg = aggregate_per_galaxy(per_point_rows)
        assert len(agg) == audit_df["galaxy"].nunique()

    def test_columns_present(self, audit_df):
        _, per_point_rows, _ = run_groupkfold(audit_df, n_folds=3, seed=0)
        agg = aggregate_per_galaxy(per_point_rows)
        for col in ("galaxy", "n_points", "hinge_mean",
                    "residual_dex_oos_mean", "r2_oos"):
            assert col in agg.columns


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_p_empirical_in_range(self, audit_df):
        result = permutation_test(audit_df, r2_obs=0.5, n_perm=19, seed=0)
        assert 0.0 <= result["p_empirical"] <= 1.0

    def test_result_keys(self, audit_df):
        result = permutation_test(audit_df, r2_obs=0.5, n_perm=19, seed=0)
        for key in ("n_perm", "r2_obs", "p_empirical", "r2_perm_mean", "r2_perm_std"):
            assert key in result

    def test_n_perm_respected(self, audit_df):
        result = permutation_test(audit_df, r2_obs=0.0, n_perm=19, seed=0)
        assert result["n_perm"] == 19

    def test_perfect_model_low_p(self):
        """A perfect model (R²=1) should have very low p_empirical."""
        rng = np.random.default_rng(42)
        n, p = 100, 4
        X = rng.normal(size=(n, p))
        y = X @ np.array([1.0, -0.5, 0.3, 0.8])
        df = pd.DataFrame(X, columns=_FEATURES)
        df["galaxy"] = np.repeat([f"G{i}" for i in range(10)], 10)
        df[_TARGET] = y
        result = permutation_test(df, r2_obs=1.0, n_perm=19, seed=0)
        assert result["p_empirical"] <= 0.15


# ---------------------------------------------------------------------------
# master_coefficients
# ---------------------------------------------------------------------------

class TestMasterCoefficients:
    def test_d_non_negative(self, audit_df):
        result = master_coefficients(audit_df)
        assert result["d"] >= 0.0

    def test_keys_present(self, audit_df):
        result = master_coefficients(audit_df)
        assert "intercept" in result
        assert "d" in result
        for feat in _FEATURES:
            assert feat in result

    def test_hinge_coeff_positive_for_correct_sign(self):
        """With correct hinge sign, hinge coeff in residual regression > 0."""
        rng = np.random.default_rng(99)
        n = 200
        log_gbar = rng.uniform(-12.5, -8.5, n)
        a0_log = np.log10(1.2e-10)
        hinge = np.maximum(0.0, a0_log - log_gbar)
        # Deeper regime → more excess velocity over baryonic
        residual_dex = 0.3 * hinge + rng.normal(0, 0.02, n)
        df = pd.DataFrame({
            "galaxy": np.repeat([f"G{i}" for i in range(20)], 10),
            "logM": rng.normal(size=n),
            "log_gbar": log_gbar,
            "log_j": rng.normal(size=n),
            "hinge": hinge,
            "residual_dex": residual_dex,
        })
        result = master_coefficients(df)
        assert result["hinge"] > 0.0, "hinge coefficient should be positive"
        assert result["d"] > 0.0


# ---------------------------------------------------------------------------
# run_audit (integration tests)
# ---------------------------------------------------------------------------

class TestRunAudit:
    def test_creates_all_required_artefacts(self, audit_csv, tmp_path):
        outdir = tmp_path / "audit_out"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        for fname in (
            "groupkfold_metrics.csv",
            "groupkfold_per_galaxy.csv",
            "oos_per_point.csv",
            "coeffs_by_fold.csv",
            "permutation_summary.json",
            "master_coeffs.json",
            "audit_summary.json",
        ):
            assert (outdir / fname).exists(), f"{fname} was not created"

    def test_permutation_summary_p_in_range(self, audit_csv, tmp_path):
        outdir = tmp_path / "perm_test"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        with open(outdir / "permutation_summary.json", encoding="utf-8") as fh:
            perm = json.load(fh)
        assert 0.0 <= perm["p_empirical"] <= 1.0

    def test_master_coeffs_d_non_negative(self, audit_csv, tmp_path):
        outdir = tmp_path / "master_test"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        with open(outdir / "master_coeffs.json", encoding="utf-8") as fh:
            master = json.load(fh)
        assert master["d"] >= 0.0

    def test_audit_summary_keys(self, audit_csv, tmp_path):
        outdir = tmp_path / "summary_test"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        with open(outdir / "audit_summary.json", encoding="utf-8") as fh:
            summary = json.load(fh)
        for key in ("n_galaxies", "n_points", "n_folds", "p_empirical",
                    "cohen_d_hinge", "status", "artefacts"):
            assert key in summary

    def test_groupkfold_metrics_columns(self, audit_csv, tmp_path):
        outdir = tmp_path / "gkf_test"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        df = pd.read_csv(outdir / "groupkfold_metrics.csv")
        for col in ("fold", "n_train", "n_test", "r2_oos", "rmse_oos"):
            assert col in df.columns

    def test_per_galaxy_csv_columns(self, audit_csv, tmp_path):
        outdir = tmp_path / "per_gal_test"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        df = pd.read_csv(outdir / "groupkfold_per_galaxy.csv")
        assert "galaxy" in df.columns
        assert "r2_oos" in df.columns
        assert "hinge_mean" in df.columns

    def test_missing_csv_raises(self, tmp_path):
        outdir = tmp_path / "no_csv"
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            run_audit(missing, outdir, seed=0, n_folds=3, n_perm=9)

    def test_missing_column_raises(self, tmp_path):
        outdir = tmp_path / "bad_csv"
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["G1"], "logM": [1.0]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            run_audit(bad_csv, outdir, seed=0, n_folds=3, n_perm=9)

    def test_returns_audit_summary_dict(self, audit_csv, tmp_path):
        outdir = tmp_path / "ret_test"
        result = run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        assert isinstance(result, dict)
        assert "status" in result

    def test_status_is_pass_for_valid_data(self, audit_csv, tmp_path):
        outdir = tmp_path / "pass_test"
        result = run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        assert result["status"] == "pass"

    def test_creates_plot(self, audit_csv, tmp_path):
        outdir = tmp_path / "plot_test"
        run_audit(audit_csv, outdir, seed=0, n_folds=3, n_perm=9)
        assert (outdir / "residual_vs_hinge_oos.png").exists()


# ---------------------------------------------------------------------------
# plot_residual_vs_hinge_oos
# ---------------------------------------------------------------------------

class TestPlotResidualVsHingeOos:
    def test_creates_png(self, tmp_path, audit_df):
        _, per_point_rows, _ = run_groupkfold(audit_df, n_folds=3, seed=0)
        out = tmp_path / "plot.png"
        plot_residual_vs_hinge_oos(per_point_rows, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_produces_artefacts(self, audit_csv, tmp_path):
        outdir = tmp_path / "cli_test"
        main([
            "--features-csv", str(audit_csv),
            "--outdir", str(outdir),
            "--seed", "0",
            "--n-folds", "3",
            "--n-perm", "9",
            "--quiet",
        ])
        assert (outdir / "audit_summary.json").exists()
        assert (outdir / "groupkfold_metrics.csv").exists()

    def test_main_missing_csv_exits(self, tmp_path):
        outdir = tmp_path / "fail_test"
        with pytest.raises(SystemExit):
            main([
                "--features-csv", str(tmp_path / "nonexistent.csv"),
                "--outdir", str(outdir),
                "--n-perm", "9",
            ])
