"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Uses synthetic data (no real SPARC download required).
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.audit_scm import (
    rmse,
    aicc,
    gaussian_loglik,
    hinge,
    build_features,
    run_groupkfold,
    permutation_test,
    compare_models,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_df():
    """Synthetic galaxy dataset with required columns."""
    rng = np.random.default_rng(0)
    n = 100
    galaxy_ids = np.repeat([f"G{i:03d}" for i in range(20)], 5)
    M_bar = 10 ** rng.uniform(8, 11, n)
    g_bar = 10 ** rng.uniform(-12, -9, n)
    j_star = 10 ** rng.uniform(1, 4, n)
    v_obs = 100 * (M_bar / 1e10) ** 0.25 * (1 + rng.normal(0, 0.05, n))
    return pd.DataFrame({
        "galaxy_id": galaxy_ids,
        "v_obs": v_obs,
        "M_bar": M_bar,
        "g_bar": g_bar,
        "j_star": j_star,
    })


@pytest.fixture()
def full_df(synthetic_df):
    """DataFrame with all engineered features for SCM full model."""
    df, _ = build_features(synthetic_df, use_hinge=True)
    return df


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestRmse:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 50)
        yp = rng.normal(0, 1, 50)
        assert rmse(y, yp) >= 0.0


class TestAicc:
    def test_increases_with_k(self):
        """More parameters should increase AICc penalty."""
        logL = -100.0
        n = 200
        assert aicc(n, 3, logL) < aicc(n, 5, logL)

    def test_finite_for_reasonable_inputs(self):
        result = aicc(100, 4, -50.0)
        assert np.isfinite(result)


class TestGaussianLoglik:
    def test_better_fit_higher_loglik(self):
        rng = np.random.default_rng(7)
        y = rng.normal(0, 1, 50)
        good_pred = y + rng.normal(0, 0.01, 50)
        bad_pred = y + rng.normal(0, 2.0, 50)
        assert gaussian_loglik(y, good_pred) > gaussian_loglik(y, bad_pred)

    def test_finite_for_reasonable_inputs(self):
        y = np.linspace(1, 10, 50)
        yp = y + 0.1
        assert np.isfinite(gaussian_loglik(y, yp))


class TestHinge:
    def test_zero_above_threshold(self):
        loggbar = np.array([-9.0, -10.0])
        result = hinge(loggbar, logg0=-10.45)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_positive_below_threshold(self):
        loggbar = np.array([-11.0, -12.0])
        result = hinge(loggbar, logg0=-10.45)
        assert np.all(result > 0)

    def test_non_negative(self):
        loggbar = np.linspace(-13, -9, 50)
        result = hinge(loggbar, logg0=-10.45)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_returns_expected_columns_with_hinge(self, synthetic_df):
        df, features = build_features(synthetic_df, use_hinge=True)
        for col in ["logMbar", "loggbar", "logj", "logv_obs", "hinge"]:
            assert col in df.columns
        assert "hinge" in features

    def test_returns_expected_columns_without_hinge(self, synthetic_df):
        df, features = build_features(synthetic_df, use_hinge=False)
        assert "hinge" not in df.columns
        assert "hinge" not in features

    def test_does_not_modify_original(self, synthetic_df):
        original_cols = set(synthetic_df.columns)
        build_features(synthetic_df, use_hinge=True)
        assert set(synthetic_df.columns) == original_cols

    def test_feature_list_length(self, synthetic_df):
        _, features_full = build_features(synthetic_df, use_hinge=True)
        _, features_no_hinge = build_features(synthetic_df, use_hinge=False)
        assert len(features_full) == len(features_no_hinge) + 1


# ---------------------------------------------------------------------------
# run_groupkfold
# ---------------------------------------------------------------------------

class TestRunGroupkfold:
    def test_returns_dataframe_and_coeffs(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, coeffs = run_groupkfold(full_df, features, k=5)
        assert isinstance(df_oof, pd.DataFrame)
        assert isinstance(coeffs, pd.DataFrame)

    def test_oof_column_added(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, _ = run_groupkfold(full_df, features, k=5)
        assert "logv_pred_oof" in df_oof.columns
        assert "resid_log_oof" in df_oof.columns

    def test_coeffs_has_fold_column(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        _, coeffs = run_groupkfold(full_df, features, k=5)
        assert "fold" in coeffs.columns

    def test_coeffs_row_count(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        k = 5
        _, coeffs = run_groupkfold(full_df, features, k=k)
        assert len(coeffs) == k

    def test_no_nan_in_predictions(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, _ = run_groupkfold(full_df, features, k=5)
        assert df_oof["logv_pred_oof"].notna().all()

    def test_residuals_match(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, _ = run_groupkfold(full_df, features, k=5)
        expected = df_oof["logv_obs"] - df_oof["logv_pred_oof"]
        np.testing.assert_allclose(df_oof["resid_log_oof"].values, expected.values)


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_returns_real_rmse_and_list(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, _ = run_groupkfold(full_df, features, k=5)
        real_rmse_val, perm_rmses = permutation_test(
            df_oof, features, k=5, n_perm=5, seed=42
        )
        assert isinstance(real_rmse_val, float)
        assert len(perm_rmses) == 5

    def test_perm_rmses_non_negative(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, _ = run_groupkfold(full_df, features, k=5)
        _, perm_rmses = permutation_test(
            df_oof, features, k=5, n_perm=5, seed=42
        )
        assert all(v >= 0.0 for v in perm_rmses)

    def test_reproducible_with_seed(self, full_df):
        features = ["logMbar", "loggbar", "logj", "hinge"]
        df_oof, _ = run_groupkfold(full_df, features, k=5)
        _, perm1 = permutation_test(df_oof, features, k=5, n_perm=5, seed=99)
        _, perm2 = permutation_test(df_oof, features, k=5, n_perm=5, seed=99)
        assert perm1 == perm2


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_dataframe(self, full_df):
        result = compare_models(full_df)
        assert isinstance(result, pd.DataFrame)

    def test_three_models_present(self, full_df):
        result = compare_models(full_df)
        assert set(result["model"]) == {"BTFR", "SCM_no_hinge", "SCM_full"}

    def test_expected_columns(self, full_df):
        result = compare_models(full_df)
        for col in ["model", "k", "n", "logL", "AICc"]:
            assert col in result.columns

    def test_aicc_finite(self, full_df):
        result = compare_models(full_df)
        assert result["AICc"].notna().all()

    def test_scm_full_has_more_params(self, full_df):
        result = compare_models(full_df).set_index("model")
        assert result.loc["SCM_full", "k"] > result.loc["BTFR", "k"]


# ---------------------------------------------------------------------------
# End-to-end: CLI via main()
# ---------------------------------------------------------------------------

class TestAuditScmCli:
    def test_creates_all_output_files(self, synthetic_df, tmp_path):
        """main() must write all four expected output artefacts."""
        import sys
        from scripts.audit_scm import main

        csv_path = tmp_path / "sparc_raw.csv"
        synthetic_df.to_csv(csv_path, index=False)
        outdir = tmp_path / "audit"

        sys.argv = [
            "audit_scm.py",
            "--input", str(csv_path),
            "--outdir", str(outdir),
            "--ref", "abc1234",
            "--kfold", "5",
            "--permutations", "5",
            "--seed", "42",
        ]
        main()

        assert (outdir / "groupkfold_predictions.csv").exists()
        assert (outdir / "coeffs_by_fold.csv").exists()
        assert (outdir / "permutation_summary.json").exists()
        assert (outdir / "model_comparison_aicc.csv").exists()

    def test_permutation_summary_keys(self, synthetic_df, tmp_path):
        """permutation_summary.json must contain expected keys."""
        import sys
        from scripts.audit_scm import main

        csv_path = tmp_path / "sparc_raw.csv"
        synthetic_df.to_csv(csv_path, index=False)
        outdir = tmp_path / "audit2"

        sys.argv = [
            "audit_scm.py",
            "--input", str(csv_path),
            "--outdir", str(outdir),
            "--ref", "testref",
            "--kfold", "5",
            "--permutations", "3",
            "--seed", "0",
        ]
        main()

        with open(outdir / "permutation_summary.json") as f:
            summary = json.load(f)

        assert "real_rmse_log" in summary
        assert "perm_rmse_mean" in summary
        assert "perm_rmse_std" in summary
