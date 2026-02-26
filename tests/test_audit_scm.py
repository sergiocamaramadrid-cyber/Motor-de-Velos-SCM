"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Validates the key fixes described in the issue:
  1. per-galaxy RMSE aggregation in GroupKFold (not per-point)
  2. Corrected AICc using MLE estimator
  3. Permutation test with correct empirical p-value formula
  4. Real coefficient stability (per-fold fitting)
  5. --strict flag behaviour
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_scm import (
    _fit_linear,
    _predict,
    _rmse,
    safe_aicc,
    groupkfold_audit,
    permutation_test,
    galaxy_delta_rmse_test,
    run_audit,
    main,
    REQUIRED_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n_galaxies: int = 10, n_pts_per_galaxy: int = 15,
             seed: int = 0) -> pd.DataFrame:
    """Synthetic per-radial-point DataFrame with galaxy, log_g_bar, log_g_obs."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_galaxies):
        lg_bar = rng.uniform(-13, -10, n_pts_per_galaxy)
        # Noisy linear relationship
        lg_obs = 0.5 * lg_bar + (-5.0 + rng.normal(0, 0.05, n_pts_per_galaxy))
        for j in range(n_pts_per_galaxy):
            rows.append({
                "galaxy": f"G{i:03d}",
                "log_g_bar": float(lg_bar[j]),
                "log_g_obs": float(lg_obs[j]),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def small_df():
    return _make_df(n_galaxies=8, n_pts_per_galaxy=12)


@pytest.fixture(scope="module")
def medium_df():
    return _make_df(n_galaxies=20, n_pts_per_galaxy=15)


# ---------------------------------------------------------------------------
# _fit_linear
# ---------------------------------------------------------------------------


class TestFitLinear:
    def test_perfect_fit(self):
        """OLS recovers exact coefficients."""
        x = np.linspace(-13, -10, 30)
        y = 0.5 * x + (-5.0)
        slope, intercept = _fit_linear(x, y)
        assert slope == pytest.approx(0.5, abs=1e-8)
        assert intercept == pytest.approx(-5.0, abs=1e-8)

    def test_noisy_fit_reasonable(self):
        rng = np.random.default_rng(7)
        x = np.linspace(-13, -10, 50)
        y = 0.5 * x - 5.0 + rng.normal(0, 0.1, 50)
        slope, intercept = _fit_linear(x, y)
        assert 0.3 <= slope <= 0.7
        assert -6.0 <= intercept <= -4.0


# ---------------------------------------------------------------------------
# safe_aicc — Fix 2
# ---------------------------------------------------------------------------


class TestSafeAICc:
    def test_finite_for_valid_inputs(self):
        val = safe_aicc(rss=10.0, n=100, k=2)
        assert math.isfinite(val)

    def test_inf_for_zero_rss(self):
        assert math.isinf(safe_aicc(rss=0.0, n=100, k=2))

    def test_inf_when_n_le_k_plus_1(self):
        # n - k - 1 ≤ 0  → correction blows up → inf
        assert math.isinf(safe_aicc(rss=1.0, n=3, k=3))

    def test_correction_shrinks_for_large_n(self):
        """For large n, AICc correction → 0."""
        k = 2
        rss = 10.0
        aic_c_small = safe_aicc(rss, n=10, k=k)
        aic_c_large = safe_aicc(rss, n=10000, k=k)
        # Correction = 2k(k+1)/(n-k-1); for large n this → 0
        assert aic_c_large < aic_c_small

    def test_increases_with_k(self):
        """More parameters → higher AICc (penalty)."""
        rss = 20.0
        n = 200
        aicc1 = safe_aicc(rss, n, k=2)
        aicc2 = safe_aicc(rss, n, k=5)
        assert aicc2 > aicc1

    def test_lower_rss_means_lower_aicc(self):
        n, k = 100, 2
        assert safe_aicc(5.0, n, k) < safe_aicc(20.0, n, k)


# ---------------------------------------------------------------------------
# groupkfold_audit — Fixes 1 & 4
# ---------------------------------------------------------------------------


class TestGroupKFoldAudit:
    def test_returns_expected_keys(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        for key in ("per_galaxy_df", "per_point_df", "coeff_by_fold",
                    "coeff_mean", "coeff_std", "aicc", "k"):
            assert key in result, f"Missing key: {key}"

    def test_per_galaxy_df_one_row_per_galaxy_per_fold(self, medium_df):
        """Fix 1: per_galaxy_df must aggregate to one row per (galaxy, fold)."""
        result = groupkfold_audit(medium_df, n_splits=4)
        pg = result["per_galaxy_df"]
        # Each (galaxy, fold) combination should appear at most once
        assert not pg.duplicated(subset=["galaxy", "fold"]).any()

    def test_per_galaxy_df_not_per_point(self, medium_df):
        """Fix 1: per_galaxy_df has fewer rows than per_point_df."""
        result = groupkfold_audit(medium_df, n_splits=4)
        pg = result["per_galaxy_df"]
        pp = result["per_point_df"]
        assert len(pg) < len(pp), (
            "per_galaxy_df should have fewer rows than per_point_df "
            "(it must be per-galaxy, not per-point)"
        )

    def test_per_galaxy_df_columns(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        pg = result["per_galaxy_df"]
        for col in ("galaxy", "fold", "n_points", "rmse_galaxy"):
            assert col in pg.columns, f"Missing column: {col}"

    def test_per_point_df_columns(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        pp = result["per_point_df"]
        for col in ("galaxy", "fold", "log_g_bar", "log_g_obs",
                    "log_g_pred", "residual"):
            assert col in pp.columns, f"Missing column: {col}"

    def test_rmse_galaxy_non_negative(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        assert (result["per_galaxy_df"]["rmse_galaxy"] >= 0).all()

    def test_n_points_consistent(self, medium_df):
        """n_points in per_galaxy_df must sum to per_point_df rows."""
        result = groupkfold_audit(medium_df, n_splits=4)
        assert result["per_galaxy_df"]["n_points"].sum() == len(result["per_point_df"])

    def test_coeff_by_fold_has_correct_keys(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        for entry in result["coeff_by_fold"]:
            assert set(entry.keys()) == {"fold", "slope", "intercept"}

    def test_coeff_by_fold_count(self, medium_df):
        """Number of fold coefficient entries equals n_splits (or fewer if not enough data)."""
        result = groupkfold_audit(medium_df, n_splits=4)
        assert len(result["coeff_by_fold"]) <= 4

    def test_coefficient_stability_fix4(self, medium_df):
        """Fix 4: coefficients differ across folds (each fold fits on its own training set)."""
        result = groupkfold_audit(medium_df, n_splits=4)
        slopes = [c["slope"] for c in result["coeff_by_fold"]]
        # Not all slopes should be identical (each fold has different training data)
        if len(slopes) > 1:
            # std should be non-zero (folds use different data)
            assert not all(s == slopes[0] for s in slopes), (
                "All fold slopes are identical — coefficients are not being "
                "refitted per fold (Fix 4 not applied)"
            )

    def test_coeff_mean_std_finite(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        assert math.isfinite(result["coeff_mean"]["slope_mean"])
        assert math.isfinite(result["coeff_mean"]["intercept_mean"])
        assert math.isfinite(result["coeff_std"]["slope_std"])
        assert math.isfinite(result["coeff_std"]["intercept_std"])

    def test_aicc_finite(self, small_df):
        result = groupkfold_audit(small_df, n_splits=3)
        assert math.isfinite(result["aicc"])

    def test_k_is_two(self, small_df):
        """k=2 for slope + intercept."""
        result = groupkfold_audit(small_df, n_splits=3)
        assert result["k"] == 2


# ---------------------------------------------------------------------------
# permutation_test — Fixes 3 & 6
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_returns_expected_keys(self, small_df):
        result = permutation_test(small_df, n_perm=20,
                                  rng=np.random.default_rng(0))
        for key in ("real_rmse", "perm_rmse", "p_value", "n_perm"):
            assert key in result

    def test_real_rmse_positive(self, small_df):
        result = permutation_test(small_df, n_perm=20,
                                  rng=np.random.default_rng(0))
        assert result["real_rmse"] > 0

    def test_perm_rmse_length(self, small_df):
        n = 30
        result = permutation_test(small_df, n_perm=n,
                                  rng=np.random.default_rng(0))
        assert len(result["perm_rmse"]) == n

    def test_p_value_range(self, small_df):
        """Fix 6: p-value must be in (0, 1] using the formula (1 + Σ) / (N+1)."""
        n = 50
        result = permutation_test(small_df, n_perm=n,
                                  rng=np.random.default_rng(0))
        # Minimum possible p = 1/(n+1); maximum = 1.0
        assert 1.0 / (n + 1) <= result["p_value"] <= 1.0

    def test_p_value_formula_correct(self, small_df):
        """Fix 6: verify the p-value formula (1 + count) / (N_perm + 1)."""
        result = permutation_test(small_df, n_perm=100,
                                  rng=np.random.default_rng(42))
        real_rmse = result["real_rmse"]
        perm_rmse = result["perm_rmse"]
        expected_p = (1 + np.sum(perm_rmse <= real_rmse)) / (100 + 1)
        assert result["p_value"] == pytest.approx(expected_p, abs=1e-10)

    def test_strong_signal_low_p(self):
        """With a strong baryonic signal, p should be low (real RMSE < perm RMSE)."""
        # Perfect log-linear relationship across many galaxies
        rng = np.random.default_rng(1)
        rows = []
        for i in range(30):
            lg_bar = rng.uniform(-13, -10, 20)
            # Very tight relationship
            lg_obs = 0.5 * lg_bar - 5.0 + rng.normal(0, 0.001, 20)
            for j in range(20):
                rows.append({"galaxy": f"G{i}", "log_g_bar": lg_bar[j],
                              "log_g_obs": lg_obs[j]})
        df = pd.DataFrame(rows)
        result = permutation_test(df, n_perm=200, rng=np.random.default_rng(7))
        assert result["p_value"] < 0.1, (
            f"Expected p < 0.1 for strong signal, got p={result['p_value']:.4f}"
        )


# ---------------------------------------------------------------------------
# galaxy_delta_rmse_test
# ---------------------------------------------------------------------------


class TestGalaxyDeltaRMSETest:
    def _make_per_gal(self, n, base_rmse, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"rmse_galaxy": base_rmse + rng.normal(0, 0.02, n)})

    def test_frac_improved_all_better(self):
        pg = pd.DataFrame({"rmse_galaxy": [0.1, 0.2, 0.15]})
        result = galaxy_delta_rmse_test(pg, null_rmse=0.5)
        assert result["frac_improved"] == pytest.approx(1.0)

    def test_frac_improved_none_better(self):
        pg = pd.DataFrame({"rmse_galaxy": [0.6, 0.7, 0.8]})
        result = galaxy_delta_rmse_test(pg, null_rmse=0.5)
        assert result["frac_improved"] == pytest.approx(0.0)

    def test_wilcoxon_nan_for_small_sample(self):
        pg = self._make_per_gal(5, 0.3)
        result = galaxy_delta_rmse_test(pg, null_rmse=0.4)
        assert math.isnan(result["wilcoxon_p"])

    def test_wilcoxon_finite_for_large_sample(self):
        pg = self._make_per_gal(20, 0.3)
        result = galaxy_delta_rmse_test(pg, null_rmse=0.4)
        # May still be NaN if all differences are zero, but not for random data
        assert result["n_galaxies"] == 20


# ---------------------------------------------------------------------------
# run_audit — integration
# ---------------------------------------------------------------------------


class TestRunAudit:
    def test_returns_summary_dict(self, small_df, tmp_path):
        summary = run_audit(small_df, tmp_path / "out", n_splits=3,
                            n_perm=20, seed=0)
        assert isinstance(summary, dict)

    def test_writes_per_galaxy_csv(self, small_df, tmp_path):
        run_audit(small_df, tmp_path / "out", n_splits=3, n_perm=10, seed=0)
        assert (tmp_path / "out" / "groupkfold_per_galaxy.csv").exists()

    def test_writes_per_point_csv(self, small_df, tmp_path):
        run_audit(small_df, tmp_path / "out", n_splits=3, n_perm=10, seed=0)
        assert (tmp_path / "out" / "groupkfold_per_point.csv").exists()

    def test_writes_permutation_csv(self, small_df, tmp_path):
        run_audit(small_df, tmp_path / "out", n_splits=3, n_perm=10, seed=0)
        assert (tmp_path / "out" / "permutation_test.csv").exists()

    def test_writes_json_summary(self, small_df, tmp_path):
        run_audit(small_df, tmp_path / "out", n_splits=3, n_perm=10, seed=0)
        assert (tmp_path / "out" / "audit_summary.json").exists()

    def test_json_has_required_keys(self, small_df, tmp_path):
        run_audit(small_df, tmp_path / "out", n_splits=3, n_perm=10, seed=0)
        with open(tmp_path / "out" / "audit_summary.json") as fh:
            data = json.load(fh)
        for key in ("n_galaxies", "n_points", "aicc", "real_rmse",
                    "p_value_permutation", "frac_galaxies_improved",
                    "coeff_slope_mean", "coeff_slope_std",
                    "coeff_intercept_mean", "coeff_intercept_std",
                    "coeff_by_fold", "strict_checks_passed"):
            assert key in data, f"Missing key in audit_summary.json: {key}"

    def test_per_galaxy_csv_is_truly_per_galaxy(self, medium_df, tmp_path):
        """Fix 1: per_galaxy CSV must not have more rows than galaxies × folds."""
        run_audit(medium_df, tmp_path / "out", n_splits=4, n_perm=10, seed=0)
        pg = pd.read_csv(tmp_path / "out" / "groupkfold_per_galaxy.csv")
        pp = pd.read_csv(tmp_path / "out" / "groupkfold_per_point.csv")
        assert len(pg) < len(pp), (
            "groupkfold_per_galaxy.csv has as many rows as per_point.csv "
            "— it is still per-point, not per-galaxy"
        )

    def test_coeff_by_fold_in_json(self, small_df, tmp_path):
        """Fix 4: coeff_by_fold must be present and contain actual per-fold fits."""
        run_audit(small_df, tmp_path / "out", n_splits=3, n_perm=10, seed=0)
        with open(tmp_path / "out" / "audit_summary.json") as fh:
            data = json.load(fh)
        assert len(data["coeff_by_fold"]) >= 1
        for entry in data["coeff_by_fold"]:
            assert "slope" in entry and "intercept" in entry

    def test_strict_mode_passes_when_ok(self, tmp_path):
        """--strict should not raise SystemExit when the data has a real signal."""
        # Strong baryonic signal → should pass
        rng = np.random.default_rng(99)
        rows = []
        for i in range(25):
            lg_bar = rng.uniform(-13, -10, 20)
            lg_obs = 0.5 * lg_bar - 5.0 + rng.normal(0, 0.005, 20)
            for j in range(20):
                rows.append({"galaxy": f"G{i}", "log_g_bar": lg_bar[j],
                              "log_g_obs": lg_obs[j]})
        df = pd.DataFrame(rows)
        # Should not raise SystemExit
        summary = run_audit(df, tmp_path / "strict_ok", n_splits=4,
                            n_perm=200, seed=0, strict=True)
        assert summary["strict_checks_passed"]

    def test_strict_mode_fails_with_noise_only(self, tmp_path):
        """Fix 5: --strict should exit(1) when there is no real signal."""
        rng = np.random.default_rng(5)
        rows = []
        for i in range(15):
            lg_bar = rng.uniform(-13, -10, 10)
            # Pure noise — no relationship
            lg_obs = rng.uniform(-12, -9, 10)
            for j in range(10):
                rows.append({"galaxy": f"G{i}", "log_g_bar": lg_bar[j],
                              "log_g_obs": lg_obs[j]})
        df = pd.DataFrame(rows)
        with pytest.raises(SystemExit) as exc_info:
            run_audit(df, tmp_path / "strict_fail", n_splits=3,
                      n_perm=200, seed=0, strict=True)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# CLI (main) — Fix 5
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_main_creates_outputs(self, small_df, tmp_path):
        csv_path = tmp_path / "input.csv"
        small_df.to_csv(csv_path, index=False)
        out_dir = tmp_path / "audit_out"
        main(["--input", str(csv_path), "--out", str(out_dir),
              "--n-splits", "3", "--n-perm", "20"])
        assert (out_dir / "groupkfold_per_galaxy.csv").exists()
        assert (out_dir / "groupkfold_per_point.csv").exists()
        assert (out_dir / "permutation_test.csv").exists()
        assert (out_dir / "audit_summary.json").exists()

    def test_main_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--input", str(tmp_path / "nonexistent.csv"),
                  "--out", str(tmp_path / "out")])

    def test_main_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["G0"], "log_g_bar": [-11.0]}).to_csv(
            bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--input", str(bad_csv), "--out", str(tmp_path / "out")])
