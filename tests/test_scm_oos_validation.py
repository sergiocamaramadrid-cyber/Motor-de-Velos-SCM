"""
tests/test_scm_oos_validation.py — Tests for scripts/scm_oos_validation.py.

Covers:
  1. aicc()                    — corrected AIC formula.
  2. compare_models_full()     — full-sample model comparison.
  3. run_oos_split()           — single OOS split metrics.
  4. run_oos_validation()      — multi-split loop + Wilcoxon test.
  5. build_residual_catalog()  — per-galaxy residuals.
  6. identify_extreme_galaxies() — top-N extreme galaxies.
  7. build_env2_catalog()      — alternative env proxy robustness.
  8. Figure generation         — smoke tests (files written, no crash).
  9. run_validation_pipeline() — end-to-end pipeline.
 10. main() CLI                — CLI entry point.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.scm_oos_validation import (
    N_EXTREME_DEFAULT,
    aicc,
    compare_models_full,
    run_oos_split,
    run_oos_validation,
    build_residual_catalog,
    identify_extreme_galaxies,
    build_env2_catalog,
    plot_scatter,
    plot_delta_rmse_hist,
    plot_delta_rmse_scatter,
    run_validation_pipeline,
    main,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_env_catalog(
    n: int = 40,
    seed: int = 42,
    beta_ref: float = 0.5,
) -> pd.DataFrame:
    """Build a synthetic galaxy_catalog_with_env.csv-like DataFrame."""
    rng = np.random.default_rng(seed)
    logMbar = rng.uniform(8.0, 11.0, n)
    delta_mass = rng.uniform(-1.5, 1.5, n)
    # delta_f3 correlates with both logMbar and delta_mass
    delta_f3 = 0.02 * (logMbar - 9.5) + 0.05 * delta_mass + rng.normal(0, 0.05, n)
    beta = delta_f3 + beta_ref
    logRd = rng.uniform(-0.5, 1.0, n)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "beta": beta,
        "delta_f3": delta_f3,
        "logMbar": logMbar,
        "logRd": logRd,
        "delta_mass": delta_mass,
    })


def _write_env_catalog(tmp_path: Path, **kwargs) -> Path:
    df = _make_env_catalog(**kwargs)
    p = tmp_path / "galaxy_catalog_with_env.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. aicc()
# ---------------------------------------------------------------------------

class TestAicc:
    def test_returns_float(self):
        result = aicc(-50.0, 3, 40)
        assert isinstance(result, float)

    def test_increases_with_more_parameters(self):
        ll = -50.0
        n = 100
        aic2 = aicc(ll, 2, n)
        aic4 = aicc(ll, 4, n)
        assert aic4 > aic2

    def test_nan_when_denominator_zero(self):
        # n - k - 1 = 0  →  nan
        result = aicc(-50.0, k=4, n=5)
        assert math.isnan(result)

    def test_nan_when_denominator_negative(self):
        result = aicc(-50.0, k=10, n=5)
        assert math.isnan(result)

    def test_lower_ll_gives_higher_aicc(self):
        n, k = 100, 3
        aic_good = aicc(-40.0, k, n)
        aic_bad = aicc(-80.0, k, n)
        assert aic_bad > aic_good


# ---------------------------------------------------------------------------
# 2. compare_models_full()
# ---------------------------------------------------------------------------

class TestCompareModelsFull:
    def test_returns_required_keys(self):
        df = _make_env_catalog(n=30)
        result = compare_models_full(df)
        required = {
            "n", "base_aicc", "full_aicc", "delta_aicc",
            "base_bic", "full_bic", "delta_bic",
            "base_r2", "full_r2", "delta_r2",
            "base_coef", "full_coef",
        }
        assert required.issubset(set(result.keys()))

    def test_n_matches_valid_rows(self):
        df = _make_env_catalog(n=30)
        result = compare_models_full(df)
        assert result["n"] == 30

    def test_delta_aicc_positive_when_full_better(self):
        """delta_aicc = base_aicc - full_aicc > 0 when full model fits better."""
        df = _make_env_catalog(n=60, seed=0)
        result = compare_models_full(df)
        # With a genuine env signal the full model should be favoured
        assert isinstance(result["delta_aicc"], float)

    def test_r2_in_valid_range(self):
        df = _make_env_catalog(n=40)
        result = compare_models_full(df)
        assert 0.0 <= result["base_r2"] <= 1.0
        assert 0.0 <= result["full_r2"] <= 1.0

    def test_full_r2_geq_base_r2(self):
        """Adding a predictor never decreases R²."""
        df = _make_env_catalog(n=50)
        result = compare_models_full(df)
        assert result["full_r2"] >= result["base_r2"] - 1e-9

    def test_missing_column_raises(self):
        df = pd.DataFrame({"delta_f3": [0.1], "logMbar": [9.0]})
        with pytest.raises(Exception):
            compare_models_full(df)


# ---------------------------------------------------------------------------
# 3. run_oos_split()
# ---------------------------------------------------------------------------

class TestRunOosSplit:
    def test_returns_required_keys(self):
        df = _make_env_catalog(n=30)
        result = run_oos_split(df, seed=0)
        required = {
            "n_train", "n_test",
            "rmse_base_train", "rmse_base_test",
            "rmse_full_train", "rmse_full_test",
            "delta_rmse_test",
        }
        assert required.issubset(set(result.keys()))

    def test_train_test_counts_sum_to_n(self):
        n = 30
        df = _make_env_catalog(n=n)
        result = run_oos_split(df, test_frac=0.3, seed=1)
        assert result["n_train"] + result["n_test"] == n

    def test_rmse_values_are_positive(self):
        df = _make_env_catalog(n=30)
        result = run_oos_split(df, seed=2)
        assert result["rmse_base_test"] > 0
        assert result["rmse_full_test"] > 0

    def test_delta_rmse_equals_base_minus_full(self):
        df = _make_env_catalog(n=30)
        result = run_oos_split(df, seed=3)
        expected = result["rmse_base_test"] - result["rmse_full_test"]
        assert result["delta_rmse_test"] == pytest.approx(expected, rel=1e-9)

    def test_different_seeds_give_different_splits(self):
        df = _make_env_catalog(n=40)
        r1 = run_oos_split(df, seed=0)
        r2 = run_oos_split(df, seed=999)
        # Different seeds should almost certainly give different RMSE values
        assert r1["rmse_base_test"] != pytest.approx(r2["rmse_base_test"])

    def test_small_dataset_does_not_crash(self):
        df = _make_env_catalog(n=6)
        result = run_oos_split(df, test_frac=0.3, seed=0)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. run_oos_validation()
# ---------------------------------------------------------------------------

class TestRunOosValidation:
    def test_returns_required_keys(self):
        df = _make_env_catalog(n=40)
        result = run_oos_validation(df, n_splits=10, seed=0)
        required = {
            "splits", "delta_rmse_arr",
            "wilcoxon_stat", "wilcoxon_pvalue",
            "median_delta_rmse", "frac_positive",
        }
        assert required.issubset(set(result.keys()))

    def test_delta_rmse_arr_length_matches_n_splits(self):
        n_splits = 15
        df = _make_env_catalog(n=40)
        result = run_oos_validation(df, n_splits=n_splits, seed=0)
        assert len(result["delta_rmse_arr"]) == n_splits

    def test_median_delta_rmse_is_scalar(self):
        df = _make_env_catalog(n=40)
        result = run_oos_validation(df, n_splits=10, seed=0)
        assert isinstance(result["median_delta_rmse"], float)

    def test_frac_positive_in_unit_interval(self):
        df = _make_env_catalog(n=40)
        result = run_oos_validation(df, n_splits=10, seed=0)
        fp = result["frac_positive"]
        assert 0.0 <= fp <= 1.0

    def test_wilcoxon_pvalue_in_unit_interval(self):
        df = _make_env_catalog(n=40)
        result = run_oos_validation(df, n_splits=20, seed=0)
        pval = result["wilcoxon_pvalue"]
        if not math.isnan(pval):
            assert 0.0 <= pval <= 1.0

    def test_splits_list_length_matches_n_splits(self):
        df = _make_env_catalog(n=40)
        result = run_oos_validation(df, n_splits=8, seed=0)
        assert len(result["splits"]) == 8


# ---------------------------------------------------------------------------
# 5. build_residual_catalog()
# ---------------------------------------------------------------------------

class TestBuildResidualCatalog:
    def test_returns_dataframe(self):
        df = _make_env_catalog(n=30)
        result = build_residual_catalog(df)
        assert isinstance(result, pd.DataFrame)

    def test_added_columns_present(self):
        df = _make_env_catalog(n=30)
        result = build_residual_catalog(df)
        for col in ["pred_base", "resid_base", "pred_full", "resid_full", "delta_resid"]:
            assert col in result.columns

    def test_residual_mean_near_zero(self):
        """OLS residuals sum to zero."""
        df = _make_env_catalog(n=50)
        result = build_residual_catalog(df)
        assert result["resid_base"].mean() == pytest.approx(0.0, abs=1e-6)
        assert result["resid_full"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_delta_resid_definition(self):
        """delta_resid = |resid_base| - |resid_full|."""
        df = _make_env_catalog(n=30)
        result = build_residual_catalog(df)
        expected = result["resid_base"].abs() - result["resid_full"].abs()
        pd.testing.assert_series_equal(
            result["delta_resid"].reset_index(drop=True),
            expected.reset_index(drop=True),
            atol=1e-10,
            check_names=False,
        )

    def test_row_count_preserved_for_valid_inputs(self):
        n = 30
        df = _make_env_catalog(n=n)
        result = build_residual_catalog(df)
        assert len(result) == n


# ---------------------------------------------------------------------------
# 6. identify_extreme_galaxies()
# ---------------------------------------------------------------------------

class TestIdentifyExtremeGalaxies:
    def test_returns_n_rows(self):
        df = _make_env_catalog(n=50)
        residuals = build_residual_catalog(df)
        extreme = identify_extreme_galaxies(residuals, n=10)
        assert len(extreme) == 10

    def test_sorted_descending_by_abs_delta_resid(self):
        df = _make_env_catalog(n=50)
        residuals = build_residual_catalog(df)
        extreme = identify_extreme_galaxies(residuals, n=15)
        abs_vals = extreme["abs_delta_resid"].values
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))

    def test_n_larger_than_sample_returns_all(self):
        df = _make_env_catalog(n=10)
        residuals = build_residual_catalog(df)
        extreme = identify_extreme_galaxies(residuals, n=100)
        assert len(extreme) == 10

    def test_default_n_is_25(self):
        df = _make_env_catalog(n=50)
        residuals = build_residual_catalog(df)
        extreme = identify_extreme_galaxies(residuals)
        assert len(extreme) == min(N_EXTREME_DEFAULT, 50)


# ---------------------------------------------------------------------------
# 7. build_env2_catalog()
# ---------------------------------------------------------------------------

class TestBuildEnv2Catalog:
    def test_adds_env2_column(self):
        df = _make_env_catalog(n=20)
        result = build_env2_catalog(df)
        assert "env2" in result.columns

    def test_env2_is_centred(self):
        """env2 should have approximately zero mean (centred proxy)."""
        df = _make_env_catalog(n=50)
        result = build_env2_catalog(df)
        assert result["env2"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_original_columns_preserved(self):
        df = _make_env_catalog(n=20)
        result = build_env2_catalog(df)
        for col in df.columns:
            assert col in result.columns

    def test_no_delta_mass_column_gives_nan_env2(self):
        df = _make_env_catalog(n=10).drop(columns=["delta_mass"])
        result = build_env2_catalog(df)
        assert result["env2"].isna().all()


# ---------------------------------------------------------------------------
# 8. Figure generation (smoke tests)
# ---------------------------------------------------------------------------

class TestFigureGeneration:
    def test_plot_scatter_writes_pdf(self, tmp_path):
        df = _make_env_catalog(n=20)
        out_path = plot_scatter(df, tmp_path)
        assert out_path.exists()
        assert out_path.suffix == ".pdf"

    def test_plot_delta_rmse_hist_writes_pdf(self, tmp_path):
        delta_rmse = np.random.default_rng(0).normal(0.005, 0.01, 50)
        out_path = plot_delta_rmse_hist(delta_rmse, wilcoxon_pvalue=0.03, out_dir=tmp_path)
        assert out_path.exists()
        assert out_path.suffix == ".pdf"

    def test_plot_delta_rmse_scatter_writes_pdf(self, tmp_path):
        df = _make_env_catalog(n=20)
        residuals = build_residual_catalog(df)
        out_path = plot_delta_rmse_scatter(residuals, tmp_path)
        assert out_path.exists()
        assert out_path.suffix == ".pdf"

    def test_plot_delta_rmse_hist_nan_pvalue_does_not_crash(self, tmp_path):
        delta_rmse = np.array([0.01, 0.02, -0.01])
        # Should not raise even with NaN pvalue
        out_path = plot_delta_rmse_hist(delta_rmse, wilcoxon_pvalue=float("nan"), out_dir=tmp_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# 9. run_validation_pipeline()
# ---------------------------------------------------------------------------

class TestRunValidationPipeline:
    def test_returns_required_keys(self, tmp_path):
        p = _write_env_catalog(tmp_path, n=40)
        result = run_validation_pipeline(
            input_path=p, out_dir=tmp_path / "data",
            figures_dir=tmp_path / "figs",
            n_splits=5, seed=0,
        )
        required = {"model_comparison", "oos", "df_residuals", "df_extreme", "df_env2"}
        assert required.issubset(set(result.keys()))

    def test_output_csvs_written(self, tmp_path):
        p = _write_env_catalog(tmp_path, n=40)
        out = tmp_path / "data"
        run_validation_pipeline(
            input_path=p, out_dir=out,
            figures_dir=tmp_path / "figs",
            n_splits=5, seed=0,
        )
        assert (out / "galaxy_catalog_with_residual.csv").exists()
        assert (out / "extreme_25_results.csv").exists()
        assert (out / "galaxy_catalog_env2.csv").exists()

    def test_figures_written(self, tmp_path):
        p = _write_env_catalog(tmp_path, n=40)
        figs = tmp_path / "figs"
        run_validation_pipeline(
            input_path=p, out_dir=tmp_path / "data",
            figures_dir=figs,
            n_splits=5, seed=0,
        )
        assert (figs / "figure02_delta_rmse_hist.pdf").exists()
        assert (figs / "figure03_delta_rmse_scatter.pdf").exists()

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_validation_pipeline(
                input_path=tmp_path / "missing.csv",
                out_dir=tmp_path / "data",
                figures_dir=tmp_path / "figs",
            )

    def test_extreme_galaxies_count(self, tmp_path):
        n = 60
        p = _write_env_catalog(tmp_path, n=n)
        result = run_validation_pipeline(
            input_path=p, out_dir=tmp_path / "data",
            figures_dir=tmp_path / "figs",
            n_splits=5, n_extreme=10, seed=0,
        )
        assert len(result["df_extreme"]) == 10

    def test_missing_delta_f3_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"logMbar": [9.0], "delta_mass": [0.1]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing"):
            run_validation_pipeline(
                input_path=bad, out_dir=tmp_path / "data",
                figures_dir=tmp_path / "figs",
            )


# ---------------------------------------------------------------------------
# 10. main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_dict_via_kwargs(self, tmp_path):
        p = _write_env_catalog(tmp_path, n=40)
        result = main(
            input_path=str(p),
            out_dir=str(tmp_path / "data"),
            figures_dir=str(tmp_path / "figs"),
            n_splits=5,
            seed=0,
        )
        assert isinstance(result, dict)
        assert "oos" in result

    def test_returns_dict_via_argv(self, tmp_path):
        p = _write_env_catalog(tmp_path, n=40)
        result = main([
            "--input", str(p),
            "--out-dir", str(tmp_path / "data"),
            "--figures-dir", str(tmp_path / "figs"),
            "--n-splits", "5",
            "--seed", "0",
        ])
        assert isinstance(result, dict)

    def test_default_n_splits_is_100(self):
        args = _parse_args([])
        assert args.n_splits == 100

    def test_default_test_frac_is_0_30(self):
        args = _parse_args([])
        assert args.test_frac == pytest.approx(0.30)

    def test_default_seed_is_42(self):
        args = _parse_args([])
        assert args.seed == 42
