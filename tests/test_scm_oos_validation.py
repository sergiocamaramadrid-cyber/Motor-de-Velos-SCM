"""
tests/test_scm_oos_validation.py — Tests for scripts/scm_oos_validation.py.

Covers:
  1. Contract validation (column detection, error messages).
  2. β-proxy mode (catalog with friction_slope / delta_from_mond).
  3. Direct-RMSE mode (catalog with pre-computed rmse_base / rmse_scm).
  4. Direct-pred mode (row-level y_true / pred_base / pred_scm, multiple rows per galaxy).
  5. Wilcoxon p-value computation.
  6. Output file creation (CSV, text summary).
  7. CLI main() entrypoint.
  8. delta_f3 column alias (slope_tail − 0.5) required by the paper contract.
  9. Column aliases (galaxy: name/galname; pred columns: yhat_base, yhat_scm, etc.).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.scm_oos_validation import (
    validate_input,
    run_oos_validation,
    _aggregate_direct_pred,
    main as oos_main,
    EXPECTED_BETA,
    DEFAULT_SEEDS,
)

# Significance threshold used in statistical tests throughout this module.
SIGNIFICANCE_THRESHOLD = 0.05

# Tolerance for galaxy-count assertions in split tests.
# The 70/30 split is integer-rounded and random-shuffled, so the actual test-set
# size can differ from int(N * 0.3) by up to ±2 galaxies due to rounding; we add
# 1 extra count as a margin against degenerate seeds.
SPLIT_TOLERANCE = 3

# ---------------------------------------------------------------------------
# Synthetic catalog helpers
# ---------------------------------------------------------------------------


def _make_beta_catalog(
    n: int = 40,
    beta_mean: float = 0.55,
    beta_std: float = 0.08,
    seed: int = 0,
    with_delta: bool = True,
    with_err: bool = True,
) -> pd.DataFrame:
    """Create a synthetic per-galaxy F3 catalog (β-proxy mode)."""
    rng = np.random.default_rng(seed)
    betas = rng.normal(beta_mean, beta_std, n)
    df = pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "friction_slope": betas,
    })
    if with_err:
        df["friction_slope_err"] = rng.uniform(0.01, 0.04, n)
    if with_delta:
        df["delta_from_mond"] = betas - EXPECTED_BETA
    return df


def _make_rmse_catalog(
    n: int = 40,
    seed: int = 0,
    scm_improves: bool = True,
) -> pd.DataFrame:
    """Create a synthetic catalog in direct-RMSE mode."""
    rng = np.random.default_rng(seed)
    rmse_base = rng.uniform(2.0, 10.0, n)
    if scm_improves:
        rmse_scm = rmse_base * rng.uniform(0.5, 0.9, n)   # always better
    else:
        rmse_scm = rmse_base * rng.uniform(0.8, 1.2, n)   # mixed
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "rmse_base": rmse_base,
        "rmse_scm": rmse_scm,
    })


# ---------------------------------------------------------------------------
# 1. Contract validation
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_beta_proxy_mode_detected(self):
        df = _make_beta_catalog()
        col_map = validate_input(df)
        assert col_map["mode"] == "beta_proxy"
        assert col_map["beta"] == "friction_slope"

    def test_beta_alias_detected(self):
        df = _make_beta_catalog()
        df = df.rename(columns={"friction_slope": "beta"})
        col_map = validate_input(df)
        assert col_map["mode"] == "beta_proxy"
        assert col_map["beta"] == "beta"

    def test_direct_rmse_mode_detected(self):
        df = _make_rmse_catalog()
        col_map = validate_input(df)
        assert col_map["mode"] == "direct_rmse"
        assert col_map["rmse_base"] == "rmse_base"
        assert col_map["rmse_scm"] == "rmse_scm"

    def test_missing_galaxy_column_raises(self):
        df = pd.DataFrame({"friction_slope": [0.5, 0.6]})
        with pytest.raises(ValueError, match="missing galaxy identifier"):
            validate_input(df)

    def test_missing_beta_and_rmse_raises(self):
        df = pd.DataFrame({"galaxy": ["A", "B"], "delta_from_mond": [0.1, 0.2]})
        with pytest.raises(ValueError, match="Cannot determine operating mode"):
            validate_input(df)

    def test_delta_column_detected(self):
        df = _make_beta_catalog(with_delta=True)
        col_map = validate_input(df)
        assert col_map.get("delta") == "delta_from_mond"

    def test_delta_f3_alias_detected(self):
        """delta_f3 = slope_tail − 0.5 is the paper's canonical column name."""
        df = _make_beta_catalog(with_delta=False)
        df["delta_f3"] = df["friction_slope"] - EXPECTED_BETA
        col_map = validate_input(df)
        assert col_map.get("delta") == "delta_f3"


# ---------------------------------------------------------------------------
# 2. β-proxy mode — core logic
# ---------------------------------------------------------------------------


class TestBetaProxyMode:
    def test_returns_required_aggregate_keys(self, tmp_path):
        df = _make_beta_catalog(n=40)
        path = tmp_path / "catalog.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        agg = result["aggregate"]
        required = {"n_valid", "pct_improved", "delta_rmse_median", "wilcoxon_p"}
        assert required.issubset(set(agg.keys()))

    def test_n_valid_approximately_30pct(self, tmp_path):
        n = 60
        df = _make_beta_catalog(n=n)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", train_frac=0.7,
                                    seeds=[42], no_figures=True)
        n_test = result["aggregate"]["n_valid"]
        # ±SPLIT_TOLERANCE galaxies: the split is integer-rounded so a 60-galaxy catalog
        # at 70/30 gives 18 test galaxies, but random shuffling can vary by ±1–2.
        # The tolerance accommodates rounding and a degenerate seed.
        assert abs(n_test - int(n * 0.3)) <= SPLIT_TOLERANCE

    def test_wilcoxon_p_is_finite(self, tmp_path):
        df = _make_beta_catalog(n=40)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        p = result["aggregate"]["wilcoxon_p"]
        assert math.isfinite(p)
        assert 0.0 <= p <= 1.0

    def test_systematic_deviation_gives_low_p(self, tmp_path):
        """If β is consistently offset from 0.5, training prior improves test set."""
        # Galaxies with β ≈ 0.7 (large systematic offset from 0.5)
        df = _make_beta_catalog(n=80, beta_mean=0.7, beta_std=0.02)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out",
                                    seeds=list(range(5)), no_figures=True)
        p = result["aggregate"]["wilcoxon_p"]
        # SCM (predict mean ≈ 0.7) should outperform base (predict 0.5)
        assert p < SIGNIFICANCE_THRESHOLD, f"Expected p < {SIGNIFICANCE_THRESHOLD} for systematic β=0.7, got p={p:.4f}"

    def test_no_deviation_gives_high_p(self, tmp_path):
        """If β ≈ 0.5, SCM offers no improvement over base — p should not be tiny."""
        df = _make_beta_catalog(n=80, beta_mean=0.5, beta_std=0.01)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out",
                                    seeds=list(range(5)), no_figures=True)
        p = result["aggregate"]["wilcoxon_p"]
        # SCM (mean ≈ 0.5) ≈ base (predict 0.5) → p should not be near 0
        assert p > SIGNIFICANCE_THRESHOLD, f"Expected p > {SIGNIFICANCE_THRESHOLD} for β≈0.5, got p={p:.4f}"

    def test_per_galaxy_df_columns(self, tmp_path):
        df = _make_beta_catalog(n=40)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        cols = set(result["per_galaxy_df"].columns)
        assert {"galaxy", "rmse_base", "rmse_scm", "delta_rmse", "delta_logL",
                "seed"}.issubset(cols)

    def test_delta_rmse_sign_convention(self, tmp_path):
        """delta_rmse = rmse_scm - rmse_base; negative means SCM wins."""
        df = _make_beta_catalog(n=80, beta_mean=0.7, beta_std=0.02)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        per_gal = result["per_galaxy_df"]
        # rmse_scm and rmse_base must be non-negative
        assert (per_gal["rmse_base"] >= 0).all()
        assert (per_gal["rmse_scm"] >= 0).all()
        # delta_rmse = rmse_scm - rmse_base
        pd.testing.assert_series_equal(
            per_gal["delta_rmse"],
            per_gal["rmse_scm"] - per_gal["rmse_base"],
            check_names=False,
        )

    def test_multiple_seeds_produce_multiple_rows(self, tmp_path):
        df = _make_beta_catalog(n=40)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        seeds = [42, 43, 44]
        result = run_oos_validation(path, tmp_path / "out", seeds=seeds,
                                    no_figures=True)
        assert len(result["all_summaries"]) == len(seeds)

    def test_accepts_parquet_input(self, tmp_path):
        df = _make_beta_catalog(n=30)
        path = tmp_path / "cat.parquet"
        df.to_parquet(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        assert result["aggregate"]["n_valid"] > 0


# ---------------------------------------------------------------------------
# 3. Direct-RMSE mode
# ---------------------------------------------------------------------------


class TestDirectRmseMode:
    def test_scm_improves_gives_low_p(self, tmp_path):
        df = _make_rmse_catalog(n=80, scm_improves=True, seed=7)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out",
                                    seeds=list(range(5)), no_figures=True)
        p = result["aggregate"]["wilcoxon_p"]
        assert p < SIGNIFICANCE_THRESHOLD, f"Expected p < {SIGNIFICANCE_THRESHOLD} when SCM always improves, got {p:.4f}"

    def test_pct_improved_near_100_when_scm_always_better(self, tmp_path):
        df = _make_rmse_catalog(n=60, scm_improves=True, seed=3)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        pct = result["aggregate"]["pct_improved"]
        assert pct > 80.0, f"Expected >80% improvement, got {pct:.1f}%"


# ---------------------------------------------------------------------------
# 4. Output file creation
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_oos_generalization_results_csv_created(self, tmp_path):
        df = _make_beta_catalog(n=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42], no_figures=True)
        assert (out / "oos_generalization_results.csv").exists()

    def test_oos_summary_txt_created(self, tmp_path):
        df = _make_beta_catalog(n=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42], no_figures=True)
        assert (out / "oos_summary.txt").exists()

    def test_summary_contains_wilcoxon_p(self, tmp_path):
        df = _make_beta_catalog(n=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42], no_figures=True)
        text = (out / "oos_summary.txt").read_text()
        assert "Wilcoxon" in text or "wilcoxon" in text.lower()
        assert "p =" in text.lower() or "p-value" in text.lower()

    def test_per_galaxy_csv_has_expected_columns(self, tmp_path):
        df = _make_beta_catalog(n=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42], no_figures=True)
        result_df = pd.read_csv(out / "oos_generalization_results.csv")
        required = {"galaxy", "rmse_base", "rmse_scm", "delta_rmse", "delta_logL", "seed"}
        assert required.issubset(set(result_df.columns))

    def test_seed_summary_csv_created(self, tmp_path):
        df = _make_beta_catalog(n=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42, 43], no_figures=True)
        assert (out / "oos_seed_summary.csv").exists()
        sdf = pd.read_csv(out / "oos_seed_summary.csv")
        assert len(sdf) == 2


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nan_beta_rows_dropped(self, tmp_path):
        df = _make_beta_catalog(n=40)
        df.loc[5:10, "friction_slope"] = float("nan")
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        assert result["aggregate"]["n_valid"] > 0

    def test_empty_catalog_raises(self, tmp_path):
        df = pd.DataFrame({"galaxy": [], "friction_slope": []})
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="empty"):
            run_oos_validation(path, tmp_path / "out", seeds=[42], no_figures=True)

    def test_minimum_viable_galaxy_count(self, tmp_path):
        """Script should handle very small catalogs (≥ MIN_TEST_GALAXIES + 1)."""
        df = _make_beta_catalog(n=10)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        # Wilcoxon might be NaN if test set < MIN_TEST_GALAXIES, but run completes
        assert "n_valid" in result["aggregate"]


# ---------------------------------------------------------------------------
# 6. CLI entrypoint
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_runs_and_produces_outputs(self, tmp_path):
        df = _make_beta_catalog(n=40)
        catalog = tmp_path / "catalog.csv"
        df.to_csv(catalog, index=False)
        out = tmp_path / "oos_out"
        result = oos_main([
            "--input", str(catalog),
            "--split", "0.7",
            "--seeds", "42",
            "--out", str(out),
            "--no-figures",
        ])
        assert (out / "oos_generalization_results.csv").exists()
        assert (out / "oos_summary.txt").exists()
        assert "aggregate" in result

    def test_cli_multiple_seeds(self, tmp_path):
        df = _make_beta_catalog(n=40)
        catalog = tmp_path / "catalog.csv"
        df.to_csv(catalog, index=False)
        out = tmp_path / "oos_multi"
        result = oos_main([
            "--input", str(catalog),
            "--seeds", "10", "20", "30",
            "--out", str(out),
            "--no-figures",
        ])
        assert len(result["all_summaries"]) == 3


# ---------------------------------------------------------------------------
# 7. Paper contract — delta_f3 column
# ---------------------------------------------------------------------------


class TestPaperContract:
    """Enforce the column contract required by the MNRAS paper.

    delta_f3 = slope_tail − 0.5 must be accepted as an alias for delta_from_mond.
    """

    def test_delta_f3_column_accepted(self, tmp_path):
        df = _make_beta_catalog(n=40, with_delta=False)
        df["delta_f3"] = df["friction_slope"] - 0.5
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        col_map = validate_input(df)
        assert col_map.get("delta") == "delta_f3"

    def test_delta_f3_equals_friction_slope_minus_half(self, tmp_path):
        df = _make_beta_catalog(n=40, with_delta=False)
        df["delta_f3"] = df["friction_slope"] - 0.5
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        # run must complete without error
        assert result["aggregate"]["n_valid"] > 0


# ---------------------------------------------------------------------------
# Helpers for direct_pred mode
# ---------------------------------------------------------------------------


def _make_pred_catalog(
    n_galaxies: int = 40,
    n_pts_per_galaxy: int = 15,
    seed: int = 0,
    scm_improves: bool = True,
    galaxy_col: str = "galaxy",
    y_true_col: str = "y_true",
    pred_base_col: str = "pred_base",
    pred_scm_col: str = "pred_scm",
) -> pd.DataFrame:
    """Create a synthetic row-level prediction catalog (direct_pred mode).

    Parameters
    ----------
    scm_improves : bool
        If True the SCM predictions are 80 % closer to y_true than the baseline
        (systematic improvement).  If False, mixed noise is added.
    galaxy_col, y_true_col, pred_base_col, pred_scm_col : str
        Column names to use (support alias testing).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gal_idx in range(n_galaxies):
        gal = f"G{gal_idx:03d}"
        y = rng.uniform(50.0, 250.0, n_pts_per_galaxy)
        noise_base = rng.normal(0, 20.0, n_pts_per_galaxy)
        if scm_improves:
            noise_scm = rng.normal(0, 5.0, n_pts_per_galaxy)   # much smaller error
        else:
            noise_scm = rng.normal(0, 18.0, n_pts_per_galaxy)  # similar to base
        for y_i, nb, ns in zip(y, noise_base, noise_scm):
            rows.append({
                galaxy_col: gal,
                y_true_col: y_i,
                pred_base_col: y_i + nb,
                pred_scm_col: y_i + ns,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. direct_pred mode
# ---------------------------------------------------------------------------


class TestDirectPredMode:
    def test_mode_detected(self):
        df = _make_pred_catalog()
        col_map = validate_input(df)
        assert col_map["mode"] == "direct_pred"
        assert col_map["y_true"] == "y_true"
        assert col_map["pred_base"] == "pred_base"
        assert col_map["pred_scm"] == "pred_scm"

    def test_direct_pred_takes_priority_over_beta_proxy(self):
        """direct_pred mode must win when both y_true and friction_slope are present."""
        df = _make_pred_catalog()
        df["friction_slope"] = 0.5  # add beta_proxy column too
        col_map = validate_input(df)
        assert col_map["mode"] == "direct_pred"

    def test_direct_pred_takes_priority_over_direct_rmse(self):
        """direct_pred mode must win when both prediction and rmse columns exist."""
        df = _make_pred_catalog()
        df["rmse_base"] = 10.0
        df["rmse_scm"] = 9.0
        col_map = validate_input(df)
        assert col_map["mode"] == "direct_pred"

    def test_aggregate_direct_pred_computes_rmse(self):
        df = _make_pred_catalog(n_galaxies=5, n_pts_per_galaxy=20)
        col_map = validate_input(df)
        agg = _aggregate_direct_pred(df, col_map)
        assert set(agg.columns) >= {"galaxy", "rmse_base", "rmse_scm", "mae_base", "mae_scm"}
        assert len(agg) == 5
        assert (agg["rmse_base"] >= 0).all()
        assert (agg["rmse_scm"] >= 0).all()

    def test_scm_improves_gives_low_p(self, tmp_path):
        df = _make_pred_catalog(n_galaxies=60, scm_improves=True, seed=7)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out",
                                    seeds=list(range(5)), no_figures=True)
        p = result["aggregate"]["wilcoxon_p"]
        assert p < SIGNIFICANCE_THRESHOLD, (
            f"Expected p < {SIGNIFICANCE_THRESHOLD} when SCM always improves, got {p:.4f}"
        )

    def test_pct_improved_near_100_when_scm_better(self, tmp_path):
        df = _make_pred_catalog(n_galaxies=60, scm_improves=True, seed=3)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        pct = result["aggregate"]["pct_improved"]
        assert pct > 80.0, f"Expected >80% improvement, got {pct:.1f}%"

    def test_outputs_csv_has_mae_columns(self, tmp_path):
        df = _make_pred_catalog(n_galaxies=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42], no_figures=True)
        result_df = pd.read_csv(out / "oos_generalization_results.csv")
        assert {"mae_base", "mae_scm", "delta_mae"}.issubset(set(result_df.columns))

    def test_delta_mae_sign_convention(self, tmp_path):
        """delta_mae = mae_scm - mae_base."""
        df = _make_pred_catalog(n_galaxies=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        per_gal = result["per_galaxy_df"]
        pd.testing.assert_series_equal(
            per_gal["delta_mae"],
            per_gal["mae_scm"] - per_gal["mae_base"],
            check_names=False,
        )

    def test_n_valid_approximately_30pct(self, tmp_path):
        n = 60
        df = _make_pred_catalog(n_galaxies=n, n_pts_per_galaxy=10)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", train_frac=0.7,
                                    seeds=[42], no_figures=True)
        n_test = result["aggregate"]["n_valid"]
        # See SPLIT_TOLERANCE at the top of this module for the justification.
        assert abs(n_test - int(n * 0.3)) <= SPLIT_TOLERANCE

    def test_output_summary_txt_created(self, tmp_path):
        df = _make_pred_catalog(n_galaxies=30)
        path = tmp_path / "cat.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "results"
        run_oos_validation(path, out, seeds=[42], no_figures=True)
        assert (out / "oos_summary.txt").exists()


# ---------------------------------------------------------------------------
# 9. Column alias tests
# ---------------------------------------------------------------------------


class TestColumnAliases:
    """Verify that all documented column aliases are accepted."""

    def test_galaxy_col_name_alias(self, tmp_path):
        df = _make_pred_catalog(galaxy_col="name")
        col_map = validate_input(df)
        assert col_map["galaxy"] == "name"
        assert col_map["mode"] == "direct_pred"

    def test_galaxy_col_galname_alias(self, tmp_path):
        df = _make_pred_catalog(galaxy_col="galname")
        col_map = validate_input(df)
        assert col_map["galaxy"] == "galname"

    def test_y_true_target_alias(self, tmp_path):
        df = _make_pred_catalog(y_true_col="target")
        col_map = validate_input(df)
        assert col_map["y_true"] == "target"
        assert col_map["mode"] == "direct_pred"

    def test_y_true_observed_alias(self, tmp_path):
        df = _make_pred_catalog(y_true_col="observed")
        col_map = validate_input(df)
        assert col_map["y_true"] == "observed"

    def test_pred_base_yhat_base_alias(self, tmp_path):
        df = _make_pred_catalog(pred_base_col="yhat_base")
        col_map = validate_input(df)
        assert col_map["pred_base"] == "yhat_base"
        assert col_map["mode"] == "direct_pred"

    def test_pred_base_pred_baseline_alias(self, tmp_path):
        df = _make_pred_catalog(pred_base_col="pred_baseline")
        col_map = validate_input(df)
        assert col_map["pred_base"] == "pred_baseline"

    def test_pred_scm_yhat_scm_alias(self, tmp_path):
        df = _make_pred_catalog(pred_scm_col="yhat_scm")
        col_map = validate_input(df)
        assert col_map["pred_scm"] == "yhat_scm"
        assert col_map["mode"] == "direct_pred"

    def test_pred_scm_pred_model_alias(self, tmp_path):
        df = _make_pred_catalog(pred_scm_col="pred_model")
        col_map = validate_input(df)
        assert col_map["pred_scm"] == "pred_model"

    def test_full_alias_catalog_runs_end_to_end(self, tmp_path):
        """Full OOS run with all alternative alias names must succeed."""
        df = _make_pred_catalog(
            n_galaxies=40,
            galaxy_col="galname",
            y_true_col="observed",
            pred_base_col="yhat_base",
            pred_scm_col="pred_model",
        )
        path = tmp_path / "cat_aliases.csv"
        df.to_csv(path, index=False)
        result = run_oos_validation(path, tmp_path / "out", seeds=[42],
                                    no_figures=True)
        assert result["aggregate"]["n_valid"] > 0
        assert math.isfinite(result["aggregate"]["wilcoxon_p"])

    def test_beta_proxy_with_name_galaxy_column(self, tmp_path):
        """beta_proxy mode must also work when the galaxy column is named 'name'."""
        df = _make_beta_catalog(n=40)
        df = df.rename(columns={"galaxy": "name"})
        col_map = validate_input(df)
        assert col_map["galaxy"] == "name"
        assert col_map["mode"] == "beta_proxy"
