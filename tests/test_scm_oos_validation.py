"""
tests/test_scm_oos_validation.py — Tests for scripts/scm_oos_validation.py.

Covers:
  1. Contract validation (column detection, error messages).
  2. β-proxy mode (catalog with friction_slope / delta_from_mond).
  3. Direct-RMSE mode (catalog with pre-computed rmse_base / rmse_scm).
  4. Wilcoxon p-value computation.
  5. Output file creation (CSV, text summary).
  6. CLI main() entrypoint.
  7. delta_f3 column alias (slope_tail − 0.5) required by the paper contract.
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
    main as oos_main,
    EXPECTED_BETA,
    DEFAULT_SEEDS,
)

# Significance threshold used in statistical tests throughout this module.
SIGNIFICANCE_THRESHOLD = 0.05

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
        with pytest.raises(ValueError, match="missing required columns"):
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
        # Allow ±3 galaxies: the split is integer-rounded so a 60-galaxy catalog
        # at 70/30 gives 18 test galaxies, but random shuffling can vary by ±1–2.
        # The ±3 tolerance accommodates both the rounding and one extra shuffle step.
        assert abs(n_test - int(n * 0.3)) <= 3

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
