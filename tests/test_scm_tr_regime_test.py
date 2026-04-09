"""
tests/test_scm_tr_regime_test.py — Tests for scripts/scm_tr_regime_test.py.

Uses synthetic data with planted correlations to verify correctness of
Fisher Z comparison, bootstrap CI, mass scan, and HC3 OLS.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.scm_tr_regime_test import (
    DEFAULT_MASS_COL,
    DEFAULT_ENV_COL,
    DEFAULT_SLOPE_COL,
    RegimeStats,
    FisherComparison,
    BootstrapSummary,
    fisher_z_from_r,
    fisher_compare_correlations,
    clean_dataframe,
    spearman_stats,
    bootstrap_spearman,
    run_mass_scan,
    run_hc3_ols,
    make_plots,
    write_summary_text,
    save_json,
    ensure_dir,
    main,
    HAS_STATSMODELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    tmp_path: Path,
    n_low: int = 40,
    n_high: int = 40,
    rho_low: float = 0.0,
    rho_high: float = -0.6,
    low_mass: float = 9.7,
    high_mass: float = 10.4,
    seed: int = 42,
    mass_col: str = DEFAULT_MASS_COL,
    env_col: str = DEFAULT_ENV_COL,
    slope_col: str = DEFAULT_SLOPE_COL,
) -> Path:
    """Build a synthetic galaxy catalog CSV with controlled Spearman correlations."""
    rng = np.random.default_rng(seed)

    def correlated_pair(n: int, rho: float) -> tuple:
        """Return (x, y) arrays with approximate Spearman rank correlation rho."""
        x = rng.normal(0, 1, n)
        noise = rng.normal(0, 1, n)
        y = rho * x + math.sqrt(max(1.0 - rho ** 2, 0.0)) * noise
        return x, y

    env_low, slope_low = correlated_pair(n_low, rho_low)
    env_high, slope_high = correlated_pair(n_high, rho_high)

    df = pd.DataFrame({
        mass_col: np.concatenate([
            rng.uniform(8.5, low_mass, n_low),
            rng.uniform(high_mass, 11.5, n_high),
        ]),
        env_col: np.concatenate([env_low, env_high]),
        slope_col: np.concatenate([slope_low, slope_high]),
    })
    p = tmp_path / "catalog.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_columns_defined(self):
        assert DEFAULT_MASS_COL == "logMbar"
        assert DEFAULT_ENV_COL == "env_proxy"
        assert DEFAULT_SLOPE_COL == "slope_tail"


# ---------------------------------------------------------------------------
# fisher_z_from_r
# ---------------------------------------------------------------------------

class TestFisherZFromR:
    def test_zero_correlation(self):
        assert fisher_z_from_r(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_positive_correlation(self):
        assert fisher_z_from_r(0.5) == pytest.approx(0.5493, abs=1e-3)

    def test_negative_correlation(self):
        assert fisher_z_from_r(-0.5) == pytest.approx(-0.5493, abs=1e-3)

    def test_clamps_at_one(self):
        # Must not raise; result is finite
        z = fisher_z_from_r(1.0)
        assert math.isfinite(z)

    def test_clamps_at_minus_one(self):
        z = fisher_z_from_r(-1.0)
        assert math.isfinite(z)

    def test_antisymmetric(self):
        assert fisher_z_from_r(0.7) == pytest.approx(-fisher_z_from_r(-0.7), abs=1e-12)


# ---------------------------------------------------------------------------
# fisher_compare_correlations
# ---------------------------------------------------------------------------

class TestFisherCompareCorrelations:
    def test_returns_dataclass(self):
        result = fisher_compare_correlations(0.3, 30, -0.5, 30)
        assert isinstance(result, FisherComparison)

    def test_required_fields(self):
        result = fisher_compare_correlations(0.3, 30, -0.5, 30)
        for field in ("z1", "z2", "se_diff", "z_stat", "p_two_sided", "n1", "n2"):
            assert hasattr(result, field)

    def test_equal_correlations_p_near_one(self):
        result = fisher_compare_correlations(0.5, 50, 0.5, 50)
        assert result.z_stat == pytest.approx(0.0, abs=1e-10)
        assert result.p_two_sided == pytest.approx(1.0, abs=1e-6)

    def test_very_different_correlations_small_p(self):
        # r1=0.0 n=100 vs r2=-0.9 n=100 → z_stat large → p small
        result = fisher_compare_correlations(0.0, 100, -0.9, 100)
        assert result.p_two_sided < 0.001

    def test_p_two_sided_between_0_and_1(self):
        result = fisher_compare_correlations(0.2, 20, -0.3, 20)
        assert 0.0 <= result.p_two_sided <= 1.0

    def test_se_diff_positive(self):
        result = fisher_compare_correlations(0.1, 20, -0.1, 20)
        assert result.se_diff > 0.0

    def test_n_stored_correctly(self):
        result = fisher_compare_correlations(0.1, 15, 0.2, 25)
        assert result.n1 == 15
        assert result.n2 == 25

    def test_raises_when_n_too_small(self):
        with pytest.raises(ValueError, match="n > 3"):
            fisher_compare_correlations(0.5, 3, 0.1, 30)

    def test_raises_when_n2_too_small(self):
        with pytest.raises(ValueError, match="n > 3"):
            fisher_compare_correlations(0.5, 30, 0.1, 2)

    def test_z1_z2_correct_sign(self):
        result = fisher_compare_correlations(0.7, 50, -0.7, 50)
        assert result.z1 > 0
        assert result.z2 < 0

    def test_z_stat_sign(self):
        # z1 > z2 → z_stat > 0
        result = fisher_compare_correlations(0.8, 50, 0.2, 50)
        assert result.z_stat > 0


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------

class TestCleanDataframe:
    def _df(self, **kwargs):
        base = {
            "logMbar": [9.5, 10.5, np.nan, 9.8],
            "env_proxy": [0.1, 0.5, 0.3, np.inf],
            "slope_tail": [-0.1, -0.3, -0.2, -0.4],
        }
        base.update(kwargs)
        return pd.DataFrame(base)

    def test_drops_nan_rows(self):
        df = self._df()
        out = clean_dataframe(df, "logMbar", "env_proxy", "slope_tail")
        assert out["logMbar"].isna().sum() == 0

    def test_drops_inf_rows(self):
        df = self._df()
        out = clean_dataframe(df, "logMbar", "env_proxy", "slope_tail")
        assert not np.isinf(out["env_proxy"]).any()

    def test_selects_only_needed_columns(self):
        df = self._df()
        df["extra"] = 99
        out = clean_dataframe(df, "logMbar", "env_proxy", "slope_tail")
        assert "extra" not in out.columns
        assert set(out.columns) == {"logMbar", "env_proxy", "slope_tail"}

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"logMbar": [9.5], "env_proxy": [0.1]})
        with pytest.raises(KeyError, match="slope_tail"):
            clean_dataframe(df, "logMbar", "env_proxy", "slope_tail")

    def test_raises_on_multiple_missing_columns(self):
        df = pd.DataFrame({"logMbar": [9.5]})
        with pytest.raises(KeyError):
            clean_dataframe(df, "logMbar", "env_proxy", "slope_tail")

    def test_all_clean_rows_preserved(self):
        df = pd.DataFrame({
            "logMbar": [9.0, 10.0, 11.0],
            "env_proxy": [0.1, 0.2, 0.3],
            "slope_tail": [-0.1, -0.2, -0.3],
        })
        out = clean_dataframe(df, "logMbar", "env_proxy", "slope_tail")
        assert len(out) == 3


# ---------------------------------------------------------------------------
# spearman_stats
# ---------------------------------------------------------------------------

class TestSpearmanStats:
    def _df(self, n=30, rho=-0.7, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        y = rho * x + math.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.normal(0, 1, n)
        return pd.DataFrame({"env_proxy": x, "slope_tail": y})

    def test_returns_regime_stats(self):
        df = self._df()
        result = spearman_stats(df, "env_proxy", "slope_tail", "high", "m>10")
        assert isinstance(result, RegimeStats)

    def test_negative_rho_planted(self):
        df = self._df(n=60, rho=-0.8)
        result = spearman_stats(df, "env_proxy", "slope_tail", "high", "m>10")
        assert result.rho_spearman < -0.3

    def test_p_value_in_range(self):
        df = self._df(n=30)
        result = spearman_stats(df, "env_proxy", "slope_tail", "high", "m>10")
        assert 0.0 <= result.p_value <= 1.0

    def test_nan_when_too_few_rows(self):
        df = pd.DataFrame({"env_proxy": [0.1], "slope_tail": [-0.1]})
        result = spearman_stats(df, "env_proxy", "slope_tail", "high", "m>10")
        assert math.isnan(result.rho_spearman)

    def test_n_matches_df_length(self):
        df = self._df(n=25)
        result = spearman_stats(df, "env_proxy", "slope_tail", "low", "m<10")
        assert result.n == 25

    def test_label_stored(self):
        df = self._df()
        result = spearman_stats(df, "env_proxy", "slope_tail", "my_label", "rule")
        assert result.label == "my_label"
        assert result.threshold_rule == "rule"


# ---------------------------------------------------------------------------
# bootstrap_spearman
# ---------------------------------------------------------------------------

class TestBootstrapSpearman:
    def _high_df(self, n=50, rho=-0.7, seed=7):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        noise = rng.normal(0, 1, n)
        y = rho * x + math.sqrt(max(1.0 - rho ** 2, 0.0)) * noise
        return pd.DataFrame({"env_proxy": x, "slope_tail": y})

    def test_returns_bootstrap_summary(self):
        df = self._high_df()
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=100, seed=0)
        assert isinstance(result, BootstrapSummary)

    def test_rho_observed_matches_direct_spearman(self):
        from scipy.stats import spearmanr
        df = self._high_df(n=40)
        rho_direct, _ = spearmanr(df["env_proxy"], df["slope_tail"])
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=100, seed=0)
        assert result.rho_observed == pytest.approx(rho_direct, abs=1e-10)

    def test_ci_contains_observed_rho(self):
        df = self._high_df(n=60)
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=500, seed=1)
        assert result.ci95_low <= result.rho_observed <= result.ci95_high

    def test_ci_low_le_ci_high(self):
        df = self._high_df(n=30)
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=200, seed=2)
        assert result.ci95_low <= result.ci95_high

    def test_frac_negative_in_range(self):
        df = self._high_df(n=50, rho=-0.7)
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=500, seed=3)
        assert 0.0 <= result.frac_negative <= 1.0

    def test_strong_negative_high_frac_negative(self):
        df = self._high_df(n=60, rho=-0.95)
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=300, seed=4)
        assert result.frac_negative > 0.9

    def test_n_boot_stored(self):
        df = self._high_df()
        result = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=123, seed=0)
        assert result.n_boot == 123

    def test_reproducible_with_same_seed(self):
        df = self._high_df()
        r1 = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=100, seed=77)
        r2 = bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=100, seed=77)
        assert r1.ci95_low == r2.ci95_low
        assert r1.ci95_high == r2.ci95_high

    def test_raises_when_too_few_rows(self):
        df = pd.DataFrame({"env_proxy": [0.1, 0.2], "slope_tail": [-0.1, -0.2]})
        with pytest.raises(ValueError, match="at least 3"):
            bootstrap_spearman(df, "env_proxy", "slope_tail", n_boot=10, seed=0)


# ---------------------------------------------------------------------------
# run_mass_scan
# ---------------------------------------------------------------------------

class TestRunMassScan:
    def _df(self, n=80, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "logMbar": rng.uniform(9.0, 11.0, n),
            "env_proxy": rng.normal(0, 1, n),
            "slope_tail": rng.normal(-0.2, 0.1, n),
        })

    def test_returns_dataframe(self):
        df = self._df()
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.5, 10.5, 0.1, 5)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = self._df()
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.5, 10.5, 0.1, 5)
        for col in ("mass_cut", "rho_high", "p_high", "n_high",
                    "rho_low", "p_low", "n_low"):
            assert col in result.columns

    def test_mass_cut_covers_range(self):
        df = self._df()
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.5, 10.5, 0.1, 1)
        assert result["mass_cut"].min() == pytest.approx(9.5, abs=0.001)
        assert result["mass_cut"].max() == pytest.approx(10.5, abs=0.001)

    def test_nan_when_below_min_n(self):
        df = self._df(n=20)
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.0, 11.0, 0.5, min_n=1000)
        assert result["rho_high"].isna().all()
        assert result["rho_low"].isna().all()

    def test_minus_log10_p_nonnegative(self):
        df = self._df(n=100)
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.5, 10.5, 0.1, 5)
        valid = result["minus_log10_p_high"].dropna()
        assert (valid >= 0).all()

    def test_rho_high_in_range(self):
        df = self._df(n=100)
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.5, 10.5, 0.1, 5)
        valid = result["rho_high"].dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_n_high_plus_n_low_le_total(self):
        df = self._df(n=80)
        result = run_mass_scan(df, "logMbar", "env_proxy", "slope_tail",
                               9.5, 10.5, 0.1, 1)
        for _, row in result.iterrows():
            assert row["n_high"] + row["n_low"] <= len(df)


# ---------------------------------------------------------------------------
# run_hc3_ols
# ---------------------------------------------------------------------------

class TestRunHC3OLS:
    def _df(self, n=40, beta=-0.4, seed=0, noise_std=0.5):
        """noise_std controls OLS residual spread; 0.5 gives a clear but noisy signal."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        y = beta * x + rng.normal(0, noise_std, n)
        return pd.DataFrame({"env_proxy": x, "slope_tail": y})

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_returns_dict_with_available_true(self):
        df = self._df()
        result = run_hc3_ols(df, "env_proxy", "slope_tail")
        assert result["available"] is True

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_negative_beta_planted(self):
        df = self._df(n=60, beta=-0.5)
        result = run_hc3_ols(df, "env_proxy", "slope_tail")
        assert result["beta_env"] < 0

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_r2_in_range(self):
        df = self._df(n=50)
        result = run_hc3_ols(df, "env_proxy", "slope_tail")
        assert 0.0 <= result["r2"] <= 1.0

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_n_stored_correctly(self):
        df = self._df(n=33)
        result = run_hc3_ols(df, "env_proxy", "slope_tail")
        assert result["n"] == 33

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_p_value_in_range(self):
        df = self._df(n=40)
        result = run_hc3_ols(df, "env_proxy", "slope_tail")
        assert 0.0 <= result["beta_env_p_hc3"] <= 1.0

    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_se_hc3_positive(self):
        df = self._df(n=40)
        result = run_hc3_ols(df, "env_proxy", "slope_tail")
        assert result["beta_env_se_hc3"] > 0


# ---------------------------------------------------------------------------
# make_plots
# ---------------------------------------------------------------------------

class TestMakePlots:
    def _scan_df(self):
        return pd.DataFrame({
            "mass_cut": np.arange(9.5, 10.6, 0.1),
            "rho_high": np.linspace(-0.3, -0.7, 11),
            "p_high": np.linspace(0.01, 0.2, 11),
            "minus_log10_p_high": np.linspace(0.7, 2.0, 11),
            "n_high": [15] * 11,
            "rho_low": np.linspace(-0.1, 0.1, 11),
            "p_low": np.linspace(0.3, 0.9, 11),
            "n_low": [12] * 11,
        })

    def test_creates_png_files(self, tmp_path):
        scan_df = self._scan_df()
        make_plots(scan_df, 10.0, 10.1, str(tmp_path))
        assert (tmp_path / "mass_scan_rho_high.png").exists()
        assert (tmp_path / "mass_scan_logp_high.png").exists()

    def test_creates_pdf_files(self, tmp_path):
        scan_df = self._scan_df()
        make_plots(scan_df, 10.0, 10.1, str(tmp_path))
        assert (tmp_path / "mass_scan_rho_high.pdf").exists()
        assert (tmp_path / "mass_scan_logp_high.pdf").exists()

    def test_handles_all_nan_rho(self, tmp_path):
        scan_df = self._scan_df()
        scan_df["rho_high"] = np.nan
        scan_df["minus_log10_p_high"] = np.nan
        make_plots(scan_df, 10.0, 10.1, str(tmp_path))
        assert (tmp_path / "mass_scan_rho_high.png").exists()


# ---------------------------------------------------------------------------
# write_summary_text
# ---------------------------------------------------------------------------

class TestWriteSummaryText:
    def _summary(self):
        return {
            "input_csv": "data/cat.csv",
            "n_total_clean": 80,
            "columns": {"mass_col": "logMbar", "env_col": "env_proxy", "slope_col": "slope_tail"},
            "cuts": {"low_cut": 10.0, "high_cut": 10.1},
            "low_regime": {
                "label": "low_mass",
                "threshold_rule": "logMbar < 10.0",
                "n": 40,
                "rho_spearman": 0.05,
                "p_value": 0.72,
            },
            "high_regime": {
                "label": "high_mass",
                "threshold_rule": "logMbar > 10.1",
                "n": 40,
                "rho_spearman": -0.48,
                "p_value": 0.002,
            },
            "fisher_comparison": {
                "z1": 0.05, "z2": -0.52, "se_diff": 0.22,
                "z_stat": 2.59, "p_two_sided": 0.0096, "n1": 40, "n2": 40,
            },
            "bootstrap_high": {
                "n_boot": 1000, "rho_observed": -0.48,
                "ci95_low": -0.65, "ci95_high": -0.28, "frac_negative": 0.98,
            },
            "hc3_high": {
                "available": True, "n": 40,
                "beta_env": -0.31, "beta_env_se_hc3": 0.08,
                "beta_env_t_hc3": -3.9, "beta_env_p_hc3": 0.0003,
                "r2": 0.22, "adj_r2": 0.20,
            },
            "scan_csv": "results/scm_tr/mass_scan.csv",
        }

    def test_creates_file(self, tmp_path):
        out = tmp_path / "summary.txt"
        write_summary_text(self._summary(), str(out))
        assert out.exists()

    def test_contains_spearman_values(self, tmp_path):
        out = tmp_path / "summary.txt"
        write_summary_text(self._summary(), str(out))
        text = out.read_text()
        assert "Spearman rho" in text

    def test_contains_fisher_section(self, tmp_path):
        out = tmp_path / "summary.txt"
        write_summary_text(self._summary(), str(out))
        text = out.read_text()
        assert "Fisher" in text

    def test_contains_bootstrap_section(self, tmp_path):
        out = tmp_path / "summary.txt"
        write_summary_text(self._summary(), str(out))
        text = out.read_text()
        assert "Bootstrap" in text

    def test_contains_hc3_section(self, tmp_path):
        out = tmp_path / "summary.txt"
        write_summary_text(self._summary(), str(out))
        text = out.read_text()
        assert "HC3" in text

    def test_skips_hc3_when_unavailable(self, tmp_path):
        summary = self._summary()
        summary["hc3_high"] = {"available": False}
        out = tmp_path / "summary.txt"
        write_summary_text(summary, str(out))
        text = out.read_text()
        assert "HC3" not in text

    def test_skips_fisher_when_none(self, tmp_path):
        summary = self._summary()
        summary["fisher_comparison"] = None
        out = tmp_path / "summary.txt"
        write_summary_text(summary, str(out))
        text = out.read_text()
        assert "Fisher" not in text


# ---------------------------------------------------------------------------
# save_json / ensure_dir
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_save_json_creates_file(self, tmp_path):
        p = str(tmp_path / "out.json")
        save_json({"a": 1, "b": [1, 2]}, p)
        assert os.path.exists(p)

    def test_save_json_roundtrip(self, tmp_path):
        data = {"x": 3.14, "y": [1, 2, 3], "z": "hello"}
        p = str(tmp_path / "data.json")
        save_json(data, p)
        loaded = json.loads(Path(p).read_text())
        assert loaded == data

    def test_ensure_dir_creates_directory(self, tmp_path):
        new_dir = str(tmp_path / "a" / "b" / "c")
        ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_ensure_dir_idempotent(self, tmp_path):
        new_dir = str(tmp_path / "mydir")
        ensure_dir(new_dir)
        ensure_dir(new_dir)  # should not raise


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        csv = _make_catalog(tmp_path)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--bootstrap", "50",
        ])
        assert isinstance(result, dict)

    def test_creates_json_output(self, tmp_path):
        csv = _make_catalog(tmp_path)
        outdir = tmp_path / "out"
        main([
            "--input", str(csv),
            "--outdir", str(outdir),
            "--bootstrap", "50",
        ])
        assert (outdir / "scm_tr_summary.json").exists()

    def test_creates_txt_output(self, tmp_path):
        csv = _make_catalog(tmp_path)
        outdir = tmp_path / "out"
        main([
            "--input", str(csv),
            "--outdir", str(outdir),
            "--bootstrap", "50",
        ])
        assert (outdir / "scm_tr_summary.txt").exists()

    def test_creates_scan_csv(self, tmp_path):
        csv = _make_catalog(tmp_path)
        outdir = tmp_path / "out"
        main([
            "--input", str(csv),
            "--outdir", str(outdir),
            "--bootstrap", "50",
        ])
        assert (outdir / "mass_scan.csv").exists()

    def test_creates_plot_files(self, tmp_path):
        csv = _make_catalog(tmp_path)
        outdir = tmp_path / "out"
        main([
            "--input", str(csv),
            "--outdir", str(outdir),
            "--bootstrap", "50",
        ])
        assert (outdir / "mass_scan_rho_high.png").exists()
        assert (outdir / "mass_scan_logp_high.png").exists()

    def test_n_total_clean_correct(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=20, n_high=20)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--bootstrap", "50",
        ])
        assert result["n_total_clean"] == 40

    def test_cuts_stored_in_result(self, tmp_path):
        csv = _make_catalog(tmp_path)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--low-cut", "9.9",
            "--high-cut", "10.2",
            "--bootstrap", "50",
        ])
        assert result["cuts"]["low_cut"] == 9.9
        assert result["cuts"]["high_cut"] == 10.2

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main([
                "--input", str(tmp_path / "no_such.csv"),
                "--outdir", str(tmp_path / "out"),
            ])

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"logMbar": [9.5, 10.5], "wrong": [1, 2]}).to_csv(bad, index=False)
        with pytest.raises(KeyError):
            main([
                "--input", str(bad),
                "--outdir", str(tmp_path / "out"),
            ])

    def test_fisher_comparison_present_when_enough_data(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=30, n_high=30)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--bootstrap", "50",
        ])
        assert result["fisher_comparison"] is not None

    def test_bootstrap_high_present(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=30, n_high=30)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--bootstrap", "100",
        ])
        assert result["bootstrap_high"] is not None

    def test_hc3_present_when_run_hc3_flag(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=30, n_high=30)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--bootstrap", "50",
            "--run-hc3",
        ])
        assert result["hc3_high"] is not None

    def test_hc3_none_without_flag(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=30, n_high=30)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--bootstrap", "50",
        ])
        assert result["hc3_high"] is None

    def test_custom_column_names(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "M": rng.uniform(9.0, 11.0, 60),
            "env": rng.normal(0, 1, 60),
            "slope": rng.normal(-0.2, 0.1, 60),
        })
        csv = tmp_path / "custom.csv"
        df.to_csv(csv, index=False)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--mass-col", "M",
            "--env-col", "env",
            "--slope-col", "slope",
            "--bootstrap", "50",
        ])
        assert result["columns"]["mass_col"] == "M"

    def test_scan_csv_is_valid(self, tmp_path):
        csv = _make_catalog(tmp_path)
        outdir = tmp_path / "out"
        main([
            "--input", str(csv),
            "--outdir", str(outdir),
            "--bootstrap", "50",
        ])
        scan = pd.read_csv(outdir / "mass_scan.csv")
        assert "mass_cut" in scan.columns
        assert "rho_high" in scan.columns

    def test_regime_n_counts_are_positive(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=20, n_high=20)
        result = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out"),
            "--low-cut", "9.8",
            "--high-cut", "10.2",
            "--bootstrap", "50",
        ])
        assert result["low_regime"]["n"] >= 0
        assert result["high_regime"]["n"] >= 0

    def test_seed_reproducibility(self, tmp_path):
        csv = _make_catalog(tmp_path, n_low=30, n_high=30)
        r1 = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out1"),
            "--bootstrap", "200",
            "--seed", "42",
        ])
        r2 = main([
            "--input", str(csv),
            "--outdir", str(tmp_path / "out2"),
            "--bootstrap", "200",
            "--seed", "42",
        ])
        assert r1["bootstrap_high"]["ci95_low"] == r2["bootstrap_high"]["ci95_low"]
