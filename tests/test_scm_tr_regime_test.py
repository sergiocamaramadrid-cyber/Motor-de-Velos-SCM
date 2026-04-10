"""
tests/test_scm_tr_regime_test.py — Tests for scripts/scm_tr_regime_test.py.

Creates synthetic SPARC-like data to verify:
- Fisher Z-transform and comparison functions
- Bootstrap Spearman routine
- Mass scan and HC3 OLS helpers
- main() end-to-end output
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.scm_tr_regime_test import (
    LOGM_THRESHOLD_DEFAULT,
    MASS_COL,
    ENV_COL,
    SLOPE_COL,
    N_BOOT_DEFAULT,
    FisherComparison,
    BootstrapSummary,
    fisher_z_from_r,
    fisher_compare_correlations,
    bootstrap_spearman,
    run_mass_scan,
    run_hc3_ols,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparc_df(n_low: int = 22, n_high: int = 54, seed: int = 7) -> pd.DataFrame:
    """Synthetic SPARC-like dataframe with correlated high-mass regime."""
    rng = np.random.default_rng(seed)
    logM_lo = rng.uniform(8.5, 10.0, n_low)
    logM_hi = rng.uniform(10.1, 11.5, n_high)
    env_lo  = rng.uniform(-1, 1, n_low)
    env_hi  = rng.normal(0, 1, n_high)
    sl_lo   = rng.normal(-0.15, 0.2, n_low)
    sl_hi   = -0.4 * env_hi + rng.normal(-0.15, 0.35, n_high)

    return pd.DataFrame({
        "galaxy":     [f"G{i:03d}" for i in range(n_low + n_high)],
        MASS_COL:     np.concatenate([logM_lo, logM_hi]),
        ENV_COL:      np.concatenate([env_lo, env_hi]),
        SLOPE_COL:    np.concatenate([sl_lo, sl_hi]),
    })


def _make_csv(tmp_path: Path, df: pd.DataFrame, name: str = "sparc_env.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# fisher_z_from_r
# ---------------------------------------------------------------------------

class TestFisherZ:
    def test_zero_returns_zero(self):
        assert fisher_z_from_r(0.0) == pytest.approx(0.0)

    def test_half_approx(self):
        assert fisher_z_from_r(0.5) == pytest.approx(math.atanh(0.5), rel=1e-6)

    def test_positive_r_positive_z(self):
        assert fisher_z_from_r(0.8) > 0

    def test_negative_r_negative_z(self):
        assert fisher_z_from_r(-0.5) < 0

    def test_symmetry(self):
        assert fisher_z_from_r(0.3) == pytest.approx(-fisher_z_from_r(-0.3), rel=1e-9)

    def test_known_value_0_9(self):
        assert fisher_z_from_r(0.9) == pytest.approx(math.atanh(0.9), rel=1e-6)

    def test_extreme_near_one(self):
        z = fisher_z_from_r(0.999)
        assert z > 3.0


# ---------------------------------------------------------------------------
# fisher_compare_correlations
# ---------------------------------------------------------------------------

class TestFisherCompare:
    def _compare(self, r1=0.5, n1=50, r2=-0.5, n2=50):
        return fisher_compare_correlations(r1, n1, r2, n2)

    def test_returns_named_tuple(self):
        result = self._compare()
        assert isinstance(result, FisherComparison)

    def test_has_required_fields(self):
        result = self._compare()
        assert hasattr(result, "rho1")
        assert hasattr(result, "n1")
        assert hasattr(result, "rho2")
        assert hasattr(result, "n2")
        assert hasattr(result, "z_stat")
        assert hasattr(result, "p_two_tail")

    def test_z_stat_is_float(self):
        result = self._compare()
        assert isinstance(result.z_stat, float)

    def test_p_in_unit_interval(self):
        result = self._compare()
        assert 0 <= result.p_two_tail <= 1

    def test_identical_correlations_pval_near_one(self):
        result = fisher_compare_correlations(0.4, 60, 0.4, 60)
        assert result.p_two_tail > 0.9

    def test_opposite_correlations_significant(self):
        result = fisher_compare_correlations(0.6, 80, -0.6, 80)
        assert result.p_two_tail < 0.05

    def test_stored_inputs(self):
        result = fisher_compare_correlations(0.3, 40, -0.3, 40)
        assert result.rho1 == pytest.approx(0.3)
        assert result.n1 == 40


# ---------------------------------------------------------------------------
# bootstrap_spearman
# ---------------------------------------------------------------------------

class TestBootstrapSpearman:
    def _boot(self, seed=99, n=60):
        rng = np.random.default_rng(seed)
        env  = rng.normal(0, 1, n)
        slope = -0.5 * env + rng.normal(0, 0.3, n)
        df = pd.DataFrame({ENV_COL: env, SLOPE_COL: slope})
        return bootstrap_spearman(df, ENV_COL, SLOPE_COL, n_boot=200, seed=0)

    def test_returns_named_tuple(self):
        result = self._boot()
        assert isinstance(result, BootstrapSummary)

    def test_median_is_float(self):
        assert isinstance(self._boot().median, float)

    def test_ci_ordering(self):
        result = self._boot()
        assert result.ci_lo <= result.median <= result.ci_hi

    def test_n_boot_stored(self):
        result = self._boot()
        assert result.n_boot == 200

    def test_negative_signal_negative_median(self):
        result = self._boot()
        assert result.median < 0

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(5)
        env = rng.normal(0, 1, 50)
        sl  = -0.4 * env + rng.normal(0, 0.3, 50)
        df = pd.DataFrame({ENV_COL: env, SLOPE_COL: sl})
        r1 = bootstrap_spearman(df, ENV_COL, SLOPE_COL, n_boot=100, seed=7)
        r2 = bootstrap_spearman(df, ENV_COL, SLOPE_COL, n_boot=100, seed=7)
        assert r1.median == pytest.approx(r2.median)


# ---------------------------------------------------------------------------
# run_mass_scan
# ---------------------------------------------------------------------------

class TestRunMassScan:
    def _df(self):
        return _make_sparc_df()

    def test_returns_dataframe(self):
        df = self._df()
        result = run_mass_scan(df, MASS_COL, ENV_COL, SLOPE_COL, scan_min=9.0,
                               scan_max=11.0, scan_step=0.5, min_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        df = self._df()
        result = run_mass_scan(df, MASS_COL, ENV_COL, SLOPE_COL)
        for col in ["threshold", "n_high", "rho_high", "pval_high"]:
            assert col in result.columns

    def test_pval_in_unit_interval(self):
        df = self._df()
        result = run_mass_scan(df, MASS_COL, ENV_COL, SLOPE_COL)
        assert (result["pval_high"] >= 0).all() and (result["pval_high"] <= 1).all()

    def test_rho_in_valid_range(self):
        df = self._df()
        result = run_mass_scan(df, MASS_COL, ENV_COL, SLOPE_COL)
        assert (result["rho_high"] >= -1).all() and (result["rho_high"] <= 1).all()

    def test_threshold_monotone(self):
        df = self._df()
        result = run_mass_scan(df, MASS_COL, ENV_COL, SLOPE_COL,
                               scan_min=9.0, scan_max=11.0, scan_step=0.2, min_n=5)
        thresholds = result["threshold"].values
        assert all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1))

    def test_min_n_filter(self):
        df = self._df()
        result = run_mass_scan(df, MASS_COL, ENV_COL, SLOPE_COL, min_n=30)
        assert (result["n_high"] >= 30).all()


# ---------------------------------------------------------------------------
# run_hc3_ols
# ---------------------------------------------------------------------------

class TestRunHC3OLS:
    def _df(self, seed=12):
        rng = np.random.default_rng(seed)
        env   = rng.normal(0, 1, 60)
        slope = -0.3 * env + rng.normal(0, 0.2, 60)
        return pd.DataFrame({ENV_COL: env, SLOPE_COL: slope})

    def test_returns_dict(self):
        assert isinstance(run_hc3_ols(self._df(), ENV_COL, SLOPE_COL), dict)

    def test_has_required_keys(self):
        result = run_hc3_ols(self._df(), ENV_COL, SLOPE_COL)
        for key in ["slope", "intercept", "slope_pval", "r2"]:
            assert key in result

    def test_slope_negative(self):
        result = run_hc3_ols(self._df(), ENV_COL, SLOPE_COL)
        assert result["slope"] < 0

    def test_r2_in_unit_interval(self):
        result = run_hc3_ols(self._df(), ENV_COL, SLOPE_COL)
        assert 0 <= result["r2"] <= 1

    def test_slope_pval_significant(self):
        result = run_hc3_ols(self._df(), ENV_COL, SLOPE_COL)
        assert result["slope_pval"] < 0.05


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_creates_output_file(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        assert (tmp_path / "scm_tr_summary.csv").exists()

    def test_returns_dict_with_required_keys(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        for key in ["low", "high", "bootstrap", "fisher"]:
            assert key in result

    def test_low_regime_keys(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        for key in ["rho", "pval", "n"]:
            assert key in result["low"]

    def test_high_regime_keys(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        for key in ["rho", "pval", "n"]:
            assert key in result["high"]

    def test_output_csv_has_correct_columns(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        out = pd.read_csv(tmp_path / "scm_tr_summary.csv")
        for col in ["regime", "n", "rho", "pval", "boot_median", "boot_ci_lo", "boot_ci_hi"]:
            assert col in out.columns

    def test_output_csv_has_two_rows(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        out = pd.read_csv(tmp_path / "scm_tr_summary.csv")
        assert len(out) == 2

    def test_output_regime_labels(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        out = pd.read_csv(tmp_path / "scm_tr_summary.csv")
        assert set(out["regime"].values) == {"low", "high"}

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv"), "--out", str(tmp_path)])

    def test_bootstrap_dict_keys(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        for key in ["median", "ci_lo", "ci_hi", "n_boot"]:
            assert key in result["bootstrap"]

    def test_custom_threshold(self, tmp_path):
        df = _make_sparc_df()
        csv = _make_csv(tmp_path, df)
        result = main([
            "--csv", str(csv), "--out", str(tmp_path),
            "--threshold", "10.5", "--n-boot", "50"
        ])
        assert result["low"]["n"] + result["high"]["n"] == len(df)
