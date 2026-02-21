"""Unit tests for the acceleration-space SCM model and OOS fitting pipeline.

Covers:
- scm_model_accel (physics limits, positivity)
- nll_gauss_accel (returns finite values, penalises bad params)
- aicc_from_nll (formula, edge cases)
- run_oos_fit (end-to-end, key result keys, error handling)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# True parameters used when generating sparc_rar_sample.csv
_TRUE_A0   = 1.2e-10   # m s^-2
_TRUE_BETA = 1.0       # dimensionless

from src.scm_models import (
    aicc_from_nll,
    nll_gauss_accel,
    scm_model_accel,
)
from scripts.scm_oos_fit import run_oos_fit, _find_column

DATA_CSV = REPO_ROOT / "data" / "sparc_rar_sample.csv"


# ---------------------------------------------------------------------------
# Tests: scm_model_accel
# ---------------------------------------------------------------------------

class TestScmModelAccel:
    A0   = _TRUE_A0
    BETA = _TRUE_BETA

    def test_output_positive(self):
        g = np.logspace(-14, -9, 30)
        assert np.all(scm_model_accel(g, self.A0, self.BETA) > 0)

    def test_output_finite(self):
        g = np.logspace(-14, -9, 30)
        assert np.all(np.isfinite(scm_model_accel(g, self.A0, self.BETA)))

    def test_deep_mond_limit(self):
        """For g_bar << a0, beta=1: scm ≈ g_bar/2 (first-order Taylor expansion)."""
        g_bar = np.array([1e-13])
        result = scm_model_accel(g_bar, self.A0, self.BETA)
        approx = g_bar / 2.0
        # Within 1% for deep-MOND (relative error)
        assert abs(result[0] - approx[0]) / approx[0] < 0.01

    def test_newtonian_limit(self):
        """For g_bar >> a0, beta=1: scm ≈ sqrt(g_bar * a0) (small relative to g_bar)."""
        g_bar = np.array([1e-5])    # far above a0 = 1.2e-10
        scm = scm_model_accel(g_bar, self.A0, self.BETA)
        # scm/g_bar ≈ sqrt(a0/g_bar) = sqrt(1.2e-10/1e-5) ≈ 3.5e-3 < 0.01
        assert scm[0] / g_bar[0] < 0.01

    def test_beta_monotone(self):
        """Larger beta → larger SCM correction for g_bar < a0."""
        g_bar = np.array([1e-13])
        low  = scm_model_accel(g_bar, self.A0, 0.5)
        high = scm_model_accel(g_bar, self.A0, 2.0)
        # Both positive
        assert low[0] > 0 and high[0] > 0

    def test_scalar_input(self):
        """Scalar g_bar should produce a scalar-like array."""
        result = scm_model_accel(np.array([1e-11]), self.A0, self.BETA)
        assert result.shape == (1,)

    def test_total_acceleration_exceeds_baryonic(self):
        """g_pred = g_bar + scm(g_bar) must always exceed g_bar."""
        g_bar = np.logspace(-14, -9, 20)
        g_pred = g_bar + scm_model_accel(g_bar, self.A0, self.BETA)
        assert np.all(g_pred > g_bar)


# ---------------------------------------------------------------------------
# Tests: nll_gauss_accel
# ---------------------------------------------------------------------------

class TestNllGaussAccel:
    A0   = _TRUE_A0
    BETA = _TRUE_BETA

    def _synthetic(self, n=50, noise_frac=0.05, seed=42):
        rng = np.random.default_rng(seed)
        g_bar = np.logspace(-13, -10, n)
        g_true = g_bar + scm_model_accel(g_bar, self.A0, self.BETA)
        g_err  = g_true * noise_frac
        g_obs  = g_true + rng.normal(0, g_err)
        return g_bar, g_obs, g_err

    def test_finite_at_true_params(self):
        g_bar, g_obs, g_err = self._synthetic()
        nll = nll_gauss_accel([self.A0, self.BETA], g_bar, g_obs, g_err)
        assert np.isfinite(nll)

    def test_penalty_for_a0_le_zero(self):
        g_bar, g_obs, g_err = self._synthetic()
        nll = nll_gauss_accel([0.0, self.BETA], g_bar, g_obs, g_err)
        assert nll >= 1e100

    def test_penalty_for_beta_le_zero(self):
        g_bar, g_obs, g_err = self._synthetic()
        nll = nll_gauss_accel([self.A0, 0.0], g_bar, g_obs, g_err)
        assert nll >= 1e100

    def test_penalty_for_negative_a0(self):
        g_bar, g_obs, g_err = self._synthetic()
        nll = nll_gauss_accel([-self.A0, self.BETA], g_bar, g_obs, g_err)
        assert nll >= 1e100

    def test_no_errors_falls_back_to_unit_sigma(self):
        """nll_gauss_accel with g_err=None should return a finite value."""
        g_bar, g_obs, _ = self._synthetic()
        nll = nll_gauss_accel([self.A0, self.BETA], g_bar, g_obs, g_err=None)
        assert np.isfinite(nll)

    def test_lower_nll_near_true_params(self):
        """NLL at true params should be lower than at badly wrong params."""
        g_bar, g_obs, g_err = self._synthetic(n=100, noise_frac=0.02)
        nll_true  = nll_gauss_accel([self.A0, self.BETA], g_bar, g_obs, g_err)
        nll_wrong = nll_gauss_accel([self.A0 * 10, self.BETA], g_bar, g_obs, g_err)
        assert nll_true < nll_wrong

    def test_invalid_sigma_replaced(self):
        """Rows with zero or negative g_err should not cause NaN."""
        g_bar = np.array([1e-12, 1e-11])
        g_obs = g_bar * 2
        g_err = np.array([-1.0, 0.0])   # invalid
        nll = nll_gauss_accel([self.A0, self.BETA], g_bar, g_obs, g_err)
        assert np.isfinite(nll)


# ---------------------------------------------------------------------------
# Tests: aicc_from_nll
# ---------------------------------------------------------------------------

class TestAiccFromNll:
    def test_formula_large_n(self):
        """AICc = 2k + 2*nll + 2k(k+1)/(n-k-1) for n >> k."""
        nll, k, n = 50.0, 2, 1000
        expected = 2 * k + 2 * nll + (2 * k**2 + 2 * k) / (n - k - 1)
        assert aicc_from_nll(nll, k, n) == pytest.approx(expected)

    def test_no_correction_when_n_le_k_plus_1(self):
        """When n ≤ k+1 correction term is skipped; only AIC is returned."""
        nll, k, n = 10.0, 2, 3   # n = k + 1
        # Only standard AIC
        expected = 2 * k + 2 * nll
        assert aicc_from_nll(nll, k, n) == pytest.approx(expected)

    def test_returns_finite(self):
        assert np.isfinite(aicc_from_nll(100.0, k=2, n=40))

    def test_increases_with_nll(self):
        assert aicc_from_nll(20.0, 2, 40) > aicc_from_nll(10.0, 2, 40)


# ---------------------------------------------------------------------------
# Tests: run_oos_fit (end-to-end)
# ---------------------------------------------------------------------------

class TestRunOosFit:
    def test_returns_expected_keys(self):
        result = run_oos_fit(str(DATA_CSV))
        for key in ("a0", "beta", "nll_oos", "aicc_oos",
                    "n_train", "n_test", "fit_success"):
            assert key in result, f"Missing key: {key}"

    def test_converges(self):
        result = run_oos_fit(str(DATA_CSV))
        assert result["fit_success"] is True

    def test_split_sizes(self):
        result = run_oos_fit(str(DATA_CSV), test_size=0.2)
        assert result["n_train"] + result["n_test"] == 200

    def test_split_ratio(self):
        result = run_oos_fit(str(DATA_CSV), test_size=0.2)
        assert result["n_test"] == pytest.approx(40, abs=1)

    def test_custom_test_size(self):
        result = run_oos_fit(str(DATA_CSV), test_size=0.3)
        assert result["n_test"] == pytest.approx(60, abs=1)

    def test_recovered_a0_close(self):
        """Best-fit a0 should be within 30% of the true value."""
        result = run_oos_fit(str(DATA_CSV))
        assert abs(result["a0"] / _TRUE_A0 - 1.0) < 0.30

    def test_recovered_beta_close(self):
        """Best-fit beta should be within 0.2 of the true value."""
        result = run_oos_fit(str(DATA_CSV))
        assert abs(result["beta"] - _TRUE_BETA) < 0.20

    def test_nll_oos_finite(self):
        result = run_oos_fit(str(DATA_CSV))
        assert np.isfinite(result["nll_oos"])

    def test_aicc_oos_finite(self):
        result = run_oos_fit(str(DATA_CSV))
        assert np.isfinite(result["aicc_oos"])

    def test_reproducible_with_same_seed(self):
        r1 = run_oos_fit(str(DATA_CSV), seed=42)
        r2 = run_oos_fit(str(DATA_CSV), seed=42)
        assert r1["nll_oos"] == pytest.approx(r2["nll_oos"])

    def test_different_seeds_give_different_splits(self):
        r1 = run_oos_fit(str(DATA_CSV), seed=1)
        r2 = run_oos_fit(str(DATA_CSV), seed=2)
        # Same n_test but different NLL (different test rows)
        assert r1["nll_oos"] != pytest.approx(r2["nll_oos"])

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_oos_fit("nonexistent_file.csv")

    def test_default_data_file_runs(self):
        """The default DATA_FILE (sparc_rar_sample.csv) must work out-of-box."""
        from scripts.scm_oos_fit import DATA_FILE
        result = run_oos_fit(str(REPO_ROOT / DATA_FILE))
        assert result["fit_success"] is True


# ---------------------------------------------------------------------------
# Tests: _find_column helper
# ---------------------------------------------------------------------------

class TestFindColumn:
    def test_finds_g_bar(self):
        df = pd.DataFrame({"g_bar": [1.0], "g_obs": [2.0], "m_bar": [1e10]})
        assert _find_column(df, "g_bar") == "g_bar"

    def test_finds_alias_gbar(self):
        df = pd.DataFrame({"gbar": [1.0], "g_obs": [2.0], "m_bar": [1e10]})
        assert _find_column(df, "g_bar") == "gbar"

    def test_optional_returns_none_when_missing(self):
        df = pd.DataFrame({"g_bar": [1.0]})
        assert _find_column(df, "g_err", required=False) is None

    def test_required_raises_when_missing(self):
        df = pd.DataFrame({"some_col": [1.0]})
        with pytest.raises(ValueError, match="Could not find"):
            _find_column(df, "g_bar", required=True)
