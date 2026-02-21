"""Unit tests for gaussian_ll, RAR, NFW models, and model_comparison_oos.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.scm_models import (
    G_SI,
    aicc_from_nll,
    gaussian_ll,
    nfw_g_dm,
    nfw_model_accel,
    nll_nfw_accel,
    nll_rar_accel,
    rar_model_accel,
    scm_model_accel,
)
from scripts.model_comparison_oos import run_comparison

DATA_CSV = REPO_ROOT / "data" / "sparc_rar_sample.csv"

# True data-generation parameters
_A0   = 1.2e-10   # m s^-2
_BETA = 1.0


# ---------------------------------------------------------------------------
# Tests: gaussian_ll
# ---------------------------------------------------------------------------

class TestGaussianLl:
    def test_scalar_sigma_finite(self):
        y    = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 3.0])
        assert np.isfinite(gaussian_ll(y, yhat, sigma=1.0))

    def test_perfect_fit_is_more_likely_than_bad_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        ll_good = gaussian_ll(y, y,          sigma=1.0)
        ll_bad  = gaussian_ll(y, y + 10.0,   sigma=1.0)
        assert ll_good > ll_bad

    def test_smaller_sigma_penalises_same_residual_more(self):
        y    = np.array([0.0, 0.0])
        yhat = np.array([1.0, 1.0])
        ll_small = gaussian_ll(y, yhat, sigma=0.1)
        ll_large = gaussian_ll(y, yhat, sigma=10.0)
        assert ll_small < ll_large

    def test_vector_sigma_finite(self):
        y     = np.array([1.0, 2.0, 3.0])
        yhat  = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.1, 0.2, 0.3])
        assert np.isfinite(gaussian_ll(y, yhat, sigma))

    def test_normalisation_term_included(self):
        """With σ ≠ 1 the log-likelihood must differ from the -0.5*chi2 term."""
        y    = np.zeros(5)
        yhat = np.zeros(5)
        sigma = 2.0
        # -0.5*chi2 = 0; normalisation = -0.5*n*ln(2*pi*sigma^2)
        expected = -0.5 * 5 * np.log(2.0 * np.pi * sigma ** 2)
        assert gaussian_ll(y, yhat, sigma) == pytest.approx(expected)

    def test_consistency_with_nll(self):
        """gaussian_ll(y, yhat, s) == -nll_gauss_accel at true SCM prediction."""
        rng  = np.random.default_rng(0)
        g_bar = np.logspace(-13, -10, 30)
        g_obs = g_bar + scm_model_accel(g_bar, _A0, _BETA) + rng.normal(0, 1e-13, 30)
        g_err = np.full(30, 1e-13)
        g_pred = g_bar + scm_model_accel(g_bar, _A0, _BETA)
        from src.scm_models import nll_gauss_accel
        nll = nll_gauss_accel([_A0, _BETA], g_bar, g_obs, g_err)
        ll  = gaussian_ll(g_obs, g_pred, g_err)
        assert ll == pytest.approx(-nll, rel=1e-9)


# ---------------------------------------------------------------------------
# Tests: rar_model_accel
# ---------------------------------------------------------------------------

class TestRarModelAccel:
    def test_output_positive(self):
        g = np.logspace(-14, -9, 20)
        assert np.all(rar_model_accel(g, _A0) > 0)

    def test_output_exceeds_g_bar(self):
        """RAR prediction must be >= g_bar (dark sector contribution)."""
        g = np.logspace(-14, -9, 20)
        assert np.all(rar_model_accel(g, _A0) >= g)

    def test_newtonian_limit(self):
        """For g_bar >> g_dagger, g_obs ≈ g_bar."""
        g = np.array([1e-5])   # >> a0 = 1.2e-10
        ratio = rar_model_accel(g, _A0) / g
        assert abs(ratio[0] - 1.0) < 1e-4

    def test_output_finite(self):
        g = np.logspace(-14, -9, 20)
        assert np.all(np.isfinite(rar_model_accel(g, _A0)))

    def test_nll_penalises_negative_g_dagger(self):
        g = np.logspace(-13, -10, 10)
        assert nll_rar_accel([-_A0], g, g, None) >= 1e100

    def test_nll_finite_at_true_value(self):
        g_bar = np.logspace(-13, -10, 20)
        g_obs = rar_model_accel(g_bar, _A0)
        assert np.isfinite(nll_rar_accel([_A0], g_bar, g_obs, None))


# ---------------------------------------------------------------------------
# Tests: nfw_g_dm / nfw_model_accel
# ---------------------------------------------------------------------------

class TestNfwModelAccel:
    # Typical NFW parameters (SI)
    LOG10_RHO_S = -21.2   # rho_s ≈ 6.3e-22 kg/m^3
    LOG10_R_S   = 20.5    # r_s ≈ 3.16e20 m ≈ 10.2 kpc

    def _make_data(self, n=20):
        """Build synthetic arrays with r_eff from the CSV generation formula."""
        g_bar = np.logspace(-13.5, -9.0, n)
        r0    = 8.0 * 3.0857e19   # 8 kpc in m
        m_bar = g_bar * r0 ** 2 / G_SI
        return g_bar, m_bar

    def test_nfw_g_dm_positive(self):
        r = np.array([2.469e20])  # 8 kpc in m
        assert nfw_g_dm(r, self.LOG10_RHO_S, self.LOG10_R_S)[0] > 0

    def test_nfw_g_dm_finite(self):
        r = np.logspace(19, 22, 10)
        assert np.all(np.isfinite(nfw_g_dm(r, self.LOG10_RHO_S, self.LOG10_R_S)))

    def test_nfw_model_exceeds_g_bar(self):
        g_bar, m_bar = self._make_data()
        g_pred = nfw_model_accel(g_bar, m_bar, self.LOG10_RHO_S, self.LOG10_R_S)
        assert np.all(g_pred >= g_bar)

    def test_nfw_nll_finite(self):
        g_bar, m_bar = self._make_data()
        g_obs = nfw_model_accel(g_bar, m_bar, self.LOG10_RHO_S, self.LOG10_R_S)
        nll = nll_nfw_accel(
            [self.LOG10_RHO_S, self.LOG10_R_S],
            g_bar, g_obs, m_bar, None,
        )
        assert np.isfinite(nll)

    def test_nfw_nll_lower_at_exact_prediction(self):
        """NLL at the true NFW prediction should be lower than at bad params.

        Use realistic g_err (10 % of signal) so that the χ² residuals are
        numerically non-negligible relative to the ln(2πσ²) normalisation.
        """
        g_bar, m_bar = self._make_data()
        g_obs = nfw_model_accel(g_bar, m_bar, self.LOG10_RHO_S, self.LOG10_R_S)
        # Realistic uncertainties: 10 % of observed value
        g_err = 0.10 * g_obs
        nll_true = nll_nfw_accel(
            [self.LOG10_RHO_S, self.LOG10_R_S], g_bar, g_obs, m_bar, g_err)
        # rho_s three orders of magnitude smaller → effectively no DM
        nll_bad  = nll_nfw_accel(
            [self.LOG10_RHO_S - 3, self.LOG10_R_S], g_bar, g_obs, m_bar, g_err)
        assert nll_true < nll_bad


# ---------------------------------------------------------------------------
# Tests: run_comparison (end-to-end)
# ---------------------------------------------------------------------------

class TestRunComparison:
    def test_returns_dataframe(self):
        df = run_comparison(str(DATA_CSV))
        assert isinstance(df, pd.DataFrame)

    def test_three_rows(self):
        df = run_comparison(str(DATA_CSV))
        assert len(df) == 3

    def test_model_names(self):
        df = run_comparison(str(DATA_CSV))
        assert set(df["name"]) == {"SCM", "RAR", "NFW"}

    def test_required_columns(self):
        df = run_comparison(str(DATA_CSV))
        for col in ("name", "k", "ll_oos", "aicc_oos",
                    "delta_aicc_vs_scm", "n_train", "n_test",
                    "fit_success", "params"):
            assert col in df.columns, f"Missing column: {col}"

    def test_k_values(self):
        df = run_comparison(str(DATA_CSV))
        assert df[df["name"] == "SCM"]["k"].iloc[0] == 2
        assert df[df["name"] == "RAR"]["k"].iloc[0] == 1
        assert df[df["name"] == "NFW"]["k"].iloc[0] == 2

    def test_delta_aicc_scm_is_zero(self):
        df = run_comparison(str(DATA_CSV))
        assert df[df["name"] == "SCM"]["delta_aicc_vs_scm"].iloc[0] == pytest.approx(0.0)

    def test_scm_best_aicc(self):
        """SCM should have the lowest AICc on the synthetic data."""
        df = run_comparison(str(DATA_CSV))
        scm_aicc = df[df["name"] == "SCM"]["aicc_oos"].iloc[0]
        for other in ["RAR", "NFW"]:
            assert scm_aicc < df[df["name"] == other]["aicc_oos"].iloc[0]

    def test_ll_finite(self):
        df = run_comparison(str(DATA_CSV))
        assert df["ll_oos"].apply(np.isfinite).all()

    def test_aicc_finite(self):
        df = run_comparison(str(DATA_CSV))
        assert df["aicc_oos"].apply(np.isfinite).all()

    def test_split_sizes(self):
        df = run_comparison(str(DATA_CSV), test_size=0.2)
        row = df.iloc[0]
        assert row["n_train"] + row["n_test"] == 200

    def test_reproducible(self):
        df1 = run_comparison(str(DATA_CSV), seed=42)
        df2 = run_comparison(str(DATA_CSV), seed=42)
        for name in ["SCM", "RAR", "NFW"]:
            ll1 = df1[df1["name"] == name]["ll_oos"].iloc[0]
            ll2 = df2[df2["name"] == name]["ll_oos"].iloc[0]
            assert ll1 == pytest.approx(ll2)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_comparison("nonexistent.csv")

    def test_scm_ll_oos_matches_scm_oos_fit(self):
        """SCM LL_OOS must match the standalone scm_oos_fit.py result."""
        from scripts.scm_oos_fit import run_oos_fit
        res_scm = run_oos_fit(str(DATA_CSV))
        df_cmp  = run_comparison(str(DATA_CSV))
        ll_cmp  = df_cmp[df_cmp["name"] == "SCM"]["ll_oos"].iloc[0]
        ll_scm  = -res_scm["nll_oos"]
        assert ll_cmp == pytest.approx(ll_scm, rel=1e-6)
