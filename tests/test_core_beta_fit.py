"""
test_core_beta_fit.py — Unit tests for src.core.beta_fit.

Validates:
    fit_beta_single  — correct slope / flag for synthetic deep-regime data
    fit_beta_batch   — correct per-galaxy aggregation; both vbar and gas+star paths
"""
import numpy as np
import pandas as pd
import pytest

from src.core.beta_fit import (
    fit_beta_single,
    fit_beta_batch,
    VELO_INERTE_LO,
    VELO_INERTE_HI,
    MIN_DEEP_POINTS,
)
from src.core.deep_regime import A0_DEFAULT, CONV as _CONV, KPC_TO_M


# ---------------------------------------------------------------------------
# Helpers to build synthetic data in the deep regime
# ---------------------------------------------------------------------------

def _synthetic_deep_points(n=8, beta=0.5, a0=A0_DEFAULT):
    """
    Generate n deep-regime (r, vrot, vbar) points that satisfy
    log(g_obs) = beta * log(g_bar) + const with the given beta.

    Strategy:
        - Fix g_bar << 0.3 * a0 for all points
        - Derive vbar and vrot from g_bar / g_obs accordingly
        - Compute radii so that vbar and vrot are physically sensible
    """
    # Choose g_bar spanning a decade in the deep regime
    g_bar_vals = np.logspace(np.log10(1e-14), np.log10(0.2 * a0 * 0.3), n)
    g_obs_vals = g_bar_vals ** beta        # exact power law

    r_kpc = np.linspace(10.0, 80.0, n)   # kpc
    conv = _CONV
    # vbar = sqrt(g_bar * r / conv), vrot = sqrt(g_obs * r / conv)
    vbar_kms = np.sqrt(g_bar_vals * r_kpc / conv)
    vrot_kms = np.sqrt(g_obs_vals * r_kpc / conv)
    return r_kpc, vrot_kms, vbar_kms


# ---------------------------------------------------------------------------
# Tests: fit_beta_single
# ---------------------------------------------------------------------------

class TestFitBetaSingle:
    def test_recovers_beta_05(self):
        r, vrot, vbar = _synthetic_deep_points(beta=0.5)
        res = fit_beta_single(r, vrot, vbar)
        assert res["n_deep"] >= MIN_DEEP_POINTS
        assert res["beta"] == pytest.approx(0.5, abs=0.05)
        assert res["velo_inerte_flag"] is True

    def test_recovers_beta_04(self):
        r, vrot, vbar = _synthetic_deep_points(beta=0.4)
        res = fit_beta_single(r, vrot, vbar)
        assert res["beta"] == pytest.approx(0.4, abs=0.05)
        assert VELO_INERTE_LO <= res["beta"] <= VELO_INERTE_HI

    def test_outside_velo_inerte_range(self):
        r, vrot, vbar = _synthetic_deep_points(beta=0.9)
        res = fit_beta_single(r, vrot, vbar)
        assert res["velo_inerte_flag"] is False

    def test_insufficient_deep_points(self):
        # All points outside deep regime → n_deep = 0, beta = NaN
        a0 = A0_DEFAULT
        r = np.array([1.0, 2.0])
        vbar = np.array([200.0, 200.0])   # g_bar >> 0.3 * a0
        vrot = np.array([210.0, 210.0])
        res = fit_beta_single(r, vrot, vbar)
        assert res["n_deep"] == 0
        assert np.isnan(res["beta"])
        assert res["velo_inerte_flag"] is False

    def test_beta_err_positive(self):
        r, vrot, vbar = _synthetic_deep_points(n=10, beta=0.5)
        # Add tiny noise to avoid zero stderr
        rng = np.random.default_rng(42)
        vrot = vrot * (1 + rng.normal(0, 0.01, size=len(vrot)))
        res = fit_beta_single(r, vrot, vbar)
        assert res["beta_err"] >= 0
        assert np.isfinite(res["beta_err"])

    def test_r_value_finite(self):
        r, vrot, vbar = _synthetic_deep_points(beta=0.5)
        res = fit_beta_single(r, vrot, vbar)
        assert np.isfinite(res["r_value"])


# ---------------------------------------------------------------------------
# Tests: fit_beta_batch
# ---------------------------------------------------------------------------

class TestFitBetaBatch:
    def _make_rc(self):
        r, vrot, vbar = _synthetic_deep_points(beta=0.5, n=10)
        df_a = pd.DataFrame({
            "galaxy_id": "GAL_A",
            "r_kpc":     r,
            "vrot_kms":  vrot,
            "vbar_kms":  vbar,
        })
        r2, vrot2, vbar2 = _synthetic_deep_points(beta=0.4, n=8)
        df_b = pd.DataFrame({
            "galaxy_id": "GAL_B",
            "r_kpc":     r2,
            "vrot_kms":  vrot2,
            "vbar_kms":  vbar2,
        })
        return pd.concat([df_a, df_b], ignore_index=True)

    def test_two_galaxies(self):
        rc = self._make_rc()
        cat = fit_beta_batch(rc)
        assert len(cat) == 2
        assert set(cat["galaxy_id"]) == {"GAL_A", "GAL_B"}

    def test_required_columns_present(self):
        rc = self._make_rc()
        cat = fit_beta_batch(rc)
        required = {"galaxy_id", "beta", "beta_err", "r_value",
                    "n_deep", "velo_inerte_flag"}
        assert required.issubset(cat.columns)

    def test_gas_plus_star_path(self):
        r, vrot, vbar = _synthetic_deep_points(beta=0.5, n=10)
        vgas  = vbar * 0.6
        vstar = np.sqrt(np.maximum(vbar ** 2 - vgas ** 2, 0.0))
        rc = pd.DataFrame({
            "galaxy_id": "GAL_GS",
            "r_kpc":     r,
            "vrot_kms":  vrot,
            "vgas_kms":  vgas,
            "vstar_kms": vstar,
        })
        cat = fit_beta_batch(rc)
        assert len(cat) == 1
        assert cat.iloc[0]["velo_inerte_flag"]

    def test_no_baryonic_column_returns_nan(self):
        r, vrot, _ = _synthetic_deep_points(beta=0.5, n=5)
        rc = pd.DataFrame({
            "galaxy_id": "GAL_NOBAR",
            "r_kpc":     r,
            "vrot_kms":  vrot,
        })
        cat = fit_beta_batch(rc)
        assert len(cat) == 1
        assert np.isnan(cat.iloc[0]["beta"])

    def test_column_types(self):
        rc = self._make_rc()
        cat = fit_beta_batch(rc)
        assert np.issubdtype(cat["n_deep"].dtype, np.integer)
        assert cat["velo_inerte_flag"].dtype == bool
        assert cat["beta"].dtype == float
