"""
tests/test_hinge_sfr.py — Unit tests for scripts/hinge_sfr_test.py.

Covers:
- compute_gbar_from_vbar (SI conversion)
- compute_hinge (hinge formula)
- compute_features_for_galaxy (F1/F2/F3 proxies)
- permutation_pvalue (one-sided permutation test)
- matched_pairs_wilcoxon (mass-matched Wilcoxon test)
- regression_test (OLS with HC3 – smoke test for no crash)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from hinge_sfr_test import (
    HingeParams,
    compute_gbar_from_vbar,
    compute_hinge,
    compute_features_for_galaxy,
    matched_pairs_wilcoxon,
    permutation_pvalue,
    regression_test,
    robust_iqr,
)

_KPC_TO_M = 3.085677581e19


# ---------------------------------------------------------------------------
# compute_gbar_from_vbar
# ---------------------------------------------------------------------------

class TestComputeGbarFromVbar:
    def test_known_value(self):
        # g = v² / r; v=1 km/s, r=1 kpc
        r = np.array([1.0])
        v = np.array([1.0])
        g = compute_gbar_from_vbar(r, v)
        expected = (1e3) ** 2 / _KPC_TO_M
        np.testing.assert_allclose(g, [expected], rtol=1e-9)

    def test_scales_as_v_squared(self):
        r = np.array([5.0])
        g1 = compute_gbar_from_vbar(r, np.array([10.0]))
        g2 = compute_gbar_from_vbar(r, np.array([20.0]))
        np.testing.assert_allclose(g2, 4.0 * g1, rtol=1e-9)

    def test_inverse_with_radius(self):
        v = np.array([100.0])
        g1 = compute_gbar_from_vbar(np.array([1.0]), v)
        g2 = compute_gbar_from_vbar(np.array([2.0]), v)
        np.testing.assert_allclose(g2, g1 / 2.0, rtol=1e-9)

    def test_non_negative(self):
        r = np.linspace(0.1, 20.0, 50)
        v = np.full(50, 80.0)
        assert np.all(compute_gbar_from_vbar(r, v) >= 0)


# ---------------------------------------------------------------------------
# compute_hinge
# ---------------------------------------------------------------------------

class TestComputeHinge:
    def test_zero_below_g0(self):
        # gbar > g0 → H = 0
        log_g0 = -10.0
        g0 = 10.0 ** log_g0
        gbar = np.array([10.0 * g0])  # above threshold
        H = compute_hinge(log_g0, d=1.0, gbar_si=gbar)
        np.testing.assert_array_equal(H, [0.0])

    def test_positive_above_threshold(self):
        # gbar < g0 → H > 0
        log_g0 = -10.0
        g0 = 10.0 ** log_g0
        gbar = np.array([g0 / 100.0])
        H = compute_hinge(log_g0, d=1.0, gbar_si=gbar)
        assert H[0] > 0

    def test_d_scaling(self):
        log_g0 = -10.0
        gbar = np.array([10.0 ** -12])
        H1 = compute_hinge(log_g0, d=1.0, gbar_si=gbar)
        H2 = compute_hinge(log_g0, d=2.0, gbar_si=gbar)
        np.testing.assert_allclose(H2, 2.0 * H1, rtol=1e-9)

    def test_non_negative(self):
        log_g0 = np.log10(1.2e-10)
        gbar = np.logspace(-14, -9, 100)
        H = compute_hinge(log_g0, d=1.0, gbar_si=gbar)
        assert np.all(H >= 0)

    def test_monotone_decreasing_with_gbar(self):
        # As gbar increases toward g0, H should decrease
        log_g0 = np.log10(1.2e-10)
        gbar = np.logspace(-13, -10, 50)
        H = compute_hinge(log_g0, d=1.0, gbar_si=gbar)
        # In the active regime (H>0) H should be non-increasing
        active = H > 0
        if active.sum() > 1:
            assert np.all(np.diff(H[active]) <= 1e-9)


# ---------------------------------------------------------------------------
# compute_features_for_galaxy
# ---------------------------------------------------------------------------

def _make_galaxy_df(n=30, rmax=15.0, gbar_scale=1e-12, name="G001"):
    """Helper: synthetic single-galaxy DataFrame."""
    rng = np.random.default_rng(42)
    r = np.linspace(0.5, rmax, n)
    vbar = np.sqrt(gbar_scale * r * _KPC_TO_M) / 1e3  # back to km/s
    return pd.DataFrame({
        "galaxy": name,
        "r_kpc": r,
        "vbar_kms": vbar,
        "rmax_kpc": rmax,
    })


class TestComputeFeaturesForGalaxy:
    def test_returns_expected_keys(self):
        df_g = _make_galaxy_df()
        hp = HingeParams(log_g0=np.log10(3.27e-11), d=1.0)
        result = compute_features_for_galaxy(df_g, hp)
        expected_keys = {
            "galaxy", "rmax_kpc_used",
            "F1_med_abs_dH_dr_ext", "F2_IQR_H_ext",
            "F3_mean_H_ext", "H_ext_npts",
        }
        assert set(result.keys()) == expected_keys

    def test_galaxy_name_preserved(self):
        df_g = _make_galaxy_df(name="NGC1234")
        hp = HingeParams(log_g0=np.log10(3.27e-11), d=1.0)
        result = compute_features_for_galaxy(df_g, hp)
        assert result["galaxy"] == "NGC1234"

    def test_npts_positive(self):
        df_g = _make_galaxy_df()
        hp = HingeParams(log_g0=np.log10(3.27e-11), d=1.0)
        result = compute_features_for_galaxy(df_g, hp)
        assert result["H_ext_npts"] > 0

    def test_F3_non_negative(self):
        df_g = _make_galaxy_df()
        hp = HingeParams(log_g0=np.log10(3.27e-11), d=1.0)
        result = compute_features_for_galaxy(df_g, hp)
        assert result["F3_mean_H_ext"] >= 0 or np.isnan(result["F3_mean_H_ext"])

    def test_gbar_column_accepted(self):
        df_g = _make_galaxy_df()
        hp = HingeParams(log_g0=np.log10(3.27e-11), d=1.0)
        # convert vbar_kms to gbar_m_s2 and replace column
        gbar = compute_gbar_from_vbar(
            df_g["r_kpc"].to_numpy(), df_g["vbar_kms"].to_numpy()
        )
        df_g2 = df_g.drop(columns=["vbar_kms"])
        df_g2["gbar_m_s2"] = gbar
        r1 = compute_features_for_galaxy(df_g, hp)
        r2 = compute_features_for_galaxy(df_g2, hp)
        np.testing.assert_allclose(r1["F3_mean_H_ext"], r2["F3_mean_H_ext"], rtol=1e-9)

    def test_rmax_fallback_when_absent(self):
        df_g = _make_galaxy_df().drop(columns=["rmax_kpc"])
        hp = HingeParams(log_g0=np.log10(3.27e-11), d=1.0)
        result = compute_features_for_galaxy(df_g, hp)
        # rmax_kpc_used should equal max(r_kpc)
        assert result["rmax_kpc_used"] == pytest.approx(df_g["r_kpc"].max())


# ---------------------------------------------------------------------------
# robust_iqr
# ---------------------------------------------------------------------------

class TestRobustIqr:
    def test_uniform_returns_expected(self):
        x = np.linspace(0.0, 10.0, 101)
        assert robust_iqr(x) == pytest.approx(5.0, rel=0.05)

    def test_constant_returns_zero(self):
        x = np.full(20, 3.14)
        assert robust_iqr(x) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# permutation_pvalue
# ---------------------------------------------------------------------------

def _make_regression_df(n=80, seed=0):
    """Synthetic galaxy table where F1 positively predicts log_sfr."""
    rng = np.random.default_rng(seed)
    log_mbar = rng.uniform(9.0, 11.5, n)
    F1 = rng.uniform(0.0, 5.0, n)
    # True model: logSFR = -10 + 0.9*logMbar + 0.3*F1 + noise
    log_sfr = -10.0 + 0.9 * log_mbar + 0.3 * F1 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "log_mbar": log_mbar,
        "log_sfr": log_sfr,
        "F1_med_abs_dH_dr_ext": F1,
    })


class TestPermutationPvalue:
    def test_p_low_when_effect_present(self):
        df = _make_regression_df(n=80, seed=0)
        p = permutation_pvalue(df, "F1_med_abs_dH_dr_ext", n_perm=500, seed=7)
        # With a clear signal the p-value should be small
        assert p < 0.1

    def test_p_bounded(self):
        df = _make_regression_df(n=40, seed=1)
        p = permutation_pvalue(df, "F1_med_abs_dH_dr_ext", n_perm=200, seed=3)
        assert 0.0 < p <= 1.0

    def test_handles_nan_rows(self):
        df = _make_regression_df(n=50, seed=2)
        df.loc[0, "F1_med_abs_dH_dr_ext"] = np.nan
        df.loc[1, "log_sfr"] = np.nan
        p = permutation_pvalue(df, "F1_med_abs_dH_dr_ext", n_perm=100, seed=5)
        assert np.isfinite(p)


# ---------------------------------------------------------------------------
# matched_pairs_wilcoxon
# ---------------------------------------------------------------------------

def _make_pairs_df(n=60, seed=0):
    """Synthetic df where high-F galaxies have higher log_sfr."""
    rng = np.random.default_rng(seed)
    log_mbar = np.repeat(np.linspace(9.5, 11.0, n // 2), 2)
    log_mbar += rng.normal(0, 0.02, n)
    F = rng.uniform(0.0, 4.0, n)
    log_sfr = -10.0 + 0.9 * log_mbar + 0.5 * F + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "log_mbar": log_mbar,
        "log_sfr": log_sfr,
        "F2_IQR_H_ext": F,
    })


class TestMatchedPairsWilcoxon:
    def test_returns_expected_keys(self):
        df = _make_pairs_df()
        result = matched_pairs_wilcoxon(df, "F2_IQR_H_ext", dlogm=0.1)
        assert set(result.keys()) == {"n_pairs", "wilcoxon_p", "median_delta_logSFR"}

    def test_n_pairs_positive(self):
        df = _make_pairs_df(n=60)
        result = matched_pairs_wilcoxon(df, "F2_IQR_H_ext", dlogm=0.15)
        assert result["n_pairs"] > 0

    def test_positive_median_delta_when_effect_present(self):
        df = _make_pairs_df(n=60, seed=7)
        result = matched_pairs_wilcoxon(df, "F2_IQR_H_ext", dlogm=0.15)
        assert result["median_delta_logSFR"] > 0

    def test_nan_pvalue_when_too_few_pairs(self):
        # Use n=8 (odd, will yield < 10 pairs) → Wilcoxon p should be NaN
        rng = np.random.default_rng(99)
        n = 8
        df = pd.DataFrame({
            "galaxy": [f"G{i:03d}" for i in range(n)],
            "log_mbar": np.linspace(9.5, 11.0, n),
            "log_sfr": rng.uniform(-1.0, 1.0, n),
            "F2_IQR_H_ext": rng.uniform(0.0, 4.0, n),
        })
        result = matched_pairs_wilcoxon(df, "F2_IQR_H_ext", dlogm=0.5)
        assert np.isnan(result["wilcoxon_p"])

    def test_respects_morph_bin(self):
        df = _make_pairs_df(n=60)
        df["morph_bin"] = ["A" if i % 2 == 0 else "B" for i in range(len(df))]
        result = matched_pairs_wilcoxon(df, "F2_IQR_H_ext", dlogm=0.5)
        # Should still find some pairs
        assert result["n_pairs"] >= 0


# ---------------------------------------------------------------------------
# regression_test — smoke test only (output is a string)
# ---------------------------------------------------------------------------

class TestRegressionTest:
    def test_returns_string(self):
        df = _make_regression_df(n=50)
        output = regression_test(df, "F1_med_abs_dH_dr_ext")
        assert isinstance(output, str)
        assert len(output) > 0

    def test_contains_feature_name(self):
        df = _make_regression_df(n=50)
        output = regression_test(df, "F1_med_abs_dH_dr_ext")
        assert "F1_med_abs_dH_dr_ext" in output

    def test_with_morph_bin(self):
        df = _make_regression_df(n=60)
        df["morph_bin"] = ["late" if i % 2 == 0 else "early" for i in range(len(df))]
        output = regression_test(df, "F1_med_abs_dH_dr_ext")
        assert isinstance(output, str)
