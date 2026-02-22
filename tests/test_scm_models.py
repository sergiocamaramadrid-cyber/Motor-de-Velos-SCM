"""Unit tests for src/scm_models.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.scm_models import (
    G0_DEFAULT,
    rar_g_obs,
    fit_g0,
    bin_rar,
    deep_regime_slope,
)


# ---------------------------------------------------------------------------
# rar_g_obs
# ---------------------------------------------------------------------------

class TestRarGObs:
    def test_newtonian_limit(self):
        """When g_bar >> g0 the formula should return ~ g_bar."""
        g_bar = np.array([1e-6, 1e-5])  # much larger than g0 ~ 1.2e-10
        result = rar_g_obs(g_bar, G0_DEFAULT)
        np.testing.assert_allclose(result, g_bar, rtol=1e-4)

    def test_deep_mond_limit(self):
        """When g_bar << g0 the formula should return ~ sqrt(g_bar * g0)."""
        g_bar = np.array([1e-15, 1e-14])  # much smaller than g0
        result = rar_g_obs(g_bar, G0_DEFAULT)
        expected = np.sqrt(g_bar * G0_DEFAULT)
        np.testing.assert_allclose(result, expected, rtol=1e-2)

    def test_positive_output(self):
        g_bar = np.logspace(-13, -8, 50)
        result = rar_g_obs(g_bar, G0_DEFAULT)
        assert np.all(result > 0)

    def test_scalar_input(self):
        result = rar_g_obs(G0_DEFAULT, G0_DEFAULT)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# fit_g0
# ---------------------------------------------------------------------------

class TestFitG0:
    def _make_data(self, g0_true=G0_DEFAULT, n=200, seed=7):
        rng = np.random.default_rng(seed)
        g_bar = 10 ** rng.uniform(-13, -8, n)
        g_obs = rar_g_obs(g_bar, g0_true) * 10 ** rng.normal(0, 0.05, n)
        return g_bar, g_obs

    def test_recovers_g0(self):
        g_bar, g_obs = self._make_data()
        result = fit_g0(g_bar, g_obs)
        assert abs(result["g0"] - G0_DEFAULT) / G0_DEFAULT < 0.05  # within 5%

    def test_returns_required_keys(self):
        g_bar, g_obs = self._make_data()
        result = fit_g0(g_bar, g_obs)
        for key in ("g0", "g0_err", "rms", "n"):
            assert key in result

    def test_n_matches_clean_data(self):
        g_bar, g_obs = self._make_data(n=150)
        result = fit_g0(g_bar, g_obs)
        assert result["n"] == 150

    def test_ignores_nonpositive(self):
        g_bar, g_obs = self._make_data(n=100)
        g_bar[0] = -1.0
        g_obs[1] = 0.0
        result = fit_g0(g_bar, g_obs)
        assert result["n"] == 98


# ---------------------------------------------------------------------------
# bin_rar
# ---------------------------------------------------------------------------

class TestBinRar:
    def _make_data(self, n=500, seed=3):
        rng = np.random.default_rng(seed)
        g_bar = 10 ** rng.uniform(-13, -8, n)
        g_obs = rar_g_obs(g_bar, G0_DEFAULT)
        return g_bar, g_obs

    def test_returns_dataframe(self):
        import pandas as pd
        g_bar, g_obs = self._make_data()
        df = bin_rar(g_bar, g_obs, n_bins=10)
        assert isinstance(df, pd.DataFrame)

    def test_column_names(self):
        g_bar, g_obs = self._make_data()
        df = bin_rar(g_bar, g_obs)
        for col in ("log_g_bar_bin", "log_g_obs_mean", "log_g_obs_std", "count"):
            assert col in df.columns

    def test_no_empty_bins_for_dense_data(self):
        g_bar, g_obs = self._make_data(n=1000)
        df = bin_rar(g_bar, g_obs, n_bins=10)
        assert len(df) == 10

    def test_counts_positive(self):
        g_bar, g_obs = self._make_data()
        df = bin_rar(g_bar, g_obs)
        assert (df["count"] > 0).all()


# ---------------------------------------------------------------------------
# deep_regime_slope
# ---------------------------------------------------------------------------

class TestDeepRegimeSlope:
    def _make_data(self, n=500, seed=5):
        rng = np.random.default_rng(seed)
        g_bar = 10 ** rng.uniform(-13, -8, n)
        g_obs = rar_g_obs(g_bar, G0_DEFAULT)
        return g_bar, g_obs

    def test_collapses_for_perfect_data(self):
        g_bar, g_obs = self._make_data(n=1000)
        result = deep_regime_slope(g_bar, g_obs)
        assert result["collapses"] is True

    def test_slope_near_half(self):
        g_bar, g_obs = self._make_data(n=1000)
        result = deep_regime_slope(g_bar, g_obs)
        assert abs(result["slope"] - 0.5) < 0.15

    def test_returns_required_keys(self):
        g_bar, g_obs = self._make_data()
        result = deep_regime_slope(g_bar, g_obs)
        for key in ("slope", "intercept", "expected_slope", "n_deep", "collapses"):
            assert key in result

    def test_too_few_points(self):
        g_bar = np.array([1e-13, 1e-13])
        g_obs = np.array([1e-12, 1e-12])
        result = deep_regime_slope(g_bar, g_obs, percentile=100.0)
        assert result["collapses"] is False


# ---------------------------------------------------------------------------
# Integration: CSV round-trip
# ---------------------------------------------------------------------------

class TestCsvRoundTrip:
    def test_sparc_csv_exists(self):
        import os
        assert os.path.exists("data/sparc_rar_sample.csv")

    def test_sparc_csv_columns(self):
        import pandas as pd
        df = pd.read_csv("data/sparc_rar_sample.csv")
        assert "g_bar" in df.columns
        assert "g_obs" in df.columns

    def test_sparc_csv_nonempty(self):
        import pandas as pd
        df = pd.read_csv("data/sparc_rar_sample.csv")
        assert len(df) > 100

    def test_pipeline_runs_on_sparc_csv(self):
        from src.scm_analysis import run_pipeline
        results = run_pipeline("data/sparc_rar_sample.csv")
        assert "fit" in results
        assert results["fit"]["g0"] > 0
        assert results["deep"]["n_deep"] > 0
