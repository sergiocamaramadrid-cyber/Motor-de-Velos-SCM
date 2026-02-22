"""
test_scm_models.py — Tests unitarios para src/scm_models.py.

Cubre:
- Función de transición ν(y)
- Modelo g_obs_model
- Residuos log10
- Bounds BOUNDS_LOG10_G0 (v0.2b: lower bound = -16.0)
- fit_g0: óptimo dentro de bounds, at_lower_bound=False para datos sintéticos
- bin_residuals / quantiles_g_bar
- print_fit_summary: [WARN] incluye log10(g0_hat) y comparación con q10/q50 de g_bar
"""

import numpy as np
import pytest

from src.scm_models import (
    BOUNDS_LOG10_G0,
    bin_residuals,
    fit_g0,
    g_obs_model,
    log10_residuals,
    nu,
    print_fit_summary,
    quantiles_g_bar,
)


# ---------------------------------------------------------------------------
# Tests de ν(y)
# ---------------------------------------------------------------------------

class TestNu:
    def test_large_y_approaches_one(self):
        """Para y >> 1, ν(y) → 1 (régimen newtoniano)."""
        y = np.array([1e4, 1e6])
        result = nu(y)
        np.testing.assert_allclose(result, 1.0, atol=1e-3)

    def test_small_y_greater_than_one(self):
        """Para cualquier y > 0, ν(y) > 1."""
        y = np.logspace(-3, 3, 50)
        assert np.all(nu(y) > 1.0)

    def test_monotone_decreasing(self):
        """ν(y) es decreciente (mayor aceleración ≡ mayor y → menor corrección).

        Se usa <= 0 para tolerar que a y muy grande ν → 1.0 en precisión float64.
        """
        y = np.logspace(-2, 4, 100)
        vals = nu(y)
        assert np.all(np.diff(vals) <= 0)

    def test_scalar_input(self):
        result = nu(np.array([1.0]))
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Tests de g_obs_model
# ---------------------------------------------------------------------------

class TestGObsModel:
    def test_deep_regime_amplification(self):
        """En régimen profundo (g_bar << g0), g_obs > g_bar."""
        g0 = 1e-10
        g_bar = np.array([1e-14, 1e-13])
        g_obs = g_obs_model(g_bar, g0)
        assert np.all(g_obs > g_bar)

    def test_newtonian_regime(self):
        """En régimen newtoniano (g_bar >> g0), g_obs ≈ g_bar."""
        g0 = 1e-10
        g_bar = np.array([1e-5, 1e-4])
        g_obs = g_obs_model(g_bar, g0)
        np.testing.assert_allclose(g_obs, g_bar, rtol=1e-3)

    def test_positive_output(self):
        g0 = 1e-10
        g_bar = np.logspace(-15, -8, 50)
        g_obs = g_obs_model(g_bar, g0)
        assert np.all(g_obs > 0)


# ---------------------------------------------------------------------------
# Tests de log10_residuals
# ---------------------------------------------------------------------------

class TestLog10Residuals:
    def test_zero_when_perfect(self):
        """Si g_obs = g_obs_model, los residuos son cero."""
        g0 = 1e-10
        g_bar = np.logspace(-14, -8, 30)
        g_obs = g_obs_model(g_bar, g0)
        resid = log10_residuals(g_obs, g_bar, g0)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)

    def test_negative_when_model_overpredicts(self):
        """Si el modelo sobre-predice, los residuos son negativos."""
        g0 = 1e-10
        g_bar = np.array([1e-14])
        g_obs_true = g_obs_model(g_bar, g0) * 0.5  # mitad del modelo
        resid = log10_residuals(g_obs_true, g_bar, g0)
        assert np.all(resid < 0)


# ---------------------------------------------------------------------------
# Tests de BOUNDS_LOG10_G0 (cambio clave v0.2b)
# ---------------------------------------------------------------------------

class TestBounds:
    def test_lower_bound_is_minus_16(self):
        """v0.2b: el lower bound de log10(g0) debe ser -16.0."""
        assert BOUNDS_LOG10_G0[0] == pytest.approx(-16.0)

    def test_upper_bound_is_minus_8(self):
        """El upper bound de log10(g0) debe ser -8.0."""
        assert BOUNDS_LOG10_G0[1] == pytest.approx(-8.0)

    def test_bounds_range_allows_very_small_g0(self):
        """El rango debe cubrir g0 tan pequeño como 1e-16."""
        lo, hi = BOUNDS_LOG10_G0
        assert lo <= -16.0
        assert hi >= -8.0


# ---------------------------------------------------------------------------
# Tests de fit_g0
# ---------------------------------------------------------------------------

class TestFitG0:
    def _make_synthetic_data(self, g0_true: float, noise_sigma: float = 0.05):
        """Genera datos sintéticos alrededor del modelo con ruido log-normal."""
        rng = np.random.default_rng(42)
        g_bar = np.logspace(-14, -9, 200)
        g_obs_clean = g_obs_model(g_bar, g0_true)
        noise = rng.normal(0, noise_sigma, size=g_bar.size)
        g_obs = g_obs_clean * 10.0 ** noise
        return g_bar, g_obs

    def test_fit_recovers_g0_within_bounds(self):
        """El ajuste devuelve g0_hat dentro del rango de bounds."""
        g0_true = 1e-10
        g_bar, g_obs = self._make_synthetic_data(g0_true)
        result = fit_g0(g_obs, g_bar)
        lo = 10.0 ** BOUNDS_LOG10_G0[0]
        hi = 10.0 ** BOUNDS_LOG10_G0[1]
        assert lo <= result["g0_hat"] <= hi

    def test_fit_recovers_g0_approximately(self):
        """El ajuste recupera g0_true con error razonable en log10."""
        g0_true = 1e-10
        g_bar, g_obs = self._make_synthetic_data(g0_true)
        result = fit_g0(g_obs, g_bar)
        assert abs(result["log10_g0_hat"] - np.log10(g0_true)) < 1.0

    def test_fit_not_at_lower_bound_for_typical_g0(self):
        """Para g0_true = 1e-10 (bien dentro de bounds), at_lower_bound=False."""
        g0_true = 1e-10
        g_bar, g_obs = self._make_synthetic_data(g0_true)
        result = fit_g0(g_obs, g_bar)
        assert not result["at_lower_bound"]

    def test_fit_detects_lower_bound_when_g0_very_small(self):
        """Si g0_true < lower_bound, at_lower_bound=True con bounds restringidos."""
        g0_true = 1e-13
        g_bar, g_obs = self._make_synthetic_data(g0_true, noise_sigma=0.01)
        # Con bounds restringidos el solver debería tocar el lower bound
        result_restricted = fit_g0(g_obs, g_bar, bounds_log10_g0=(-12.0, -8.0))
        assert result_restricted["at_lower_bound"]

        # Con bounds ampliados (v0.2b) el solver puede salir del lower bound
        result_wide = fit_g0(g_obs, g_bar, bounds_log10_g0=(-16.0, -8.0))
        assert not result_wide["at_lower_bound"]

    def test_rms_decreases_with_wider_bounds_when_optimal_outside(self):
        """Ampliar bounds reduce el RMS cuando el óptimo estaba fuera del rango."""
        g0_true = 1e-13
        g_bar, g_obs = self._make_synthetic_data(g0_true, noise_sigma=0.01)
        result_narrow = fit_g0(g_obs, g_bar, bounds_log10_g0=(-12.0, -8.0))
        result_wide = fit_g0(g_obs, g_bar, bounds_log10_g0=(-16.0, -8.0))
        assert result_wide["rms"] <= result_narrow["rms"]

    def test_result_keys(self):
        """El resultado contiene todas las claves esperadas."""
        g_bar, g_obs = self._make_synthetic_data(1e-10)
        result = fit_g0(g_obs, g_bar)
        for key in ("g0_hat", "log10_g0_hat", "rms", "lower_bound", "upper_bound", "at_lower_bound"):
            assert key in result


# ---------------------------------------------------------------------------
# Tests de bin_residuals
# ---------------------------------------------------------------------------

class TestBinResiduals:
    def test_returns_list_of_dicts(self):
        g0 = 1e-10
        g_bar = np.logspace(-14, -9, 100)
        g_obs = g_obs_model(g_bar, g0)
        bins = bin_residuals(g_obs, g_bar, g0, n_bins=5)
        assert isinstance(bins, list)
        assert all(isinstance(b, dict) for b in bins)

    def test_perfect_model_zero_residuals(self):
        """Con datos perfectos, los residuos medianos son ≈ 0."""
        g0 = 1e-10
        g_bar = np.logspace(-14, -9, 200)
        g_obs = g_obs_model(g_bar, g0)
        bins = bin_residuals(g_obs, g_bar, g0, n_bins=5)
        for b in bins:
            assert abs(b["median_residual"]) < 1e-8

    def test_bin_keys(self):
        g0 = 1e-10
        g_bar = np.logspace(-14, -9, 50)
        g_obs = g_obs_model(g_bar, g0)
        bins = bin_residuals(g_obs, g_bar, g0, n_bins=3)
        for b in bins:
            assert "g_bar_center" in b
            assert "median_residual" in b
            assert "n_points" in b


# ---------------------------------------------------------------------------
# Tests de quantiles_g_bar
# ---------------------------------------------------------------------------

class TestQuantilesGBar:
    def test_order(self):
        g_bar = np.logspace(-14, -9, 1000)
        q = quantiles_g_bar(g_bar)
        assert q["q10"] <= q["q50"] <= q["q90"]

    def test_keys(self):
        g_bar = np.logspace(-14, -9, 100)
        q = quantiles_g_bar(g_bar)
        assert set(q.keys()) == {"q10", "q50", "q90"}


# ---------------------------------------------------------------------------
# Tests de print_fit_summary — telemetría [WARN] con log10 y comparación q10/q50
# ---------------------------------------------------------------------------

class TestPrintFitSummary:
    def _make_fit_result(self, at_lower_bound: bool = False) -> dict:
        return {
            "g0_hat": 1.2e-10,
            "log10_g0_hat": -9.921,
            "rms": 0.1234,
            "lower_bound": 1e-16,
            "upper_bound": 1e-8,
            "at_lower_bound": at_lower_bound,
        }

    def test_scm_v02_tag_in_output(self, capsys):
        """La primera línea contiene la etiqueta [SCM v0.2]."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result()
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        assert "[SCM v0.2]" in out
        assert "g0_hat=" in out

    def test_no_warn_when_not_at_bound(self, capsys):
        """Sin tocar el lower bound, no se emite [WARN]."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=False)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        assert "[WARN]" not in out

    def test_warn_emitted_when_at_lower_bound(self, capsys):
        """Al tocar el lower bound se emite una línea [WARN]."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=True)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        assert "[WARN]" in out
        assert "lower" in out

    def test_warn_includes_g0_hat_value(self, capsys):
        """La línea [WARN] incluye el valor de g0_hat."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=True)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        warn_line = next(l for l in out.splitlines() if "[WARN]" in l)
        assert "g0_hat=" in warn_line

    def test_warn_includes_log10_g0_hat(self, capsys):
        """La línea [WARN] incluye log10(g0_hat) para ubicar el óptimo en escala."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=True)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        warn_line = next(l for l in out.splitlines() if "[WARN]" in l)
        assert "log10(g0_hat)" in warn_line

    def test_warn_includes_q10_comparison(self, capsys):
        """La línea [WARN] incluye la comparación con q10 de g_bar."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=True)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        warn_line = next(l for l in out.splitlines() if "[WARN]" in l)
        assert "q10" in warn_line

    def test_warn_includes_q50_comparison(self, capsys):
        """La línea [WARN] incluye la comparación con q50 de g_bar."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=True)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        warn_line = next(l for l in out.splitlines() if "[WARN]" in l)
        assert "q50" in warn_line

    def test_warn_log10_g0_hat_value_correct(self, capsys):
        """El valor de log10(g0_hat) en la línea [WARN] coincide con fit_result."""
        g_bar = np.logspace(-13, -10, 100)
        fit = self._make_fit_result(at_lower_bound=True)
        bins = bin_residuals(g_obs_model(g_bar, fit["g0_hat"]), g_bar, fit["g0_hat"])
        print_fit_summary(fit, bins, g_bar)
        out = capsys.readouterr().out
        warn_line = next(l for l in out.splitlines() if "[WARN]" in l)
        # log10(1.2e-10) ≈ -9.921, which should appear in the warn line
        assert "-9.921" in warn_line or "-9.92" in warn_line
