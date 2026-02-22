"""Unit tests for src/scm_models.py."""

import numpy as np
import pytest

from src.scm_models import (
    v_baryonic,
    v_velos,
    v_total,
    residuals,
    chi2_reduced,
    baryonic_tully_fisher,
)


class TestVBaryonic:
    def test_zero_components_gives_zero(self):
        r = np.array([1.0, 2.0, 5.0])
        zeros = np.zeros(3)
        result = v_baryonic(r, zeros, zeros, zeros)
        np.testing.assert_array_equal(result, zeros)

    def test_positive_components(self):
        r = np.array([1.0])
        v_gas = np.array([50.0])
        v_disk = np.array([80.0])
        v_bul = np.array([0.0])
        result = v_baryonic(r, v_gas, v_disk, v_bul, upsilon_disk=1.0)
        expected = np.sqrt(50.0 ** 2 + 80.0 ** 2)
        np.testing.assert_allclose(result, [expected], rtol=1e-6)

    def test_upsilon_scaling(self):
        r = np.array([1.0])
        zeros = np.array([0.0])
        v_disk = np.array([100.0])
        result_1 = v_baryonic(r, zeros, v_disk, zeros, upsilon_disk=1.0)
        result_4 = v_baryonic(r, zeros, v_disk, zeros, upsilon_disk=4.0)
        np.testing.assert_allclose(result_4, 2.0 * result_1, rtol=1e-6)

    def test_shape_preserved(self):
        r = np.linspace(0.5, 20, 50)
        v_comp = np.full(50, 100.0)
        zeros = np.zeros(50)
        result = v_baryonic(r, v_comp, zeros, zeros)
        assert result.shape == (50,)


class TestVVelos:
    def test_zero_radius(self):
        result = v_velos(np.array([0.0]))
        np.testing.assert_array_equal(result, [0.0])

    def test_increases_with_radius(self):
        r = np.array([1.0, 4.0, 9.0])
        v = v_velos(r)
        assert np.all(np.diff(v) > 0)

    def test_scales_with_sqrt_a0(self):
        r = np.array([5.0])
        v1 = v_velos(r, a0=1.2e-10)
        v2 = v_velos(r, a0=4.8e-10)
        np.testing.assert_allclose(v2, 2.0 * v1, rtol=1e-6)

    def test_non_negative(self):
        r = np.linspace(0, 30, 100)
        v = v_velos(r)
        assert np.all(v >= 0)


class TestVTotal:
    def test_includes_velos_contribution(self):
        r = np.array([10.0])
        zeros = np.zeros(1)
        v_with = v_total(r, zeros, zeros, zeros, include_velos=True)
        v_without = v_total(r, zeros, zeros, zeros, include_velos=False)
        assert v_with[0] > v_without[0]

    def test_without_velos_equals_baryonic(self):
        r = np.array([5.0])
        v_gas = np.array([30.0])
        v_disk = np.array([60.0])
        v_bul = np.array([10.0])
        vt = v_total(r, v_gas, v_disk, v_bul, include_velos=False)
        vb = v_baryonic(r, v_gas, v_disk, v_bul)
        np.testing.assert_allclose(vt, vb, rtol=1e-6)

    def test_shape(self):
        n = 30
        r = np.linspace(1, 15, n)
        v_comp = np.full(n, 80.0)
        zeros = np.zeros(n)
        result = v_total(r, v_comp, zeros, zeros)
        assert result.shape == (n,)


class TestResiduals:
    def test_perfect_fit(self):
        v = np.array([100.0, 150.0, 200.0])
        err = np.array([5.0, 5.0, 5.0])
        res = residuals(v, err, v)
        np.testing.assert_array_equal(res, np.zeros(3))

    def test_sign_and_magnitude(self):
        v_obs = np.array([100.0])
        v_err = np.array([10.0])
        v_pred = np.array([90.0])
        res = residuals(v_obs, v_err, v_pred)
        np.testing.assert_allclose(res, [1.0], rtol=1e-9)

    def test_zero_error_handled(self):
        v_obs = np.array([100.0])
        v_err = np.array([0.0])
        v_pred = np.array([90.0])
        res = residuals(v_obs, v_err, v_pred)
        assert np.isfinite(res[0])


class TestChi2Reduced:
    def test_perfect_fit_gives_zero(self):
        v = np.array([100.0, 150.0, 200.0])
        err = np.array([5.0, 5.0, 5.0])
        assert chi2_reduced(v, err, v) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        v_obs = np.array([100.0, 200.0, 300.0])
        err = np.array([10.0, 10.0, 10.0])
        # Each residual = 1 → chi2 = 3/dof(=1) = 3 for n_free=2
        v_pred = v_obs - 10.0
        result = chi2_reduced(v_obs, err, v_pred, n_free=2)
        assert result == pytest.approx(3.0, rel=1e-6)

    def test_positive(self):
        v = np.random.default_rng(0).uniform(50, 300, 20)
        err = np.full(20, 10.0)
        v_pred = v + np.random.default_rng(1).normal(0, 5, 20)
        assert chi2_reduced(v, err, v_pred) >= 0.0


class TestBaryonicTullyFisher:
    def test_higher_velocity_gives_higher_mass(self):
        m1 = baryonic_tully_fisher(100.0)
        m2 = baryonic_tully_fisher(200.0)
        assert m2 > m1

    def test_scales_as_fourth_power(self):
        m1 = baryonic_tully_fisher(100.0)
        m2 = baryonic_tully_fisher(200.0)
        ratio = m2 / m1
        assert ratio == pytest.approx(16.0, rel=1e-6)

    def test_array_input(self):
        v = np.array([100.0, 150.0, 200.0])
        m = baryonic_tully_fisher(v)
        assert m.shape == (3,)
        assert np.all(m > 0)
