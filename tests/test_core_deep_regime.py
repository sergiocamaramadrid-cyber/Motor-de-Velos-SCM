"""
test_core_deep_regime.py — Unit tests for src.core.deep_regime.

Validates:
    compute_g_obs   — (km/s)²/kpc → m/s² conversion
    compute_g_bar   — identical conversion for baryonic velocity
    deep_mask       — boolean selection below threshold × a0
"""
import numpy as np
import pytest

from src.core.deep_regime import (
    compute_g_obs,
    compute_g_bar,
    deep_mask,
    KPC_TO_M,
    A0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
)

_CONV = 1e6 / KPC_TO_M


class TestComputeGObs:
    def test_scalar_round_trip(self):
        v, r = 100.0, 10.0          # km/s, kpc
        expected = v ** 2 / r * _CONV
        assert compute_g_obs(v, r) == pytest.approx(expected, rel=1e-9)

    def test_array_shape(self):
        v = np.array([50.0, 100.0, 150.0])
        r = np.array([1.0,  5.0,   10.0])
        g = compute_g_obs(v, r)
        assert g.shape == (3,)
        assert np.all(g > 0)

    def test_zero_radius_guard(self):
        # Near-zero radius must not cause division by zero
        g = compute_g_obs(100.0, 0.0)
        assert np.isfinite(g)

    def test_zero_velocity_gives_zero(self):
        g = compute_g_obs(0.0, 5.0)
        assert g == pytest.approx(0.0, abs=1e-30)


class TestComputeGBar:
    def test_matches_g_obs_for_same_inputs(self):
        v, r = 80.0, 8.0
        assert compute_g_bar(v, r) == pytest.approx(compute_g_obs(v, r), rel=1e-9)

    def test_array(self):
        vb = np.array([20.0, 40.0, 60.0])
        r  = np.array([2.0,  4.0,  8.0])
        g  = compute_g_bar(vb, r)
        assert g.shape == (3,)
        assert np.all(g > 0)


class TestDeepMask:
    def test_all_deep(self):
        # Very small g_bar << a0
        g_bar = np.full(5, 1e-13)       # well below 0.3 × 1.2e-10
        mask  = deep_mask(g_bar)
        assert mask.all()

    def test_none_deep(self):
        # g_bar >> a0
        g_bar = np.full(5, 1e-8)
        mask  = deep_mask(g_bar)
        assert not mask.any()

    def test_mixed(self):
        g_bar = np.array([1e-13, 1e-13, 1e-8, 1e-8])
        mask  = deep_mask(g_bar)
        assert mask[:2].all()
        assert not mask[2:].any()

    def test_custom_threshold(self):
        a0        = 1.2e-10
        threshold = 0.5
        g_bar     = np.array([0.4 * a0, 0.6 * a0])
        mask      = deep_mask(g_bar, a0=a0, threshold=threshold)
        assert mask[0]
        assert not mask[1]
