import numpy as np
import pytest

from BayesLineFit_mod import lnprob_vertical


def test_lnprob_vertical_uses_squared_x_error_term():
    x = np.array([1.0, 2.0])
    y = np.array([2.1, 3.9])
    err_x = np.array([0.3, 0.4])
    err_y = np.array([0.2, 0.25])
    theta = (1.5, 0.5, 0.1)

    got = lnprob_vertical(theta, x, y, err_x, err_y)

    slope, intercept, sigma_int = theta
    variance = err_y**2 + (slope * err_x) ** 2 + sigma_int**2
    residual = y - (slope * x + intercept)
    expected = -0.5 * np.sum((residual**2) / variance + np.log(2.0 * np.pi * variance))

    assert got == pytest.approx(expected)


def test_lnprob_vertical_rejects_negative_intrinsic_scatter():
    val = lnprob_vertical((1.0, 0.0, -1e-3), [1.0], [1.0], [0.1], [0.1])
    assert val == -np.inf
