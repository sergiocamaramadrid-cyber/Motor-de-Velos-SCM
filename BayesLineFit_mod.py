"""Bayesian line-fit utilities.

This module keeps the vertical log-probability used by the classic
Desmond & Lelli (2019) line-fit workflow. The error propagation term follows

    var = err_y^2 + (slope * err_x)^2 + sigma_int^2

which correctly squares the x-uncertainty contribution.
"""

from __future__ import annotations

import numpy as np


def lnprob_vertical(theta, x_arr, y_arr, err_x_arr, err_y_arr):
    """Return vertical Gaussian log-probability with intrinsic scatter.

    Parameters
    ----------
    theta : sequence[float]
        ``(slope, intercept, sigma_int)``.
    x_arr, y_arr : array-like
        Data coordinates.
    err_x_arr, err_y_arr : array-like
        1σ observational uncertainties for x and y.
    """
    slope, intercept, sigma_int = [float(v) for v in theta]
    if sigma_int < 0:
        return -np.inf

    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    err_x = np.asarray(err_x_arr, dtype=float)
    err_y = np.asarray(err_y_arr, dtype=float)

    model = slope * x + intercept
    variance = err_y**2 + (slope * err_x) ** 2 + sigma_int**2
    if np.any(variance <= 0):
        return -np.inf

    residual = y - model
    return float(-0.5 * np.sum((residual**2) / variance + np.log(2.0 * np.pi * variance)))
