"""
scm_models.py — SCM v0.2 (Motor de Velos)

Defines the SCM rotation-curve model and parameter fitting utilities.

Model
-----
The SCM interpolation function relates the observed centripetal acceleration
g_obs to the baryonic acceleration g_bar via a single free parameter g0:

    g_obs = g_bar / (1 - exp(-sqrt(g_bar / g0)))

This is the standard MOND-style interpolation (McGaugh et al. 2016) used as
the basis of the SCM Condensación-Fluida framework.

References
----------
McGaugh, S. S., Lelli, F., & Schombert, J. M. (2016). Radial Acceleration
Relation in Rotationally Supported Galaxies. PRL, 117, 201101.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def scm_g_obs(g_bar: ArrayLike, g0: float) -> np.ndarray:
    """Return predicted centripetal acceleration for baryonic acceleration *g_bar*.

    Parameters
    ----------
    g_bar:
        Baryonic gravitational acceleration [m s^-2].
    g0:
        SCM characteristic acceleration scale [m s^-2].

    Returns
    -------
    numpy.ndarray
        Predicted observed acceleration.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    x = np.sqrt(np.abs(g_bar) / g0)
    return g_bar / (1.0 - np.exp(-x))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# Default parameter bounds (log10 scale for g0)
_G0_BOUNDS_LOG10 = (-11.5, -9.5)  # typical range ~3e-12 – 3e-10 m/s^2


def fit_g0(
    g_bar: ArrayLike,
    g_obs: ArrayLike,
    g0_init: float = 1.2e-10,
    bounds: tuple[float, float] | None = None,
) -> dict:
    """Fit the SCM g0 parameter by minimising the orthogonal scatter.

    Parameters
    ----------
    g_bar:
        Baryonic acceleration values [m s^-2].
    g_obs:
        Observed centripetal acceleration values [m s^-2].
    g0_init:
        Initial guess for g0 [m s^-2].
    bounds:
        (lower, upper) bounds for g0. Defaults to ``_G0_BOUNDS_LOG10``
        expressed back in linear units.

    Returns
    -------
    dict with keys:
        ``g0_hat``   – best-fit g0 [m s^-2]
        ``g0_err``   – estimated 1-σ uncertainty (from inverse Hessian)
        ``residuals``– array of log-residuals  log10(g_obs/g_pred)
        ``at_bound`` – True if g0_hat is within 1 % of either bound
        ``success``  – optimisation success flag
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    if bounds is None:
        lo = 10 ** _G0_BOUNDS_LOG10[0]
        hi = 10 ** _G0_BOUNDS_LOG10[1]
    else:
        lo, hi = bounds

    # Work in log10 space for numerical stability
    log10_g0_init = np.log10(g0_init)
    log10_bounds = [(np.log10(lo), np.log10(hi))]

    def _loss(params: np.ndarray) -> float:
        g0 = 10.0 ** params[0]
        g_pred = scm_g_obs(g_bar, g0)
        resid = np.log10(g_obs) - np.log10(g_pred)
        return float(np.sum(resid ** 2))

    result = minimize(
        _loss,
        x0=[log10_g0_init],
        method="L-BFGS-B",
        bounds=log10_bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )

    g0_hat = 10.0 ** result.x[0]

    # 1-σ uncertainty from inverse Hessian (finite-difference approximation)
    h = 1e-4
    f0 = _loss(result.x)
    fp = _loss(result.x + h)
    fm = _loss(result.x - h)
    hess = (fp - 2 * f0 + fm) / h ** 2
    if hess > 0:
        var_log10 = 1.0 / hess
        # Propagate to linear scale: Δg0 ≈ g0 * ln(10) * Δlog10(g0)
        g0_err = g0_hat * np.log(10) * np.sqrt(var_log10)
    else:
        g0_err = float("nan")

    g_pred = scm_g_obs(g_bar, g0_hat)
    residuals = np.log10(g_obs) - np.log10(g_pred)

    # Check whether the optimum is within 1 % of either bound (log10 scale)
    log10_g0_hat = result.x[0]
    tol = 0.01 * (log10_bounds[0][1] - log10_bounds[0][0])
    at_bound = (
        abs(log10_g0_hat - log10_bounds[0][0]) < tol
        or abs(log10_g0_hat - log10_bounds[0][1]) < tol
    )

    return {
        "g0_hat": g0_hat,
        "g0_err": g0_err,
        "residuals": residuals,
        "at_bound": at_bound,
        "success": result.success,
        "g_pred": g_pred,
    }
