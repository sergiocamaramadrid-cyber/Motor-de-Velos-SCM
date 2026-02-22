"""
scm_models.py — RAR (Radial Acceleration Relation) models and utilities.

Core formula (McGaugh et al. 2016):
    g_obs = g_bar / (1 - exp(-sqrt(g_bar / g0)))

where g0 is the characteristic acceleration scale (~1.2e-10 m/s²).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import binned_statistic

# Default acceleration scale (SI, m/s²)
G0_DEFAULT = 1.2e-10


# ---------------------------------------------------------------------------
# RAR formula
# ---------------------------------------------------------------------------

def rar_g_obs(g_bar: np.ndarray, g0: float) -> np.ndarray:
    """Return predicted g_obs given baryonic acceleration g_bar and scale g0.

    Uses the interpolating function from McGaugh et al. 2016:
        g_obs = g_bar / (1 - exp(-sqrt(g_bar / g0)))

    Parameters
    ----------
    g_bar : array-like
        Baryonic gravitational acceleration in m/s² (must be > 0).
    g0 : float
        Characteristic acceleration scale in m/s².

    Returns
    -------
    np.ndarray
        Predicted observed acceleration in m/s².
    """
    g_bar = np.asarray(g_bar, dtype=float)
    x = np.sqrt(np.abs(g_bar) / g0)
    return g_bar / (1.0 - np.exp(-x))


# ---------------------------------------------------------------------------
# g0 fitting
# ---------------------------------------------------------------------------

def fit_g0(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_init: float = G0_DEFAULT,
) -> dict:
    """Fit the acceleration scale g0 to observed data via least-squares.

    Minimises the residuals in log10 space:
        residual_i = log10(g_obs_i) - log10(rar_g_obs(g_bar_i, g0))

    Parameters
    ----------
    g_bar : array-like
        Baryonic acceleration values (m/s²), must be > 0.
    g_obs : array-like
        Observed acceleration values (m/s²), must be > 0.
    g0_init : float, optional
        Initial guess for g0 (m/s²).

    Returns
    -------
    dict with keys:
        ``g0``      — best-fit value (m/s²)
        ``g0_err``  — 1-sigma uncertainty (m/s²) from covariance matrix
        ``rms``     — RMS of log10 residuals (dex)
        ``n``       — number of data points used
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    mask = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar) & np.isfinite(g_obs)
    gb = g_bar[mask]
    go = g_obs[mask]

    log_go = np.log10(go)

    def model(g_bar_arr, g0):
        return np.log10(rar_g_obs(g_bar_arr, g0))

    popt, pcov = curve_fit(model, gb, log_go, p0=[g0_init], bounds=(1e-12, 1e-8))
    g0_fit = float(popt[0])
    g0_err = float(np.sqrt(pcov[0, 0]))

    residuals = log_go - model(gb, g0_fit)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return {"g0": g0_fit, "g0_err": g0_err, "rms": rms, "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_rar(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Bin the RAR data in equal-width log10(g_bar) bins.

    Parameters
    ----------
    g_bar : array-like
        Baryonic acceleration values (m/s²).
    g_obs : array-like
        Observed acceleration values (m/s²).
    n_bins : int, optional
        Number of bins (default 20).

    Returns
    -------
    pd.DataFrame with columns:
        ``log_g_bar_bin`` — bin centre in log10(g_bar)
        ``log_g_obs_mean`` — mean of log10(g_obs) within the bin
        ``log_g_obs_std``  — standard deviation (dex)
        ``count``          — number of data points in bin
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    mask = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar) & np.isfinite(g_obs)
    log_gb = np.log10(g_bar[mask])
    log_go = np.log10(g_obs[mask])

    edges = np.linspace(log_gb.min(), log_gb.max(), n_bins + 1)

    mean_stat, _, bnum = binned_statistic(log_gb, log_go, statistic="mean", bins=edges)
    std_stat, _, _ = binned_statistic(log_gb, log_go, statistic="std", bins=edges)
    count_stat, _, _ = binned_statistic(log_gb, log_go, statistic="count", bins=edges)

    centres = 0.5 * (edges[:-1] + edges[1:])

    df = pd.DataFrame({
        "log_g_bar_bin": centres,
        "log_g_obs_mean": mean_stat,
        "log_g_obs_std": std_stat,
        "count": count_stat.astype(int),
    })
    return df.dropna(subset=["log_g_obs_mean"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deep-regime diagnostics
# ---------------------------------------------------------------------------

def deep_regime_slope(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0: float = G0_DEFAULT,
    percentile: float = 20.0,
) -> dict:
    """Check whether the deep-MOND regime (g_bar << g0) obeys g_obs ∝ sqrt(g_bar).

    In the deep-MOND limit: log10(g_obs) = 0.5*log10(g_bar) + 0.5*log10(g0)
    The expected slope in log-log space is 0.5.

    Parameters
    ----------
    g_bar : array-like
        Baryonic acceleration (m/s²).
    g_obs : array-like
        Observed acceleration (m/s²).
    g0 : float, optional
        Acceleration scale used to define the deep-regime threshold.
    percentile : float, optional
        Use the lowest ``percentile``% of g_bar values as the deep-regime sample.

    Returns
    -------
    dict with keys:
        ``slope``     — measured log-log slope in deep regime
        ``intercept`` — log-log intercept
        ``expected_slope`` — 0.5 (MOND prediction)
        ``n_deep``    — number of points in deep regime
        ``collapses`` — True if |slope - 0.5| < 0.15
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    mask = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar) & np.isfinite(g_obs)
    gb = g_bar[mask]
    go = g_obs[mask]

    threshold = np.percentile(gb, percentile)
    deep = gb <= threshold

    log_gb = np.log10(gb[deep])
    log_go = np.log10(go[deep])

    if deep.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan,
                "expected_slope": 0.5, "n_deep": int(deep.sum()),
                "collapses": False}

    coeffs = np.polyfit(log_gb, log_go, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    return {
        "slope": slope,
        "intercept": intercept,
        "expected_slope": 0.5,
        "n_deep": int(deep.sum()),
        "collapses": abs(slope - 0.5) < 0.15,
    }
