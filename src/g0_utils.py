"""g0_utils.py — Reusable guard-rail utilities for the g0 (a0) optimizer.

Provides :func:`g0_touches_bounds`, a log10-space proximity checker that
detects when a fitted Motor-de-Velos acceleration scale is suspiciously
close to the boundary of its search interval.
"""

from __future__ import annotations

import numpy as np


def g0_touches_bounds(
    g0_hat: float,
    bounds: tuple,
    tol_log10: float = 0.05,
) -> tuple:
    """Return (touches_lower, touches_upper) using log10-space proximity.

    Guards against non-finite or non-positive g0_hat and validates bounds
    before performing the log10 comparison.

    Parameters
    ----------
    g0_hat    : float  Fitted acceleration scale [m s⁻²].
    bounds    : (lo, hi)  Search interval [m s⁻²]; must satisfy 0 < lo < hi.
    tol_log10 : float  Proximity threshold in log10 decades (default 0.05).

    Returns
    -------
    (touches_lower, touches_upper) : (bool, bool)
        Both False when g0_hat is non-finite or <= 0.

    Raises
    ------
    ValueError  If bounds are invalid (non-finite, <= 0, or lo >= hi).
    """
    lo, hi = float(bounds[0]), float(bounds[1])

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(f"Invalid bounds: {bounds}")

    if (not np.isfinite(g0_hat)) or g0_hat <= 0:
        return False, False

    log10_g0 = float(np.log10(g0_hat))
    log10_lo = float(np.log10(lo))
    log10_hi = float(np.log10(hi))

    touches_lower = abs(log10_g0 - log10_lo) < float(tol_log10)
    touches_upper = abs(log10_g0 - log10_hi) < float(tol_log10)
    return bool(touches_lower), bool(touches_upper)
