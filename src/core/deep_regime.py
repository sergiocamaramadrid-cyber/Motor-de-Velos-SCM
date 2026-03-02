"""
deep_regime.py — Vectorized deep-regime filtering for galaxy rotation curves.

Provides three pure, NumPy-only functions:

    compute_g_obs   — centripetal acceleration from observed velocities
    compute_g_bar   — centripetal acceleration from baryonic velocities
    deep_mask       — boolean mask selecting the deep-MOND/velos regime

All accelerations are in m/s².

Unit convention
---------------
    velocity  : km/s
    radius    : kpc
    g_obs/bar : m/s²

Conversion factor:
    (km/s)² / kpc = 1e6 m²s⁻² / (3.085677581e19 m) = _CONV m/s²
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KPC_TO_M: float = 3.085677581e19        # 1 kpc → metres (IAU 2012)
CONV: float = 1e6 / KPC_TO_M            # (km/s)²/kpc → m/s²
_CONV = CONV                             # backward-compat alias
_MIN_R: float = 1e-10                   # guard: avoid dividing by near-zero radius

A0_DEFAULT: float = 1.2e-10             # characteristic acceleration, m/s²
DEEP_THRESHOLD_DEFAULT: float = 0.3    # deep regime: g_bar < threshold × a0
                                        # 0.3 × 1.2e-10 ≈ 3.6e-11 m/s²
                                        # (McGaugh+2016 deep-MOND convention)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_g_obs(vrot_kms: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    """Centripetal acceleration from the observed rotation curve.

    g_obs = V_obs² / r

    Parameters
    ----------
    vrot_kms : array_like
        Observed rotation velocity in km/s.
    r_kpc : array_like
        Galactocentric radius in kpc.

    Returns
    -------
    ndarray
        g_obs in m/s².
    """
    v = np.asarray(vrot_kms, dtype=float)
    r = np.maximum(np.asarray(r_kpc, dtype=float), _MIN_R)
    return v ** 2 / r * _CONV


def compute_g_bar(vbar_kms: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    """Centripetal acceleration from the baryonic velocity.

    g_bar = V_bar² / r

    Parameters
    ----------
    vbar_kms : array_like
        Baryonic velocity magnitude in km/s.  Pass the combined
        sqrt(V_gas² + V_star²) or V_bar directly.
    r_kpc : array_like
        Galactocentric radius in kpc.

    Returns
    -------
    ndarray
        g_bar in m/s².
    """
    vb = np.asarray(vbar_kms, dtype=float)
    r = np.maximum(np.asarray(r_kpc, dtype=float), _MIN_R)
    return vb ** 2 / r * _CONV


def deep_mask(
    g_bar: np.ndarray,
    a0: float = A0_DEFAULT,
    threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> np.ndarray:
    """Boolean mask selecting deep-regime radial points.

    Deep regime is defined as:   g_bar < threshold × a0

    Parameters
    ----------
    g_bar : array_like
        Baryonic centripetal acceleration in m/s².
    a0 : float
        Characteristic acceleration in m/s² (default 1.2e-10 m/s²).
    threshold : float
        Fraction of a0 used as the deep-regime cutoff (default 0.3).

    Returns
    -------
    ndarray of bool
    """
    return np.asarray(g_bar, dtype=float) < threshold * a0
