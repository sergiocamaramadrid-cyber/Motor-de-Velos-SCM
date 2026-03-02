"""
beta_fit.py — Vectorized β (deep-slope) fitting for the SCM pipeline.

The β exponent is the slope of log(g_obs) vs log(g_bar) in the deep
baryonic-acceleration regime.  MOND and the Motor de Velos both predict
β ≈ 0.5.

Provides:
    fit_beta_single  — fit β for one galaxy given arrays
    fit_beta_batch   — fit β for all galaxies in an rc_points DataFrame

Usage
-----
    from src.core.beta_fit import fit_beta_batch

    catalog = fit_beta_batch(rc_df)   # one row per galaxy
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .deep_regime import (
    compute_g_obs,
    compute_g_bar,
    deep_mask,
    CONV as _CONV,
    A0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VELO_INERTE_LO: float = 0.35   # lower bound of β consistent with velos/MOND
VELO_INERTE_HI: float = 0.65   # upper bound
MIN_DEEP_POINTS: int = 2        # minimum deep-regime points to compute β


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_beta_single(
    r_kpc: np.ndarray,
    vrot_kms: np.ndarray,
    vbar_kms: np.ndarray,
    a0: float = A0_DEFAULT,
    threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> Dict[str, object]:
    """Fit β for a single galaxy.

    Parameters
    ----------
    r_kpc : array_like
        Galactocentric radii in kpc.
    vrot_kms : array_like
        Observed rotation velocity in km/s.
    vbar_kms : array_like
        Baryonic velocity magnitude in km/s (sqrt(V_gas² + V_star²) or V_bar).
    a0 : float
        Characteristic acceleration in m/s².
    threshold : float
        Deep-regime threshold as fraction of a0.

    Returns
    -------
    dict with keys:
        beta              — slope of log g_obs vs log g_bar in deep regime
        beta_err          — standard error of beta
        r_value           — Pearson r of the deep-regime fit
        n_deep            — number of deep-regime points used
        velo_inerte_flag  — True if n_deep ≥ 2 and beta ∈ [0.35, 0.65]
    """
    result: Dict[str, object] = {
        "beta": float("nan"),
        "beta_err": float("nan"),
        "r_value": float("nan"),
        "n_deep": 0,
        "velo_inerte_flag": False,
    }

    r = np.asarray(r_kpc, dtype=float)
    vrot = np.asarray(vrot_kms, dtype=float)
    vbar = np.asarray(vbar_kms, dtype=float)

    g_obs = compute_g_obs(vrot, r)
    g_bar = compute_g_bar(vbar, r)

    valid = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar) & np.isfinite(g_obs)
    if valid.sum() < MIN_DEEP_POINTS:
        return result

    dm = deep_mask(g_bar, a0=a0, threshold=threshold) & valid
    n_deep = int(dm.sum())
    result["n_deep"] = n_deep

    if n_deep >= MIN_DEEP_POINTS:
        slope, _, r_val, _, stderr = linregress(
            np.log10(g_bar[dm]), np.log10(g_obs[dm])
        )
        result["beta"] = float(slope)
        result["beta_err"] = float(stderr)
        result["r_value"] = float(r_val)
        result["velo_inerte_flag"] = bool(VELO_INERTE_LO <= slope <= VELO_INERTE_HI)

    return result


def fit_beta_batch(
    rc_df: pd.DataFrame,
    a0: float = A0_DEFAULT,
    threshold: float = DEEP_THRESHOLD_DEFAULT,
    vbar_col: Optional[str] = None,
) -> pd.DataFrame:
    """Fit β for all galaxies in an rc_points-contract DataFrame.

    Auto-detects the baryonic velocity column:
        1. *vbar_col* if provided and present
        2. ``vbar_kms``  — direct combined baryonic velocity
        3. sqrt(``vgas_kms``² + ``vstar_kms``²)

    Parameters
    ----------
    rc_df : DataFrame
        Must contain columns: ``galaxy_id``, ``r_kpc``, ``vrot_kms``, and at
        least one of the baryonic velocity options above.
    a0 : float
        Characteristic acceleration in m/s².
    threshold : float
        Deep-regime threshold as fraction of a0.
    vbar_col : str, optional
        Explicit column name for baryonic velocity (overrides auto-detect).

    Returns
    -------
    DataFrame
        One row per galaxy with columns:
        ``galaxy_id``, ``beta``, ``beta_err``, ``r_value``, ``n_deep``,
        ``velo_inerte_flag``.
        Sorted by ``galaxy_id``.
    """
    rows: List[Dict[str, object]] = []

    for gid, grp in rc_df.groupby("galaxy_id", sort=True):
        grp = grp.reset_index(drop=True)

        # Resolve baryonic velocity
        if vbar_col and vbar_col in grp.columns:
            vbar = grp[vbar_col].values.astype(float)
        elif "vbar_kms" in grp.columns:
            vbar = grp["vbar_kms"].values.astype(float)
        elif "vgas_kms" in grp.columns and "vstar_kms" in grp.columns:
            vbar = np.sqrt(
                np.maximum(
                    grp["vgas_kms"].values.astype(float) ** 2
                    + grp["vstar_kms"].values.astype(float) ** 2,
                    0.0,
                )
            )
        else:
            rows.append({
                "galaxy_id": gid,
                "beta": float("nan"),
                "beta_err": float("nan"),
                "r_value": float("nan"),
                "n_deep": 0,
                "velo_inerte_flag": False,
            })
            continue

        stats = fit_beta_single(
            r_kpc=grp["r_kpc"].values,
            vrot_kms=grp["vrot_kms"].values,
            vbar_kms=vbar,
            a0=a0,
            threshold=threshold,
        )
        rows.append({"galaxy_id": gid, **stats})

    if not rows:
        return pd.DataFrame(
            columns=["galaxy_id", "beta", "beta_err", "r_value",
                     "n_deep", "velo_inerte_flag"]
        )

    df = pd.DataFrame(rows)[
        ["galaxy_id", "beta", "beta_err", "r_value", "n_deep", "velo_inerte_flag"]
    ]
    df["n_deep"] = df["n_deep"].astype(int)
    df["beta"] = df["beta"].astype(float)
    df["beta_err"] = df["beta_err"].astype(float)
    df["r_value"] = df["r_value"].astype(float)
    df["velo_inerte_flag"] = df["velo_inerte_flag"].astype(bool)
    return df.reset_index(drop=True)
