"""
population_stats.py — Population-level statistical summaries of the F3/β catalog.

Operates on a per-galaxy catalog DataFrame produced by
``src.core.beta_fit.fit_beta_batch`` or equivalent.  Expected columns:

    galaxy_id, beta, beta_err, r_value, n_deep, velo_inerte_flag

Optional columns (used when available):

    log_mstar   — log₁₀ stellar mass / M☉
    quality     — integer data-quality flag (e.g. 1 = best)
    survey      — survey name string

Public API
----------
    beta_summary(catalog)            — scalar statistics of the β distribution
    beta_vs_mass(catalog, ...)       — median β per mass bin
    beta_by_quality(catalog, ...)    — median β per quality tier
    beta_by_survey(catalog, ...)     — median β per survey label
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_col(df: pd.DataFrame, col: str, fn: str) -> None:
    if col not in df.columns:
        raise KeyError(f"{fn}: required column '{col}' not found in catalog.")


def _finite_beta(df: pd.DataFrame) -> pd.Series:
    """Return the beta column filtered to finite (non-NaN) values."""
    return df["beta"].dropna()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def beta_summary(catalog: pd.DataFrame) -> pd.Series:
    """Scalar statistics of the β distribution.

    Parameters
    ----------
    catalog : DataFrame
        Per-galaxy catalog with at least a ``beta`` column.

    Returns
    -------
    pd.Series
        Index: n_total, n_valid, n_velo_inerte, mean, median, std, q16, q84
    """
    _require_col(catalog, "beta", "beta_summary")
    beta = _finite_beta(catalog)
    n_vi = (
        catalog["velo_inerte_flag"].sum()
        if "velo_inerte_flag" in catalog.columns
        else np.nan
    )
    stats = {
        "n_total":       len(catalog),
        "n_valid":       len(beta),
        "n_velo_inerte": int(n_vi),
        "mean":          float(beta.mean()) if len(beta) else float("nan"),
        "median":        float(beta.median()) if len(beta) else float("nan"),
        "std":           float(beta.std(ddof=1)) if len(beta) > 1 else float("nan"),
        "q16":           float(np.nanpercentile(beta, 16)) if len(beta) else float("nan"),
        "q84":           float(np.nanpercentile(beta, 84)) if len(beta) else float("nan"),
    }
    return pd.Series(stats)


def beta_vs_mass(
    catalog: pd.DataFrame,
    mass_col: str = "log_mstar",
    n_bins: int = 6,
) -> pd.DataFrame:
    """Median β per stellar-mass bin.

    Parameters
    ----------
    catalog : DataFrame
        Must contain ``beta`` and *mass_col*.
    mass_col : str
        Column name for stellar mass (default ``log_mstar``).
    n_bins : int
        Number of equal-width bins (default 6).

    Returns
    -------
    DataFrame
        Columns: ``mass_bin_center``, ``beta_median``, ``beta_std``, ``n``
    """
    _require_col(catalog, "beta",    "beta_vs_mass")
    _require_col(catalog, mass_col, "beta_vs_mass")

    df = catalog[["beta", mass_col]].dropna()
    if df.empty:
        return pd.DataFrame(
            columns=["mass_bin_center", "beta_median", "beta_std", "n"]
        )

    bins = np.linspace(df[mass_col].min(), df[mass_col].max(), n_bins + 1)
    labels = 0.5 * (bins[:-1] + bins[1:])
    df = df.copy()
    df["_bin"] = pd.cut(df[mass_col], bins=bins, labels=labels, include_lowest=True)

    agg = (
        df.groupby("_bin", observed=True)["beta"]
        .agg(beta_median="median", beta_std="std", n="count")
        .reset_index()
        .rename(columns={"_bin": "mass_bin_center"})
    )
    agg["mass_bin_center"] = agg["mass_bin_center"].astype(float)
    return agg


def beta_by_quality(
    catalog: pd.DataFrame,
    quality_col: str = "quality",
) -> pd.DataFrame:
    """Median β per data-quality tier.

    Parameters
    ----------
    catalog : DataFrame
        Must contain ``beta`` and *quality_col*.
    quality_col : str
        Column name for quality flag (default ``quality``).

    Returns
    -------
    DataFrame
        Columns: *quality_col*, ``beta_median``, ``beta_std``, ``n``
    """
    _require_col(catalog, "beta",       "beta_by_quality")
    _require_col(catalog, quality_col, "beta_by_quality")

    df = catalog[[quality_col, "beta"]].dropna()
    agg = (
        df.groupby(quality_col)["beta"]
        .agg(beta_median="median", beta_std="std", n="count")
        .reset_index()
    )
    return agg


def beta_by_survey(
    catalog: pd.DataFrame,
    survey_col: str = "survey",
) -> pd.DataFrame:
    """Median β per survey label.

    Parameters
    ----------
    catalog : DataFrame
        Must contain ``beta`` and *survey_col*.
    survey_col : str
        Column name for survey identifier (default ``survey``).

    Returns
    -------
    DataFrame
        Columns: *survey_col*, ``beta_median``, ``beta_std``, ``n``
    """
    _require_col(catalog, "beta",      "beta_by_survey")
    _require_col(catalog, survey_col, "beta_by_survey")

    df = catalog[[survey_col, "beta"]].dropna()
    agg = (
        df.groupby(survey_col)["beta"]
        .agg(beta_median="median", beta_std="std", n="count")
        .reset_index()
    )
    return agg
