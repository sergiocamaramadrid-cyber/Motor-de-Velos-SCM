"""
bias_diagnostics.py — Selection-bias and survey-comparison diagnostics.

Operates on per-galaxy β catalogs.  Provides:

    selection_bias_check(catalog)    — checks whether the sample is biased
                                       toward high or low β (compares to the
                                       MOND/velos null hypothesis β = 0.5)
    survey_comparison(cat_a, cat_b)  — compares β distributions between two
                                       surveys (Mann-Whitney U test)
    n_deep_distribution(catalog)     — histogram of n_deep points per galaxy

All tests are non-parametric (rank-based) to be robust against non-normality.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


# Expected β under MOND / Motor de Velos deep prediction
_BETA_NULL: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_beta(df: pd.DataFrame) -> np.ndarray:
    return df["beta"].dropna().values


def _require_col(df: pd.DataFrame, col: str, fn: str) -> None:
    if col not in df.columns:
        raise KeyError(f"{fn}: required column '{col}' not found.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def selection_bias_check(catalog: pd.DataFrame) -> Dict[str, object]:
    """Test whether sample β values are centred on the null hypothesis β = 0.5.

    Uses a one-sample Wilcoxon signed-rank test against the hypothesis
    ``median(β) = 0.5``.

    Parameters
    ----------
    catalog : DataFrame
        Per-galaxy catalog with a ``beta`` column.

    Returns
    -------
    dict with keys:
        n             — number of galaxies with finite β
        beta_median   — sample median of β
        beta_mean     — sample mean of β
        wilcoxon_stat — W statistic
        p_value       — two-sided p-value (small ⟹ bias detected)
        biased        — True if p_value < 0.05
    """
    _require_col(catalog, "beta", "selection_bias_check")
    beta = _finite_beta(catalog)
    result: Dict[str, object] = {
        "n":             len(beta),
        "beta_median":   float(np.median(beta)) if len(beta) else float("nan"),
        "beta_mean":     float(np.mean(beta))   if len(beta) else float("nan"),
        "wilcoxon_stat": float("nan"),
        "p_value":       float("nan"),
        "biased":        False,
    }

    if len(beta) < 10:
        return result

    shifted = beta - _BETA_NULL
    # Drop zeros (Wilcoxon requires non-zero differences)
    shifted = shifted[shifted != 0.0]
    if len(shifted) < 10:
        return result

    stat, pval = wilcoxon(shifted, alternative="two-sided")
    result["wilcoxon_stat"] = float(stat)
    result["p_value"]       = float(pval)
    result["biased"]        = bool(pval < 0.05)
    return result


def survey_comparison(
    cat_a: pd.DataFrame,
    cat_b: pd.DataFrame,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, object]:
    """Compare β distributions between two survey catalogs.

    Uses a two-sample Mann-Whitney U test (non-parametric).

    Parameters
    ----------
    cat_a, cat_b : DataFrame
        Per-galaxy catalogs with a ``beta`` column.
    label_a, label_b : str
        Survey labels for reporting.

    Returns
    -------
    dict with keys:
        label_a, label_b
        n_a, n_b
        median_a, median_b
        mw_stat, p_value
        significant   — True if p_value < 0.05
    """
    _require_col(cat_a, "beta", "survey_comparison")
    _require_col(cat_b, "beta", "survey_comparison")

    beta_a = _finite_beta(cat_a)
    beta_b = _finite_beta(cat_b)

    result: Dict[str, object] = {
        "label_a":     label_a,
        "label_b":     label_b,
        "n_a":         len(beta_a),
        "n_b":         len(beta_b),
        "median_a":    float(np.median(beta_a)) if len(beta_a) else float("nan"),
        "median_b":    float(np.median(beta_b)) if len(beta_b) else float("nan"),
        "mw_stat":     float("nan"),
        "p_value":     float("nan"),
        "significant": False,
    }

    if len(beta_a) < 3 or len(beta_b) < 3:
        return result

    stat, pval = mannwhitneyu(beta_a, beta_b, alternative="two-sided")
    result["mw_stat"]     = float(stat)
    result["p_value"]     = float(pval)
    result["significant"] = bool(pval < 0.05)
    return result


def n_deep_distribution(catalog: pd.DataFrame) -> pd.DataFrame:
    """Histogram of n_deep per galaxy.

    Parameters
    ----------
    catalog : DataFrame
        Per-galaxy catalog with an ``n_deep`` column.

    Returns
    -------
    DataFrame
        Columns: ``n_deep``, ``count``, ``fraction``
    """
    _require_col(catalog, "n_deep", "n_deep_distribution")
    counts = (
        catalog["n_deep"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"n_deep": "n_deep", "count": "count"})
    )
    counts.columns = ["n_deep", "count"]
    counts["fraction"] = counts["count"] / counts["count"].sum()
    return counts
