#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scm_validation.py — Statistical validation for SCM regime-transition tests.

Provides:
  - compute_score     : Spearman-based score for a candidate cut x_c
  - compute_p_perm    : Permutation p-value against the null hypothesis
  - compute_sigma_xc  : Bootstrap stability of the optimal cut
"""

import numpy as np
from scipy.stats import spearmanr


def compute_score(data, cut, Y, E, M):
    """Return |rho_high| - |rho_low| for a candidate mass cut.

    Parameters
    ----------
    data : pandas.DataFrame
    cut  : float  — mass threshold
    Y    : str    — target column (e.g. 'slope_tail')
    E    : str    — environment column (e.g. 'delta_mass_std')
    M    : str    — mass column (e.g. 'logM')

    Returns
    -------
    float or np.nan
    """
    low = data[data[M] < cut]
    high = data[data[M] >= cut]

    if len(low) < 10 or len(high) < 10:
        return np.nan

    rho_low, _ = spearmanr(low[E], low[Y])
    rho_high, _ = spearmanr(high[E], high[Y])

    return abs(rho_high) - abs(rho_low)


def compute_p_perm(df, cut, Y, E, M, n_perm=1000):
    """Permutation test for the regime-transition score at *cut*.

    Parameters
    ----------
    df     : pandas.DataFrame
    cut    : float  — mass threshold (best x_c from the run)
    Y      : str    — target column
    E      : str    — environment column (permuted)
    M      : str    — mass column
    n_perm : int    — number of permutations (default 1000)

    Returns
    -------
    p_perm     : float  — fraction of permuted scores >= observed (plus-1 corrected)
    real_score : float  — observed score at *cut*
    """
    real_score = compute_score(df, cut, Y, E, M)

    rng = np.random.default_rng(42)
    perm_scores = []

    for _ in range(n_perm):
        tmp = df.copy()
        tmp[E] = rng.permutation(tmp[E].values)
        perm_scores.append(compute_score(tmp, cut, Y, E, M))

    perm_scores = np.array(perm_scores)

    p_perm = (np.sum(perm_scores >= real_score) + 1) / (n_perm + 1)
    return float(p_perm), float(real_score)


def compute_sigma_xc(df, Y, E, M, n_boot=500, cut_min=9.0, cut_max=11.5, cut_step=0.05):
    """Bootstrap standard deviation of the optimal mass cut.

    Parameters
    ----------
    df       : pandas.DataFrame
    Y        : str    — target column
    E        : str    — environment column
    M        : str    — mass column
    n_boot   : int    — bootstrap iterations (default 500)
    cut_min  : float  — lower bound of the cut grid (default 9.0)
    cut_max  : float  — upper bound of the cut grid (default 11.5)
    cut_step : float  — step size of the cut grid (default 0.05)

    Returns
    -------
    float — standard deviation of the best cut across bootstrap samples
    """
    rng = np.random.default_rng(42)
    cuts = np.arange(cut_min, cut_max, cut_step)
    boot_cuts = []

    for _ in range(n_boot):
        sample = df.sample(len(df), replace=True, random_state=rng)
        scores = [(c, compute_score(sample, c, Y, E, M)) for c in cuts]
        scores = [(c, s) for c, s in scores if not np.isnan(s)]

        if scores:
            best_cut = max(scores, key=lambda x: x[1])[0]
            boot_cuts.append(best_cut)

    return float(np.std(boot_cuts)) if boot_cuts else float("nan")
