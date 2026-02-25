"""
diagnostics.py — Multicollinearity and structural diagnostics for the SCM feature set.

PR #57 — Multicollinearity & Structural Diagnostics (VIF Layer)
Sergio Cámara Madrid — Motor de Velos SCM v0.6.0

Analyses the internal feature geometry of the SCM predictor set::

    {log M_bar,  log g_bar,  log j_*}

Diagnostics implemented
-----------------------
1. Variance Inflation Factor (VIF) per predictor  → ``vif_results.csv``
2. Condition number of the standardised design matrix → ``condition_number.txt``
3. Partial correlation of each predictor with log g_obs  → ``partial_correlation.json``
4. Combined summary  → ``diagnostics_summary.json``

Frozen constants (v0.6.0)
--------------------------
``G0_HINGE`` = 3.5 × 10⁻¹¹ m/s²  — the SCM hinge acceleration at which the
Velo Inerte begins to dominate galactic dynamics.  This value is *frozen* for
all v0.6.0 analyses and must not be altered by downstream scripts.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from .scm_models import KPC_TO_M

# ---------------------------------------------------------------------------
# Frozen SCM constants (v0.6.0)
# ---------------------------------------------------------------------------

#: Frozen hinge acceleration (m/s²).  Below this threshold the Velo Inerte
#: dominates; above it the baryonic (Newtonian) term is sufficient.
#: Registered under the Modelo de Condensación Fluida — Sergio Cámara Madrid.
G0_HINGE: float = 3.5e-11

# ---------------------------------------------------------------------------
# VIF interpretation thresholds
# ---------------------------------------------------------------------------

VIF_THRESHOLD_LOW: float = 5.0   # VIF < 5  → Acceptable
VIF_THRESHOLD_HIGH: float = 10.0  # VIF ≥ 10 → Problematic

# Condition-number interpretation thresholds
COND_THRESHOLD_MODERATE: float = 30.0
COND_THRESHOLD_ILL: float = 100.0


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _log_j_star(r_kpc: np.ndarray, g_obs: np.ndarray) -> np.ndarray:
    """Compute log₁₀(j_*) from radius and observed centripetal acceleration.

    The local specific angular momentum is approximated as::

        j_*(r) = r · v_c(r)

    where the circular velocity is recovered from the centripetal relation::

        v_c(r) = sqrt(g_obs · r · KPC_TO_M) / 1 000   [km/s]

    so::

        j_*(r) = r_kpc · v_c_kms   [kpc km/s]

    All inputs must be strictly positive.

    Parameters
    ----------
    r_kpc : ndarray
        Galactocentric radii in kpc.
    g_obs : ndarray
        Observed centripetal acceleration in m/s².

    Returns
    -------
    ndarray
        log₁₀(j_*) in units of kpc km/s.
    """
    v_c_kms = np.sqrt(np.maximum(g_obs * r_kpc * KPC_TO_M, 0.0)) / 1.0e3
    j_star = r_kpc * v_c_kms
    return np.log10(np.maximum(j_star, 1.0e-30))


def build_feature_matrix(
    compare_df: pd.DataFrame,
    per_galaxy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-radial-point SCM feature matrix.

    Merges per-point kinematic data with galaxy-level baryonic mass to
    produce a row for every valid (galaxy, radius) pair with columns:

    ``['galaxy', 'r_kpc', 'log_M_bar', 'log_g_bar', 'log_j_star',
       'log_v_obs', 'velo_dominated']``

    All logarithms are base-10.  The ``velo_dominated`` flag marks points
    where ``g_bar < G0_HINGE`` (Velo Inerte regime).

    Parameters
    ----------
    compare_df : pd.DataFrame
        Per-radial-point data produced by the SCM pipeline
        (``universal_term_comparison_full.csv``).
        Required columns: ``galaxy``, ``r_kpc``, ``g_bar``, ``g_obs``.
    per_galaxy_df : pd.DataFrame
        Per-galaxy summary (``per_galaxy_summary.csv``).
        Required columns: ``galaxy``, ``M_bar_BTFR_Msun``.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per valid (galaxy, radius) point.
    """
    df = compare_df.merge(
        per_galaxy_df[["galaxy", "M_bar_BTFR_Msun"]],
        on="galaxy",
        how="inner",
    )

    valid = (
        (df["M_bar_BTFR_Msun"] > 0)
        & (df["g_bar"] > 0)
        & (df["g_obs"] > 0)
        & (df["r_kpc"] > 0)
    )
    df = df[valid].copy().reset_index(drop=True)

    df["log_M_bar"] = np.log10(df["M_bar_BTFR_Msun"])
    df["log_g_bar"] = np.log10(df["g_bar"])
    df["log_j_star"] = _log_j_star(df["r_kpc"].values, df["g_obs"].values)
    df["log_v_obs"] = np.log10(
        np.sqrt(np.maximum(df["g_obs"].values * df["r_kpc"].values * KPC_TO_M, 1e-30))
        / 1.0e3  # km/s
    )
    df["velo_dominated"] = df["g_bar"] < G0_HINGE

    return df[
        ["galaxy", "r_kpc", "log_M_bar", "log_g_bar", "log_j_star",
         "log_v_obs", "velo_dominated"]
    ].reset_index(drop=True)


def build_audit_table(
    compare_df: pd.DataFrame,
    per_galaxy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the galaxy-level audit table (``sparc_global.csv``).

    Aggregates per-radial-point features to one row per galaxy using the
    median.  The resulting table:

    * contains exactly the galaxies present in *per_galaxy_df*
    * has a gapless integer index (0 … N-1)
    * uses base-10 logarithms for all scale-invariant quantities
    * marks galaxies where the median baryonic acceleration is below
      ``G0_HINGE`` as Velo-dominated

    Parameters
    ----------
    compare_df : pd.DataFrame
        Per-radial-point pipeline output.
    per_galaxy_df : pd.DataFrame
        Per-galaxy pipeline summary.

    Returns
    -------
    pd.DataFrame
        Columns:
        ``['galaxy', 'log_M_bar', 'log_g_bar', 'log_j_star',
           'log_v_obs', 'velo_dominated', 'n_points']``
    """
    feat = build_feature_matrix(compare_df, per_galaxy_df)

    agg = (
        feat.groupby("galaxy")
        .agg(
            log_M_bar=("log_M_bar", "median"),
            log_g_bar=("log_g_bar", "median"),
            log_j_star=("log_j_star", "median"),
            log_v_obs=("log_v_obs", "median"),
            n_points=("r_kpc", "count"),
        )
        .reset_index()
    )

    # Propagate galaxy-level M_bar for any galaxy with no radial points
    all_galaxies = per_galaxy_df[
        per_galaxy_df["M_bar_BTFR_Msun"] > 0
    ][["galaxy", "M_bar_BTFR_Msun"]].copy()
    all_galaxies["log_M_bar_ref"] = np.log10(all_galaxies["M_bar_BTFR_Msun"])

    audit = all_galaxies.merge(agg, on="galaxy", how="left")

    # For galaxies with no per-point data, fill log_M_bar from galaxy table
    missing_log_m = audit["log_M_bar"].isna()
    audit.loc[missing_log_m, "log_M_bar"] = audit.loc[
        missing_log_m, "log_M_bar_ref"
    ]
    audit.loc[audit["n_points"].isna(), "n_points"] = 0

    # Velo-dominated flag: True when median g_bar < G0_HINGE
    audit["velo_dominated"] = 10.0 ** audit["log_g_bar"].fillna(np.nan) < G0_HINGE

    audit = audit.sort_values("galaxy").reset_index(drop=True)

    return audit[
        ["galaxy", "log_M_bar", "log_g_bar", "log_j_star",
         "log_v_obs", "velo_dominated", "n_points"]
    ]


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factor (VIF) for each predictor column.

    For each column *i*:

    .. math::

        \\mathrm{VIF}(X_i) = \\frac{1}{1 - R_i^2}

    where :math:`R_i^2` is the coefficient of determination from regressing
    :math:`X_i` on all remaining columns (with intercept).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix of *p* predictors for *n* observations.
        No intercept column should be included.

    Returns
    -------
    ndarray, shape (p,)
        VIF for each predictor.  Returns ``inf`` when a predictor is an
        exact linear combination of the others.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    vifs = np.empty(p)
    for i in range(p):
        X_other = np.delete(X, i, axis=1)
        X_aug = np.column_stack([np.ones(n), X_other])
        coef, _, _, _ = np.linalg.lstsq(X_aug, X[:, i], rcond=None)
        y_hat = X_aug @ coef
        ss_res = float(np.sum((X[:, i] - y_hat) ** 2))
        ss_tot = float(np.sum((X[:, i] - X[:, i].mean()) ** 2))
        if ss_tot < 1.0e-12:
            vifs[i] = np.inf
        else:
            r2 = 1.0 - ss_res / ss_tot
            vifs[i] = 1.0 / (1.0 - r2) if r2 < 1.0 - 1.0e-12 else np.inf
    return vifs


# ---------------------------------------------------------------------------
# Condition number
# ---------------------------------------------------------------------------

def compute_condition_number(X: np.ndarray) -> float:
    """Condition number κ of the standardised design matrix.

    .. math::

        \\kappa = \\frac{\\sigma_{\\max}}{\\sigma_{\\min}}

    where :math:`\\sigma` are singular values of the column-standardised
    (zero-mean, unit-variance) version of *X*.

    Parameters
    ----------
    X : ndarray, shape (n, p)

    Returns
    -------
    float
        Condition number.  Returns ``inf`` when the matrix is singular.
    """
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0)
    std = np.where(std < 1.0e-12, 1.0, std)
    X_std = (X - X.mean(axis=0)) / std
    sv = np.linalg.svd(X_std, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 1.0e-12 else np.inf


# ---------------------------------------------------------------------------
# Partial correlations
# ---------------------------------------------------------------------------

def compute_partial_correlations(
    X: np.ndarray,
    y: np.ndarray,
    predictor_names: list,
) -> dict:
    """Partial correlation of each predictor in *X* with response *y*.

    For predictor *i*, the partial correlation with *y* controlling for all
    other predictors is the Pearson correlation between:

    * the residuals of regressing :math:`X_i` on the other columns, and
    * the residuals of regressing :math:`y` on the other columns.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix (no intercept).
    y : ndarray, shape (n,)
        Response variable.
    predictor_names : list of str
        Names for each column of *X*.

    Returns
    -------
    dict
        ``{predictor_name: partial_correlation}`` where values are in [-1, 1].
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    result = {}
    for i, name in enumerate(predictor_names):
        X_other = np.delete(X, i, axis=1)
        X_aug = np.column_stack([np.ones(n), X_other])

        coef_xi, _, _, _ = np.linalg.lstsq(X_aug, X[:, i], rcond=None)
        resid_xi = X[:, i] - X_aug @ coef_xi

        coef_y, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        resid_y = y - X_aug @ coef_y

        std_xi = float(resid_xi.std())
        std_y = float(resid_y.std())
        if std_xi < 1.0e-12 or std_y < 1.0e-12:
            result[name] = 0.0
        else:
            result[name] = float(np.corrcoef(resid_xi, resid_y)[0, 1])
    return result


# ---------------------------------------------------------------------------
# Master diagnostics runner
# ---------------------------------------------------------------------------

def run_diagnostics(
    feature_df: pd.DataFrame,
    out_dir,
    verbose: bool = True,
) -> dict:
    """Run the full PR #57 multicollinearity and structural diagnostics.

    Uses the per-radial-point feature matrix produced by
    :func:`build_feature_matrix`.  The VIF, condition number and partial
    correlations are computed on *galaxy-level medians* to avoid
    pseudo-replication inflating the sample size.

    Outputs written to *out_dir*
    ----------------------------
    * ``vif_results.csv``
    * ``condition_number.txt``
    * ``partial_correlation.json``
    * ``diagnostics_summary.json``

    Parameters
    ----------
    feature_df : pd.DataFrame
        Per-radial-point feature matrix (output of :func:`build_feature_matrix`).
        Required columns: ``galaxy``, ``log_M_bar``, ``log_g_bar``,
        ``log_j_star``, ``log_v_obs``.
    out_dir : str or Path
        Directory for output files (created if needed).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Full diagnostics summary (also written as ``diagnostics_summary.json``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate to galaxy-level medians to remove within-galaxy correlation
    predictor_cols = ["log_M_bar", "log_g_bar", "log_j_star"]
    response_col = "log_v_obs"

    gal = (
        feature_df.groupby("galaxy")[predictor_cols + [response_col]]
        .median()
        .dropna()
        .reset_index()
    )

    X = gal[predictor_cols].values
    y = gal[response_col].values
    n_obs = len(gal)

    if verbose:
        print(
            f"[diagnostics] n_galaxies={n_obs}  "
            f"G0_HINGE={G0_HINGE:.2e} m/s² (frozen)"
        )

    # ------------------------------------------------------------------ VIF
    vifs = compute_vif(X)
    vif_records = []
    for name, v in zip(predictor_cols, vifs):
        if v < VIF_THRESHOLD_LOW:
            interpretation = "Acceptable"
        elif v < VIF_THRESHOLD_HIGH:
            interpretation = "Moderate collinearity"
        else:
            interpretation = "Problematic"
        vif_records.append(
            {"predictor": name, "VIF": round(float(v), 6),
             "interpretation": interpretation}
        )
    vif_df = pd.DataFrame(vif_records)
    vif_df.to_csv(out_dir / "vif_results.csv", index=False)
    if verbose:
        print("[diagnostics] VIF results:")
        print(vif_df.to_string(index=False))

    # ------------------------------------------- Condition number
    kappa = compute_condition_number(X)
    if kappa < COND_THRESHOLD_MODERATE:
        cond_interp = "Stable"
    elif kappa < COND_THRESHOLD_ILL:
        cond_interp = "Moderate sensitivity"
    else:
        cond_interp = "Ill-conditioned"
    cond_lines = [
        f"condition_number: {kappa:.6f}",
        f"interpretation: {cond_interp}",
        f"threshold_moderate: {COND_THRESHOLD_MODERATE}",
        f"threshold_ill_conditioned: {COND_THRESHOLD_ILL}",
        f"g0_hinge_frozen: {G0_HINGE}",
    ]
    (out_dir / "condition_number.txt").write_text(
        "\n".join(cond_lines) + "\n", encoding="utf-8"
    )
    if verbose:
        print(f"[diagnostics] Condition number: {kappa:.4f} ({cond_interp})")

    # -------------------------------------- Partial correlations (vs log_v_obs)
    partials = compute_partial_correlations(X, y, predictor_cols)
    with open(out_dir / "partial_correlation.json", "w", encoding="utf-8") as fh:
        json.dump(partials, fh, indent=2)
    if verbose:
        print(f"[diagnostics] Partial correlations (vs log_v_obs): {partials}")

    # ------------------------------------------------------ Combined summary
    summary = {
        "scm_version": "v0.6.0",
        "g0_hinge_frozen_m_s2": G0_HINGE,
        "n_galaxies": n_obs,
        "predictors": predictor_cols,
        "vif": {r["predictor"]: r["VIF"] for r in vif_records},
        "vif_interpretations": {
            r["predictor"]: r["interpretation"] for r in vif_records
        },
        "condition_number": round(float(kappa), 6),
        "condition_number_interpretation": cond_interp,
        "partial_correlations_vs_log_v_obs": partials,
    }
    with open(out_dir / "diagnostics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if verbose:
        print(f"[diagnostics] Summary written to {out_dir}")

    return summary
