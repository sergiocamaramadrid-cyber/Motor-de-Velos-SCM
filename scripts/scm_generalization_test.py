"""
scripts/scm_generalization_test.py — SCM Leave-One-Out Generalization Test

Evaluates whether the SCM predictive model (slope_tail ~ E_SCM + logMbar)
generalises beyond the training sample using Leave-One-Out cross-validation
and a permutation null baseline.

Model
-----
    slope_tail  ~  env_proxy_formal  +  logMbar     (OLS)

where env_proxy_formal = E_SCM = log10(MHI / Rdisk²) is the formal HI surface
density proxy.

Metrics
-------
- RMSE_IS   : in-sample RMSE (fitted on all N galaxies)
- RMSE_LOO  : mean LOO RMSE across N folds
- rho_LOO   : Spearman ρ(y_true, y_pred_LOO)
- p_LOO     : two-sided p-value of rho_LOO
- RMSE_null_p95 : 95th percentile of RMSE from permuted-target null distribution

Public API
----------
load_dataset(path)                                   → pd.DataFrame
run_loo_cv(df, feature_cols, target_col)             → dict
run_permutation_baseline(df, feature_cols,
                         target_col, n_perm, seed)   → dict
summary_table(loo_result, perm_result)               → pd.DataFrame
main(argv=None)                                      → dict

Usage
-----
Default paths::

    python scripts/scm_generalization_test.py

Custom paths::

    python scripts/scm_generalization_test.py \\
        --dataset data/scm_canonical_dataset.csv \\
        --out results/generalization \\
        --n-perm 1000 \\
        --seed 42

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_DEFAULT = "data/scm_canonical_dataset.csv"
OUT_DIR_DEFAULT = "results/generalization"
N_PERM_DEFAULT = 1000
RANDOM_SEED_DEFAULT = 42

FEATURE_COLS_DEFAULT = ["env_proxy_formal", "logMbar"]
TARGET_COL_DEFAULT = "slope_tail"

# Required columns in the dataset
REQUIRED_COLS = {"galaxy", "logMbar", "slope_tail", "env_proxy_formal"}

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# 1. load_dataset
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the SCM canonical dataset.

    Parameters
    ----------
    path : str or Path
        CSV file with at minimum: ``galaxy``, ``logMbar``, ``slope_tail``,
        ``env_proxy_formal``.

    Returns
    -------
    pd.DataFrame
        Dataset with rows containing no NaN in the required columns dropped.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    df = pd.read_csv(p)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # Drop rows with NaN in required columns
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLS)).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        import warnings

        warnings.warn(
            f"load_dataset: dropped {dropped} rows with NaN in required columns.",
            UserWarning,
            stacklevel=2,
        )

    return df


# ---------------------------------------------------------------------------
# 2. _fit_ols  (internal helper)
# ---------------------------------------------------------------------------


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS and return coefficients (intercept first).

    Uses numpy least-squares when statsmodels is unavailable.
    """
    X_const = np.column_stack([np.ones(len(X)), X])
    if _HAS_STATSMODELS:
        res = sm.OLS(y, X_const).fit()
        return res.params
    # Fallback: numpy lstsq
    coef, *_ = np.linalg.lstsq(X_const, y, rcond=None)
    return coef


def _predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predict from OLS coefficients (intercept first)."""
    X_const = np.column_stack([np.ones(len(X)), X])
    return X_const @ coef


# ---------------------------------------------------------------------------
# 3. run_loo_cv
# ---------------------------------------------------------------------------


def run_loo_cv(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL_DEFAULT,
) -> dict:
    """Run Leave-One-Out cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature and target columns.
    feature_cols : list of str, optional
        Predictors.  Defaults to ``FEATURE_COLS_DEFAULT``.
    target_col : str
        Response variable.  Defaults to ``TARGET_COL_DEFAULT``.

    Returns
    -------
    dict with keys:
        ``n``           – number of galaxies
        ``rmse_is``     – in-sample RMSE
        ``rmse_loo``    – LOO RMSE
        ``rho_loo``     – Spearman ρ between y_true and y_pred_loo
        ``p_loo``       – two-sided p-value of rho_loo
        ``predictions`` – DataFrame(galaxy, y_true, y_pred_is, y_pred_loo)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_DEFAULT

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"run_loo_cv: missing columns {missing}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    n = len(df)

    # In-sample fit
    coef_is = _fit_ols(X, y)
    y_pred_is = _predict(coef_is, X)
    rmse_is = float(np.sqrt(np.mean((y - y_pred_is) ** 2)))

    # LOO
    y_pred_loo = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coef_i = _fit_ols(X[mask], y[mask])
        y_pred_loo[i] = _predict(coef_i, X[i : i + 1])[0]

    rmse_loo = float(np.sqrt(np.mean((y - y_pred_loo) ** 2)))
    rho, p_val = spearmanr(y, y_pred_loo)

    galaxy_col = df["galaxy"].values if "galaxy" in df.columns else np.arange(n)
    predictions = pd.DataFrame(
        {
            "galaxy": galaxy_col,
            "y_true": np.round(y, 6),
            "y_pred_is": np.round(y_pred_is, 6),
            "y_pred_loo": np.round(y_pred_loo, 6),
            "residual_loo": np.round(y - y_pred_loo, 6),
        }
    )

    return {
        "n": n,
        "rmse_is": round(rmse_is, 6),
        "rmse_loo": round(rmse_loo, 6),
        "rho_loo": round(float(rho), 6),
        "p_loo": round(float(p_val), 8),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# 4. run_permutation_baseline
# ---------------------------------------------------------------------------


def run_permutation_baseline(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL_DEFAULT,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = RANDOM_SEED_DEFAULT,
) -> dict:
    """Estimate the null LOO-RMSE distribution by permuting the target.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str, optional
    target_col : str
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        ``n_perm``            – actual number of permutations run
        ``rmse_null_mean``    – mean null RMSE
        ``rmse_null_std``     – std of null RMSE
        ``rmse_null_p95``     – 95th percentile of null RMSE
        ``rmse_null_p05``     – 5th percentile of null RMSE
        ``null_rmse_values``  – np.ndarray of length n_perm
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_DEFAULT

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed)

    null_rmse = np.empty(n_perm)
    for p in range(n_perm):
        y_perm = rng.permutation(y)
        loo_resid_sq = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            coef_i = _fit_ols(X[mask], y_perm[mask])
            pred_i = _predict(coef_i, X[i : i + 1])[0]
            loo_resid_sq[i] = (y_perm[i] - pred_i) ** 2
        null_rmse[p] = np.sqrt(np.mean(loo_resid_sq))

    return {
        "n_perm": n_perm,
        "rmse_null_mean": round(float(null_rmse.mean()), 6),
        "rmse_null_std": round(float(null_rmse.std()), 6),
        "rmse_null_p95": round(float(np.percentile(null_rmse, 95)), 6),
        "rmse_null_p05": round(float(np.percentile(null_rmse, 5)), 6),
        "null_rmse_values": null_rmse,
    }


# ---------------------------------------------------------------------------
# 5. summary_table
# ---------------------------------------------------------------------------


def summary_table(loo_result: dict, perm_result: dict) -> pd.DataFrame:
    """Assemble a summary DataFrame from LOO and permutation results.

    Parameters
    ----------
    loo_result : dict
        Output of :func:`run_loo_cv`.
    perm_result : dict
        Output of :func:`run_permutation_baseline`.

    Returns
    -------
    pd.DataFrame
        One row per metric with columns ``metric``, ``value``,
        ``interpretation``.
    """
    beats_null = loo_result["rmse_loo"] < perm_result["rmse_null_p95"]
    rows = [
        {
            "metric": "n_galaxies",
            "value": loo_result["n"],
            "interpretation": "Sample size",
        },
        {
            "metric": "rmse_is",
            "value": loo_result["rmse_is"],
            "interpretation": "In-sample RMSE",
        },
        {
            "metric": "rmse_loo",
            "value": loo_result["rmse_loo"],
            "interpretation": "LOO RMSE",
        },
        {
            "metric": "rho_loo",
            "value": loo_result["rho_loo"],
            "interpretation": f"Spearman rho LOO (p={loo_result['p_loo']:.4g})",
        },
        {
            "metric": "p_loo",
            "value": loo_result["p_loo"],
            "interpretation": "Two-sided p-value of rho_LOO",
        },
        {
            "metric": "rmse_null_p95",
            "value": perm_result["rmse_null_p95"],
            "interpretation": f"Null RMSE p95 (n_perm={perm_result['n_perm']})",
        },
        {
            "metric": "beats_null",
            "value": int(beats_null),
            "interpretation": "1 = model LOO RMSE < null p95",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the full generalization test pipeline.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments.  If *None*, uses ``sys.argv[1:]``.

    Returns
    -------
    dict with keys:
        ``dataset_path``, ``n``, ``loo``, ``permutation``, ``summary``,
        ``out_dir``.
    """
    parser = argparse.ArgumentParser(
        description="SCM Leave-One-Out Generalization Test"
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_DEFAULT,
        help=f"Path to canonical dataset CSV (default: {DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--out",
        default=OUT_DIR_DEFAULT,
        help=f"Output directory (default: {OUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=N_PERM_DEFAULT,
        help=f"Number of permutations for null baseline (default: {N_PERM_DEFAULT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED_DEFAULT,
        help=f"Random seed (default: {RANDOM_SEED_DEFAULT})",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load
    df = load_dataset(args.dataset)
    n = len(df)

    # --- LOO CV
    loo = run_loo_cv(df)

    # --- Permutation baseline
    perm = run_permutation_baseline(df, n_perm=args.n_perm, seed=args.seed)

    # --- Summary
    summary = summary_table(loo, perm)

    # --- Write outputs
    loo["predictions"].to_csv(out_dir / "loo_predictions.csv", index=False)

    perm_df = pd.DataFrame({"null_rmse": perm["null_rmse_values"]})
    perm_df.to_csv(out_dir / "permutation_baseline.csv", index=False)

    summary.to_csv(out_dir / "generalization_summary.csv", index=False)

    # JSON summary (exclude numpy array)
    json_payload = {
        "dataset_path": str(args.dataset),
        "n": n,
        "rmse_is": loo["rmse_is"],
        "rmse_loo": loo["rmse_loo"],
        "rho_loo": loo["rho_loo"],
        "p_loo": loo["p_loo"],
        "rmse_null_mean": perm["rmse_null_mean"],
        "rmse_null_std": perm["rmse_null_std"],
        "rmse_null_p95": perm["rmse_null_p95"],
        "beats_null": int(loo["rmse_loo"] < perm["rmse_null_p95"]),
    }
    with open(out_dir / "generalization_summary.json", "w") as fh:
        json.dump(json_payload, fh, indent=2)

    # Text report
    beats = "YES" if json_payload["beats_null"] else "NO"
    report = (
        f"{_SEP}\n"
        f"SCM Generalization Test — N={n}\n"
        f"{_SEP}\n"
        f"Model: slope_tail ~ env_proxy_formal + logMbar\n\n"
        f"In-sample RMSE : {loo['rmse_is']:.4f}\n"
        f"LOO RMSE       : {loo['rmse_loo']:.4f}\n"
        f"Spearman rho   : {loo['rho_loo']:.4f}  (p = {loo['p_loo']:.4g})\n\n"
        f"Null p95 RMSE  : {perm['rmse_null_p95']:.4f}  (n_perm={args.n_perm})\n"
        f"Beats null     : {beats}\n"
        f"{_SEP}\n"
    )
    with open(out_dir / "generalization_summary.txt", "w") as fh:
        fh.write(report)

    print(report, end="")

    return {
        "dataset_path": str(args.dataset),
        "n": n,
        "loo": {k: v for k, v in loo.items() if k != "predictions"},
        "permutation": {k: v for k, v in perm.items() if k != "null_rmse_values"},
        "summary": summary,
        "out_dir": str(out_dir),
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
