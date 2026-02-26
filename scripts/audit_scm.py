"""
scripts/audit_scm.py — Audit the SCM global feature model.

Fits an OLS linear model

    v_obs ~ logM + log_gbar + log_j

on a per-galaxy global CSV, performs k-fold cross-validation, and runs
a permutation test to assess overall model significance.

Outputs written to --outdir:
    audit_coefs.csv        — OLS coefficients and full-sample fit metrics
    audit_kfold.csv        — per-fold CV metrics (RMSE and R²)
    audit_permutations.csv — permutation-test results (R² distribution)
    audit.log              — full run log (also printed to stdout)

Usage
-----
    python scripts/audit_scm.py \\
        --input  data/_smoke_sparc_global.csv \\
        --outdir results/_smoke_audit \\
        --seed   123 \\
        --kfold  3 \\
        --permutations 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES = ["logM", "log_gbar", "log_j"]
TARGET = "v_obs"
GALAXY_COL = "galaxy_id"
_SEP = "=" * 64

# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build [1, logM, log_gbar, log_j] design matrix (intercept first)."""
    return np.column_stack([np.ones(len(df))] + [df[f].values for f in FEATURES])


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit OLS and return (coef, y_pred, r2, rmse).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix.
    y : ndarray, shape (n,)
        Target vector.

    Returns
    -------
    coef : ndarray, shape (p,)
    y_pred : ndarray, shape (n,)
    r2 : float
    rmse : float
    """
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return coef, y_pred, r2, rmse


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------


def kfold_cv(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """K-fold cross-validation on the OLS model.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with feature and target columns.
    k : int
        Number of folds.
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    pd.DataFrame
        Per-fold metrics with columns ``['fold', 'n_val', 'rmse', 'r2']``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    folds = np.array_split(idx, k)

    records = []
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train = _design_matrix(df.iloc[train_idx])
        y_train = df.iloc[train_idx][TARGET].values
        X_val = _design_matrix(df.iloc[val_idx])
        y_val = df.iloc[val_idx][TARGET].values

        coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        y_pred = X_val @ coef
        ss_res = float(np.sum((y_val - y_pred) ** 2))
        ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        records.append({"fold": i + 1, "n_val": len(val_idx), "rmse": rmse, "r2": r2})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def permutation_test(df: pd.DataFrame, n_perms: int, seed: int) -> dict:
    """Permutation test for overall model significance.

    Shuffles the target ``v_obs`` *n_perms* times and compares the resulting
    R² distribution to the true R².  The empirical p-value is the fraction
    of permuted R² values that are ≥ the true R².

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    n_perms : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        r2_true        — R² on the unshuffled data
        r2_perm_mean   — mean R² across permutations
        r2_perm_std    — std of permuted R²
        p_value        — empirical p-value
        n_permutations — n_perms
    """
    rng = np.random.default_rng(seed)
    X = _design_matrix(df)
    y = df[TARGET].values

    _, _, r2_true, _ = _ols_fit(X, y)

    perm_r2 = np.empty(n_perms)
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        _, _, r2_p, _ = _ols_fit(X, y_perm)
        perm_r2[i] = r2_p

    p_value = float(np.mean(perm_r2 >= r2_true))

    return {
        "r2_true": float(r2_true),
        "r2_perm_mean": float(np.mean(perm_r2)),
        "r2_perm_std": float(np.std(perm_r2)),
        "p_value": p_value,
        "n_permutations": n_perms,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_report(
    df: pd.DataFrame,
    coef_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    perm: dict,
    args: argparse.Namespace,
) -> list[str]:
    """Return a list of report lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — Global Feature Audit",
        _SEP,
        f"  Input        : {args.input}",
        f"  N_rows       : {len(df)}",
        f"  N_galaxies   : {df[GALAXY_COL].nunique()}",
        f"  Features     : {', '.join(FEATURES)}",
        f"  Target       : {TARGET}",
        f"  Seed         : {args.seed}",
        "",
        "  OLS coefficients (full sample)",
        "  " + "-" * 40,
    ]
    for _, row in coef_df.iterrows():
        lines.append(f"    {row['term']:<12} {row['coef']:+.6f}")

    r2_full = coef_df.loc[coef_df["term"] == "_r2", "coef"].values
    rmse_full = coef_df.loc[coef_df["term"] == "_rmse", "coef"].values
    lines += [
        "",
        f"  Full-sample R²   : {r2_full[0]:.4f}" if len(r2_full) else "",
        f"  Full-sample RMSE : {rmse_full[0]:.6f}" if len(rmse_full) else "",
        "",
        f"  {args.kfold}-fold cross-validation",
        "  " + "-" * 40,
        f"  {'Fold':<6} {'N_val':>6} {'RMSE':>10} {'R²':>8}",
    ]
    for _, row in cv_df.iterrows():
        lines.append(
            f"  {int(row['fold']):<6} {int(row['n_val']):>6} "
            f"{row['rmse']:>10.6f} {row['r2']:>8.4f}"
        )
    cv_rmse_mean = cv_df["rmse"].mean()
    cv_r2_mean = cv_df["r2"].mean()
    lines += [
        "  " + "-" * 40,
        f"  CV mean RMSE : {cv_rmse_mean:.6f}",
        f"  CV mean R²   : {cv_r2_mean:.4f}",
        "",
        f"  Permutation test  (n_perms={perm['n_permutations']})",
        "  " + "-" * 40,
        f"  True R²      : {perm['r2_true']:.4f}",
        f"  Perm R² mean : {perm['r2_perm_mean']:.4f}",
        f"  Perm R² std  : {perm['r2_perm_std']:.4f}",
        f"  p-value      : {perm['p_value']:.4f}",
        _SEP,
    ]
    return [ln for ln in lines if ln is not None]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the SCM global feature model with k-fold CV and permutation test."
    )
    parser.add_argument(
        "--input", required=True, metavar="CSV",
        help="Input CSV with columns: galaxy_id, logM, log_gbar, log_j, v_obs.",
    )
    parser.add_argument(
        "--outdir", required=True, metavar="DIR",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fold assignment and permutations (default: 42).",
    )
    parser.add_argument(
        "--kfold", type=int, default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--permutations", type=int, default=100,
        help="Number of permutations for the significance test (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {GALAXY_COL, TARGET} | set(FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Full-sample OLS fit
    X = _design_matrix(df)
    y = df[TARGET].values
    coef, _, r2_full, rmse_full = _ols_fit(X, y)

    coef_records = [{"term": "intercept", "coef": coef[0]}]
    for feat, c in zip(FEATURES, coef[1:]):
        coef_records.append({"term": feat, "coef": float(c)})
    coef_records.append({"term": "_r2", "coef": r2_full})
    coef_records.append({"term": "_rmse", "coef": rmse_full})
    coef_df = pd.DataFrame(coef_records)

    # K-fold CV
    cv_df = kfold_cv(df, k=args.kfold, seed=args.seed)

    # Permutation test
    perm = permutation_test(df, n_perms=args.permutations, seed=args.seed)
    perm_df = pd.DataFrame([perm])

    # Format report
    report_lines = _format_report(df, coef_df, cv_df, perm, args)
    for line in report_lines:
        print(line)

    # Write outputs
    coef_df.to_csv(out_dir / "audit_coefs.csv", index=False)
    cv_df.to_csv(out_dir / "audit_kfold.csv", index=False)
    perm_df.to_csv(out_dir / "audit_permutations.csv", index=False)
    (out_dir / "audit.log").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\n  Results written to {out_dir}")


if __name__ == "__main__":
    main()
