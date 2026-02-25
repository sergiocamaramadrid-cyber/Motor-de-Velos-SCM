"""
scripts/audit_scm.py — Out-of-sample statistical audit of the SCM model.

Methodology
-----------
1. GroupKFold cross-validation, grouped by galaxy: the model is never
   trained on data from a galaxy that appears in the test fold.
   A simple OLS linear model (log_g_obs = a·log_g_bar + b) is fitted on the
   training galaxies and evaluated on the held-out galaxies.
   Metric: per-galaxy RMSE of (log_g_obs_pred − log_g_obs).

2. Permutation test: ``log_g_bar`` is shuffled *within* each galaxy to
   destroy the signal while preserving galaxy-level distributions.
   The p-value is the fraction of permutations whose mean cross-validated
   RMSE ≤ the observed RMSE (lower p → stronger evidence of real signal).

Strict checks (``--strict`` flag)
----------------------------------
* p-value (permutation) ≤ 0.05
* ≥ 50 % of galaxies show RMSE improvement over the null (real RMSE < median
  null RMSE for that galaxy).

Outputs (written to ``--out`` directory)
-----------------------------------------
  audit_summary.json         — top-level verdict, p-value, pass/fail flags
  groupkfold_per_galaxy.csv  — per-galaxy RMSE across folds
  groupkfold_per_point.csv   — per-point predictions and residuals
  permutation_test.csv       — mean OOS RMSE for each permutation

Usage
-----
::

    python scripts/audit_scm.py \\
        --input results/universal_term_comparison_full.csv \\
        --out results/audit/scm_v1 \\
        --n-splits 5 \\
        --n-perm 200 \\
        --seed 42 \\
        --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required columns (allow flexible galaxy-column aliases)
# ---------------------------------------------------------------------------

_GALAXY_ALIASES = ("galaxy", "galaxy_id", "Galaxy", "name")
_LOG_GBAR_ALIASES = ("log_g_bar", "log_gbar", "log_g_bar_obs")
_LOG_GOBS_ALIASES = ("log_g_obs", "log_gobs", "log_g_obs_obs")


def _find_col(df: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    for a in aliases:
        if a in df.columns:
            return a
    raise ValueError(
        f"Required column '{label}' not found in CSV. "
        f"Tried aliases: {list(aliases)}. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# GroupKFold (pure numpy — no scikit-learn dependency)
# ---------------------------------------------------------------------------

def _group_kfold_splits(
    groups: np.ndarray, n_splits: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) pairs split by group.

    Parameters
    ----------
    groups : 1-D array of group labels (one per sample).
    n_splits : number of folds.
    rng : random Generator for shuffling group order.

    Returns
    -------
    List of (train_indices, test_indices) as integer index arrays.
    """
    unique = np.array(sorted(set(groups)))
    rng.shuffle(unique)
    fold_assignments = np.array_split(unique, n_splits)
    splits = []
    all_idx = np.arange(len(groups))
    for fold_groups in fold_assignments:
        test_mask = np.isin(groups, fold_groups)
        splits.append((all_idx[~test_mask], all_idx[test_mask]))
    return splits


# ---------------------------------------------------------------------------
# OLS linear model helpers
# ---------------------------------------------------------------------------

def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = a·x + b by ordinary least squares.

    Returns
    -------
    (a, b) : slope and intercept.
    """
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.dot(x - x_mean, x - x_mean))
    if denom == 0.0:
        return 0.0, float(y_mean)
    a = float(np.dot(x - x_mean, y - y_mean) / denom)
    b = float(y_mean - a * x_mean)
    return a, b


def _ols_predict(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


# ---------------------------------------------------------------------------
# Cross-validation engine
# ---------------------------------------------------------------------------

def _run_cv(
    df: pd.DataFrame,
    col_galaxy: str,
    col_x: str,
    col_y: str,
    n_splits: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run GroupKFold CV.

    Returns
    -------
    per_galaxy : DataFrame with columns galaxy, fold, rmse, n_points.
    per_point  : DataFrame with all input rows plus pred and residual columns.
    """
    groups = df[col_galaxy].values
    x = df[col_x].values.astype(float)
    y = df[col_y].values.astype(float)

    splits = _group_kfold_splits(groups, n_splits, rng)

    point_records: list[dict] = []
    galaxy_records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        a, b = _ols_fit(x[train_idx], y[train_idx])
        y_pred_test = _ols_predict(x[test_idx], a, b)
        residuals = y[test_idx] - y_pred_test

        # per-point records
        for i, (idx, pred, res) in enumerate(
            zip(test_idx, y_pred_test, residuals)
        ):
            point_records.append(
                {
                    "fold": fold_idx,
                    col_galaxy: groups[idx],
                    col_x: x[idx],
                    col_y: y[idx],
                    "pred": pred,
                    "residual": res,
                }
            )

        # per-galaxy RMSE within this fold
        test_galaxies = np.unique(groups[test_idx])
        for gal in test_galaxies:
            gal_mask = groups[test_idx] == gal
            gal_res = residuals[gal_mask]
            rmse = float(np.sqrt(np.mean(gal_res ** 2)))
            galaxy_records.append(
                {
                    "fold": fold_idx,
                    col_galaxy: gal,
                    "rmse": rmse,
                    "n_points": int(gal_mask.sum()),
                    "slope": a,
                    "intercept": b,
                }
            )

    per_galaxy = pd.DataFrame(galaxy_records)
    per_point = pd.DataFrame(point_records)
    return per_galaxy, per_point


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _permute_within_groups(
    x: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle x within each group independently."""
    x_perm = x.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        x_perm[idx] = rng.permutation(x_perm[idx])
    return x_perm


def _cv_mean_rmse(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute mean per-galaxy RMSE across all folds."""
    rmse_list: list[float] = []
    for train_idx, test_idx in splits:
        a, b = _ols_fit(x[train_idx], y[train_idx])
        test_galaxies = np.unique(groups[test_idx])
        y_pred_test = _ols_predict(x[test_idx], a, b)
        residuals = y[test_idx] - y_pred_test
        for gal in test_galaxies:
            gal_mask = groups[test_idx] == gal
            rmse_list.append(float(np.sqrt(np.mean(residuals[gal_mask] ** 2))))
    return float(np.mean(rmse_list)) if rmse_list else float("nan")


def _run_permutation_test(
    df: pd.DataFrame,
    col_galaxy: str,
    col_x: str,
    col_y: str,
    n_splits: int,
    n_perm: int,
    seed: int,
    observed_rmse: float,
) -> tuple[pd.DataFrame, float]:
    """Run the permutation null test.

    log_g_bar is shuffled within each galaxy to destroy the between-galaxy
    signal while preserving within-galaxy distributions.

    Returns
    -------
    perm_df : DataFrame with columns perm_id, mean_rmse.
    p_value : fraction of null RMSEs ≤ observed_rmse.
    """
    rng = np.random.default_rng(seed + 1)  # different seed from CV split
    groups = df[col_galaxy].values
    x = df[col_x].values.astype(float)
    y = df[col_y].values.astype(float)

    # Fix the fold splits for all permutations (deterministic, same grouping)
    cv_rng = np.random.default_rng(seed)
    splits = _group_kfold_splits(groups, n_splits, cv_rng)

    null_rmses: list[float] = []
    for _ in range(n_perm):
        x_perm = _permute_within_groups(x, groups, rng)
        null_rmses.append(_cv_mean_rmse(x_perm, y, groups, splits))

    null_arr = np.array(null_rmses)
    # p-value: fraction of null runs whose RMSE is ≤ observed (one-sided)
    # A low p-value means the null rarely achieves RMSE as low as the real model
    p_value = float(np.mean(null_arr <= observed_rmse))

    perm_df = pd.DataFrame(
        {"perm_id": np.arange(n_perm), "mean_rmse": null_rmses}
    )
    return perm_df, p_value


# ---------------------------------------------------------------------------
# Fraction of galaxies that improve over the null
# ---------------------------------------------------------------------------

def _frac_galaxies_improved(
    per_galaxy_real: pd.DataFrame,
    perm_df: pd.DataFrame,
    df: pd.DataFrame,
    col_galaxy: str,
    col_x: str,
    col_y: str,
    n_splits: int,
    seed: int,
) -> float:
    """Compute fraction of galaxies where real RMSE < median null RMSE.

    For each galaxy that appears in the OOS folds, compare its observed RMSE
    to the median RMSE it would have under the null distribution.
    """
    # Build per-galaxy null RMSEs from n_perm permutations
    rng_perm = np.random.default_rng(seed + 1)
    groups = df[col_galaxy].values
    x = df[col_x].values.astype(float)
    y = df[col_y].values.astype(float)

    cv_rng = np.random.default_rng(seed)
    splits = _group_kfold_splits(groups, n_splits, cv_rng)

    n_perm = len(perm_df)
    # Per-galaxy null RMSE accumulator
    gal_null: dict[str, list[float]] = {}

    for _ in range(n_perm):
        x_perm = _permute_within_groups(x, groups, rng_perm)
        for train_idx, test_idx in splits:
            a, b = _ols_fit(x_perm[train_idx], y[train_idx])
            y_pred_test = _ols_predict(x_perm[test_idx], a, b)
            residuals = y[test_idx] - y_pred_test
            for gal in np.unique(groups[test_idx]):
                gal_mask = groups[test_idx] == gal
                rmse = float(np.sqrt(np.mean(residuals[gal_mask] ** 2)))
                gal_null.setdefault(gal, []).append(rmse)

    # Per-galaxy real RMSE (average across folds if galaxy appears multiple times)
    real_rmse_by_gal = (
        per_galaxy_real.groupby(col_galaxy)["rmse"].mean().to_dict()
    )

    improved = 0
    total = 0
    for gal, null_list in gal_null.items():
        if gal not in real_rmse_by_gal:
            continue
        median_null = float(np.median(null_list))
        if real_rmse_by_gal[gal] < median_null:
            improved += 1
        total += 1

    return float(improved / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main audit routine
# ---------------------------------------------------------------------------

def run_audit(
    input_csv: Path,
    out_dir: Path,
    n_splits: int = 5,
    n_perm: int = 200,
    seed: int = 42,
    strict: bool = False,
) -> dict:
    """Run the full SCM audit and return the summary dict.

    Parameters
    ----------
    input_csv : Path
        CSV file with per-radial-point data. Must contain galaxy, log_g_bar,
        and log_g_obs columns (or accepted aliases).
    out_dir : Path
        Directory to write all output artefacts.
    n_splits : int
        Number of GroupKFold splits.
    n_perm : int
        Number of permutations for the null test.
    seed : int
        Master random seed (used for CV splits and permutations).
    strict : bool
        If True, raises SystemExit with code 1 when strict checks fail.

    Returns
    -------
    dict : audit_summary contents.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and validate input
    # ------------------------------------------------------------------
    df = pd.read_csv(input_csv)
    col_galaxy = _find_col(df, _GALAXY_ALIASES, "galaxy")
    col_x = _find_col(df, _LOG_GBAR_ALIASES, "log_g_bar")
    col_y = _find_col(df, _LOG_GOBS_ALIASES, "log_g_obs")

    # Drop rows with missing values in the three key columns
    df = df[[col_galaxy, col_x, col_y]].dropna().reset_index(drop=True)

    n_galaxies = df[col_galaxy].nunique()
    n_points = len(df)

    if n_galaxies < n_splits:
        raise ValueError(
            f"n_splits={n_splits} but only {n_galaxies} galaxies available. "
            "Reduce --n-splits."
        )

    # ------------------------------------------------------------------
    # GroupKFold cross-validation
    # ------------------------------------------------------------------
    cv_rng = np.random.default_rng(seed)
    per_galaxy, per_point = _run_cv(
        df, col_galaxy, col_x, col_y, n_splits, cv_rng
    )

    # Observed mean OOS RMSE
    observed_rmse = float(per_galaxy["rmse"].mean())

    # ------------------------------------------------------------------
    # Permutation test
    # ------------------------------------------------------------------
    perm_df, p_value = _run_permutation_test(
        df, col_galaxy, col_x, col_y,
        n_splits, n_perm, seed, observed_rmse,
    )

    # ------------------------------------------------------------------
    # Fraction of galaxies improved over null
    # ------------------------------------------------------------------
    frac_improved = _frac_galaxies_improved(
        per_galaxy, perm_df, df, col_galaxy, col_x, col_y, n_splits, seed
    )

    # ------------------------------------------------------------------
    # Strict checks
    # ------------------------------------------------------------------
    p_pass = p_value <= 0.05
    frac_pass = (not np.isnan(frac_improved)) and (frac_improved >= 0.50)
    overall_pass = p_pass and frac_pass

    verdict = "PASS" if overall_pass else "FAIL"
    null_mean_rmse = float(perm_df["mean_rmse"].mean())

    summary = {
        "input_csv": str(input_csv),
        "n_galaxies": n_galaxies,
        "n_points": n_points,
        "n_splits": n_splits,
        "n_perm": n_perm,
        "seed": seed,
        "observed_mean_oos_rmse": observed_rmse,
        "null_mean_rmse": null_mean_rmse,
        "p_value_permutation": p_value,
        "frac_galaxies_improved": frac_improved,
        "strict_checks": {
            "p_value_le_0.05": p_pass,
            "frac_improved_ge_0.50": frac_pass,
        },
        "verdict": verdict,
    }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    per_galaxy.to_csv(out_dir / "groupkfold_per_galaxy.csv", index=False)
    per_point.to_csv(out_dir / "groupkfold_per_point.csv", index=False)
    perm_df.to_csv(out_dir / "permutation_test.csv", index=False)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Out-of-sample statistical audit of the SCM model using "
            "GroupKFold cross-validation and permutation testing."
        )
    )
    parser.add_argument(
        "--input", required=True, metavar="CSV",
        help="Input CSV with galaxy, log_g_bar, log_g_obs columns.",
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output directory for audit artefacts.",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, dest="n_splits",
        help="Number of GroupKFold splits (default: 5).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=200, dest="n_perm",
        help="Number of permutations for the null test (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Exit with code 1 if strict checks fail "
            "(p-value > 0.05 or < 50%% of galaxies improved)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_audit(
        input_csv=Path(args.input),
        out_dir=Path(args.out),
        n_splits=args.n_splits,
        n_perm=args.n_perm,
        seed=args.seed,
        strict=args.strict,
    )

    # Print summary to stdout
    sep = "=" * 60
    print(sep)
    print("  SCM Audit Summary")
    print(sep)
    print(f"  Galaxies       : {summary['n_galaxies']}")
    print(f"  Radial points  : {summary['n_points']}")
    print(f"  GroupKFold k   : {summary['n_splits']}")
    print(f"  Permutations   : {summary['n_perm']}")
    print(f"  Seed           : {summary['seed']}")
    print()
    print(f"  OOS RMSE (real): {summary['observed_mean_oos_rmse']:.6f}")
    print(f"  OOS RMSE (null): {summary['null_mean_rmse']:.6f}")
    print(f"  p-value        : {summary['p_value_permutation']:.4f}")
    print(f"  Frac improved  : {summary['frac_galaxies_improved']:.3f}")
    print()
    checks = summary["strict_checks"]
    print(f"  p-value ≤ 0.05 : {'PASS' if checks['p_value_le_0.05'] else 'FAIL'}")
    print(f"  frac ≥ 50%     : {'PASS' if checks['frac_improved_ge_0.50'] else 'FAIL'}")
    print()
    print(f"  Verdict        : {summary['verdict']}")
    print(sep)
    print(f"\n  Results written to {args.out}/")

    if args.strict and summary["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
