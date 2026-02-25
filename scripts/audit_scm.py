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
   The p-value uses the corrected estimator::

       p = (1 + #{null RMSE ≤ observed RMSE}) / (N_perm + 1)

   This avoids a p-value of exactly 0 and is the standard form for
   permutation tests (Phipson & Smyth 2010).

3. Per-galaxy p-values are computed with the same corrected formula and
   corrected for multiple comparisons using the Benjamini-Hochberg (FDR)
   procedure.

Strict checks (``--strict`` flag)
----------------------------------
* Global p-value ≤ 0.05
* ≥ 50 % of galaxies show RMSE improvement over the null (real RMSE < median
  null RMSE for that galaxy).

Outputs (written to ``--out`` directory)
-----------------------------------------
  audit_summary.json         — top-level verdict, p-value, folds detail,
                               perm RMSE mean/std, and all key metrics
  groupkfold_per_galaxy.csv  — one row per galaxy: mean/std RMSE across folds,
                               null RMSE stats, per-galaxy p-value (raw + BH)
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
# Multiple-comparisons correction (Benjamini-Hochberg FDR)
# ---------------------------------------------------------------------------

def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for an array of p-values.

    NaN entries are preserved as NaN in the output.

    Parameters
    ----------
    p_values : 1-D array of raw p-values (may contain NaN).

    Returns
    -------
    1-D array of BH-adjusted p-values, clipped to [0, 1].
    """
    result = p_values.copy().astype(float)
    valid_mask = ~np.isnan(p_values)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return result

    p_valid = p_values[valid_mask].astype(float)
    sorted_idx = np.argsort(p_valid)          # ascending rank
    ranks = np.empty(n_valid, dtype=float)
    ranks[sorted_idx] = np.arange(1, n_valid + 1)

    # BH adjusted: p_adj[i] = p[i] * m / rank[i]
    adjusted = np.minimum(p_valid * n_valid / ranks, 1.0)

    # Enforce monotonicity (take cumulative min from the right in sorted order)
    sorted_adjusted = adjusted[sorted_idx]
    for i in range(n_valid - 2, -1, -1):
        sorted_adjusted[i] = min(sorted_adjusted[i], sorted_adjusted[i + 1])
    adjusted[sorted_idx] = sorted_adjusted

    result[valid_mask] = adjusted
    return result


# ---------------------------------------------------------------------------
# Permutation analysis (single pass — global + per-galaxy)
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


def _run_permutation_analysis(
    df: pd.DataFrame,
    col_galaxy: str,
    col_x: str,
    col_y: str,
    n_splits: int,
    n_perm: int,
    seed: int,
    observed_rmse: float,
    real_rmse_by_gal: dict[str, float],
) -> tuple[pd.DataFrame, float, dict[str, dict]]:
    """Run the permutation null test in a single pass.

    Computes both the global p-value and per-galaxy null statistics.
    ``log_g_bar`` is shuffled within each galaxy to destroy the signal while
    preserving per-galaxy distributions.

    The p-value uses the corrected estimator (Phipson & Smyth 2010)::

        p = (1 + #{null RMSE ≤ observed RMSE}) / (N_perm + 1)

    Parameters
    ----------
    observed_rmse : float
        Mean per-galaxy OOS RMSE from the real (un-permuted) run.
    real_rmse_by_gal : dict
        Mapping galaxy → mean OOS RMSE from the real run.

    Returns
    -------
    perm_df : DataFrame with columns perm_id, mean_rmse.
    p_value : corrected global permutation p-value.
    gal_null_stats : dict mapping galaxy → dict with null RMSE stats and
        per-galaxy p-value (raw).
    """
    rng = np.random.default_rng(seed + 1)  # different seed from CV split
    groups = df[col_galaxy].values
    x = df[col_x].values.astype(float)
    y = df[col_y].values.astype(float)

    # Fix fold splits for all permutations (deterministic)
    cv_rng = np.random.default_rng(seed)
    splits = _group_kfold_splits(groups, n_splits, cv_rng)

    null_global_rmses: list[float] = []
    # Per-galaxy null RMSE accumulator: gal → list[float]
    gal_null: dict[str, list[float]] = {}

    for _ in range(n_perm):
        x_perm = _permute_within_groups(x, groups, rng)
        perm_gal_rmses: list[float] = []
        for train_idx, test_idx in splits:
            a, b = _ols_fit(x_perm[train_idx], y[train_idx])
            y_pred_test = _ols_predict(x_perm[test_idx], a, b)
            residuals = y[test_idx] - y_pred_test
            for gal in np.unique(groups[test_idx]):
                gal_mask = groups[test_idx] == gal
                rmse = float(np.sqrt(np.mean(residuals[gal_mask] ** 2)))
                perm_gal_rmses.append(rmse)
                gal_null.setdefault(gal, []).append(rmse)
        null_global_rmses.append(
            float(np.mean(perm_gal_rmses)) if perm_gal_rmses else float("nan")
        )

    null_arr = np.array(null_global_rmses)
    # Corrected p-value (Phipson & Smyth 2010): avoids p = 0
    p_value = float((np.sum(null_arr <= observed_rmse) + 1) / (n_perm + 1))

    perm_df = pd.DataFrame(
        {"perm_id": np.arange(n_perm), "mean_rmse": null_global_rmses}
    )

    # Build per-galaxy null stats and raw p-values
    gal_null_stats: dict[str, dict] = {}
    for gal, null_list in gal_null.items():
        null_arr_gal = np.array(null_list)
        real = real_rmse_by_gal.get(gal, float("nan"))
        p_gal = float(
            (np.sum(null_arr_gal <= real) + 1) / (len(null_arr_gal) + 1)
        )
        gal_null_stats[gal] = {
            "null_rmse_median": float(np.median(null_arr_gal)),
            "null_rmse_mean": float(np.mean(null_arr_gal)),
            "null_rmse_std": float(np.std(null_arr_gal)),
            "improved": (not np.isnan(real)) and (real < float(np.median(null_arr_gal))),
            "p_value_galaxy": p_gal,
        }

    return perm_df, p_value, gal_null_stats


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
    per_galaxy_folds, per_point = _run_cv(
        df, col_galaxy, col_x, col_y, n_splits, cv_rng
    )

    # Observed mean OOS RMSE (mean across all per-galaxy fold records)
    observed_rmse = float(per_galaxy_folds["rmse"].mean())

    # Per-galaxy mean OOS RMSE (averaged across folds)
    real_rmse_by_gal: dict[str, float] = (
        per_galaxy_folds.groupby(col_galaxy)["rmse"].mean().to_dict()
    )

    # ------------------------------------------------------------------
    # Permutation analysis (single pass: global + per-galaxy)
    # ------------------------------------------------------------------
    perm_df, p_value, gal_null_stats = _run_permutation_analysis(
        df, col_galaxy, col_x, col_y,
        n_splits, n_perm, seed, observed_rmse, real_rmse_by_gal,
    )

    null_mean_rmse = float(perm_df["mean_rmse"].mean())
    null_std_rmse = float(perm_df["mean_rmse"].std())

    # ------------------------------------------------------------------
    # Fraction of galaxies improved over null
    # ------------------------------------------------------------------
    improved_count = sum(
        1 for stats in gal_null_stats.values() if stats["improved"]
    )
    total_count = len(gal_null_stats)
    frac_improved = float(improved_count / total_count) if total_count > 0 else float("nan")

    # ------------------------------------------------------------------
    # Build aggregated per-galaxy CSV (one row per galaxy)
    # ------------------------------------------------------------------
    agg = (
        per_galaxy_folds.groupby(col_galaxy)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            n_folds=("fold", "count"),
            n_points=("n_points", "sum"),
        )
        .reset_index()
    )
    # Attach null stats and per-galaxy p-values
    for col_name, key in [
        ("null_rmse_median", "null_rmse_median"),
        ("null_rmse_mean", "null_rmse_mean"),
        ("null_rmse_std", "null_rmse_std"),
        ("improved", "improved"),
        ("p_value_galaxy", "p_value_galaxy"),
    ]:
        agg[col_name] = agg[col_galaxy].map(
            lambda g, k=key: gal_null_stats.get(g, {}).get(k, float("nan"))
        )
    # Benjamini-Hochberg correction on per-galaxy p-values
    agg["p_value_bh"] = _bh_correction(agg["p_value_galaxy"].values)

    # ------------------------------------------------------------------
    # Build folds detail for audit_summary.json
    # ------------------------------------------------------------------
    folds_detail: dict[str, dict] = {}
    for fold_idx, fold_df in per_galaxy_folds.groupby("fold"):
        folds_detail[f"fold_{fold_idx}"] = {
            "n_test_galaxies": int(len(fold_df)),
            "n_test_points": int(fold_df["n_points"].sum()),
            "mean_rmse": float(fold_df["rmse"].mean()),
            "slope": float(fold_df["slope"].iloc[0]),
            "intercept": float(fold_df["intercept"].iloc[0]),
        }

    # ------------------------------------------------------------------
    # Strict checks
    # ------------------------------------------------------------------
    p_pass = p_value <= 0.05
    frac_pass = (not np.isnan(frac_improved)) and (frac_improved >= 0.50)
    overall_pass = p_pass and frac_pass

    verdict = "PASS" if overall_pass else "FAIL"

    summary = {
        "input_csv": str(input_csv),
        "n_galaxies": n_galaxies,
        "n_points": n_points,
        "n_splits": n_splits,
        "n_perm": n_perm,
        "seed": seed,
        # Primary keys (referee-facing names)
        "rmse_real": observed_rmse,
        "rmse_perm_mean": null_mean_rmse,
        "rmse_perm_std": null_std_rmse,
        "p_value": p_value,
        "frac_galaxies_improved": frac_improved,
        # Legacy aliases (backward compatible)
        "observed_mean_oos_rmse": observed_rmse,
        "null_mean_rmse": null_mean_rmse,
        "p_value_permutation": p_value,
        # Per-fold detail
        "folds": folds_detail,
        # Strict checks
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

    agg.to_csv(out_dir / "groupkfold_per_galaxy.csv", index=False)
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
    print(f"  OOS RMSE (real): {summary['rmse_real']:.6f}")
    print(f"  OOS RMSE (null): {summary['rmse_perm_mean']:.6f}  ± {summary['rmse_perm_std']:.6f}")
    print(f"  p-value        : {summary['p_value']:.4f}")
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
