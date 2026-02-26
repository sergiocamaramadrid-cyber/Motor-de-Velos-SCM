"""
scripts/audit_scm.py — Statistical audit of the Motor de Velos SCM model.

Performs three independent checks to validate that the RAR/SCM model is not
a tautology and generalises across galaxies:

  1. GroupKFold cross-validation (split by galaxy to prevent hierarchy leakage).
     Outputs per-galaxy RMSE (aggregated from radial points within each galaxy)
     and per-point residuals.

  2. Permutation test of log_g_bar within each galaxy (within-galaxy shuffle
     destroys the baryonic signal while preserving galaxy-level structure).
     Reports empirical p-value using the two-sided formula:
       p = (1 + Σ[perm_rmse ≤ real_rmse]) / (N_perm + 1)

  3. Coefficient stability: for each GroupKFold fold the model is refitted
     on training galaxies only, and the fitted slope and intercept are recorded.
     Reports mean ± std across folds.

AICc is computed using the MLE estimator σ² = RSS/n, which makes it comparable
across models without requiring external error estimates:
  AICc = n·ln(RSS/n) + 2k + 2k(k+1)/(n−k−1) + n·ln(2π) + n

Model: linear regression  log_g_obs = slope·log_g_bar + intercept
  (fitted per fold / globally via ordinary least squares)

Outputs written to --out:
  groupkfold_per_galaxy.csv  — per-galaxy RMSE from GroupKFold
  groupkfold_per_point.csv   — per-point predictions and residuals
  permutation_test.csv       — RMSE distribution under null (N_perm rows)
  audit_summary.json         — summary with p-value, stability, AICc, verdict

Usage
-----
    python scripts/audit_scm.py \\
        --input results/universal_term_comparison_full.csv \\
        --out   results/diagnostics/audit_scm

With strict mode (exits with code 1 if checks fail)::

    python scripts/audit_scm.py \\
        --input results/universal_term_comparison_full.csv \\
        --out   results/diagnostics/audit_scm \\
        --strict

Optional flags::

    --n-splits   N   Number of GroupKFold splits (default: 5)
    --n-perm     N   Number of permutations in the permutation test (default: 200)
    --seed       N   Random seed for reproducibility (default: 42)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Required columns in the input CSV
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"galaxy", "log_g_bar", "log_g_obs"}

# ---------------------------------------------------------------------------
# Linear model helpers
# ---------------------------------------------------------------------------


def _fit_linear(log_g_bar: np.ndarray,
                log_g_obs: np.ndarray) -> tuple[float, float]:
    """Fit log_g_obs = slope·log_g_bar + intercept via OLS.

    Parameters
    ----------
    log_g_bar, log_g_obs : 1-D arrays
        Input data (must have length ≥ 2).

    Returns
    -------
    slope, intercept : float
    """
    x = log_g_bar - log_g_bar.mean()
    slope = float(np.dot(x, log_g_obs - log_g_obs.mean()) / np.dot(x, x))
    intercept = float(log_g_obs.mean() - slope * log_g_bar.mean())
    return slope, intercept


def _predict(log_g_bar: np.ndarray, slope: float,
             intercept: float) -> np.ndarray:
    """Apply the linear model."""
    return slope * log_g_bar + intercept


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Fix 2 — Corrected AICc using MLE estimator (σ² = RSS/n)
# ---------------------------------------------------------------------------


def safe_aicc(rss: float, n: int, k: int) -> float:
    """AICc with MLE noise estimate σ² = RSS/n.

    AICc = n·ln(RSS/n) + 2k + 2k(k+1)/(n−k−1) + n·(1 + ln(2π))

    The constant n·(1 + ln(2π)) is included so that ΔAICc values are
    interpretable across models with the same n.

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of data points.
    k : int
        Number of free parameters (e.g., 2 for slope + intercept).

    Returns
    -------
    float
        AICc value.  Returns +inf if rss ≤ 0 or n ≤ k + 1.
    """
    if rss <= 0 or n <= k + 1:
        return float("inf")
    aic = n * np.log(rss / n) + 2.0 * k + n * (1.0 + np.log(2.0 * np.pi))
    correction = 2.0 * k * (k + 1) / max(n - k - 1, 1)
    return float(aic + correction)


# ---------------------------------------------------------------------------
# Fix 1 & 4 — GroupKFold audit with per-galaxy RMSE and coefficient stability
# ---------------------------------------------------------------------------


def groupkfold_audit(df: pd.DataFrame, n_splits: int = 5,
                     rng: np.random.Generator | None = None
                     ) -> dict:
    """GroupKFold cross-validation split by galaxy.

    Each fold trains a linear model on all radial points from training galaxies
    and evaluates it on radial points from held-out (test) galaxies.
    Per-galaxy RMSE is computed by aggregating the residuals of each galaxy's
    radial points within the fold — so ``groupkfold_per_galaxy.csv`` really is
    one row per galaxy, not one row per radial point.

    Coefficient stability is measured by saving the fitted (slope, intercept)
    for every fold and reporting mean ± std.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``galaxy``, ``log_g_bar``, ``log_g_obs``.
    n_splits : int
        Number of GroupKFold splits (must be ≤ number of galaxies).
    rng : np.random.Generator or None
        Random generator for fold assignment shuffling.

    Returns
    -------
    dict with keys:
        per_galaxy_df   — DataFrame: one row per (galaxy, fold) with
                          galaxy_id, fold, n_points, rmse_galaxy
        per_point_df    — DataFrame: one row per radial point with
                          galaxy, fold, log_g_bar, log_g_obs, log_g_pred, residual
        coeff_by_fold   — list of dicts {fold, slope, intercept} per fold
        coeff_mean      — dict {slope_mean, intercept_mean}
        coeff_std       — dict {slope_std, intercept_std}
        aicc            — AICc of the global (all-data) fit
        k               — number of free parameters used (2: slope + intercept)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    galaxies = df["galaxy"].unique()
    n_gal = len(galaxies)
    n_splits_actual = min(n_splits, n_gal)

    # Assign each galaxy to a fold
    perm = rng.permutation(n_gal)
    fold_ids = np.array_split(perm, n_splits_actual)
    galaxy_to_fold = {}
    for fold_idx, indices in enumerate(fold_ids):
        for i in indices:
            galaxy_to_fold[galaxies[i]] = fold_idx

    per_galaxy_rows: list[dict] = []
    per_point_rows: list[dict] = []
    coeff_by_fold: list[dict] = []

    for fold_idx in range(n_splits_actual):
        test_galaxies = {g for g, f in galaxy_to_fold.items() if f == fold_idx}
        train_mask = ~df["galaxy"].isin(test_galaxies)
        test_mask = df["galaxy"].isin(test_galaxies)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 2:
            continue  # not enough training data for this fold

        # Fix 4: fit coefficients ONLY on training galaxies for this fold
        slope, intercept = _fit_linear(
            train_df["log_g_bar"].to_numpy(),
            train_df["log_g_obs"].to_numpy(),
        )
        coeff_by_fold.append({"fold": fold_idx, "slope": slope,
                               "intercept": intercept})

        # Evaluate on test galaxies — aggregate per galaxy (Fix 1)
        for gal_id, gal_df in test_df.groupby("galaxy"):
            lg_bar = gal_df["log_g_bar"].to_numpy()
            lg_obs = gal_df["log_g_obs"].to_numpy()
            lg_pred = _predict(lg_bar, slope, intercept)
            rmse_gal = _rmse(lg_obs, lg_pred)

            per_galaxy_rows.append({
                "galaxy": gal_id,
                "fold": fold_idx,
                "n_points": len(gal_df),
                "rmse_galaxy": rmse_gal,
            })

            for j in range(len(lg_bar)):
                per_point_rows.append({
                    "galaxy": gal_id,
                    "fold": fold_idx,
                    "log_g_bar": float(lg_bar[j]),
                    "log_g_obs": float(lg_obs[j]),
                    "log_g_pred": float(lg_pred[j]),
                    "residual": float(lg_obs[j] - lg_pred[j]),
                })

    per_galaxy_df = pd.DataFrame(per_galaxy_rows)
    per_point_df = pd.DataFrame(per_point_rows)

    # Coefficient stability summary
    if coeff_by_fold:
        slopes = np.array([c["slope"] for c in coeff_by_fold])
        intercepts = np.array([c["intercept"] for c in coeff_by_fold])
        coeff_mean = {"slope_mean": float(slopes.mean()),
                      "intercept_mean": float(intercepts.mean())}
        coeff_std = {"slope_std": float(slopes.std(ddof=1)) if len(slopes) > 1 else 0.0,
                     "intercept_std": (float(intercepts.std(ddof=1))
                                       if len(intercepts) > 1 else 0.0)}
    else:
        coeff_mean = {"slope_mean": float("nan"), "intercept_mean": float("nan")}
        coeff_std = {"slope_std": float("nan"), "intercept_std": float("nan")}

    # Global AICc (full dataset fit) — Fix 2
    k = 2  # slope + intercept
    lg_bar_all = df["log_g_bar"].to_numpy()
    lg_obs_all = df["log_g_obs"].to_numpy()
    slope_all, intercept_all = _fit_linear(lg_bar_all, lg_obs_all)
    lg_pred_all = _predict(lg_bar_all, slope_all, intercept_all)
    rss_all = float(np.sum((lg_obs_all - lg_pred_all) ** 2))
    aic_c = safe_aicc(rss_all, len(lg_bar_all), k)

    return {
        "per_galaxy_df": per_galaxy_df,
        "per_point_df": per_point_df,
        "coeff_by_fold": coeff_by_fold,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "aicc": aic_c,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Fix 3 + 6 — Permutation test with correct empirical p-value
# ---------------------------------------------------------------------------


def permutation_test(df: pd.DataFrame, n_perm: int = 200,
                     rng: np.random.Generator | None = None) -> dict:
    """Within-galaxy permutation test of the log_g_bar signal.

    For each permutation, log_g_bar values are shuffled *within* each galaxy
    (preserving the galaxy-level distribution) and the model RMSE is
    recomputed.  This tests whether the baryonic signal is informative beyond
    galaxy-level scatter.

    The empirical two-sided p-value uses the formula recommended for
    permutation tests with a finite number of permutations:
      p = (1 + Σ[perm_rmse ≤ real_rmse]) / (N_perm + 1)

    Note: "lower is better" for RMSE, so a small p-value means the real data
    fit better than most permuted datasets — consistent with the baryonic
    signal being real.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``galaxy``, ``log_g_bar``, ``log_g_obs``.
    n_perm : int
        Number of permutations.
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    dict with keys:
        real_rmse    — RMSE of the model fit on the original data
        perm_rmse    — array of RMSE values under the null
        p_value      — empirical p-value (lower is better convention)
        n_perm       — number of permutations actually performed
    """
    if rng is None:
        rng = np.random.default_rng(42)

    lg_bar = df["log_g_bar"].to_numpy()
    lg_obs = df["log_g_obs"].to_numpy()
    galaxy_ids = df["galaxy"].to_numpy()

    # Real RMSE
    slope_r, intercept_r = _fit_linear(lg_bar, lg_obs)
    real_rmse = _rmse(lg_obs, _predict(lg_bar, slope_r, intercept_r))

    # Permutation RMSE distribution
    perm_rmse_list: list[float] = []
    unique_galaxies = np.unique(galaxy_ids)

    for _ in range(n_perm):
        # Shuffle log_g_bar within each galaxy independently
        perm_lg_bar = lg_bar.copy()
        for gal in unique_galaxies:
            mask = galaxy_ids == gal
            perm_lg_bar[mask] = rng.permutation(lg_bar[mask])

        slope_p, intercept_p = _fit_linear(perm_lg_bar, lg_obs)
        perm_rmse_list.append(_rmse(lg_obs, _predict(perm_lg_bar, slope_p,
                                                      intercept_p)))

    perm_rmse = np.array(perm_rmse_list)

    # Fix 6: correct empirical p-value (lower RMSE is better)
    p_value = float((1 + np.sum(perm_rmse <= real_rmse)) / (n_perm + 1))

    return {
        "real_rmse": float(real_rmse),
        "perm_rmse": perm_rmse,
        "p_value": p_value,
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Wilcoxon test on per-galaxy ΔRMSE (paired comparison)
# ---------------------------------------------------------------------------


def galaxy_delta_rmse_test(per_galaxy_df: pd.DataFrame,
                            null_rmse: float) -> dict:
    """Wilcoxon signed-rank test on per-galaxy RMSE vs a null RMSE.

    The null RMSE is typically the mean permutation RMSE from
    :func:`permutation_test`.

    Parameters
    ----------
    per_galaxy_df : pd.DataFrame
        Output of :func:`groupkfold_audit` with column ``rmse_galaxy``.
    null_rmse : float
        Reference RMSE to compare against (e.g., mean perm RMSE).

    Returns
    -------
    dict with keys:
        frac_improved     — fraction of galaxies with rmse_galaxy < null_rmse
        wilcoxon_stat     — Wilcoxon test statistic (nan if < 10 galaxies)
        wilcoxon_p        — Wilcoxon p-value (nan if < 10 galaxies)
        n_galaxies        — number of galaxies tested
    """
    rmse_vals = per_galaxy_df["rmse_galaxy"].to_numpy()
    delta = rmse_vals - null_rmse
    frac_improved = float(np.mean(delta < 0))
    n = len(rmse_vals)

    if n < 10:
        return {
            "frac_improved": frac_improved,
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
            "n_galaxies": n,
        }

    try:
        stat, p = wilcoxon(delta, alternative="two-sided")
        return {
            "frac_improved": frac_improved,
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(p),
            "n_galaxies": n,
        }
    except ValueError:
        return {
            "frac_improved": frac_improved,
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
            "n_galaxies": n,
        }


# ---------------------------------------------------------------------------
# Top-level audit runner
# ---------------------------------------------------------------------------


def run_audit(df: pd.DataFrame, out_dir: Path, n_splits: int = 5,
              n_perm: int = 200, seed: int = 42,
              strict: bool = False) -> dict:
    """Run all three audit checks and write outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns ``galaxy``, ``log_g_bar``, ``log_g_obs``.
    out_dir : Path
        Output directory (created if absent).
    n_splits : int
        GroupKFold splits.
    n_perm : int
        Permutation test repetitions.
    seed : int
        Random seed.
    strict : bool
        If True, raise SystemExit(1) when checks fail.

    Returns
    -------
    dict
        Summary of all audit results (also written as audit_summary.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- 1. GroupKFold cross-validation ----
    kfold_result = groupkfold_audit(df, n_splits=n_splits,
                                    rng=np.random.default_rng(seed))
    per_gal_df = kfold_result["per_galaxy_df"]
    per_pt_df = kfold_result["per_point_df"]

    per_gal_df.to_csv(out_dir / "groupkfold_per_galaxy.csv", index=False)
    per_pt_df.to_csv(out_dir / "groupkfold_per_point.csv", index=False)

    # ---- 2. Permutation test ----
    perm_result = permutation_test(df, n_perm=n_perm,
                                   rng=np.random.default_rng(seed + 1))
    perm_df = pd.DataFrame({"perm_rmse": perm_result["perm_rmse"]})
    perm_df.to_csv(out_dir / "permutation_test.csv", index=False)

    # ---- 3. Wilcoxon on per-galaxy ΔRMSE ----
    null_rmse = float(perm_result["perm_rmse"].mean())
    wilcox_result = galaxy_delta_rmse_test(per_gal_df, null_rmse)

    # ---- Build summary ----
    summary: dict = {
        "n_galaxies": int(df["galaxy"].nunique()),
        "n_points": int(len(df)),
        "n_splits": int(n_splits),
        "n_perm": int(n_perm),
        "seed": int(seed),
        # AICc
        "aicc": float(kfold_result["aicc"]),
        "k_params": int(kfold_result["k"]),
        # GroupKFold
        "cv_mean_rmse_per_galaxy": (
            float(per_gal_df["rmse_galaxy"].mean())
            if len(per_gal_df) else float("nan")
        ),
        "cv_std_rmse_per_galaxy": (
            float(per_gal_df["rmse_galaxy"].std(ddof=1))
            if len(per_gal_df) > 1 else float("nan")
        ),
        # Coefficient stability (Fix 4)
        "coeff_slope_mean": kfold_result["coeff_mean"]["slope_mean"],
        "coeff_slope_std": kfold_result["coeff_std"]["slope_std"],
        "coeff_intercept_mean": kfold_result["coeff_mean"]["intercept_mean"],
        "coeff_intercept_std": kfold_result["coeff_std"]["intercept_std"],
        "coeff_by_fold": kfold_result["coeff_by_fold"],
        # Permutation test
        "real_rmse": float(perm_result["real_rmse"]),
        "perm_rmse_mean": float(null_rmse),
        "p_value_permutation": float(perm_result["p_value"]),
        # Wilcoxon per-galaxy
        "frac_galaxies_improved": float(wilcox_result["frac_improved"]),
        "wilcoxon_stat": wilcox_result["wilcoxon_stat"],
        "wilcoxon_p": wilcox_result["wilcoxon_p"],
        "n_galaxies_tested": int(wilcox_result["n_galaxies"]),
    }

    # Strict mode checks (Fix 5)
    checks_passed = True
    strict_messages: list[str] = []

    p_val = summary["p_value_permutation"]
    frac_imp = summary["frac_galaxies_improved"]

    if p_val > 0.05:
        checks_passed = False
        strict_messages.append(
            f"FAIL: permutation p-value={p_val:.4f} > 0.05 "
            "(baryonic signal not significant)"
        )

    if frac_imp < 0.50:
        checks_passed = False
        strict_messages.append(
            f"FAIL: only {frac_imp*100:.1f}% of galaxies improved "
            "(need ≥ 50%)"
        )

    summary["strict_checks_passed"] = bool(checks_passed)
    summary["strict_messages"] = strict_messages

    # Write JSON summary
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Print report
    _print_report(summary)

    if strict and not checks_passed:
        for msg in strict_messages:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)

    return summary


def _print_report(summary: dict) -> None:
    sep = "=" * 72
    print(sep)
    print("  Motor de Velos SCM — Statistical Audit Report")
    print(sep)
    print(f"  Galaxies          : {summary['n_galaxies']}")
    print(f"  Radial points     : {summary['n_points']}")
    print(f"  GroupKFold splits : {summary['n_splits']}")
    print(f"  N permutations    : {summary['n_perm']}")
    print()
    print(f"  [AICc]")
    print(f"    AICc (global)   : {summary['aicc']:.4f}")
    print(f"    k parameters    : {summary['k_params']}")
    print()
    print(f"  [GroupKFold — per-galaxy RMSE]")
    print(f"    Mean RMSE       : {summary['cv_mean_rmse_per_galaxy']:.4f}")
    print(f"    Std  RMSE       : {summary['cv_std_rmse_per_galaxy']:.4f}")
    print()
    print(f"  [Coefficient stability across folds]")
    print(f"    slope  mean±std : "
          f"{summary['coeff_slope_mean']:.4f} ± {summary['coeff_slope_std']:.4f}")
    print(f"    intercept mean±std: "
          f"{summary['coeff_intercept_mean']:.4f} ± "
          f"{summary['coeff_intercept_std']:.4f}")
    print()
    print(f"  [Permutation test]")
    print(f"    Real RMSE       : {summary['real_rmse']:.4f}")
    print(f"    Perm RMSE mean  : {summary['perm_rmse_mean']:.4f}")
    print(f"    p-value         : {summary['p_value_permutation']:.4f}")
    print()
    print(f"  [Per-galaxy Δ RMSE (vs. null)]")
    print(f"    Frac improved   : {summary['frac_galaxies_improved']*100:.1f}%")
    wp = summary.get("wilcoxon_p")
    if wp is not None and not (isinstance(wp, float) and np.isnan(wp)):
        print(f"    Wilcoxon p      : {wp:.4f}")
    else:
        print(f"    Wilcoxon p      : N/A (< 10 galaxies)")
    print()
    verdict = "✅ PASSED" if summary.get("strict_checks_passed", True) else "⚠️  FAILED"
    print(f"  Verdict: {verdict}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical audit of the Motor de Velos SCM model."
    )
    parser.add_argument(
        "--input", required=True, metavar="CSV",
        help=(
            "Per-radial-point CSV with columns: galaxy, log_g_bar, log_g_obs. "
            "Typically results/universal_term_comparison_full.csv."
        ),
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output directory for audit results.",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, dest="n_splits",
        help="Number of GroupKFold splits (default: 5).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=200, dest="n_perm",
        help="Number of permutations in the permutation test (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Exit with code 1 if checks fail: "
            "permutation p > 0.05 or < 50%% galaxies improved."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path = Path(args.input)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}.\n"
            f"Required: {sorted(REQUIRED_COLS)}"
        )

    run_audit(
        df,
        out_dir=Path(args.out),
        n_splits=args.n_splits,
        n_perm=args.n_perm,
        seed=args.seed,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
