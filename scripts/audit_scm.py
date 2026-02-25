"""
scripts/audit_scm.py — Rigorous out-of-sample (OOS) audit for the Motor de Velos SCM.

Methodology
-----------
The unit of observation is a *galaxy*.  Three nested models are compared:

  BTFR      v_obs = β·logM + C
  no_hinge  v_obs = β·logM + C + a·log_gbar
  full_scm  v_obs = β·logM + C + a·log_gbar + b·max(0, logg0−log_gbar)^d
            physical constraint:  d ≥ 0

Evaluation protocol
~~~~~~~~~~~~~~~~~~~
* GroupKFold (default K=5) splits *galaxies* into non-overlapping train/test
  folds — no data-leakage across galaxies.
* All three models are **fit on TRAIN only** and evaluated on TEST.
* Per-galaxy RMSE is recorded for every test fold.

Permutation test (hard null)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The global ``log_gbar`` column is shuffled N_PERM times, completely breaking
the RAR structure.  The full SCM model is refit from scratch on the shuffled
data using the same GroupKFold splits.  The resulting distribution of mean
RMSE forms the structural null.

  p_empirical = #{perm_rmse ≤ real_rmse} / N_PERM
  (fraction of permutations that are at least as good as the real model)

Wilcoxon one-sided test
~~~~~~~~~~~~~~~~~~~~~~~~
After collecting per-galaxy signed deltas (rmse_full − rmse_btfr) across all
test folds, a one-sided Wilcoxon signed-rank test checks whether the full SCM
systematically improves on BTFR (H1: median Δ < 0).

Artefacts (6 standard outputs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  groupkfold_metrics.csv   fold-level RMSE for all three models
  gal_results.csv          per-galaxy RMSE + signed residuals
  permutation_runs.csv     perm_rmse for each permutation
  coeffs_by_fold.csv       fitted coefficients (β, C, a, b, d, logg0) by fold
  permutation_summary.json p-value + summary statistics
  audit_summary.txt        human-readable audit report

Usage
-----
    python scripts/audit_scm.py \\
        --csv results/audit/sparc_global.csv \\
        --out results/audit/oos_audit \\
        --n-splits 5 \\
        --n-perm 200

Or programmatically::

    from scripts.audit_scm import main
    main(["--csv", "...", "--out", "..."])
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SPLITS_DEFAULT = 5
N_PERM_DEFAULT = 200
LOGG0_INIT = -10.45   # initial guess for hinge threshold
REQUIRED_COLS = {"galaxy_id", "logM", "log_gbar", "v_obs"}
_SEP = "=" * 72


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _predict_btfr(logM: np.ndarray, params: np.ndarray) -> np.ndarray:
    """v_obs = β·logM + C"""
    beta, C = params
    return beta * logM + C


def _fit_btfr(logM: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS fit of BTFR model.  Returns [beta, C]."""
    A = np.column_stack([logM, np.ones(len(logM))])
    params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return params


def _predict_no_hinge(logM: np.ndarray, log_gbar: np.ndarray,
                      params: np.ndarray) -> np.ndarray:
    """v_obs = β·logM + C + a·log_gbar"""
    beta, C, a = params
    return beta * logM + C + a * log_gbar


def _fit_no_hinge(logM: np.ndarray, log_gbar: np.ndarray,
                  y: np.ndarray) -> np.ndarray:
    """OLS fit of no-hinge model.  Returns [beta, C, a]."""
    A = np.column_stack([logM, np.ones(len(logM)), log_gbar])
    params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return params


def _predict_full(logM: np.ndarray, log_gbar: np.ndarray,
                  params: np.ndarray) -> np.ndarray:
    """v_obs = β·logM + C + a·log_gbar + b·relu(logg0−log_gbar)^d
    where relu(x) = max(0, x) and d ≥ 0.
    """
    beta, C, a, b, d, logg0 = params
    d_safe = max(float(d), 0.0)
    hinge = np.maximum(0.0, logg0 - log_gbar) ** d_safe
    return beta * logM + C + a * log_gbar + b * hinge


def _fit_full(logM: np.ndarray, log_gbar: np.ndarray,
              y: np.ndarray) -> np.ndarray:
    """Bounded minimisation of MSE for the full SCM model.

    Physical constraint: d ≥ 0.
    Returns [beta, C, a, b, d, logg0].
    """
    def _mse(p: np.ndarray) -> float:
        pred = _predict_full(logM, log_gbar, p)
        return float(np.mean((y - pred) ** 2))

    # Warm start from no-hinge OLS
    p_nh = _fit_no_hinge(logM, log_gbar, y)
    p0 = np.array([p_nh[0], p_nh[1], p_nh[2], 0.0, 1.0, LOGG0_INIT])

    bounds = [
        (None, None),   # beta
        (None, None),   # C
        (None, None),   # a
        (None, None),   # b
        (0.0, None),    # d  ← physical constraint d ≥ 0
        (None, None),   # logg0
    ]
    result = minimize(_mse, p0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 1000, "ftol": 1e-12})
    return result.x


# ---------------------------------------------------------------------------
# Per-galaxy RMSE helpers
# ---------------------------------------------------------------------------

def _rmse_per_galaxy(
    gal_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.Series:
    """Return Series galaxy_id → RMSE over the test rows."""
    df = pd.DataFrame({"galaxy_id": gal_ids, "y_true": y_true, "y_pred": y_pred})
    return df.groupby("galaxy_id").apply(
        lambda g: float(np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2)))
    )


def _resid_per_galaxy(
    gal_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.Series:
    """Return Series galaxy_id → mean signed residual (y_true − y_pred)."""
    df = pd.DataFrame({"galaxy_id": gal_ids, "resid": y_true - y_pred})
    return df.groupby("galaxy_id")["resid"].mean()


# ---------------------------------------------------------------------------
# Main audit routine
# ---------------------------------------------------------------------------

def run_audit(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS_DEFAULT,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = 0,
) -> dict:
    """Execute the full OOS audit.

    Parameters
    ----------
    df : pd.DataFrame
        Per-galaxy data with columns galaxy_id, logM, log_gbar, v_obs.
    n_splits : int
        Number of GroupKFold splits.
    n_perm : int
        Number of permutations for the hard permutation test.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys:
        fold_records    list of fold-level dicts
        gal_records     list of per-galaxy result dicts
        perm_rmses      list of mean RMSE values under permutation
        coeff_records   list of coefficient dicts per fold
        wilcoxon_stat   Wilcoxon test statistic
        wilcoxon_p      one-sided p-value (H1: median Δ < 0)
        real_mean_rmse  mean test RMSE (full SCM) over all folds
        p_empirical     empirical p-value from permutation test
    """
    rng = np.random.default_rng(seed)

    logM = df["logM"].values
    log_gbar = df["log_gbar"].values
    y = df["v_obs"].values
    groups = df["galaxy_id"].values

    gkf = GroupKFold(n_splits=n_splits)

    fold_records: list[dict] = []
    gal_records_map: dict[str, dict] = {}
    coeff_records: list[dict] = []

    # -----------------------------------------------------------------------
    # GroupKFold evaluation
    # -----------------------------------------------------------------------
    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(logM, y, groups)
    ):
        logM_tr, logM_te = logM[train_idx], logM[test_idx]
        lgb_tr, lgb_te = log_gbar[train_idx], log_gbar[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        gal_te = groups[test_idx]

        # Fit on TRAIN only
        p_btfr = _fit_btfr(logM_tr, y_tr)
        p_nh = _fit_no_hinge(logM_tr, lgb_tr, y_tr)
        p_full = _fit_full(logM_tr, lgb_tr, y_tr)

        # Predict on TEST
        pred_btfr = _predict_btfr(logM_te, p_btfr)
        pred_nh = _predict_no_hinge(logM_te, lgb_te, p_nh)
        pred_full = _predict_full(logM_te, lgb_te, p_full)

        # Fold-level RMSE
        rmse_btfr = float(np.sqrt(np.mean((y_te - pred_btfr) ** 2)))
        rmse_nh = float(np.sqrt(np.mean((y_te - pred_nh) ** 2)))
        rmse_full = float(np.sqrt(np.mean((y_te - pred_full) ** 2)))

        fold_records.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "rmse_btfr": rmse_btfr,
            "rmse_no_hinge": rmse_nh,
            "rmse_full": rmse_full,
        })

        # Per-galaxy RMSE and residuals
        rmse_btfr_gal = _rmse_per_galaxy(gal_te, y_te, pred_btfr)
        rmse_nh_gal = _rmse_per_galaxy(gal_te, y_te, pred_nh)
        rmse_full_gal = _rmse_per_galaxy(gal_te, y_te, pred_full)
        resid_btfr_gal = _resid_per_galaxy(gal_te, y_te, pred_btfr)

        for gal_id in rmse_full_gal.index:
            gal_records_map[gal_id] = {
                "galaxy_id": gal_id,
                "fold": fold_idx,
                "rmse_btfr": float(rmse_btfr_gal[gal_id]),
                "rmse_no_hinge": float(rmse_nh_gal[gal_id]),
                "rmse_full": float(rmse_full_gal[gal_id]),
                "resid_btfr": float(resid_btfr_gal[gal_id]),
            }

        # Coefficients: enforce d >= 0 in output
        beta, C, a, b, d, logg0 = p_full
        coeff_records.append({
            "fold": fold_idx,
            "beta": float(beta),
            "C": float(C),
            "a": float(a),
            "b": float(b),
            "d": float(max(d, 0.0)),
            "logg0": float(logg0),
        })

    gal_records = list(gal_records_map.values())
    real_mean_rmse = float(np.mean([r["rmse_full"] for r in fold_records]))

    # -----------------------------------------------------------------------
    # Wilcoxon one-sided test (H1: median(rmse_full − rmse_btfr) < 0)
    # -----------------------------------------------------------------------
    deltas = np.array([r["rmse_full"] - r["rmse_btfr"] for r in gal_records])
    if len(deltas) >= 10:
        wstat, wpval = wilcoxon(deltas, alternative="less")
    else:
        wstat, wpval = float("nan"), float("nan")

    # -----------------------------------------------------------------------
    # Hard permutation test (global shuffle of log_gbar)
    # -----------------------------------------------------------------------
    perm_rmses: list[float] = []
    for _ in range(n_perm):
        log_gbar_perm = rng.permutation(log_gbar)  # global shuffle
        perm_fold_rmses: list[float] = []
        for train_idx, test_idx in gkf.split(logM, y, groups):
            logM_tr = logM[train_idx]
            lgb_tr_p = log_gbar_perm[train_idx]
            y_tr = y[train_idx]
            logM_te = logM[test_idx]
            lgb_te_p = log_gbar_perm[test_idx]
            y_te = y[test_idx]

            p_perm = _fit_full(logM_tr, lgb_tr_p, y_tr)
            pred_p = _predict_full(logM_te, lgb_te_p, p_perm)
            perm_fold_rmses.append(
                float(np.sqrt(np.mean((y_te - pred_p) ** 2)))
            )
        perm_rmses.append(float(np.mean(perm_fold_rmses)))

    perm_arr = np.array(perm_rmses)
    p_empirical = float(np.mean(perm_arr <= real_mean_rmse))

    return {
        "fold_records": fold_records,
        "gal_records": gal_records,
        "perm_rmses": perm_rmses,
        "coeff_records": coeff_records,
        "wilcoxon_stat": float(wstat) if not np.isnan(wstat) else None,
        "wilcoxon_p": float(wpval) if not np.isnan(wpval) else None,
        "real_mean_rmse": real_mean_rmse,
        "p_empirical": p_empirical,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_artefacts(results: dict, out_dir: Path) -> None:
    """Write the 6 standard audit artefacts to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(results["fold_records"])
    fold_df.to_csv(out_dir / "groupkfold_metrics.csv", index=False)

    gal_df = pd.DataFrame(results["gal_records"])
    gal_df.to_csv(out_dir / "gal_results.csv", index=False)

    perm_df = pd.DataFrame({"perm_rmse": results["perm_rmses"],
                             "run": range(len(results["perm_rmses"]))})
    perm_df.to_csv(out_dir / "permutation_runs.csv", index=False)

    coeff_df = pd.DataFrame(results["coeff_records"])
    coeff_df.to_csv(out_dir / "coeffs_by_fold.csv", index=False)

    perm_arr = np.array(results["perm_rmses"])
    summary = {
        "real_mean_rmse": results["real_mean_rmse"],
        "perm_mean_rmse": float(perm_arr.mean()) if len(perm_arr) else None,
        "perm_std_rmse": float(perm_arr.std()) if len(perm_arr) else None,
        "p_empirical": results["p_empirical"],
        "wilcoxon_stat": results["wilcoxon_stat"],
        "wilcoxon_p": results["wilcoxon_p"],
        "n_perm": len(results["perm_rmses"]),
        "n_galaxies": len(results["gal_records"]),
    }
    (out_dir / "permutation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _write_audit_summary(results, fold_df, gal_df, out_dir / "audit_summary.txt")


def _write_audit_summary(results: dict, fold_df: pd.DataFrame,
                         gal_df: pd.DataFrame, path: Path) -> None:
    """Write human-readable audit report."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — OOS Audit Report",
        "  GroupKFold + hard permutation + physical hinge (d ≥ 0)",
        _SEP,
        f"  N_galaxies       : {len(gal_df)}",
        f"  N_splits         : {len(fold_df)}",
        f"  N_perm           : {len(results['perm_rmses'])}",
        "",
        "  Mean RMSE by model (test folds):",
        f"    BTFR             : {fold_df['rmse_btfr'].mean():.5f}",
        f"    No-hinge         : {fold_df['rmse_no_hinge'].mean():.5f}",
        f"    Full SCM         : {fold_df['rmse_full'].mean():.5f}",
        "",
        "  Permutation test:",
        f"    Real RMSE (full) : {results['real_mean_rmse']:.5f}",
    ]
    perm_arr = np.array(results["perm_rmses"])
    if len(perm_arr):
        lines.append(f"    Perm mean RMSE   : {perm_arr.mean():.5f}")
        lines.append(f"    Perm std RMSE    : {perm_arr.std():.5f}")
    lines += [
        f"    p_empirical      : {results['p_empirical']:.4f}",
        "",
        "  Wilcoxon one-sided test (H1: median Δ < 0):",
    ]
    if results["wilcoxon_p"] is not None:
        lines.append(f"    stat             : {results['wilcoxon_stat']:.2f}")
        lines.append(f"    p-value          : {results['wilcoxon_p']:.4f}")
    else:
        lines.append("    (insufficient galaxies for Wilcoxon test)")
    deltas = gal_df["rmse_full"] - gal_df["rmse_btfr"]
    frac_improved = float((deltas < 0).mean())
    lines += [
        "",
        f"  Galaxies improved (Δ < 0) : {frac_improved:.1%}",
        _SEP,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rigorous OOS audit: GroupKFold + hard permutation + hinge SCM."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Per-galaxy audit CSV (galaxy_id, logM, log_gbar, v_obs). "
             "If omitted, defaults to results/audit/sparc_global.csv",
    )
    parser.add_argument(
        "--out",
        default="results/audit/oos_audit",
        help="Output directory for audit artefacts. "
             "Default: results/audit/oos_audit",
    )
    parser.add_argument(
        "--n-splits", type=int, default=N_SPLITS_DEFAULT, dest="n_splits",
        help=f"Number of GroupKFold splits (default: {N_SPLITS_DEFAULT}).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=N_PERM_DEFAULT, dest="n_perm",
        help=f"Number of permutations (default: {N_PERM_DEFAULT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the OOS audit pipeline.

    Returns the results dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)

    # Auto-detect input if --csv was not provided
    if args.csv is None:
        default_path = Path("results/audit/sparc_global.csv")
        if default_path.exists():
            args.csv = str(default_path)
            print(f"[audit_scm] Using default input: {default_path}")
        else:
            raise ValueError(
                "No --csv provided and default results/audit/sparc_global.csv not found. "
                "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
            )

    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        print(
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' "
            "to generate per-galaxy audit data.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"ERROR: CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=list(REQUIRED_COLS)).reset_index(drop=True)
    if len(df) < args.n_splits:
        print(
            f"ERROR: need at least n_splits={args.n_splits} rows, "
            f"got {len(df)}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = run_audit(df, n_splits=args.n_splits, n_perm=args.n_perm,
                        seed=args.seed)
    out_dir = Path(args.out)
    _write_artefacts(results, out_dir)

    # Print audit summary to stdout
    (out_dir / "audit_summary.txt").read_text(encoding="utf-8").splitlines()
    for line in (out_dir / "audit_summary.txt").read_text(encoding="utf-8").splitlines():
        print(line)

    print(f"\n  Artefacts written to {out_dir}")
    return results


if __name__ == "__main__":
    main()
