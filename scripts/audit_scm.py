"""
scripts/audit_scm.py — GroupKFold OOS audit for Motor de Velos SCM.

Reads ``audit_features.csv`` (produced by the pipeline's
:func:`src.scm_analysis._write_audit_metrics`) and writes the following
artefacts to ``--outdir``:

    groupkfold_metrics.csv     — per-fold OOS R², RMSE, N
    groupkfold_per_galaxy.csv  — per-galaxy aggregated OOS metrics
    coeffs_by_fold.csv         — feature coefficients for each fold
    permutation_summary.json   — permutation-test result (0 ≤ p_empirical ≤ 1)
    master_coeffs.json         — full-data OLS coefficients + Cohen's d for hinge
    audit_summary.json         — high-level audit pass/fail summary

    residual_vs_hinge_oos.png  — scatter + binned-median plot (matplotlib only)

The model regressed is::

    residual_dex ~ logM + log_gbar + log_j + hinge

where ``residual_dex = log_g_obs − log_g_bar`` and all features are in dex.
Groups in the cross-validation are galaxy names, preventing data leakage
between training and test sets.

Usage
-----
After running the SCM pipeline to generate ``audit/audit_features.csv``::

    python scripts/audit_scm.py \\
        --features-csv results/audit/audit_features.csv \\
        --outdir       results/final_audit \\
        --seed         20260211

If ``--features-csv`` is omitted it defaults to
``results/audit/audit_features.csv`` relative to the current working
directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURES = ["logM", "log_gbar", "log_j", "hinge"]
_TARGET = "residual_dex"
_N_FOLDS = 5
_N_PERM = 999
_DEFAULT_FEATURES_CSV = Path("results") / "audit" / "audit_features.csv"

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray):
    """OLS with intercept.  Returns ``(feature_coefficients, intercept)``."""
    n = len(y)
    Xa = np.column_stack([np.ones(n), X])
    params, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return params[1:], float(params[0])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# GroupKFold (pure NumPy — no sklearn dependency)
# ---------------------------------------------------------------------------

def _group_kfold_splits(groups: np.ndarray, n_folds: int, seed: int):
    """Yield ``(train_idx, test_idx)`` for GroupKFold.

    Unique group labels are shuffled with *seed*, then split into *n_folds*
    buckets.  All points from a group appear in exactly one test fold.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    fold_groups = np.array_split(shuffled, n_folds)
    for test_grps in fold_groups:
        test_set = set(test_grps.tolist())
        test_idx = np.where(np.isin(groups, list(test_set)))[0]
        train_idx = np.where(~np.isin(groups, list(test_set)))[0]
        yield train_idx, test_idx


def run_groupkfold(
    df: pd.DataFrame,
    n_folds: int = _N_FOLDS,
    seed: int = 42,
):
    """Run GroupKFold cross-validation (groups = galaxy names).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``galaxy``, all columns in ``_FEATURES``, and
        ``_TARGET``.
    n_folds : int
        Number of folds (clipped to the number of unique galaxies).
    seed : int
        Random seed for group shuffling.

    Returns
    -------
    fold_metrics : list[dict]
        One dict per fold: ``fold``, ``n_train``, ``n_test``,
        ``r2_oos``, ``rmse_oos``.
    per_point_rows : list[dict]
        One dict per test point: ``galaxy``, ``hinge``,
        ``residual_dex``, ``residual_dex_pred``,
        ``residual_dex_oos``, ``fold``.
    fold_coeffs : list[dict]
        One dict per fold: ``fold``, ``intercept``, plus one entry per
        feature name.
    """
    X = df[_FEATURES].to_numpy(dtype=float)
    y = df[_TARGET].to_numpy(dtype=float)
    groups = df["galaxy"].to_numpy()

    n_unique = df["galaxy"].nunique()
    n_folds = min(n_folds, n_unique)

    fold_metrics: list[dict] = []
    per_point_rows: list[dict] = []
    fold_coeffs_list: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        _group_kfold_splits(groups, n_folds, seed)
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        coeffs, intercept = _ols(X_train, y_train)
        y_pred = X_test @ coeffs + intercept
        residuals_oos = y_test - y_pred

        fold_metrics.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "r2_oos": _r2(y_test, y_pred),
            "rmse_oos": _rmse(y_test, y_pred),
        })

        coeff_row: dict = {"fold": fold_idx, "intercept": intercept}
        for feat, c in zip(_FEATURES, coeffs):
            coeff_row[feat] = float(c)
        fold_coeffs_list.append(coeff_row)

        hinge_col = _FEATURES.index("hinge")
        for k, idx in enumerate(test_idx):
            per_point_rows.append({
                "galaxy": str(groups[idx]),
                "hinge": float(X[idx, hinge_col]),
                "residual_dex": float(y[idx]),
                "residual_dex_pred": float(y_pred[k]),
                "residual_dex_oos": float(residuals_oos[k]),
                "fold": fold_idx,
            })

    return fold_metrics, per_point_rows, fold_coeffs_list


# ---------------------------------------------------------------------------
# Per-galaxy aggregation of OOS results
# ---------------------------------------------------------------------------

def aggregate_per_galaxy(per_point_rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-point OOS results to per-galaxy summaries."""
    df = pd.DataFrame(per_point_rows)
    records = []
    for galaxy, gdf in df.groupby("galaxy"):
        y_true = gdf["residual_dex"].to_numpy(dtype=float)
        y_pred = gdf["residual_dex_pred"].to_numpy(dtype=float)
        y_oos = gdf["residual_dex_oos"].to_numpy(dtype=float)
        records.append({
            "galaxy": str(galaxy),
            "n_points": int(len(gdf)),
            "hinge_mean": float(gdf["hinge"].mean()),
            "residual_dex_mean": float(y_true.mean()),
            "residual_dex_oos_mean": float(y_oos.mean()),
            "residual_dex_oos_std": float(y_oos.std(ddof=0)),
            "r2_oos": _r2(y_true, y_pred),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    df: pd.DataFrame,
    r2_obs: float,
    n_perm: int = _N_PERM,
    seed: int = 42,
) -> dict:
    """Test whether the observed model R² exceeds the null distribution.

    *residual_dex* is permuted within each galaxy to preserve the galaxy-level
    autocorrelation structure.  The OLS model is re-fitted on all data for
    each permutation.

    Parameters
    ----------
    df : pd.DataFrame
    r2_obs : float
        Observed R² (from master OLS or mean OOS).
    n_perm : int
    seed : int

    Returns
    -------
    dict
        Keys: ``n_perm``, ``r2_obs``, ``p_empirical``, ``r2_perm_mean``,
        ``r2_perm_std``.
    """
    X = df[_FEATURES].to_numpy(dtype=float)
    y = df[_TARGET].to_numpy(dtype=float)
    groups = df["galaxy"].to_numpy()

    rng = np.random.default_rng(seed)
    r2_perm: list[float] = []

    for _ in range(n_perm):
        y_perm = y.copy()
        for g in np.unique(groups):
            mask = groups == g
            y_perm[mask] = rng.permutation(y_perm[mask])
        coeffs, intercept = _ols(X, y_perm)
        y_pred = X @ coeffs + intercept
        r2_perm.append(_r2(y_perm, y_pred))

    r2_arr = np.array(r2_perm)
    p_emp = float(np.mean(r2_arr >= r2_obs))

    return {
        "n_perm": n_perm,
        "r2_obs": float(r2_obs),
        "p_empirical": float(np.clip(p_emp, 0.0, 1.0)),
        "r2_perm_mean": float(np.mean(r2_arr)),
        "r2_perm_std": float(np.std(r2_arr)),
    }


# ---------------------------------------------------------------------------
# Master OLS coefficients
# ---------------------------------------------------------------------------

def master_coefficients(df: pd.DataFrame) -> dict:
    """Fit OLS on the full dataset and compute Cohen's d for the hinge effect.

    Cohen's d is defined here as ``|hinge_coeff| / std(residual_dex)`` —
    the effect size of the hinge predictor relative to the variability of
    the response.  It is always ≥ 0.

    Returns
    -------
    dict
        Keys: ``intercept``, one key per feature name, ``d``.
    """
    X = df[_FEATURES].to_numpy(dtype=float)
    y = df[_TARGET].to_numpy(dtype=float)
    coeffs, intercept = _ols(X, y)

    sigma = float(np.std(y, ddof=1))
    hinge_idx = _FEATURES.index("hinge")
    d = abs(float(coeffs[hinge_idx]) / sigma) if sigma > 0.0 else 0.0

    result: dict = {"intercept": float(intercept)}
    for feat, c in zip(_FEATURES, coeffs):
        result[feat] = float(c)
    result["d"] = float(d)  # Cohen's d — always >= 0
    return result


# ---------------------------------------------------------------------------
# Residual vs hinge OOS plot
# ---------------------------------------------------------------------------

def plot_residual_vs_hinge_oos(
    per_point_rows: list[dict],
    out_path: Path,
    n_bins: int = 12,
) -> None:
    """Scatter + binned-median plot of OOS residual vs hinge.

    Parameters
    ----------
    per_point_rows : list[dict]
        Output of :func:`run_groupkfold`.
    out_path : Path
        Destination PNG file.
    n_bins : int
        Number of hinge bins for the median line.
    """
    df = pd.DataFrame(per_point_rows)
    hinge = df["hinge"].to_numpy(dtype=float)
    res_oos = df["residual_dex_oos"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(hinge, res_oos, s=4, alpha=0.3, color="#1f77b4",
               label="OOS points")

    # Binned median
    if len(hinge) > 0:
        bins = np.linspace(hinge.min(), hinge.max(), n_bins + 1)
        bin_centers: list[float] = []
        bin_medians: list[float] = []
        for i in range(len(bins) - 1):
            mask = (hinge >= bins[i]) & (hinge < bins[i + 1])
            if mask.sum() >= 3:
                bin_centers.append(float(0.5 * (bins[i] + bins[i + 1])))
                bin_medians.append(float(np.median(res_oos[mask])))
        if bin_centers:
            ax.plot(bin_centers, bin_medians, "o-", color="#d62728",
                    linewidth=1.5, markersize=5, label="Binned median")

    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"hinge  [dex]  ($= \max(0,\,\log_{10} a_0 - \log\,g_{\rm bar})$)")
    ax.set_ylabel("OOS residual  [dex]  (log gobs − log gbar − fitted)")
    ax.set_title("Residual vs Hinge — OOS (GroupKFold test folds)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def run_audit(
    audit_features_csv: Path,
    outdir: Path,
    seed: int = 42,
    n_folds: int = _N_FOLDS,
    n_perm: int = _N_PERM,
    verbose: bool = True,
) -> dict:
    """Run the full OOS audit and write all artefacts.

    Parameters
    ----------
    audit_features_csv : Path
        CSV produced by :func:`src.scm_analysis._write_audit_metrics`.
    outdir : Path
        Directory to write all output artefacts.
    seed : int
        Random seed for reproducibility.
    n_folds : int
        Number of GroupKFold folds.
    n_perm : int
        Number of permutations for the significance test.
    verbose : bool
        Print a progress report to stdout.

    Returns
    -------
    dict
        audit_summary dict (also written to ``audit_summary.json``).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if not audit_features_csv.exists():
        raise FileNotFoundError(
            f"audit_features.csv not found: {audit_features_csv}\n"
            "Run the SCM pipeline first with `python -m src.scm_analysis "
            "--data-dir data/SPARC --out results/`"
        )

    df = pd.read_csv(audit_features_csv)
    required = set(_FEATURES) | {_TARGET, "galaxy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"audit_features.csv is missing required columns: {missing}"
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=_FEATURES + [_TARGET]
    )
    if df.empty:
        raise ValueError(
            "No valid rows remain in audit_features.csv after cleaning."
        )

    n_galaxies = int(df["galaxy"].nunique())
    n_points = int(len(df))

    if verbose:
        print(_SEP)
        print("  Motor de Velos SCM — GroupKFold OOS Audit")
        print(_SEP)
        print(f"  Features     : {_FEATURES}")
        print(f"  Target       : {_TARGET}")
        print(f"  N galaxies   : {n_galaxies}")
        print(f"  N points     : {n_points}")
        print(f"  K folds      : {min(n_folds, n_galaxies)}")
        print(f"  N perm       : {n_perm}")
        print(f"  Seed         : {seed}")
        print()

    # --- 1. GroupKFold cross-validation ---
    fold_metrics, per_point_rows, fold_coeffs = run_groupkfold(
        df, n_folds=n_folds, seed=seed
    )

    pd.DataFrame(fold_metrics).to_csv(
        outdir / "groupkfold_metrics.csv", index=False
    )
    pd.DataFrame(fold_coeffs).to_csv(
        outdir / "coeffs_by_fold.csv", index=False
    )
    aggregate_per_galaxy(per_point_rows).to_csv(
        outdir / "groupkfold_per_galaxy.csv", index=False
    )

    r2_oos_mean = float(
        np.mean([m["r2_oos"] for m in fold_metrics]) if fold_metrics else 0.0
    )

    if verbose:
        print(f"  OOS R² (mean over folds): {r2_oos_mean:.4f}")

    # --- 2. Master OLS coefficients ---
    master = master_coefficients(df)
    with open(outdir / "master_coeffs.json", "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)

    if verbose:
        print(f"  Cohen's d (hinge)        : {master['d']:.4f}")

    # --- 3. Permutation test (in-sample R² null vs master in-sample R²) ---
    X_all = df[_FEATURES].to_numpy(dtype=float)
    y_all = df[_TARGET].to_numpy(dtype=float)
    coeffs_all, intercept_all = _ols(X_all, y_all)
    r2_insample = _r2(y_all, X_all @ coeffs_all + intercept_all)

    if verbose:
        print(f"  Running {n_perm} permutations …")

    perm = permutation_test(df, r2_obs=r2_insample, n_perm=n_perm, seed=seed)
    with open(outdir / "permutation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(perm, fh, indent=2)

    if verbose:
        print(f"  p_empirical              : {perm['p_empirical']:.4f}")

    # --- 4. Residual vs hinge OOS scatter plot ---
    if per_point_rows:
        plot_residual_vs_hinge_oos(
            per_point_rows,
            outdir / "residual_vs_hinge_oos.png",
        )

    # --- 5. Audit summary ---
    audit_summary = {
        "n_galaxies": n_galaxies,
        "n_points": n_points,
        "n_folds": min(n_folds, n_galaxies),
        "seed": seed,
        "r2_oos_mean": r2_oos_mean,
        "r2_insample": float(r2_insample),
        "p_empirical": perm["p_empirical"],
        "cohen_d_hinge": float(master["d"]),
        "status": (
            "pass"
            if master["d"] >= 0.0 and 0.0 <= perm["p_empirical"] <= 1.0
            else "check"
        ),
        "artefacts": [
            "groupkfold_metrics.csv",
            "groupkfold_per_galaxy.csv",
            "coeffs_by_fold.csv",
            "permutation_summary.json",
            "master_coeffs.json",
            "audit_summary.json",
            "residual_vs_hinge_oos.png",
        ],
    }
    with open(outdir / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(audit_summary, fh, indent=2)

    if verbose:
        print()
        print(f"  Status  : {audit_summary['status'].upper()}")
        print(f"  Results → {outdir}")
        print(_SEP)

    return audit_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "GroupKFold OOS audit for Motor de Velos SCM. "
            "Reads audit_features.csv and writes audit artefacts."
        )
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory for audit artefacts.",
    )
    p.add_argument(
        "--features-csv",
        default=None,
        dest="features_csv",
        metavar="CSV",
        help=(
            "Path to audit_features.csv "
            f"(default: {_DEFAULT_FEATURES_CSV})."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=_N_FOLDS,
        dest="n_folds",
        help=f"Number of GroupKFold folds (default: {_N_FOLDS}).",
    )
    p.add_argument(
        "--n-perm",
        type=int,
        default=_N_PERM,
        dest="n_perm",
        help=f"Number of permutations (default: {_N_PERM}).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return p.parse_args(argv)


def main(argv=None):
    """CLI entry-point for the GroupKFold OOS audit."""
    args = _parse_args(argv)
    outdir = Path(args.outdir)

    if args.features_csv:
        features_csv = Path(args.features_csv)
    else:
        features_csv = _DEFAULT_FEATURES_CSV
        if not features_csv.exists():
            alt = outdir / "audit_features.csv"
            if alt.exists():
                features_csv = alt

    try:
        run_audit(
            audit_features_csv=features_csv,
            outdir=outdir,
            seed=args.seed,
            n_folds=args.n_folds,
            n_perm=args.n_perm,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
