#!/usr/bin/env python3
"""
scripts/audit_scm.py

Rigorous out-of-sample audit for the SCM framework.

Key methodological guarantees:
  - True OOS: parameters are fit ONLY on TRAIN folds (GroupKFold by galaxy),
    evaluated on TEST folds.
  - Hard permutation test: shuffle log_gbar ACROSS galaxies (global shuffle),
    not within-galaxy. This breaks the RAR-like association rather than only
    internal ordering.
  - Physical constraint: if fitted hinge coefficient d < 0 (unphysical),
    refit without the hinge term (set d = 0).

Outputs (written under --outdir):
  - groupkfold_metrics.csv
  - groupkfold_per_galaxy.csv
  - coeffs_by_fold.csv
  - permutation_summary.json
  - master_coeffs.json
  - audit_summary.json

Example:
  python scripts/audit_scm.py --input data/sparc_global.csv --outdir results/audit --seed 123
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.stats import wilcoxon


# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _rmse_from_sse_n(sse: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(math.sqrt(sse / n))


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _lstsq_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ordinary least squares using numpy lstsq.
    Returns (weights, rss).
    """
    w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    if residuals.size > 0:
        rss = float(residuals[0])
    else:
        # If residuals not returned (e.g., rank-deficient), compute manually
        yhat = X @ w
        rss = float(np.sum((y - yhat) ** 2))
    return w, rss


def _design_btfr(logM: np.ndarray) -> np.ndarray:
    # log v = beta*logM + C
    return np.column_stack([logM, np.ones_like(logM)])


def _design_no_hinge(logM: np.ndarray, log_gbar: np.ndarray, log_j: np.ndarray) -> np.ndarray:
    # log v = beta*logM + C + a*log_gbar + b*log_j
    return np.column_stack([logM, np.ones_like(logM), log_gbar, log_j])


def _hinge_term(log_gbar: np.ndarray, logg0: float) -> np.ndarray:
    return np.maximum(0.0, logg0 - log_gbar)


def _design_full(logM: np.ndarray, log_gbar: np.ndarray, log_j: np.ndarray, logg0: float) -> np.ndarray:
    # log v = beta*logM + C + a*log_gbar + b*log_j + d*max(0, logg0 - log_gbar)
    h = _hinge_term(log_gbar, logg0)
    return np.column_stack([logM, np.ones_like(logM), log_gbar, log_j, h])


@dataclass
class FitResult:
    beta: float
    C: float
    a: float
    b: float
    d: float
    logg0: float
    rss: float
    k: int  # number of parameters in the fitted model (effective)

    def to_row(self, fold: int, model: str) -> dict:
        return {
            "fold": fold,
            "model": model,
            "beta": self.beta,
            "C": self.C,
            "a": self.a,
            "b": self.b,
            "d": self.d,
            "logg0": self.logg0,
            "rss_train": self.rss,
            "k": self.k,
        }


def fit_btfr(y_train: np.ndarray, logM: np.ndarray) -> FitResult:
    X = _design_btfr(logM)
    w, rss = _lstsq_fit(X, y_train)
    beta, C = w[0], w[1]
    return FitResult(beta=_safe_float(beta), C=_safe_float(C), a=0.0, b=0.0, d=0.0, logg0=float("nan"), rss=rss, k=2)


def fit_scm_no_hinge(y_train: np.ndarray, logM: np.ndarray, log_gbar: np.ndarray, log_j: np.ndarray) -> FitResult:
    X = _design_no_hinge(logM, log_gbar, log_j)
    w, rss = _lstsq_fit(X, y_train)
    beta, C, a, b = w[0], w[1], w[2], w[3]
    return FitResult(beta=_safe_float(beta), C=_safe_float(C), a=_safe_float(a), b=_safe_float(b), d=0.0, logg0=float("nan"), rss=rss, k=4)


def fit_scm_full(
    y_train: np.ndarray,
    logM: np.ndarray,
    log_gbar: np.ndarray,
    log_j: np.ndarray,
    logg0: float,
) -> FitResult:
    """
    Fit full SCM model with hinge term.
    If d < 0 (unphysical), drop hinge and refit (d=0).
    """
    X = _design_full(logM, log_gbar, log_j, logg0)
    w, rss = _lstsq_fit(X, y_train)
    beta, C, a, b, d = w[0], w[1], w[2], w[3], w[4]
    d = _safe_float(d)
    if not np.isfinite(d):
        d = float("nan")

    if np.isfinite(d) and d < 0.0:
        # Unphysical hinge strength -> refit without hinge.
        X2 = _design_no_hinge(logM, log_gbar, log_j)
        w2, rss2 = _lstsq_fit(X2, y_train)
        beta2, C2, a2, b2 = w2[0], w2[1], w2[2], w2[3]
        return FitResult(
            beta=_safe_float(beta2),
            C=_safe_float(C2),
            a=_safe_float(a2),
            b=_safe_float(b2),
            d=0.0,
            logg0=logg0,
            rss=rss2,
            k=4,  # hinge dropped
        )

    return FitResult(
        beta=_safe_float(beta),
        C=_safe_float(C),
        a=_safe_float(a),
        b=_safe_float(b),
        d=_safe_float(d),
        logg0=logg0,
        rss=rss,
        k=5,
    )


def predict_btfr(fr: FitResult, logM: np.ndarray) -> np.ndarray:
    X = _design_btfr(logM)
    w = np.array([fr.beta, fr.C], dtype=float)
    return X @ w


def predict_no_hinge(fr: FitResult, logM: np.ndarray, log_gbar: np.ndarray, log_j: np.ndarray) -> np.ndarray:
    X = _design_no_hinge(logM, log_gbar, log_j)
    w = np.array([fr.beta, fr.C, fr.a, fr.b], dtype=float)
    return X @ w


def predict_full(fr: FitResult, logM: np.ndarray, log_gbar: np.ndarray, log_j: np.ndarray) -> np.ndarray:
    if fr.d == 0.0 or not np.isfinite(fr.d) or fr.k == 4:
        # hinge absent
        return predict_no_hinge(fr, logM, log_gbar, log_j)
    X = _design_full(logM, log_gbar, log_j, fr.logg0)
    w = np.array([fr.beta, fr.C, fr.a, fr.b, fr.d], dtype=float)
    return X @ w


# -----------------------------
# Loading & validation
# -----------------------------

def load_and_validate(
    input_csv: str,
    galaxy_col: str,
    y_col: str,
    logM_col: str,
    log_gbar_col: str,
    log_j_col: str,
    strict: bool,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required = [galaxy_col, y_col, logM_col, log_gbar_col, log_j_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numeric
    for c in [y_col, logM_col, log_gbar_col, log_j_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaN in required numeric fields
    before = len(df)
    df = df.dropna(subset=[y_col, logM_col, log_gbar_col, log_j_col]).copy()
    after = len(df)

    # galaxy id should not be empty
    df[galaxy_col] = df[galaxy_col].astype(str)
    empty_mask = df[galaxy_col].str.strip().eq("") | df[galaxy_col].isna()
    if empty_mask.any():
        n_bad = int(empty_mask.sum())
        msg = f"{n_bad} rows have empty {galaxy_col}"
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg} (dropping)", file=sys.stderr)
        df = df.loc[~empty_mask].copy()

    # Duplicates are valid for radial rows, so do NOT fail.
    dup_mask = df.duplicated(subset=[galaxy_col], keep=False)
    # This "duplicate" definition is intentionally weak (galaxy-level), because input can be multi-row per galaxy.
    # We only warn if there are exact duplicated FULL ROWS.
    dup_full = df.duplicated(keep=False)
    if dup_full.any():
        n_dup = int(dup_full.sum())
        print(f"WARNING: detected {n_dup} duplicated full rows (allowed; check upstream ingestion).", file=sys.stderr)

    # Basic sanity
    if len(df) == 0:
        raise ValueError("No valid rows after validation.")
    if df[galaxy_col].nunique() < 2:
        raise ValueError("Need at least 2 galaxies for GroupKFold.")

    if after < before:
        print(f"INFO: dropped {before - after} rows with NaNs in required numeric fields.", file=sys.stderr)

    return df


# -----------------------------
# GroupKFold audit
# -----------------------------

@dataclass
class FoldOOSMetrics:
    fold: int
    n_test_rows: int
    n_test_galaxies: int
    rmse_btfr: float
    rmse_no_hinge: float
    rmse_full: float

    def to_row(self) -> dict:
        return {
            "fold": self.fold,
            "n_test_rows": self.n_test_rows,
            "n_test_galaxies": self.n_test_galaxies,
            "rmse_btfr": self.rmse_btfr,
            "rmse_no_hinge": self.rmse_no_hinge,
            "rmse_full": self.rmse_full,
        }


def groupkfold_audit(
    df: pd.DataFrame,
    galaxy_col: str,
    y_col: str,
    logM_col: str,
    log_gbar_col: str,
    log_j_col: str,
    kfold: int,
    seed: int,
    logg0: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit BTFR / SCM(no hinge) / SCM(full) on TRAIN folds, evaluate on TEST.
    Return:
      - fold_metrics_df
      - per_galaxy_df (aggregated over all OOS rows per galaxy)
      - coeffs_by_fold_df
      - headline summary dict
    """
    rng = np.random.default_rng(seed)
    # GroupKFold is deterministic given ordering; we can shuffle rows once for robustness.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    groups = df[galaxy_col].values
    y = df[y_col].to_numpy(dtype=float)
    logM = df[logM_col].to_numpy(dtype=float)
    log_gbar = df[log_gbar_col].to_numpy(dtype=float)
    log_j = df[log_j_col].to_numpy(dtype=float)

    gkf = GroupKFold(n_splits=kfold)

    fold_rows: List[dict] = []
    coeff_rows: List[dict] = []

    # Accumulators for per-galaxy RMSE across OOS predictions
    gal_sse = {}  # galaxy -> dict(model->sse)
    gal_n = {}    # galaxy -> n points

    # Also accumulate overall OOS SSE for fold metrics
    for fold, (train_idx, test_idx) in enumerate(gkf.split(logM, y, groups)):
        y_tr, y_te = y[train_idx], y[test_idx]
        logM_tr, logM_te = logM[train_idx], logM[test_idx]
        gbar_tr, gbar_te = log_gbar[train_idx], log_gbar[test_idx]
        j_tr, j_te = log_j[train_idx], log_j[test_idx]

        # Fit on TRAIN only
        fr_btfr = fit_btfr(y_tr, logM_tr)
        fr_no = fit_scm_no_hinge(y_tr, logM_tr, gbar_tr, j_tr)
        fr_full = fit_scm_full(y_tr, logM_tr, gbar_tr, j_tr, logg0=logg0)

        coeff_rows.append(fr_btfr.to_row(fold=fold, model="btfr"))
        coeff_rows.append(fr_no.to_row(fold=fold, model="scm_no_hinge"))
        coeff_rows.append(fr_full.to_row(fold=fold, model="scm_full"))

        # Predict on TEST only
        yhat_btfr = predict_btfr(fr_btfr, logM_te)
        yhat_no = predict_no_hinge(fr_no, logM_te, gbar_te, j_te)
        yhat_full = predict_full(fr_full, logM_te, gbar_te, j_te)

        # Fold-level metrics
        sse_btfr = float(np.sum((y_te - yhat_btfr) ** 2))
        sse_no = float(np.sum((y_te - yhat_no) ** 2))
        sse_full = float(np.sum((y_te - yhat_full) ** 2))
        n_te = int(len(test_idx))
        n_gal_te = int(pd.Series(groups[test_idx]).nunique())

        fold_rows.append(
            FoldOOSMetrics(
                fold=fold,
                n_test_rows=n_te,
                n_test_galaxies=n_gal_te,
                rmse_btfr=_rmse_from_sse_n(sse_btfr, n_te),
                rmse_no_hinge=_rmse_from_sse_n(sse_no, n_te),
                rmse_full=_rmse_from_sse_n(sse_full, n_te),
            ).to_row()
        )

        # Per-galaxy accumulators
        gals_te = groups[test_idx]
        for i, gal in enumerate(gals_te):
            if gal not in gal_sse:
                gal_sse[gal] = {"btfr": 0.0, "scm_no_hinge": 0.0, "scm_full": 0.0}
                gal_n[gal] = 0
            gal_sse[gal]["btfr"] += float((y_te[i] - yhat_btfr[i]) ** 2)
            gal_sse[gal]["scm_no_hinge"] += float((y_te[i] - yhat_no[i]) ** 2)
            gal_sse[gal]["scm_full"] += float((y_te[i] - yhat_full[i]) ** 2)
            gal_n[gal] += 1

    fold_df = pd.DataFrame(fold_rows)
    coeff_df = pd.DataFrame(coeff_rows)

    # Build per-galaxy DF
    gal_rows: List[dict] = []
    for gal, sse_map in gal_sse.items():
        n = int(gal_n[gal])
        rmse_btfr = _rmse_from_sse_n(sse_map["btfr"], n)
        rmse_no = _rmse_from_sse_n(sse_map["scm_no_hinge"], n)
        rmse_full = _rmse_from_sse_n(sse_map["scm_full"], n)
        gal_rows.append(
            {
                "galaxy_id": gal,
                "n_points_oos": n,
                "rmse_btfr": rmse_btfr,
                "rmse_no_hinge": rmse_no,
                "rmse_full": rmse_full,
                "delta_rmse_full_minus_btfr": rmse_full - rmse_btfr,
                "delta_rmse_full_minus_no_hinge": rmse_full - rmse_no,
            }
        )
    gal_df = pd.DataFrame(gal_rows).sort_values("galaxy_id").reset_index(drop=True)

    # Wilcoxon at galaxy-level deltas (alternative="less"): are deltas < 0?
    # Need finite values.
    d1 = gal_df["delta_rmse_full_minus_btfr"].to_numpy(dtype=float)
    d2 = gal_df["delta_rmse_full_minus_no_hinge"].to_numpy(dtype=float)
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]

    wilc_btfr = None
    wilc_no = None
    if len(d1) >= 10:
        try:
            w = wilcoxon(d1, alternative="less")
            wilc_btfr = {"statistic": float(w.statistic), "pvalue": float(w.pvalue), "n": int(len(d1))}
        except Exception:
            wilc_btfr = {"statistic": None, "pvalue": None, "n": int(len(d1))}
    else:
        wilc_btfr = {"statistic": None, "pvalue": None, "n": int(len(d1))}

    if len(d2) >= 10:
        try:
            w = wilcoxon(d2, alternative="less")
            wilc_no = {"statistic": float(w.statistic), "pvalue": float(w.pvalue), "n": int(len(d2))}
        except Exception:
            wilc_no = {"statistic": None, "pvalue": None, "n": int(len(d2))}
    else:
        wilc_no = {"statistic": None, "pvalue": None, "n": int(len(d2))}

    # Headline summary
    summary = {
        "groupkfold": {
            "kfold": int(kfold),
            "n_rows": int(len(df)),
            "n_galaxies": int(pd.Series(groups).nunique()),
            "rmse_btfr_mean": float(fold_df["rmse_btfr"].mean()),
            "rmse_btfr_std": float(fold_df["rmse_btfr"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
            "rmse_no_hinge_mean": float(fold_df["rmse_no_hinge"].mean()),
            "rmse_no_hinge_std": float(fold_df["rmse_no_hinge"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
            "rmse_full_mean": float(fold_df["rmse_full"].mean()),
            "rmse_full_std": float(fold_df["rmse_full"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        },
        "galaxy_level": {
            "n_galaxies_oos": int(len(gal_df)),
            "median_delta_full_minus_btfr": float(np.nanmedian(gal_df["delta_rmse_full_minus_btfr"].values)),
            "median_delta_full_minus_no_hinge": float(np.nanmedian(gal_df["delta_rmse_full_minus_no_hinge"].values)),
            "wilcoxon_full_vs_btfr": wilc_btfr,
            "wilcoxon_full_vs_no_hinge": wilc_no,
        },
        "logg0": float(logg0),
    }

    return fold_df, gal_df, coeff_df, summary


# -----------------------------
# Permutation test (hard shuffle)
# -----------------------------

def permutation_test(
    df: pd.DataFrame,
    galaxy_col: str,
    y_col: str,
    logM_col: str,
    log_gbar_col: str,
    log_j_col: str,
    kfold: int,
    seed: int,
    logg0: float,
    n_perm: int,
) -> dict:
    """
    Hard permutation test:
      - Shuffle log_gbar across ALL rows (between galaxies).
      - For each permutation, refit FULL SCM on TRAIN for each fold, evaluate on TEST.

    Empirical p-value:
      p = (1 + #{perm_rmse <= real_rmse}) / (1 + N_perm)
    where rmse is mean fold rmse_full.
    """
    rng = np.random.default_rng(seed)

    # Compute real (non-permuted) score first
    fold_df_real, _, _, _ = groupkfold_audit(
        df=df,
        galaxy_col=galaxy_col,
        y_col=y_col,
        logM_col=logM_col,
        log_gbar_col=log_gbar_col,
        log_j_col=log_j_col,
        kfold=kfold,
        seed=seed,
        logg0=logg0,
    )
    real_rmse = float(fold_df_real["rmse_full"].mean())

    # Prepare arrays (we'll reuse same folds via deterministic shuffle+GroupKFold)
    df_base = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    groups = df_base[galaxy_col].values
    y = df_base[y_col].to_numpy(dtype=float)
    logM = df_base[logM_col].to_numpy(dtype=float)
    log_j = df_base[log_j_col].to_numpy(dtype=float)

    gkf = GroupKFold(n_splits=kfold)

    perm_rmse_means: List[float] = []

    # Pre-extract the original gbar column, shuffle it each perm
    gbar_orig = df_base[log_gbar_col].to_numpy(dtype=float)

    for p in range(n_perm):
        gbar_perm = rng.permutation(gbar_orig)  # global shuffle across rows

        fold_rmses: List[float] = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(logM, y, groups)):
            y_tr, y_te = y[train_idx], y[test_idx]
            logM_tr, logM_te = logM[train_idx], logM[test_idx]
            gbar_tr, gbar_te = gbar_perm[train_idx], gbar_perm[test_idx]
            j_tr, j_te = log_j[train_idx], log_j[test_idx]

            fr_full = fit_scm_full(y_tr, logM_tr, gbar_tr, j_tr, logg0=logg0)
            yhat_full = predict_full(fr_full, logM_te, gbar_te, j_te)
            sse = float(np.sum((y_te - yhat_full) ** 2))
            fold_rmses.append(_rmse_from_sse_n(sse, int(len(test_idx))))

        perm_rmse_means.append(float(np.mean(fold_rmses)))

    perm_arr = np.array(perm_rmse_means, dtype=float)
    perm_mean = float(np.mean(perm_arr))
    perm_std = float(np.std(perm_arr, ddof=1)) if len(perm_arr) > 1 else 0.0

    # Empirical p-value (lower is "better" for real if real_rmse is small)
    count = int(np.sum(perm_arr <= real_rmse))
    p_emp = float((1 + count) / (1 + n_perm))

    return {
        "real_rmse_full_mean": real_rmse,
        "perm_rmse_full_mean_mean": perm_mean,
        "perm_rmse_full_mean_std": perm_std,
        "n_perm": int(n_perm),
        "count_perm_leq_real": count,
        "p_empirical": p_emp,
        "perm_rmse_means": perm_rmse_means,  # keep for transparency; can be large but useful
        "definition": "Hard shuffle of log_gbar across all rows; refit full model on TRAIN per fold; score is mean fold RMSE.",
        "p_definition": "p = (1 + #{perm_rmse <= real_rmse}) / (1 + N_perm)",
    }


# -----------------------------
# Master coefficients (descriptive)
# -----------------------------

def fit_master_coeffs(
    df: pd.DataFrame,
    y_col: str,
    logM_col: str,
    log_gbar_col: str,
    log_j_col: str,
    logg0: float,
) -> dict:
    """
    Global fit over full dataset (DESCRIPTIVE ONLY).
    The science claim is based on OOS GroupKFold results.
    """
    y = df[y_col].to_numpy(dtype=float)
    logM = df[logM_col].to_numpy(dtype=float)
    gbar = df[log_gbar_col].to_numpy(dtype=float)
    j = df[log_j_col].to_numpy(dtype=float)

    fr_btfr = fit_btfr(y, logM)
    fr_no = fit_scm_no_hinge(y, logM, gbar, j)
    fr_full = fit_scm_full(y, logM, gbar, j, logg0=logg0)

    return {
        "btfr": fr_btfr.__dict__,
        "scm_no_hinge": fr_no.__dict__,
        "scm_full": fr_full.__dict__,
        "note": "Descriptive fit on the full dataset. Do not use for claims; use OOS GroupKFold audit outputs.",
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rigorous OOS audit for SCM (GroupKFold + hard permutation).")
    p.add_argument("--input", required=True, help="CSV with columns: galaxy_id, logM, log_gbar, log_j, y (log v_obs).")
    p.add_argument("--outdir", default="results/audit", help="Output directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for row shuffling and permutations.")
    p.add_argument("--kfold", type=int, default=5, help="Number of GroupKFold splits.")
    p.add_argument("--permutations", type=int, default=200, help="Number of hard permutations.")
    p.add_argument("--logg0", type=float, default=-10.45, help="Fixed log10(g0) threshold for hinge term.")
    p.add_argument("--strict", action="store_true", help="Strict validation (fail on empty galaxy_id, etc.).")

    # Column mapping
    p.add_argument("--galaxy-col", default="galaxy_id")
    p.add_argument("--y-col", default="v_obs", help="Target column (assumed already log10(v_obs)).")
    p.add_argument("--logM-col", default="logM")
    p.add_argument("--log-gbar-col", default="log_gbar")
    p.add_argument("--log-j-col", default="log_j")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    df = load_and_validate(
        input_csv=args.input,
        galaxy_col=args.galaxy_col,
        y_col=args.y_col,
        logM_col=args.logM_col,
        log_gbar_col=args.log_gbar_col,
        log_j_col=args.log_j_col,
        strict=bool(args.strict),
    )

    # 1) GroupKFold OOS audit
    fold_df, gal_df, coeff_df, headline = groupkfold_audit(
        df=df,
        galaxy_col=args.galaxy_col,
        y_col=args.y_col,
        logM_col=args.logM_col,
        log_gbar_col=args.log_gbar_col,
        log_j_col=args.log_j_col,
        kfold=int(args.kfold),
        seed=int(args.seed),
        logg0=float(args.logg0),
    )

    fold_df.to_csv(outdir / "groupkfold_metrics.csv", index=False)
    gal_df.to_csv(outdir / "groupkfold_per_galaxy.csv", index=False)
    coeff_df.to_csv(outdir / "coeffs_by_fold.csv", index=False)

    # 2) Hard permutation test
    perm = permutation_test(
        df=df,
        galaxy_col=args.galaxy_col,
        y_col=args.y_col,
        logM_col=args.logM_col,
        log_gbar_col=args.log_gbar_col,
        log_j_col=args.log_j_col,
        kfold=int(args.kfold),
        seed=int(args.seed),
        logg0=float(args.logg0),
        n_perm=int(args.permutations),
    )
    _write_json(outdir / "permutation_summary.json", {
        "real_rmse_full_mean": perm["real_rmse_full_mean"],
        "perm_rmse_full_mean_mean": perm["perm_rmse_full_mean_mean"],
        "perm_rmse_full_mean_std": perm["perm_rmse_full_mean_std"],
        "n_perm": perm["n_perm"],
        "count_perm_leq_real": perm["count_perm_leq_real"],
        "p_empirical": perm["p_empirical"],
        "definition": perm["definition"],
        "p_definition": perm["p_definition"],
    })

    # 3) Master coefficients (descriptive)
    master = fit_master_coeffs(
        df=df,
        y_col=args.y_col,
        logM_col=args.logM_col,
        log_gbar_col=args.log_gbar_col,
        log_j_col=args.log_j_col,
        logg0=float(args.logg0),
    )
    _write_json(outdir / "master_coeffs.json", master)

    # 4) One source of truth summary (headlines)
    audit_summary = {
        "input": str(args.input),
        "outdir": str(outdir),
        "seed": int(args.seed),
        "kfold": int(args.kfold),
        "permutations": int(args.permutations),
        "logg0": float(args.logg0),
        "headline": headline,
        "permutation": {
            "real_rmse_full_mean": perm["real_rmse_full_mean"],
            "p_empirical": perm["p_empirical"],
            "perm_rmse_full_mean_mean": perm["perm_rmse_full_mean_mean"],
            "perm_rmse_full_mean_std": perm["perm_rmse_full_mean_std"],
        },
        "artifacts": {
            "groupkfold_metrics_csv": str(outdir / "groupkfold_metrics.csv"),
            "groupkfold_per_galaxy_csv": str(outdir / "groupkfold_per_galaxy.csv"),
            "coeffs_by_fold_csv": str(outdir / "coeffs_by_fold.csv"),
            "permutation_summary_json": str(outdir / "permutation_summary.json"),
            "master_coeffs_json": str(outdir / "master_coeffs.json"),
            "audit_summary_json": str(outdir / "audit_summary.json"),
        },
        "notes": [
            "Parameters are fit exclusively on TRAIN folds; evaluation is on TEST folds (true OOS).",
            "Permutation is a hard shuffle of log_gbar across rows (between galaxies).",
            "If hinge coefficient d < 0, hinge is dropped and the model is refit (physical constraint).",
            "Scientific claims should be based on OOS results and permutation p-value, not master_coeffs.json.",
        ],
    }
    _write_json(outdir / "audit_summary.json", audit_summary)

    # Console summary (short)
    hk = headline["groupkfold"]
    gal = headline["galaxy_level"]
    print("\n=== SCM AUDIT SUMMARY (OOS) ===")
    print(f"Rows: {hk['n_rows']} | Galaxies: {hk['n_galaxies']} | KFold: {hk['kfold']}")
    print(f"RMSE (BTFR)      : {hk['rmse_btfr_mean']:.5f} ± {hk['rmse_btfr_std']:.5f}")
    print(f"RMSE (No hinge)  : {hk['rmse_no_hinge_mean']:.5f} ± {hk['rmse_no_hinge_std']:.5f}")
    print(f"RMSE (SCM full)  : {hk['rmse_full_mean']:.5f} ± {hk['rmse_full_std']:.5f}")
    print(f"Median ΔRMSE full-btfr     : {gal['median_delta_full_minus_btfr']:.5f}")
    print(f"Median ΔRMSE full-no_hinge : {gal['median_delta_full_minus_no_hinge']:.5f}")
    w1 = gal["wilcoxon_full_vs_btfr"]
    w2 = gal["wilcoxon_full_vs_no_hinge"]
    if w1 and w1.get("pvalue") is not None:
        print(f"Wilcoxon(full vs btfr)     p={w1['pvalue']:.4f}  stat={w1['statistic']:.1f}  n={w1['n']}")
    if w2 and w2.get("pvalue") is not None:
        print(f"Wilcoxon(full vs no_hinge) p={w2['pvalue']:.4f}  stat={w2['statistic']:.1f}  n={w2['n']}")
    print(f"Permutation p (hard shuffle, N={perm['n_perm']}): {perm['p_empirical']:.4f}")
    print(f"  Perm RMSE mean ± std: {perm['perm_rmse_full_mean_mean']:.5f} ± {perm['perm_rmse_full_mean_std']:.5f}")
    print(f"\nResults written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
