#!/usr/bin/env python3
"""
test_rg_proxy.py

RG-style proxy test for the SCM framework.

Operational definition:
    proxy_rg = F3_outer - F3_inner

where F3_inner and F3_outer are the log-slope fits of the inner and outer
halves of the external region r >= r_threshold * Rmax.

This is not a continuous derivative dF3/dlogr; it is a finite-difference
proxy for radial evolution in the outer tail.

Inputs:
    --points    CSV/Parquet with per-point data:
                galaxy, r_kpc, v_obs_kms
    --galaxies  CSV with galaxy-level metadata:
                galaxy, logMbar, logRd, logSigmaHI_out
    --out       output directory

Outputs:
    results/rg_proxy/rg_per_galaxy.csv
    results/rg_proxy/rg_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


MIN_OUTER_POINTS = 4
BOOTSTRAP_N = 1000
RNG_SEED = 42


def _safe_read_table(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def split_outer_part(
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    r_threshold: float = 0.7,
) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Split the external region (r >= r_threshold * Rmax) into two radius-ordered halves.
    Returns DataFrames with columns: logr, logv
    """
    r = np.asarray(r_kpc, dtype=float)
    v = np.asarray(v_obs_kms, dtype=float)

    mask = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    r = r[mask]
    v = v[mask]

    if len(r) < MIN_OUTER_POINTS:
        return None, None

    r_max = np.max(r)
    outer_mask = r >= (r_threshold * r_max)
    if np.sum(outer_mask) < MIN_OUTER_POINTS:
        return None, None

    r_outer = r[outer_mask]
    v_outer = v[outer_mask]

    order = np.argsort(r_outer)
    r_outer = r_outer[order]
    v_outer = v_outer[order]

    logr = np.log10(r_outer)
    logv = np.log10(v_outer)

    n = len(logr)
    mid = n // 2
    if mid < 2 or (n - mid) < 2:
        return None, None

    inner = pd.DataFrame({"logr": logr[:mid], "logv": logv[:mid]})
    outer = pd.DataFrame({"logr": logr[mid:], "logv": logv[mid:]})
    return inner, outer


def slope_from_df(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Fit logv ~ logr and return:
        slope, slope_stderr, r_value
    """
    if df is None or len(df) < 2:
        return np.nan, np.nan, np.nan

    x = df["logr"].to_numpy(dtype=float)
    y = df["logv"].to_numpy(dtype=float)

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan

    res = stats.linregress(x, y)
    return float(res.slope), float(res.stderr), float(res.rvalue)


def compute_rg_proxy_for_galaxy(
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    r_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute finite-difference RG proxy:
        proxy_rg = F3_outer - F3_inner
    """
    inner_df, outer_df = split_outer_part(r_kpc, v_obs_kms, r_threshold=r_threshold)
    if inner_df is None or outer_df is None:
        return {
            "F3_inner": np.nan,
            "F3_outer": np.nan,
            "proxy_rg": np.nan,
            "proxy_rg_err": np.nan,
            "n_inner": 0,
            "n_outer": 0,
        }

    f3_inner, err_inner, r_inner = slope_from_df(inner_df)
    f3_outer, err_outer, r_outer = slope_from_df(outer_df)

    if not np.isfinite(f3_inner) or not np.isfinite(f3_outer):
        return {
            "F3_inner": np.nan,
            "F3_outer": np.nan,
            "proxy_rg": np.nan,
            "proxy_rg_err": np.nan,
            "n_inner": len(inner_df),
            "n_outer": len(outer_df),
        }

    proxy = f3_outer - f3_inner
    proxy_err = np.sqrt(err_inner**2 + err_outer**2)

    return {
        "F3_inner": f3_inner,
        "F3_outer": f3_outer,
        "proxy_rg": proxy,
        "proxy_rg_err": proxy_err,
        "n_inner": len(inner_df),
        "n_outer": len(outer_df),
        "r_inner_fit": r_inner,
        "r_outer_fit": r_outer,
    }


def _linear_regression_with_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    OLS using numpy only.
    Returns coefficients, R2, RMSE, and approximate p-values from t-stats.
    """
    n = len(y)
    p = X.shape[1]

    X1 = np.column_stack([np.ones(n), X])
    beta, residuals, rank, s = np.linalg.lstsq(X1, y, rcond=None)
    y_pred = X1 @ beta
    resid = y - y_pred

    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid**2)))

    dof = n - (p + 1)
    if dof > 0:
        sigma2 = rss / dof
        cov = sigma2 * np.linalg.pinv(X1.T @ X1)
        se = np.sqrt(np.diag(cov))
        tvals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=dof))
    else:
        se = np.full_like(beta, np.nan, dtype=float)
        tvals = np.full_like(beta, np.nan, dtype=float)
        pvals = np.full_like(beta, np.nan, dtype=float)

    return {
        "intercept": float(beta[0]),
        "coef_logSigmaHI_out": float(beta[1]),
        "coef_logMbar": float(beta[2]),
        "coef_logRd": float(beta[3]),
        "stderr_intercept": float(se[0]),
        "stderr_logSigmaHI_out": float(se[1]),
        "stderr_logMbar": float(se[2]),
        "stderr_logRd": float(se[3]),
        "p_intercept": float(pvals[0]),
        "p_logSigmaHI_out": float(pvals[1]),
        "p_logMbar": float(pvals[2]),
        "p_logRd": float(pvals[3]),
        "R2_full": float(r2),
        "RMSE": float(rmse),
        "y_pred": y_pred,
    }


def analyze_rg_proxy(
    points_file: str | Path,
    galaxies_file: str | Path,
    out_dir: str | Path,
    r_threshold: float = 0.7,
    bootstrap_n: int = BOOTSTRAP_N,
    seed: int = RNG_SEED,
) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = _safe_read_table(Path(points_file))
    galaxies = pd.read_csv(galaxies_file)

    _validate_columns(pts, ["galaxy", "r_kpc", "v_obs_kms"], "points_file")
    _validate_columns(
        galaxies,
        ["galaxy", "logMbar", "logRd", "logSigmaHI_out"],
        "galaxies_file",
    )

    galaxies = galaxies.drop_duplicates(subset=["galaxy"]).copy()

    rows = []
    for gal in galaxies["galaxy"]:
        sub = pts.loc[pts["galaxy"] == gal, ["r_kpc", "v_obs_kms"]].copy()
        if sub.empty:
            continue

        rg = compute_rg_proxy_for_galaxy(
            sub["r_kpc"].to_numpy(),
            sub["v_obs_kms"].to_numpy(),
            r_threshold=r_threshold,
        )
        if not np.isfinite(rg["proxy_rg"]):
            continue

        meta = galaxies.loc[galaxies["galaxy"] == gal].iloc[0]
        if (
            not np.isfinite(meta["logMbar"])
            or not np.isfinite(meta["logRd"])
            or not np.isfinite(meta["logSigmaHI_out"])
        ):
            continue

        rows.append(
            {
                "galaxy": gal,
                "proxy_rg": rg["proxy_rg"],
                "proxy_rg_err": rg["proxy_rg_err"],
                "F3_inner": rg["F3_inner"],
                "F3_outer": rg["F3_outer"],
                "n_inner": rg["n_inner"],
                "n_outer": rg["n_outer"],
                "logMbar": float(meta["logMbar"]),
                "logRd": float(meta["logRd"]),
                "logSigmaHI_out": float(meta["logSigmaHI_out"]),
            }
        )

    if not rows:
        raise RuntimeError("No galaxies with valid RG proxy values.")

    df = pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True)

    x_env = df["logSigmaHI_out"].to_numpy(dtype=float)
    y = df["proxy_rg"].to_numpy(dtype=float)

    if len(df) >= 3:
        pearson_r, pearson_p = stats.pearsonr(x_env, y)
        spearman_rho, spearman_p = stats.spearmanr(x_env, y)
    else:
        pearson_r, pearson_p, spearman_rho, spearman_p = [np.nan] * 4

    X = df[["logSigmaHI_out", "logMbar", "logRd"]].to_numpy(dtype=float)
    reg = _linear_regression_with_stats(X, y)
    df["predicted_proxy_rg"] = reg["y_pred"]
    df.to_csv(out_dir / "rg_per_galaxy.csv", index=False)

    rng = np.random.default_rng(seed)
    boot_coef_env = []
    for _ in range(bootstrap_n):
        idx = rng.choice(len(df), size=len(df), replace=True)
        Xb = X[idx]
        yb = y[idx]
        reg_b = _linear_regression_with_stats(Xb, yb)
        boot_coef_env.append(reg_b["coef_logSigmaHI_out"])

    ci_low, ci_high = np.percentile(boot_coef_env, [2.5, 97.5])

    summary = {
        "status": "ok",
        "n_galaxies": int(len(df)),
        "r_threshold": float(r_threshold),
        "observable_definition": "proxy_rg = F3_outer - F3_inner in the external region",
        "pearson_r_env": float(pearson_r),
        "pearson_p_env": float(pearson_p),
        "spearman_rho_env": float(spearman_rho),
        "spearman_p_env": float(spearman_p),
        "coef_logSigmaHI_out": reg["coef_logSigmaHI_out"],
        "coef_logMbar": reg["coef_logMbar"],
        "coef_logRd": reg["coef_logRd"],
        "stderr_logSigmaHI_out": reg["stderr_logSigmaHI_out"],
        "p_logSigmaHI_out": reg["p_logSigmaHI_out"],
        "bootstrap_ci_logSigmaHI_out": [float(ci_low), float(ci_high)],
        "R2_full": reg["R2_full"],
        "RMSE": reg["RMSE"],
        "intercept": reg["intercept"],
        "bootstrap_n": int(bootstrap_n),
        "seed": int(seed),
    }

    with open(out_dir / "rg_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RG-style proxy test for SCM.")
    parser.add_argument("--points", required=True, help="Per-point data (CSV or Parquet).")
    parser.add_argument("--galaxies", required=True, help="Galaxy-level metadata CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--r-threshold",
        type=float,
        default=0.7,
        help="Outer-region threshold as fraction of Rmax.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    summary = analyze_rg_proxy(
        points_file=args.points,
        galaxies_file=args.galaxies,
        out_dir=args.out,
        r_threshold=args.r_threshold,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
