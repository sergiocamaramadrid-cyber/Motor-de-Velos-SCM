#!/usr/bin/env python3
"""
test_f3_local_recurrence.py

Intra-galaxy recurrence test for the SCM framework using SPARC-style rotmod files.

Core observable
---------------
    F3_local = local slope of log10(Vobs) vs log10(r)

For each galaxy, the script computes F3_local on a centered rolling window and
tests whether F3_{i+1} is better described by:

    (1) null model:     y = c
    (2) recurrence:     y = c + a * F3_i
    (3) recurrence+bar: y = c + a * F3_i + b * Δlog10(Vbar_i)

where y = F3_{i+1}.

Outputs
-------
- per_galaxy_f3_recurrence.csv
- top20_f3_recurrence_improve.csv
- top20_f3_recurrence_worsen.csv
- skipped_galaxies.csv
- executive_summary.json

Expected rotmod columns (standard SPARC fallback mapping)
---------------------------------------------------------
    col0 = radius_kpc
    col1 = vobs_kms
    col2 = err_vobs_kms
    col3 = vgas_kms
    col4 = vdisk_kms
    col5 = vbul_kms
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12
MIN_POINTS_PER_GALAXY = 10
DEFAULT_WINDOW = 5
DEFAULT_BOOTSTRAP = 500
MIN_RECURRENCE_PAIRS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test F3_local recurrence on SPARC rotmod data."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing SPARC data. Prefers data_dir/rotmod/*_rotmod.dat, "
             "falls back to data_dir/*_rotmod.dat.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/f3_local_recurrence"),
        help="Output directory.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Centered rolling window size for local slope. Prefer odd values.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=MIN_POINTS_PER_GALAXY,
        help="Minimum raw radial points per galaxy to attempt analysis.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="Bootstrap iterations per galaxy for recurrence slope CI.",
    )
    return parser.parse_args()


def robust_aicc(rss: float, n: int, k: int) -> float:
    if n <= k + 1:
        return np.inf
    rss = max(rss, EPS)
    aic = n * math.log(rss / n) + 2 * k
    correction = (2 * k * (k + 1)) / max(n - k - 1, EPS)
    return aic + correction


def rmse(residuals: np.ndarray) -> float:
    if residuals.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(residuals ** 2)))


def weighted_lstsq(
    X: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Weighted least squares via sqrt(weight) transform.

    Returns
    -------
    beta : ndarray
        Best-fit coefficients.
    residuals : ndarray
        Unweighted residuals in y-space.
    rss : float
        Weighted residual sum of squares.
    rmse_val : float
        Unweighted RMSE in y-space.
    """
    if w is None:
        sw = np.ones_like(y, dtype=float)
    else:
        sw = np.sqrt(np.clip(w.astype(float), EPS, np.inf))

    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    residuals = y - X @ beta
    rss = float(np.sum((sw * residuals) ** 2))
    return beta, residuals, rss, rmse(residuals)


def local_slope_loglog(r: np.ndarray, v: np.ndarray, window: int) -> np.ndarray:
    """
    Centered rolling local slope of log10(v) vs log10(r).

    Returns NaN where the window cannot be computed.
    """
    n = len(r)
    out = np.full(n, np.nan, dtype=float)

    if window < 3:
        raise ValueError("window must be >= 3")
    if window % 2 == 0:
        window -= 1  # force odd
    half = window // 2

    valid = (r > 0) & (v > 0) & np.isfinite(r) & np.isfinite(v)
    logr = np.full(n, np.nan, dtype=float)
    logv = np.full(n, np.nan, dtype=float)
    logr[valid] = np.log10(r[valid])
    logv[valid] = np.log10(v[valid])

    for i in range(half, n - half):
        sl = slice(i - half, i + half + 1)
        x = logr[sl]
        y = logv[sl]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            continue
        x = x[m]
        y = y[m]
        if np.nanstd(x) < EPS:
            continue
        A = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[i] = beta[1]

    return out


def load_rotmod_file(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < 3:
        raise ValueError(f"{path.name}: expected at least 3 columns, got {arr.shape[1]}")

    df = pd.DataFrame(
        {
            "r_kpc": arr[:, 0],
            "vobs_kms": arr[:, 1],
            "err_vobs_kms": arr[:, 2],
        }
    )

    df["vgas_kms"] = arr[:, 3] if arr.shape[1] > 3 else 0.0
    df["vdisk_kms"] = arr[:, 4] if arr.shape[1] > 4 else 0.0
    df["vbul_kms"] = arr[:, 5] if arr.shape[1] > 5 else 0.0

    df["vbar_kms"] = np.sqrt(
        np.clip(df["vgas_kms"], 0, np.inf) ** 2
        + np.clip(df["vdisk_kms"], 0, np.inf) ** 2
        + np.clip(df["vbul_kms"], 0, np.inf) ** 2
    )

    return df


def discover_rotmod_files(data_dir: Path) -> List[Path]:
    """
    Discover rotmod files with preference for <data_dir>/rotmod, while keeping
    backward compatibility with legacy layouts that place files directly under
    <data_dir>.
    """
    rotmod_dir = data_dir / "rotmod"
    if rotmod_dir.is_dir():
        files = sorted(rotmod_dir.glob("*_rotmod.dat"))
        if files:
            return files
    return sorted(data_dir.glob("*_rotmod.dat"))


def build_pair_table(galaxy: str, df: pd.DataFrame, window: int) -> pd.DataFrame:
    sub = df.sort_values("r_kpc").reset_index(drop=True).copy()

    sub["f3_local"] = local_slope_loglog(
        r=sub["r_kpc"].to_numpy(dtype=float),
        v=sub["vobs_kms"].to_numpy(dtype=float),
        window=window,
    )

    valid_bar = (sub["vbar_kms"] > 0) & np.isfinite(sub["vbar_kms"])
    sub["log_vbar"] = np.nan
    sub.loc[valid_bar, "log_vbar"] = np.log10(sub.loc[valid_bar, "vbar_kms"])
    sub["delta_log_vbar"] = sub["log_vbar"].diff()
    sub["f3_next"] = sub["f3_local"].shift(-1)

    pair = sub.loc[
        :,
        [
            "r_kpc",
            "vobs_kms",
            "err_vobs_kms",
            "vbar_kms",
            "f3_local",
            "f3_next",
            "delta_log_vbar",
        ],
    ].copy()
    pair["galaxy"] = galaxy

    mask = (
        np.isfinite(pair["f3_local"])
        & np.isfinite(pair["f3_next"])
        & np.isfinite(pair["r_kpc"])
        & (pair["r_kpc"] > 0)
    )
    return pair.loc[mask].reset_index(drop=True)


def bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    w: Optional[np.ndarray] = None,
    seed: int = 12345,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return np.nan, np.nan

    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        wb = None if w is None else w[idx]
        X = np.column_stack([np.ones_like(xb), xb])
        beta, _, _, _ = weighted_lstsq(X, yb, wb)
        slopes.append(beta[1])

    if not slopes:
        return np.nan, np.nan

    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def analyze_galaxy(
    galaxy: str,
    df: pd.DataFrame,
    window: int,
    n_boot: int,
) -> Optional[Dict[str, float]]:
    pair = build_pair_table(galaxy=galaxy, df=df, window=window)
    n_pairs = len(pair)
    if n_pairs < MIN_RECURRENCE_PAIRS:
        return None

    y = pair["f3_next"].to_numpy(dtype=float)
    x_prev = pair["f3_local"].to_numpy(dtype=float)
    x_bar = pair["delta_log_vbar"].to_numpy(dtype=float)

    err = pair["err_vobs_kms"].to_numpy(dtype=float)
    valid_err = np.isfinite(err) & (err > 0)
    median_err = np.median(err[valid_err]) if valid_err.any() else 1.0
    err = np.where(valid_err, err, median_err)
    w = 1.0 / np.clip(err, EPS, np.inf) ** 2

    # Null model
    X0 = np.ones((n_pairs, 1), dtype=float)
    beta0, resid0, rss0, rmse0 = weighted_lstsq(X0, y, w)
    aicc0 = robust_aicc(rss0, n_pairs, k=1)

    # Recurrence model
    X1 = np.column_stack([np.ones(n_pairs), x_prev])
    beta1, resid1, rss1, rmse1 = weighted_lstsq(X1, y, w)
    aicc1 = robust_aicc(rss1, n_pairs, k=2)

    # Recurrence + local baryonic control
    mask2 = np.isfinite(x_bar)
    if mask2.sum() >= 6:
        y2 = y[mask2]
        w2 = w[mask2]
        X2 = np.column_stack([np.ones(mask2.sum()), x_prev[mask2], x_bar[mask2]])
        beta2, resid2, rss2, rmse2 = weighted_lstsq(X2, y2, w2)
        aicc2 = robust_aicc(rss2, len(y2), k=3)

        coeff_prev_bar = float(beta2[1])
        coeff_bar = float(beta2[2])
        rmse2_val = float(rmse2)
        aicc2_val = float(aicc2)
        delta_aicc_control = float(aicc2 - aicc0)
        improves_aicc_recbar = bool(aicc2_val < aicc0)
    else:
        coeff_prev_bar = np.nan
        coeff_bar = np.nan
        rmse2_val = np.nan
        aicc2_val = np.nan
        delta_aicc_control = np.nan
        improves_aicc_recbar = False

    boot_lo, boot_hi = bootstrap_slope(
        x=x_prev,
        y=y,
        n_boot=n_boot,
        w=w,
        seed=12345 + (abs(hash(galaxy)) % 100000),
    )

    return {
        "galaxy": galaxy,
        "n_raw": int(len(df)),
        "n_pairs": int(n_pairs),
        "window": int(window),
        "null_intercept": float(beta0[0]),
        "rec_intercept": float(beta1[0]),
        "rec_slope": float(beta1[1]),
        "rec_slope_boot_ci_lo": boot_lo,
        "rec_slope_boot_ci_hi": boot_hi,
        "rmse_null": float(rmse0),
        "rmse_rec": float(rmse1),
        "delta_rmse": float(rmse1 - rmse0),
        "aicc_null": float(aicc0),
        "aicc_rec": float(aicc1),
        "delta_aicc": float(aicc1 - aicc0),
        "improves_aicc": bool(aicc1 < aicc0),
        "improves_rmse": bool(rmse1 < rmse0),
        "recbar_slope_prev": coeff_prev_bar,
        "recbar_slope_dlogvbar": coeff_bar,
        "rmse_recbar": rmse2_val,
        "aicc_recbar": aicc2_val,
        "delta_aicc_recbar_vs_null": delta_aicc_control,
        "improves_aicc_recbar": improves_aicc_recbar,
    }


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "n_galaxies": 0,
            "pct_delta_aicc_negative": np.nan,
            "pct_delta_rmse_negative": np.nan,
            "median_delta_aicc": np.nan,
            "median_delta_rmse": np.nan,
            "median_rec_slope": np.nan,
            "pct_delta_aicc_negative_recbar": np.nan,
            "median_delta_aicc_recbar": np.nan,
        }

    valid_delta_aicc = np.isfinite(df["delta_aicc"])
    valid_delta_rmse = np.isfinite(df["delta_rmse"])
    valid_rec_slope = np.isfinite(df["rec_slope"])
    valid_control = np.isfinite(df["delta_aicc_recbar_vs_null"])

    return {
        "n_galaxies": int(len(df)),
        "pct_delta_aicc_negative": (
            float(100.0 * np.mean(df.loc[valid_delta_aicc, "delta_aicc"] < 0))
            if valid_delta_aicc.any()
            else np.nan
        ),
        "pct_delta_rmse_negative": (
            float(100.0 * np.mean(df.loc[valid_delta_rmse, "delta_rmse"] < 0))
            if valid_delta_rmse.any()
            else np.nan
        ),
        "median_delta_aicc": (
            float(np.median(df.loc[valid_delta_aicc, "delta_aicc"]))
            if valid_delta_aicc.any()
            else np.nan
        ),
        "median_delta_rmse": (
            float(np.median(df.loc[valid_delta_rmse, "delta_rmse"]))
            if valid_delta_rmse.any()
            else np.nan
        ),
        "median_rec_slope": (
            float(np.median(df.loc[valid_rec_slope, "rec_slope"]))
            if valid_rec_slope.any()
            else np.nan
        ),
        "pct_delta_aicc_negative_recbar": (
            float(100.0 * np.mean(df.loc[valid_control, "delta_aicc_recbar_vs_null"] < 0))
            if valid_control.any()
            else np.nan
        ),
        "median_delta_aicc_recbar": (
            float(np.median(df.loc[valid_control, "delta_aicc_recbar_vs_null"]))
            if valid_control.any()
            else np.nan
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_rotmod_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {args.data_dir}")

    rows: List[Dict[str, float]] = []
    skipped: List[Tuple[str, str]] = []

    for path in files:
        galaxy = path.name.replace("_rotmod.dat", "")
        try:
            df = load_rotmod_file(path)
            if len(df) < args.min_points:
                skipped.append((galaxy, f"too_few_points:{len(df)}"))
                continue

            result = analyze_galaxy(
                galaxy=galaxy,
                df=df,
                window=args.window,
                n_boot=args.bootstrap,
            )
            if result is None:
                skipped.append((galaxy, "insufficient_pairs"))
                continue

            rows.append(result)
        except Exception as exc:
            skipped.append((galaxy, f"error:{exc}"))

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_aicc", ascending=True).reset_index(drop=True)
    summary = summarize(out)
    summary["n_files_found"] = int(len(files))
    summary["n_skipped"] = int(len(skipped))
    summary["window"] = int(args.window)
    summary["min_points"] = int(args.min_points)
    summary["bootstrap"] = int(args.bootstrap)

    out_csv = args.out_dir / "per_galaxy_f3_recurrence.csv"
    top_imp_csv = args.out_dir / "top20_f3_recurrence_improve.csv"
    top_wor_csv = args.out_dir / "top20_f3_recurrence_worsen.csv"
    skipped_csv = args.out_dir / "skipped_galaxies.csv"
    summary_json = args.out_dir / "executive_summary.json"

    out.to_csv(out_csv, index=False)
    out.head(20).to_csv(top_imp_csv, index=False)
    if not out.empty:
        out.sort_values("delta_aicc", ascending=False).head(20).to_csv(top_wor_csv, index=False)
    else:
        out.to_csv(top_wor_csv, index=False)
    pd.DataFrame(skipped, columns=["galaxy", "reason"]).to_csv(skipped_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
