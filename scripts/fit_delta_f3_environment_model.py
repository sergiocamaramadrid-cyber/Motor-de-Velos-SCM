#!/usr/bin/env python3
"""
fit_delta_f3_environment_model.py

Multivariable linear model for environmental signal in delta_f3:

    delta_f3 ~ logSigmaHI_out + logMbar + logRd + inclination
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["delta_f3", "logSigmaHI_out", "logMbar", "logRd", "inclination"]
PREDICTORS = ["logSigmaHI_out", "logMbar", "logRd", "inclination"]


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    n_initial = int(len(df))
    n_used = int(len(clean))
    n_removed = int(n_initial - n_used)
    return clean, n_initial, n_used, n_removed


def _design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[PREDICTORS].to_numpy(dtype=float)
    y = df["delta_f3"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(df), dtype=float), x])
    return x, y


def _ols_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    resid = y - yhat
    n, p = x.shape
    dof = max(n - p, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.diag(sigma2 * xtx_inv))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return beta, stderr, r2


def _bootstrap_betas(df: pd.DataFrame, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    betas: list[np.ndarray] = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        x, y = _design_matrix(sample)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        betas.append(beta)
    return np.asarray(betas, dtype=float)


def run_model(
    df: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    x, y = _design_matrix(df)
    beta, stderr, r2 = _ols_stats(x, y)
    boots = _bootstrap_betas(df, n_bootstrap=n_bootstrap, seed=seed)
    ci_low = np.percentile(boots, 2.5, axis=0)
    ci_high = np.percentile(boots, 97.5, axis=0)

    names = ["intercept", *PREDICTORS]
    coef_df = pd.DataFrame(
        {
            "variable": names,
            "coefficient": beta,
            "std_error": stderr,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        }
    )
    summary: dict[str, float | int | str] = {
        "model": "delta_f3 ~ logSigmaHI_out + logMbar + logRd + inclination",
        "R2": float(r2),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
    return coef_df, summary


def save_outputs(
    outdir: Path,
    coef_df: pd.DataFrame,
    summary: dict[str, float | int | str],
    n_initial: int,
    n_used: int,
    n_removed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(outdir / "delta_f3_environment_coefficients.csv", index=False)
    summary = {
        **summary,
        "n_initial": int(n_initial),
        "n_used": int(n_used),
        "n_removed_nan": int(n_removed),
    }
    with (outdir / "delta_f3_environment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sparc_175_master.csv")
    parser.add_argument("--out", default="results/delta_f3_environment_model")
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    check_columns(df)
    clean, n_initial, n_used, n_removed = prepare_data(df)
    coef_df, summary = run_model(clean, n_bootstrap=args.bootstrap_n, seed=args.seed)
    save_outputs(Path(args.out), coef_df, summary, n_initial, n_used, n_removed)


if __name__ == "__main__":
    main()
