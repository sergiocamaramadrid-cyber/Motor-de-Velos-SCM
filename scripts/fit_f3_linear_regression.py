#!/usr/bin/env python3
"""
fit_f3_linear_regression.py

Linear regression of F3 against environmental and structural predictors.

Model:
    F3 ~ logSigmaHI_out + logMbar + logRd

Features
--------
• No sklearn dependency (uses numpy.linalg.lstsq)
• Explicit NaN filtering
• Reproducible output (CSV + JSON)
• Reports number of galaxies used
• Reports R²
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "F3",
    "logSigmaHI_out",
    "logMbar",
    "logRd",
]
MIN_REQUIRED_SAMPLES = 10


def check_columns(df: pd.DataFrame) -> None:
    """Validate that all required regression columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Drop rows with NaNs in required columns and return row-count stats."""
    df_clean = df.dropna(subset=REQUIRED_COLUMNS)

    n_initial = len(df)
    n_used = len(df_clean)
    n_removed = n_initial - n_used

    return df_clean, n_initial, n_used, n_removed


def run_regression(df: pd.DataFrame) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Run OLS by least squares and return intercept, coefficients, R² and ŷ."""
    y = df["F3"].values

    x = df[["logSigmaHI_out", "logMbar", "logRd"]].values

    # add intercept
    x = np.column_stack([np.ones(len(x)), x])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)

    intercept = float(beta[0])
    coefs = beta[1:]

    y_pred = x @ beta

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return intercept, coefs, r2, y_pred


def save_results(
    outdir: Path,
    intercept: float,
    coefs: np.ndarray,
    r2: float,
    n_initial: int,
    n_used: int,
    n_removed: int,
) -> dict[str, float | int | str]:
    outdir.mkdir(parents=True, exist_ok=True)

    coef_table = pd.DataFrame(
        {
            "variable": [
                "logSigmaHI_out",
                "logMbar",
                "logRd",
            ],
            "coefficient": coefs,
        }
    )

    coef_table.to_csv(outdir / "f3_regression_coefficients.csv", index=False)

    summary = {
        "status": "ok",
        "model": "F3 ~ logSigmaHI_out + logMbar + logRd",
        "intercept": float(intercept),
        "R2": float(r2),
        "n_initial": int(n_initial),
        "n_used": int(n_used),
        "n_removed_nan": int(n_removed),
    }

    with open(outdir / "f3_regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nRegression summary")
    print("------------------")
    print(f"Intercept: {intercept:.6f}")
    print(f"R2: {r2:.4f}")
    print(f"Galaxies used: {n_used}/{n_initial}")
    print(f"Rows removed (NaN): {n_removed}")

    print("\nCoefficients:")
    for name, val in zip(["logSigmaHI_out", "logMbar", "logRd"], coefs):
        print(f"{name}: {val:.6f}")
    return summary


def save_insufficient_sample(
    outdir: Path, n_initial: int, n_used: int, n_removed: int
) -> dict[str, float | int | str]:
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variable": [], "coefficient": []}).to_csv(
        outdir / "f3_regression_coefficients.csv", index=False
    )
    summary = {
        "status": "insufficient_sample",
        "n_samples": int(n_used),
        "required": int(MIN_REQUIRED_SAMPLES),
        "n_initial": int(n_initial),
        "n_used": int(n_used),
        "n_removed_nan": int(n_removed),
    }
    with open(outdir / "f3_regression_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Insufficient sample for regression: n_samples={n_used}, "
        f"required={MIN_REQUIRED_SAMPLES}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sparc_175_master.csv")
    parser.add_argument(
        "--out",
        default="results/f3_regression",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional CSV summary output path kept for CI backward compatibility.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    check_columns(df)

    df, n_initial, n_used, n_removed = prepare_data(df)
    if n_used < MIN_REQUIRED_SAMPLES:
        summary = save_insufficient_sample(
            Path(args.out), n_initial=n_initial, n_used=n_used, n_removed=n_removed
        )
        if args.summary_out:
            summary_out = Path(args.summary_out)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([summary]).to_csv(summary_out, index=False)
        return

    intercept, coefs, r2, _ = run_regression(df)

    summary = save_results(
        Path(args.out),
        intercept,
        coefs,
        r2,
        n_initial,
        n_used,
        n_removed,
    )

    if args.summary_out:
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(summary_out, index=False)


if __name__ == "__main__":
    main()
