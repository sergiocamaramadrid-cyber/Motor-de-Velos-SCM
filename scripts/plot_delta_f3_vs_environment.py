#!/usr/bin/env python3
"""
plot_delta_f3_vs_environment.py

Generate publication-style figure:
  delta_f3 vs logSigmaHI_out with fitted line and bootstrap band.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["delta_f3", "logSigmaHI_out"]


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return float(beta[0]), float(beta[1])


def bootstrap_band(x: np.ndarray, y: np.ndarray, n_bootstrap: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    preds = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        inter, slope = fit_line(x[idx], y[idx])
        preds.append(inter + slope * x_grid)
    pred_arr = np.asarray(preds, dtype=float)
    low = np.percentile(pred_arr, 2.5, axis=0)
    high = np.percentile(pred_arr, 97.5, axis=0)
    return x_grid, low, high


def make_plot(df: pd.DataFrame, out_pdf: Path, n_bootstrap: int = 500, seed: int = 42) -> None:
    clean = df.dropna(subset=REQUIRED_COLUMNS)
    x = clean["logSigmaHI_out"].to_numpy(dtype=float)
    y = clean["delta_f3"].to_numpy(dtype=float)
    inter, slope = fit_line(x, y)
    x_grid, y_low, y_high = bootstrap_band(x, y, n_bootstrap=n_bootstrap, seed=seed)
    y_fit = inter + slope * x_grid

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.8))
    plt.scatter(x, y, alpha=0.75, label="Galaxies")
    plt.plot(x_grid, y_fit, color="tab:red", linewidth=1.8, label="Linear fit")
    plt.fill_between(x_grid, y_low, y_high, color="tab:red", alpha=0.2, label="Bootstrap 95% band")
    plt.xlabel("logSigmaHI_out")
    plt.ylabel("delta_f3")
    plt.title("delta_f3 vs environmental gas surface density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sparc_175_master.csv")
    parser.add_argument("--out", default="results/figures/delta_f3_vs_environment.pdf")
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    check_columns(df)
    make_plot(df, Path(args.out), n_bootstrap=args.bootstrap_n, seed=args.seed)


if __name__ == "__main__":
    main()
