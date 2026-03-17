#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.experimental.persistence_law import build_bins, fit_parameters

# Numerical floor to avoid unstable sigma = lambda / beta when beta ~ 0.
EPS = 1e-10


def bootstrap(df: pd.DataFrame, n_boot: int = 1000, n_bins: int = 5) -> np.ndarray:
    results = []

    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True)
        r_obs = build_bins(sample, n_bins=n_bins)

        lam, beta, _ = fit_parameters(r_obs)
        sigma = lam / beta if abs(beta) > EPS else np.nan

        results.append((lam, beta, sigma))

    return np.array(results, dtype=float)


def summarize(arr: np.ndarray) -> np.ndarray:
    percentiles = [2.5, 16, 50, 84, 97.5]
    return np.nanpercentile(arr, percentiles)


def main(input_csv: str, n_boot: int = 1000, n_bins: int = 5) -> None:
    df = pd.read_csv(input_csv)

    res = bootstrap(df, n_boot=n_boot, n_bins=n_bins)

    lam_stats = summarize(res[:, 0])
    beta_stats = summarize(res[:, 1])
    sigma_stats = summarize(res[:, 2])

    print("\n=== Bootstrap Results ===")
    print("Percentiles: 2.5, 16, 50, 84, 97.5")
    print(f"lambda: {lam_stats}")
    print(f"beta:   {beta_stats}")
    print(f"sigma:  {sigma_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    main(args.input, n_boot=args.n_boot, n_bins=args.bins)
