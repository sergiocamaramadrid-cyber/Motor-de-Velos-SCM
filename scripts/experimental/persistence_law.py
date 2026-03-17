#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

A0 = 1.2e-10


def recurrence(r: float, lam: float, beta: float) -> float:
    return (1 - lam) * r + beta * r * (1 - r)


def simulate_sequence(r0: float, lam: float, beta: float, n: int) -> np.ndarray:
    r = [r0]
    for _ in range(n - 1):
        r.append(recurrence(r[-1], lam, beta))
    return np.array(r, dtype=float)


def loss(params: np.ndarray, r_obs: np.ndarray) -> float:
    lam, beta = params
    if lam < 0 or beta < 0:
        return 1e9
    r_pred = simulate_sequence(r_obs[0], lam, beta, len(r_obs))
    return float(np.mean((r_pred - r_obs) ** 2))


def fit_parameters(r_obs: np.ndarray) -> tuple[float, float, float]:
    res = minimize(loss, x0=[0.1, 0.5], args=(r_obs,), method="Nelder-Mead")
    lam, beta = res.x
    return float(lam), float(beta), float(res.fun)


def compute_aicc(n: int, rss: float, k: int) -> float:
    if n <= k + 1:
        return float("inf")
    rss = max(float(rss), 1e-12)
    return float(n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1))


def build_bins(df: pd.DataFrame, n_bins: int = 5) -> np.ndarray:
    df = df.copy()
    if "Mbar" in df.columns:
        mbar = df["Mbar"]
    elif "logMbar" in df.columns:
        mbar = 10.0 ** df["logMbar"]
    elif "r_kpc" in df.columns:
        # Fallback to scale bins when mass proxies are unavailable.
        mbar = df["r_kpc"]
    else:
        raise ValueError("Input must contain either 'Mbar', 'logMbar', or 'r_kpc'.")

    df["logM"] = np.log10(mbar)
    df["bin"] = pd.qcut(df["logM"], n_bins, labels=False, duplicates="drop")

    grouped = df.groupby("bin", observed=True).agg({"g_obs": "mean", "g_bar": "mean"})
    return (grouped["g_obs"] / grouped["g_bar"]).to_numpy(dtype=float)


def main(input_csv: str, n_bins: int = 5) -> None:
    df = pd.read_csv(input_csv)
    missing = {"g_obs", "g_bar"} - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Input file is missing required columns: {missing_list}")
    df = df[df["g_bar"] < 0.3 * A0]

    r_obs = build_bins(df, n_bins=n_bins)
    lam, beta, mse = fit_parameters(r_obs)

    r_pred = simulate_sequence(r_obs[0], lam, beta, len(r_obs))

    rss = float(np.sum((r_pred - r_obs) ** 2))
    aicc_model = compute_aicc(len(r_obs), rss, k=2)

    r_null = np.full_like(r_obs, r_obs.mean())
    rss_null = float(np.sum((r_null - r_obs) ** 2))
    aicc_null = compute_aicc(len(r_obs), rss_null, k=1)

    print("\n=== Persistence Law Fit ===")
    print(f"lambda = {lam:.4f}")
    print(f"beta   = {beta:.4f}")
    print(f"sigma  = {lam / beta:.4f}")
    print(f"MSE    = {mse:.6f}")
    print(f"AICc(model) = {aicc_model:.2f}")
    print(f"AICc(null)  = {aicc_null:.2f}")

    if aicc_model < aicc_null:
        print("✔ Modelo mejor que nulo")
    else:
        print("✘ No supera modelo nulo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    main(args.input, args.bins)
