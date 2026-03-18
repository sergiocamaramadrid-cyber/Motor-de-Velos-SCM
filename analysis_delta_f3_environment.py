#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis_delta_f3_environment.py

Evaluate whether environment information (logSigmaHI_out) improves ΔF3
prediction relative to a baryonic-mass baseline model.

Includes:
- OLS fit (baseline vs environment)
- AICc (when valid)
- Out-of-sample validation (train/test)
- Reproducible text output for paper workflows
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ============================================================
# UTILIDADES
# ============================================================


def compute_aicc(rss: float, n: int, k: int) -> float:
    """Compute AICc safely."""
    if n <= k + 1:
        return float("inf")
    safe_rss = max(float(rss), np.finfo(float).tiny)
    aic = n * np.log(safe_rss / n) + 2 * k
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[LinearRegression, float, np.ndarray]:
    """Fit OLS and return model, RSS and predictions."""
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    rss = float(np.sum((y - y_pred) ** 2))
    return model, rss, y_pred


def resolve_column(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    """Resolve a valid column from aliases."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No valid column found for {name}: {candidates}")


# ============================================================
# MAIN
# ============================================================


def run_analysis(
    input_path: str,
    outdir: str = "results/delta_f3_environment",
    delta_col: str = "delta_f3",
    test_size: float = 0.3,
    seed: int = 42,
) -> dict[str, float]:
    os.makedirs(outdir, exist_ok=True)

    # ---------------- LOAD ----------------
    df = pd.read_csv(input_path)

    mass_col = resolve_column(df, ["logMbar", "Mbar", "logM"], "mass")
    hi_col = resolve_column(df, ["logSigmaHI_out", "SigmaHI_out"], "HI")

    # target
    if delta_col not in df.columns:
        if "F3" not in df.columns:
            raise ValueError(
                f"Target column '{delta_col}' not found and 'F3' column not available."
            )
        df = df.sort_values(mass_col).reset_index(drop=True)
        df[delta_col] = df["F3"].diff().fillna(0)

    df = df.dropna(subset=[delta_col, mass_col, hi_col]).reset_index(drop=True)

    y = df[delta_col].to_numpy(dtype=float)
    x_base = df[[mass_col]].to_numpy(dtype=float)
    x_env = df[[mass_col, hi_col]].to_numpy(dtype=float)

    n = len(y)

    # ---------------- FIT IN-SAMPLE ----------------
    model_base, rss_base, _ = ols_fit(x_base, y)
    model_env, rss_env, _ = ols_fit(x_env, y)

    k_base = x_base.shape[1] + 1
    k_env = x_env.shape[1] + 1

    aicc_base = compute_aicc(rss_base, n, k_base)
    aicc_env = compute_aicc(rss_env, n, k_env)

    delta_aicc = float(aicc_env - aicc_base)

    # ---------------- OOS ----------------
    xb_tr, xb_te, y_tr, y_te = train_test_split(
        x_base, y, test_size=test_size, random_state=seed
    )

    xe_tr, xe_te, _, _ = train_test_split(
        x_env, y, test_size=test_size, random_state=seed
    )

    model_base_oos = LinearRegression().fit(xb_tr, y_tr)
    model_env_oos = LinearRegression().fit(xe_tr, y_tr)

    y_pred_base = model_base_oos.predict(xb_te)
    y_pred_env = model_env_oos.predict(xe_te)

    rmse_base = float(np.sqrt(mean_squared_error(y_te, y_pred_base)))
    rmse_env = float(np.sqrt(mean_squared_error(y_te, y_pred_env)))

    delta_rmse = float(rmse_env - rmse_base)

    coef_hi = float(model_env_oos.coef_[-1])

    # ---------------- PRINT ----------------
    print("\n--- MODELOS (IN-SAMPLE) ---")
    print(f"AICc base       : {aicc_base}")
    print(f"AICc entorno    : {aicc_env}")
    print(f"ΔAICc (env-base): {delta_aicc}")

    if np.isinf(aicc_base) or np.isinf(aicc_env):
        print("⚠️ AICc not valid (n <= k+1)")

    print("\n--- VALIDACIÓN OOS ---")
    print(f"RMSE base   : {rmse_base:.6f}")
    print(f"RMSE entorno: {rmse_env:.6f}")
    print(f"ΔRMSE       : {delta_rmse:.6f}")

    print("\n--- COEFICIENTE ENTORNO ---")
    print(f"coef_HI = {coef_hi:.6f}")

    # ---------------- SAVE ----------------
    out_file = os.path.join(outdir, "results.txt")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("ENVIRONMENT ANALYSIS (ΔF3)\n")
        f.write("==========================\n\n")

        f.write(f"N = {n}\n\n")

        f.write("AICc:\n")
        f.write(f"base = {aicc_base}\n")
        f.write(f"env  = {aicc_env}\n")
        f.write(f"ΔAICc = {delta_aicc}\n\n")

        f.write("OOS:\n")
        f.write(f"RMSE_base = {rmse_base}\n")
        f.write(f"RMSE_env  = {rmse_env}\n")
        f.write(f"ΔRMSE     = {delta_rmse}\n\n")

        f.write("coef_HI:\n")
        f.write(f"{coef_hi}\n")

    print(f"\nResults saved to: {out_file}")

    return {
        "aicc_base": float(aicc_base),
        "aicc_env": float(aicc_env),
        "delta_aicc": float(delta_aicc),
        "rmse_base": float(rmse_base),
        "rmse_env": float(rmse_env),
        "delta_rmse": float(delta_rmse),
        "coef_hi": float(coef_hi),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="results/delta_f3_environment")
    parser.add_argument("--delta-col", default="delta_f3")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(
        input_path=args.input,
        outdir=args.outdir,
        delta_col=args.delta_col,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
