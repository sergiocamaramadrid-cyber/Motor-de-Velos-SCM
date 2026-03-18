#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EPS = 1e-12

# ============================================================
# CONFIG
# ============================================================
INPUT = "data/sparc_175_master_sample.csv"
OUTDIR = "results/delta_f3_environment"


# ============================================================
# FUNCTIONS
# ============================================================
def compute_aicc(rss: float, n: int, k: int) -> float:
    rss = max(float(rss), EPS)
    if n <= k + 1:
        return float("inf")
    # Classical Gaussian AIC from RSS: AIC = n*ln(RSS/n) + 2k
    aic = n * np.log(rss / n) + 2 * k
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    rss = float(np.sum((y - yhat) ** 2))
    return beta, yhat, rss


def run_analysis(input_path: str = INPUT, outdir: str = OUTDIR) -> dict[str, float]:
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_path)

    rdisk_col = "Rdisk" if "Rdisk" in df.columns else "logRd" if "logRd" in df.columns else None
    if rdisk_col is None:
        raise ValueError("Missing required column: Rdisk (or alias logRd)")

    incl_col = "inclination" if "inclination" in df.columns else None
    required = ["delta_f3", "logMbar", rdisk_col, "logSigmaHI_out"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if incl_col is not None:
        required.append(incl_col)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    y = df["delta_f3"].to_numpy(dtype=float)

    controls = [df["logMbar"].to_numpy(dtype=float), df[rdisk_col].to_numpy(dtype=float)]
    if incl_col is not None:
        controls.append(df[incl_col].to_numpy(dtype=float))
    x_controls = np.column_stack(controls)
    x_env = df["logSigmaHI_out"].to_numpy(dtype=float).reshape(-1, 1)

    beta0 = float(np.mean(y))
    rss0 = float(np.sum((y - beta0) ** 2))
    aicc0 = compute_aicc(rss0, len(y), 1)

    _, _, rss_ctrl = ols_fit(x_controls, y)
    aicc_ctrl = compute_aicc(rss_ctrl, len(y), x_controls.shape[1] + 1)

    x_full = np.column_stack([x_controls, x_env])
    beta_full, _, rss_full = ols_fit(x_full, y)
    aicc_full = compute_aicc(rss_full, len(y), x_full.shape[1] + 1)

    x_train, x_test, y_train, y_test = train_test_split(x_full, y, test_size=0.3, random_state=42)
    beta_train, _, _ = ols_fit(x_train, y_train)
    x_test_aug = np.column_stack([x_test, np.ones(len(x_test))])
    y_pred = x_test_aug @ beta_train
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    x_train_ctrl, x_test_ctrl, _, _ = train_test_split(
        x_controls, y, test_size=0.3, random_state=42
    )
    beta_ctrl, _, _ = ols_fit(x_train_ctrl, y_train)
    x_test_ctrl_aug = np.column_stack([x_test_ctrl, np.ones(len(x_test_ctrl))])
    y_pred_ctrl = x_test_ctrl_aug @ beta_ctrl
    rmse_ctrl = float(np.sqrt(np.mean((y_test - y_pred_ctrl) ** 2)))

    delta_rmse = float(rmse - rmse_ctrl)
    coef_hi = float(beta_full[x_controls.shape[1]])

    print("\n=== MODELS (AICc) ===")
    print(f"Null: {aicc0:.3f}")
    print(f"Controles: {aicc_ctrl:.3f}")
    print(f"Controles+HI: {aicc_full:.3f}")
    print(f"ΔAICc (HI vs controles): {aicc_full - aicc_ctrl:.3f}")

    print("\n=== OOS ===")
    print(f"RMSE controles: {rmse_ctrl:.5f}")
    print(f"RMSE + HI: {rmse:.5f}")
    print(f"ΔRMSE: {delta_rmse:.5f}")

    print("\n=== HI COEFFICIENT ===")
    print(f"logSigmaHI_out coef: {coef_hi:.6f}")

    results = {
        "AICc_nulo": float(aicc0),
        "AICc_controles": float(aicc_ctrl),
        "AICc_full": float(aicc_full),
        "Delta_AICc": float(aicc_full - aicc_ctrl),
        "RMSE_ctrl": float(rmse_ctrl),
        "RMSE_full": float(rmse),
        "Delta_RMSE": float(delta_rmse),
        "coef_HI": float(coef_hi),
    }

    with open(os.path.join(outdir, "results.txt"), "w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"\nResults saved to {outdir}/results.txt")
    return results


def main() -> None:
    run_analysis(INPUT, OUTDIR)


if __name__ == "__main__":
    main()
