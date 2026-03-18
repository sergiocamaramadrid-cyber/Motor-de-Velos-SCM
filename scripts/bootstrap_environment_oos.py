#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bootstrap_environment_oos.py

Bootstrap OOS para evaluar si el entorno (logSigmaHI_out)
mejora la predicción de ΔF3 más allá de la masa bariónica.

Salida:
- resultados_bootstrap.txt
- bootstrap_HI.png
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV = "data/sparc_175_master.csv"
OUTDIR = "results/bootstrap_environment"
N_BOOT = 500
TEST_SIZE = 0.3
SEED = 42


# ============================================================
# CORE
# ============================================================
def _resolve_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No se encontró columna para {label}. Opciones: {list(candidates)}")


def run_bootstrap(
    input_csv: str = INPUT_CSV,
    outdir: str = OUTDIR,
    n_boot: int = N_BOOT,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> dict[str, float]:
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # columnas esperadas (robusto a aliases)
    mass_col = _resolve_column(df, ["logMbar", "Mbar", "logM"], "masa")
    hi_col = _resolve_column(df, ["logSigmaHI_out", "SigmaHI_out"], "entorno HI")

    # target
    if "delta_f3" not in df.columns:
        if "F3" not in df.columns:
            raise ValueError("Falta target: requiere 'delta_f3' o, alternativamente, 'F3' para derivarlo")
        df = df.sort_values(mass_col).reset_index(drop=True)
        df["delta_f3"] = df["F3"].diff().fillna(0)

    df = df.dropna(subset=["delta_f3", mass_col, hi_col]).reset_index(drop=True)

    x_base = df[[mass_col]].to_numpy(dtype=float)
    x_env = df[[mass_col, hi_col]].to_numpy(dtype=float)
    y = df["delta_f3"].to_numpy(dtype=float)

    # ============================================================
    # BOOTSTRAP
    # ============================================================
    rmse_base: list[float] = []
    rmse_env: list[float] = []
    coef_hi: list[float] = []

    np.random.seed(seed)

    for i in range(n_boot):
        xb_tr, xb_te, y_tr, y_te = train_test_split(x_base, y, test_size=test_size, random_state=i)
        xe_tr, xe_te, _, _ = train_test_split(x_env, y, test_size=test_size, random_state=i)

        # base
        m_base = LinearRegression().fit(xb_tr, y_tr)
        yb = m_base.predict(xb_te)
        rmse_b = float(np.sqrt(mean_squared_error(y_te, yb)))

        # env
        m_env = LinearRegression().fit(xe_tr, y_tr)
        ye = m_env.predict(xe_te)
        rmse_e = float(np.sqrt(mean_squared_error(y_te, ye)))

        rmse_base.append(rmse_b)
        rmse_env.append(rmse_e)
        coef_hi.append(float(m_env.coef_[-1]))

    rmse_base_arr = np.array(rmse_base, dtype=float)
    rmse_env_arr = np.array(rmse_env, dtype=float)
    delta = rmse_env_arr - rmse_base_arr
    coef_hi_arr = np.array(coef_hi, dtype=float)

    # ============================================================
    # STATS
    # ============================================================
    mean_delta = float(delta.mean())
    std_delta = float(delta.std())

    prop_improve = float(np.mean(delta < 0))
    p_empirical = float(np.mean(delta >= 0))

    mean_coef = float(coef_hi_arr.mean())
    std_coef = float(coef_hi_arr.std())
    prop_coef_pos = float(np.mean(coef_hi_arr > 0))

    # ============================================================
    # PRINT
    # ============================================================
    print("\n=== BOOTSTRAP OOS ===")
    print(f"ΔRMSE mean = {mean_delta:.6f} ± {std_delta:.6f}")
    print(f"% mejora (ΔRMSE<0) = {prop_improve * 100:.1f}%")
    print(f"p_empirical = {p_empirical:.4f}")
    print(f"coef_HI = {mean_coef:.6f} ± {std_coef:.6f}")
    print(f"% coef_HI > 0 = {prop_coef_pos * 100:.1f}%")

    # ============================================================
    # PLOT
    # ============================================================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(delta, bins=30)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Distribución ΔRMSE")

    plt.subplot(1, 2, 2)
    plt.hist(coef_hi_arr, bins=30)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Distribución coef_HI")

    plt.tight_layout()
    plt.savefig(f"{outdir}/bootstrap_HI.png", dpi=150)
    plt.close()

    # ============================================================
    # SAVE
    # ============================================================
    with open(f"{outdir}/resultados_bootstrap.txt", "w", encoding="utf-8") as f:
        f.write("BOOTSTRAP OOS RESULTS\n")
        f.write("=====================\n")
        f.write(f"N_boot = {n_boot}\n\n")
        f.write(f"ΔRMSE mean = {mean_delta:.6f} ± {std_delta:.6f}\n")
        f.write(f"% mejora = {prop_improve * 100:.2f}\n")
        f.write(f"p_empirical = {p_empirical:.6f}\n\n")
        f.write(f"coef_HI = {mean_coef:.6f} ± {std_coef:.6f}\n")
        f.write(f"% coef_HI > 0 = {prop_coef_pos * 100:.2f}\n")

    return {
        "delta_rmse_mean": mean_delta,
        "delta_rmse_std": std_delta,
        "prop_improve": prop_improve,
        "p_empirical": p_empirical,
        "coef_hi_mean": mean_coef,
        "coef_hi_std": std_coef,
        "prop_coef_hi_pos": prop_coef_pos,
    }


def main() -> None:
    run_bootstrap()


if __name__ == "__main__":
    main()
