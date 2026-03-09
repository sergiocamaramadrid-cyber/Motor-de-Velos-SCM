#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
INPUT_CSV = RESULTS_DIR / "per_galaxy_beta.csv"
OUTPUT_FIG = RESULTS_DIR / "fig_beta_environment.png"


def main() -> int:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No existe {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {"logSigmaHI_out", "beta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df[["logSigmaHI_out", "beta"]].dropna()
    if len(df) < 3:
        raise ValueError("No hay suficientes puntos válidos para la figura (mínimo 3).")

    x = df["logSigmaHI_out"]
    y = df["beta"]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.75)

    xline = pd.Series(sorted(x))
    yline = intercept + slope * xline
    plt.plot(xline, yline, label=f"slope={slope:.3f}, p={p_value:.3g}")

    plt.xlabel("log Σ_HI,out (M☉ pc⁻²)")
    plt.ylabel("β (deep-regime slope)")
    plt.title("Environmental Test: β vs HI Surface Density")
    plt.grid(True, alpha=0.3)
    plt.legend()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.close()

    print(f"Figure saved to {OUTPUT_FIG}")
    print(f"slope = {slope:.6f}")
    print(f"intercept = {intercept:.6f}")
    print(f"r = {r_value:.6f}")
    print(f"p-value = {p_value:.6g}")
    print(f"stderr = {std_err:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
