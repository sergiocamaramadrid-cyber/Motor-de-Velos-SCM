#!/usr/bin/env python3
"""
generate_env_figure.py

Generate the environment-correlation figure for the SCM framework.

Primary use:
- Read real F3 results from results/f3_catalog_real.csv
- Read SPARC basic properties from data/SPARC/sparc_basic.csv
- Build the HI-based environmental proxy:
      logSigmaHI = log10( MHI / (pi * Rdisk^2) )
- Merge by galaxy
- Compute Spearman rho
- Save:
    results/figure_env_correlation.pdf
    results/figure_env_correlation.png

This script is intentionally lightweight and does not require scipy.
Spearman rho is computed via rank correlation using numpy/pandas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


F3_CSV = Path("results/f3_catalog_real.csv")
SPARC_BASIC_CSV = Path("data/SPARC/sparc_basic.csv")
OUT_PDF = Path("results/figure_env_correlation.pdf")
OUT_PNG = Path("results/figure_env_correlation.png")


def compute_spearman_rho(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average").to_numpy(dtype=float)
    yr = y.rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> int:
    if not F3_CSV.exists():
        print(f"ERROR: missing {F3_CSV}")
        return 1
    if not SPARC_BASIC_CSV.exists():
        print(f"ERROR: missing {SPARC_BASIC_CSV}")
        return 1

    f3 = pd.read_csv(F3_CSV)
    sparc = pd.read_csv(SPARC_BASIC_CSV)

    required_f3 = {"galaxy", "F3"}
    if not required_f3.issubset(f3.columns):
        print(f"ERROR: {F3_CSV} must contain columns {sorted(required_f3)}")
        print(f"Found: {list(f3.columns)}")
        return 1

    required_sparc = {"galaxy", "MHI", "Rdisk"}
    if not required_sparc.issubset(sparc.columns):
        print(f"ERROR: {SPARC_BASIC_CSV} must contain columns {sorted(required_sparc)}")
        print(f"Found: {list(sparc.columns)}")
        return 1

    sparc = sparc.copy()
    sparc.loc[sparc["Rdisk"] <= 0, "Rdisk"] = np.nan
    sparc["logSigmaHI"] = np.log10(sparc["MHI"] / (np.pi * sparc["Rdisk"] ** 2))
    sparc = sparc.replace([np.inf, -np.inf], np.nan)

    merged = f3.merge(
        sparc[["galaxy", "logSigmaHI"]],
        on="galaxy",
        how="inner",
    ).dropna(subset=["F3", "logSigmaHI"])

    if len(merged) < 3:
        print("ERROR: not enough valid merged rows to make figure")
        return 1

    x = merged["logSigmaHI"].to_numpy(dtype=float)
    y = merged["F3"].to_numpy(dtype=float)

    rho = compute_spearman_rho(merged["logSigmaHI"], merged["F3"])
    slope, intercept = np.polyfit(x, y, 1)

    xs = np.linspace(x.min(), x.max(), 200)
    ys = slope * xs + intercept

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(x, y, alpha=0.85)
    plt.plot(xs, ys, linewidth=2)

    plt.xlabel(r"log $\Sigma_{\rm HI}$")
    plt.ylabel("F3")
    plt.title("Environmental modulation of outer disk dynamics")
    plt.text(
        0.03,
        0.97,
        f"N = {len(merged)}\nSpearman $\\rho$ = {rho:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    print(f"Figure saved: {OUT_PDF}")
    print(f"Figure saved: {OUT_PNG}")
    print(f"N = {len(merged)}")
    print(f"Spearman rho = {rho:.6f}")
    print(f"Linear slope = {slope:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
