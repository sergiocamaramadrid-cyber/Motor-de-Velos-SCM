#!/usr/bin/env python3
"""
generate_env_figure.py — F3 vs environment correlation plot (SCM framework)

- Detects F3 or delta_f3 automatically
- Detects delta_mass or delta_mass_yang
- Computes Spearman correlation WITHOUT scipy
- Uses explicit input CSV candidates
- Writes fallback file if no valid input is found
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
INPUT_CSV_CANDIDATES = [
    Path("results/environment_interaction.csv"),
    Path("results/delta_mass_yang_sparc.csv"),
    Path("results/delta_f3/sparc_delta_f3_catalog.csv"),
]

OUTPUT_DIR = Path("docs/paper1/figures")
OUTPUT_PDF = OUTPUT_DIR / "figure_env_correlation.pdf"
MISSING_FILE = OUTPUT_DIR / "MISSING_SOURCE.txt"
# =================================================


def find_input_csv():
    for path in INPUT_CSV_CANDIDATES:
        if path.exists():
            return path
    return None


def detect_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def compute_spearman(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    return np.corrcoef(xr, yr)[0, 1]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = find_input_csv()
    if csv_path is None:
        with open(MISSING_FILE, "w") as f:
            f.write("No valid input CSV found.\n")
            f.write("Expected one of:\n")
            for p in INPUT_CSV_CANDIDATES:
                f.write(f" - {p}\n")
        print("❌ No input CSV found. See MISSING_SOURCE.txt")
        sys.exit(1)

    print(f"Using input: {csv_path}")
    df = pd.read_csv(csv_path)

    # Detect columns
    f3_col = detect_column(df.columns, ["F3", "delta_f3", "f3"])
    env_col = detect_column(df.columns, ["delta_mass", "delta_mass_yang"])

    if f3_col is None or env_col is None:
        print("ERROR: Required columns not found.")
        print("Available columns:", list(df.columns))
        sys.exit(1)

    print(f"Detected columns → F3: {f3_col}, ENV: {env_col}")

    # Clean data
    data = df[[f3_col, env_col]].dropna()
    if len(data) < 5:
        print("ERROR: Not enough data points.")
        sys.exit(1)

    x = data[env_col].values
    y = data[f3_col].values

    rho = compute_spearman(x, y)
    print(f"Spearman rho = {rho:.3f} (N={len(x)})")

    # Linear fit (visual guide only)
    coeffs = np.polyfit(x, y, 1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = np.polyval(coeffs, x_fit)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.7)
    plt.plot(x_fit, y_fit)

    plt.xlabel(env_col)
    plt.ylabel(f3_col)
    plt.title(f"Environment vs F3 (rho={rho:.2f}, N={len(x)})")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    plt.close()

    print(f"✅ Figure saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
