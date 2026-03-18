#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intra-galaxy analysis: radial ΔF3 versus local baryonic structure.

For each galaxy, fit:
    ΔF3(r) ~ a * dlogSB_dlogr + b * log10(gbar_mid) + c

Where:
- ΔF3(r) is the forward radial difference of F3
- dlogSB_dlogr is the local logarithmic slope of the surface-brightness profile
- log10(gbar_mid) is the local baryonic acceleration proxy at ring midpoints

This is an intra-galaxy, radial-structure test. It is not an inter-galaxy
recurrence model.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

EPS = 1e-30
MIN_RINGS = 6
MIN_PAIRS = 4


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_df(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required radial columns: {missing}")

    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required_cols)

    # Physical guards
    out = out[out["r"] > 0].copy()
    out = out[out["SB"] > 0].copy()
    out = out[out["gbar"] > 0].copy()

    return out


def compute_f3(gobs: np.ndarray, gbar: np.ndarray) -> np.ndarray:
    gbar_safe = np.maximum(gbar, EPS)
    return (gobs - gbar_safe) / gbar_safe


def analyze_single_galaxy(group: pd.DataFrame) -> dict | None:
    group = group.sort_values("r").reset_index(drop=True)

    if len(group) < MIN_RINGS:
        return None

    r = group["r"].to_numpy(dtype=float)
    gbar = np.maximum(group["gbar"].to_numpy(dtype=float), EPS)
    gobs = group["gobs"].to_numpy(dtype=float)
    sb = np.maximum(group["SB"].to_numpy(dtype=float), EPS)

    # Observable
    f3 = compute_f3(gobs, gbar)

    # Forward radial difference
    delta_f3 = f3[1:] - f3[:-1]

    # Local logarithmic slope: d log SB / d log r
    dlogsb = np.diff(np.log(sb))
    dlogr = np.diff(np.log(r))
    valid_grad = np.abs(dlogr) > EPS
    if not np.any(valid_grad):
        return None
    grad_logsb = dlogsb[valid_grad] / dlogr[valid_grad]

    # Midpoint local potential proxy
    loggbar_mid = np.log10(0.5 * (gbar[1:] + gbar[:-1]))[valid_grad]
    delta_f3 = delta_f3[valid_grad]

    n = min(len(delta_f3), len(grad_logsb), len(loggbar_mid))
    if n < MIN_PAIRS:
        return None

    y = delta_f3[:n]
    x1 = grad_logsb[:n]
    x2 = loggbar_mid[:n]

    finite_mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    y = y[finite_mask]
    x1 = x1[finite_mask]
    x2 = x2[finite_mask]
    if len(y) < MIN_PAIRS:
        return None

    # Full model: y = a*x1 + b*x2 + c
    X = np.column_stack([x1, x2, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_grad, b_gbar, c = coef
    y_pred = X @ coef
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # Null model: constant only
    y_null = np.full_like(y, np.mean(y))
    rmse_null = float(np.sqrt(mean_squared_error(y, y_null)))

    return {
        "a_grad": float(a_grad),
        "b_gbar": float(b_gbar),
        "c": float(c),
        "rmse": rmse,
        "rmse_null": rmse_null,
        "delta_rmse": float(rmse - rmse_null),
        "n_rings": int(len(group)),
        "n_pairs": int(len(y)),
    }


def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    required = ["galaxy", "r", "gbar", "gobs", "SB"]
    df = clean_df(df, required)
    df = df.sort_values(["galaxy", "r"]).reset_index(drop=True)

    rows = []
    for galaxy, group in df.groupby("galaxy", sort=False):
        result = analyze_single_galaxy(group)
        if result is None:
            continue
        result["galaxy"] = galaxy
        rows.append(result)

    if not rows:
        raise ValueError("No valid galaxies were available for intra-galaxy fitting.")

    return pd.DataFrame(rows)


def save_outputs(res_df: pd.DataFrame, outdir: str) -> None:
    ensure_dir(outdir)

    csv_path = os.path.join(outdir, "intra_galaxy_fits.csv")
    txt_path = os.path.join(outdir, "summary.txt")
    fig_path = os.path.join(outdir, "coef_hist.png")

    res_df.to_csv(csv_path, index=False)

    n_gal = len(res_df)
    mean_a = res_df["a_grad"].mean()
    std_a = res_df["a_grad"].std()
    mean_b = res_df["b_gbar"].mean()
    std_b = res_df["b_gbar"].std()
    mean_delta = res_df["delta_rmse"].mean()
    frac_improve = (res_df["delta_rmse"] < 0).mean() * 100.0

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("INTRA-GALAXY GRADIENT ANALYSIS\n")
        f.write("=============================\n\n")
        f.write(f"Galaxies analyzed: {n_gal}\n")
        f.write(f"a_grad mean: {mean_a:.6f} ± {std_a:.6f}\n")
        f.write(f"b_gbar mean: {mean_b:.6f} ± {std_b:.6f}\n")
        f.write(f"ΔRMSE mean: {mean_delta:.6f}\n")
        f.write(f"% galaxies improved (ΔRMSE < 0): {frac_improve:.2f}\n")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(res_df["a_grad"], bins=20)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.0)
    plt.title("Local gradient coefficient (a_grad)")
    plt.xlabel("a_grad")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(res_df["b_gbar"], bins=20)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.0)
    plt.title("Local baryonic acceleration coefficient (b_gbar)")
    plt.xlabel("b_gbar")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intra-galaxy radial ΔF3 analysis versus local baryonic structure."
    )
    parser.add_argument("--input", required=True, help="Input CSV with radial profiles.")
    parser.add_argument("--outdir", default="results/intra_galaxy", help="Output directory.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    res_df = run_analysis(df)
    save_outputs(res_df, args.outdir)

    print("\n=== INTRA-GALAXY RESULTS ===")
    print(f"Galaxies analyzed: {len(res_df)}")
    print(f"a_grad mean: {res_df['a_grad'].mean():.4f} ± {res_df['a_grad'].std():.4f}")
    print(f"b_gbar mean: {res_df['b_gbar'].mean():.4f} ± {res_df['b_gbar'].std():.4f}")
    print(f"ΔRMSE mean: {res_df['delta_rmse'].mean():.4f}")
    print(f"% galaxies improved (ΔRMSE < 0): {(res_df['delta_rmse'] < 0).mean() * 100:.1f}%")
    print(f"\nOutputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
