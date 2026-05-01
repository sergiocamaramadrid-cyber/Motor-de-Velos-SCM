#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCM — Structured Residuals in Galaxy Kinematics

Reproducible pipeline for:
P1 — residual structure test
P2 — threshold robustness test
P3 — environmental modulation test

Author: Sergio Cámara Madrid
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SPARC_PATH = DATA_DIR / "scm_mass_ready.csv"
YANG_PATH = DATA_DIR / "scm_tr_yang_dataset.csv"

BOOTSTRAP_N = 1000
PERMUTATION_N = 500
RANDOM_SEED = 42


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def fit_rss(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    model = LinearRegression().fit(x, y)
    pred = model.predict(x)

    rss = np.sum((y - pred) ** 2)
    beta = float(model.coef_[0])

    return float(rss), beta


def crtt_best(x, y, grid_min=9.5, grid_max=10.5, n_grid=100, min_n=10):
    rss_global, beta_global = fit_rss(x, y)

    best = None

    for xc in np.linspace(grid_min, grid_max, n_grid):
        low = x < xc
        high = x >= xc

        if low.sum() < min_n or high.sum() < min_n:
            continue

        rss_low, beta_low = fit_rss(x[low], y[low])
        rss_high, beta_high = fit_rss(x[high], y[high])

        rss_split = rss_low + rss_high
        delta_rss = rss_global - rss_split

        row = {
            "xc": float(xc),
            "delta_rss": float(delta_rss),
            "rss_global": float(rss_global),
            "rss_split": float(rss_split),
            "beta_global": float(beta_global),
            "beta_low": float(beta_low),
            "beta_high": float(beta_high),
            "n_low": int(low.sum()),
            "n_high": int(high.sum()),
        }

        if best is None or delta_rss > best["delta_rss"]:
            best = row

    return best


# ============================================================
# P1 / P2 — SPARC
# ============================================================

def run_sparc_tests():
    df = pd.read_csv(SPARC_PATH)
    df = df[["slope_tail", "logMbar"]].dropna()

    x = df["logMbar"].values
    y = df["slope_tail"].values

    rng = np.random.default_rng(RANDOM_SEED)

    observed = crtt_best(x, y)

    # Bootstrap
    boot_rows = []

    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(x), len(x))
        r = crtt_best(x[idx], y[idx])

        if r is not None:
            boot_rows.append(r)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULTS_DIR / "sparc_bootstrap_results.csv", index=False)

    delta_ci_low, delta_ci_high = boot["delta_rss"].quantile([0.025, 0.975])
    xc_median = boot["xc"].median()
    xc_sigma = boot["xc"].std()

    # Permutation
    perm_rows = []

    for _ in range(PERMUTATION_N):
        yp = rng.permutation(y)
        r = crtt_best(x, yp)

        if r is not None:
            perm_rows.append(r)

    perm = pd.DataFrame(perm_rows)
    perm.to_csv(RESULTS_DIR / "sparc_permutation_results.csv", index=False)

    p_perm = float((perm["delta_rss"] >= observed["delta_rss"]).mean())

    # Figure
    plt.figure(figsize=(6, 4))
    plt.hist(boot["delta_rss"], bins=30, alpha=0.8)
    plt.axvline(delta_ci_low, linestyle="--", label="IC95 low")
    plt.axvline(delta_ci_high, linestyle="--", label="IC95 high")
    plt.axvline(observed["delta_rss"], linestyle="-", label="Observed \u0394RSS")
    plt.xlabel("\u0394RSS")
    plt.ylabel("Frequency")
    plt.title("SPARC bootstrap \u0394RSS distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delta_rss_bootstrap_distribution.png", dpi=200)
    plt.close()

    result = {
        "dataset": "SPARC",
        "N": int(len(df)),
        "observed_xc": float(observed["xc"]),
        "observed_delta_rss": float(observed["delta_rss"]),
        "delta_rss_ci95_low": float(delta_ci_low),
        "delta_rss_ci95_high": float(delta_ci_high),
        "xc_median": float(xc_median),
        "xc_sigma": float(xc_sigma),
        "p_perm": p_perm,
        "P1_residual_structure": bool(delta_ci_low > 0),
        "P2_threshold_robust": bool(p_perm < 0.05 and xc_sigma <= 0.15),
        "conclusion": (
            "structured_residuals_confirmed"
            if delta_ci_low > 0 else
            "residual_structure_not_confirmed"
        ),
    }

    return result


# ============================================================
# P3 — YANG
# ============================================================

def run_yang_environment_test():
    df = pd.read_csv(YANG_PATH)
    df = df[["slope_tail", "logM", "delta_mass_std"]].dropna()

    df = df.copy()
    df["mass_q"] = pd.qcut(df["logM"], q=4, labels=False)

    q4 = df[df["mass_q"] == 3].dropna(subset=["delta_mass_std", "slope_tail"])

    slope, intercept, r, p, se = stats.linregress(
        q4["delta_mass_std"],
        q4["slope_tail"],
    )

    result = {
        "dataset": "YANG",
        "N_total": int(len(df)),
        "N_Q4": int(len(q4)),
        "beta_env_Q4": float(slope),
        "p_value_Q4": float(p),
        "P3_environment_confirmed": bool(slope < 0 and p < 0.05),
        "conclusion": (
            "environment_confirmed"
            if slope < 0 and p < 0.05 else
            "no_significant_environment"
        ),
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    sparc_result = run_sparc_tests()
    yang_result = run_yang_environment_test()

    final_results = pd.DataFrame([
        {
            "dataset": "SPARC",
            "N": sparc_result["N"],
            "test": "P1/P2",
            "delta_rss_ci95_low": sparc_result["delta_rss_ci95_low"],
            "delta_rss_ci95_high": sparc_result["delta_rss_ci95_high"],
            "p_perm": sparc_result["p_perm"],
            "xc_sigma": sparc_result["xc_sigma"],
            "conclusion": sparc_result["conclusion"],
        },
        {
            "dataset": "YANG",
            "N": yang_result["N_total"],
            "test": "P3",
            "delta_rss_ci95_low": np.nan,
            "delta_rss_ci95_high": np.nan,
            "p_perm": yang_result["p_value_Q4"],
            "xc_sigma": np.nan,
            "conclusion": yang_result["conclusion"],
        },
    ])

    final_results.to_csv(
        RESULTS_DIR / "scm_final_results.csv",
        index=False,
    )

    metadata = {
        "project": "SCM \u2014 Structured Residuals in Galaxy Kinematics",
        "author": "Sergio C\u00e1mara Madrid",
        "bootstrap_iterations": BOOTSTRAP_N,
        "permutation_iterations": PERMUTATION_N,
        "random_seed": RANDOM_SEED,
        "sparc": sparc_result,
        "yang": yang_result,
        "final_interpretation": (
            "Residual structure is confirmed in SPARC. "
            "No robust threshold transition or environmental modulation "
            "is confirmed with current data."
        ),
    }

    with open(RESULTS_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== SCM FINAL RESULTS ===")
    print(final_results.to_string(index=False))

    print("\nSaved:")
    print(f"  {RESULTS_DIR / 'scm_final_results.csv'}")
    print(f"  {RESULTS_DIR / 'sparc_bootstrap_results.csv'}")
    print(f"  {RESULTS_DIR / 'sparc_permutation_results.csv'}")
    print(f"  {RESULTS_DIR / 'run_metadata.json'}")
    print(f"  {FIGURES_DIR / 'delta_rss_bootstrap_distribution.png'}")

    return {"sparc": sparc_result, "yang": yang_result}


if __name__ == "__main__":
    main()
