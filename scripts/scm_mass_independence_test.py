#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scm_mass_independence_test.py — Test that the bifurcation signal survives
mass residualisation.

Loads the combined SPARC + MOJAVE dataset (after normalisation), partials
out mass, and verifies that the quadratic hinge term retains statistical
significance.  This script is the mass-independence shield for the SCM
unique-law claim.

Usage
-----
    python scripts/scm_mass_independence_test.py

Inputs (data/processed/)
-------------------------
    scm_level2_thermo_features.csv  — SPARC sample
    scm_bh_regime_labeled_final.csv — MOJAVE sample

Outputs (stdout + results/)
----------------------------
    scm_mass_independence_result.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SPARC_CSV = os.path.join(DATA_DIR, "scm_level2_thermo_features.csv")
MOJ_CSV = os.path.join(DATA_DIR, "scm_bh_regime_labeled_final.csv")


def _load_and_normalise():
    """Return combined DataFrame with columns x_res, y, m."""
    if not os.path.exists(SPARC_CSV):
        sys.exit(f"ERROR: {SPARC_CSV} not found.")
    if not os.path.exists(MOJ_CSV):
        sys.exit(f"ERROR: {MOJ_CSV} not found.")

    sparc = pd.read_csv(SPARC_CSV).dropna(subset=["grad_p", "v_res", "logMbar"])
    moj = pd.read_csv(MOJ_CSV).dropna(subset=["energy_proxy", "residual_theta"])

    sparc["x"] = (sparc["grad_p"] - sparc["grad_p"].mean()) / sparc["grad_p"].std()
    sparc["y"] = (sparc["v_res"] ** 2 - (sparc["v_res"] ** 2).mean()) / (
        sparc["v_res"] ** 2
    ).std()
    sparc["m"] = (sparc["logMbar"] - sparc["logMbar"].mean()) / sparc["logMbar"].std()

    E0 = moj["energy_proxy"].mean()
    moj["x"] = (moj["energy_proxy"] - E0) / moj["energy_proxy"].std()
    moj["y"] = (
        moj["residual_theta"] ** 2 - (moj["residual_theta"] ** 2).mean()
    ) / (moj["residual_theta"] ** 2).std()
    moj["m"] = np.nan

    df = pd.concat([sparc[["x", "y", "m"]], moj[["x", "y", "m"]]], ignore_index=True)

    mask = df["m"].notna()
    X_mass = sm.add_constant(df.loc[mask, ["m"]])
    mass_model = sm.OLS(df.loc[mask, "x"], X_mass).fit()

    df["x_res"] = df["x"].copy()
    df.loc[mask, "x_res"] = mass_model.resid
    df["Hx2"] = (df["x_res"] ** 2) * (df["x_res"] >= 0)

    return df.dropna(), mass_model


def run_mass_independence_test(n_perm: int = 500, seed: int = 42) -> dict:
    """Run the full mass-independence shield test.

    Returns
    -------
    dict with keys: p_perm, coef_Hx2, pvalue_Hx2, mass_r_squared,
                    mass_p_value_m, criterion_pass
    """
    d, mass_model = _load_and_normalise()

    X = sm.add_constant(d[["x_res", "Hx2"]])
    model = sm.OLS(d["y"], X).fit()

    obs_t = model.tvalues["Hx2"]
    rng = np.random.default_rng(seed)
    perm_t = []

    for _ in range(n_perm):
        d_p = d.copy()
        d_p["x_res"] = rng.permutation(d_p["x_res"].values)
        d_p["Hx2"] = (d_p["x_res"] ** 2) * (d_p["x_res"] >= 0)
        X_p = sm.add_constant(d_p[["x_res", "Hx2"]])
        m_p = sm.OLS(d_p["y"], X_p).fit()
        perm_t.append(m_p.tvalues["Hx2"])

    perm_t = np.array(perm_t)
    p_perm = float((np.sum(np.abs(perm_t) >= abs(obs_t)) + 1) / (n_perm + 1))

    result = {
        "n": int(len(d)),
        "n_perm": n_perm,
        "coef_Hx2": float(model.params["Hx2"]),
        "pvalue_Hx2": float(model.pvalues["Hx2"]),
        "p_perm": p_perm,
        "mass_r_squared": float(mass_model.rsquared),
        "criterion_pass": p_perm < 0.05,
    }

    return result


def main():
    print("=== SCM MASS INDEPENDENCE TEST ===")
    result = run_mass_independence_test()

    print(f"n combined:      {result['n']}")
    print(f"coef Hx2:        {result['coef_Hx2']:.4f}")
    print(f"p-value Hx2:     {result['pvalue_Hx2']:.4f}")
    print(f"p_perm:          {result['p_perm']:.4f}")
    print(f"mass R²:         {result['mass_r_squared']:.4f}")
    print(f"CRITERIO: {'PASS' if result['criterion_pass'] else 'FAIL'}")

    out = os.path.join(RESULTS_DIR, "scm_mass_independence_result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out}")

    return result


if __name__ == "__main__":
    main()
