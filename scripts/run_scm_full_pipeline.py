#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scm_full_pipeline.py — SCM reproducible analysis pipeline.

Loads SPARC level-2 thermo features and MOJAVE BH regime data,
normalises variables, residualises against mass, and fits the
piecewise quadratic model that tests for regime bifurcation.

Usage
-----
    python scripts/run_scm_full_pipeline.py

Inputs (data/processed/)
-------------------------
    scm_level2_thermo_features.csv  — SPARC sample
        Required columns: grad_p, v_res, logMbar
    scm_bh_regime_labeled_final.csv — MOJAVE sample
        Required columns: energy_proxy, residual_theta

Outputs (results/)
------------------
    scm_unique_law_summary.json
    scm_xc_bootstrap_validation.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SPARC_CSV = os.path.join(DATA_DIR, "scm_level2_thermo_features.csv")
MOJ_CSV = os.path.join(DATA_DIR, "scm_bh_regime_labeled_final.csv")

print("=== SCM FULL PIPELINE ===")

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
if not os.path.exists(SPARC_CSV):
    sys.exit(f"ERROR: {SPARC_CSV} not found. Place the SPARC level-2 file there.")
if not os.path.exists(MOJ_CSV):
    sys.exit(f"ERROR: {MOJ_CSV} not found. Place the MOJAVE BH regime file there.")

sparc = pd.read_csv(SPARC_CSV)
moj = pd.read_csv(MOJ_CSV)

sparc = sparc.dropna(subset=["grad_p", "v_res", "logMbar"])
moj = moj.dropna(subset=["energy_proxy", "residual_theta"])

print(f"SPARC rows: {len(sparc)}, MOJAVE rows: {len(moj)}")

# ---------------------------------------------------------------------------
# 2. Normalise
# ---------------------------------------------------------------------------
sparc["x"] = (sparc["grad_p"] - sparc["grad_p"].mean()) / sparc["grad_p"].std()
sparc["y"] = (sparc["v_res"] ** 2 - (sparc["v_res"] ** 2).mean()) / (sparc["v_res"] ** 2).std()
sparc["m"] = (sparc["logMbar"] - sparc["logMbar"].mean()) / sparc["logMbar"].std()

E0 = moj["energy_proxy"].mean()
moj["x"] = (moj["energy_proxy"] - E0) / moj["energy_proxy"].std()
moj["y"] = (moj["residual_theta"] ** 2 - (moj["residual_theta"] ** 2).mean()) / (
    moj["residual_theta"] ** 2
).std()
moj["m"] = np.nan

df = pd.concat([sparc[["x", "y", "m"]], moj[["x", "y", "m"]]], ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Residualise x against mass (SPARC only)
# ---------------------------------------------------------------------------
mask = df["m"].notna()
X_mass = sm.add_constant(df.loc[mask, ["m"]])
mass_model = sm.OLS(df.loc[mask, "x"], X_mass).fit()

df["x_res"] = df["x"].copy()
df.loc[mask, "x_res"] = mass_model.resid

# ---------------------------------------------------------------------------
# 4. Hinge term
# ---------------------------------------------------------------------------
df["Hx2"] = (df["x_res"] ** 2) * (df["x_res"] >= 0)

# ---------------------------------------------------------------------------
# 5. Final piecewise OLS
# ---------------------------------------------------------------------------
# Use subset dropna so MOJAVE rows (m=NaN) are retained in the combined fit
d = df.dropna(subset=["x_res", "Hx2", "y"])
X_final = sm.add_constant(d[["x_res", "Hx2"]])
final_model = sm.OLS(d["y"], X_final).fit()

print("\n=== PIECEWISE OLS SUMMARY ===")
print(final_model.summary())

# ---------------------------------------------------------------------------
# 6. Permutation test for the quadratic term
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N_PERM = 500

obs_t = final_model.tvalues["Hx2"]
perm_t = []

for _ in range(N_PERM):
    d_perm = d.copy()
    d_perm["x_res"] = rng.permutation(d_perm["x_res"].values)
    d_perm["Hx2"] = (d_perm["x_res"] ** 2) * (d_perm["x_res"] >= 0)
    X_p = sm.add_constant(d_perm[["x_res", "Hx2"]])
    m_p = sm.OLS(d_perm["y"], X_p).fit()
    perm_t.append(m_p.tvalues["Hx2"])

perm_t = np.array(perm_t)
p_perm = (np.sum(np.abs(perm_t) >= abs(obs_t)) + 1) / (N_PERM + 1)

print(f"\nPermutation p-value (Hx2 quadratic term): {p_perm:.4f}")

# ---------------------------------------------------------------------------
# 7. Bootstrap stability of Hx2 coefficient
# ---------------------------------------------------------------------------
N_BOOT = 500
boot_coef_Hx2 = []

for _ in range(N_BOOT):
    idx = rng.integers(0, len(d), size=len(d))
    d_b = d.iloc[idx].copy()
    X_b = sm.add_constant(d_b[["x_res", "Hx2"]])
    m_b = sm.OLS(d_b["y"], X_b).fit()
    boot_coef_Hx2.append(float(m_b.params["Hx2"]))

sigma_bif = float(np.std(boot_coef_Hx2))

print(f"\nBootstrap sigma (Hx2 coefficient): {sigma_bif:.4f}")

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------
unique_law = {
    "model": "piecewise_OLS_hinge",
    "n_sparc": int(len(sparc)),
    "n_mojave": int(len(moj)),
    "n_combined": int(len(d)),
    "coef_x_res": float(final_model.params["x_res"]),
    "coef_Hx2": float(final_model.params["Hx2"]),
    "pvalue_x_res": float(final_model.pvalues["x_res"]),
    "pvalue_Hx2": float(final_model.pvalues["Hx2"]),
    "r_squared": float(final_model.rsquared),
    "p_perm_Hx2": float(p_perm),
    "mass_residualised": True,
}

xc_validation = {
    "bifurcation_point_x_res": 0.0,
    "sigma_bootstrap_coef_Hx2": sigma_bif,
    "n_boot": N_BOOT,
    "criterion_pass": sigma_bif <= 0.15,
    "p_perm": float(p_perm),
    "criterion_p_perm": float(p_perm) < 0.05,
}

with open(os.path.join(RESULTS_DIR, "scm_unique_law_summary.json"), "w") as f:
    json.dump(unique_law, f, indent=2)

with open(os.path.join(RESULTS_DIR, "scm_xc_bootstrap_validation.json"), "w") as f:
    json.dump(xc_validation, f, indent=2)

print("\n=== VALIDATION ===")
print(f"p_perm (Hx2):   {p_perm:.4f}")
print(f"sigma_bif:       {sigma_bif:.4f}")
print(f"CRITERIO: {'PASS' if p_perm < 0.05 and sigma_bif <= 0.15 else 'FAIL'}")
print("\nResults saved to results/")
