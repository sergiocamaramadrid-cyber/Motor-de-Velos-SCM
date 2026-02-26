"""
scripts/calc_vif.py — Compute Variance Inflation Factors (VIF) for the
Motor de Velos SCM predictor variables.

This script diagnoses multicollinearity among the four predictors used in
the global SCM regression:

    logM      — log10 baryonic mass
    log_gbar  — log10 characteristic baryonic acceleration
    log_j     — log10 proxy for specific angular momentum
    hinge     — max(0, log_g0 − log_gbar)  [frozen g0 = 10^{−10.45} m/s²]

A VIF < 5 for all variables indicates no problematic multicollinearity.

Usage
-----
Run the main pipeline first to generate the input file::

    python -m src.scm_analysis --data-dir data/SPARC --out results/

Then compute VIF::

    python scripts/calc_vif.py

Results are written to ``results/audit/vif_table.csv``.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# Load per-galaxy audit data produced by run_pipeline
df = pd.read_csv("results/audit/sparc_global.csv")

# Build predictor matrix
X = pd.DataFrame()
X["logM"] = df["logM"]
X["log_gbar"] = df["log_gbar"]
X["log_j"] = df["log_j"]

# Hinge variable with frozen g0 (log10 scale)
logg0 = -10.45
X["hinge"] = np.maximum(0, logg0 - df["log_gbar"])

# Drop rows with any NaN so VIF computation is well-defined
X = X.dropna()

# Compute VIF for each predictor
vif = pd.DataFrame()
vif["variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(X.shape[1])]

# Save results
Path("results/audit").mkdir(parents=True, exist_ok=True)
vif.to_csv("results/audit/vif_table.csv", index=False)

print("\n=== RESULTADOS VIF ===")
print(vif)
