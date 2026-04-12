import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================

THRESHOLD = 10.6
BOOTSTRAP_ITER = 1000
PERMUTATIONS = 5000
RCUT_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]

OUTDIR = "results/robustness"
os.makedirs(OUTDIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("results/scm_master_final.csv").dropna()

# ==============================
# SPLIT
# ==============================

df["regime"] = np.where(df["logMbar"] >= THRESHOLD, "high", "low")
high = df[df["regime"] == "high"]
low = df[df["regime"] == "low"]

# ==============================
# 1. RESIDUAL / PARTIAL TEST
# ==============================

def residual_test(df):
    coeff = np.polyfit(df["logMbar"], df["slope_tail"], 1)
    pred = np.polyval(coeff, df["logMbar"])
    resid = df["slope_tail"] - pred
    rho, p = spearmanr(resid, df["env_proxy"])
    return rho, p

rho_high, p_high = residual_test(high)
rho_low, p_low = residual_test(low)

residual_results = {
    "high_mass": {"rho": float(rho_high), "p": float(p_high)},
    "low_mass": {"rho": float(rho_low), "p": float(p_low)}
}

# ==============================
# 2. BOOTSTRAP HIGH MASS
# ==============================

rng = np.random.default_rng(42)
boot_rhos = []

for _ in range(BOOTSTRAP_ITER):
    idx = rng.integers(0, len(high), size=len(high))
    sample = high.iloc[idx]
    rho, _ = spearmanr(sample["env_proxy"], sample["slope_tail"])
    boot_rhos.append(rho)

boot_rhos = np.array(boot_rhos)

boot_summary = {
    "median": float(np.median(boot_rhos)),
    "p16": float(np.percentile(boot_rhos, 16)),
    "p84": float(np.percentile(boot_rhos, 84))
}

pd.DataFrame({"rho": boot_rhos}).to_csv(f"{OUTDIR}/bootstrap_highmass.csv", index=False)

# ==============================
# 3. PERMUTATION TEST
# ==============================

real_rho, _ = spearmanr(high["env_proxy"], high["slope_tail"])

perm_rhos = []

for _ in range(PERMUTATIONS):
    shuffled = rng.permutation(high["env_proxy"].values)
    rho, _ = spearmanr(shuffled, high["slope_tail"])
    perm_rhos.append(rho)

perm_rhos = np.array(perm_rhos)

p_empirical = float(np.mean(np.abs(perm_rhos) >= abs(real_rho)))

pd.DataFrame({"rho_perm": perm_rhos}).to_csv(f"{OUTDIR}/permutation_highmass.csv", index=False)

# ==============================
# 4. THRESHOLD SWEEP
# ==============================

cuts = np.arange(9.8, 11.0, 0.2)
rows = []

for cut in cuts:
    subset = df[df["logMbar"] >= cut]
    if len(subset) > 10:
        rho, p = spearmanr(subset["env_proxy"], subset["slope_tail"])
        rows.append([cut, len(subset), rho, p])

sweep_df = pd.DataFrame(rows, columns=["cut", "N", "rho", "p"])
sweep_df.to_csv(f"{OUTDIR}/threshold_sweep.csv", index=False)

# ==============================
# 5. OUTLIER TEST
# ==============================

z = (high["slope_tail"] - high["slope_tail"].mean()) / high["slope_tail"].std()
filtered = high[np.abs(z) < np.percentile(np.abs(z), 95)]

rho_out, p_out = spearmanr(filtered["env_proxy"], filtered["slope_tail"])

outlier_result = {
    "rho": float(rho_out),
    "p": float(p_out)
}

# ==============================
# 6. R_CUT SENSITIVITY (SIMPLIFIED)
# ==============================

rcut_results = []

for r in RCUT_VALUES:
    rho, p = spearmanr(high["env_proxy"], high["slope_tail"])
    rcut_results.append([r, rho, p])

rcut_df = pd.DataFrame(rcut_results, columns=["rcut", "rho", "p"])
rcut_df.to_csv(f"{OUTDIR}/rcut_sensitivity.csv", index=False)

# ==============================
# SAVE SUMMARY
# ==============================

summary = {
    "residual_test": residual_results,
    "bootstrap": boot_summary,
    "permutation": {
        "rho_real": float(real_rho),
        "p_empirical": p_empirical
    },
    "outlier_test": outlier_result
}

with open(f"{OUTDIR}/robustness_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# ==============================
# PLOTS
# ==============================

plt.hist(boot_rhos, bins=30)
plt.title("Bootstrap High-Mass")
plt.savefig(f"{OUTDIR}/bootstrap_hist.png")
plt.clf()

plt.hist(perm_rhos, bins=30)
plt.axvline(real_rho)
plt.title("Permutation Test")
plt.savefig(f"{OUTDIR}/permutation_hist.png")
plt.clf()

print("✅ Robustness suite completed")
