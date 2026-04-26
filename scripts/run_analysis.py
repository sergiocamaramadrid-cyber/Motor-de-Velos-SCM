import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

DATA_PATH = "data/mojave_vizier_table3.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df = df[["r15", "alphaApp15"]].dropna().copy()
df = df.rename(columns={"alphaApp15": "theta_jet"})
df["logr15"] = np.log10(df["r15"])

cut = 10

df["regime"] = np.where(df["r15"] < cut, "LOW", "HIGH")

low = df[df["regime"] == "LOW"]
high = df[df["regime"] == "HIGH"]

ks = stats.ks_2samp(low["theta_jet"], high["theta_jet"])
rho, p = stats.spearmanr(high["logr15"], high["theta_jet"])

print("TOTAL:", len(df))
print("LOW:", len(low))
print("HIGH:", len(high))
print("KS p:", ks.pvalue)
print("rho:", rho)
print("p:", p)

summary = df.groupby("regime")["theta_jet"].agg(
    N="count",
    Mean="mean",
    Median="median",
    Std="std"
).round(2)

summary.to_csv(os.path.join(RESULTS_DIR, "table_descriptive.csv"))

plt.figure(figsize=(6, 5))
plt.scatter(low["logr15"], low["theta_jet"], s=12, alpha=0.5, label="LOW")
plt.scatter(high["logr15"], high["theta_jet"], s=12, alpha=0.8, label="HIGH")
plt.axvline(np.log10(cut), linestyle="--")
plt.xlabel(r"$\log r_{15}$")
plt.ylabel(r"$\theta_{\mathrm{jet}}$ (deg)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig1_transition.png"), dpi=300)

plt.figure(figsize=(6, 5))
plt.scatter(high["logr15"], high["theta_jet"], s=14, alpha=0.8)
plt.xlabel(r"$\log r_{15}$")
plt.ylabel(r"$\theta_{\mathrm{jet}}$ (deg)")
plt.title(r"HIGH regime: $\rho=-0.331$, $p=0.0019$")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig2_high.png"), dpi=300)
