"""Figure 2 — Histogram of ΔRMSE per galaxy across OOS seeds."""
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/scm_oos/oos_generalization_results.csv")

# Use median delta_rmse per galaxy to avoid seed repetition
per_galaxy = df.groupby("galaxy")["delta_rmse"].median().reset_index()

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(per_galaxy["delta_rmse"], bins=20, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0, color="k", lw=1, ls="--")

ax.set_xlabel(r"Median $\Delta$RMSE (SCM $-$ baseline)")
ax.set_ylabel("Number of galaxies")
ax.set_title(r"Distribution of $\Delta$RMSE per galaxy (OOS)")
fig.tight_layout()

fig.savefig("figures/figure02_delta_rmse_hist.png", dpi=300)
fig.savefig("figures/figure02_delta_rmse_hist.pdf")
print("Figure 2 saved.")
