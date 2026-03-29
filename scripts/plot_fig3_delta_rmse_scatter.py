"""Figure 3 — Per-galaxy ΔRMSE scatter (sorted by improvement)."""
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/scm_oos/oos_generalization_results.csv")

per_galaxy = (
    df.groupby("galaxy")["delta_rmse"]
    .median()
    .reset_index()
    .sort_values("delta_rmse")
    .reset_index(drop=True)
)

colors = ["#d62728" if v >= 0 else "steelblue" for v in per_galaxy["delta_rmse"]]

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(per_galaxy.index, per_galaxy["delta_rmse"], c=colors, s=20, alpha=0.85)
ax.axhline(0, color="k", lw=1, ls="--")

ax.set_xlabel("Galaxy (sorted by $\\Delta$RMSE)")
ax.set_ylabel(r"Median $\Delta$RMSE (SCM $-$ baseline)")
ax.set_title("Per-galaxy OOS improvement — SCM vs baseline")
fig.tight_layout()

fig.savefig("figures/figure03_delta_rmse_scatter.png", dpi=300)
fig.savefig("figures/figure03_delta_rmse_scatter.pdf")
print("Figure 3 saved.")
