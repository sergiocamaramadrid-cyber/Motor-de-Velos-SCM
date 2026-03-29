"""Figure 1 — Scatter plot: baseline vs SCM predictions."""
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/predictions/sparc_predictions.csv")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df["pred_base"], df["pred_scm"], alpha=0.5, s=8, color="steelblue")

minv = min(df["pred_base"].min(), df["pred_scm"].min())
maxv = max(df["pred_base"].max(), df["pred_scm"].max())
ax.plot([minv, maxv], [minv, maxv], "k--", lw=1, label="Identity")

ax.set_xlabel("Baseline prediction (km s$^{-1}$)")
ax.set_ylabel("SCM prediction (km s$^{-1}$)")
ax.set_title("Baseline vs SCM predictions — SPARC")
ax.legend()
fig.tight_layout()

fig.savefig("figures/figure01_scatter.png", dpi=300)
fig.savefig("figures/figure01_scatter.pdf")
print("Figure 1 saved.")
