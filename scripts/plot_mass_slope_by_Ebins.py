import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/scm_canonical_dataset.csv")
df = df.dropna(subset=["slope_tail","logMbar","env_proxy_formal"])
df["E_bin"] = pd.qcut(df["env_proxy_formal"], q=4, duplicates="drop")

plt.figure(figsize=(8,6))
markers = ["o","s","^","D"]

for i, (b, g) in enumerate(df.groupby("E_bin", observed=False)):
    plt.scatter(g["logMbar"], g["slope_tail"], alpha=0.75, label=str(b), marker=markers[i % 4])
    if len(g) >= 3:
        z = np.polyfit(g["logMbar"], g["slope_tail"], 1)
        x = np.linspace(g["logMbar"].min(), g["logMbar"].max(), 100)
        plt.plot(x, z[0]*x + z[1], linestyle="--", linewidth=1.5)

plt.xlabel("logMbar")
plt.ylabel("Outer slope (dlogV/dlogr)")
plt.title("Mass–slope relation modulated by E_SCM")
plt.legend(title="E_SCM bins", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("results/figures/fig_mass_slope_Ebins.pdf")
plt.show()
