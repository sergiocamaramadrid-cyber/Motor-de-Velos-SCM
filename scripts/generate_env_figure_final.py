import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA = Path("data/scm_final_dataset_79.csv")
OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

x = df["delta_mass_std"]
y = df["slope_tail"]

coef = np.polyfit(x, y, 1)
line = np.poly1d(coef)

order = np.argsort(x.to_numpy())
x_sorted = x.to_numpy()[order]

plt.figure(figsize=(6, 5))
plt.scatter(x, y, alpha=0.7)
plt.plot(x_sorted, line(x_sorted))

plt.xlabel("Environmental density (delta_mass_std)")
plt.ylabel("Outer slope (F3 / slope_tail)")
plt.title("Environmental modulation of outer disk dynamics")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTDIR / "figure_env_correlation.pdf")
plt.savefig(OUTDIR / "figure_env_correlation.png", dpi=300)
plt.show()
