import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("data/processed/scm_bh_regime_labeled_final.csv")

d = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["theta_jet", "logr15", "energy_proxy"]
).copy()

d["efficiency"] = d["energy_proxy"] / (10 ** d["logr15"])
d["geom_factor"] = d["theta_jet"] / d["logr15"]

print("N:", len(d))

print("theta vs energy:", spearmanr(d["theta_jet"], d["energy_proxy"]))
print("logr15 vs energy:", spearmanr(d["logr15"], d["energy_proxy"]))
print("eff vs theta:", spearmanr(d["efficiency"], d["theta_jet"]))
print("geom vs energy:", spearmanr(d["geom_factor"], d["energy_proxy"]))
