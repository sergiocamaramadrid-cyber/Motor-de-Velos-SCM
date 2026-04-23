import pandas as pd
from scm_bh_generic import SCMGeneric

df = pd.read_csv("data/processed/bh_catalog_clean.csv")

if df.empty:
    raise ValueError("Dataset vacío o no cargado correctamente")

scm = SCMGeneric()
scm.load_data(df)

print("\n--- SIMPLE ---")
print(scm.spearman("logM_BH", "theta_jet"))
print(scm.spearman("logL_bol", "theta_jet"))

thr = df["logM_BH"].median()

print("\n--- SPLIT ---")
print(scm.split_test("logL_bol", "theta_jet", "logM_BH", thr))

scm.add_E()

print("\n--- COMPOSITE ---")
print(scm.spearman("E_BH", "theta_jet"))

print("\n--- ROBUSTNESS ---")
print(scm.permutation("E_BH", "theta_jet"))
print(scm.bootstrap("E_BH", "theta_jet"))
