import pandas as pd
import statsmodels.api as sm

print("=== ANALISIS SCM: beta vs densidad HI externa ===")

# cargar catálogo generado previamente
df = pd.read_parquet("results/f3_catalog_v2.parquet")

# eliminar filas con datos faltantes
df = df.dropna(subset=["beta","logSigmaHI_out","logMbar","inc"]) 

print(f"Numero de galaxias analizadas: {len(df)}")

# variables independientes
X = df[["logSigmaHI_out","logMbar","inc"]]
X = sm.add_constant(X)

# variable dependiente
y = df["beta"]

# regresion
model = sm.OLS(y, X).fit()

print("\n=== RESULTADOS REGRESION ===")
print(model.summary())

coef = model.params["logSigmaHI_out"]
pval = model.pvalues["logSigmaHI_out"]

print("\n=== RESULTADO CLAVE SCM ===")
print(f"coef_logSigmaHI_out = {coef:.5f}")
print(f"p_value = {pval:.5f}")

if pval < 0.01 and coef > 0:
    print("Interpretacion: señal positiva consistente con la hipótesis del Velo.")
else:
    print("Interpretacion: no se observa señal estadistica fuerte.")