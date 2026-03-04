#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    ap = argparse.ArgumentParser(
        description="SCM: regression test beta (friction_slope) vs external HI density (logSigmaHI_out)"
    )
    ap.add_argument("--catalog", default="results/f3_catalog_v2.parquet",
                    help="Path to Parquet catalog (default: results/f3_catalog_v2.parquet)")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Significance threshold for the key predictor (default: 0.01)")
    args = ap.parse_args()

    print("=== ANALISIS SCM: beta vs densidad HI externa ===")
    print(f"catalog: {args.catalog}")

    try:
        df = pd.read_parquet(args.catalog)
    except Exception as e:
        print(f"ERROR: cannot read catalog parquet: {e}", file=sys.stderr)
        return 2

    # Map beta column
    if "friction_slope" in df.columns:
        beta_col = "friction_slope"
    elif "beta" in df.columns:
        beta_col = "beta"
    else:
        print("ERROR: catalog missing beta column (expected 'friction_slope' or 'beta')", file=sys.stderr)
        print(f"Columns: {list(df.columns)}", file=sys.stderr)
        return 3

    required = [beta_col, "logSigmaHI_out", "logMbar", "inc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing required columns: {missing}", file=sys.stderr)
        print(f"Columns: {list(df.columns)}", file=sys.stderr)
        return 4

    # Clean rows
    df = df.dropna(subset=required).copy()
    df = df[np.isfinite(df[required]).all(axis=1)]

    print(f"Numero de galaxias analizadas: {len(df)}")
    if len(df) < 10:
        print("WARNING: very small N; results may be unstable.", file=sys.stderr)

    # Design matrix
    X = df[["logSigmaHI_out", "logMbar", "inc"]]
    X = sm.add_constant(X)
    y = df[beta_col]

    # OLS + robust SE (HC3)
    model = sm.OLS(y, X).fit(cov_type="HC3")

    print("\n=== RESULTADOS REGRESION (OLS, robust HC3) ===")
    print(model.summary())

    coef = float(model.params["logSigmaHI_out"])
    pval = float(model.pvalues["logSigmaHI_out"])

    print("\n=== RESULTADO CLAVE SCM ===")
    print(f"beta_col = {beta_col}")
    print(f"coef_logSigmaHI_out = {coef:.6f}")
    print(f"p_value = {pval:.6g}")

    if (pval < args.alpha) and (coef > 0):
        print("Interpretacion: señal positiva consistente con la hipótesis del Velo.")
        return 0
    else:
        print("Interpretacion: no se observa señal estadistica fuerte (o signo contrario).")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())