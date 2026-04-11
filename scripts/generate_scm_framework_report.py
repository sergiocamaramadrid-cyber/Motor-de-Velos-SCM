#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCM Framework PDF Report Generator (repo-ready)

- Compatible con:
    mass: logM / logMbar
    target: F3_SCM / slope_tail
    env: env_proxy (obligatorio)

- Estadística:
    OLS + HC3
    Bootstrap IC95%
    Permutación p-valor

- Portable: sin Colab, sin rutas hardcodeadas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm

# ----------------------------
# CONFIG
# ----------------------------
RANDOM_SEED = 42

# ----------------------------
# ARGUMENTOS
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mass-cut", type=float, default=10.5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=5000)
    return p.parse_args()

# ----------------------------
# DETECCIÓN COLUMNAS
# ----------------------------
def resolve_columns(df):

    def find(options):
        for o in options:
            if o in df.columns:
                return o
        return None

    mass = find(["logM", "logMbar"])
    target = find(["F3_SCM", "slope_tail", "F3"])
    env = find(["env_proxy"])

    if not mass:
        raise ValueError("No se encontró columna de masa")
    if not target:
        raise ValueError("No se encontró columna de pendiente")
    if not env:
        raise ValueError("Se requiere env_proxy para SCM")

    return mass, env, target

# ----------------------------
# LIMPIEZA
# ----------------------------
def clean(df, cols):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=cols)

# ----------------------------
# MODELO
# ----------------------------
def fit(df, mass, env, target):
    X = df[[mass, env]]
    X = sm.add_constant(X)
    y = df[target]
    return sm.OLS(y, X).fit(cov_type="HC3")

# ----------------------------
# BOOTSTRAP
# ----------------------------
def bootstrap_ci(df, mass, env, target, n):
    rng = np.random.default_rng(RANDOM_SEED)
    betas = []

    for _ in range(n):
        sample = df.sample(len(df), replace=True, random_state=int(rng.integers(0, 2**31)))
        try:
            m = fit(sample, mass, env, target)
            betas.append(m.params[env])
        except Exception:
            pass

    lo, hi = np.percentile(betas, [2.5, 97.5])
    return lo, hi

# ----------------------------
# PERMUTACIÓN
# ----------------------------
def permutation_p(df, mass, env, target, n, beta_obs):
    rng = np.random.default_rng(RANDOM_SEED)
    count = 0

    for _ in range(n):
        perm = df.copy()
        perm[env] = rng.permutation(df[env].values)
        try:
            m = fit(perm, mass, env, target)
            if abs(m.params[env]) >= abs(beta_obs):
                count += 1
        except Exception:
            pass

    return (count + 1) / (n + 1)

# ----------------------------
# MAIN
# ----------------------------
def main(argv=None):
    args = parse_args() if argv is None else _parse_args_from(argv)

    df = pd.read_csv(args.input)
    mass, env, target = resolve_columns(df)

    df = clean(df, [mass, env, target])

    df_high = df[df[mass] >= args.mass_cut]

    if len(df_high) < 10:
        raise ValueError("Submuestra demasiado pequeña")

    model = fit(df_high, mass, env, target)

    beta = model.params[env]
    pval = model.pvalues[env]
    r2 = model.rsquared

    ci_lo, ci_hi = bootstrap_ci(df_high, mass, env, target, args.bootstrap)
    p_perm = permutation_p(df_high, mass, env, target, args.permutations, beta)

    # ---------------- PDF ----------------
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:

        # portada
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        fig.text(0.5, 0.85, "Framework SCM - Motor de Velos", ha="center", fontsize=18, weight="bold")
        fig.text(0.5, 0.75, f"N total: {len(df)} | Alta masa: {len(df_high)}", ha="center")
        pdf.savefig(fig)
        plt.close()

        # tabla
        fig, ax = plt.subplots()
        ax.axis("off")
        table = [
            ["β_env", f"{beta:.4f}"],
            ["p (HC3)", f"{pval:.4g}"],
            ["IC95%", f"[{ci_lo:.4f},{ci_hi:.4f}]"],
            ["p perm", f"{p_perm:.4g}"],
            ["R²", f"{r2:.4f}"],
        ]
        ax.table(cellText=table, loc="center")
        pdf.savefig(fig)
        plt.close()

        # scatter
        fig, ax = plt.subplots()
        sc = ax.scatter(df_high[env], df_high[target], c=df_high[mass], cmap="viridis")
        plt.colorbar(sc, ax=ax, label=mass)
        ax.set_xlabel(env)
        ax.set_ylabel(target)
        pdf.savefig(fig)
        plt.close()

    print("✅ PDF generado:", output)

    return {
        "output": str(output),
        "n_total": len(df),
        "n_high": len(df_high),
        "beta_env": beta,
        "pval_hc3": pval,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_perm": p_perm,
        "r2": r2,
    }


def _parse_args_from(argv):
    """Parse a list of argument strings (for programmatic use)."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mass-cut", type=float, default=10.5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=5000)
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
