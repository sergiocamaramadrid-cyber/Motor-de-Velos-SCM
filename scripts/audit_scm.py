#!/usr/bin/env python
"""
scripts/audit_scm.py — Automated statistical audit of the SCM model.

Performs:
  - GroupKFold cross-validation by galaxy
  - Permutation test within galaxy
  - AICc model comparison (BTFR vs SCM without hinge vs SCM full)
  - Saves metrics and coefficients
  - Reproducible with --seed

Expected input CSV columns:
  galaxy_id, v_obs, M_bar, g_bar, j_star

Usage
-----
    python scripts/audit_scm.py \\
        --input data/SPARC/sparc_raw.csv \\
        --outdir results/audit \\
        --ref $(git rev-parse --short HEAD) \\
        --kfold 5 \\
        --permutations 200 \\
        --seed 42
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -------------------------
# Utilities
# -------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def aicc(n, k, logL):
    aic = 2 * k - 2 * logL
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def gaussian_loglik(y_true, y_pred):
    resid = y_true - y_pred
    sigma2 = np.var(resid, ddof=1)
    n = len(y_true)
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)


def hinge(loggbar, logg0):
    return np.maximum(0.0, logg0 - loggbar)


# -------------------------
# Model builder
# -------------------------

def build_features(df, use_hinge=True, logg0=-10.45):
    df = df.copy()
    df["logMbar"] = np.log10(df["M_bar"])
    df["loggbar"] = np.log10(df["g_bar"])
    df["logj"] = np.log10(df["j_star"])
    df["logv_obs"] = np.log10(df["v_obs"])

    features = ["logMbar", "loggbar", "logj"]

    if use_hinge:
        df["hinge"] = hinge(df["loggbar"].values, logg0)
        features.append("hinge")

    return df, features


# -------------------------
# GroupKFold OOS
# -------------------------

def run_groupkfold(df, features, k=5, seed=123):
    gkf = GroupKFold(n_splits=k)
    X = df[features].values
    y = df["logv_obs"].values
    groups = df["galaxy_id"].values

    oof_pred = np.zeros_like(y)
    coeffs = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = LinearRegression()
        model.fit(X[tr], y[tr])
        oof_pred[te] = model.predict(X[te])

        coeff_row = dict(zip(features, model.coef_))
        coeff_row["intercept"] = model.intercept_
        coeff_row["fold"] = fold
        coeffs.append(coeff_row)

    df = df.copy()
    df["logv_pred_oof"] = oof_pred
    df["resid_log_oof"] = df["logv_obs"] - df["logv_pred_oof"]

    return df, pd.DataFrame(coeffs)


# -------------------------
# Permutation test
# -------------------------

def permutation_test(df, features, k=5, n_perm=100, seed=123):
    rng = np.random.default_rng(seed)
    real_rmse = rmse(df["logv_obs"], df["logv_pred_oof"])

    perm_rmses = []

    for i in range(n_perm):
        df_perm = df.copy()
        df_perm["loggbar"] = (
            df_perm.groupby("galaxy_id")["loggbar"]
            .transform(lambda x: rng.permutation(x.values))
        )

        df_perm, _ = run_groupkfold(df_perm, features, k=k, seed=seed)
        perm_rmses.append(rmse(df_perm["logv_obs"], df_perm["logv_pred_oof"]))

    return real_rmse, perm_rmses


# -------------------------
# Model comparison AICc
# -------------------------

def compare_models(df):
    results = []

    models = {
        "BTFR": ["logMbar"],
        "SCM_no_hinge": ["logMbar", "loggbar", "logj"],
        "SCM_full": ["logMbar", "loggbar", "logj", "hinge"],
    }

    for name, feats in models.items():
        X = df[feats].values
        y = df["logv_obs"].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        logL = gaussian_loglik(y, y_pred)
        n = len(y)
        k = len(feats) + 1  # + intercept

        results.append({
            "model": name,
            "k": k,
            "n": n,
            "logL": logL,
            "AICc": aicc(n, k, logL),
        })

    return pd.DataFrame(results)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ref", required=True)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    # Build full SCM
    df_full, features_full = build_features(df, use_hinge=True)

    # GroupKFold
    df_oof, coeffs = run_groupkfold(
        df_full, features_full, k=args.kfold, seed=args.seed
    )

    df_oof.to_csv(os.path.join(args.outdir, "groupkfold_predictions.csv"), index=False)
    coeffs.to_csv(os.path.join(args.outdir, "coeffs_by_fold.csv"), index=False)

    # Permutation
    real_rmse, perm_rmses = permutation_test(
        df_oof, features_full, k=args.kfold,
        n_perm=args.permutations, seed=args.seed
    )

    with open(os.path.join(args.outdir, "permutation_summary.json"), "w") as f:
        json.dump({
            "real_rmse_log": real_rmse,
            "perm_rmse_mean": float(np.mean(perm_rmses)),
            "perm_rmse_std": float(np.std(perm_rmses)),
        }, f, indent=2)

    # AICc comparison
    model_comp = compare_models(df_full)
    model_comp.to_csv(os.path.join(args.outdir, "model_comparison_aicc.csv"), index=False)

    print("Audit complete.")
    print("Results saved in:", args.outdir)


if __name__ == "__main__":
    main()
