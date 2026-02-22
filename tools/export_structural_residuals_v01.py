#!/usr/bin/env python3
# tools/export_structural_residuals_v01.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def _pick(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No column for {label}. Tried={{candidates}}. Available={{list(df.columns)}}")

def _maybe_pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _load_hc3_regression():
    """
    Prefer using the project's hc3_regression if available.
    Fallback to simple OLS if not (keeps pipeline working).
    """
    try:
        from scm_env_protocol import hc3_regression  # type: ignore
        return hc3_regression, "project_hc3_regression"
    except Exception:
        pass

    def _ols(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope = beta[0], beta[1]

        yhat = intercept + slope * x
        resid = y - yhat
        dof = max(len(y) - 2, 1)
        s2 = np.sum(resid ** 2) / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(s2 * xtx_inv))
        return {
            "intercept": float(intercept),
            "slope": float(slope),
            "se_intercept": float(se[0]),
            "se_slope": float(se[1]),
        }

    return _ols, "fallback_ols"

def compute_per_galaxy_residuals(df, fit_on="all", id_col=None):
    """
    residual_logv = log_velocity - (intercept + slope * log_mass)
    fit_on:
      - all: single global fit
      - by_env: separate fit per env_class==0/1 (if present)
    """
    logm = _pick(df, ["log_mass", "log_mbar", "logM", "log_m"], "log_mass")
    logv = _pick(df, ["log_velocity", "logV", "log_v", "log_vflat"], "log_velocity")

    df_out = df.copy()

    if id_col is None:
        id_col = _maybe_pick(df_out, ["galaxy_id", "name", "galaxy", "id"])
    if id_col is None:
        df_out["galaxy_id"] = np.arange(len(df_out), dtype=int)
        id_col = "galaxy_id"
        df_out["__id_is_synthetic__"] = True
    else:
        df_out["__id_is_synthetic__"] = False

    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna(subset=[logm, logv]).copy()

    hc3_regression, fit_backend = _load_hc3_regression()

    for c in [
        "logv_pred", "residual_logv", "fit_group",
        "fit_intercept", "fit_slope", "fit_se_intercept_hc3", "fit_se_slope_hc3",
        "fit_backend",
    ]:
        if c not in df_out.columns:
            df_out[c] = np.nan if c not in ["fit_group", "fit_backend"] else ""

    def _fit_apply(mask, group_label):
        res = hc3_regression(df_out.loc[mask, logm].values, df_out.loc[mask, logv].values)
        intercept = float(res.get("intercept", np.nan))
        slope = float(res.get("slope", np.nan))
        se_i = float(res.get("se_intercept", np.nan))
        se_s = float(res.get("se_slope", np.nan))

        pred = intercept + slope * df_out.loc[mask, logm]
        df_out.loc[mask, "logv_pred"] = pred
        df_out.loc[mask, "residual_logv"] = df_out.loc[mask, logv] - pred
        df_out.loc[mask, "fit_group"] = group_label
        df_out.loc[mask, "fit_intercept"] = intercept
        df_out.loc[mask, "fit_slope"] = slope
        df_out.loc[mask, "fit_se_intercept_hc3"] = se_i
        df_out.loc[mask, "fit_se_slope_hc3"] = se_s
        df_out.loc[mask, "fit_backend"] = fit_backend

    if fit_on == "all":
        _fit_apply(np.ones(len(df_out), dtype=bool), "all")
    elif fit_on == "by_env":
        if "env_class" not in df_out.columns:
            raise ValueError("fit_on='by_env' requires env_class column (0/1).")
        for env in [0, 1]:
            m = (df_out["env_class"] == env)
            if int(m.sum()) >= 3:
                _fit_apply(m, f"env_{{env}}")
            else:
                df_out.loc[m, "fit_group"] = f"env_{{env}}_insufficient_n"
    else:
        raise ValueError("fit_on must be 'all' or 'by_env'")

    keep = [
        id_col, "__id_is_synthetic__",
        logm, logv,
        "logv_pred", "residual_logv",
        "fit_group", "fit_backend", "fit_intercept", "fit_slope", "fit_se_intercept_hc3", "fit_se_slope_hc3",
        "env_class",
        "logSigma5", "log_sigma5", "Sigma5", "sigma5",
        "morph_type", "is_barred", "gas_fraction",
        "surf_brightness", "distance_Mpc",
    ]
    keep = [c for c in keep if c in df_out.columns]

    out = df_out[keep].copy()
    out = out.rename(columns={{id_col: "galaxy_id", logm: "log_mass", logv: "log_velocity"}})

    if "logSigma5" not in out.columns:
        if "log_sigma5" in out.columns:
            out = out.rename(columns={{"log_sigma5": "logSigma5"}})
        elif "Sigma5" in out.columns:
            out["logSigma5"] = np.log10(pd.to_numeric(out["Sigma5"], errors="coerce"))
        elif "sigma5" in out.columns:
            out["logSigma5"] = np.log10(pd.to_numeric(out["sigma5"], errors="coerce"))

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("df_master", nargs="?", default="df_master.csv", help="Path to df_master.csv (default: df_master.csv")
    ap.add_argument("--outdir", default="scm_results_final/diagnostics", help="Output directory")
    ap.add_argument("--by-env", action="store_true", help="Also export fit_by_env if env_class exists")
    args = ap.parse_args()

    in_path = Path(args.df_master)
    if not in_path.exists():
        raise FileNotFoundError(f"df_master not found: {{in_path.resolve()}}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    residuals_all = compute_per_galaxy_residuals(df, fit_on="all")
    out_all = outdir / "per_galaxy_residuals_fit_all.csv"
    residuals_all.to_csv(out_all, index=False)

    print(f"[OK] wrote: {{out_all}}")
    print(f"[INFO] rows={{len(residuals_all)}} cols={{list(residuals_all.columns)}}")
    print(residuals_all.head(5).to_string(index=False))

    if args.by_env and ("env_class" in df.columns):
        residuals_env = compute_per_galaxy_residuals(df, fit_on="by_env")
        out_env = outdir / "per_galaxy_residuals_fit_by_env.csv"
        residuals_env.to_csv(out_env, index=False)
        print(f"[OK] wrote: {{out_env}}")

if __name__ == "__main__":
    main()