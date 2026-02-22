#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr, kendalltau

_EPS = 1e-300


# -------------------------
# Model: nu(y)=1/(1-exp(-sqrt(y)))
# -------------------------
def nu_exponential_sqrt(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, np.inf)
    s = np.sqrt(y)
    denom = -np.expm1(-s)            # 1 - exp(-s)
    denom = np.maximum(denom, 1e-15) # numeric guard
    return 1.0 / denom


def g_pred_scm_v02(g_bar: np.ndarray, g0: float) -> np.ndarray:
    g_bar = np.asarray(g_bar, dtype=float)
    g_bar = np.clip(g_bar, 0.0, np.inf)
    g0 = float(g0)
    if (not np.isfinite(g0)) or g0 <= 0:
        raise ValueError("g0 must be finite and > 0")
    y = g_bar / g0
    return g_bar * nu_exponential_sqrt(y)


# -------------------------
# Helpers
# -------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@dataclass
class ColMap:
    galaxy: str
    gbar: str
    gobs: str
    mass: Optional[str] = None
    env: Optional[str] = None
    morph: Optional[str] = None
    barred: Optional[str] = None
    gasfrac: Optional[str] = None


def autodetect_columns(df: pd.DataFrame) -> ColMap:
    galaxy = _pick_col(df, ["galaxy", "galaxy_id", "name", "id"])
    if galaxy is None:
        raise KeyError("No galaxy ID column found. Tried: galaxy/galaxy_id/name/id")

    gbar = _pick_col(df, ["g_bar", "gbar", "g_bar_ms2", "gbar_mks", "gbar_cgs"])
    gobs = _pick_col(df, ["g_obs", "gobs", "g_obs_ms2", "gobs_mks", "gobs_cgs"])
    if gbar is None or gobs is None:
        raise KeyError("Missing g_bar/g_obs columns (need both). Tried common variants.")

    mass = _pick_col(df, ["log_mbar", "log_mass", "logMbar", "log_m", "mbar", "Mbar"])
    env = _pick_col(df, ["logSigma5", "log_sigma5", "Sigma5", "sigma5", "env", "env_class", "R_max", "r_max", "Rmax"])
    morph = _pick_col(df, ["morph_type", "morph", "type"])
    barred = _pick_col(df, ["is_barred", "barred", "has_bar"])
    gasfrac = _pick_col(df, ["gas_fraction", "fgas", "f_gas"])

    return ColMap(galaxy=galaxy, gbar=gbar, gobs=gobs, mass=mass, env=env, morph=morph, barred=barred, gasfrac=gasfrac)


# -------------------------
# Fit g0 (per galaxy)
# -------------------------
def fit_g0_for_galaxy(
    sub: pd.DataFrame,
    gbar_col: str,
    gobs_col: str,
    bounds_log10_g0: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Returns: (g0_hat, rms_dex) in log10 space for that galaxy.
    """
    gbar = _to_numeric(sub[gbar_col]).to_numpy(dtype=float)
    gobs = _to_numeric(sub[gobs_col]).to_numpy(dtype=float)
    m = np.isfinite(gbar) & np.isfinite(gobs) & (gbar > 0) & (gobs > 0)
    gbar = gbar[m]
    gobs = gobs[m]
    if len(gbar) < 5:
        return (np.nan, np.nan)

    def objective(log10_g0: float) -> float:
        g0 = 10.0 ** float(log10_g0)
        gpred = g_pred_scm_v02(gbar, g0)
        r = np.log10(np.maximum(gobs, _EPS)) - np.log10(np.maximum(gpred, _EPS))
        return float(np.median(np.abs(r)))

    res = minimize_scalar(objective, bounds=bounds_log10_g0, method="bounded", options={"xatol": 1e-3})
    g0_hat = float(10.0 ** res.x)

    gpred = g_pred_scm_v02(gbar, g0_hat)
    r = np.log10(np.maximum(gobs, _EPS)) - np.log10(np.maximum(gpred, _EPS))
    rms_dex = float(np.sqrt(np.mean(r**2)))
    return g0_hat, rms_dex


# -------------------------
# Matched pairs
# -------------------------
def matched_pairs_delta(
    df_gal: pd.DataFrame,
    mass_col: str,
    env_col: str,
    g0_col: str = "g0_hat",
    mass_tol: float = 0.10,
    env_min_sep: float = 0.50,
    max_pairs_per_gal: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Pair galaxies with similar mass but separated environment; compute Δlog10(g0).
    """
    rng = np.random.default_rng(int(seed))
    d = df_gal[[mass_col, env_col, g0_col]].copy()
    d[mass_col] = _to_numeric(d[mass_col])
    d[env_col] = _to_numeric(d[env_col])
    d[g0_col] = _to_numeric(d[g0_col])
    d = d.dropna()
    if len(d) < 8:
        return pd.DataFrame()

    d["log10_g0"] = np.log10(np.maximum(d[g0_col].to_numpy(dtype=float), _EPS))

    rows = []
    idx = d.index.to_list()

    for i in idx:
        mi = float(d.loc[i, mass_col])
        ei = float(d.loc[i, env_col])
        cand = d[
            (d[mass_col] >= mi - mass_tol) & (d[mass_col] <= mi + mass_tol) &
            (np.abs(d[env_col] - ei) >= env_min_sep)
        ].copy()
        if len(cand) == 0:
            continue

        take = min(int(max_pairs_per_gal), len(cand))
        chosen = cand.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
        for j, rj in chosen.iterrows():
            rows.append({
                "i": i,
                "j": j,
                "mass_i": mi,
                "mass_j": float(rj[mass_col]),
                "env_i": ei,
                "env_j": float(rj[env_col]),
                "log10_g0_i": float(d.loc[i, "log10_g0"]),
                "log10_g0_j": float(rj["log10_g0"]),
                "delta_log10_g0": float(d.loc[i, "log10_g0"] - float(rj["log10_g0"])),
            })

    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    boots = []
    n = len(x)
    for _ in range(int(n_boot)):
        samp = x[rng.integers(0, n, size=n)]
        boots.append(float(np.median(samp)))
    boots = np.asarray(boots, dtype=float)
    return float(np.median(x)), float(np.quantile(boots, 0.16)), float(np.quantile(boots, 0.84))


# -------------------------
# Main
# -------------------------
def _preprocess_argv(argv: List[str]) -> List[str]:
    """Merge '--g0-bounds -16,-8' into '--g0-bounds=-16,-8' so argparse accepts it."""
    result: List[str] = []
    i = 0
    while i < len(argv):
        if (
            argv[i] == "--g0-bounds"
            and i + 1 < len(argv)
            and argv[i + 1].startswith("-")
            and "," in argv[i + 1]
        ):
            result.append(f"--g0-bounds={argv[i + 1]}")
            i += 2
        else:
            result.append(argv[i])
            i += 1
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (point-level: galaxy,r_kpc,g_bar,g_obs + optional covariates).")
    ap.add_argument("--out", default="results", help="Output directory.")
    ap.add_argument("--min-points", type=int, default=20, help="Min points per galaxy to fit g0.")
    ap.add_argument("--g0-bounds", default="-16,-8", help="log10 bounds, e.g. '-16,-8' -> (1e-16,1e-8).")
    ap.add_argument("--env-merge", default=None, help="Optional CSV to merge env/covariates (must include galaxy id).")
    ap.add_argument("--mass-tol", type=float, default=0.10, help="Matched-pair mass tolerance (dex).")
    ap.add_argument("--env-min-sep", type=float, default=0.50, help="Matched-pair minimum env separation (dex/units).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(_preprocess_argv(sys.argv[1:]))

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    diagdir = os.path.join(outdir, "diagnostics")
    os.makedirs(diagdir, exist_ok=True)

    lo_s, hi_s = args.g0_bounds.split(",")
    bounds = (float(lo_s), float(hi_s))

    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    if args.env_merge:
        df2 = pd.read_csv(args.env_merge, encoding="utf-8-sig")
        cm_tmp = autodetect_columns(df)
        galcol = cm_tmp.galaxy
        if galcol not in df2.columns:
            alt = _pick_col(df2, ["galaxy", "galaxy_id", "name", "id"])
            if alt is None:
                raise KeyError("env-merge provided but no galaxy id column found in that file.")
            df2 = df2.rename(columns={alt: galcol})
        df = df.merge(df2, on=galcol, how="left")

    cm = autodetect_columns(df)

    rows = []
    for gal, sub in df.groupby(cm.galaxy):
        if len(sub) < int(args.min_points):
            continue
        g0_hat, rms = fit_g0_for_galaxy(sub, cm.gbar, cm.gobs, bounds_log10_g0=bounds)
        if not np.isfinite(g0_hat):
            continue

        row = {
            "galaxy": gal,
            "n_points": int(len(sub)),
            "g0_hat": float(g0_hat),
            "log10_g0_hat": float(np.log10(g0_hat)),
            "rms_dex": float(rms),
            "gbar_min": float(np.nanmin(_to_numeric(sub[cm.gbar]))),
            "gbar_max": float(np.nanmax(_to_numeric(sub[cm.gbar]))),
        }

        for _, col in [("mass", cm.mass), ("env", cm.env), ("morph", cm.morph), ("barred", cm.barred), ("gasfrac", cm.gasfrac)]:
            if col is not None and col in sub.columns:
                v = sub[col].dropna()
                row[col] = v.iloc[0] if len(v) else np.nan

        rows.append(row)

    df_gal = pd.DataFrame(rows)
    out_gal = os.path.join(diagdir, "g0_per_galaxy.csv")
    df_gal.to_csv(out_gal, index=False)
    print(f"[OK] wrote {out_gal} rows={len(df_gal)}")

    def corr_report(xcol: str, label: str):
        if xcol not in df_gal.columns:
            print(f"[SKIP] {label}: column '{xcol}' not present.")
            return None
        x = _to_numeric(df_gal[xcol]).to_numpy()
        y = _to_numeric(df_gal["log10_g0_hat"]).to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            print(f"[SKIP] {label}: not enough data (n={m.sum()}).")
            return None
        rho, p = spearmanr(x[m], y[m])
        tau, pk = kendalltau(x[m], y[m])
        print(f"[CORR] {label}: Spearman rho={rho:.3f} p={p:.2e} | Kendall tau={tau:.3f} p={pk:.2e}")
        return {"covariate": label, "col": xcol, "spearman_rho": float(rho), "spearman_p": float(p),
                "kendall_tau": float(tau), "kendall_p": float(pk), "n": int(m.sum())}

    reports = []
    if cm.env is not None:
        reports.append(corr_report(cm.env, "env"))
    if cm.mass is not None:
        reports.append(corr_report(cm.mass, "mass"))
    if cm.morph is not None:
        reports.append(corr_report(cm.morph, "morph"))
    if cm.barred is not None:
        reports.append(corr_report(cm.barred, "barred"))
    if cm.gasfrac is not None:
        reports.append(corr_report(cm.gasfrac, "gas_fraction"))

    reports = [r for r in reports if r is not None]
    if reports:
        out_corr = os.path.join(diagdir, "g0_covariate_correlations.csv")
        pd.DataFrame(reports).to_csv(out_corr, index=False)
        print(f"[OK] wrote {out_corr}")

    if cm.mass is not None and cm.env is not None and (cm.mass in df_gal.columns) and (cm.env in df_gal.columns):
        pairs = matched_pairs_delta(
            df_gal,
            mass_col=cm.mass,
            env_col=cm.env,
            mass_tol=float(args.mass_tol),
            env_min_sep=float(args.env_min_sep),
            seed=int(args.seed),
        )
        out_pairs = os.path.join(diagdir, "g0_matched_pairs.csv")
        pairs.to_csv(out_pairs, index=False)
        print(f"[OK] wrote {out_pairs} rows={len(pairs)}")

        if len(pairs) > 0:
            med, lo, hi = bootstrap_ci(pairs["delta_log10_g0"].to_numpy(), n_boot=2000, seed=int(args.seed))
            print(f"[PAIRS] median Δlog10(g0)={med:.4f}  (p16={lo:.4f}, p84={hi:.4f})  n_pairs={len(pairs)}")
    else:
        print("[SKIP] matched-pairs: need both mass and env columns (or merge env table).")

    detected = {
        "galaxy": cm.galaxy,
        "g_bar": cm.gbar,
        "g_obs": cm.gobs,
        "mass": cm.mass,
        "env": cm.env,
        "morph": cm.morph,
        "barred": cm.barred,
        "gas_fraction": cm.gasfrac,
        "g0_bounds_log10": bounds,
        "min_points": int(args.min_points),
    }
    out_meta = os.path.join(diagdir, "g0_universality_meta.json")
    pd.Series(detected).to_json(out_meta, indent=2)
    print(f"[OK] wrote {out_meta}")


if __name__ == "__main__":
    main()
