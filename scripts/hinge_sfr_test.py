#!/usr/bin/env python3
"""
hinge_sfr_test.py
Operational test: hinge-derived "friction" proxies (F1/F2/F3) vs SFR,
controlling for Mbar.  Designed to plug into SCM-Motor de Velos workflow.

Theory
------
The hinge per radial point is defined as:

    H(r_i) = d * max(0, log10(g0) - log10(g_bar(r_i)))

where g_bar is the baryonic centripetal acceleration at r_i.

Three "friction" proxies are computed over the outer region (r > 0.7 * Rmax):

    F1 = median |dH/dr|        (radial gradient — structural activity)
    F2 = IQR(H)                (variability / rugosity of hinge)
    F3 = mean(H)               (integrated hinge power)

Statistical tests
-----------------
For each proxy Fk the script runs:

1. OLS regression (HC3 robust errors):
       log SFR = a + b*log Mbar + c*Fk [+ morph dummies] + eps

2. Permutation test (H1: c_obs > 0):
       5000 permutations of Fk while keeping log Mbar fixed.

3. Matched-pairs Wilcoxon (one-sided, H1: delta > 0):
       Pairs matched by log Mbar (±0.1 dex) and, if available, morph_bin.

Inputs (recommended)
--------------------
profiles.csv      -- per radial point: galaxy, r_kpc, vbar_kms (or
                     gbar_m_s2), rmax_kpc (optional)
galaxy_table.csv  -- per galaxy: galaxy, log_mbar, log_sfr (or sfr),
                     morph_bin (optional), quality_flag (optional)

Outputs (written to results/hinge_sfr/)
----------------------------------------
hinge_features.csv
regression_<feature>.txt
permutation_<feature>.txt
matched_pairs_<feature>.txt
"""

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon


@dataclass
class HingeParams:
    log_g0: float  # log10(g0) in SI (m/s²) — from SCM best-fit
    d: float       # hinge amplitude — from SCM best-fit (or 1.0 for proxy test)


# 1 kpc in metres (IAU 2012)
_KPC_TO_M = 3.085677581e19


def ensure_dir(path: str) -> None:
    """Create *path* (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def robust_iqr(x: np.ndarray) -> float:
    """Return the inter-quartile range of *x*, ignoring NaNs."""
    q75, q25 = np.nanpercentile(x, [75, 25])
    return float(q75 - q25)


def compute_gbar_from_vbar(r_kpc: np.ndarray, vbar_kms: np.ndarray) -> np.ndarray:
    """Convert (r_kpc, v_bar_kms) to baryonic centripetal acceleration in m/s².

    g_bar = v_bar² / r  (centripetal relation)

    Parameters
    ----------
    r_kpc : array_like
        Galactocentric radii in **kiloparsecs** (kpc).  Typical galaxy radii
        span 0.1–100 kpc; values far outside this range trigger a
        unit-mismatch warning.
    vbar_kms : array_like
        Baryonic rotation velocity in **km/s**.  Typical values span
        10–500 km/s; values far outside this range trigger a unit-mismatch
        warning.

    Returns
    -------
    ndarray
        Baryonic centripetal acceleration in m/s² at each radial point.

    Warns
    -----
    UserWarning
        If r_kpc contains values > 5000 kpc (possible pc or m input),
        negative values, or if vbar_kms contains values > 2000 km/s
        (possible m/s input).
    """
    r_kpc = np.asarray(r_kpc, dtype=float)
    vbar_kms = np.asarray(vbar_kms, dtype=float)

    # --- Unit-sanity checks (non-fatal: warn rather than raise) ---
    r_finite = r_kpc[np.isfinite(r_kpc)]
    v_finite = vbar_kms[np.isfinite(vbar_kms)]

    if r_finite.size and r_finite.max() > 5000:
        warnings.warn(
            f"r_kpc contains values up to {r_finite.max():.4g} kpc, which is "
            "unusually large for a galaxy rotation-curve radius.  Check that "
            "radii are in kpc, not pc or m.",
            UserWarning,
            stacklevel=2,
        )
    if r_finite.size and (r_finite < 0).any():
        warnings.warn(
            "r_kpc contains negative values; galactocentric radii must be "
            "non-negative.",
            UserWarning,
            stacklevel=2,
        )
    if v_finite.size and v_finite.max() > 2000:
        warnings.warn(
            f"vbar_kms contains values up to {v_finite.max():.4g} km/s, which "
            "is unusually large for a baryonic rotation velocity.  Check that "
            "velocities are in km/s, not m/s.",
            UserWarning,
            stacklevel=2,
        )

    r_m = r_kpc * _KPC_TO_M
    v_m_s = vbar_kms * 1e3
    return (v_m_s ** 2) / np.maximum(r_m, 1e-30)


def compute_hinge(log_g0: float, d: float, gbar_si: np.ndarray) -> np.ndarray:
    """Compute the hinge term H(r) = d * max(0, log10(g0) - log10(g_bar)).

    Parameters
    ----------
    log_g0 : float
        log10 of the characteristic acceleration g0 (SI, m/s²).
    d : float
        Hinge amplitude (dimensionless scaling factor).
    gbar_si : array_like
        Baryonic centripetal acceleration in m/s² at each radial point.

    Returns
    -------
    ndarray
        Hinge values H(r) ≥ 0 at each radial point.
    """
    log_gbar = np.log10(np.maximum(gbar_si, 1e-60))
    return d * np.maximum(0.0, log_g0 - log_gbar)


def compute_features_for_galaxy(df_g: pd.DataFrame, hp: HingeParams) -> dict:
    """Compute F1, F2, F3 friction proxies for a single galaxy.

    Parameters
    ----------
    df_g : pd.DataFrame
        Per-radial-point data for one galaxy.  Required columns:
        ``galaxy``, ``r_kpc``, and either ``gbar_m_s2`` or ``vbar_kms``.
        Optional: ``rmax_kpc``.
    hp : HingeParams
        Global SCM hinge parameters.

    Returns
    -------
    dict
        Keys: galaxy, rmax_kpc_used, F1_med_abs_dH_dr_ext,
        F2_IQR_H_ext, F3_mean_H_ext, H_ext_npts.
    """
    r = df_g["r_kpc"].to_numpy(dtype=float)

    if "gbar_m_s2" in df_g.columns:
        gbar = df_g["gbar_m_s2"].to_numpy(dtype=float)
    else:
        gbar = compute_gbar_from_vbar(r, df_g["vbar_kms"].to_numpy(dtype=float))

    H = compute_hinge(hp.log_g0, hp.d, gbar)

    # Define the "outer" region used for all three proxies
    if "rmax_kpc" in df_g.columns and np.isfinite(df_g["rmax_kpc"].iloc[0]):
        rmax = float(df_g["rmax_kpc"].iloc[0])
    else:
        rmax = float(np.nanmax(r))

    mask = r > 0.7 * rmax
    if mask.sum() < 5:
        # fallback to outer half when the outer-30% region is sparse
        mask = r > 0.5 * rmax

    r_ext = r[mask]
    H_ext = H[mask]

    # Sort by radius for the gradient estimate
    order = np.argsort(r_ext)
    r_ext = r_ext[order]
    H_ext = H_ext[order]

    # F1: median of |dH/dr| (finite differences)
    dH = np.diff(H_ext)
    dr = np.diff(r_ext)
    grad = np.abs(dH / np.maximum(dr, 1e-12))
    F1 = float(np.nanmedian(grad)) if grad.size else np.nan

    # F2: IQR of H in the outer region
    F2 = robust_iqr(H_ext) if H_ext.size else np.nan

    # F3: mean of H in the outer region
    F3 = float(np.nanmean(H_ext)) if H_ext.size else np.nan

    return {
        "galaxy": df_g["galaxy"].iloc[0],
        "rmax_kpc_used": rmax,
        "F1_med_abs_dH_dr_ext": F1,
        "F2_IQR_H_ext": F2,
        "F3_mean_H_ext": F3,
        "H_ext_npts": int(H_ext.size),
    }


def regression_test(df: pd.DataFrame, feature: str) -> str:
    """OLS regression log_SFR ~ log_Mbar + feature [+ morph dummies].

    Uses HC3 heteroskedasticity-robust covariance to guard against
    outlier-driven false significance.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy-level table with columns ``log_sfr``, ``log_mbar``,
        *feature*, and optionally ``morph_bin``.
    feature : str
        Name of the friction-proxy column.

    Returns
    -------
    str
        Full statsmodels OLS summary as text.
    """
    y = df["log_sfr"].astype(float)
    X_cols = ["log_mbar", feature]

    if "morph_bin" in df.columns:
        dummies = pd.get_dummies(
            df["morph_bin"], prefix="morph", drop_first=True
        ).astype(float)
        X = pd.concat([df[X_cols], dummies], axis=1)
    else:
        X = df[X_cols].copy()

    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
    return model.summary().as_text()


def permutation_pvalue(
    df: pd.DataFrame,
    feature: str,
    n_perm: int = 5000,
    seed: int = 7,
) -> float:
    """One-sided permutation p-value for the friction-proxy coefficient.

    Shuffles *feature* relative to log_SFR while keeping log_Mbar fixed,
    so the null distribution reflects "no partial effect of Fk at fixed mass".

    H1: the OLS coefficient of *feature* > 0 (more friction → more SFR).

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy-level table.
    feature : str
        Name of the friction-proxy column.
    n_perm : int
        Number of permutations (default 5000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        One-sided p-value P(c_perm ≥ c_obs | H0).
    """
    rng = np.random.default_rng(seed)

    y = df["log_sfr"].astype(float).to_numpy()
    x_m = df["log_mbar"].astype(float).to_numpy()
    x_f = df[feature].astype(float).to_numpy()

    # Drop rows with any NaN
    mask = np.isfinite(y) & np.isfinite(x_m) & np.isfinite(x_f)
    y, x_m, x_f = y[mask], x_m[mask], x_f[mask]

    X = np.column_stack([np.ones_like(x_m), x_m, x_f])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    c_obs = beta[-1]

    more_extreme = 0
    for _ in range(n_perm):
        x_fp = rng.permutation(x_f)
        Xp = np.column_stack([np.ones_like(x_m), x_m, x_fp])
        betap = np.linalg.lstsq(Xp, y, rcond=None)[0]
        if betap[-1] >= c_obs:
            more_extreme += 1

    return (more_extreme + 1) / (n_perm + 1)


def matched_pairs_wilcoxon(
    df: pd.DataFrame,
    feature: str,
    dlogm: float = 0.1,
) -> dict:
    """Mass-matched-pair test: does higher Fk predict higher log SFR?

    Pairs are formed by matching galaxies on log_Mbar (within *dlogm* dex)
    and, if available, within the same ``morph_bin``.  Within each pair the
    galaxy with higher *feature* value is labelled "highF".

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy-level table.
    feature : str
        Name of the friction-proxy column.
    dlogm : float
        Maximum allowed |Δlog Mbar| for a valid pair (default 0.1 dex).

    Returns
    -------
    dict
        Keys: n_pairs, wilcoxon_p (one-sided, alternative="greater"),
        median_delta_logSFR.
    """
    cols = ["galaxy", "log_mbar", "log_sfr", feature]
    if "morph_bin" in df.columns:
        cols.append("morph_bin")
    work = df[cols].copy().dropna().sort_values("log_mbar").reset_index(drop=True)

    used: set = set()
    deltas = []

    for i in range(len(work)):
        gi = work.loc[i, "galaxy"]
        if gi in used:
            continue

        mi = float(work.loc[i, "log_mbar"])
        cond = np.abs(work["log_mbar"] - mi) <= dlogm

        if "morph_bin" in work.columns:
            cond &= work["morph_bin"] == work.loc[i, "morph_bin"]

        candidates = work[cond & work["galaxy"].apply(lambda g: g not in used)].copy()
        if len(candidates) < 2:
            continue

        candidates["_dm"] = np.abs(candidates["log_mbar"] - mi)
        candidates = candidates.sort_values("_dm")

        row_j = None
        for _, rj in candidates.iterrows():
            if rj["galaxy"] != gi:
                row_j = rj
                break
        if row_j is None:
            continue

        used.add(gi)
        used.add(row_j["galaxy"])

        row_i = work.loc[i]
        if float(row_i[feature]) >= float(row_j[feature]):
            high, low = row_i, row_j
        else:
            high, low = row_j, row_i

        deltas.append(float(high["log_sfr"]) - float(low["log_sfr"]))

    deltas_arr = np.array(deltas, dtype=float)
    n_pairs = int(deltas_arr.size)
    median_delta = float(np.nanmedian(deltas_arr)) if n_pairs else np.nan

    if n_pairs < 10:
        return {
            "n_pairs": n_pairs,
            "wilcoxon_p": np.nan,
            "median_delta_logSFR": median_delta,
        }

    stat = wilcoxon(deltas_arr, alternative="greater", zero_method="wilcox")
    return {
        "n_pairs": n_pairs,
        "wilcoxon_p": float(stat.pvalue),
        "median_delta_logSFR": median_delta,
    }


def main(argv=None) -> None:
    """Run the full hinge-friction vs SFR test pipeline.

    By default reads ``profiles.csv`` and ``galaxy_table.csv`` from the
    current working directory.  Use ``--profiles`` / ``--galaxy-table`` to
    point to files in any location (e.g. ``data/hinge_sfr/``).

    Customise ``--log-g0`` and ``--d`` to match your SCM best-fit values.
    """
    parser = argparse.ArgumentParser(
        description="Hinge-friction vs SFR test (F1/F2/F3 proxies)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profiles",
        default="profiles.csv",
        help="Per-radial-point CSV (columns: galaxy, r_kpc, vbar_kms or "
             "gbar_m_s2, rmax_kpc [optional])",
    )
    parser.add_argument(
        "--galaxy-table",
        dest="galaxy_table",
        default="galaxy_table.csv",
        help="Per-galaxy CSV (columns: galaxy, log_mbar, log_sfr or sfr, "
             "morph_bin [optional])",
    )
    parser.add_argument(
        "--out",
        default="results/hinge_sfr",
        help="Output directory",
    )
    parser.add_argument(
        "--log-g0",
        dest="log_g0",
        type=float,
        default=np.log10(3.27e-11),
        help="log10(g0) in SI m/s² — use your SCM best-fit value",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=1.0,
        help="Hinge amplitude d — use your SCM best-fit value",
    )
    args = parser.parse_args(argv)

    outdir = args.out
    ensure_dir(outdir)

    hp = HingeParams(log_g0=args.log_g0, d=args.d)

    profiles = pd.read_csv(args.profiles)
    gal = pd.read_csv(args.galaxy_table)

    # Compute friction features per galaxy
    feats = []
    for _gname, df_g in profiles.groupby("galaxy"):
        feats.append(compute_features_for_galaxy(df_g, hp))
    feats_df = pd.DataFrame(feats)
    feats_df.to_csv(os.path.join(outdir, "hinge_features.csv"), index=False)

    # Merge with galaxy-level table
    df = gal.merge(feats_df, on="galaxy", how="inner")

    # Convert linear SFR to log if necessary
    if "log_sfr" not in df.columns and "sfr" in df.columns:
        df["log_sfr"] = np.log10(np.maximum(df["sfr"].astype(float), 1e-12))

    features = ["F1_med_abs_dH_dr_ext", "F2_IQR_H_ext", "F3_mean_H_ext"]
    for feature in features:
        # 1. OLS regression with HC3 robust errors
        summ = regression_test(df, feature)
        with open(
            os.path.join(outdir, f"regression_{feature}.txt"), "w", encoding="utf-8"
        ) as fh:
            fh.write(summ)

        # 2. Permutation test (one-sided: coef > 0)
        p_perm = permutation_pvalue(df, feature, n_perm=5000, seed=7)
        with open(
            os.path.join(outdir, f"permutation_{feature}.txt"), "w", encoding="utf-8"
        ) as fh:
            fh.write(
                f"Permutation p-value (H1: feature coef > 0): {p_perm:.6g}\n"
            )

        # 3. Mass-matched Wilcoxon test
        mp = matched_pairs_wilcoxon(df, feature, dlogm=0.1)
        with open(
            os.path.join(outdir, f"matched_pairs_{feature}.txt"), "w", encoding="utf-8"
        ) as fh:
            fh.write(f"n_pairs = {mp['n_pairs']}\n")
            fh.write(f"median_delta_logSFR = {mp['median_delta_logSFR']:.6g}\n")
            fh.write(f"wilcoxon_p (greater) = {mp['wilcoxon_p']:.6g}\n")

    print(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
