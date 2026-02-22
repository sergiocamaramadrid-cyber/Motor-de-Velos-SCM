"""
g0_universality.py — Test whether the RAR acceleration scale g₀ is universal.

Fits g₀ per galaxy and per galaxy sub-group (mass proxy, morphology proxy),
then tests whether the fitted g₀ values are consistent with a single universal
value using a KS test and inter-quartile statistics.

Typical usage
-------------
    from src.g0_universality import run_universality_analysis
    results = run_universality_analysis("data/sparc_rar_sample.csv")
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.scm_models import fit_g0, G0_DEFAULT

# Minimum number of data points required to fit g₀ for a single galaxy
MIN_POINTS_PER_GALAXY = 5


# ---------------------------------------------------------------------------
# Per-galaxy g₀ fitting
# ---------------------------------------------------------------------------

def fit_g0_per_galaxy(
    df: pd.DataFrame,
    min_points: int = MIN_POINTS_PER_GALAXY,
) -> pd.DataFrame:
    """Fit g₀ independently for each galaxy in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``galaxy``, ``g_bar``, ``g_obs``.
    min_points : int, optional
        Galaxies with fewer than this many valid data points are skipped.

    Returns
    -------
    pd.DataFrame with one row per fitted galaxy and columns:
        ``galaxy``, ``n_points``, ``g0``, ``g0_err``, ``rms``,
        ``log10_g0``, ``median_g_bar`` (mass proxy),
        ``max_r_kpc`` (size proxy, if column present).
    """
    records = []
    for galaxy, group in df.groupby("galaxy"):
        gb = group["g_bar"].values
        go = group["g_obs"].values

        mask = (gb > 0) & (go > 0) & np.isfinite(gb) & np.isfinite(go)
        if mask.sum() < min_points:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_g0(gb[mask], go[mask])
        except Exception:
            continue

        row: dict = {
            "galaxy": galaxy,
            "n_points": int(mask.sum()),
            "g0": res["g0"],
            "g0_err": res["g0_err"],
            "rms": res["rms"],
            "log10_g0": np.log10(res["g0"]),
            "median_g_bar": float(np.median(gb[mask])),
        }
        if "r_kpc" in group.columns:
            row["max_r_kpc"] = float(group["r_kpc"].max())

        records.append(row)

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Group-level g₀ fitting
# ---------------------------------------------------------------------------

def fit_g0_by_group(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Fit a single g₀ for each category defined by *group_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``g_bar``, ``g_obs``, and the *group_col* column.
    group_col : str
        Column used to partition galaxies into groups (e.g. ``mass_quartile``).

    Returns
    -------
    pd.DataFrame with columns:
        ``group``, ``n_galaxies``, ``n_points``, ``g0``, ``g0_err``,
        ``rms``, ``log10_g0``.
    """
    records = []
    for grp_val, group in df.groupby(group_col):
        gb = group["g_bar"].values
        go = group["g_obs"].values

        mask = (gb > 0) & (go > 0) & np.isfinite(gb) & np.isfinite(go)
        n_gals = group["galaxy"].nunique() if "galaxy" in group.columns else np.nan

        if mask.sum() < MIN_POINTS_PER_GALAXY:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_g0(gb[mask], go[mask])
        except Exception:
            continue

        records.append({
            "group": grp_val,
            "n_galaxies": n_gals,
            "n_points": int(mask.sum()),
            "g0": res["g0"],
            "g0_err": res["g0_err"],
            "rms": res["rms"],
            "log10_g0": np.log10(res["g0"]),
        })

    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Universality statistics
# ---------------------------------------------------------------------------

def universality_stats(
    per_galaxy: pd.DataFrame,
    reference_g0: float = G0_DEFAULT,
) -> dict:
    """Compute summary statistics for the per-galaxy g₀ distribution.

    Parameters
    ----------
    per_galaxy : pd.DataFrame
        Output of :func:`fit_g0_per_galaxy`.
    reference_g0 : float, optional
        Reference (canonical MOND) value for comparison.

    Returns
    -------
    dict with keys:
        ``n_galaxies``          — number of galaxies fitted
        ``g0_median``           — median g₀ across galaxies
        ``g0_std``              — standard deviation of g₀
        ``log10_g0_std``        — standard deviation of log10(g₀) (dex scatter)
        ``g0_p16``, ``g0_p84`` — 16th and 84th percentiles
        ``reference_g0``        — reference value supplied
        ``ref_within_1sigma``   — True if reference falls within median ± std
    """
    g0_arr = per_galaxy["g0"].values
    log10_g0 = per_galaxy["log10_g0"].values

    median = float(np.median(g0_arr))
    std = float(np.std(g0_arr))
    log_std = float(np.std(log10_g0))

    return {
        "n_galaxies": len(per_galaxy),
        "g0_median": median,
        "g0_std": std,
        "log10_g0_std": log_std,
        "g0_p16": float(np.percentile(g0_arr, 16)),
        "g0_p84": float(np.percentile(g0_arr, 84)),
        "reference_g0": reference_g0,
        "ref_within_1sigma": bool(abs(reference_g0 - median) <= std),
    }


def ks_test_mass_quartiles(
    df: pd.DataFrame,
    per_galaxy: pd.DataFrame,
    n_quartiles: int = 4,
) -> dict:
    """KS test comparing g₀ distributions between mass quartiles.

    Galaxies are split into *n_quartiles* groups by their median g_bar
    (a proxy for total baryonic mass).  The KS statistic between the
    lowest and highest quartile g₀ distributions is reported.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (used to annotate per_galaxy).
    per_galaxy : pd.DataFrame
        Output of :func:`fit_g0_per_galaxy`.
    n_quartiles : int, optional
        Number of mass groups (default 4).

    Returns
    -------
    dict with keys:
        ``ks_stat``, ``ks_pvalue``, ``low_median_g0``, ``high_median_g0``,
        ``delta_log10_g0``, ``is_universal`` (True if p > 0.05).
    """
    # Assign mass quartile to each galaxy in per_galaxy
    pg = per_galaxy.copy()
    pg["mass_quartile"] = pd.qcut(
        pg["median_g_bar"], q=n_quartiles, labels=False, duplicates="drop"
    )

    low = pg[pg["mass_quartile"] == pg["mass_quartile"].min()]["g0"].values
    high = pg[pg["mass_quartile"] == pg["mass_quartile"].max()]["g0"].values

    if len(low) < 3 or len(high) < 3:
        return {
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "low_median_g0": float(np.median(low)) if len(low) else np.nan,
            "high_median_g0": float(np.median(high)) if len(high) else np.nan,
            "delta_log10_g0": np.nan,
            "is_universal": False,  # indeterminate due to insufficient data
        }

    stat, pval = ks_2samp(low, high)
    low_med = float(np.median(low))
    high_med = float(np.median(high))

    return {
        "ks_stat": float(stat),
        "ks_pvalue": float(pval),
        "low_median_g0": low_med,
        "high_median_g0": high_med,
        "delta_log10_g0": float(np.log10(high_med) - np.log10(low_med)),
        "is_universal": bool(pval > 0.05),
    }


# ---------------------------------------------------------------------------
# Full universality pipeline
# ---------------------------------------------------------------------------

def run_universality_analysis(
    csv_path: str,
    n_quartiles: int = 4,
    min_points: int = MIN_POINTS_PER_GALAXY,
    out_dir: str | None = None,
) -> dict:
    """Run the complete g₀ universality analysis.

    Steps
    -----
    1. Load CSV and clean data.
    2. Fit g₀ per galaxy.
    3. Compute universality summary statistics.
    4. Split by mass quartile and fit g₀ per group.
    5. KS test between low- and high-mass quartiles.
    6. (Optional) save CSV results.

    Parameters
    ----------
    csv_path : str
        Path to SPARC RAR CSV (must include ``galaxy``, ``g_bar``, ``g_obs``).
    n_quartiles : int, optional
        Number of mass groups for the group analysis (default 4).
    min_points : int, optional
        Minimum data points required per galaxy (default 5).
    out_dir : str, optional
        Directory to write result CSVs.

    Returns
    -------
    dict with keys:
        ``per_galaxy``   — DataFrame of per-galaxy g₀ fits
        ``stats``        — universality summary statistics dict
        ``group_fits``   — DataFrame of per-mass-quartile g₀ fits
        ``ks``           — KS test results dict
    """
    from src.scm_analysis import load_sparc_csv
    from pathlib import Path

    df = load_sparc_csv(csv_path)

    if "galaxy" not in df.columns:
        raise ValueError("CSV must contain a 'galaxy' column for per-galaxy analysis.")

    per_galaxy = fit_g0_per_galaxy(df, min_points=min_points)
    stats = universality_stats(per_galaxy)

    # Assign mass quartile label to each data point for group fitting
    galaxy_quartile = per_galaxy[["galaxy", "median_g_bar"]].copy()
    galaxy_quartile["mass_quartile"] = pd.qcut(
        galaxy_quartile["median_g_bar"], q=n_quartiles, labels=False, duplicates="drop"
    ).astype("Int64")
    df_annotated = df.merge(
        galaxy_quartile[["galaxy", "mass_quartile"]], on="galaxy", how="left"
    )
    df_annotated = df_annotated.dropna(subset=["mass_quartile"])

    group_fits = fit_g0_by_group(df_annotated, group_col="mass_quartile")
    ks = ks_test_mass_quartiles(df, per_galaxy, n_quartiles=n_quartiles)

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        per_galaxy.to_csv(p / "g0_per_galaxy.csv", index=False)
        group_fits.to_csv(p / "g0_by_mass_quartile.csv", index=False)

    return {
        "per_galaxy": per_galaxy,
        "stats": stats,
        "group_fits": group_fits,
        "ks": ks,
    }


def print_universality_summary(results: dict) -> None:
    """Print a human-readable universality analysis summary."""
    stats = results["stats"]
    ks = results["ks"]
    per_galaxy = results["per_galaxy"]
    group_fits = results["group_fits"]

    print("=" * 65)
    print("g₀ UNIVERSALITY ANALYSIS")
    print("=" * 65)
    print(f"Galaxies fitted:      {stats['n_galaxies']}")
    print()
    print("--- Per-galaxy g₀ distribution ---")
    print(f"  Median g₀          = {stats['g0_median']:.4e} m/s²")
    print(f"  Std g₀             = {stats['g0_std']:.4e} m/s²")
    print(f"  Scatter log10(g₀)  = {stats['log10_g0_std']:.3f} dex")
    print(f"  g₀ [16, 84]%       = [{stats['g0_p16']:.4e}, {stats['g0_p84']:.4e}] m/s²")
    print(f"  Reference g₀       = {stats['reference_g0']:.4e} m/s²")
    print(f"  Ref within ±1σ?    = {'YES ✓' if stats['ref_within_1sigma'] else 'NO ✗'}")
    print()
    print("--- Per-mass-quartile g₀ fits ---")
    for _, row in group_fits.iterrows():
        print(
            f"  Quartile {int(row['group'])}: "
            f"g₀={row['g0']:.4e} ±{row['g0_err']:.2e} m/s²"
            f"  ({int(row['n_galaxies'])} galaxies, {int(row['n_points'])} pts)"
        )
    print()
    print("--- KS test (low-mass vs high-mass quartile) ---")
    if ks["ks_pvalue"] is not None and not np.isnan(ks["ks_pvalue"]):
        print(f"  KS statistic       = {ks['ks_stat']:.4f}")
        print(f"  p-value            = {ks['ks_pvalue']:.4f}")
        print(f"  Δ log10(g₀)        = {ks['delta_log10_g0']:+.3f} dex")
        verdict = "UNIVERSAL (p > 0.05)" if ks["is_universal"] else "NON-UNIVERSAL (p ≤ 0.05)"
        print(f"  Verdict            = {verdict}")
    else:
        print("  Insufficient data for KS test.")
    print("=" * 65)
