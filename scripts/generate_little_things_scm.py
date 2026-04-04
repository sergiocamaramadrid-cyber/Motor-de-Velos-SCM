"""
scripts/generate_little_things_scm.py — SCM F3 analysis for LITTLE THINGS galaxies.

Applies the F3 (friction slope) measurement to the LITTLE THINGS dwarf-galaxy
sample using global photometric and kinematic properties.

For each galaxy the deep-regime slope β is estimated from the single global
(g_bar, g_obs) data point via:

    F3 = (log10(g_obs) − 0.5 · log10(a0)) / log10(g_bar)

which equals 0.5 exactly when g_obs = √(g_bar · a0)  (deep MOND / SCM
prediction).  The residual δF3 = F3 − 0.5 quantifies the deviation from the
SCM prediction.

g_obs is derived from the flat rotation velocity Vlast and the effective radius
R_eff = j / Vlast  (specific angular momentum divided by flat velocity).

Usage
-----
::

    python scripts/generate_little_things_scm.py

    python scripts/generate_little_things_scm.py \\
        --csv data/little_things_global.csv \\
        --out results_little_things_scm \\
        --a0 1.2e-10 \\
        --deep-threshold 0.3

Outputs written to --out DIR (default: results_little_things_scm):
  little_things_scm_catalog.csv   — full per-galaxy F3 catalog
  scm_clean_sample.csv            — reliable-fit subset
  scm_clean_with_residual.csv     — clean subset with explicit δF3 residual
  summary.json                    — aggregate statistics
  faseA_f3_vs_vlast.png           — Phase-A diagnostic: F3 vs log10(Vlast)
  scatter_f3_vlast.png            — scatter: F3 vs log10(Vlast)
  hist_f3.png                     — histogram of F3 values
  hist_delta_f3.png               — histogram of δF3 values
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.stats import spearmanr as _spearmanr
from scipy.stats import linregress as _linregress

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

KPC_TO_M: float = 3.085677581e19   # metres per kiloparsec (IAU 2012)
KMS_TO_MS: float = 1.0e3           # m/s per km/s

A0_DEFAULT: float = 1.2e-10        # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT: float = 0.3  # deep regime: g_bar < threshold × a0

EXPECTED_F3_MOND: float = 0.5      # expected friction slope in deep-MOND limit

# Required columns in the input CSV
REQUIRED_COLS: list[str] = ["galaxy_id", "logM", "logVobs", "log_gbar", "log_j"]

# Constant: log10(KPC_TO_M)
_LOG10_KPC_TO_M: float = float(np.log10(KPC_TO_M))


# ---------------------------------------------------------------------------
# Per-galaxy F3 computation
# ---------------------------------------------------------------------------

def compute_log_gobs(
    logVobs: float | np.ndarray,
    log_j: float | np.ndarray,
) -> float | np.ndarray:
    """Compute log10(g_obs / m·s⁻²) from flat velocity and specific angular momentum.

    Derivation (SI throughout):

        R_eff [kpc] = j [kpc·km/s] / Vlast [km/s]
                    = 10^log_j / 10^logVobs = 10^(log_j - logVobs)

        g_obs [m/s²] = Vlast² [m/s]² / R_eff [m]
                     = (10^logVobs × KMS_TO_MS)² / (10^(log_j−logVobs) × KPC_TO_M)

        log10(g_obs) = 2·logVobs + log10(KMS_TO_MS²)
                     − (log_j − logVobs) − log10(KPC_TO_M)
                     = 3·logVobs − log_j + 6 − log10(KPC_TO_M)

    Parameters
    ----------
    logVobs : float or array
        log10(Vlast / km·s⁻¹).
    log_j : float or array
        log10(j / kpc·km·s⁻¹).

    Returns
    -------
    float or ndarray
        log10(g_obs / m·s⁻²).
    """
    return 3.0 * np.asarray(logVobs, dtype=float) - np.asarray(log_j, dtype=float) + 6.0 - _LOG10_KPC_TO_M


def compute_f3(
    log_gobs: float | np.ndarray,
    log_gbar: float | np.ndarray,
    a0: float = A0_DEFAULT,
) -> float | np.ndarray:
    """Compute the per-galaxy F3 (friction slope) proxy from a single global point.

    Definition:

        F3 = (log10(g_obs) − 0.5·log10(a0)) / log10(g_bar)

    This equals exactly 0.5 when g_obs = √(g_bar · a0)  (deep-MOND/SCM
    prediction).  Galaxies above this line give F3 > 0.5; below give F3 < 0.5.

    Parameters
    ----------
    log_gobs : float or array
        log10(g_obs / m·s⁻²).
    log_gbar : float or array
        log10(g_bar / m·s⁻²).
    a0 : float
        Characteristic acceleration in m/s².

    Returns
    -------
    float or ndarray
        F3 friction-slope proxy.
    """
    log_a0 = np.log10(a0)
    return (np.asarray(log_gobs, dtype=float) - 0.5 * log_a0) / np.asarray(log_gbar, dtype=float)


def compute_reliable(
    log_gbar: float | np.ndarray,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> bool | np.ndarray:
    """Flag galaxies that lie in the deep MOND regime (g_bar < threshold × a0).

    Parameters
    ----------
    log_gbar : float or array
        log10(g_bar / m·s⁻²).
    a0 : float
        Characteristic acceleration in m/s².
    deep_threshold : float
        Fraction of a0 defining the deep regime.

    Returns
    -------
    bool or ndarray of bool
    """
    threshold_log = np.log10(deep_threshold * a0)
    return np.asarray(log_gbar, dtype=float) < threshold_log


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

def build_catalog(
    df: pd.DataFrame,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """Compute F3 columns and return the full SCM catalog DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with required columns (galaxy_id, logM, logVobs,
        log_gbar, log_j).
    a0 : float
        Characteristic acceleration in m/s².
    deep_threshold : float
        Deep-regime threshold as fraction of a0.

    Returns
    -------
    pd.DataFrame
        Full catalog with added columns:
        log_gobs, friction_slope, friction_slope_err,
        delta_F3, reliable, velo_inerte_flag.
    """
    cat = df.copy()
    cat["log_gobs"] = compute_log_gobs(cat["logVobs"].values, cat["log_j"].values)
    cat["friction_slope"] = compute_f3(
        cat["log_gobs"].values, cat["log_gbar"].values, a0=a0
    )
    cat["friction_slope_err"] = float("nan")
    cat["delta_F3"] = cat["friction_slope"] - EXPECTED_F3_MOND
    cat["reliable"] = compute_reliable(cat["log_gbar"].values, a0=a0, deep_threshold=deep_threshold)
    cat["velo_inerte_flag"] = cat["reliable"]
    return cat


# ---------------------------------------------------------------------------
# Mass-correlation and detrending
# ---------------------------------------------------------------------------

def compute_mass_detrend(cat: pd.DataFrame) -> pd.DataFrame:
    """Add *f3_mass_residual* column: F3 after removing the logVobs linear trend.

    For galaxies that are marked reliable the OLS fit of *friction_slope* on
    *logVobs* is computed and the residuals stored in ``f3_mass_residual``.
    Non-reliable galaxies receive NaN.

    The OLS coefficients are returned via the ``_ols_slope`` and
    ``_ols_intercept`` attributes attached to the returned DataFrame for use
    by callers (e.g. figure functions).

    Parameters
    ----------
    cat : pd.DataFrame
        Catalog as returned by *build_catalog*.  Must contain columns
        ``friction_slope``, ``logVobs``, and ``reliable``.

    Returns
    -------
    pd.DataFrame
        Copy of *cat* with an added ``f3_mass_residual`` column.
    """
    cat = cat.copy()
    cat["f3_mass_residual"] = float("nan")

    clean_mask = cat["reliable"] & cat["friction_slope"].notna()
    if clean_mask.sum() >= 2:
        f3 = cat.loc[clean_mask, "friction_slope"].values
        logV = cat.loc[clean_mask, "logVobs"].values
        slope, intercept, *_ = _linregress(logV, f3)
        residuals = f3 - (slope * logV + intercept)
        cat.loc[clean_mask, "f3_mass_residual"] = residuals
        cat.attrs["_ols_slope"] = slope
        cat.attrs["_ols_intercept"] = intercept
    else:
        cat.attrs["_ols_slope"] = float("nan")
        cat.attrs["_ols_intercept"] = float("nan")

    return cat


def compute_mass_correlation_stats(cat: pd.DataFrame) -> dict:
    """Compute Spearman correlations between F3 and logVobs (mass proxy).

    Two-step analysis:

    1. **Raw** Spearman ρ(F3, logVobs) — shows that the outer slope carries
       a mass dependence.

    2. **Rank-detrended residual**: the OLS of *rank(F3)* on *rank(logVobs)*
       is fitted; the rank residuals are then correlated (Spearman) with
       logVobs.  This partial-Spearman approach removes the monotone mass
       trend; if the residual ρ is close to zero the mass dependence fully
       accounts for the F3 variation.

    Parameters
    ----------
    cat : pd.DataFrame
        Catalog containing ``friction_slope``, ``logVobs``, and ``reliable``.

    Returns
    -------
    dict with keys:
        spearman_f3_vlast_rho    — raw Spearman ρ
        spearman_f3_vlast_p      — raw p-value
        ols_slope                — OLS slope (value-space, F3 ~ a·logVobs + b)
        ols_intercept            — OLS intercept
        spearman_resid_vlast_rho — rank-detrended residual Spearman ρ
        spearman_resid_vlast_p   — rank-detrended residual p-value
    """
    clean = cat[cat["reliable"] & cat["friction_slope"].notna()]
    n = len(clean)
    if n < 4:
        return {
            "spearman_f3_vlast_rho": float("nan"),
            "spearman_f3_vlast_p": float("nan"),
            "ols_slope": float("nan"),
            "ols_intercept": float("nan"),
            "spearman_resid_vlast_rho": float("nan"),
            "spearman_resid_vlast_p": float("nan"),
        }

    f3 = clean["friction_slope"].values
    logV = clean["logVobs"].values

    rho_raw, p_raw = _spearmanr(f3, logV)

    slope, intercept, *_ = _linregress(logV, f3)

    # Rank-detrend: regress rank(F3) on rank(logVobs), compute rank residuals,
    # then measure their Spearman correlation with logVobs (partial Spearman).
    r_f3 = pd.Series(f3).rank().values
    r_v = pd.Series(logV).rank().values
    slope_r, intercept_r, *_ = _linregress(r_v, r_f3)
    resid_r = r_f3 - (slope_r * r_v + intercept_r)
    rho_resid, p_resid = _spearmanr(resid_r, logV)

    return {
        "spearman_f3_vlast_rho": float(rho_raw),
        "spearman_f3_vlast_p": float(p_raw),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "spearman_resid_vlast_rho": float(rho_resid),
        "spearman_resid_vlast_p": float(p_resid),
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(cat: pd.DataFrame) -> dict:
    """Compute aggregate F3 statistics from the catalog.

    Parameters
    ----------
    cat : pd.DataFrame
        Full catalog as returned by *build_catalog*.

    Returns
    -------
    dict
        Summary with keys: n_galaxies, n_reliable, f3_mean, f3_median,
        f3_std, delta_f3_mean, delta_f3_median, delta_f3_std,
        t_stat, p_value_ttest, consistent_mond, plus mass-correlation
        keys from *compute_mass_correlation_stats*.
    """
    clean = cat[cat["reliable"]]["friction_slope"].dropna()
    n_galaxies = int(len(cat))
    n_reliable = int(len(clean))

    f3_mean = float(clean.mean()) if n_reliable > 0 else float("nan")
    f3_median = float(clean.median()) if n_reliable > 0 else float("nan")
    f3_std = float(clean.std()) if n_reliable > 0 else float("nan")

    delta = cat[cat["reliable"]]["delta_F3"].dropna()
    delta_mean = float(delta.mean()) if n_reliable > 0 else float("nan")
    delta_median = float(delta.median()) if n_reliable > 0 else float("nan")
    delta_std = float(delta.std()) if n_reliable > 0 else float("nan")

    if n_reliable >= 2:
        t_result = _scipy_stats.ttest_1samp(clean.values, EXPECTED_F3_MOND)
        t_stat = float(t_result.statistic)
        p_value = float(t_result.pvalue)
    else:
        t_stat = float("nan")
        p_value = float("nan")

    consistent = (p_value > 0.05) if not np.isnan(p_value) else False

    corr_stats = compute_mass_correlation_stats(cat)

    return {
        "n_galaxies": n_galaxies,
        "n_reliable": n_reliable,
        "f3_mean": f3_mean,
        "f3_median": f3_median,
        "f3_std": f3_std,
        "delta_f3_mean": delta_mean,
        "delta_f3_median": delta_median,
        "delta_f3_std": delta_std,
        "t_stat": t_stat,
        "p_value_ttest": p_value,
        "consistent_mond": consistent,
        **corr_stats,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_faseA_f3_vs_vlast(cat: pd.DataFrame, out_path: Path) -> None:
    """Phase-A diagnostic: F3 vs log10(Vlast), annotated with SCM prediction and
    OLS mass-trend line + Spearman ρ."""
    reliable = cat["reliable"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("white")

    ax.scatter(
        cat.loc[reliable, "logVobs"],
        cat.loc[reliable, "friction_slope"],
        s=40, color="steelblue", edgecolors="none", alpha=0.85,
        label="Reliable (deep regime)",
        zorder=3,
    )
    not_rel = ~reliable
    if not_rel.any():
        ax.scatter(
            cat.loc[not_rel, "logVobs"],
            cat.loc[not_rel, "friction_slope"],
            s=40, color="gray", edgecolors="none", alpha=0.6,
            marker="^", label="Not reliable",
            zorder=3,
        )

    ax.axhline(EXPECTED_F3_MOND, color="tomato", linewidth=1.5, linestyle="--",
               label=f"SCM prediction (F3 = {EXPECTED_F3_MOND})", zorder=2)

    # OLS mass-trend line and Spearman annotation (reliable galaxies only)
    corr = compute_mass_correlation_stats(cat)
    if not np.isnan(corr["ols_slope"]):
        logV_clean = cat.loc[reliable, "logVobs"].values
        x_fit = np.linspace(logV_clean.min(), logV_clean.max(), 100)
        y_fit = corr["ols_slope"] * x_fit + corr["ols_intercept"]
        ax.plot(x_fit, y_fit, color="darkorange", linewidth=1.5, linestyle="-",
                label="OLS mass trend", zorder=2)
        rho_str = f"ρ = {corr['spearman_f3_vlast_rho']:.2f}"
        p_str = f"p = {corr['spearman_f3_vlast_p']:.3f}"
        ax.text(0.97, 0.95, f"{rho_str}, {p_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="darkorange",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel(r"$\log_{10}(V_{\rm last}\,/\,\rm km\,s^{-1})$", fontsize=11)
    ax.set_ylabel(r"$F_3$ (friction slope)", fontsize=11)
    ax.set_title("Phase A — F3 vs $V_{\\rm last}$ (LITTLE THINGS)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_scatter_f3_vlast(cat: pd.DataFrame, out_path: Path) -> None:
    """Simple scatter plot: F3 vs log10(Vlast) with Spearman ρ annotation."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor("white")

    ax.scatter(
        cat["logVobs"], cat["friction_slope"],
        s=30, color="black", edgecolors="none", alpha=0.75,
        zorder=3,
    )
    ax.axhline(EXPECTED_F3_MOND, color="tomato", linewidth=1.2, linestyle="--",
               label=f"F3 = {EXPECTED_F3_MOND}", zorder=2)

    corr = compute_mass_correlation_stats(cat)
    if not np.isnan(corr["spearman_f3_vlast_rho"]):
        rho_str = f"ρ = {corr['spearman_f3_vlast_rho']:.2f}"
        p_str = f"p = {corr['spearman_f3_vlast_p']:.3f}"
        ax.text(0.97, 0.95, f"Spearman {rho_str}, {p_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel(r"$\log_{10}(V_{\rm last}\,/\,\rm km\,s^{-1})$", fontsize=10)
    ax.set_ylabel(r"$F_3$", fontsize=10)
    ax.set_title("F3 vs $V_{\\rm last}$  (LITTLE THINGS)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_hist_f3(cat: pd.DataFrame, out_path: Path) -> None:
    """Histogram of F3 (friction slope) values — reliable galaxies."""
    values = cat[cat["reliable"]]["friction_slope"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor("white")

    ax.hist(values, bins=10, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(EXPECTED_F3_MOND, color="tomato", linewidth=1.5, linestyle="--",
               label=f"F3 = {EXPECTED_F3_MOND} (MOND)")
    ax.set_xlabel(r"$F_3$ (friction slope)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(r"Distribution of $F_3$ — LITTLE THINGS", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_hist_delta_f3(cat: pd.DataFrame, out_path: Path) -> None:
    """Histogram of δF3 = F3 − 0.5 values — reliable galaxies."""
    values = cat[cat["reliable"]]["delta_F3"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor("white")

    ax.hist(values, bins=10, color="mediumseagreen", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="tomato", linewidth=1.5, linestyle="--",
               label=r"$\delta F_3 = 0$ (MOND)")
    ax.set_xlabel(r"$\delta F_3 = F_3 - 0.5$", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(r"Distribution of $\delta F_3$ — LITTLE THINGS", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_little_things_scm(
    csv_path: Path,
    out_dir: Path,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    no_figures: bool = False,
) -> dict:
    """Run the LITTLE THINGS SCM F3 analysis pipeline.

    Parameters
    ----------
    csv_path : Path
        Input CSV with columns: galaxy_id, logM, logVobs, log_gbar, log_j.
    out_dir : Path
        Directory for all outputs.
    a0 : float
        Characteristic acceleration in m/s².
    deep_threshold : float
        Deep-regime threshold as fraction of a0.
    no_figures : bool
        Skip figure generation when True (useful for testing).

    Returns
    -------
    dict
        Summary statistics.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build full catalog and add mass-detrended residuals
    cat = build_catalog(df, a0=a0, deep_threshold=deep_threshold)
    cat = compute_mass_detrend(cat)

    # Full catalog
    cat.to_csv(out_dir / "little_things_scm_catalog.csv", index=False)

    # Clean sample (reliable fits only)
    clean = cat[cat["reliable"]].reset_index(drop=True)
    clean.to_csv(out_dir / "scm_clean_sample.csv", index=False)

    # Clean sample with explicit residual column (delta_F3 already present;
    # write as a dedicated file for downstream consumers)
    clean_res = clean.copy()
    clean_res["residual"] = clean_res["delta_F3"]
    clean_res.to_csv(out_dir / "scm_clean_with_residual.csv", index=False)

    # Summary
    summary = compute_summary(cat)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: None if np.isnan(x) else x),
        encoding="utf-8",
    )

    # Figures
    if not no_figures:
        _save_faseA_f3_vs_vlast(cat, out_dir / "faseA_f3_vs_vlast.png")
        _save_scatter_f3_vlast(cat, out_dir / "scatter_f3_vlast.png")
        _save_hist_f3(cat, out_dir / "hist_f3.png")
        _save_hist_delta_f3(cat, out_dir / "hist_delta_f3.png")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCM F3 analysis for LITTLE THINGS dwarf galaxies. "
            "Computes per-galaxy friction slope F3 from global kinematic data "
            "and generates catalog, clean sample, residuals, summary, and figures."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent.parent / "data" / "little_things_global.csv"),
        metavar="FILE",
        help="Input dataset CSV (default: data/little_things_global.csv).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "results_little_things_scm"),
        metavar="DIR",
        help="Output directory (default: results_little_things_scm).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=f"Deep-regime threshold as fraction of a0 (default: {DEEP_THRESHOLD_DEFAULT}).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        dest="no_figures",
        help="Skip figure generation.",
    )
    return parser.parse_args(argv)


def _print_summary(summary: dict) -> None:
    sep = "=" * 70
    print(sep)
    print("  Motor de Velos SCM — LITTLE THINGS F3 Analysis")
    print(sep)
    print(f"  Galaxies total  : {summary['n_galaxies']}")
    print(f"  Reliable (deep) : {summary['n_reliable']}")
    print()
    if summary["n_reliable"] > 0:
        print(f"  F3 mean         : {summary['f3_mean']:.4f}")
        print(f"  F3 median       : {summary['f3_median']:.4f}")
        print(f"  F3 std          : {summary['f3_std']:.4f}")
        print(f"  δF3 mean        : {summary['delta_f3_mean']:+.4f}")
        print(f"  Expected (MOND) : {EXPECTED_F3_MOND:.4f}")
        if not np.isnan(summary.get("t_stat", float("nan"))):
            print(f"  t-statistic     : {summary['t_stat']:+.4f}")
            print(f"  p-value         : {summary['p_value_ttest']:.4e}")
        print()
        if summary["consistent_mond"]:
            print(f"  ✅  Estado A — F3 consistent with {EXPECTED_F3_MOND} "
                  f"(p = {summary['p_value_ttest']:.3f} > 0.05)")
        else:
            print(f"  ⚠️  Estado B — F3 deviates from {EXPECTED_F3_MOND} "
                  f"(p = {summary['p_value_ttest']:.3e} < 0.05)")
        print()
        rho_raw = summary.get("spearman_f3_vlast_rho", float("nan"))
        p_raw   = summary.get("spearman_f3_vlast_p",   float("nan"))
        rho_res = summary.get("spearman_resid_vlast_rho", float("nan"))
        p_res   = summary.get("spearman_resid_vlast_p",   float("nan"))
        if not np.isnan(rho_raw):
            print(f"  Spearman ρ(F3, Vlast) : {rho_raw:+.3f}  (p = {p_raw:.3e})")
            print(f"  Spearman ρ(resid, Vlast) [mass-removed]: "
                  f"{rho_res:+.3f}  (p = {p_res:.3f})")
    print(sep)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: parse arguments, run pipeline, print summary."""
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    summary = run_little_things_scm(
        csv_path=csv_path,
        out_dir=out_dir,
        a0=args.a0,
        deep_threshold=args.deep_threshold,
        no_figures=args.no_figures,
    )

    _print_summary(summary)
    print(f"\n  Results written to: {out_dir}")
    return summary


if __name__ == "__main__":
    main()
