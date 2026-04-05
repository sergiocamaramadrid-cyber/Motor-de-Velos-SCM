#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_env_analysis.py — Environmental OLS analysis on the SPARC sample.

Merges the per-galaxy F3 catalog with the SPARC global table, derives an
HI-based environmental proxy, fits OLS models (base and full), and tests
whether the proxy explains the outer-slope residuals via Spearman correlation.

Inputs
------
f3_catalog.csv
    Required columns: ``galaxy`` (or ``Galaxy``), ``F3`` (or ``friction_slope``
    / ``beta``), ``n_deep`` (or ``n_tail_points``).
sparc_basic.csv
    Required columns: ``Galaxy`` (or ``galaxy``), ``Inc``, ``L36``, ``MHI``,
    ``Rdisk`` (or ``Re``).

Outputs (written to --out directory)
--------------------------------------
galaxy_catalog_with_env.csv
    Merged per-galaxy table with all derived columns.
summary.txt
    Plain-text summary: N, Spearman ρ, p-value, ΔAIC, OLS env coefficient.

Usage
-----
::

    python scripts/run_env_analysis.py \\
        --f3-catalog  data/f3_catalog_real.csv \\
        --sparc-basic data/sparc_basic.csv

    python scripts/run_env_analysis.py \\
        --f3-catalog  data/f3_catalog_real.csv \\
        --sparc-basic data/sparc_basic.csv \\
        --out         results/env_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default mass-to-light ratio at 3.6 μm (McGaugh & Schombert 2015)
UPSILON_36: float = 0.5  # Msun / Lsun

# Helium correction factor for total gas mass
HE_CORRECTION: float = 1.33

# Unit scale: L36 and MHI are in 1e9 Lsun / Msun
UNIT_SCALE: float = 1.0e9

# Default F3 reference value for delta_f3 = F3 - F3_REF
F3_REF: float = 0.5


# ---------------------------------------------------------------------------
# Column-name resolution helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in *df*, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_galaxy_key(df: pd.DataFrame) -> pd.Series:
    """Return the galaxy-name column as a stripped string Series."""
    col = _resolve_column(df, ["galaxy", "Galaxy", "name", "galname"])
    if col is None:
        raise ValueError(
            f"No galaxy-name column found. Columns present: {list(df.columns)}"
        )
    return df[col].astype(str).str.strip()


def _resolve_beta(df: pd.DataFrame) -> pd.Series:
    """Return the β / friction_slope column (prefer F3, then friction_slope, beta)."""
    col = _resolve_column(df, ["F3", "friction_slope", "beta", "slope_tail"])
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _resolve_n_deep(df: pd.DataFrame) -> pd.Series:
    """Return the n_deep / n_tail_points column."""
    col = _resolve_column(df, ["n_deep", "n_tail_points", "n_deep_points"])
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _resolve_rdisk(df_sparc: pd.DataFrame) -> pd.Series:
    """Return the disk-radius column (Rdisk, Re, or r_eff)."""
    col = _resolve_column(df_sparc, ["Rdisk", "Re", "r_eff", "re"])
    if col is None:
        return pd.Series(np.nan, index=df_sparc.index)
    return pd.to_numeric(df_sparc[col], errors="coerce")


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------

def load_data(f3_path: str | Path, sparc_path: str | Path) -> pd.DataFrame:
    """Load F3 catalog and SPARC basic table and merge them.

    Parameters
    ----------
    f3_path : path-like
        F3 catalog CSV.
    sparc_path : path-like
        SPARC basic table CSV.

    Returns
    -------
    pd.DataFrame
        Merged table with columns:
        ``galaxy_id, slope_tail, n_tail_points, inc_deg, logM, Rmax, MHI, Rdisk``.
    """
    df_f3 = pd.read_csv(f3_path)
    df_sp = pd.read_csv(sparc_path)

    # --- F3 side ---
    f3_clean = pd.DataFrame({
        "galaxy_id": _resolve_galaxy_key(df_f3),
        "slope_tail": _resolve_beta(df_f3),
        "n_tail_points": _resolve_n_deep(df_f3),
    })

    # --- SPARC side ---
    L36 = pd.to_numeric(df_sp.get("L36", pd.Series(dtype=float)), errors="coerce")
    MHI = pd.to_numeric(df_sp.get("MHI", pd.Series(dtype=float)), errors="coerce")

    inc_col = _resolve_column(df_sp, ["Inc", "inc", "Inc_deg"])
    inc_vals = (
        pd.to_numeric(df_sp[inc_col], errors="coerce")
        if inc_col else pd.Series(np.nan, index=df_sp.index)
    )

    M_bar = UPSILON_36 * L36 * UNIT_SCALE + HE_CORRECTION * MHI * UNIT_SCALE
    logM = np.where(M_bar > 0, np.log10(M_bar), np.nan)

    Rdisk = _resolve_rdisk(df_sp)

    sp_clean = pd.DataFrame({
        "galaxy_id": _resolve_galaxy_key(df_sp),
        "inc_deg": inc_vals,
        "logM": logM,
        "Rmax": Rdisk,
        "MHI": MHI,
        "Rdisk": Rdisk,
    })

    df = f3_clean.merge(sp_clean, on="galaxy_id", how="left")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def compute_env_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_f3`` and ``env_proxy`` columns to *df* (in-place copy).

    ``delta_f3  = slope_tail - F3_REF``
    ``env_proxy = log10(MHI) - 2 * log10(Rdisk)``

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new columns.
    """
    df = df.copy()
    df["delta_f3"] = df["slope_tail"] - F3_REF
    df["env_proxy"] = np.log10(df["MHI"]) - 2 * np.log10(df["Rdisk"])
    return df


# ---------------------------------------------------------------------------
# OLS models
# ---------------------------------------------------------------------------

def run_ols(df: pd.DataFrame) -> tuple[Any, Any, pd.DataFrame]:
    """Fit base and full OLS models and add a residual column.

    The base model regresses ``delta_f3`` on ``logM`` and ``Rmax``.
    The full model adds ``env_proxy`` as a third predictor.

    Both models are fitted with HC3 heteroskedasticity-robust standard errors
    (``cov_type="HC3"``), consistent with the robustness-analysis pipeline.

    Only rows with no NaN in any required column are used.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_env_proxy`.

    Returns
    -------
    model_base : statsmodels RegressionResultsWrapper
    model_full : statsmodels RegressionResultsWrapper
    df_fit : pd.DataFrame
        Subset of *df* used for fitting, with an added ``residual`` column.
    """
    required = ["delta_f3", "logM", "Rmax", "env_proxy"]
    df_fit = df.loc[df[required].notna().all(axis=1)].copy()

    X_base = sm.add_constant(df_fit[["logM", "Rmax"]])
    model_base = sm.OLS(df_fit["delta_f3"], X_base).fit(cov_type="HC3")
    df_fit["residual"] = df_fit["delta_f3"] - model_base.predict(X_base)

    X_full = sm.add_constant(df_fit[["logM", "Rmax", "env_proxy"]])
    model_full = sm.OLS(df_fit["delta_f3"], X_full).fit(cov_type="HC3")

    return model_base, model_full, df_fit.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(df_fit: pd.DataFrame, model_base: Any, model_full: Any) -> dict:
    """Return a dictionary of key statistics.

    Parameters
    ----------
    df_fit : pd.DataFrame
        Output of :func:`run_ols` (contains ``residual`` and ``env_proxy``).
    model_base : statsmodels result
        Base OLS model.
    model_full : statsmodels result
        Full OLS model.

    Returns
    -------
    dict with keys: N, rho, p, delta_aic, coef_env, p_env
    """
    rho, p = spearmanr(df_fit["residual"], df_fit["env_proxy"])
    delta_aic = model_base.aic - model_full.aic
    return {
        "N": int(len(df_fit)),
        "rho": float(rho),
        "p": float(p),
        "delta_aic": float(delta_aic),
        "coef_env": float(model_full.params["env_proxy"]),
        "p_env": float(model_full.pvalues["env_proxy"]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    stats: dict,
    out_dir: str | Path = "results/env_analysis",
) -> None:
    """Write results to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged table (output of :func:`run_ols`).
    stats : dict
        Output of :func:`compute_stats`.
    out_dir : path-like
        Output directory (created if missing).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "galaxy_catalog_with_env.csv", index=False)

    summary_lines = [
        "===== RESULTADOS =====",
        f"N = {stats['N']}",
        f"rho = {stats['rho']:.3f}",
        f"p = {stats['p']:.4f}",
        f"ΔAIC = {stats['delta_aic']:.3f}",
        f"coef_env = {stats['coef_env']:.4f}",
        f"p_env = {stats['p_env']:.4f}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summary_text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Environmental OLS analysis on the SPARC sample."
    )
    parser.add_argument(
        "--f3-catalog",
        required=True,
        dest="f3_catalog",
        metavar="FILE",
        help="F3 catalog CSV (columns: galaxy, F3/friction_slope/beta, n_deep).",
    )
    parser.add_argument(
        "--sparc-basic",
        required=True,
        dest="sparc_basic",
        metavar="FILE",
        help="SPARC basic table CSV (columns: Galaxy, Inc, L36, MHI, Rdisk/Re).",
    )
    parser.add_argument(
        "--out",
        default="results/env_analysis",
        metavar="DIR",
        help="Output directory (default: results/env_analysis).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for CLI usage."""
    args = _parse_args(argv)

    df = load_data(args.f3_catalog, args.sparc_basic)
    df = compute_env_proxy(df)
    model_base, model_full, df_fit = run_ols(df)
    stats = compute_stats(df_fit, model_base, model_full)
    save_outputs(df_fit, stats, args.out)

    print("\n===== RESULTADOS =====")
    print(f"N = {stats['N']}")
    print(f"rho = {stats['rho']:.3f}")
    print(f"p = {stats['p']:.4f}")
    print(f"ΔAIC = {stats['delta_aic']:.3f}")
    print(f"coef_env = {stats['coef_env']:.4f}")
    print(f"p_env = {stats['p_env']:.4f}")
    print(f"\nResultados guardados en: {args.out}")


if __name__ == "__main__":
    main()
