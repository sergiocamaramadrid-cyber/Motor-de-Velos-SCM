"""
scm_analysis.py — Analysis pipeline for the SPARC RAR dataset.

Typical usage
-------------
    python -m src.scm_analysis --data-dir data/ --out results/

or programmatically:

    from src.scm_analysis import run_pipeline
    results = run_pipeline("data/sparc_rar_sample.csv")
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scm_models import fit_g0, bin_rar, deep_regime_slope, G0_DEFAULT


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_sparc_csv(path: str | Path) -> pd.DataFrame:
    """Load a SPARC RAR CSV file.

    Accepts columns ``g_bar``/``g_obs`` (linear, m/s²) or
    ``log_g_bar``/``log_g_obs`` (log10 dex).  Converts log columns to linear.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    pd.DataFrame with at minimum columns ``g_bar`` and ``g_obs``.
    """
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    if "log_g_bar" in cols and "log_g_obs" in cols:
        df["g_bar"] = 10.0 ** df["log_g_bar"]
        df["g_obs"] = 10.0 ** df["log_g_obs"]
    elif "g_bar" not in cols or "g_obs" not in cols:
        raise ValueError(
            f"CSV must contain 'g_bar'/'g_obs' or 'log_g_bar'/'log_g_obs'. "
            f"Found: {cols}"
        )

    df = df.dropna(subset=["g_bar", "g_obs"])
    df = df[(df["g_bar"] > 0) & (df["g_obs"] > 0)]
    return df.reset_index(drop=True)


def run_pipeline(
    csv_path: str | Path,
    n_bins: int = 20,
    deep_percentile: float = 20.0,
    g0_init: float = G0_DEFAULT,
) -> dict:
    """Run the full RAR analysis pipeline.

    Steps
    -----
    1. Load and inspect the CSV.
    2. Fit g0 (acceleration scale).
    3. Bin the data in log(g_bar) space.
    4. Diagnose the deep-regime slope.

    Parameters
    ----------
    csv_path : str or Path
        Path to the SPARC RAR CSV.
    n_bins : int, optional
        Number of log-bins for the binned RAR.
    deep_percentile : float, optional
        Fraction (%) of lowest g_bar points to treat as deep regime.
    g0_init : float, optional
        Initial guess for g0 in m/s².

    Returns
    -------
    dict with keys:
        ``df``          — cleaned DataFrame
        ``fit``         — g0 fit results dict
        ``bins``        — binned RAR DataFrame
        ``deep``        — deep-regime diagnostics dict
    """
    df = load_sparc_csv(csv_path)

    fit = fit_g0(df["g_bar"].values, df["g_obs"].values, g0_init=g0_init)
    bins = bin_rar(df["g_bar"].values, df["g_obs"].values, n_bins=n_bins)
    deep = deep_regime_slope(
        df["g_bar"].values, df["g_obs"].values,
        g0=fit["g0"], percentile=deep_percentile,
    )

    return {"df": df, "fit": fit, "bins": bins, "deep": deep}


def print_summary(results: dict) -> None:
    """Print a human-readable summary of pipeline results."""
    fit = results["fit"]
    deep = results["deep"]
    df = results["df"]

    print("=" * 60)
    print("SPARC RAR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Data points (clean):  {len(df)}")
    print(f"Galaxies:             {df['galaxy'].nunique() if 'galaxy' in df.columns else 'N/A'}")
    print()
    print("--- g0 fit ---")
    print(f"  g0        = {fit['g0']:.4e} m/s²")
    print(f"  g0_err    = {fit['g0_err']:.4e} m/s²")
    print(f"  RMS (dex) = {fit['rms']:.4f}")
    print(f"  N points  = {fit['n']}")
    print()
    print("--- Deep regime (low g_bar) ---")
    print(f"  Points in deep regime : {deep['n_deep']}")
    print(f"  Measured slope        : {deep['slope']:.3f}")
    print(f"  Expected slope (MOND) : {deep['expected_slope']:.3f}")
    print(f"  Regime collapses?     : {'YES ✓' if deep['collapses'] else 'NO ✗'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run SPARC RAR analysis: fit g0, bin, check deep regime."
    )
    parser.add_argument(
        "--csv",
        default="data/sparc_rar_sample.csv",
        help="Path to SPARC RAR CSV (default: data/sparc_rar_sample.csv)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=20,
        help="Number of log-bins (default: 20)",
    )
    parser.add_argument(
        "--deep-pct", type=float, default=20.0,
        help="Percentile threshold for deep regime (default: 20)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Directory to save results (optional)",
    )
    args = parser.parse_args()

    results = run_pipeline(args.csv, n_bins=args.n_bins, deep_percentile=args.deep_pct)
    print_summary(results)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        results["bins"].to_csv(out / "rar_bins.csv", index=False)
        fit = results["fit"]
        deep = results["deep"]
        summary_lines = [
            f"g0={fit['g0']:.6e}",
            f"g0_err={fit['g0_err']:.6e}",
            f"rms_dex={fit['rms']:.6f}",
            f"n={fit['n']}",
            f"deep_slope={deep['slope']:.6f}",
            f"deep_collapses={deep['collapses']}",
        ]
        (out / "rar_summary.txt").write_text("\n".join(summary_lines) + "\n")
        print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    _cli()
