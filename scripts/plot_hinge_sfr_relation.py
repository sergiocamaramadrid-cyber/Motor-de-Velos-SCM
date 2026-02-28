#!/usr/bin/env python3
"""
plot_hinge_sfr_relation.py
==========================
Standalone script that generates the key paper figure:

    Residual log SFR (mass-corrected) vs mean hinge activation (F3)

How it works
------------
1. Fit OLS: log_sfr = a + b * log_mbar   (mass-only baseline)
2. Compute residuals: e_i = log_sfr_i − ŷ_i
3. Scatter-plot e_i vs F3_mean_H_ext
4. Overlay OLS best-fit line through the residuals

Inputs
------
hinge_features.csv   per-galaxy friction proxies (from hinge_sfr_test.py)
galaxy_table.csv     per-galaxy log_mbar, log_sfr

Outputs
-------
results/paper_figures/hinge_sfr_relation.png   (300 dpi raster)

Usage
-----
Default paths::

    python scripts/plot_hinge_sfr_relation.py

Custom paths::

    python scripts/plot_hinge_sfr_relation.py \\
        --features   results/hinge_sfr/hinge_features.csv \\
        --galaxy-table  data/hinge_sfr/galaxy_table.csv \\
        --out-dir    results/paper_figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for CI / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES = os.path.join("results", "hinge_sfr", "hinge_features.csv")
_DEFAULT_GALAXY_TABLE = os.path.join("data", "hinge_sfr", "galaxy_table.csv")
_DEFAULT_OUT_DIR = os.path.join("results", "paper_figures")
_OUT_FILENAME = "hinge_sfr_relation.png"


# ---------------------------------------------------------------------------
# Core functions (testable)
# ---------------------------------------------------------------------------

def load_data(features_path: str, galaxy_table_path: str) -> pd.DataFrame:
    """Merge hinge features with galaxy photometry table.

    Parameters
    ----------
    features_path : str
        Path to ``hinge_features.csv`` produced by ``hinge_sfr_test.py``.
    galaxy_table_path : str
        Path to ``galaxy_table.csv`` with at minimum columns
        ``galaxy``, ``log_mbar``, ``log_sfr``.

    Returns
    -------
    pd.DataFrame
        Merged table with columns ``galaxy``, ``log_mbar``, ``log_sfr``,
        and ``F3_mean_H_ext``.
    """
    features = pd.read_csv(features_path)
    gal = pd.read_csv(galaxy_table_path)
    df = gal.merge(features[["galaxy", "F3_mean_H_ext"]], on="galaxy", how="inner")
    return df


def compute_mass_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the mass-only OLS prediction from log SFR.

    Fits ``log_sfr ~ log_mbar`` and adds column ``logSFR_residual``
    containing ``log_sfr − ŷ``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``log_mbar`` and ``log_sfr``.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column ``logSFR_residual``.
    """
    df = df.copy()
    X_mass = sm.add_constant(df["log_mbar"])
    model_mass = sm.OLS(df["log_sfr"], X_mass).fit()
    df["logSFR_residual"] = df["log_sfr"] - model_mass.predict(X_mass)
    return df


def fit_hinge_relation(df: pd.DataFrame) -> dict:
    """Fit OLS of ``logSFR_residual ~ F3_mean_H_ext``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``logSFR_residual`` and ``F3_mean_H_ext``.

    Returns
    -------
    dict with keys:
        coef        float — slope coefficient
        pvalue      float — two-sided p-value for the slope
        x_fit       np.ndarray — 100-point x grid for the fit line
        y_fit       np.ndarray — fitted y values on that grid
        summary     str — full OLS summary string
    """
    x = df["F3_mean_H_ext"]
    y = df["logSFR_residual"]
    X_hinge = sm.add_constant(x)
    model = sm.OLS(y, X_hinge).fit()

    x_fit = np.linspace(float(x.min()), float(x.max()), 100)
    y_fit = model.params.iloc[0] + model.params.iloc[1] * x_fit

    return {
        "coef": float(model.params.iloc[1]),
        "pvalue": float(model.pvalues.iloc[1]),
        "x_fit": x_fit,
        "y_fit": y_fit,
        "summary": model.summary().as_text(),
    }


def make_figure(df: pd.DataFrame, fit: dict) -> plt.Figure:
    """Create scatter + OLS fit figure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``F3_mean_H_ext`` and ``logSFR_residual``.
    fit : dict
        Output of :func:`fit_hinge_relation`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        df["F3_mean_H_ext"],
        df["logSFR_residual"],
        alpha=0.7,
        s=25,
        color="steelblue",
        label="SPARC galaxies",
    )
    ax.plot(
        fit["x_fit"],
        fit["y_fit"],
        color="crimson",
        linewidth=1.8,
        label=f"OLS fit  (coef={fit['coef']:+.3f}, p={fit['pvalue']:.4f})",
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Mean hinge activation (F3)", fontsize=12)
    ax.set_ylabel("Residual log SFR (mass-corrected)", fontsize=12)
    ax.set_title("Hinge activation vs star formation rate (SPARC)", fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_dir: str) -> str:
    """Save *fig* as a 300-dpi PNG under *out_dir*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str
        Directory to write the file into (created if absent).

    Returns
    -------
    str
        Full path of the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, _OUT_FILENAME)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return outfile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate residual log SFR vs hinge activation figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features",
        default=_DEFAULT_FEATURES,
        help="Path to hinge_features.csv.",
    )
    parser.add_argument(
        "--galaxy-table",
        dest="galaxy_table",
        default=_DEFAULT_GALAXY_TABLE,
        help="Path to galaxy_table.csv.",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=_DEFAULT_OUT_DIR,
        help="Output directory for the figure.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Entry point."""
    args = _parse_args(argv)
    df = load_data(args.features, args.galaxy_table)
    df = compute_mass_residuals(df)
    fit = fit_hinge_relation(df)
    fig = make_figure(df, fit)
    outfile = save_figure(fig, args.out_dir)
    print(f"Figure saved to: {outfile}")
    print(fit["summary"])
    plt.close(fig)


if __name__ == "__main__":
    main()
