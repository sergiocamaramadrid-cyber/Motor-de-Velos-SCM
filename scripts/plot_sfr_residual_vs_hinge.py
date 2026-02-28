#!/usr/bin/env python3
"""
plot_sfr_residual_vs_hinge.py
==============================
Generate the main paper figure: log SFR residual vs F3 (mean outer hinge).

The figure shows the partial effect of the SCM hinge activation proxy (F3)
on log SFR after removing the linear dependence on log Mbar.  This is the
natural visual complement to the regression result

    c(F3) = +0.112 ± 0.045,  p = 0.013

reported in the paper.

How it works
------------
1. Fit OLS: log_sfr = a + b * log_mbar  (mass-only baseline)
2. Compute residuals: e_i = log_sfr_i − ŷ_i
3. Scatter-plot e_i vs F3_mean_H_ext, colour-coded by morphology bin
4. Overlay the OLS best-fit line and 1-σ confidence band for
       e = c * F3 + noise
5. Annotate with the HC3-robust coefficient and p-value

Inputs
------
hinge_features.csv      per-galaxy friction proxies (from hinge_sfr_test.py)
galaxy_table.csv        per-galaxy log_mbar, log_sfr, morph_bin

Outputs
-------
results/paper_figures/sfr_residual_vs_F3.pdf   (publication-quality)
results/paper_figures/sfr_residual_vs_F3.png   (raster preview)

Usage
-----
Default (reads data/hinge_sfr/)::

    python scripts/plot_sfr_residual_vs_hinge.py

Custom paths::

    python scripts/plot_sfr_residual_vs_hinge.py \\
        --features  results/hinge_sfr/hinge_features.csv \\
        --galaxy-table  data/hinge_sfr/galaxy_table.csv \\
        --out-dir   results/paper_figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for CI / headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_FEATURES = "results/hinge_sfr/hinge_features.csv"
_DEFAULT_GALAXY_TABLE = "data/hinge_sfr/galaxy_table.csv"
_DEFAULT_OUT_DIR = "results/paper_figures"

# Morph-bin colour palette (MNRAS-friendly greyscale-safe colours)
_MORPH_COLOURS = {
    "late": "#1f77b4",   # blue
    "inter": "#ff7f0e",  # orange
    "early": "#2ca02c",  # green
}
_MORPH_MARKERS = {"late": "o", "inter": "s", "early": "^"}
_MORPH_LABELS = {"late": "Late (Sd/Sm/Im)", "inter": "Inter (Sa–Sc)", "early": "Early (S0/E)"}

# Floor applied to linear SFR before log10 to avoid log(0) errors
_MIN_SFR_FOR_LOG = 1e-12  # M☉/yr

# Alpha for the 1-σ confidence band (68% ≈ 1σ for a normal distribution)
_CI_ALPHA_1SIGMA = 0.32


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_sfr_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with an added ``sfr_resid`` column.

    Residuals are the log-SFR values after removing the OLS-fitted linear
    dependence on log Mbar:

        log_sfr = a + b * log_mbar  [+ morph dummies if morph_bin present]

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``log_mbar``, ``log_sfr``.  Optional: ``morph_bin``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``sfr_resid`` column.
    """
    work = df.dropna(subset=["log_mbar", "log_sfr", "F3_mean_H_ext"]).copy()

    X = sm.add_constant(work["log_mbar"].astype(float), has_constant="add")
    model = sm.OLS(work["log_sfr"].astype(float), X, missing="drop").fit()
    work["sfr_resid"] = model.resid
    return work


def fit_residual_vs_F3(df: pd.DataFrame) -> dict:
    """OLS of sfr_resid ~ F3 with HC3 errors.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``sfr_resid`` and ``F3_mean_H_ext``.

    Returns
    -------
    dict with keys: coef, se, pvalue, x_fit, y_fit, ci_lo, ci_hi
    """
    work = df.dropna(subset=["sfr_resid", "F3_mean_H_ext"])
    x = work["F3_mean_H_ext"].astype(float)
    y = work["sfr_resid"].astype(float)

    X = sm.add_constant(x, has_constant="add")
    res = sm.OLS(y, X).fit(cov_type="HC3")

    coef = float(res.params.iloc[-1])
    se = float(res.bse.iloc[-1])
    pvalue = float(res.pvalues.iloc[-1])

    # Prediction grid for the fit line and 1-σ band
    x_fit = np.linspace(float(x.min()), float(x.max()), 200)
    X_fit = sm.add_constant(x_fit, has_constant="add")
    pred = res.get_prediction(X_fit)
    y_fit = pred.predicted_mean
    ci = pred.conf_int(alpha=_CI_ALPHA_1SIGMA)
    ci_lo = ci[:, 0]
    ci_hi = ci[:, 1]

    return {
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, fit: dict) -> plt.Figure:
    """Create and return the publication-quality scatter figure.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy table with ``F3_mean_H_ext``, ``sfr_resid``, and optionally
        ``morph_bin``.
    fit : dict
        Output of :func:`fit_residual_vs_F3`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    # Scatter: colour by morphology if available
    if "morph_bin" in df.columns:
        for mbin, grp in df.groupby("morph_bin"):
            c = _MORPH_COLOURS.get(mbin, "grey")
            m = _MORPH_MARKERS.get(mbin, "o")
            lbl = _MORPH_LABELS.get(mbin, mbin)
            ax.scatter(
                grp["F3_mean_H_ext"],
                grp["sfr_resid"],
                c=c, marker=m, s=30, alpha=0.75, linewidths=0.4,
                edgecolors="k", label=lbl, zorder=3,
            )
    else:
        ax.scatter(
            df["F3_mean_H_ext"],
            df["sfr_resid"],
            c="steelblue", s=30, alpha=0.75, linewidths=0.4,
            edgecolors="k", zorder=3,
        )

    # Best-fit line
    ax.plot(fit["x_fit"], fit["y_fit"], color="crimson", lw=1.8, zorder=4,
            label=f"OLS fit  ($c = {fit['coef']:+.3f}$, $p = {fit['pvalue']:.3f}$)")

    # 1-σ confidence band
    ax.fill_between(
        fit["x_fit"], fit["ci_lo"], fit["ci_hi"],
        color="crimson", alpha=0.15, zorder=2, label=r"$1\sigma$ confidence band",
    )

    # Zero line
    ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=1)

    # Annotation box with the HC3 result
    ann = (
        r"$c = {:+.3f} \pm {:.3f}$".format(fit["coef"], fit["se"])
        + "\n"
        + r"$p = {:.4f}$".format(fit["pvalue"])
    )
    ax.text(
        0.97, 0.05, ann,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax.set_xlabel(r"$F_3 = \langle H(r) \rangle_{\rm ext}$  [outer mean hinge]", fontsize=11)
    ax.set_ylabel(
        r"$\log \mathrm{SFR}$ residual  (at fixed $\log M_{\rm bar}$)", fontsize=11
    )
    ax.set_title(
        r"SCM hinge activation vs SFR residual  (SPARC sample)", fontsize=11
    )

    if "morph_bin" in df.columns:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8, loc="upper left",
                  framealpha=0.9, edgecolor="0.7")
    else:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="0.7")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(features_path: str, galaxy_table_path: str) -> pd.DataFrame:
    """Load and merge hinge features with galaxy table.

    Parameters
    ----------
    features_path : str
        Path to ``hinge_features.csv`` produced by ``hinge_sfr_test.py``.
    galaxy_table_path : str
        Path to ``galaxy_table.csv``.

    Returns
    -------
    pd.DataFrame
        Merged table with all columns needed for the figure.
    """
    feats = pd.read_csv(features_path)
    gal = pd.read_csv(galaxy_table_path)
    df = gal.merge(feats[["galaxy", "F3_mean_H_ext"]], on="galaxy", how="inner")
    if "log_sfr" not in df.columns and "sfr" in df.columns:
        df["log_sfr"] = np.log10(np.maximum(df["sfr"].astype(float), _MIN_SFR_FOR_LOG))
    return df


def save_figure(fig: plt.Figure, out_dir: str) -> list[str]:
    """Save *fig* as PDF and PNG under *out_dir*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str
        Directory to write files into (created if absent).

    Returns
    -------
    list of str
        Paths of the saved files.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"sfr_residual_vs_F3.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate log SFR residual vs F3 (main paper figure).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features",
        default=_DEFAULT_FEATURES,
        help="Path to hinge_features.csv (output of hinge_sfr_test.py).",
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
    df = compute_sfr_residuals(df)
    fit = fit_residual_vs_F3(df)
    fig = make_figure(df, fit)
    saved = save_figure(fig, args.out_dir)
    for p in saved:
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
