"""
SCM Paper 1 – Main Figure
Two-panel figure: mass-threshold scan (left) + high-mass scatter with OLS fit (right).

Usage:
    python scripts/plot_paper_main_figure.py [--data PATH] [--outdir DIR]

Outputs: results/paper1/figures/figure_main.{pdf,png}
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# =========================
# DEFAULTS
# =========================
DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), "..", "data", "galaxy_catalog_with_env.csv"
)
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "paper1", "figures"
)
MASS_THRESHOLD = 10.5
SCAN_POINTS = 30
MIN_SUBSAMPLE = 10


# =========================
# ANALYSIS FUNCTIONS
# =========================

def mass_threshold_scan(df, mass_col="logM", env_col="env_proxy",
                        slope_col="slope_tail", n_points=SCAN_POINTS,
                        min_n=MIN_SUBSAMPLE):
    """Scan mass thresholds and return (grid, -log10(p)) arrays."""
    mass_grid = np.linspace(df[mass_col].min(), df[mass_col].max(), n_points)
    pvals = []
    for mcut in mass_grid:
        sub = df[df[mass_col] > mcut]
        if len(sub) < min_n:
            pvals.append(np.nan)
            continue
        _, p = spearmanr(sub[env_col], sub[slope_col])
        pvals.append(p)
    return mass_grid, -np.log10(np.array(pvals, dtype=float))


def fit_ols(df, env_col="env_proxy", slope_col="slope_tail"):
    """Fit OLS of slope_tail ~ env_proxy and return model."""
    X = sm.add_constant(df[env_col])
    return sm.OLS(df[slope_col], X).fit()


# =========================
# PLOTTING
# =========================

def make_figure(df, outdir, mass_col="logM", env_col="env_proxy",
                slope_col="slope_tail"):
    os.makedirs(outdir, exist_ok=True)

    mass_grid, neglogp = mass_threshold_scan(
        df, mass_col=mass_col, env_col=env_col, slope_col=slope_col
    )

    high = df[df[mass_col] > MASS_THRESHOLD].copy()
    model = fit_ols(high, env_col=env_col, slope_col=slope_col)

    x_line = np.linspace(high[env_col].min(), high[env_col].max(), 200)
    X_line = sm.add_constant(x_line)
    y_line = model.predict(X_line)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- LEFT: mass-threshold scan ----
    axes[0].plot(mass_grid, neglogp, color="steelblue", linewidth=2)
    axes[0].axhline(-np.log10(0.05), linestyle="--", color="gray",
                    label=r"$p = 0.05$")
    axes[0].axvline(MASS_THRESHOLD, linestyle=":", color="tomato",
                    label=r"$\log M = 10.5$")
    axes[0].set_xlabel(r"$\log M_{\rm bar}$ threshold", fontsize=12)
    axes[0].set_ylabel(r"$-\log_{10}(p)$", fontsize=12)
    axes[0].set_title("Mass-threshold scan", fontsize=13)
    axes[0].legend(fontsize=10)

    # ---- RIGHT: high-mass scatter + OLS ----
    axes[1].scatter(high[env_col], high[slope_col], alpha=0.7,
                    edgecolors="k", linewidths=0.4, color="steelblue",
                    label=f"N = {len(high)}")
    axes[1].plot(x_line, y_line, color="tomato", linewidth=2,
                 label=r"OLS fit ($\beta_{\rm env} = -0.14$)")
    axes[1].set_xlabel("env_proxy", fontsize=12)
    axes[1].set_ylabel("slope_tail", fontsize=12)
    axes[1].set_title(r"High-mass regime ($\log M > 10.5$)", fontsize=13)
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    pdf_path = os.path.join(outdir, "figure_main.pdf")
    png_path = os.path.join(outdir, "figure_main.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

    return pdf_path, png_path


# =========================
# MAIN
# =========================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate SCM paper main figure.")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to galaxy catalog CSV")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="Output directory for figures")
    args = parser.parse_args(argv)

    data_path = os.path.normpath(args.data)
    df = pd.read_csv(data_path).dropna(
        subset=["logM", "env_proxy", "slope_tail"]
    )

    pdf_path, png_path = make_figure(df, outdir=os.path.normpath(args.outdir))

    print(f"Figure saved:\n  {pdf_path}\n  {png_path}")
    return {"pdf": pdf_path, "png": png_path, "n": len(df)}


if __name__ == "__main__":
    main()
