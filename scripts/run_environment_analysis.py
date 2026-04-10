"""
scripts/run_environment_analysis.py — Environmental correlation analysis for SPARC outer slopes.

Tests whether the outer rotation-curve slope (slope_tail, F3_SCM) correlates with an
environmental proxy (env_proxy), both globally and after splitting by baryonic mass.
A residual test controls for the mass-driven component using OLS.

Theory
------
Environmental modulation of galaxy outer dynamics may emerge only above a characteristic
baryonic mass scale.  This script quantifies:
  1. Global Spearman correlation (env_proxy vs slope_tail)
  2. Mass-split Spearman (low vs high mass)
  3. Residual test: OLS slope_tail ~ logMbar, then Spearman(residuals, env_proxy)

Usage
-----
    python scripts/run_environment_analysis.py

    python scripts/run_environment_analysis.py \\
        --csv data/galaxy_catalog_with_env.csv \\
        --threshold 10.05 \\
        --out results/paper1_environment \\
        --no-figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGM_THRESHOLD_DEFAULT: float = 10.05
MASS_COL: str = "logMbar"
ENV_COL: str = "env_proxy"
SLOPE_COL: str = "slope_tail"
CSV_DEFAULT: str = "data/galaxy_catalog_with_env.csv"
OUT_DEFAULT: str = "results/paper1_environment"

# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def compute_global_spearman(df: pd.DataFrame, env_col: str, slope_col: str) -> dict:
    """Return Spearman correlation between env_proxy and slope_tail for all galaxies."""
    rho, pval = spearmanr(df[env_col], df[slope_col])
    return {"rho": float(rho), "pval": float(pval), "n": len(df)}


def compute_mass_split(
    df: pd.DataFrame,
    mass_col: str,
    env_col: str,
    slope_col: str,
    threshold: float,
) -> dict:
    """Compute Spearman correlations for low- and high-mass subsamples."""
    low = df[df[mass_col] < threshold]
    high = df[df[mass_col] >= threshold]

    rho_low, pval_low = spearmanr(low[env_col], low[slope_col])
    rho_high, pval_high = spearmanr(high[env_col], high[slope_col])

    return {
        "low": {"rho": float(rho_low), "pval": float(pval_low), "n": len(low)},
        "high": {"rho": float(rho_high), "pval": float(pval_high), "n": len(high)},
    }


def compute_residual_test(
    df: pd.DataFrame,
    mass_col: str,
    env_col: str,
    slope_col: str,
    threshold: float,
) -> dict:
    """OLS of slope_tail ~ logMbar; Spearman(residuals, env_proxy) globally and per regime."""
    x = df[mass_col].values
    y = df[slope_col].values

    model = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    slope_ols = float(model.params[1])
    intercept_ols = float(model.params[0])
    r2 = float(model.rsquared)
    pval_ols = float(model.pvalues[1])

    residuals = y - model.fittedvalues

    df2 = df.copy()
    df2["_resid"] = residuals

    low = df2[df2[mass_col] < threshold]
    high = df2[df2[mass_col] >= threshold]

    rho_g, pval_g = spearmanr(df2[env_col], df2["_resid"])
    rho_l, pval_l = spearmanr(low[env_col], low["_resid"])
    rho_h, pval_h = spearmanr(high[env_col], high["_resid"])

    return {
        "ols": {
            "slope": slope_ols,
            "intercept": intercept_ols,
            "r2": r2,
            "pval": pval_ols,
        },
        "global": {"rho": float(rho_g), "pval": float(pval_g), "n": len(df2)},
        "low": {"rho": float(rho_l), "pval": float(pval_l), "n": len(low)},
        "high": {"rho": float(rho_h), "pval": float(pval_h), "n": len(high)},
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x-sorted arrays for a simple OLS regression line."""
    coeffs = np.polyfit(x, y, 1)
    x_sorted = np.linspace(x.min(), x.max(), 200)
    return x_sorted, np.polyval(coeffs, x_sorted)


def generate_figure_env_correlation(
    df: pd.DataFrame,
    env_col: str,
    slope_col: str,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Scatter plot of env_proxy vs slope_tail with OLS fit and Spearman annotation."""
    out_dir = Path(out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rho, pval = spearmanr(df[env_col], df[slope_col])
    x = df[env_col].values
    y = df[slope_col].values
    x_fit, y_fit = _ols_line(x, y)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(x, y, s=20, alpha=0.7, color="steelblue", label="Galaxies")
    ax.plot(x_fit, y_fit, color="firebrick", linewidth=1.5, label="OLS fit")
    ax.set_xlabel("Environmental proxy (env_proxy)", fontsize=11)
    ax.set_ylabel("Outer slope (slope_tail)", fontsize=11)
    ax.set_title("Environmental correlation (SPARC outer slope)", fontsize=12)
    ax.annotate(
        f"Spearman ρ = {rho:.3f}\np = {pval:.3e}\nN = {len(df)}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    png_path = out_dir / "figure_env_correlation.png"
    pdf_path = out_dir / "figure_env_correlation.pdf"
    fig.savefig(png_path, dpi=150, facecolor="white")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def generate_figure_env_residual_split(
    df: pd.DataFrame,
    mass_col: str,
    env_col: str,
    slope_col: str,
    threshold: float,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """Two-panel figure: residuals vs env for low-mass (left) and high-mass (right)."""
    out_dir = Path(out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df[mass_col].values
    y = df[slope_col].values
    coeffs = np.polyfit(x, y, 1)
    residuals = y - np.polyval(coeffs, x)

    df2 = df.copy()
    df2["_resid"] = residuals
    low = df2[df2[mass_col] < threshold]
    high = df2[df2[mass_col] >= threshold]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    fig.suptitle("Residual vs environment by mass regime", fontsize=13)

    for ax, subset, label, color in [
        (axes[0], low, f"Low-mass (N={len(low)})", "steelblue"),
        (axes[1], high, f"High-mass (N={len(high)})", "darkorange"),
    ]:
        ex = subset[env_col].values
        ey = subset["_resid"].values
        rho, pval = spearmanr(ex, ey)
        x_fit, y_fit = _ols_line(ex, ey)
        ax.set_facecolor("white")
        ax.scatter(ex, ey, s=20, alpha=0.7, color=color)
        ax.plot(x_fit, y_fit, color="firebrick", linewidth=1.5)
        ax.set_xlabel("Environmental proxy (env_proxy)", fontsize=10)
        ax.set_ylabel("Residual (slope_tail − OLS fit)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.annotate(
            f"ρ = {rho:.3f}, p = {pval:.3f}\nN = {len(subset)}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    fig.tight_layout()

    png_path = out_dir / "figure_env_residual_split.png"
    pdf_path = out_dir / "figure_env_residual_split.pdf"
    fig.savefig(png_path, dpi=150, facecolor="white")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------


def build_summary_results_csv(
    global_result: dict,
    split_result: dict,
    residual_result: dict,
) -> pd.DataFrame:
    """Build summary DataFrame with one row per statistical test."""
    rows = [
        {
            "test": "global_spearman",
            "regime": "all",
            "n": global_result["n"],
            "rho": global_result["rho"],
            "pval": global_result["pval"],
        },
        {
            "test": "mass_split",
            "regime": "low",
            "n": split_result["low"]["n"],
            "rho": split_result["low"]["rho"],
            "pval": split_result["low"]["pval"],
        },
        {
            "test": "mass_split",
            "regime": "high",
            "n": split_result["high"]["n"],
            "rho": split_result["high"]["rho"],
            "pval": split_result["high"]["pval"],
        },
        {
            "test": "residual_spearman",
            "regime": "all",
            "n": residual_result["global"]["n"],
            "rho": residual_result["global"]["rho"],
            "pval": residual_result["global"]["pval"],
        },
        {
            "test": "residual_spearman",
            "regime": "low",
            "n": residual_result["low"]["n"],
            "rho": residual_result["low"]["rho"],
            "pval": residual_result["low"]["pval"],
        },
        {
            "test": "residual_spearman",
            "regime": "high",
            "n": residual_result["high"]["n"],
            "rho": residual_result["high"]["rho"],
            "pval": residual_result["high"]["pval"],
        },
    ]
    return pd.DataFrame(rows, columns=["test", "regime", "n", "rho", "pval"])


def build_summary_md(
    global_result: dict,
    split_result: dict,
    residual_result: dict,
    out_dir: str | Path,
    threshold: float = LOGM_THRESHOLD_DEFAULT,
) -> Path:
    """Write a Markdown summary of all results; return path to the file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "summary.md"

    content = f"""# paper1_environment: Results Summary

## Global correlation
- N = {global_result['n']}
- Spearman ρ = {global_result['rho']:.3f}
- p = {global_result['pval']:.3e}

## Mass split (threshold = {threshold})
### Low-mass (N = {split_result['low']['n']})
- ρ = {split_result['low']['rho']:.3f}
- p = {split_result['low']['pval']:.3f}

### High-mass (N = {split_result['high']['n']})
- ρ = {split_result['high']['rho']:.3f}
- p = {split_result['high']['pval']:.4f}

## Residual test (slope_tail ~ logM)
OLS fit: slope = {residual_result['ols']['slope']:.4f}, intercept = {residual_result['ols']['intercept']:.4f}, R² = {residual_result['ols']['r2']:.4f}, p = {residual_result['ols']['pval']:.4f}

Residual vs env_proxy:
- Global: ρ = {residual_result['global']['rho']:.4f}, p = {residual_result['global']['pval']:.5f}
- Low mass: ρ = {residual_result['low']['rho']:.4f}, p = {residual_result['low']['pval']:.3f}
- High mass: ρ = {residual_result['high']['rho']:.4f}, p = {residual_result['high']['pval']:.5f}
"""
    md_path.write_text(content)
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> dict:
    """Orchestrate the full environmental analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Environmental correlation analysis for SPARC outer slopes."
    )
    parser.add_argument(
        "--csv",
        default=CSV_DEFAULT,
        help=f"Path to input CSV (default: {CSV_DEFAULT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOGM_THRESHOLD_DEFAULT,
        help=f"logMbar mass split threshold (default: {LOGM_THRESHOLD_DEFAULT})",
    )
    parser.add_argument(
        "--out",
        default=OUT_DEFAULT,
        help=f"Output directory (default: {OUT_DEFAULT})",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir = Path(args.out)

    # --- Analysis ---
    global_result = compute_global_spearman(df, ENV_COL, SLOPE_COL)
    split_result = compute_mass_split(df, MASS_COL, ENV_COL, SLOPE_COL, args.threshold)
    residual_result = compute_residual_test(df, MASS_COL, ENV_COL, SLOPE_COL, args.threshold)

    # --- Figures ---
    figures: dict = {}
    if not args.no_figures:
        png1, pdf1 = generate_figure_env_correlation(df, ENV_COL, SLOPE_COL, out_dir)
        png2, pdf2 = generate_figure_env_residual_split(
            df, MASS_COL, ENV_COL, SLOPE_COL, args.threshold, out_dir
        )
        figures = {
            "env_correlation": (png1, pdf1),
            "env_residual_split": (png2, pdf2),
        }

    # --- Tables ---
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary_results_csv(global_result, split_result, residual_result)
    csv_out = tables_dir / "summary_results.csv"
    summary_df.to_csv(csv_out, index=False)

    # --- Summary markdown ---
    md_path = build_summary_md(
        global_result,
        split_result,
        residual_result,
        out_dir,
        threshold=args.threshold,
    )

    print(f"Global:    ρ={global_result['rho']:.3f}  p={global_result['pval']:.3e}  N={global_result['n']}")
    print(f"Low-mass:  ρ={split_result['low']['rho']:.3f}   p={split_result['low']['pval']:.3f}  N={split_result['low']['n']}")
    print(f"High-mass: ρ={split_result['high']['rho']:.3f}  p={split_result['high']['pval']:.4f} N={split_result['high']['n']}")
    print(f"OLS slope={residual_result['ols']['slope']:.4f} R²={residual_result['ols']['r2']:.4f}")
    print(f"Summary written to: {md_path}")

    return {
        "global": global_result,
        "split": split_result,
        "residual": residual_result,
        "figures": figures,
        "tables": {"summary_results": csv_out},
    }


if __name__ == "__main__":
    main()
