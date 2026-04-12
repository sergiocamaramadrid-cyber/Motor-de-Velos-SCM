"""
Figure 1 for SCM-TR paper: Mass-threshold emergence + Environmental correlation.

Panel A — Mass-threshold scan: significance (-log10 p) of env–slope Spearman
           correlation as a function of baryonic mass cut.
Panel B — High-mass regime: env_proxy vs slope_tail scatter with linear
           regression and Spearman annotation.

Outputs:
    results/robustness/figure_mass_environment.png  (300 dpi)
    results/robustness/figure_mass_environment.pdf
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

FIGURE_CAPTION = (
    "Left: Significance of the environmental correlation as a function of "
    "baryonic mass threshold. The signal emerges only above log M ≈ 10.6, "
    "indicating a regime-dependent behavior. "
    "Right: Correlation between environmental proxy and outer rotation curve "
    "slope in the high-mass regime. The negative correlation is statistically "
    "significant and robust."
)

THRESHOLD = 10.6
SCAN_MIN = 9.8
SCAN_MAX = 11.0
SCAN_STEP = 0.1
MIN_N = 10

OUT_DIR = pathlib.Path("results/robustness")


def run_mass_scan(
    df: pd.DataFrame,
    mass_col: str = "logMbar",
    env_col: str = "env_proxy",
    slope_col: str = "slope_tail",
    scan_min: float = SCAN_MIN,
    scan_max: float = SCAN_MAX,
    scan_step: float = SCAN_STEP,
    min_n: int = MIN_N,
) -> tuple[np.ndarray, np.ndarray]:
    cuts = np.arange(scan_min, scan_max, scan_step)
    pvals: list[float] = []
    for cut in cuts:
        subset = df[df[mass_col] >= cut]
        if len(subset) > min_n:
            _, p = spearmanr(subset[env_col], subset[slope_col])
            pvals.append(float(p))
        else:
            pvals.append(float("nan"))
    return cuts, np.array(pvals)


def high_mass_stats(
    df: pd.DataFrame,
    threshold: float = THRESHOLD,
    mass_col: str = "logMbar",
    env_col: str = "env_proxy",
    slope_col: str = "slope_tail",
) -> dict:
    high = df[df[mass_col] >= threshold].copy()
    x = high[env_col].values
    y = high[slope_col].values
    rho, p = spearmanr(x, y)
    slope, intercept, *_ = linregress(x, y)
    return dict(x=x, y=y, rho=rho, p=p, slope=slope, intercept=intercept, n=len(high))


def generate_figure(
    df: pd.DataFrame,
    out_dir: pathlib.Path = OUT_DIR,
    threshold: float = THRESHOLD,
    mass_col: str = "logMbar",
    env_col: str = "env_proxy",
    slope_col: str = "slope_tail",
) -> plt.Figure:
    cuts, pvals = run_mass_scan(df, mass_col, env_col, slope_col)
    yvals = -np.log10(pvals)

    stats = high_mass_stats(df, threshold, mass_col, env_col, slope_col)
    x, y = stats["x"], stats["y"]
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = stats["slope"] * xfit + stats["intercept"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel A: mass-threshold scan
    ax = axes[0]
    ax.plot(cuts, yvals, color="steelblue", lw=2, marker="o", ms=4)
    ax.axvline(threshold, color="crimson", ls="--", lw=1.4, label=f"log M = {threshold}")
    ax.axhline(-np.log10(0.05), color="gray", ls=":", lw=1.2, label="p = 0.05")
    ax.set_xlabel(r"$\log\,M_{\rm cut}$", fontsize=13)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=13)
    ax.set_title("Mass-threshold scan", fontsize=13)
    ax.legend(fontsize=10)

    # ---- Panel B: high-mass env-slope correlation
    ax = axes[1]
    ax.scatter(x, y, color="steelblue", alpha=0.75, zorder=3, label=f"N={stats['n']}")
    ax.plot(xfit, yfit, color="crimson", lw=1.8)
    ax.set_xlabel("env_proxy", fontsize=13)
    ax.set_ylabel("slope_tail", fontsize=13)
    ax.set_title(f"High-mass regime (log M ≥ {threshold})", fontsize=13)
    ax.text(
        0.05, 0.95,
        f"ρ = {stats['rho']:.2f}\np = {stats['p']:.3e}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Generate mass-environment figure")
    parser.add_argument("--input", default="results/scm_master_final.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input).dropna(subset=["logMbar", "env_proxy", "slope_tail"])

    fig = generate_figure(df, out_dir, args.threshold)

    png_path = out_dir / "figure_mass_environment.png"
    pdf_path = out_dir / "figure_mass_environment.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"✅ Figure generated → {png_path}")
    print(f"✅ Figure generated → {pdf_path}")

    return dict(png_path=str(png_path), pdf_path=str(pdf_path))


if __name__ == "__main__":
    main()
