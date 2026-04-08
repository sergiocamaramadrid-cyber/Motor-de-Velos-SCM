#!/usr/bin/env python3
"""
Generate the publication-ready LITTLE THINGS (N=26) scatter figure.

Plots delta_mass_std (environmental proxy, z-score of log_j) versus delta_f3
(outer-slope residual = slope_tail - 0.5) with a Spearman correlation
annotation and an OLS trend line.

Output: figure01_env_little_things.png + .pdf (300 dpi) by default.

Public API
----------
DELTA_MASS_STD : np.ndarray  -- hardcoded x values (N=26)
DELTA_F3       : np.ndarray  -- hardcoded y values (N=26)
compute_stats(x, y) -> dict  -- Spearman rho/p + OLS coef
generate_figure(x, y, out_path) -> plt.Figure
main(argv=None) -> dict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Hardcoded LITTLE THINGS (N=26) data
# ---------------------------------------------------------------------------

DELTA_MASS_STD = np.array([
    -2.4381,  0.0430,  0.4215,  1.3116,  0.8289,  1.0209, -1.2282,  0.1198,
    -0.3952,  0.2016, -1.4448,  0.1513,  0.8232,  0.9578,  0.3238,  0.2921,
     1.0715,  0.8388, -0.3725, -1.7098, -0.9900,  0.1250, -0.7250,  0.5246,
     1.3631, -1.1150,
])

DELTA_F3 = np.array([
    -0.3188, -0.3320, -0.3847, -0.4020, -0.2731, -0.3927, -0.3851, -0.3240,
    -0.5734, -0.5611, -0.3559, -0.6177, -0.4331, -0.4373, -0.3847, -0.3731,
    -0.5361, -0.4623, -0.8731, -0.3059, -0.3473, -0.4081, -0.1973, -0.3081,
    -0.4183, -0.3723,
])

_OUT_DEFAULT = Path("figure01_env_little_things.png")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Return Spearman rho/p and OLS (slope, intercept) for arrays x, y."""
    rho, p_val = spearmanr(x, y)
    coef = np.polyfit(x, y, 1)
    return {
        "rho": float(rho),
        "p_val": float(p_val),
        "ols_slope": float(coef[0]),
        "ols_intercept": float(coef[1]),
        "n": len(x),
    }


def generate_figure(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path = _OUT_DEFAULT,
) -> plt.Figure:
    """Create scatter + OLS trend figure and save to *out_path*."""
    stats = compute_stats(x, y)
    rho = stats["rho"]
    p_val = stats["p_val"]

    coef = np.array([stats["ols_slope"], stats["ols_intercept"]])
    line = np.poly1d(coef)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x, y, color='black', edgecolor='white', s=60, alpha=0.8)
    ax.plot(x_line, line(x_line), 'r--', linewidth=1.5)

    ax.set_xlabel(r'$\delta_{\mathrm{mass\_std}}$ (environment)', fontsize=12)
    ax.set_ylabel(r'$\delta_{f3}$', fontsize=12)
    ax.set_title('LITTLE THINGS sample (N=26)', fontsize=11)

    ax.text(
        0.05, 0.95,
        f'Spearman $\\rho = {rho:.2f}$\n$p = {p_val:.3f}$',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LITTLE THINGS (N=26) environmental scatter figure."
    )
    p.add_argument(
        "--out", default=str(_OUT_DEFAULT),
        help="Output PNG path (default: figure01_env_little_things.png); PDF saved alongside automatically.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point. Returns dict with stats and output path."""
    args = _parse_args(argv)
    out_path = Path(args.out)

    x = DELTA_MASS_STD
    y = DELTA_F3

    fig = generate_figure(x, y, out_path=out_path)
    plt.close(fig)

    stats = compute_stats(x, y)
    stats["out_path"] = str(out_path)
    stats["pdf_path"] = str(out_path.with_suffix(".pdf"))
    print(
        f"Wrote figure → {out_path} + {out_path.with_suffix('.pdf')}  "
        f"(Spearman rho={stats['rho']:.2f}, p={stats['p_val']:.3f})"
    )
    return stats


if __name__ == "__main__":
    result = main()
    sys.exit(0)
