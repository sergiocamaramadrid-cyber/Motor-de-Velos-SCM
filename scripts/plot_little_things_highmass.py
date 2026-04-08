#!/usr/bin/env python3
"""
Generate the publication-ready LITTLE THINGS high-mass (N=13) scatter figure.

Plots delta_mass_std (environmental proxy, z-score of log_j) versus delta_f3
(outer-slope residual = slope_tail - 0.5) for the high-mass subsample defined
by the fixed mass threshold logMbar >= 7.8 (documented in
results/paper1_environment/RESULTS.md).

Output: figure02_env_little_things_highmass.png + .pdf (300 dpi) by default.

Public API
----------
DELTA_MASS_STD_HM : np.ndarray  -- hardcoded x values (N=13, logM >= 7.8)
DELTA_F3_HM       : np.ndarray  -- hardcoded y values (N=13)
LOGM_CUT          : float       -- fixed mass threshold (7.8)
compute_stats(x, y) -> dict     -- Spearman rho/p + OLS coef
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
# Fixed mass threshold (documented in results/paper1_environment/RESULTS.md)
# ---------------------------------------------------------------------------

LOGM_CUT: float = 7.8

# ---------------------------------------------------------------------------
# Hardcoded high-mass subsample data (logMbar >= 7.8, N=13)
# Source: results/lt_sample_catalog.csv filtered by logM >= LOGM_CUT
# Galaxy order: DDO46, DDO47, DDO50, DDO52, DDO63, DDO87, DDO101,
#               DDO126, DDO133, DDO168, Haro29, NGC1569, NGC2366
# ---------------------------------------------------------------------------

DELTA_MASS_STD_HM = np.array([
    0.4215, 1.3116, 0.8289, 1.0209, 0.1198, 0.8232, 0.9578,
    0.3238, 0.2921, 0.8388, 0.1250, 0.5246, 1.3631,
])

DELTA_F3_HM = np.array([
    -0.3847, -0.4020, -0.2731, -0.3927, -0.3240, -0.4331, -0.4373,
    -0.3847, -0.3731, -0.4623, -0.4081, -0.3081, -0.4183,
])

_OUT_DEFAULT = Path("figure02_env_little_things_highmass.png")


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
    ax.set_title(
        f'LITTLE THINGS high-mass subsample (N={stats["n"]}, '
        f'$\\log M_{{\\rm bar}} \\geq {LOGM_CUT}$)',
        fontsize=11,
    )

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
        description=(
            "Generate LITTLE THINGS high-mass (N=13, logM>={}) "
            "environmental scatter figure.".format(LOGM_CUT)
        )
    )
    p.add_argument(
        "--out", default=str(_OUT_DEFAULT),
        help=(
            "Output PNG path (default: figure02_env_little_things_highmass.png); "
            "PDF saved alongside automatically."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point. Returns dict with stats and output path."""
    args = _parse_args(argv)
    out_path = Path(args.out)

    x = DELTA_MASS_STD_HM
    y = DELTA_F3_HM

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
