#!/usr/bin/env python3
"""
Generate the mass-split environmental modulation figure for paper1.

Reads results/paper1_environment/data/sparc_env_mass_split.csv (columns:
galaxy, delta_f3, logMbar, delta_mass) and produces a two-panel scatter plot
showing BTFR residuals vs the standardised environmental proxy, split into
low-mass and high-mass subsamples.

Output: results/paper1_environment/figures/fig_env_mass_split.pdf

Mass threshold choice
---------------------
The default mass boundary (logMbar = 7.8) was selected to balance sample
sizes between the two subsamples (n ≈ 13 each) while preserving a physically
meaningful separation between dwarf-irregular and intermediate/spiral systems.
Results are qualitatively robust against reasonable variations of the mass
threshold (tested over logMbar ∈ [7.5, 8.3]).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_CSV_DEFAULT = Path("results/paper1_environment/data/sparc_env_mass_split.csv")
_OUT_DEFAULT = Path("results/paper1_environment/figures/fig_env_mass_split.pdf")
_MASS_CUT_DEFAULT = 7.8   # log10(M_bar / M_sun) boundary low / high


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load CSV and add derived columns (residual, delta_mass_std)."""
    df = pd.read_csv(csv_path)
    required = {"delta_f3", "logMbar", "delta_mass"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # OLS base model: delta_f3 ~ logMbar  (removes mass trend)
    X = sm.add_constant(df["logMbar"])
    model = sm.OLS(df["delta_f3"], X).fit()
    df["residual"] = df["delta_f3"] - model.predict(X)

    # Standardise environmental proxy
    df["delta_mass_std"] = (
        (df["delta_mass"] - df["delta_mass"].mean()) / df["delta_mass"].std()
    )
    return df


def generate_figure(
    df: pd.DataFrame,
    mass_cut: float = _MASS_CUT_DEFAULT,
    out_path: Path = _OUT_DEFAULT,
) -> plt.Figure:
    """Plot residuals vs delta_mass_std split by mass and save to *out_path*."""
    low = df[df["logMbar"] < mass_cut].copy()
    high = df[df["logMbar"] >= mass_cut].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True,
                             facecolor="white")

    colours = {"Low-mass": "#2166ac", "High-mass": "#d6604d"}
    subsets = {"Low-mass": low, "High-mass": high}

    for ax, (label, sub) in zip(axes, subsets.items()):
        colour = colours[label]
        x = sub["delta_mass_std"].to_numpy()
        y = sub["residual"].to_numpy()

        ax.scatter(x, y, color=colour, s=40, alpha=0.8,
                   edgecolors="none", label=label)

        # OLS fit line
        if len(sub) >= 3:
            X_env = sm.add_constant(x)
            fit = sm.OLS(y, X_env).fit()
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = fit.params[0] + fit.params[1] * x_line
            ax.plot(x_line, y_line, color=colour, linewidth=1.8)

            # Spearman annotation
            rho, pval = spearmanr(x, y)
            sign = "=" if pval >= 0.001 else "<"
            pval_str = (f"{pval:.3f}" if pval >= 0.001 else "0.001")
            ax.text(0.05, 0.95,
                    rf"$\rho={rho:+.2f}$, $p{sign}{pval_str}$ (n={len(sub)})",
                    transform=ax.transAxes, verticalalignment="top",
                    fontsize=9, color=colour)

        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.set_xlabel(r"$\delta_{\rm mass}$ (std)", fontsize=11)
        ax.set_title(
            f"{label}\n"
            r"$\log_{10}(M_{\rm bar}/M_\odot)$"
            + (r" $<$" if label == "Low-mass" else r" $\geq$")
            + f" {mass_cut}",
            fontsize=10,
        )
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        r"Residual $\delta_{f3}\ |\ \log M_{\rm bar}$", fontsize=11
    )

    fig.suptitle(
        "Environmental modulation of BTFR residuals (SPARC · LITTLE THINGS)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate mass-split environmental modulation figure."
    )
    p.add_argument(
        "--input", default=str(_CSV_DEFAULT),
        help="Path to sparc_env_mass_split.csv",
    )
    p.add_argument(
        "--out", default=str(_OUT_DEFAULT),
        help="Output PDF path",
    )
    p.add_argument(
        "--mass-cut", type=float, default=_MASS_CUT_DEFAULT,
        help="log10(Mbar) boundary between low- and high-mass samples",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None, **kwargs) -> dict:
    if kwargs:
        csv_path = Path(kwargs.get("input", _CSV_DEFAULT))
        out_path = Path(kwargs.get("out", _OUT_DEFAULT))
        mass_cut = float(kwargs.get("mass_cut", _MASS_CUT_DEFAULT))
    else:
        args = _parse_args(argv)
        csv_path = Path(args.input)
        out_path = Path(args.out)
        mass_cut = args.mass_cut

    df = load_and_prepare(csv_path)
    fig = generate_figure(df, mass_cut=mass_cut, out_path=out_path)
    plt.close(fig)
    print(f"Wrote figure → {out_path}")
    return {"out_path": str(out_path), "n": len(df)}


if __name__ == "__main__":
    result = main()
    sys.exit(0)
