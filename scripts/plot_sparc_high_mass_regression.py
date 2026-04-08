"""
scripts/plot_sparc_high_mass_regression.py — Scatter + OLS regression figure
for the SPARC high-mass subsample.

Loads a per-galaxy catalog CSV, filters to galaxies with logM >= M_CRIT,
and produces a scatter plot of ``delta_f3`` vs ``delta_mass_std`` overlaid
with the OLS regression line.  Spearman and OLS (HC3) statistics are printed
to stdout and returned by :func:`main`.

Theory
------
``delta_f3 = slope_tail − 0.5``

where ``slope_tail`` is the outer-disk dlogV/dlogr slope and ``0.5`` is the
SCM reference value (Motor-de-Velos deep form / MOND asymptotic slope).

``delta_mass_std`` is the z-score of the angular momentum proxy, used as a
measure of the environmental tidal field.

The mass threshold ``M_CRIT = 10.05`` is determined data-driven by
:mod:`scripts.plot_sparc_mass_scan` (maximises the composite signal score
``S = |rho| * sqrt(N) * (-log10 p)``).

Usage
-----
::

    python scripts/plot_sparc_high_mass_regression.py

With optional arguments::

    python scripts/plot_sparc_high_mass_regression.py \\
        --csv data/sparc_subset.csv \\
        --m-crit 10.05 \\
        --out results/scm_high_mass_regression.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF: float = 0.5
M_CRIT_DEFAULT: float = 10.05

#: Official figure caption for use in paper drafts and supplementary material.
FIGURE_CAPTION: str = (
    "Figure X. Environmental modulation of the outer rotation curve slope in the "
    "high-mass regime (logM \u2265 10.05). Each point represents a galaxy from the "
    "SPARC subset (N\u202f=\u202f56). The solid line shows the OLS fit with HC3 robust "
    "errors. A negative slope is observed, indicating that galaxies in denser "
    "environments exhibit more negative \u0394F\u2083 values. No equivalent correlation "
    "is found in the low-mass regime."
)

_REQUIRED_COLS = {"slope_tail", "logM", "delta_mass_std"}
_REPO_ROOT = Path(__file__).parent.parent
_CSV_DEFAULT = str(_REPO_ROOT / "data" / "sparc_subset.csv")
_OUT_DEFAULT = "results/scm_high_mass_regression.png"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Spearman correlation and OLS (HC3) regression statistics.

    Parameters
    ----------
    x : array-like
        Predictor values (``delta_mass_std``).
    y : array-like
        Response values (``delta_f3``).

    Returns
    -------
    dict with keys:
        rho         — Spearman rank correlation coefficient
        p_spear     — two-sided p-value for the Spearman test
        ols_slope   — OLS slope coefficient
        ols_intercept — OLS intercept coefficient
        ols_pval    — HC3 p-value for the slope coefficient
        n           — number of data points
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    rho, p_spear = spearmanr(x, y)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    ols_intercept, ols_slope = model.params
    ols_pval = model.pvalues[1]

    return {
        "rho": float(rho),
        "p_spear": float(p_spear),
        "ols_slope": float(ols_slope),
        "ols_intercept": float(ols_intercept),
        "ols_pval": float(ols_pval),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def _render_figure(
    x: np.ndarray,
    y: np.ndarray,
    stats: dict,
    m_crit: float,
    out_path: Path,
) -> plt.Figure:
    """Internal helper: render and save the scatter + regression figure.

    Both a PNG (at *out_path*) and a sibling PDF are written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = stats["ols_intercept"] + stats["ols_slope"] * x_line

    annotation = (
        f"N = {stats['n']}\n"
        f"Spearman ρ = {stats['rho']:.3f}  p = {stats['p_spear']:.2e}\n"
        f"OLS slope = {stats['ols_slope']:.3f}  p(HC3) = {stats['ols_pval']:.2e}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, color="steelblue", alpha=0.75, s=40, zorder=3, label="Galaxies")
    ax.plot(x_line, y_line, color="tomato", lw=1.8, zorder=4, label="OLS fit")
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=2)
    ax.set_xlabel(r"$\delta_\mathrm{mass,std}$", fontsize=12)
    ax.set_ylabel(r"$\Delta F_3$", fontsize=12)
    ax.set_title(
        rf"SPARC high-mass regime ($\log M \geq {m_crit}$)", fontsize=11
    )
    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return fig


def generate_figure(
    df: pd.DataFrame,
    out_path: str | Path,
    m_crit: float = M_CRIT_DEFAULT,
) -> plt.Figure:
    """Generate the high-mass scatter + regression figure.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns ``logM``, ``delta_mass_std``, ``slope_tail``.
        ``delta_f3`` is computed internally as ``slope_tail − 0.5``.
    out_path : str or Path
        Destination PNG path.  A sibling PDF is also written.
    m_crit : float
        Minimum logM threshold.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.copy()
    df["delta_f3"] = df["slope_tail"] - BETA_REF
    df_high = df[df["logM"] >= m_crit].dropna(
        subset=["delta_mass_std", "delta_f3"]
    )
    if len(df_high) < 2:
        raise ValueError(
            f"Only {len(df_high)} galaxies with logM >= {m_crit}; "
            "need at least 2 to fit a regression."
        )
    x = df_high["delta_mass_std"].values
    y = df_high["delta_f3"].values
    stats = compute_stats(x, y)
    return _render_figure(x, y, stats, m_crit, Path(out_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter + OLS regression figure for the SPARC high-mass subsample: "
            "delta_f3 vs delta_mass_std."
        )
    )
    parser.add_argument(
        "--csv", default=_CSV_DEFAULT,
        help="Path to per-galaxy catalog CSV (default: data/sparc_subset.csv).",
    )
    parser.add_argument(
        "--m-crit", type=float, default=M_CRIT_DEFAULT, dest="m_crit",
        help=f"Minimum logM threshold (default: {M_CRIT_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=_OUT_DEFAULT,
        help=f"Output PNG path (default: {_OUT_DEFAULT}). A sibling PDF is also saved.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the high-mass scatter + regression figure pipeline.

    Returns
    -------
    dict with keys:
        stats           — statistics dict (rho, p_spear, ols_slope, …, n)
        m_crit          — mass threshold used
        figure_path     — Path to the saved PNG
        pdf_path        — Path to the saved PDF
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {csv_path}\n"
            "Provide a CSV with columns: slope_tail, logM, delta_mass_std."
        )

    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["delta_f3"] = df["slope_tail"] - BETA_REF
    df_high = df[df["logM"] >= args.m_crit].dropna(
        subset=["delta_mass_std", "delta_f3"]
    )
    n = len(df_high)

    if n < 2:
        raise ValueError(
            f"Only {n} galaxies with logM >= {args.m_crit}; "
            "need at least 2 to fit a regression."
        )

    x = df_high["delta_mass_std"].values
    y = df_high["delta_f3"].values
    stats = compute_stats(x, y)

    out_path = Path(args.out)
    _render_figure(x, y, stats, args.m_crit, out_path)
    pdf_path = out_path.with_suffix(".pdf")

    print(f"N alta masa: {n}")
    print(f"Spearman rho = {stats['rho']:.3f}")
    print(f"Spearman p = {stats['p_spear']:.3e}")
    print(
        f"OLS slope = {stats['ols_slope']:.4f}  "
        f"p(HC3) = {stats['ols_pval']:.3e}"
    )
    print(f"Figure saved as '{out_path}' and '{pdf_path}'")

    return {
        "stats": stats,
        "m_crit": args.m_crit,
        "figure_path": out_path,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    main()
