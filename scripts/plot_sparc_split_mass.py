"""
scripts/plot_sparc_split_mass.py — Split-by-mass environmental correlation figure.

Reads a per-galaxy catalog CSV (e.g. ``sparc_subset.csv``) and produces a
two-panel scatter figure comparing the environmental proxy ``delta_f3`` vs
``delta_mass_std`` for low-mass and high-mass subsamples split at the sample
median of ``logM``.

Theory
------
``delta_f3 = slope_tail − 0.5``

where ``slope_tail`` is the outer-disk dlogV/dlogr slope and ``0.5`` is the
SCM reference value (Motor-de-Velos deep form / MOND asymptotic slope).

Usage
-----
::

    python scripts/plot_sparc_split_mass.py \\
        --csv sparc_subset.csv

With optional output path::

    python scripts/plot_sparc_split_mass.py \\
        --csv sparc_subset.csv \\
        --out results/SPARC_split_mass_environment.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF: float = 0.5          # SCM / MOND reference outer-disk slope
_REQUIRED_COLS = {"slope_tail", "logM", "delta_mass_std"}
_REPO_ROOT = Path(__file__).parent.parent
_CSV_DEFAULT = str(_REPO_ROOT / "data" / "sparc_subset.csv")
_OUT_DEFAULT = "SPARC_split_mass_environment.png"

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Spearman and OLS statistics for a scatter of (x, y) pairs.

    Parameters
    ----------
    x, y : array_like
        Paired arrays of equal length (NaN-free).

    Returns
    -------
    dict with keys:
        n              — number of data points
        rho            — Spearman correlation coefficient
        p_spear        — two-tailed Spearman p-value
        ols_slope      — OLS regression slope
        ols_intercept  — OLS regression intercept
        r2             — R² (Pearson r squared)
        p_ols          — two-tailed OLS slope p-value
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rho, p_spear = spearmanr(x, y)
    slope, intercept, r_value, p_ols, _stderr = linregress(x, y)
    return {
        "n": n,
        "rho": float(rho),
        "p_spear": float(p_spear),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "r2": float(r_value ** 2),
        "p_ols": float(p_ols),
    }


def split_by_mass(
    df: pd.DataFrame,
    median_logM: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split *df* into low-mass and high-mass subsamples at *median_logM*.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with at least a ``logM`` column.
    median_logM : float or None
        Split point.  Computed from ``df['logM'].median()`` when *None*.

    Returns
    -------
    low_mass : pd.DataFrame
        Rows where ``logM < median_logM``.
    high_mass : pd.DataFrame
        Rows where ``logM >= median_logM``.
    median_logM : float
        The split value used.
    """
    if median_logM is None:
        median_logM = float(df["logM"].median())
    low_mass = df[df["logM"] < median_logM].copy()
    high_mass = df[df["logM"] >= median_logM].copy()
    return low_mass, high_mass, median_logM


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _build_split_data(
    df: pd.DataFrame,
    median_logM: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict, dict]:
    """Compute delta_f3, split by mass, and return statistics for both halves.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns ``slope_tail``, ``logM``, ``delta_mass_std``.
    median_logM : float or None
        Optional fixed split point; computed from the data when *None*.

    Returns
    -------
    low_mass, high_mass, median_logM_used, stats_low, stats_high
    """
    df = df.copy()
    df["delta_f3"] = df["slope_tail"] - BETA_REF
    low_mass, high_mass, median_logM_used = split_by_mass(df, median_logM)
    stats_low = compute_stats(
        low_mass["delta_mass_std"].to_numpy(),
        low_mass["delta_f3"].to_numpy(),
    )
    stats_high = compute_stats(
        high_mass["delta_mass_std"].to_numpy(),
        high_mass["delta_f3"].to_numpy(),
    )
    return low_mass, high_mass, median_logM_used, stats_low, stats_high


def _render_figure(
    low_mass: pd.DataFrame,
    high_mass: pd.DataFrame,
    stats_low: dict,
    stats_high: dict,
    median_logM: float,
    out_path: Path,
) -> plt.Figure:
    """Draw the two-panel figure from pre-computed split data and save to disk.

    Internal helper used by both ``generate_figure`` and ``main``.
    Both a PNG (at *out_path*) and a sibling PDF are written.

    Parameters
    ----------
    low_mass, high_mass : pd.DataFrame
        Per-galaxy subsamples with columns ``delta_mass_std`` and ``delta_f3``.
    stats_low, stats_high : dict
        Output of ``compute_stats`` for each subsample.
    median_logM : float
        Mass split value used as the panel titles.
    out_path : Path
        Destination path for the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_x = pd.concat([low_mass["delta_mass_std"], high_mass["delta_mass_std"]])
    all_y = pd.concat([low_mass["delta_f3"], high_mass["delta_f3"]])
    x_lim = (float(all_x.min()) - 0.1, float(all_x.max()) + 0.1)
    y_lim = (float(all_y.min()) - 0.1, float(all_y.max()) + 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # ------ Left panel: low mass ------
    ax1.scatter(
        low_mass["delta_mass_std"], low_mass["delta_f3"],
        color="black", s=30, alpha=0.7,
    )
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title(f"Low Mass (logM < {median_logM:.2f})", fontsize=12)
    ax1.set_xlabel(r"$\delta_{\rm mass,std}$", fontsize=11)
    ax1.set_ylabel(r"$\Delta F_3$", fontsize=11)
    text_low = (
        f"N = {stats_low['n']}\n"
        f"$\\beta$ = {stats_low['ols_slope']:.3f} (p = {stats_low['p_ols']:.3f})\n"
        f"$\\rho$ = {stats_low['rho']:.2f} (p = {stats_low['p_spear']:.2f})"
    )
    ax1.text(
        0.05, 0.95, text_low, transform=ax1.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ------ Right panel: high mass ------
    ax2.scatter(
        high_mass["delta_mass_std"], high_mass["delta_f3"],
        color="black", s=30, alpha=0.7,
    )
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    # Regression line
    x_range = np.linspace(
        float(high_mass["delta_mass_std"].min()),
        float(high_mass["delta_mass_std"].max()),
        100,
    )
    y_fit = stats_high["ols_slope"] * x_range + stats_high["ols_intercept"]
    ax2.plot(
        x_range, y_fit, "r--", linewidth=1.5,
        label=f"$\\beta$ = {stats_high['ols_slope']:.3f}",
    )

    ax2.set_title(f"High Mass (logM \u2265 {median_logM:.2f})", fontsize=12)
    ax2.set_xlabel(r"$\delta_{\rm mass,std}$", fontsize=11)
    text_high = (
        f"N = {stats_high['n']}\n"
        f"$\\beta$ = {stats_high['ols_slope']:.3f} (p = {stats_high['p_ols']:.3f})\n"
        f"$\\rho$ = {stats_high['rho']:.2f} (p = {stats_high['p_spear']:.3f})"
    )
    ax2.text(
        0.05, 0.95, text_high, transform=ax2.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax2.legend(loc="lower right")

    ax1.set_xlim(*x_lim)
    ax1.set_ylim(*y_lim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    return fig


def generate_figure(
    df: pd.DataFrame,
    out_path: str | Path,
    median_logM: float | None = None,
) -> plt.Figure:
    """Generate and save the split-by-mass two-panel scatter figure.

    Both a PNG (at *out_path*) and a sibling PDF are written.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns ``slope_tail``, ``logM``, ``delta_mass_std``.
        ``delta_f3`` is computed internally as ``slope_tail − BETA_REF``.
    out_path : str or Path
        Destination path for the PNG.  A sibling ``.pdf`` is also saved.
    median_logM : float or None
        Optional fixed split point; computed from the data when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    low_mass, high_mass, median_logM, stats_low, stats_high = _build_split_data(
        df, median_logM
    )
    return _render_figure(
        low_mass, high_mass, stats_low, stats_high, median_logM, Path(out_path)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split-by-mass environmental correlation figure: "
            "delta_f3 vs delta_mass_std for low- and high-mass subsamples."
        )
    )
    parser.add_argument(
        "--csv", default=_CSV_DEFAULT,
        help="Path to per-galaxy catalog CSV (default: data/sparc_subset.csv).",
    )
    parser.add_argument(
        "--out", default=_OUT_DEFAULT,
        help=f"Output PNG path (default: {_OUT_DEFAULT}). A sibling PDF is also saved.",
    )
    parser.add_argument(
        "--median-logM", type=float, default=None, dest="median_logM",
        help="Fix the mass-split point (default: computed from data median).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the split-by-mass figure pipeline.

    Returns
    -------
    dict with keys:
        median_logM     — mass split point used
        stats_low       — statistics dict for the low-mass panel
        stats_high      — statistics dict for the high-mass panel
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

    df = df[list(_REQUIRED_COLS)].dropna()

    out_path = Path(args.out)
    low_mass, high_mass, median_logM, stats_low, stats_high = _build_split_data(
        df, args.median_logM
    )
    _render_figure(low_mass, high_mass, stats_low, stats_high, median_logM, out_path)

    pdf_path = out_path.with_suffix(".pdf")
    print(f"Figure saved as '{out_path}' and '{pdf_path}'")
    print(f"Median logM: {median_logM:.3f}")
    print(
        f"Low  Mass (N={stats_low['n']}): "
        f"rho={stats_low['rho']:.3f} p={stats_low['p_spear']:.3f} "
        f"beta={stats_low['ols_slope']:.3f} p_ols={stats_low['p_ols']:.3f}"
    )
    print(
        f"High Mass (N={stats_high['n']}): "
        f"rho={stats_high['rho']:.3f} p={stats_high['p_spear']:.3f} "
        f"beta={stats_high['ols_slope']:.3f} p_ols={stats_high['p_ols']:.3f}"
    )

    return {
        "median_logM": median_logM,
        "stats_low": stats_low,
        "stats_high": stats_high,
        "figure_path": out_path,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    main()
