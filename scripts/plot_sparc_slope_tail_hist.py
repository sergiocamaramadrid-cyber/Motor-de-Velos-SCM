"""
scripts/plot_sparc_slope_tail_hist.py — Histogram of outer-disk slope_tail for
SPARC high-mass galaxies.

The script merges a SPARC summary table (galaxy, Mstar) with the per-galaxy
outer-disk slope catalog produced by ``sparc_slope_tail.py``, filters to the
high-mass subsample (log10(Mstar) > LOGM_CUT), and saves a histogram of
slope_tail together with a reference axvline.

Usage
-----
Default paths::

    python scripts/plot_sparc_slope_tail_hist.py

Custom paths::

    python scripts/plot_sparc_slope_tail_hist.py \\
        --sparc-csv  data/sparc_basic.csv \\
        --slopes-csv results/slope_tail.csv \\
        --out        results/fig_slope_tail_high_mass.png \\
        --logm-cut   10.0 \\
        --axvline    -0.15 \\
        --bins       15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGM_CUT_DEFAULT: float = 10.0       # log10(Mstar/M_sun) threshold
AXVLINE_DEFAULT: float = -0.15       # reference slope (SCM / Motor-de-Velos)
BINS_DEFAULT: int = 15
SPARC_CSV_DEFAULT = "data/sparc_basic.csv"
SLOPES_CSV_DEFAULT = "results/slope_tail.csv"
OUTPUT_PNG_DEFAULT = "results/fig_slope_tail_high_mass.png"

FIGURE_CAPTION: str = (
    "Distribution of the outer-disk velocity slope (slope_tail) for SPARC "
    "galaxies with log10(M★/M☉) > {logm_cut:.1f}.  "
    "The dashed vertical line marks slope_tail = {axvline:.2f}, the "
    "reference value predicted by the Motor-de-Velos SCM framework."
).format(logm_cut=LOGM_CUT_DEFAULT, axvline=AXVLINE_DEFAULT)

# Required columns
_SPARC_REQUIRED = {"galaxy", "Mstar"}
_SLOPES_REQUIRED = {"galaxy", "slope_tail"}


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_stats(slopes: np.ndarray) -> dict:
    """Compute summary statistics for a slope_tail array.

    Parameters
    ----------
    slopes : array_like
        Array of slope_tail values (already filtered to the target subsample).

    Returns
    -------
    dict with keys:
        n        — number of galaxies
        mean     — arithmetic mean
        median   — median
        std      — standard deviation (ddof=1, NaN if n < 2)
        min      — minimum
        max      — maximum
    """
    slopes = np.asarray(slopes, dtype=float)
    n = int(len(slopes))
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": n,
        "mean": float(np.mean(slopes)),
        "median": float(np.median(slopes)),
        "std": float(np.std(slopes, ddof=1)) if n >= 2 else float("nan"),
        "min": float(np.min(slopes)),
        "max": float(np.max(slopes)),
    }


# ---------------------------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------------------------

def load_and_merge(
    sparc_csv: str | Path,
    slopes_csv: str | Path,
) -> pd.DataFrame:
    """Load the SPARC summary table and slope catalog, merge on ``galaxy``.

    Parameters
    ----------
    sparc_csv : str or Path
        CSV with columns ``galaxy`` and ``Mstar`` (stellar mass in M_sun).
    slopes_csv : str or Path
        CSV with columns ``galaxy`` and ``slope_tail``.

    Returns
    -------
    pd.DataFrame
        Merged table with columns ``galaxy``, ``Mstar``, ``slope_tail``,
        and ``logM``.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    ValueError
        If required columns are missing from either file.
    """
    sparc_csv = Path(sparc_csv)
    slopes_csv = Path(slopes_csv)

    if not sparc_csv.exists():
        raise FileNotFoundError(
            f"SPARC summary CSV not found: {sparc_csv}"
        )
    if not slopes_csv.exists():
        raise FileNotFoundError(
            f"Slope-tail CSV not found: {slopes_csv}\n"
            "Run 'python scripts/sparc_slope_tail.py' first."
        )

    df_sparc = pd.read_csv(sparc_csv)
    df_slopes = pd.read_csv(slopes_csv)

    missing_sparc = _SPARC_REQUIRED - set(df_sparc.columns)
    if missing_sparc:
        raise ValueError(
            f"SPARC CSV missing required columns: {missing_sparc}"
        )

    missing_slopes = _SLOPES_REQUIRED - set(df_slopes.columns)
    if missing_slopes:
        raise ValueError(
            f"Slopes CSV missing required columns: {missing_slopes}"
        )

    df = df_sparc.merge(df_slopes, on="galaxy", how="inner")
    df["logM"] = np.log10(df["Mstar"])
    return df


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(
    df: pd.DataFrame,
    out_path: str | Path,
    logm_cut: float = LOGM_CUT_DEFAULT,
    axvline: float = AXVLINE_DEFAULT,
    bins: int = BINS_DEFAULT,
) -> plt.Figure:
    """Plot and save the slope_tail histogram for the high-mass subsample.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame with columns ``logM`` and ``slope_tail``.
    out_path : str or Path
        Destination PNG file.  A sibling PDF is also written automatically.
    logm_cut : float
        log10(Mstar) threshold; only galaxies with ``logM > logm_cut`` are shown.
    axvline : float
        x-position of the reference vertical line.
    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    high_mass = df[df["logM"] > logm_cut].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(high_mass["slope_tail"], bins=bins, color="steelblue",
            edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(axvline, color="crimson", linestyle="--", linewidth=1.5,
               label=f"slope = {axvline:.2f}")

    ax.set_xlabel("slope_tail")
    ax.set_ylabel("Number of galaxies")
    ax.set_title(
        f"SPARC high-mass slope_tail distribution\n"
        f"(log10(M★) > {logm_cut:.1f}, N = {len(high_mass)})"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a histogram of outer-disk slope_tail for SPARC "
            "high-mass galaxies."
        )
    )
    parser.add_argument(
        "--sparc-csv", default=SPARC_CSV_DEFAULT, dest="sparc_csv",
        help=f"SPARC summary CSV with galaxy + Mstar (default: {SPARC_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--slopes-csv", default=SLOPES_CSV_DEFAULT, dest="slopes_csv",
        help=f"Slope-tail CSV (default: {SLOPES_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUTPUT_PNG_DEFAULT,
        help=f"Output PNG path (default: {OUTPUT_PNG_DEFAULT}).",
    )
    parser.add_argument(
        "--logm-cut", type=float, default=LOGM_CUT_DEFAULT, dest="logm_cut",
        help=f"log10(Mstar) threshold for high-mass subsample (default: {LOGM_CUT_DEFAULT}).",
    )
    parser.add_argument(
        "--axvline", type=float, default=AXVLINE_DEFAULT,
        help=f"Reference slope for axvline (default: {AXVLINE_DEFAULT}).",
    )
    parser.add_argument(
        "--bins", type=int, default=BINS_DEFAULT,
        help=f"Number of histogram bins (default: {BINS_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: merge data, generate figure, and print summary.

    Returns
    -------
    dict with keys:
        stats        — summary statistics dict from :func:`compute_stats`
        logm_cut     — the log10(Mstar) threshold used
        n_merged     — galaxies after inner join
        n_high_mass  — galaxies in the high-mass subsample
        figure_path  — absolute path to the PNG (str)
        pdf_path     — absolute path to the sibling PDF (str)
    """
    args = _parse_args(argv)

    df = load_and_merge(args.sparc_csv, args.slopes_csv)
    high_mass = df[df["logM"] > args.logm_cut]
    stats = compute_stats(high_mass["slope_tail"].values)

    generate_figure(
        df,
        out_path=args.out,
        logm_cut=args.logm_cut,
        axvline=args.axvline,
        bins=args.bins,
    )

    out = Path(args.out)
    print(f"Saved: {out}")

    return {
        "stats": stats,
        "logm_cut": args.logm_cut,
        "n_merged": len(df),
        "n_high_mass": stats["n"],
        "figure_path": str(out.resolve()),
        "pdf_path": str(out.with_suffix(".pdf").resolve()),
    }


if __name__ == "__main__":
    main()
