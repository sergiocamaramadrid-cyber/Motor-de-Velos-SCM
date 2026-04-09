"""
scripts/plot_env_mass_scan.py — Environmental correlation vs mass-threshold scan.

For each mass threshold in a configurable list the script selects the
high-mass subsample (logM > threshold), computes the Spearman rank correlation
between ``env_proxy`` and ``slope_tail``, and plots the resulting ρ vs
threshold curve with per-point N and p-value annotations.

The default input is ``data/galaxy_catalog_env.csv`` with required columns
``logM``, ``env_proxy``, and ``slope_tail``.

Usage
-----
Default paths::

    python scripts/plot_env_mass_scan.py

Custom paths and thresholds::

    python scripts/plot_env_mass_scan.py \\
        --catalog   data/galaxy_catalog_env.csv \\
        --out       results/fig_env_mass_scan.png \\
        --thresholds 9.8 10.0 10.05 10.2 10.3 \\
        --n-min     10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLDS_DEFAULT: list[float] = [9.8, 10.0, 10.05, 10.2, 10.3]
N_MIN_DEFAULT: int = 10
CATALOG_CSV_DEFAULT = "data/galaxy_catalog_env.csv"
OUTPUT_PNG_DEFAULT = "results/fig_env_mass_scan.png"

_REQUIRED_COLUMNS = {"logM", "env_proxy", "slope_tail"}

FIGURE_CAPTION: str = (
    "Spearman rank correlation between environmental proxy (env_proxy) and "
    "outer-disk slope (slope_tail) as a function of the stellar-mass threshold "
    "(log10 M★/M☉) used to define the high-mass subsample.  "
    "Each point is annotated with the subsample size N and two-tailed p-value. "
    "Produced by the Motor-de-Velos SCM framework."
)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_scan(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    n_min: int = N_MIN_DEFAULT,
) -> pd.DataFrame:
    """Compute Spearman ρ(env_proxy, slope_tail) at each mass threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog with at least columns ``logM``, ``env_proxy``,
        and ``slope_tail``.
    thresholds : list of float, optional
        Mass thresholds (log10 M★) to scan.  Defaults to
        ``THRESHOLDS_DEFAULT``.
    n_min : int
        Minimum subsample size required to compute a correlation.  If
        ``len(sub) < n_min`` the row is populated with NaN for ``rho``
        and ``p``.

    Returns
    -------
    pd.DataFrame
        One row per threshold with columns:
        ``threshold``, ``n``, ``rho``, ``p``.

    Raises
    ------
    ValueError
        If ``df`` is missing any of the required columns.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")

    if thresholds is None:
        thresholds = THRESHOLDS_DEFAULT

    records = []
    for t in thresholds:
        sub = df[df["logM"] > t].dropna(subset=["env_proxy", "slope_tail"])
        n = len(sub)
        if n < n_min:
            records.append({"threshold": t, "n": n, "rho": float("nan"),
                            "p": float("nan")})
            continue
        rho, p = spearmanr(sub["env_proxy"].values, sub["slope_tail"].values)
        records.append({"threshold": t, "n": n, "rho": float(rho),
                        "p": float(p)})

    return pd.DataFrame(records, columns=["threshold", "n", "rho", "p"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_catalog(catalog_csv: str | Path) -> pd.DataFrame:
    """Load the galaxy environment catalog and validate required columns.

    Parameters
    ----------
    catalog_csv : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    catalog_csv = Path(catalog_csv)
    if not catalog_csv.exists():
        raise FileNotFoundError(
            f"Galaxy catalog not found: {catalog_csv}\n"
            "Expected columns: logM, env_proxy, slope_tail."
        )
    df = pd.read_csv(catalog_csv)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog CSV missing required columns: {missing}"
        )
    return df


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(
    scan_df: pd.DataFrame,
    out_path: str | Path,
    thresholds: list[float] | None = None,
) -> plt.Figure:
    """Plot and save the ρ vs mass-threshold scan.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Output of :func:`compute_scan` (columns: threshold, n, rho, p).
    out_path : str or Path
        Destination PNG.  A sibling PDF is written automatically.
    thresholds : list of float, optional
        Kept for API symmetry; the x-axis is taken from ``scan_df.threshold``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = scan_df["threshold"].values
    rhos = scan_df["rho"].values
    ns = scan_df["n"].values
    ps = scan_df["p"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    # Only connect finite rho points
    finite = np.isfinite(rhos)
    ax.plot(xs[finite], rhos[finite], marker="o", color="steelblue",
            linewidth=1.5, markersize=6, label="Spearman ρ")
    # Mark NaN points with a hollow marker
    if not np.all(finite):
        ax.plot(xs[~finite], np.zeros(np.sum(~finite)), marker="x",
                linestyle="none", color="grey", markersize=8,
                label="N < n_min (not computed)")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for i, (x, rho, n, p) in enumerate(zip(xs, rhos, ns, ps)):
        if not math.isfinite(rho):
            label_text = f"N={int(n)}\n(n<n_min)"
            y_pos = 0.0
        else:
            p_str = f"{p:.1e}" if not math.isnan(p) else "nan"
            label_text = f"N={int(n)}\np={p_str}"
            y_pos = rho
        ax.annotate(
            label_text,
            xy=(x, y_pos),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            va="bottom",
        )

    ax.set_xlabel("Mass cut: logM > threshold")
    ax.set_ylabel("Spearman rho (env_proxy vs slope_tail)")
    ax.set_title(
        "SPARC high-mass: environmental correlation vs mass threshold"
    )
    ax.legend(fontsize=9)

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
            "Plot Spearman rho(env_proxy, slope_tail) vs mass threshold "
            "for the SPARC high-mass subsample."
        )
    )
    parser.add_argument(
        "--catalog", default=CATALOG_CSV_DEFAULT,
        help=f"Galaxy environment catalog CSV (default: {CATALOG_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUTPUT_PNG_DEFAULT,
        help=f"Output PNG path (default: {OUTPUT_PNG_DEFAULT}).",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=THRESHOLDS_DEFAULT,
        metavar="T",
        help="Space-separated list of logM thresholds to scan "
             f"(default: {THRESHOLDS_DEFAULT}).",
    )
    parser.add_argument(
        "--n-min", type=int, default=N_MIN_DEFAULT, dest="n_min",
        help=f"Minimum subsample size for a valid correlation (default: {N_MIN_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: load catalog, run scan, generate figure.

    Returns
    -------
    dict with keys:
        scan_df      — pd.DataFrame from :func:`compute_scan`
        thresholds   — list of thresholds used
        n_min        — minimum subsample size used
        figure_path  — absolute path to the PNG (str)
        pdf_path     — absolute path to the sibling PDF (str)
    """
    args = _parse_args(argv)

    df = load_catalog(args.catalog)
    scan_df = compute_scan(df, thresholds=args.thresholds, n_min=args.n_min)
    generate_figure(scan_df, out_path=args.out, thresholds=args.thresholds)

    out = Path(args.out)
    print(f"Saved: {out}")

    return {
        "scan_df": scan_df,
        "thresholds": args.thresholds,
        "n_min": args.n_min,
        "figure_path": str(out.resolve()),
        "pdf_path": str(out.with_suffix(".pdf").resolve()),
    }


if __name__ == "__main__":
    main()
