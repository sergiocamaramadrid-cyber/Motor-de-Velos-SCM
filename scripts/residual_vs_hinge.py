"""
scripts/residual_vs_hinge.py — Residual vs Hinge OOS diagnostic (PR #70).

Reads ``oos_per_point.csv`` (produced by ``scripts/audit_scm.py``) and
produces a scatter + binned-median plot of OOS residuals versus the
deep-regime hinge feature::

    hinge = max(0, log10(a0) − log g_bar)

The plot uses OOS residuals only — those from GroupKFold **test** folds,
never in-sample predictions.  This is the key invariant of PR #70: the
scatter must not mix training-fold fitted values with test-fold targets.

OOS residual definition
-----------------------
For each radial point *i* assigned to test fold *k*:

    residual_dex_oos[i] = residual_dex[i] − (X[i] @ coeffs_k + intercept_k)

where ``coeffs_k`` / ``intercept_k`` are OLS coefficients fitted on the
**training** folds (all galaxies NOT in fold *k*).

Usage
-----
Default paths (after running ``audit_scm.py --outdir results/final_audit``)::

    python scripts/residual_vs_hinge.py

Explicit options::

    python scripts/residual_vs_hinge.py \\
        --csv   results/final_audit/oos_per_point.csv \\
        --out   results/final_audit/residual_vs_hinge_oos.png \\
        --bins  12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSV_DEFAULT = "results/final_audit/oos_per_point.csv"
_OUT_DEFAULT = "results/final_audit/residual_vs_hinge_oos.png"
_BINS_DEFAULT = 12
_MIN_BIN_POINTS = 3  # minimum points per bin for a stable median

_REQUIRED_COLS = {"hinge", "residual_dex_oos", "galaxy", "fold"}


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def plot_residual_vs_hinge(
    df: pd.DataFrame,
    out_path: Path,
    n_bins: int = _BINS_DEFAULT,
) -> None:
    """Scatter + binned-median plot of OOS residual vs hinge.

    Parameters
    ----------
    df : pd.DataFrame
        Per-radial-point OOS table.  Must contain columns
        ``hinge`` and ``residual_dex_oos``.
    out_path : Path
        Destination PNG file.
    n_bins : int
        Number of hinge bins for the binned-median overlay.

    Notes
    -----
    * Only matplotlib is used — no seaborn dependency.
    * Plot uses *OOS* residuals (``residual_dex_oos``), never in-sample.
    * Bins with fewer than ``_MIN_BIN_POINTS`` points are skipped.
    """
    hinge = df["hinge"].to_numpy(dtype=float)
    res_oos = df["residual_dex_oos"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter: individual OOS points
    ax.scatter(
        hinge, res_oos,
        s=4, alpha=0.3, color="#1f77b4",
        label="OOS points (test folds)",
    )

    # Binned median overlay
    if len(hinge) >= _MIN_BIN_POINTS:
        bins = np.linspace(hinge.min(), hinge.max(), n_bins + 1)
        bin_centers: list[float] = []
        bin_medians: list[float] = []
        bin_q25: list[float] = []
        bin_q75: list[float] = []
        for i in range(len(bins) - 1):
            # Use <= for the upper bound of the last bin to include hinge.max()
            upper_op = (hinge <= bins[i + 1]) if i == len(bins) - 2 else (hinge < bins[i + 1])
            mask = (hinge >= bins[i]) & upper_op
            if mask.sum() >= _MIN_BIN_POINTS:
                vals = res_oos[mask]
                bin_centers.append(float(0.5 * (bins[i] + bins[i + 1])))
                bin_medians.append(float(np.median(vals)))
                bin_q25.append(float(np.percentile(vals, 25)))
                bin_q75.append(float(np.percentile(vals, 75)))

        if bin_centers:
            bc = np.array(bin_centers)
            bm = np.array(bin_medians)
            bq25 = np.array(bin_q25)
            bq75 = np.array(bin_q75)
            ax.plot(bc, bm, "o-", color="#d62728",
                    linewidth=1.5, markersize=5, label="Binned median")
            ax.fill_between(bc, bq25, bq75,
                            color="#d62728", alpha=0.15, label="IQR [25–75%]")

    # Reference line at zero residual
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)

    # Axis labels + title
    ax.set_xlabel(
        r"hinge  [dex]  "
        r"($= \max(0,\,\log_{10} a_0 - \log\,g_{\rm bar})$)"
    )
    ax.set_ylabel(
        r"OOS residual  [dex]  "
        r"($\log g_{\rm obs} - \log g_{\rm bar} - \hat{f}$)"
    )
    ax.set_title(
        "Residual vs Hinge — OOS only (GroupKFold test folds)"
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Residual vs Hinge OOS diagnostic plot (PR #70). "
            "Reads oos_per_point.csv and produces a scatter + binned-median figure."
        )
    )
    p.add_argument(
        "--csv",
        default=_CSV_DEFAULT,
        metavar="CSV",
        help=f"Path to oos_per_point.csv (default: {_CSV_DEFAULT}).",
    )
    p.add_argument(
        "--out",
        default=_OUT_DEFAULT,
        metavar="PNG",
        help=f"Output PNG path (default: {_OUT_DEFAULT}).",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=_BINS_DEFAULT,
        metavar="N",
        help=f"Number of hinge bins for the median overlay (default: {_BINS_DEFAULT}).",
    )
    return p.parse_args(argv)


def main(argv=None):
    """CLI entry-point for the Residual vs Hinge OOS diagnostic."""
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        print(
            f"ERROR: OOS CSV not found: {csv_path}\n"
            "Run 'python scripts/audit_scm.py --outdir <dir>' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        print(
            f"ERROR: oos_per_point.csv is missing required columns: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["hinge", "residual_dex_oos"]
    )
    if df.empty:
        print(
            "ERROR: No valid rows remain after cleaning oos_per_point.csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_residual_vs_hinge(df, out_path, n_bins=args.bins)
    print(f"Plot written to {out_path}")


if __name__ == "__main__":
    main()
