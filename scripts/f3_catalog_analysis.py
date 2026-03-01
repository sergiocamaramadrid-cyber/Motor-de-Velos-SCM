"""
scripts/f3_catalog_analysis.py — F3 Catalog Analysis

Reads a per-galaxy catalog CSV that contains at minimum the columns
``friction_slope`` and ``flag``, and prints a summary block covering:

  - total row count
  - number of galaxies with a valid (non-NaN) slope
  - consistency flags (1 = consistent with β=0.5, 0 = inconsistent, NaN = insufficient data)
  - mean and std of friction_slope
  - one-sample t-test of the mean vs. the deep-MOND / deep-velos prediction β=0.5

Usage
-----
Against the deterministic CI fixture (synthetic flat-rotation-curve data,
β≈1 expected — for tooling verification only, not a physical result)::

    python scripts/f3_catalog_analysis.py --catalog results/f3_catalog_synthetic_flat.csv

For the physically targeted SPARC-LSB deep-regime measurement, first generate
the real catalog::

    python scripts/generate_f3_catalog.py --data-dir data/SPARC --out results/f3_catalog_real.csv

then analyze it::

    python scripts/f3_catalog_analysis.py --catalog results/f3_catalog_real.csv

Note
----
``results/f3_catalog_synthetic_flat.csv`` is a committed CI fixture derived
from ``results/universal_term_comparison_full.csv`` (synthetic flat-rotation-
curve data). In that dataset g_obs and g_bar both scale as V²/r with constant
V, so the expected per-galaxy slope is β≈1, not the deep-MOND/deep-velos value
of β=0.5. It should not be interpreted as a physical result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Deep-MOND / deep-velos expected slope
BETA_REF: float = 0.5

# Required columns in the catalog CSV
REQUIRED_COLS: list[str] = ["friction_slope", "flag"]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_catalog(catalog_path: Path) -> dict:
    """Load the F3 catalog and compute summary statistics.

    Parameters
    ----------
    catalog_path : Path
        Path to the per-galaxy catalog CSV.

    Returns
    -------
    dict with keys:
        total_rows, n_valid, n_consistent, n_inconsistent, n_nan,
        mean_slope, std_slope, t_stat, p_val
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    df = pd.read_csv(catalog_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Catalog is missing required columns: {missing}.\n"
            f"Found columns: {df.columns.tolist()}"
        )

    total_rows = len(df)
    valid_mask = df["friction_slope"].notna()
    n_valid = int(valid_mask.sum())
    n_nan = total_rows - n_valid

    # flag column may contain NaN where slope is NaN
    n_consistent = int((df["flag"] == 1).sum())
    n_inconsistent = int((df["flag"] == 0).sum())

    slopes = df.loc[valid_mask, "friction_slope"].values

    if len(slopes) == 0:
        mean_slope = float("nan")
        std_slope = float("nan")
        t_stat = float("nan")
        p_val = float("nan")
    elif len(slopes) == 1:
        mean_slope = float(slopes[0])
        std_slope = float("nan")
        t_stat = float("nan")
        p_val = float("nan")
    else:
        mean_slope = float(np.mean(slopes))
        std_slope = float(np.std(slopes, ddof=1))
        t_stat, p_val = stats.ttest_1samp(slopes, BETA_REF)
        t_stat = float(t_stat)
        p_val = float(p_val)

    return {
        "total_rows": total_rows,
        "n_valid": n_valid,
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "n_nan": n_nan,
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "t_stat": t_stat,
        "p_val": p_val,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary(result: dict) -> None:
    """Print the F3 catalog analysis summary block."""

    def _fmt(v: float) -> str:
        return f"{v:.6f}" if np.isfinite(v) else "nan"

    print()
    print("=== F3 CATALOG ANALYSIS SUMMARY ===")
    print()
    print(f"Total rows in catalog: {result['total_rows']}")
    print(f"Galaxies analyzed (valid slope): {result['n_valid']}")
    print()
    print(f"Consistent with β=0.5 (flag=1): {result['n_consistent']}")
    print(f"Inconsistent (flag=0): {result['n_inconsistent']}")
    print(f"Insufficient data (NaN): {result['n_nan']}")
    print()
    print(f"Mean friction_slope (β): {_fmt(result['mean_slope'])}")
    print(f"Std friction_slope: {_fmt(result['std_slope'])}")
    print(f"t-statistic vs β=0.5: {_fmt(result['t_stat'])}")
    print(f"p-value: {_fmt(result['p_val'])}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F3 per-galaxy catalog analysis — friction slope statistics."
    )
    parser.add_argument(
        "--catalog",
        required=True,
        metavar="FILE",
        help="Per-galaxy catalog CSV with 'friction_slope' and 'flag' columns.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the F3 catalog analysis and print results.

    Returns the result dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)
    result = analyze_catalog(catalog_path)
    print_summary(result)
    return result


if __name__ == "__main__":
    main()
