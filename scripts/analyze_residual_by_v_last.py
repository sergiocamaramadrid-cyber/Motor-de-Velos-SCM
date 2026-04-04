"""
scripts/analyze_residual_by_v_last.py — Mann-Whitney U test on F3 residuals.

Splits galaxies into two groups by the median of ``v_last`` (last measured
rotation velocity) and tests whether the F3 acceleration residual
(``f3_residual``) differs between those groups using the Mann-Whitney U test.

The test addresses the question: *Does the SCM model residual depend on the
asymptotic rotation velocity of the galaxy?*  A significant result (small p)
indicates that the F3 residual is correlated with ``v_last``, which may point
to model systematics or genuine physical differences between low- and
high-velocity galaxies.

Input
-----
``results/scm_clean_with_residual.csv``  (or ``--input PATH``)
  Required columns: ``f3_residual``, ``v_last``

Output
------
Printed summary:
  - Sample sizes for low- and high-``v_last`` groups
  - Median ``f3_residual`` in each group
  - Mann-Whitney U statistic and two-sided p-value

Usage
-----
::

    python scripts/analyze_residual_by_v_last.py

    python scripts/analyze_residual_by_v_last.py \\
        --input results/scm_clean_with_residual.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "results/scm_clean_with_residual.csv"
REQUIRED_COLS = ["f3_residual", "v_last"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_residual_catalog(csv_path: Path) -> pd.DataFrame:
    """Load and validate the residual catalog.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file produced by ``generate_scm_residual_catalog.py``.

    Returns
    -------
    pd.DataFrame
        Validated catalog with at least ``f3_residual`` and ``v_last`` columns.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Residual catalog not found: {csv_path}\n"
            "Run generate_scm_residual_catalog.py first."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}"
        )
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_residual_by_v_last(df: pd.DataFrame) -> dict:
    """Run the Mann-Whitney U test comparing F3 residuals by v_last group.

    Rows with non-finite ``f3_residual`` or ``v_last`` are dropped before
    analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns ``f3_residual`` and ``v_last``.

    Returns
    -------
    dict
        Keys:
        - ``n_low`` / ``n_high``: sample sizes
        - ``median_low`` / ``median_high``: group medians of ``f3_residual``
        - ``v_last_median``: the split threshold
        - ``statistic``: Mann-Whitney U statistic
        - ``p_value``: two-sided p-value
    """
    df_clean = df[
        np.isfinite(df["f3_residual"]) & np.isfinite(df["v_last"])
    ].copy()

    if df_clean.empty:
        raise ValueError(
            "No finite rows remain after cleaning — cannot run Mann-Whitney U test."
        )

    median_v = df_clean["v_last"].median()

    low = df_clean.loc[df_clean["v_last"] <= median_v, "f3_residual"]
    high = df_clean.loc[df_clean["v_last"] > median_v, "f3_residual"]

    stat, p = mannwhitneyu(low, high, alternative="two-sided")

    return {
        "n_total": len(df_clean),
        "n_low": len(low),
        "n_high": len(high),
        "v_last_median": float(median_v),
        "median_low": float(low.median()),
        "median_high": float(high.median()),
        "statistic": float(stat),
        "p_value": float(p),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    """Print a formatted summary of the Mann-Whitney U test results."""
    sep = "=" * 60
    print(sep)
    print("  Motor de Velos SCM — F3 Residual vs V_last Analysis")
    print(sep)
    print(f"  Total galaxies (clean): {results['n_total']}")
    print(f"  Split threshold v_last median: {results['v_last_median']:.2f} km/s")
    print()
    print("  Low vs High V_last — F3 residual comparison")
    print(f"    n_low  = {results['n_low']},  median f3_residual = {results['median_low']:.4f}")
    print(f"    n_high = {results['n_high']},  median f3_residual = {results['median_high']:.4f}")
    print()
    print(f"  Mann-Whitney U statistic: {results['statistic']:.1f}")
    print(f"  p-value (two-sided):      {results['p_value']:.4g}")
    if results["p_value"] < 0.05:
        print("  → Significant difference (p < 0.05)")
    else:
        print("  → No significant difference (p ≥ 0.05)")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mann-Whitney U test: does f3_residual differ between "
            "low- and high-v_last galaxy groups?"
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        metavar="FILE",
        help=f"Input residual catalog CSV (default: {DEFAULT_INPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: load catalog, run analysis, print results."""
    args = _parse_args(argv)
    csv_path = Path(args.input)
    df = load_residual_catalog(csv_path)
    results = analyze_residual_by_v_last(df)
    print_results(results)
    return results


if __name__ == "__main__":
    main()
