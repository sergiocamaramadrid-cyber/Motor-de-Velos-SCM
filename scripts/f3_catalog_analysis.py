"""
scripts/f3_catalog_analysis.py — Statistical analysis of the F3 per-galaxy β catalog.

Reads the per-galaxy friction-slope catalog produced by generate_f3_catalog.py
and prints a fully unambiguous summary that distinguishes:

  - Total catalog rows
  - Galaxies with a valid (finite) slope — "analyzed"
  - Consistent with β = 0.5 within 2σ  (velo_inerte_flag = 1)
  - Inconsistent at ≥ 2σ               (velo_inerte_flag = 0)
  - Insufficient deep-regime data       (velo_inerte_flag = NaN)

Ensemble statistics (mean, std) and a one-sample t-test vs β = 0.5 are
reported for the valid sub-sample.

velo_inerte_flag semantics
--------------------------
  1   → |β − 0.5| ≤ 2·stderr  (consistent with deep-MOND / deep-velos)
  0   → |β − 0.5| > 2·stderr  (statistically inconsistent)
  NaN → n_deep < 2, regression not possible

Usage
-----
::

    python scripts/f3_catalog_analysis.py

    python scripts/f3_catalog_analysis.py \\
        --catalog results/f3_catalog_real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CATALOG_DEFAULT = "results/f3_catalog_real.csv"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_catalog(df: pd.DataFrame) -> dict:
    """Compute summary statistics from a per-galaxy β catalog DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``friction_slope``, ``velo_inerte_flag``.

    Returns
    -------
    dict with keys:
        n_total_rows   — total rows in the catalog
        n_analyzed     — rows with a finite friction_slope
        n_consistent   — velo_inerte_flag == 1
        n_inconsistent — velo_inerte_flag == 0
        n_nan          — velo_inerte_flag is NaN
        beta_mean      — mean of friction_slope over valid rows
        beta_std       — sample std (ddof=1) of friction_slope over valid rows
        t_stat         — one-sample t-statistic vs β = 0.5
        p_value        — two-tailed p-value of the t-test
    """
    required = {"friction_slope", "velo_inerte_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog DataFrame missing required columns: {missing}.\n"
            "Regenerate with generate_f3_catalog.py."
        )

    # total rows
    n_total_rows = len(df)

    # rows with valid (finite) slope
    df_valid = df[np.isfinite(df["friction_slope"])]
    n_analyzed = len(df_valid)

    # flags
    n_consistent = int((df_valid["velo_inerte_flag"] == 1).sum())
    n_inconsistent = int((df_valid["velo_inerte_flag"] == 0).sum())
    n_nan = int(df["velo_inerte_flag"].isna().sum())

    if n_analyzed < 2:
        return {
            "n_total_rows": n_total_rows,
            "n_analyzed": n_analyzed,
            "n_consistent": n_consistent,
            "n_inconsistent": n_inconsistent,
            "n_nan": n_nan,
            "beta_mean": float("nan"),
            "beta_std": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    beta_mean = float(df_valid["friction_slope"].mean())
    beta_std = float(df_valid["friction_slope"].std(ddof=1))
    t_stat, p_value = stats.ttest_1samp(df_valid["friction_slope"], 0.5)

    return {
        "n_total_rows": n_total_rows,
        "n_analyzed": n_analyzed,
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "n_nan": n_nan,
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def print_summary(result: dict) -> None:
    """Print the standardized F3 catalog analysis summary."""
    print("\n=== F3 CATALOG ANALYSIS SUMMARY ===")
    print(f"Total rows in catalog:            {result['n_total_rows']}")
    print(f"Galaxies analyzed (valid slope):  {result['n_analyzed']}")
    print(f"Consistent with β=0.5 (flag=1):   {result['n_consistent']}")
    print(f"Inconsistent (flag=0):            {result['n_inconsistent']}")
    print(f"Insufficient data (NaN):          {result['n_nan']}")
    print("")
    print(f"Mean friction_slope (β): {result['beta_mean']:.6f}")
    print(f"Std friction_slope:      {result['beta_std']:.6f}")
    print(f"t-statistic vs β=0.5:    {result['t_stat']:.6f}")
    print(f"p-value:                 {result['p_value']:.6e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the F3 per-galaxy friction-slope catalog.\n"
            "Prints an unambiguous statistical summary."
        )
    )
    parser.add_argument(
        "--catalog", default=CATALOG_DEFAULT,
        help=f"Per-galaxy β catalog CSV (default: {CATALOG_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the catalog analysis and print results.

    Returns the result dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}\n"
            "Run 'python scripts/generate_f3_catalog.py' first."
        )

    df = pd.read_csv(catalog_path)
    result = analyze_catalog(df)
    print_summary(result)
    return result


if __name__ == "__main__":
    main()
