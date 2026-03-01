"""
scripts/f3_catalog_analysis.py — Statistical analysis of the F3 catalog.

Reads the per-galaxy F3 catalog produced by generate_f3_catalog.py and
prints the following summary block:

    N galaxies analyzed: <N>
    Mean friction_slope: <mean>
    Std friction_slope: <std>
    Consistent (velo_inerte_flag=1): <n_consistent>
    Inconsistent (velo_inerte_flag=0): <n_inconsistent>
    p-value: <p>

The p-value comes from a one-sample t-test of the friction_slope values
against the null hypothesis μ = 0.5 (the MOND/deep-velos prediction).
Only galaxies with a finite (non-NaN) friction_slope are included.

Usage
-----
    python scripts/f3_catalog_analysis.py \\
        --catalog results/f3_catalog_real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

# MOND / deep-velos reference slope
EXPECTED_SLOPE = 0.5


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_catalog(df: pd.DataFrame) -> dict:
    """Compute population statistics for the F3 friction-slope catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with at least columns ``friction_slope`` and
        ``velo_inerte_flag``.

    Returns
    -------
    dict with keys:
        n_analyzed      — int, galaxies with finite friction_slope
        mean_slope      — float, mean friction_slope
        std_slope       — float, sample std of friction_slope
        n_consistent    — int, count where velo_inerte_flag == 1
        n_inconsistent  — int, count where velo_inerte_flag == 0
        p_value         — float, one-sample t-test p-value vs β = 0.5
    """
    slopes = df["friction_slope"].dropna()
    n = len(slopes)

    if n == 0:
        return {
            "n_analyzed": 0,
            "mean_slope": float("nan"),
            "std_slope": float("nan"),
            "n_consistent": 0,
            "n_inconsistent": int(len(df)),
            "p_value": float("nan"),
        }

    mean_slope = float(slopes.mean())
    std_slope = float(slopes.std(ddof=1)) if n > 1 else float("nan")

    n_consistent = int((df["velo_inerte_flag"] == 1).sum())
    n_inconsistent = int((df["velo_inerte_flag"] == 0).sum())

    if n < 2:
        p_value = float("nan")
    else:
        _, p_value = ttest_1samp(slopes.values, EXPECTED_SLOPE)
        p_value = float(p_value)

    return {
        "n_analyzed": n,
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "p_value": p_value,
    }


def format_summary(stats: dict) -> str:
    """Format the statistics dict as the canonical summary block."""
    lines = [
        f"N galaxies analyzed: {stats['n_analyzed']}",
        f"Mean friction_slope: {stats['mean_slope']:.6f}"
        if not np.isnan(stats["mean_slope"])
        else "Mean friction_slope: nan",
        f"Std friction_slope: {stats['std_slope']:.6f}"
        if not np.isnan(stats["std_slope"])
        else "Std friction_slope: nan",
        f"Consistent (velo_inerte_flag=1): {stats['n_consistent']}",
        f"Inconsistent (velo_inerte_flag=0): {stats['n_inconsistent']}",
        f"p-value: {stats['p_value']:.6e}"
        if not np.isnan(stats["p_value"])
        else "p-value: nan",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the F3 per-galaxy friction-slope catalog."
    )
    parser.add_argument(
        "--catalog", required=True, metavar="FILE",
        help="Path to the F3 catalog CSV (output of generate_f3_catalog.py).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run analysis and print summary block.

    Returns the statistics dict so callers can inspect values programmatically.
    """
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}\n"
            "Run generate_f3_catalog.py first."
        )

    df = pd.read_csv(catalog_path)
    required = {"friction_slope", "velo_inerte_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog missing required columns: {missing}. "
            "Regenerate with generate_f3_catalog.py."
        )

    stats = analyze_catalog(df)
    print(format_summary(stats))
    return stats


if __name__ == "__main__":
    main()
