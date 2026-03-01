"""
scripts/f3_catalog_analysis.py — Ensemble analysis of the F3 friction-slope catalog.

Reads the per-galaxy catalog produced by generate_f3_catalog.py and computes:
  - Descriptive statistics (mean, std, median) of friction_slope β
  - One-sample t-test against the deep-velos prediction μ = 0.5
  - Count of galaxies with velo_inerte_flag = 1 ("consistent" with β ≈ 0.5)

Canonical printed summary block
--------------------------------
================================================================
  Motor de Velos SCM — F3 Friction-Slope Catalog Analysis
================================================================
  Catalog           : results/f3_catalog_real.csv
  N total rows      : <N_total>
  N analyzed (non-NaN β): <N_analyzed>
  ----------------------------------------------------------------
  mean β            : <mean>
  std β             : <std>
  median β          : <median>
  ----------------------------------------------------------------
  One-sample t-test (H0: μ_β = 0.5)
    t-statistic     : <t>
    p-value         : <p>
    Verdict         : <verdict>
  ----------------------------------------------------------------
  velo_inerte_flag = 1: <count> / <N_analyzed>
================================================================

Usage
-----
With default paths::

    python scripts/f3_catalog_analysis.py

Explicit options::

    python scripts/f3_catalog_analysis.py \\
        --catalog results/f3_catalog_real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_DEFAULT = "results/f3_catalog_real.csv"
EXPECTED_SLOPE = 0.5
_SEP = "=" * 64
_DASH = "-" * 64

# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_catalog(catalog: pd.DataFrame) -> dict:
    """Compute ensemble statistics for the F3 friction-slope catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must have columns: friction_slope, velo_inerte_flag.
        Rows with NaN friction_slope are excluded from analysis.

    Returns
    -------
    dict with keys:
        n_total        — total rows in catalog
        n_analyzed     — rows with non-NaN friction_slope
        mean           — mean β
        std            — std β
        median         — median β
        t_stat         — one-sample t-statistic (H0: μ = 0.5)
        p_value        — two-tailed p-value
        n_consistent   — count of galaxies with velo_inerte_flag == 1
        verdict        — descriptive string based on p-value
    """
    required = {"friction_slope", "velo_inerte_flag"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(
            f"Catalog missing required columns: {missing}."
        )

    n_total = len(catalog)
    valid = catalog.dropna(subset=["friction_slope"])
    n_analyzed = len(valid)

    if n_analyzed < 2:
        return {
            "n_total": n_total,
            "n_analyzed": n_analyzed,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n_consistent": 0,
            "verdict": "⚠️  Insufficient galaxies for ensemble analysis (need ≥ 2).",
        }

    slopes = valid["friction_slope"].to_numpy()
    mean_b = float(np.mean(slopes))
    std_b = float(np.std(slopes, ddof=1))
    median_b = float(np.median(slopes))
    t_stat, p_value = ttest_1samp(slopes, EXPECTED_SLOPE)
    n_consistent = int((valid["velo_inerte_flag"] == 1).sum())

    if p_value > 0.05:
        verdict = (
            f"✅  Mean β = {mean_b:.3f} is consistent with μ = 0.5 "
            f"(p = {p_value:.3f} > 0.05, cannot reject H0)."
        )
    else:
        verdict = (
            f"⚠️  Mean β = {mean_b:.3f} deviates from μ = 0.5 "
            f"(p = {p_value:.3e} ≤ 0.05, reject H0)."
        )

    return {
        "n_total": n_total,
        "n_analyzed": n_analyzed,
        "mean": mean_b,
        "std": std_b,
        "median": median_b,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_consistent": n_consistent,
        "verdict": verdict,
    }


def format_summary(result: dict, catalog_path: str) -> list[str]:
    """Format the analysis summary as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — F3 Friction-Slope Catalog Analysis",
        _SEP,
        f"  Catalog           : {catalog_path}",
        f"  N total rows      : {result['n_total']}",
        f"  N analyzed (non-NaN β): {result['n_analyzed']}",
        _DASH,
    ]
    if not np.isnan(result["mean"]):
        lines += [
            f"  mean β            : {result['mean']:.4f}",
            f"  std β             : {result['std']:.4f}",
            f"  median β          : {result['median']:.4f}",
            _DASH,
            "  One-sample t-test (H0: μ_β = 0.5)",
            f"    t-statistic     : {result['t_stat']:.4f}",
            f"    p-value         : {result['p_value']:.4e}",
            f"    Verdict         : {result['verdict']}",
            _DASH,
            f"  velo_inerte_flag = 1: {result['n_consistent']} / {result['n_analyzed']}",
        ]
    else:
        lines.append(f"  Verdict: {result['verdict']}")
    lines.append(_SEP)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble analysis of the F3 friction-slope catalog."
    )
    parser.add_argument(
        "--catalog", default=CATALOG_DEFAULT,
        help=f"Per-galaxy catalog CSV (default: {CATALOG_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the F3 catalog analysis and print the summary block.

    Returns the result dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}\n"
            "Run 'python scripts/generate_f3_catalog.py' first."
        )

    catalog = pd.read_csv(catalog_path)
    result = analyze_catalog(catalog)

    lines = format_summary(result, str(catalog_path))
    for line in lines:
        print(line)

    return result


if __name__ == "__main__":
    main()
