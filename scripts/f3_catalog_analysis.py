"""
scripts/f3_catalog_analysis.py — Ensemble statistics for the F3 friction-slope catalog.

Reads the per-galaxy catalog produced by generate_f3_catalog.py and computes:

  • Ensemble mean and standard deviation of friction_slope
  • Median friction_slope
  • One-sample t-test against the MOND prediction β = 0.5
  • Fraction of galaxies with velo_inerte_flag = 1
    (velo_inerte_flag = 1 → fitted β consistent with β = 0.5 within 2σ;
     velo_inerte_flag = 0 → fitted β deviates from β = 0.5 by ≥ 2σ)

Usage
-----
::

    python scripts/f3_catalog_analysis.py

    python scripts/f3_catalog_analysis.py \\
        --catalog results/f3_catalog.csv \\
        --ref-slope 0.5 \\
        --out results/f3_catalog_stats.csv
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

CATALOG_DEFAULT: str = "results/f3_catalog.csv"
OUT_DEFAULT: str = "results/f3_catalog_stats.csv"
REF_SLOPE: float = 0.5   # MOND / deep-velos prediction
_SEP = "=" * 64

# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_catalog(
    catalog: pd.DataFrame,
    ref_slope: float = REF_SLOPE,
) -> dict:
    """Compute ensemble statistics for friction_slope values.

    Parameters
    ----------
    catalog : pd.DataFrame
        Per-galaxy catalog with at least a ``friction_slope`` column.
    ref_slope : float
        Reference slope for the one-sample t-test (default: 0.5).

    Returns
    -------
    dict with keys:
        n_galaxies          — total galaxies in catalog
        n_fitted            — galaxies with a valid (non-NaN) friction_slope
        mean_slope          — ensemble mean of friction_slope
        std_slope           — ensemble std of friction_slope
        median_slope        — ensemble median of friction_slope
        t_stat              — t-statistic from one-sample t-test vs ref_slope
        p_value             — two-tailed p-value of the t-test
        velo_inerte_frac    — fraction of fitted galaxies whose velo_inerte_flag = 1
                              (i.e. fitted β is consistent with β = 0.5 within 2σ)
        ref_slope           — reference slope used for t-test
    """
    required = {"friction_slope"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")

    slopes = catalog["friction_slope"].dropna().to_numpy(dtype=float)
    n_galaxies = len(catalog)
    n_fitted = len(slopes)

    if n_fitted == 0:
        return {
            "n_galaxies": n_galaxies,
            "n_fitted": 0,
            "mean_slope": float("nan"),
            "std_slope": float("nan"),
            "median_slope": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "velo_inerte_frac": float("nan"),
            "ref_slope": ref_slope,
        }

    mean_slope = float(np.mean(slopes))
    std_slope = float(np.std(slopes, ddof=1)) if n_fitted > 1 else float("nan")
    median_slope = float(np.median(slopes))

    if n_fitted >= 2:
        t_stat, p_value = ttest_1samp(slopes, ref_slope)
        t_stat = float(t_stat)
        p_value = float(p_value)
    else:
        t_stat = float("nan")
        p_value = float("nan")

    # velo_inerte_frac: fraction of fitted galaxies with velo_inerte_flag = 1
    # (flag = 1 → β consistent with 0.5 within 2σ; flag = 0 → deviant by ≥ 2σ)
    if "velo_inerte_flag" in catalog.columns:
        flags = catalog.loc[catalog["friction_slope"].notna(), "velo_inerte_flag"]
        n_flagged = int((flags == 1.0).sum())
        velo_inerte_frac = n_flagged / n_fitted if n_fitted > 0 else float("nan")
    else:
        velo_inerte_frac = float("nan")

    return {
        "n_galaxies": n_galaxies,
        "n_fitted": n_fitted,
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "median_slope": median_slope,
        "t_stat": t_stat,
        "p_value": p_value,
        "velo_inerte_frac": float(velo_inerte_frac),
        "ref_slope": ref_slope,
    }


def format_report(stats: dict) -> list[str]:
    """Format the analysis report as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — F3 Catalog Ensemble Analysis",
        _SEP,
        f"  Galaxies in catalog  : {stats['n_galaxies']}",
        f"  Galaxies fitted      : {stats['n_fitted']}",
    ]
    if stats["n_fitted"] == 0:
        lines += ["", "  ⚠️  No fitted galaxies — cannot compute ensemble statistics.", _SEP]
        return lines

    lines += [
        "",
        f"  Mean β               : {stats['mean_slope']:.4f}",
        f"  Std β                : {stats['std_slope']:.4f}",
        f"  Median β             : {stats['median_slope']:.4f}",
        f"  Reference β (MOND)   : {stats['ref_slope']:.4f}",
        "",
        "  One-sample t-test (H0: mean β = ref β)",
        f"    t-statistic        : {stats['t_stat']:.4f}",
        f"    p-value            : {stats['p_value']:.4e}",
    ]
    if not np.isnan(stats["p_value"]):
        if stats["p_value"] < 0.05:
            lines.append(
                f"    → Significant deviation from β={stats['ref_slope']} "
                f"(p={stats['p_value']:.4e} < 0.05)"
            )
        else:
            lines.append(
                f"    → No significant deviation from β={stats['ref_slope']} "
                f"(p={stats['p_value']:.4e} ≥ 0.05)"
            )
    if not np.isnan(stats["velo_inerte_frac"]):
        lines += [
            "",
            f"  Velo-inerte fraction : {stats['velo_inerte_frac']:.3f}",
        ]
    lines.append(_SEP)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble statistics and t-test for the F3 friction-slope catalog."
    )
    parser.add_argument(
        "--catalog", default=CATALOG_DEFAULT,
        help=f"Per-galaxy F3 catalog CSV (default: {CATALOG_DEFAULT}).",
    )
    parser.add_argument(
        "--ref-slope", type=float, default=REF_SLOPE, dest="ref_slope",
        help=f"Reference slope for the t-test (default: {REF_SLOPE}).",
    )
    parser.add_argument(
        "--out", default=OUT_DEFAULT,
        help=f"Output stats CSV path (default: {OUT_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the catalog analysis and return the statistics dict."""
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}\n"
            "Run 'python scripts/generate_f3_catalog.py' first."
        )

    catalog = pd.read_csv(catalog_path)
    stats = analyze_catalog(catalog, ref_slope=args.ref_slope)

    report_lines = format_report(stats)
    for line in report_lines:
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(out_path, index=False)
    print(f"\n  Stats written to {out_path}")

    return stats


if __name__ == "__main__":
    main()
