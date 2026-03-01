"""
scripts/f3_catalog_analysis.py — Statistical analysis of the F3 friction-slope catalog.

Theory
------
In the deep-MOND / deep-velos regime (g_bar ≪ a0):

    log10(g_obs) = β · log10(g_bar) + const

where the Motor de Velos framework predicts β = 0.5 universally.

This script tests that prediction empirically:

  1. Computes the sample mean and standard deviation of the per-galaxy β values.
  2. Reports the mean deviation Δ = mean(β) − 0.5.
  3. Lists galaxies flagged as ``velo_inerte`` (|β − 0.5| > 2σ per galaxy).
  4. Runs a one-sample t-test against H₀: mean(β) = 0.5.

Usage
-----
::

    python scripts/f3_catalog_analysis.py

    python scripts/f3_catalog_analysis.py \\
        --csv  results/f3_catalog.csv \\
        --out  results/diagnostics/f3_analysis

Output columns in summary CSV
------------------------------
N, mean_beta, std_beta, delta_mean, n_velo_inerte, t_stat, p_value
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

EXPECTED_SLOPE = 0.5   # MOND / Motor de Velos universal prediction
CSV_DEFAULT = "results/f3_catalog.csv"
_SEP = "=" * 64

# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_catalog(df: pd.DataFrame) -> dict:
    """Compute F3 friction-slope statistics.

    Parameters
    ----------
    df : pd.DataFrame
        F3 catalog with at least the column ``friction_slope``.
        Optional columns ``friction_slope_err`` and ``velo_inerte_flag``
        are used when present.

    Returns
    -------
    dict with keys:
        N               — number of galaxies analysed
        mean_beta       — sample mean of friction_slope
        std_beta        — sample standard deviation of friction_slope
        delta_mean      — mean_beta − 0.5
        n_velo_inerte   — count of galaxies with velo_inerte_flag == True
        t_stat          — one-sample t-statistic (H₀: mean = 0.5)
        p_value         — two-tailed p-value for the t-test
        flagged_df      — DataFrame of flagged galaxies (may be empty)
    """
    clean = df[np.isfinite(df["friction_slope"])].copy()
    beta = clean["friction_slope"].values

    n = len(beta)
    mean_beta = float(beta.mean())
    std_beta = float(beta.std(ddof=1))
    delta = mean_beta - EXPECTED_SLOPE

    # Flagged galaxies (velo_inerte_flag column preferred; fallback to >2σ)
    if "velo_inerte_flag" in clean.columns:
        flagged = clean[clean["velo_inerte_flag"].astype(bool)]
    else:
        flagged = clean[np.abs(clean["friction_slope"] - EXPECTED_SLOPE) > 2 * std_beta]

    t_stat, p_value = ttest_1samp(beta, EXPECTED_SLOPE)

    return {
        "N": n,
        "mean_beta": mean_beta,
        "std_beta": std_beta,
        "delta_mean": delta,
        "n_velo_inerte": len(flagged),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "flagged_df": flagged,
    }


def format_report(result: dict, csv_path: str) -> list[str]:
    """Format the analysis report as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — F3 Friction-Slope Catalog Analysis",
        _SEP,
        f"  CSV             : {csv_path}",
        f"  Expected β      : {EXPECTED_SLOPE}",
        "",
        "===== RESULTADOS F3 CATALOG =====",
        f"N galaxias: {result['N']}",
        f"Media β: {round(result['mean_beta'], 6)}",
        f"Std β: {round(result['std_beta'], 6)}",
        f"Δ medio: {round(result['delta_mean'], 6)}",
        "",
        f"Galaxias con desviación >2σ: {result['n_velo_inerte']}",
    ]

    flagged = result["flagged_df"]
    if len(flagged) > 0:
        show_cols = [c for c in ["galaxy", "friction_slope", "friction_slope_err"]
                     if c in flagged.columns]
        lines.append(flagged[show_cols].to_string(index=False))

    lines += [
        "",
        "Test contra β=0.5",
        f"t: {round(result['t_stat'], 6)}",
        f"p: {round(result['p_value'], 8)}",
        "",
    ]

    # Interpretation
    p = result["p_value"]
    delta = result["delta_mean"]
    std = result["std_beta"]
    if p > 0.05:
        verdict = (
            f"✅  Result compatible with β = 0.5  "
            f"(p = {p:.4f} > 0.05; Δ = {delta:+.4f})"
        )
    elif abs(delta) <= 2 * std / np.sqrt(result["N"]):
        verdict = (
            f"ℹ️   Marginal deviation: p = {p:.4f}, Δ = {delta:+.4f} "
            f"(within 2·SEM)"
        )
    else:
        verdict = (
            f"⚠️   Significant deviation from β = 0.5 detected: "
            f"p = {p:.4e}, Δ = {delta:+.4f}"
        )

    lines += [f"  Verdict: {verdict}", _SEP]
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Statistical analysis of the F3 per-galaxy friction-slope catalog."
        )
    )
    parser.add_argument(
        "--csv", default=CSV_DEFAULT, metavar="FILE",
        help=f"F3 catalog CSV (default: {CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write f3_analysis.csv and f3_analysis.log to this directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the F3 catalog analysis.  Returns the result dict."""
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"F3 catalog not found: {csv_path}\n"
            "Run 'python scripts/generate_f3_catalog.py --synthetic' first."
        )

    df = pd.read_csv(csv_path)
    if "friction_slope" not in df.columns:
        raise ValueError(
            f"CSV missing required column 'friction_slope'.  "
            f"Columns found: {list(df.columns)}"
        )

    result = analyze_catalog(df)
    report_lines = format_report(result, str(csv_path))
    for line in report_lines:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV (scalar statistics only)
        summary = {k: v for k, v in result.items() if k != "flagged_df"}
        pd.DataFrame([summary]).to_csv(out_dir / "f3_analysis.csv", index=False)

        # Flagged-galaxies CSV
        result["flagged_df"].to_csv(out_dir / "f3_flagged.csv", index=False)

        # Log
        (out_dir / "f3_analysis.log").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        print(f"\n  Results written to {out_dir}")

    return result


if __name__ == "__main__":
    main()
