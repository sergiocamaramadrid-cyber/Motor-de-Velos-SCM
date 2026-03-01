"""
scripts/f3_catalog_analysis.py — Statistical analysis of the F3 β catalog.

Reads the per-galaxy F3 catalog produced by ``generate_f3_catalog.py`` and
reports the distribution of the deep-regime slope β across the galaxy sample.

On real SPARC LSB data the expected deep-regime value is β ≈ 0.5 (MOND /
Motor-de-Velos deep form).  On synthetic flat-rotation-curve data β ≈ 1.0.

Usage
-----
Official physical-measurement analysis::

    python scripts/f3_catalog_analysis.py \\
        --catalog results/f3_catalog_real.csv

With optional output directory::

    python scripts/f3_catalog_analysis.py \\
        --catalog results/f3_catalog_real.csv \\
        --out results/f3_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

_SEP = "=" * 64
EXPECTED_BETA_MOND = 0.5
ALPHA_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_f3_catalog(df: pd.DataFrame) -> dict:
    """Compute summary statistics of the β distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Per-galaxy catalog as produced by ``generate_f3_catalog.py``.
        Must contain columns ``beta`` and ``reliable``.

    Returns
    -------
    dict with keys:
        n_galaxies        — total galaxies in catalog
        n_reliable        — galaxies with reliable β fit
        beta_mean         — mean β (reliable only)
        beta_median       — median β (reliable only)
        beta_std          — std-dev of β (reliable only)
        beta_mean_all     — mean β (all, ignoring NaN)
        delta_from_mond   — beta_mean − 0.5
        t_stat            — one-sample t-statistic: (β̄ − 0.5) / (σ/√N)
        p_value_ttest     — two-tailed p-value of the t-test vs β = 0.5
        consistent_mond   — True if p_value_ttest > 0.05 (State A); False = State B
    """
    required = {"beta", "reliable"}
    missing = required - set(df.columns)
    if missing:
        # Accept SCM framework canonical names as aliases
        alias = {"beta": "friction_slope", "reliable": "velo_inerte_flag"}
        still_missing = {c for c in missing if alias.get(c, c) not in df.columns}
        if still_missing:
            raise ValueError(f"Catalog missing required columns: {still_missing}")
        df = df.copy()
        if "beta" not in df.columns:
            df["beta"] = df["friction_slope"]
        if "reliable" not in df.columns:
            df["reliable"] = df["velo_inerte_flag"]

    reliable = df[df["reliable"]]["beta"].dropna()
    all_beta = df["beta"].dropna()

    n_galaxies = len(df)
    n_reliable = len(reliable)

    beta_mean = float(reliable.mean()) if n_reliable > 0 else float("nan")
    beta_median = float(reliable.median()) if n_reliable > 0 else float("nan")
    beta_std = float(reliable.std()) if n_reliable > 0 else float("nan")
    beta_mean_all = float(all_beta.mean()) if len(all_beta) > 0 else float("nan")
    delta = beta_mean - EXPECTED_BETA_MOND

    # One-sample t-test: H0: β = 0.5
    if n_reliable >= 2:
        t_result = _scipy_stats.ttest_1samp(reliable.values, EXPECTED_BETA_MOND)
        t_stat = float(t_result.statistic)
        p_value_ttest = float(t_result.pvalue)
    else:
        t_stat = float("nan")
        p_value_ttest = float("nan")

    return {
        "n_galaxies": n_galaxies,
        "n_reliable": n_reliable,
        "beta_mean": beta_mean,
        "beta_median": beta_median,
        "beta_std": beta_std,
        "beta_mean_all": beta_mean_all,
        "delta_from_mond": delta,
        "t_stat": t_stat,
        "p_value_ttest": p_value_ttest,
        "consistent_mond": (p_value_ttest > ALPHA_THRESHOLD) if not np.isnan(p_value_ttest) else False,
    }


def format_analysis_report(stats: dict, catalog_path: str) -> list[str]:
    """Format the analysis report as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — F3 Catalog Analysis",
        _SEP,
        f"  Catalog      : {catalog_path}",
        f"  N galaxies   : {stats['n_galaxies']}",
        f"  N reliable β : {stats['n_reliable']}",
        "",
    ]
    if stats["n_reliable"] > 0:
        lines += [
            f"  β mean       : {stats['beta_mean']:.4f}",
            f"  β median     : {stats['beta_median']:.4f}",
            f"  β std-dev    : {stats['beta_std']:.4f}",
            f"  Expected (MOND): {EXPECTED_BETA_MOND:.4f}",
            f"  Δ from 0.5   : {stats['delta_from_mond']:+.4f}",
        ]
        if not np.isnan(stats.get("t_stat", float("nan"))):
            lines += [
                f"  t-statistic  : {stats['t_stat']:+.4f}",
                f"  p-value      : {stats['p_value_ttest']:.4e}",
            ]
        lines.append("")
        if stats["consistent_mond"]:
            verdict = (
                "✅  Estado A — β distribution consistent with β = 0.5 "
                f"(p = {stats['p_value_ttest']:.3f} > {ALPHA_THRESHOLD})"
            )
        else:
            verdict = (
                f"⚠️  Estado B — β distribution deviates from β = 0.5 "
                f"(β̄ = {stats['beta_mean']:.3f}, p = {stats['p_value_ttest']:.3e} < {ALPHA_THRESHOLD})"
            )
        lines.append(f"  Verdict: {verdict}")
    else:
        lines.append("  ⚠️  No galaxies with reliable β — insufficient deep-regime data.")
    lines.append(_SEP)

    # Machine-readable summary block (canonical format for physical verdict)
    lines += [
        "",
        "=== F3 CATALOG ANALYSIS SUMMARY ===",
        f"Galaxias: {stats['n_reliable']}",
        f"Mean β: {stats['beta_mean']:.4f}" if not np.isnan(stats["beta_mean"])
        else "Mean β: nan",
        f"Std β: {stats['beta_std']:.4f}" if not np.isnan(stats.get("beta_std", float("nan")))
        else "Std β: nan",
        f"p-value: {stats['p_value_ttest']:.4e}" if not np.isnan(stats.get("p_value_ttest", float("nan")))
        else "p-value: nan",
    ]
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of the F3 per-galaxy β catalog."
    )
    parser.add_argument(
        "--catalog", default="results/f3_catalog_real.csv",
        help="Path to the F3 catalog CSV (default: results/f3_catalog_real.csv).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write f3_analysis.csv and .log to this directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the F3 catalog analysis and print results.

    Returns the statistics dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}\n"
            "Run 'python scripts/generate_f3_catalog.py --data-dir data/SPARC "
            "--out results/f3_catalog_real.csv' first."
        )

    df = pd.read_csv(catalog_path)
    stats = analyze_f3_catalog(df)

    report_lines = format_analysis_report(stats, str(catalog_path))
    for line in report_lines:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([stats]).to_csv(out_dir / "f3_analysis.csv", index=False)
        (out_dir / "f3_analysis.log").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        print(f"\n  Results written to {out_dir}")

    return stats


if __name__ == "__main__":
    main()
