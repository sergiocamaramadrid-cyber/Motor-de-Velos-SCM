"""
scm_analyze.py — Population analysis CLI.

Reads the per-galaxy β catalog produced by ``scm_catalog`` and writes a
suite of population-level statistics (bias check, β distribution, β vs mass,
β by quality/survey).

Usage
-----
    python -m src.cli.scm_analyze \\
        --catalog results/f3_catalog.parquet \\
        --out-dir results/population
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.analysis.population_stats import (
    beta_summary,
    beta_vs_mass,
    beta_by_quality,
    beta_by_survey,
)
from src.analysis.bias_diagnostics import (
    selection_bias_check,
    n_deep_distribution,
)
from scripts.contract_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SCM analyze — population statistics on a β catalog."
    )
    ap.add_argument("--catalog", required=True,
                    help="Input per-galaxy β catalog parquet.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for CSV summary files.")
    return ap.parse_args()


def main() -> None:
    args    = parse_args()
    cat_path = Path(args.catalog)
    out_dir  = Path(args.out_dir)

    if not cat_path.exists():
        raise SystemExit(f"❌ Catalog not found: {cat_path}")

    ensure_dir(out_dir)

    catalog = pd.read_parquet(cat_path)

    # 1) β summary
    summary = beta_summary(catalog)
    print("\n── β summary ──────────────────────────────")
    print(summary.to_string())
    summary.to_frame("value").to_csv(out_dir / "beta_summary.csv")

    # 2) Bias check
    bias = selection_bias_check(catalog)
    print("\n── selection bias check ────────────────────")
    for k, v in bias.items():
        print(f"  {k}: {v}")
    pd.Series(bias).to_frame("value").to_csv(out_dir / "bias_check.csv")

    # 3) n_deep distribution
    ndep = n_deep_distribution(catalog)
    ndep.to_csv(out_dir / "n_deep_distribution.csv", index=False)

    # 4) Optional: β vs mass (only if log_mstar present)
    if "log_mstar" in catalog.columns:
        bvm = beta_vs_mass(catalog)
        bvm.to_csv(out_dir / "beta_vs_mass.csv", index=False)
        print("\n── β vs mass written.")

    # 5) Optional: β by quality (only if quality present)
    if "quality" in catalog.columns:
        bq = beta_by_quality(catalog)
        bq.to_csv(out_dir / "beta_by_quality.csv", index=False)
        print("── β by quality written.")

    # 6) Optional: β by survey (only if survey present)
    if "survey" in catalog.columns:
        bs = beta_by_survey(catalog)
        bs.to_csv(out_dir / "beta_by_survey.csv", index=False)
        print("── β by survey written.")

    print(f"\n✅ Analysis complete → {out_dir}")


if __name__ == "__main__":
    main()
