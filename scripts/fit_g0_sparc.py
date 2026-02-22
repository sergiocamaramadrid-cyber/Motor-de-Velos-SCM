#!/usr/bin/env python3
"""
fit_g0_sparc.py — Inspect the SPARC RAR CSV, fit g0, bin, and check the
deep-MOND regime in one go.

Usage
-----
    python scripts/fit_g0_sparc.py
    python scripts/fit_g0_sparc.py --csv data/sparc_rar_sample.csv --out results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scm_analysis import load_sparc_csv, run_pipeline, print_summary


def inspect_csv(path: str) -> pd.DataFrame:
    """Print the rows, columns, and first 5 rows of the CSV."""
    df = pd.read_csv(path)
    print(f"rows:    {len(df)}")
    print(f"columns: {df.columns.tolist()}")
    print("\nhead(5):")
    print(df.head(5).to_string(index=False))
    print()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect SPARC CSV, fit g0, bin, check deep regime."
    )
    parser.add_argument(
        "--csv",
        default="data/sparc_rar_sample.csv",
        help="Path to SPARC RAR CSV (default: data/sparc_rar_sample.csv)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=20,
        help="Number of log-bins (default: 20)",
    )
    parser.add_argument(
        "--deep-pct", type=float, default=20.0,
        help="Percentile threshold for deep regime (default: 20)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Directory to write CSV outputs (optional)",
    )
    args = parser.parse_args()

    # Step 1 – inspect
    print("=== Step 1: CSV inspection ===")
    inspect_csv(args.csv)

    # Step 2-4 – pipeline (fit g0, bin, deep regime)
    results = run_pipeline(
        args.csv,
        n_bins=args.n_bins,
        deep_percentile=args.deep_pct,
    )
    print_summary(results)

    # Optional output
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        results["bins"].to_csv(out / "rar_bins.csv", index=False)
        fit = results["fit"]
        deep = results["deep"]
        lines = [
            f"g0={fit['g0']:.6e}",
            f"g0_err={fit['g0_err']:.6e}",
            f"rms_dex={fit['rms']:.6f}",
            f"n={fit['n']}",
            f"deep_slope={deep['slope']:.6f}",
            f"deep_collapses={deep['collapses']}",
        ]
        (out / "rar_summary.txt").write_text("\n".join(lines) + "\n")
        print(f"Results saved to {out}/")


if __name__ == "__main__":
    main()
