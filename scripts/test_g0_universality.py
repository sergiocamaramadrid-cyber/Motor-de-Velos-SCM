#!/usr/bin/env python3
"""
test_g0_universality.py — Run the g₀ universality analysis on the SPARC dataset.

Tests whether the RAR acceleration scale g₀ is truly universal across galaxies
or whether it varies systematically with galaxy mass or other properties.

Usage
-----
    python scripts/test_g0_universality.py
    python scripts/test_g0_universality.py --csv data/sparc_rar_sample.csv --out results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.g0_universality import run_universality_analysis, print_universality_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether g₀ is universal across galaxies in the SPARC dataset. "
            "Fits g₀ per galaxy and per mass quartile, then applies a KS test."
        )
    )
    parser.add_argument(
        "--csv",
        default="data/sparc_rar_sample.csv",
        help="Path to SPARC RAR CSV (default: data/sparc_rar_sample.csv)",
    )
    parser.add_argument(
        "--quartiles", type=int, default=4,
        help="Number of mass groups to split galaxies into (default: 4)",
    )
    parser.add_argument(
        "--min-pts", type=int, default=5,
        help="Minimum data points required per galaxy for g₀ fit (default: 5)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Directory to save result CSVs (optional)",
    )
    args = parser.parse_args()

    results = run_universality_analysis(
        args.csv,
        n_quartiles=args.quartiles,
        min_points=args.min_pts,
        out_dir=args.out,
    )
    print_universality_summary(results)

    if args.out:
        print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
