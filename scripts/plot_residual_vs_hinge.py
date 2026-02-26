"""
scripts/plot_residual_vs_hinge.py — Standalone diagnostic plot: residual vs hinge.

For every radial point in the per-radial-point audit CSV this script computes:

    hinge    = max(0, log10(a0) − log10(g_bar))
    residual = log10(g_obs) − log10(g_bar)

and produces:

    <out-dir>/audit/residual_vs_hinge.png  — scatter + binned-median plot
    <out-dir>/audit/residual_vs_hinge.csv  — point-level data for traceability

The plot shows whether the velos/hinge term is "cleaning up" residuals in the
deep regime (hinge > 0, where g_bar < a0).  The expected MOND slope is 0.5
because in the deep-MOND limit log g_obs ≈ 0.5·log g_bar + 0.5·log a0,
which implies residual ≈ 0.5·hinge.

Usage
-----
With the per-radial-point CSV produced by the pipeline::

    python scripts/plot_residual_vs_hinge.py \\
        --csv results/universal_term_comparison_full.csv \\
        --out results/

With a custom a0::

    python scripts/plot_residual_vs_hinge.py \\
        --csv results/universal_term_comparison_full.csv \\
        --out results/ \\
        --a0 1.2e-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.scm_analysis import _A0_DEFAULT, _write_residual_vs_hinge


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate residual-vs-hinge diagnostic plot from audit CSV."
    )
    parser.add_argument(
        "--csv", required=True, metavar="FILE",
        help=(
            "Per-radial-point CSV with columns log_g_bar and log_g_obs "
            "(e.g. results/universal_term_comparison_full.csv)."
        ),
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output root directory; plot is written to <out>/audit/.",
    )
    parser.add_argument(
        "--a0", type=float, default=_A0_DEFAULT, metavar="A0",
        help=f"Characteristic acceleration in m/s² (default: {_A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--bins", type=int, default=10, metavar="N",
        help="Number of quantile bins for the binned-median trend (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            "Expected log_g_bar and log_g_obs (produced by run_pipeline)."
        )

    out_dir = Path(args.out)
    audit_dir = out_dir / "audit"
    _write_residual_vs_hinge(df, audit_dir, a0=args.a0, n_bins=args.bins)
    print(f"Diagnostic plot written to {audit_dir / 'residual_vs_hinge.png'}")
    print(f"Point-level CSV written to  {audit_dir / 'residual_vs_hinge.csv'}")


if __name__ == "__main__":
    main()
