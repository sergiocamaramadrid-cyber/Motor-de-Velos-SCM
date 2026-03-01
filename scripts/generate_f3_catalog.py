"""
scripts/generate_f3_catalog.py — Generate the F3 per-galaxy friction-slope catalog.

This is the **official physical pipeline** for producing ``f3_catalog_real.csv``
from real observational data (e.g. SPARC).  It accepts either:

  a) a pre-computed per-radial-point CSV (``--csv``) that already contains
     ``log_g_bar`` and ``log_g_obs`` columns (output of ``run_pipeline``), or

  b) a SPARC data directory (``--data-dir``), in which case the full SCM
     pipeline is run first to generate ``universal_term_comparison_full.csv``
     and then the per-galaxy friction slopes are computed from that file.

For each galaxy a linear regression

    log10(g_obs) = β · log10(g_bar) + const

is fitted over all available radial points, yielding the per-galaxy
``friction_slope`` (β).  A galaxy is flagged as *consistent with β=0.5* when

    |β − 0.5| ≤ 2 · stderr

Output columns: ``galaxy_id``, ``n_points``, ``friction_slope``,
``slope_stderr``, ``flag``.

Usage
-----
From a pre-computed per-radial-point CSV::

    python scripts/generate_f3_catalog.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --out  results/f3_catalog_real.csv

From raw SPARC data (runs the full SCM pipeline internally)::

    python scripts/generate_f3_catalog.py \\
        --data-dir data/SPARC \\
        --out      results/f3_catalog_real.csv

The analysis step is then::

    python scripts/f3_catalog_analysis.py --catalog results/f3_catalog_real.csv
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Expected deep-MOND / deep-velos slope (used for the consistency flag)
BETA_REF: float = 0.5

# Minimum radial points required to attempt a regression
MIN_POINTS: int = 2

# Required columns in the per-radial-point CSV
REQUIRED_COLS: list[str] = ["galaxy", "log_g_bar", "log_g_obs"]


# ---------------------------------------------------------------------------
# Core: compute per-galaxy friction slopes
# ---------------------------------------------------------------------------

def compute_friction_slopes(csv_path: Path) -> pd.DataFrame:
    """Compute per-galaxy friction slopes from a per-radial-point CSV.

    Parameters
    ----------
    csv_path : Path
        CSV with at least columns ``galaxy``, ``log_g_bar``, ``log_g_obs``.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy_id``, ``n_points``, ``friction_slope``,
        ``slope_stderr``, ``flag``.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Per-radial-point CSV not found: {csv_path}\n"
            "Run the SCM pipeline first:  "
            "python -m src.scm_analysis --data-dir data/SPARC --out results/"
        )

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Per-radial-point CSV is missing required columns: {missing}.\n"
            f"Found columns: {df.columns.tolist()}"
        )

    records: list[dict] = []
    for galaxy, gdf in df.groupby("galaxy"):
        log_gbar = gdf["log_g_bar"].values
        log_gobs = gdf["log_g_obs"].values
        n = len(log_gbar)

        if n < MIN_POINTS:
            records.append({
                "galaxy_id": galaxy,
                "n_points": n,
                "friction_slope": float("nan"),
                "slope_stderr": float("nan"),
                "flag": float("nan"),
            })
            continue

        slope, _intercept, _r, _p, stderr = linregress(log_gbar, log_gobs)
        flag = 1 if abs(float(slope) - BETA_REF) <= 2.0 * float(stderr) else 0
        records.append({
            "galaxy_id": galaxy,
            "n_points": n,
            "friction_slope": round(float(slope), 8),
            "slope_stderr": round(float(stderr), 8),
            "flag": flag,
        })

    catalog = pd.DataFrame(records)
    return catalog.sort_values("galaxy_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the F3 per-galaxy friction-slope catalog from a "
            "per-radial-point CSV or from raw SPARC data."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        metavar="FILE",
        help=(
            "Pre-computed per-radial-point CSV with galaxy, log_g_bar, "
            "log_g_obs columns (output of run_pipeline)."
        ),
    )
    source.add_argument(
        "--data-dir",
        metavar="DIR",
        dest="data_dir",
        help=(
            "SPARC data directory.  The full SCM pipeline is run first to "
            "generate universal_term_comparison_full.csv, then friction "
            "slopes are computed from that file."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="FILE",
        help="Output path for the friction-slope catalog CSV.",
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=1.2e-10,
        help="Characteristic velos acceleration in m/s² (default: 1.2e-10). "
             "Used only when --data-dir triggers a pipeline run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pipeline progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run the catalog generation pipeline.

    Returns the catalog DataFrame so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    out_path = Path(args.out)

    if args.data_dir is not None:
        # Run the full SCM pipeline to produce universal_term_comparison_full.csv
        data_dir = Path(args.data_dir)
        # Write pipeline outputs to a sibling directory of --out so that the
        # caller can inspect them; fall back to a temp dir if needed.
        pipeline_out = out_path.parent / "_pipeline_out"
        pipeline_out.mkdir(parents=True, exist_ok=True)

        if not args.quiet:
            print(f"Running SCM pipeline on {data_dir} → {pipeline_out} …",
                  flush=True)

        # Import here to avoid a heavy top-level import when --csv is used.
        from src.scm_analysis import run_pipeline  # noqa: PLC0415
        run_pipeline(data_dir, pipeline_out, a0=args.a0,
                     verbose=not args.quiet)

        csv_path = pipeline_out / "universal_term_comparison_full.csv"
    else:
        csv_path = Path(args.csv)

    if not args.quiet:
        print(f"Computing per-galaxy friction slopes from {csv_path} …", flush=True)

    catalog = compute_friction_slopes(csv_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    n_valid = int(catalog["friction_slope"].notna().sum())
    n_consistent = int((catalog["flag"] == 1).sum())

    if not args.quiet:
        print(f"Catalog written to {out_path}")
        print(f"  Galaxies: {len(catalog)}  valid slopes: {n_valid}  "
              f"consistent with β=0.5: {n_consistent}")

    return catalog


if __name__ == "__main__":
    main()
