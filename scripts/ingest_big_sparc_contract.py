"""
scripts/ingest_big_sparc_contract.py — Passthrough ingestion for BIG-SPARC data.

Reads pre-downloaded BIG-SPARC table files (``galaxies.*`` and
``rc_points.*``), validates the SCM data contract, derives ``vbar_kms`` from
component velocities when needed, and writes a reproducible Parquet file ready
for contract-based pipeline v2 analysis.

Expected input files (auto-detected as CSV or Parquet)
------------------------------------------------------
  <data-dir>/galaxies.{csv,parquet,pq}
  <data-dir>/rc_points.{csv,parquet,pq}

``rc_points`` must contain at minimum the columns required by
:func:`contract_utils.REQUIRED_COLS` (or the component columns
``vgas_kms``, ``vdisk_kms``, ``vbul_kms`` from which ``vbar_kms`` is
derived automatically).

Usage
-----
::

    python scripts/ingest_big_sparc_contract.py \\
        --data-dir data/big_sparc \\
        --out      data/big_sparc/contract

With ``--dry-run`` the script validates the data and prints a summary without
writing any file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from contract_utils import (
    REQUIRED_COLS,
    compute_vbar_kms,
    read_table,
    validate_contract,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TABLE_STEMS = ("galaxies", "rc_points")
_EXTENSIONS = (".csv", ".parquet", ".pq")
OUTPUT_FILENAME = "big_sparc_contract.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_table(data_dir: Path, stem: str) -> Path:
    """Return the first matching file for *stem* with a supported extension."""
    for ext in _EXTENSIONS:
        candidate = data_dir / (stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find '{stem}' table in {data_dir}. "
        f"Expected one of: {[stem + e for e in _EXTENSIONS]}"
    )


def _merge_tables(galaxies: pd.DataFrame, rc_points: pd.DataFrame) -> pd.DataFrame:
    """Merge galaxy-level and per-point tables on the 'galaxy' key."""
    key = "galaxy"
    if key not in rc_points.columns:
        raise ValueError(
            f"rc_points table missing join key '{key}'. "
            "Cannot merge with galaxies table."
        )
    # galaxy table may contain metadata we want to propagate; left-join so we
    # keep all rotation-curve points even if the galaxy table is a subset.
    if key in galaxies.columns:
        # avoid duplicating columns that exist in both tables
        extra_cols = [key] + [
            c for c in galaxies.columns if c != key and c not in rc_points.columns
        ]
        merged = rc_points.merge(galaxies[extra_cols], on=key, how="left")
    else:
        merged = rc_points.copy()
    return merged


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def ingest(data_dir: str | Path, out_dir: str | Path | None = None,
           dry_run: bool = False) -> pd.DataFrame:
    """Run the BIG-SPARC contract ingestion.

    Parameters
    ----------
    data_dir:
        Directory containing ``galaxies.*`` and ``rc_points.*`` files.
    out_dir:
        Directory where ``big_sparc_contract.parquet`` is written.  Ignored
        when *dry_run* is ``True``.
    dry_run:
        If ``True``, validate only — do not write output.

    Returns
    -------
    pd.DataFrame
        Contract-validated, sorted DataFrame.
    """
    data_dir = Path(data_dir)

    # --- locate and read input tables ----------------------------------------
    gal_path = _find_table(data_dir, "galaxies")
    rc_path = _find_table(data_dir, "rc_points")

    print(f"  Reading galaxies  : {gal_path}")
    print(f"  Reading rc_points : {rc_path}")

    galaxies = read_table(gal_path)
    rc_points = read_table(rc_path)

    # --- merge ---------------------------------------------------------------
    df = _merge_tables(galaxies, rc_points)

    # --- derive vbar_kms if absent ------------------------------------------
    df = compute_vbar_kms(df)

    # --- validate contract ---------------------------------------------------
    validate_contract(df)

    # --- sort for reproducibility -------------------------------------------
    df = df.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)

    n_galaxies = df["galaxy"].nunique()
    n_points = len(df)
    print(
        f"  Contract OK — {n_galaxies} galaxies, {n_points} radial points"
    )

    # --- write output --------------------------------------------------------
    if not dry_run:
        out_path = Path(out_dir) / OUTPUT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"  Written : {out_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest pre-downloaded BIG-SPARC tables into a contract-compliant "
            "Parquet file ready for SCM pipeline v2."
        )
    )
    parser.add_argument(
        "--data-dir", default="data/big_sparc", metavar="DIR",
        help="Directory containing galaxies.* and rc_points.* (default: data/big_sparc).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help=(
            "Output directory for big_sparc_contract.parquet "
            "(default: same as --data-dir)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only; do not write any file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    out_dir = args.out if args.out else args.data_dir
    return ingest(data_dir=args.data_dir, out_dir=out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
