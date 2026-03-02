"""
scripts/ingest_sparc_contract.py — Convert SPARC rotation-curve data to
the SCM v2 unified data contract (Parquet).

Reads the standard SPARC layout:
  <sparc-dir>/SPARC_Lelli2016c.csv        (or .mrt)
  <sparc-dir>/<Galaxy>_rotmod.dat         (or in <sparc-dir>/raw/)

and writes a single contract-compliant ``sparc_contract.parquet`` ready
for the v2 β-catalog generator.

Output columns (SCM data contract)
-----------------------------------
  galaxy          — galaxy name
  r_kpc           — galactocentric radius (kpc)
  vobs_kms        — observed rotation velocity (km/s)
  vobs_err_kms    — error on observed rotation velocity (km/s)
  vgas_kms        — gas contribution velocity (km/s)
  vdisk_kms       — stellar-disk contribution velocity (km/s)
  vbul_kms        — bulge contribution velocity (km/s)
  vbar_kms        — total baryonic velocity (quadrature sum, km/s)

Usage
-----
::

    python scripts/ingest_sparc_contract.py \\
        --sparc-dir data/SPARC \\
        --out-dir   data/SPARC/processed_contract

With ``--dry-run`` the script validates and reports without writing any file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from contract_utils import compute_vbar_kms, validate_contract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_FILENAME = "sparc_contract.parquet"

_GALAXY_TABLE_CANDIDATES = [
    "SPARC_Lelli2016c.csv",
    "SPARC_Lelli2016c.mrt",
]

# Column mapping from rotmod.dat → contract names
_ROTMOD_NAMES = ["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
                 "SBdisk", "SBbul"]
_ROTMOD_KEEP = ["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]
_CONTRACT_RENAME = {
    "r":          "r_kpc",
    "v_obs":      "vobs_kms",
    "v_obs_err":  "vobs_err_kms",
    "v_gas":      "vgas_kms",
    "v_disk":     "vdisk_kms",
    "v_bul":      "vbul_kms",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_galaxy_table(sparc_dir: Path) -> Path:
    """Return the path to the SPARC galaxy summary table."""
    for name in _GALAXY_TABLE_CANDIDATES:
        for prefix in (sparc_dir, sparc_dir / "raw", sparc_dir / "processed"):
            candidate = prefix / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {sparc_dir}. "
        f"Expected one of: {_GALAXY_TABLE_CANDIDATES}"
    )


def _load_galaxy_table(path: Path) -> pd.DataFrame:
    """Read the SPARC galaxy summary table (CSV or MRT)."""
    sep = "," if path.suffix == ".csv" else r"\s+"
    df = pd.read_csv(path, sep=sep, comment="#")
    if "Galaxy" not in df.columns:
        raise ValueError(
            f"Galaxy table {path} is missing the 'Galaxy' column."
        )
    return df


def _find_rotmod(sparc_dir: Path, galaxy: str) -> Path:
    """Return the rotmod.dat path for *galaxy*, searching standard locations."""
    filename = f"{galaxy}_rotmod.dat"
    for prefix in (sparc_dir, sparc_dir / "raw"):
        candidate = prefix / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Rotation curve '{filename}' not found in {sparc_dir} or {sparc_dir / 'raw'}"
    )


def _read_rotmod(path: Path) -> pd.DataFrame:
    """Read a SPARC ``_rotmod.dat`` file and return contract-renamed columns."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=_ROTMOD_NAMES,
    )[_ROTMOD_KEEP].rename(columns=_CONTRACT_RENAME)
    return df


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def ingest(sparc_dir: str | Path, out_dir: str | Path | None = None,
           dry_run: bool = False) -> pd.DataFrame:
    """Ingest all SPARC galaxies into a single contract-compliant DataFrame.

    Parameters
    ----------
    sparc_dir:
        Root SPARC directory (contains galaxy table and rotmod files).
    out_dir:
        Directory where ``sparc_contract.parquet`` is written.  Ignored when
        *dry_run* is ``True``.
    dry_run:
        If ``True``, validate only — do not write output.

    Returns
    -------
    pd.DataFrame
        Contract-validated, sorted DataFrame.
    """
    sparc_dir = Path(sparc_dir)

    table_path = _find_galaxy_table(sparc_dir)
    print(f"  Galaxy table : {table_path}")
    galaxy_table = _load_galaxy_table(table_path)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    rows: list[pd.DataFrame] = []
    skipped = 0
    for name in galaxy_names:
        try:
            rc_path = _find_rotmod(sparc_dir, name)
        except FileNotFoundError:
            skipped += 1
            continue
        rc = _read_rotmod(rc_path)
        rc.insert(0, "galaxy", name)
        rows.append(rc)

    if not rows:
        raise RuntimeError(
            f"No rotation-curve files found in {sparc_dir}. "
            "Check that *_rotmod.dat files exist."
        )

    df = pd.concat(rows, ignore_index=True)

    # Derive vbar_kms from component columns
    df = compute_vbar_kms(df)

    # Validate contract
    validate_contract(df)

    # Sort for reproducibility
    df = df.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)

    n_galaxies = df["galaxy"].nunique()
    n_points = len(df)
    print(
        f"  N_galaxies   : {n_galaxies}"
        + (f"  ({skipped} skipped — rotmod not found)" if skipped else "")
    )
    print(f"  N_points     : {n_points}")

    if not dry_run:
        out_path = Path(out_dir) / OUTPUT_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"  Written      : {out_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SPARC rotation-curve data to the SCM v2 unified "
            "data contract (Parquet)."
        )
    )
    parser.add_argument(
        "--sparc-dir", default="data/SPARC", metavar="DIR",
        help="Root SPARC directory (default: data/SPARC).",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help=(
            "Output directory for sparc_contract.parquet "
            "(default: <sparc-dir>/processed_contract)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only; do not write any file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    out_dir = (
        args.out_dir
        if args.out_dir
        else str(Path(args.sparc_dir) / "processed_contract")
    )
    return ingest(sparc_dir=args.sparc_dir, out_dir=out_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
