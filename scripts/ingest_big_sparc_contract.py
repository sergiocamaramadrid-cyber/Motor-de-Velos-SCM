"""
scripts/ingest_big_sparc_contract.py — BIG-SPARC passthrough ingestor.

Reads pre-downloaded BIG-SPARC tables (``galaxies.*`` and ``rc_points.*``),
validates the SCM data contract, derives ``vbar_kms`` via quadrature when
needed, and writes a contract-compliant Parquet file sorted by
``(galaxy, r_kpc)`` for deterministic downstream analysis.

Expected input files (CSV or Parquet, auto-detected)
------------------------------------------------------
galaxies.*
    One row per galaxy.  Must contain at minimum:

    =========  =========================================================
    galaxy     Unique string identifier (e.g. ``"NGC0300"``).
    =========  =========================================================

rc_points.*
    One row per rotation-curve point.  Must contain at minimum:

    =============  ======================================================
    galaxy         Foreign key matching ``galaxies.galaxy``.
    r_kpc          Galactocentric radius (kpc).
    vobs_kms       Observed circular velocity (km/s).
    vobs_err_kms   Uncertainty on ``vobs_kms`` (km/s).
    =============  ======================================================

    Optional velocity-component columns used to derive ``vbar_kms``:

    ========  =============================================================
    v_gas     Gas circular velocity (km/s).
    v_disk    Disk stellar circular velocity (km/s).
    v_bul     Bulge circular velocity (km/s, default 0).
    ========  =============================================================

    If ``vbar_kms`` is already present in ``rc_points.*`` it is used as-is
    and the component columns are not required.

Output
------
<out_dir>/big_sparc_contract.parquet
    Contract-compliant Parquet file sorted by ``(galaxy, r_kpc)``.  Columns
    follow :data:`scripts.contract_utils.CONTRACT_COLUMNS`.

Usage
-----
::

    python -m scripts.ingest_big_sparc_contract \\
        --galaxies data/BIG-SPARC/galaxies.csv \\
        --rc-points data/BIG-SPARC/rc_points.csv \\
        --out data/BIG-SPARC/processed

    # or as a direct script:
    python scripts/ingest_big_sparc_contract.py \\
        --galaxies data/BIG-SPARC/galaxies.csv \\
        --rc-points data/BIG-SPARC/rc_points.csv \\
        --out data/BIG-SPARC/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow both ``python -m scripts.ingest_big_sparc_contract`` and
# ``python scripts/ingest_big_sparc_contract.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.contract_utils import (
    CONTRACT_COLUMNS,
    compute_vbar_kms,
    read_table,
    validate_contract,
)

# ---------------------------------------------------------------------------
# Required columns for each input table
# ---------------------------------------------------------------------------

_GALAXIES_REQUIRED: list[str] = ["galaxy"]

_RC_REQUIRED_BASE: list[str] = ["galaxy", "r_kpc", "vobs_kms", "vobs_err_kms"]

_RC_COMPONENT_COLS: list[str] = ["v_gas", "v_disk"]  # v_bul is optional


# ---------------------------------------------------------------------------
# Core ingestion function
# ---------------------------------------------------------------------------

def ingest(
    galaxies_path: Path,
    rc_points_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """Run the BIG-SPARC passthrough ingestion pipeline.

    Parameters
    ----------
    galaxies_path : Path
        Path to the galaxies table (CSV or Parquet).
    rc_points_path : Path
        Path to the rotation-curve points table (CSV or Parquet).
    out_dir : Path
        Directory where the output Parquet file is written.

    Returns
    -------
    pd.DataFrame
        Contract-compliant DataFrame that was written to disk.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    ValueError
        If required columns are missing or the join produces an empty table.
    """
    # ------------------------------------------------------------------
    # Load input tables
    # ------------------------------------------------------------------
    galaxies = read_table(galaxies_path)
    rc = read_table(rc_points_path)

    # ------------------------------------------------------------------
    # Validate input columns
    # ------------------------------------------------------------------
    _check_cols(galaxies, _GALAXIES_REQUIRED, galaxies_path)
    _check_cols(rc, _RC_REQUIRED_BASE, rc_points_path)

    # ------------------------------------------------------------------
    # Derive vbar_kms when not already present
    # ------------------------------------------------------------------
    if "vbar_kms" not in rc.columns:
        missing_comp = [c for c in _RC_COMPONENT_COLS if c not in rc.columns]
        if missing_comp:
            raise ValueError(
                f"'vbar_kms' not found in {rc_points_path} and component "
                f"columns {missing_comp} are also missing — cannot derive "
                f"baryonic velocity."
            )
        v_bul = rc["v_bul"].values if "v_bul" in rc.columns else None
        rc = rc.copy()
        rc["vbar_kms"] = compute_vbar_kms(
            rc["v_gas"].values,
            rc["v_disk"].values,
            v_bul,
        )

    # ------------------------------------------------------------------
    # Join to ensure only galaxies present in the galaxies table are kept
    # ------------------------------------------------------------------
    valid_galaxies = set(galaxies["galaxy"].unique())
    rc_filtered = rc[rc["galaxy"].isin(valid_galaxies)].copy()

    if rc_filtered.empty:
        raise ValueError(
            "After joining galaxies and rc_points tables the result is empty. "
            "Check that the 'galaxy' foreign key matches between both files."
        )

    # ------------------------------------------------------------------
    # Select and order contract columns (extras are silently dropped)
    # ------------------------------------------------------------------
    out_df = rc_filtered[CONTRACT_COLUMNS].copy()

    # ------------------------------------------------------------------
    # Validate contract compliance
    # ------------------------------------------------------------------
    validate_contract(out_df, source=str(rc_points_path))

    # ------------------------------------------------------------------
    # Sort for deterministic downstream analysis
    # ------------------------------------------------------------------
    out_df = out_df.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "big_sparc_contract.parquet"
    out_df.to_parquet(out_path, index=False)

    return out_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_cols(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in '{source}': {missing}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BIG-SPARC passthrough ingestor: reads galaxies + rc_points, "
            "validates the SCM contract, derives vbar_kms, and writes sorted "
            "Parquet output."
        )
    )
    parser.add_argument(
        "--galaxies",
        required=True,
        metavar="FILE",
        help="Path to the galaxies table (CSV or Parquet).",
    )
    parser.add_argument(
        "--rc-points",
        required=True,
        dest="rc_points",
        metavar="FILE",
        help="Path to the rotation-curve points table (CSV or Parquet).",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for big_sparc_contract.parquet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_df = ingest(
        galaxies_path=Path(args.galaxies),
        rc_points_path=Path(args.rc_points),
        out_dir=Path(args.out),
    )
    print(
        f"Ingested {len(out_df['galaxy'].unique())} galaxies, "
        f"{len(out_df)} rotation-curve points → "
        f"{Path(args.out) / 'big_sparc_contract.parquet'}"
    )


if __name__ == "__main__":
    main()
