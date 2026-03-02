"""
scripts/ingest_big_sparc_contract.py — BIG-SPARC passthrough ingestor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.contract_utils import CONTRACT_COLUMNS, compute_vbar_kms, read_table, validate_contract

_GALAXIES_REQUIRED: list[str] = ["galaxy"]
_RC_REQUIRED_BASE: list[str] = ["galaxy", "r_kpc", "vobs_kms", "vobs_err_kms"]
_RC_COMPONENT_COLS: list[str] = ["v_gas", "v_disk"]


def ingest(galaxies_path: Path, rc_points_path: Path, out_dir: Path) -> pd.DataFrame:
    galaxies = read_table(galaxies_path)
    rc = read_table(rc_points_path)

    _check_cols(galaxies, _GALAXIES_REQUIRED, galaxies_path)
    _check_cols(rc, _RC_REQUIRED_BASE, rc_points_path)

    if "vbar_kms" not in rc.columns:
        missing_comp = [c for c in _RC_COMPONENT_COLS if c not in rc.columns]
        if missing_comp:
            raise ValueError(
                f"'vbar_kms' not found in {rc_points_path} and component columns {missing_comp} are also missing — cannot derive baryonic velocity."
            )
        v_bul = rc["v_bul"].values if "v_bul" in rc.columns else None
        rc = rc.copy()
        rc["vbar_kms"] = compute_vbar_kms(rc["v_gas"].values, rc["v_disk"].values, v_bul)

    valid_galaxies = set(galaxies["galaxy"].unique())
    rc_filtered = rc[rc["galaxy"].isin(valid_galaxies)].copy()
    if rc_filtered.empty:
        raise ValueError(
            "After joining galaxies and rc_points tables the result is empty. Check that the 'galaxy' foreign key matches between both files."
        )

    out_df = rc_filtered[CONTRACT_COLUMNS].copy()
    validate_contract(out_df, source=str(rc_points_path))
    out_df = out_df.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "big_sparc_contract.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_df


def _check_cols(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in '{source}': {missing}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BIG-SPARC passthrough ingestor: reads galaxies + rc_points, validates the SCM contract, derives vbar_kms, and writes sorted Parquet output."
        )
    )
    parser.add_argument("--galaxies", required=True, metavar="FILE", help="Path to the galaxies table (CSV or Parquet).")
    parser.add_argument("--rc-points", required=True, dest="rc_points", metavar="FILE", help="Path to the rotation-curve points table (CSV or Parquet).")
    parser.add_argument("--out", required=True, metavar="DIR", help="Output directory for big_sparc_contract.parquet.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_df = ingest(Path(args.galaxies), Path(args.rc_points), Path(args.out))
    print(
        f"Ingested {len(out_df['galaxy'].unique())} galaxies, {len(out_df)} rotation-curve points → {Path(args.out) / 'big_sparc_contract.parquet'}"
    )


if __name__ == "__main__":
    main()
