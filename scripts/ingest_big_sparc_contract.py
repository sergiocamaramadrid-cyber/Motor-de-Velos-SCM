#!/usr/bin/env python3
"""
ingest_big_sparc_contract.py

Normaliza BIG-SPARC (cuando lo descargues) al contrato interno SCM:
- galaxies
- rc_points

NOTE: This ingestor expects pre-downloaded tables; BIG-SPARC download not included.
      Obtain the BIG-SPARC tables independently and place them in --input-dir.

Uso:
python scripts/ingest_big_sparc_contract.py \
  --input-dir data/BIG_SPARC_raw \
  --out-dir data/BIG_SPARC/processed

El input debe contener:
- galaxies.(csv|parquet)
- rc_points.(csv|parquet)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts.contract_utils import (
        ensure_dir,
        read_table,
        validate_galaxies_df,
        validate_rc_points_df,
        compute_vbar_kms,
    )
except ImportError:
    from contract_utils import (  # type: ignore[no-redef]
        ensure_dir,
        read_table,
        validate_galaxies_df,
        validate_rc_points_df,
        compute_vbar_kms,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing galaxies.* and rc_points.*")
    ap.add_argument("--out-dir", required=True, help="Output directory for normalized parquet files")
    return ap.parse_args()


def find_table(input_dir: Path, stem: str) -> Path:
    for ext in [".parquet", ".pq", ".csv"]:
        p = input_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {stem}.(csv|parquet) in {input_dir}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    galaxies_path = find_table(input_dir, "galaxies")
    rc_path = find_table(input_dir, "rc_points")

    df_gal = read_table(galaxies_path)
    df_rc = read_table(rc_path)

    # --- Validate
    v1 = validate_galaxies_df(df_gal)
    v2 = validate_rc_points_df(df_rc)

    if not v1.ok or not v2.ok:
        print("❌ Contract validation failed.")
        for e in v1.errors + v2.errors:
            print(" -", e)
        raise SystemExit(2)

    # --- Ensure vbar exists (derived if needed)
    if "vbar_kms" not in df_rc.columns:
        df_rc["vbar_kms"] = compute_vbar_kms(df_rc)

    # --- Keep only necessary + keep extra cols (harmless)
    # Sorting improves determinism
    df_gal = df_gal.sort_values("galaxy_id").reset_index(drop=True)
    df_rc = df_rc.sort_values(["galaxy_id", "r_kpc"]).reset_index(drop=True)

    out_gal = out_dir / "galaxies.parquet"
    out_rc = out_dir / "rc_points.parquet"

    df_gal.to_parquet(out_gal, index=False)
    df_rc.to_parquet(out_rc, index=False)

    print("✅ Ingest OK")
    print(" - galaxies:", out_gal)
    print(" - rc_points:", out_rc)
    print(f" - N_galaxies: {len(df_gal)}")
    print(f" - N_points:   {len(df_rc)}")


if __name__ == "__main__":
    main()
