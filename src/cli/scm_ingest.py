"""
scm_ingest.py — Unified ingest CLI for the SCM pipeline.

Converts survey data into the SCM internal contract (galaxies.parquet +
rc_points.parquet).

Supports:
    --survey sparc      — standard SPARC *_rotmod.dat directory
    --survey big-sparc  — BIG-SPARC directory or single catalog file

Usage
-----
    python -m src.cli.scm_ingest \\
        --survey sparc \\
        --data-path data/SPARC \\
        --out-dir   data/SPARC/processed_contract \\
        [--workers 4]

    python -m src.cli.scm_ingest \\
        --survey big-sparc \\
        --data-path data/BIG-SPARC/rotmod_files \\
        --out-dir   data/BIG-SPARC/processed_contract \\
        --workers 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.adapters.big_sparc_adapter import ingest_big_sparc
from scripts.contract_utils import ensure_dir, validate_galaxies_df, validate_rc_points_df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SCM ingest — convert survey data to the internal parquet contract."
    )
    ap.add_argument(
        "--survey", required=True,
        choices=["sparc", "big-sparc"],
        help="Source survey identifier.",
    )
    ap.add_argument(
        "--data-path", required=True,
        help="Directory of *_rotmod.dat files or path to a single catalog file.",
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="Output directory for galaxies.parquet and rc_points.parquet.",
    )
    ap.add_argument(
        "--pattern", default="*_rotmod.dat",
        help="Glob pattern used when data-path is a directory (default: *_rotmod.dat).",
    )
    ap.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default 1).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir   = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[scm_ingest] survey={args.survey}  data-path={data_path}  workers={args.workers}")

    df_gal, df_rc = ingest_big_sparc(
        path=data_path,
        pattern=args.pattern,
        workers=args.workers,
    )

    v1 = validate_galaxies_df(df_gal)
    v2 = validate_rc_points_df(df_rc)
    if not v1.ok or not v2.ok:
        for e in v1.errors + v2.errors:
            print("❌", e)
        raise SystemExit(2)

    gal_path = out_dir / "galaxies.parquet"
    rc_path  = out_dir / "rc_points.parquet"
    df_gal.to_parquet(gal_path, index=False)
    df_rc.to_parquet(rc_path,   index=False)

    print("✅ Ingest OK")
    print(f"   galaxies  → {gal_path}  ({len(df_gal)} rows)")
    print(f"   rc_points → {rc_path}   ({len(df_rc)} rows)")


if __name__ == "__main__":
    main()
