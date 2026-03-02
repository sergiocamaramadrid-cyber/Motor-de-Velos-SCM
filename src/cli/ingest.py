"""
ingest.py — Unified survey ingest CLI.

Usage
-----
    python src/cli/ingest.py --survey sparc   --input data/SPARC   --out data/SPARC/contract_v2
    python src/cli/ingest.py --survey big-sparc --input data/BIG-SPARC --out data/BIG-SPARC/contract

Options
-------
    --survey     sparc | big-sparc
    --input      Input directory (rotmod files) or path to catalog file
    --out        Output directory; receives galaxies.parquet and rc_points.parquet
    --instrument Instrument tag (optional, e.g. 'ASKAP')
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when this file is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.contract_utils import ensure_dir
from src.adapters.base import IngestConfig
from src.adapters.sparc import SPARCAdapter
from src.adapters.big_sparc import BIGSPARCAdapter


ADAPTERS = {
    "sparc":     SPARCAdapter(),
    "big-sparc": BIGSPARCAdapter(),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SCM ingest — convert survey data to the internal parquet contract."
    )
    ap.add_argument("--survey",     required=True, choices=list(ADAPTERS),
                    help="Source survey identifier.")
    ap.add_argument("--input",      required=True,
                    help="Input directory (rotmod files) or catalog file path.")
    ap.add_argument("--out",        required=True,
                    help="Output directory for galaxies.parquet and rc_points.parquet.")
    ap.add_argument("--instrument", default=None,
                    help="Instrument tag (optional, e.g. ASKAP).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir    = Path(args.out)
    ensure_dir(out_dir)

    cfg     = IngestConfig(survey=args.survey, instrument=args.instrument)
    adapter = ADAPTERS[args.survey]

    df_gal, df_rc = adapter.ingest(input_path, cfg)

    df_gal.to_parquet(out_dir / "galaxies.parquet",  index=False)
    df_rc.to_parquet(out_dir  / "rc_points.parquet", index=False)

    print("✅ Ingest OK")
    print(" - survey:    ", args.survey)
    print(" - galaxies:  ", len(df_gal))
    print(" - rc_points: ", len(df_rc))


if __name__ == "__main__":
    main()
