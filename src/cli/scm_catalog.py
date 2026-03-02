"""
scm_catalog.py — F3 / β catalog generation CLI.

Reads the SCM contract (galaxies.parquet + rc_points.parquet) produced by
``scm_ingest`` and writes a per-galaxy β catalog parquet.

Usage
-----
    python -m src.cli.scm_catalog \\
        --data-dir data/SPARC/processed_contract \\
        --out      results/f3_catalog.parquet \\
        [--a0 1.2e-10] \\
        [--deep-threshold 0.3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.core.beta_fit import fit_beta_batch, A0_DEFAULT, DEEP_THRESHOLD_DEFAULT
from scripts.contract_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SCM catalog — generate per-galaxy β (deep-slope) catalog."
    )
    ap.add_argument("--data-dir",        required=True,
                    help="Directory with galaxies.parquet and rc_points.parquet.")
    ap.add_argument("--out",             required=True,
                    help="Output parquet path.")
    ap.add_argument("--a0",              type=float, default=A0_DEFAULT,
                    help=f"Characteristic acceleration in m/s² (default {A0_DEFAULT}).")
    ap.add_argument("--deep-threshold",  type=float, default=DEEP_THRESHOLD_DEFAULT,
                    help=f"Deep-regime threshold as fraction of a0 "
                         f"(default {DEEP_THRESHOLD_DEFAULT}).")
    return ap.parse_args()


def main() -> None:
    args    = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    for fname in ("galaxies.parquet", "rc_points.parquet"):
        if not (data_dir / fname).exists():
            raise SystemExit(f"❌ Missing contract file: {data_dir / fname}")

    ensure_dir(out_path.parent)

    df_rc = pd.read_parquet(data_dir / "rc_points.parquet")

    catalog = fit_beta_batch(df_rc, a0=args.a0, threshold=args.deep_threshold)
    catalog.to_parquet(out_path, index=False)

    n_valid = catalog["beta"].notna().sum()
    print("✅ Catalog OK")
    print(f"   N_galaxies   : {len(catalog)}")
    print(f"   N_valid_beta : {n_valid}")
    print(f"   velo_inerte  : {catalog['velo_inerte_flag'].sum()} / {len(catalog)}")
    if n_valid:
        print(f"   beta median  : {catalog['beta'].median():.3f}")
        print(f"   n_deep total : {catalog['n_deep'].sum()}")
    print(f"   output       : {out_path}")


if __name__ == "__main__":
    main()
