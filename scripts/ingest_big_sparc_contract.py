from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.contract_utils import compute_vbar_kms, read_table, validate_contract
except ImportError:  # pragma: no cover
    from contract_utils import compute_vbar_kms, read_table, validate_contract


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIG-SPARC contract passthrough ingestor.")
    parser.add_argument("--galaxies", required=True, help="Path to galaxies.* table.")
    parser.add_argument("--rc-points", required=True, dest="rc_points",
                        help="Path to rc_points.* table.")
    parser.add_argument("--out-dir", required=True, dest="out_dir",
                        help="Directory where parquet contract tables will be written.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    galaxies = read_table(args.galaxies).copy()
    rc_points = read_table(args.rc_points).copy()
    validate_contract(galaxies, rc_points)

    if "vbar_kms" not in rc_points.columns:
        rc_points["vbar_kms"] = compute_vbar_kms(rc_points)

    galaxies = galaxies.sort_values(["galaxy"]).reset_index(drop=True)
    rc_points = rc_points.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)

    galaxies_path = out_dir / "galaxies.parquet"
    rc_points_path = out_dir / "rc_points.parquet"
    galaxies.to_parquet(galaxies_path, index=False)
    rc_points.to_parquet(rc_points_path, index=False)
    return galaxies_path, rc_points_path


if __name__ == "__main__":
    main()
