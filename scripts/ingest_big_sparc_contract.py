from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.contract_utils import compute_vbar_kms, read_table, validate_contract
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from contract_utils import compute_vbar_kms, read_table, validate_contract


def ingest_contract(galaxies_path: str | Path, rc_points_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    galaxies = read_table(galaxies_path)
    rc_points = read_table(rc_points_path)

    validate_contract(galaxies, ["galaxy"], "galaxies")
    validate_contract(rc_points, ["galaxy", "r_kpc", "vobs_kms"], "rc_points")

    if "vbar_kms" not in rc_points.columns:
        validate_contract(rc_points, ["vgas_kms", "vdisk_kms", "vbul_kms"], "rc_points")
        rc_points = rc_points.copy()
        rc_points["vbar_kms"] = compute_vbar_kms(rc_points)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    galaxies_out = out_dir / "galaxies.parquet"
    rc_points_out = out_dir / "rc_points.parquet"

    galaxies.sort_values(["galaxy"]).reset_index(drop=True).to_parquet(galaxies_out, index=False)
    rc_points.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True).to_parquet(rc_points_out, index=False)
    return galaxies_out, rc_points_out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest BIG-SPARC contract tables into deterministic parquet files.")
    parser.add_argument("--galaxies", required=True, help="Path to galaxies table (CSV/Parquet).")
    parser.add_argument("--rc-points", required=True, dest="rc_points", help="Path to rc_points table (CSV/Parquet).")
    parser.add_argument("--out", required=True, help="Output directory for deterministic parquet tables.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = _parse_args(argv)
    outputs = ingest_contract(args.galaxies, args.rc_points, args.out)
    print(f"Written: {outputs[0]}")
    print(f"Written: {outputs[1]}")
    return outputs


if __name__ == "__main__":
    main()
