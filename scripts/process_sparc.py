#!/usr/bin/env python3
"""
Build a consolidated SPARC rotation-curves table from ``*_rotmod.dat`` files.

Default I/O:
  input : data/SPARC
  output: results/SPARC/rotation_curves-v1.0.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd


def _find_rotmod_files(sparc_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (sparc_dir / "rotmod", sparc_dir / "raw", sparc_dir):
        if base.exists():
            candidates.extend(base.glob("*_rotmod.dat"))
    return sorted({p.resolve() for p in candidates})


def _read_rotmod(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if raw.shape[1] >= 6:
        cols = {"radius": 0, "v_obs": 1, "v_gas": 3, "v_disk": 4, "v_bulge": 5}
    elif raw.shape[1] == 5:
        cols = {"radius": 0, "v_obs": 1, "v_gas": 2, "v_disk": 3, "v_bulge": 4}
    else:
        raise ValueError(
            f"Unsupported rotmod format in {path}: expected 5 or >=6 columns, got {raw.shape[1]}"
        )
    return pd.DataFrame({name: raw.iloc[:, idx].astype(float) for name, idx in cols.items()})


def consolidate_sparc(
    input_dir: str | Path = "data/SPARC",
    output_file: str | Path = "results/SPARC/rotation_curves-v1.0.csv",
) -> pd.DataFrame:
    sparc_dir = Path(input_dir)
    if not sparc_dir.exists():
        raise FileNotFoundError(f"SPARC input directory not found: {sparc_dir}")

    files = _find_rotmod_files(sparc_dir)
    if not files:
        raise FileNotFoundError(
            f"No *_rotmod.dat files found in {sparc_dir / 'rotmod'}, {sparc_dir / 'raw'}, or {sparc_dir}"
        )

    rows: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []
    for path in files:
        galaxy = path.name.replace("_rotmod.dat", "")
        try:
            rc = _read_rotmod(path)
        except (pd.errors.ParserError, ValueError) as exc:
            skipped.append((galaxy, str(exc)))
            continue
        rc.insert(0, "galaxy", galaxy)
        rc["v_bar"] = (rc["v_gas"] ** 2 + rc["v_disk"] ** 2 + rc["v_bulge"] ** 2) ** 0.5
        rows.append(rc)

    if not rows:
        raise ValueError(
            f"All {len(files)} rotmod files failed to parse; "
            f"{len(skipped)} files were skipped due to parse/value-conversion errors."
        )

    for galaxy, msg in skipped:
        warnings.warn(f"Skipped {galaxy}: {msg}", stacklevel=2)

    out = pd.concat(rows, ignore_index=True)
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["galaxy", "radius", "v_obs", "v_gas", "v_disk", "v_bulge", "v_bar"]
    )
    out = out[out["radius"] > 0].copy()
    out = out.sort_values(["galaxy", "radius"]).reset_index(drop=True)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate SPARC *_rotmod.dat files into rotation_curves-v1.0.csv."
    )
    parser.add_argument(
        "--input",
        default="data/SPARC",
        help="Directory containing SPARC rotmod data (checks ./rotmod, ./raw, and root).",
    )
    parser.add_argument(
        "--out",
        default="results/SPARC/rotation_curves-v1.0.csv",
        help="Output CSV path (default: results/SPARC/rotation_curves-v1.0.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        out = consolidate_sparc(input_dir=args.input, output_file=args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("-" * 30)
    print("✅ Consolidación SPARC completada.")
    print(f"Nuevo artefacto: {args.out}")
    print(f"Filas: {len(out)}")
    print(f"Galaxias únicas: {out['galaxy'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
