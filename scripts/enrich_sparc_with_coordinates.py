#!/usr/bin/env python3
"""
Enrich consolidated SPARC rotation curves with sky coordinates and distance.

Default I/O:
  metadata: data/SPARC/SPARC_Lelli2016c.csv
  input   : results/SPARC/rotation_curves-v1.0.csv
  output  : results/SPARC/rotation_curves-v1.1-coords.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _resolve_column(df: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    by_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        col = by_lower.get(candidate.lower())
        if col is not None:
            return col
    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}")
    return None


def enrich_with_coordinates(
    input_file: str | Path = "results/SPARC/rotation_curves-v1.0.csv",
    metadata_file: str | Path = "data/SPARC/SPARC_Lelli2016c.csv",
    output_file: str | Path = "results/SPARC/rotation_curves-v1.1-coords.csv",
) -> pd.DataFrame:
    input_path = Path(input_file)
    metadata_path = Path(metadata_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input catalog not found: {input_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"SPARC metadata file not found: {metadata_path}")

    df_curves = pd.read_csv(input_path)
    df_meta = pd.read_csv(metadata_path)

    galaxy_col = _resolve_column(df_curves, ["galaxy"])
    name_col = _resolve_column(df_meta, ["Name", "Galaxy"])
    ra_col = _resolve_column(df_meta, ["RA"])
    dec_col = _resolve_column(df_meta, ["Dec", "DEC"])
    d_col = _resolve_column(df_meta, ["D"])
    type_col = _resolve_column(df_meta, ["Type", "T"], required=False)

    df_curves = df_curves.copy()
    df_meta = df_meta.copy()

    df_curves["galaxy_key"] = (
        df_curves[galaxy_col].astype(str).str.upper().str.strip().str.replace(" ", "", regex=False)
    )
    df_meta["name_key"] = (
        df_meta[name_col].astype(str).str.upper().str.strip().str.replace(" ", "", regex=False)
    )

    columns = ["name_key", ra_col, dec_col, d_col]
    if type_col is not None:
        columns.append(type_col)
    meta_subset = df_meta[columns].drop_duplicates(subset=["name_key"], keep="first").copy()

    rename_map = {
        ra_col: "RA",
        dec_col: "Dec",
        d_col: "D",
    }
    if type_col is not None:
        rename_map[type_col] = "Type"
    meta_subset = meta_subset.rename(columns=rename_map)

    enriched_df = df_curves.merge(meta_subset, left_on="galaxy_key", right_on="name_key", how="left")
    enriched_df = enriched_df.drop(columns=["galaxy_key", "name_key"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_path, index=False)
    return enriched_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich SPARC consolidated rotation curves with RA/Dec/Distance metadata."
    )
    parser.add_argument(
        "--metadata",
        default="data/SPARC/SPARC_Lelli2016c.csv",
        help="Path to SPARC metadata CSV (default: data/SPARC/SPARC_Lelli2016c.csv).",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        default="results/SPARC/rotation_curves-v1.0.csv",
        help="Path to input consolidated curves CSV (default: results/SPARC/rotation_curves-v1.0.csv).",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        default="results/SPARC/rotation_curves-v1.1-coords.csv",
        help="Path to output enriched CSV (default: results/SPARC/rotation_curves-v1.1-coords.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        enriched_df = enrich_with_coordinates(
            input_file=args.input_file,
            metadata_file=args.metadata,
            output_file=args.output_file,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    assigned = enriched_df["RA"].notna().sum() if "RA" in enriched_df.columns else 0
    coverage = 100.0 * assigned / len(enriched_df) if len(enriched_df) else 0.0
    print("-" * 30)
    print("✅ Enriquecimiento completado.")
    print(f"Nuevo artefacto: {args.output_file}")
    print(f"Galaxias con coordenadas asignadas: {coverage:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
