#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER_CANDIDATES = ["logMbar", "r_kpc", "radius_kpc", "bin_mass_log", "bin_radius"]
DELTA_CANDIDATES = ["delta_f3", "DeltaF3", "dF3"]


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No valid column found for {label}. Candidates: {candidates}")


def build_delta_f3_persistence_input(df: pd.DataFrame) -> pd.DataFrame:
    if "galaxy" not in df.columns:
        raise ValueError("Missing required 'galaxy' column in input CSV.")

    order_col = pick_column(df, ORDER_CANDIDATES, "order_var")
    delta_col = pick_column(df, DELTA_CANDIDATES, "delta_f3")

    keep = ["galaxy", order_col, delta_col]
    for optional_col in ["fit_ok", "reliable", "quality_flag"]:
        if optional_col in df.columns:
            keep.append(optional_col)

    out = df[keep].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[order_col, delta_col])

    if "fit_ok" in out.columns:
        out = out[out["fit_ok"].astype(bool)]
    if "reliable" in out.columns:
        out = out[out["reliable"].astype(bool)]

    out = out.rename(
        columns={
            order_col: "order_var",
            delta_col: "delta_f3",
        }
    ).sort_values(["galaxy", "order_var"]).reset_index(drop=True)

    if len(out) < 3:
        raise ValueError("Too few valid rows after filtering to build ΔF3 persistence input.")

    return out


def main(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    out = build_delta_f3_persistence_input(df)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Exported rows: {len(out)}")
    print(f"Columns: {list(out.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
