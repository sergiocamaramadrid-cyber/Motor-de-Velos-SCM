#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_f3_catalog_from_contract.py

Generate an F3 catalog from a contract CSV with radial SPARC-like data.

Required input columns:
- galaxy
- r
- gbar
- gobs

Optional columns preserved if present:
- SB, Vobs, Vbar, logMbar, Rdisk, type, inclination

Output:
- galaxy, r, gbar, gobs, F3, delta_f3 (+ passthrough columns)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EPS = 1e-30


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_f3(gobs: np.ndarray, gbar: np.ndarray) -> np.ndarray:
    gbar_safe = np.maximum(np.asarray(gbar, dtype=float), EPS)
    gobs_arr = np.asarray(gobs, dtype=float)
    return (gobs_arr - gbar_safe) / gbar_safe


def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df, ["galaxy", "r", "gbar", "gobs"])

    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["galaxy", "r", "gbar", "gobs"]).copy()
    out = out[out["r"] > 0].copy()
    out = out[out["gbar"] > 0].copy()

    out = out.sort_values(["galaxy", "r"]).reset_index(drop=True)
    out["F3"] = compute_f3(out["gobs"].to_numpy(), out["gbar"].to_numpy())

    out["delta_f3"] = (
        out.groupby("galaxy", sort=False)["F3"]
        .diff()
        .astype(float)
    )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F3 catalog from contract CSV.")
    parser.add_argument("--input", required=True, help="Input contract CSV")
    parser.add_argument("--output", default="data/f3_catalog.csv", help="Output CSV")
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.output) or ".")

    df = pd.read_csv(args.input)
    out = build_catalog(df)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(out)}")
    print(f"Galaxies: {out['galaxy'].nunique()}")
    print(f"NaN total: {int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
