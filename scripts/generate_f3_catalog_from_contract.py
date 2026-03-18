#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_f3_catalog_from_contract.py

Generate an F3 catalog from a contract CSV with radial SPARC-like data.

Expected input columns:
- galaxy
- r
- gbar
- gobs

Optional columns preserved if present:
- SB
- Vobs
- Vbar
- logMbar
- Rdisk
- type
- inclination

Output columns:
- galaxy
- r
- gbar
- gobs
- F3
- delta_f3
plus any optional passthrough columns found in the input.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EPS = 1e-30
KPC_TO_M = 3.085677581e19


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


def _with_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"galaxy", "r", "gbar", "gobs"}.issubset(out.columns):
        return out

    velocity_cols = {"galaxy", "r_kpc", "vbar_kms", "vobs_kms"}
    if velocity_cols.issubset(out.columns):
        r_m = np.maximum(out["r_kpc"].to_numpy(dtype=float) * KPC_TO_M, EPS)
        out["r"] = out["r_kpc"].to_numpy(dtype=float)
        out["gbar"] = (out["vbar_kms"].to_numpy(dtype=float) * 1000.0) ** 2 / r_m
        out["gobs"] = (out["vobs_kms"].to_numpy(dtype=float) * 1000.0) ** 2 / r_m
        return out

    return out


def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = _with_required_columns(df)
    validate_columns(out, ["galaxy", "r", "gbar", "gobs"])

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["galaxy", "r", "gbar", "gobs"]).copy()
    out = out[out["r"] > 0].copy()
    out = out[out["gbar"] > 0].copy()

    out["F3"] = compute_f3(out["gobs"].to_numpy(), out["gbar"].to_numpy())
    out = out.sort_values(["galaxy", "r"]).reset_index(drop=True)

    out["delta_f3"] = out.groupby("galaxy", sort=False)["F3"].diff().astype(float)
    # Backward-compatible aliases kept for existing pipeline consumers.
    out["f3_scm"] = out["F3"].astype(float)
    out["fit_ok"] = np.isfinite(out["F3"].to_numpy(dtype=float))
    out["quality_flag"] = np.where(out["fit_ok"], "ok", "invalid")
    out["beta"] = out["F3"].astype(float)
    out["beta_err"] = np.nan
    out["reliable"] = out["fit_ok"]
    out["friction_slope"] = out["F3"].astype(float)
    out["velo_inerte_flag"] = out["fit_ok"]

    return out


def _resolve_output_path(raw_output: str) -> str:
    output = os.path.expanduser(raw_output)
    if os.path.isdir(output):
        return os.path.join(output, "f3_catalog.csv")

    _, ext = os.path.splitext(output)
    if not ext:
        return os.path.join(output, "f3_catalog.csv")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F3 catalog from contract CSV.")
    parser.add_argument("--input", required=True, help="Input contract CSV")
    parser.add_argument("--output", default="data/f3_catalog.csv", help="Output CSV")
    args = parser.parse_args()

    output_path = _resolve_output_path(args.output)
    ensure_dir(os.path.dirname(output_path) or ".")

    df = pd.read_csv(args.input)
    out = build_catalog(df)
    out.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(out)}")
    print(f"Galaxies: {out['galaxy'].nunique()}")
    print(f"NaN total: {int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
