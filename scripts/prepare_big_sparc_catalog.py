"""
Prepare a BIG-SPARC catalog CSV for run_big_sparc_veil_test.py.

This utility normalizes either:
  1) A table that already contains galaxy/g_obs/g_bar, or
  2) A contract-style table with galaxy/r_kpc/vobs_kms/vbar_kms

into a CSV with the columns expected by the veil test runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from scripts.contract_utils import read_table
except ImportError:
    if __name__ == "__main__":
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from scripts.contract_utils import read_table
    else:
        raise


_REQ_DIRECT = {"galaxy", "g_obs", "g_bar"}
_REQ_CONTRACT = {"galaxy", "r_kpc", "vobs_kms", "vbar_kms"}
_KPC_TO_M = 3.085677581491367e19


def _compute_accel_from_contract(df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQ_CONTRACT - set(df.columns)
    if missing:
        raise ValueError(
            "Input is missing required columns for conversion. "
            f"Need either {_REQ_DIRECT} or {_REQ_CONTRACT}; missing {sorted(missing)}."
        )

    out = df.copy()
    radius_m = out["r_kpc"].astype(float).to_numpy() * _KPC_TO_M
    vobs_mps = out["vobs_kms"].astype(float).to_numpy() * 1_000.0
    vbar_mps = out["vbar_kms"].astype(float).to_numpy() * 1_000.0

    valid = np.isfinite(radius_m) & (radius_m > 0.0)
    if not np.any(valid):
        raise ValueError("No valid rows with r_kpc > 0 available to compute accelerations.")

    g_obs = np.full(len(out), np.nan, dtype=float)
    g_bar = np.full(len(out), np.nan, dtype=float)
    g_obs[valid] = (vobs_mps[valid] ** 2) / radius_m[valid]
    g_bar[valid] = (vbar_mps[valid] ** 2) / radius_m[valid]

    out["g_obs"] = g_obs
    out["g_bar"] = g_bar
    return out


def prepare_catalog(input_path: Path, out_path: Path) -> pd.DataFrame:
    df = read_table(input_path)

    if not _REQ_DIRECT.issubset(df.columns):
        df = _compute_accel_from_contract(df)

    cols = ["galaxy", "g_obs", "g_bar"]
    for opt in ("logMbar", "logSigmaHI_out"):
        if opt in df.columns:
            cols.append(opt)

    out = df[cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["galaxy", "g_obs", "g_bar"])
    out = out[(out["g_obs"] > 0.0) & (out["g_bar"] > 0.0)]
    out = out.sort_values(["galaxy"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data/big_sparc_catalog.csv for the BIG-SPARC veil test."
    )
    parser.add_argument("--input", required=True, help="Input CSV/Parquet table.")
    parser.add_argument(
        "--out",
        default="data/big_sparc_catalog.csv",
        help="Output CSV path (default: data/big_sparc_catalog.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out = prepare_catalog(Path(args.input), Path(args.out))
    print(f"Wrote {len(out)} rows to {Path(args.out)}")


if __name__ == "__main__":
    main()
