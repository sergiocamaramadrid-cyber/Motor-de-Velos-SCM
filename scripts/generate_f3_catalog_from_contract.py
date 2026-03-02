"""
scripts/generate_f3_catalog_from_contract.py — F3 catalog generator from a contract-compliant table.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.contract_utils import read_table, validate_contract

_VFLAT_MIN_DEFAULT: float = 80.0
_MBAR_MAX_DEFAULT: float = 10.5
_MIN_DEEP_DEFAULT: int = 3
_VBAR_DEEP_DEFAULT: float = 50.0


def _compute_galaxy_stats(sub: pd.DataFrame, min_deep: int, vbar_deep: float) -> dict:
    galaxy = sub["galaxy"].iloc[0]
    n_points = len(sub)

    r_thresh = sub["r_kpc"].quantile(0.80)
    outer = sub[sub["r_kpc"] >= r_thresh]
    vflat = float(outer["vobs_kms"].median()) if len(outer) > 0 else float(sub["vobs_kms"].median())

    vbar_max = float(sub["vbar_kms"].abs().max())
    log_mbar_proxy = 4.0 * np.log10(max(vbar_max, 1e-6))

    deep_mask = sub["vbar_kms"].abs() < vbar_deep
    deep_pts = sub[deep_mask]
    deep_n = int(deep_mask.sum())

    deep_slope: float = float("nan")
    if deep_n >= min_deep:
        log_vbar = np.log10(deep_pts["vbar_kms"].abs().clip(lower=1e-6).values)
        log_vobs = np.log10(deep_pts["vobs_kms"].abs().clip(lower=1e-6).values)
        if np.std(log_vbar) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(log_vbar, log_vobs, 1)
            deep_slope = float(coeffs[0])

    return {
        "galaxy": galaxy,
        "n_points": n_points,
        "vflat_kms": round(vflat, 3),
        "log_mbar_proxy": round(log_mbar_proxy, 4),
        "deep_n": deep_n,
        "deep_slope": round(deep_slope, 4) if not np.isnan(deep_slope) else float("nan"),
    }


def generate_catalog(
    input_path: Path,
    out_dir: Path,
    vflat_min: float = _VFLAT_MIN_DEFAULT,
    mbar_max: float = _MBAR_MAX_DEFAULT,
    min_deep: int = _MIN_DEEP_DEFAULT,
    vbar_deep: float = _VBAR_DEEP_DEFAULT,
) -> pd.DataFrame:
    df = read_table(input_path)
    validate_contract(df, source=str(input_path))

    rows = []
    for _, sub in df.groupby("galaxy", sort=True):
        rows.append(_compute_galaxy_stats(sub, min_deep=min_deep, vbar_deep=vbar_deep))
    catalog = pd.DataFrame(rows)

    has_slope = catalog["deep_slope"].notna()
    catalog["f3_flag"] = (
        (catalog["vflat_kms"] >= vflat_min)
        & (catalog["log_mbar_proxy"] <= mbar_max)
        & has_slope
        & (catalog["deep_slope"] < 0.6)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "f3_catalog.csv", index=False)
    return catalog


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate F3 catalog from contract-compliant table.")
    parser.add_argument("--input", required=True, metavar="FILE", help="Contract-compliant table (Parquet or CSV).")
    parser.add_argument("--out", default=".", metavar="DIR", help="Output directory for f3_catalog.csv (default: current dir).")
    parser.add_argument("--vflat-min", type=float, default=_VFLAT_MIN_DEFAULT, dest="vflat_min", metavar="FLOAT")
    parser.add_argument("--mbar-max", type=float, default=_MBAR_MAX_DEFAULT, dest="mbar_max", metavar="FLOAT")
    parser.add_argument("--min-deep", type=int, default=_MIN_DEEP_DEFAULT, dest="min_deep", metavar="INT")
    parser.add_argument("--vbar-deep", type=float, default=_VBAR_DEEP_DEFAULT, dest="vbar_deep", metavar="FLOAT")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    catalog = generate_catalog(
        input_path=Path(args.input),
        out_dir=Path(args.out),
        vflat_min=args.vflat_min,
        mbar_max=args.mbar_max,
        min_deep=args.min_deep,
        vbar_deep=args.vbar_deep,
    )
    n_f3 = int(catalog["f3_flag"].sum())
    print(
        f"F3 catalog: {len(catalog)} galaxies, {n_f3} flagged as F3 (vflat_min={args.vflat_min}, mbar_max={args.mbar_max}, min_deep={args.min_deep})"
    )
    print(f"Written to: {Path(args.out) / 'f3_catalog.csv'}")


if __name__ == "__main__":
    main()
