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

try:
    from scripts.contract_utils import read_table, validate_contract
except ImportError:
    # Allow direct execution via `python scripts/generate_f3_catalog_from_contract.py`
    # without affecting import-time behavior when used as a library.
    if __name__ == "__main__":
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from scripts.contract_utils import read_table, validate_contract
    else:
        raise
_VFLAT_MIN_DEFAULT: float = 80.0
_MBAR_MAX_DEFAULT: float = 10.5
_MIN_DEEP_DEFAULT: int = 3
_VBAR_DEEP_DEFAULT: float = 50.0
_EXPECTED_SLOPE_DEFAULT: float = 0.5
_TAIL_POINTS_DEFAULT: int = 5


def _compute_galaxy_stats(
    sub: pd.DataFrame,
    min_deep: int,
    vbar_deep: float,
    expected_slope: float,
    tail_points: int,
) -> dict:
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
    tail_n = 0
    tail_r_min = float("nan")
    tail_r_max = float("nan")

    deep_slope: float = float("nan")
    if deep_n >= min_deep:
        tail_pts = deep_pts.sort_values("r_kpc").tail(tail_points)
        tail_n = int(len(tail_pts))
        if tail_n > 0:
            tail_r_min = float(tail_pts["r_kpc"].min())
            tail_r_max = float(tail_pts["r_kpc"].max())
        log_vbar = np.log10(tail_pts["vbar_kms"].abs().clip(lower=1e-6).values)
        log_vobs = np.log10(tail_pts["vobs_kms"].abs().clip(lower=1e-6).values)
        if np.std(log_vbar) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(log_vbar, log_vobs, 1)
            deep_slope = float(coeffs[0])

    delta_f3 = float("nan") if np.isnan(deep_slope) else float(deep_slope - expected_slope)

    return {
        "galaxy": galaxy,
        "n_points": n_points,
        "vflat_kms": round(vflat, 3),
        "log_mbar_proxy": round(log_mbar_proxy, 4),
        "deep_n": deep_n,
        "n_tail_points": tail_n,
        "tail_points_used": int(tail_points),
        "tail_r_min": round(tail_r_min, 4) if not np.isnan(tail_r_min) else float("nan"),
        "tail_r_max": round(tail_r_max, 4) if not np.isnan(tail_r_max) else float("nan"),
        "deep_slope": round(deep_slope, 4) if not np.isnan(deep_slope) else float("nan"),
        "F3_slope": round(deep_slope, 4) if not np.isnan(deep_slope) else float("nan"),
        "expected_slope": expected_slope,
        "delta_f3": round(delta_f3, 4) if not np.isnan(delta_f3) else float("nan"),
    }


def generate_catalog(
    input_path: Path,
    out_dir: Path,
    vflat_min: float = _VFLAT_MIN_DEFAULT,
    mbar_max: float = _MBAR_MAX_DEFAULT,
    min_deep: int = _MIN_DEEP_DEFAULT,
    vbar_deep: float = _VBAR_DEEP_DEFAULT,
    expected_slope: float = _EXPECTED_SLOPE_DEFAULT,
    tail_points: int = _TAIL_POINTS_DEFAULT,
) -> pd.DataFrame:
    if tail_points < 3:
        raise ValueError("tail_points must be >= 3")

    df = read_table(input_path)
    validate_contract(df, source=str(input_path))

    rows = []
    for _, sub in df.groupby("galaxy", sort=True):
        rows.append(
            _compute_galaxy_stats(
                sub,
                min_deep=min_deep,
                vbar_deep=vbar_deep,
                expected_slope=expected_slope,
                tail_points=tail_points,
            )
        )
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
    parser.add_argument(
        "--tail-points",
        type=int,
        default=_TAIL_POINTS_DEFAULT,
        dest="tail_points",
        metavar="INT",
        help=f"Number of outer deep-regime points used to fit deep_slope (default: {_TAIL_POINTS_DEFAULT}, min: 3).",
    )
    args = parser.parse_args(argv)
    if args.tail_points < 3:
        parser.error("--tail-points must be >= 3")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    catalog = generate_catalog(
        input_path=Path(args.input),
        out_dir=Path(args.out),
        vflat_min=args.vflat_min,
        mbar_max=args.mbar_max,
        min_deep=args.min_deep,
        vbar_deep=args.vbar_deep,
        tail_points=args.tail_points,
    )
    n_f3 = int(catalog["f3_flag"].sum())
    print(
        f"F3 catalog: {len(catalog)} galaxies, {n_f3} flagged as F3 (vflat_min={args.vflat_min}, mbar_max={args.mbar_max}, min_deep={args.min_deep}, tail_points={args.tail_points})"
    )
    delta = catalog["delta_f3"].dropna()
    pos = int((delta > 0).sum())
    neg = int((delta < 0).sum())
    tail_counts = catalog.loc[delta.index, "n_tail_points"]
    if len(delta) > 1 and tail_counts.nunique(dropna=True) > 1:
        corr = float(delta.corr(tail_counts))
    else:
        corr = float("nan")
    print(
        "delta_f3 quick stats: "
        f"median={float(delta.median()):.4f}, std={float(delta.std(ddof=1)):.4f}, "
        f"count_gt0={pos}, count_lt0={neg}, corr_with_n_tail_points={corr:.4f}"
    )
    print(f"Written to: {Path(args.out) / 'f3_catalog.csv'}")


if __name__ == "__main__":
    main()
