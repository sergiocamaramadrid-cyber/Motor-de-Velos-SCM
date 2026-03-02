from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.contract_utils import read_table, validate_contract
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from contract_utils import read_table, validate_contract

_CONV = 1e6 / 3.085677581e16


def generate_f3_catalog(
    rc_points_path: str | Path,
    out_dir: str | Path,
    a0: float = 1.2e-10,
    deep_threshold: float = 0.3,
    min_deep: int = 10,
) -> Path:
    rc_points = read_table(rc_points_path)
    validate_contract(rc_points, ["galaxy", "r_kpc", "vobs_kms", "vbar_kms"], "rc_points")

    r = np.maximum(rc_points["r_kpc"].to_numpy(dtype=float), 1e-10)
    g_bar = rc_points["vbar_kms"].to_numpy(dtype=float) ** 2 / r * _CONV
    g_obs = rc_points["vobs_kms"].to_numpy(dtype=float) ** 2 / r * _CONV

    rc_points = rc_points.copy()
    rc_points["log_g_bar"] = np.log10(np.maximum(g_bar, 1e-30))
    rc_points["log_g_obs"] = np.log10(np.maximum(g_obs, 1e-30))
    rc_points["is_deep"] = g_bar < (deep_threshold * a0)

    rows = []
    for galaxy, group in rc_points.groupby("galaxy", sort=True):
        deep = group[group["is_deep"]]
        n_deep = int(len(deep))
        if n_deep >= min_deep:
            slope, intercept = np.polyfit(deep["log_g_bar"], deep["log_g_obs"], 1)
        else:
            slope, intercept = np.nan, np.nan
        rows.append(
            {
                "galaxy": galaxy,
                "n_points": int(len(group)),
                "n_deep": n_deep,
                "beta_slope": float(slope) if np.isfinite(slope) else np.nan,
                "beta_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "f3_catalog.parquet"
    pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True).to_parquet(out_path, index=False)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-galaxy F3/beta catalog from rc_points contract table.")
    parser.add_argument("--rc-points", required=True, dest="rc_points", help="Path to rc_points table (CSV/Parquet).")
    parser.add_argument("--out", required=True, help="Output directory for catalog parquet.")
    parser.add_argument("--a0", type=float, default=1.2e-10, help="Characteristic acceleration in m/s².")
    parser.add_argument("--deep-threshold", type=float, default=0.3, dest="deep_threshold", help="Deep mask threshold as fraction of a0.")
    parser.add_argument("--min-deep", type=int, default=10, dest="min_deep", help="Minimum deep points required to fit beta slope.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    out_path = generate_f3_catalog(
        rc_points_path=args.rc_points,
        out_dir=args.out,
        a0=args.a0,
        deep_threshold=args.deep_threshold,
        min_deep=args.min_deep,
    )
    print(f"Written: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
