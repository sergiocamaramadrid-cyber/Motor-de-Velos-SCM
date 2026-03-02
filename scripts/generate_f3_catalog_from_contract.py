from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.contract_utils import compute_vbar_kms, read_table, validate_contract
except ImportError:  # pragma: no cover
    from contract_utils import compute_vbar_kms, read_table, validate_contract

CONV = 1e6 / 3.085677581e19
G0_DEFAULT = 1.2e-10
DEEP_THRESHOLD_DEFAULT = 0.3
# Numerical floor to avoid division-by-zero when malformed rows contain r_kpc<=0.
MIN_SAFE_RADIUS_KPC = 1e-12


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-galaxy F3 catalog.")
    parser.add_argument("--input-dir", required=True, help="Directory with ingested parquet.")
    parser.add_argument("--out", required=True, help="Output catalog (.csv or .parquet).")
    parser.add_argument("--g0", type=float, default=G0_DEFAULT, help="Reference acceleration.")
    parser.add_argument("--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
                        dest="deep_threshold", help="Deep-regime mask factor.")
    parser.add_argument("--min-deep", type=int, default=10, dest="min_deep",
                        help="Minimum deep points required to report beta.")
    return parser.parse_args(argv)


def _beta_for_galaxy(df: pd.DataFrame, g0: float, deep_threshold: float, min_deep: int) -> dict:
    r = df["r_kpc"].to_numpy(dtype=float)
    vbar = df["vbar_kms"].to_numpy(dtype=float)
    vobs = df["v_obs_kms"].to_numpy(dtype=float)

    safe_r = np.maximum(r, MIN_SAFE_RADIUS_KPC)
    g_bar = (vbar ** 2 / safe_r) * CONV
    g_obs = (vobs ** 2 / safe_r) * CONV
    deep = g_bar < (deep_threshold * g0)
    n_deep = int(deep.sum())

    if n_deep >= min_deep:
        beta = float(np.polyfit(np.log10(g_bar[deep]), np.log10(g_obs[deep]), 1)[0])
    else:
        beta = float("nan")

    return {"n_points": int(len(df)), "n_deep": n_deep, "beta": beta}


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir)

    galaxies = read_table(input_dir / "galaxies.parquet").copy()
    rc_points = read_table(input_dir / "rc_points.parquet").copy()
    validate_contract(galaxies, rc_points)

    if "vbar_kms" not in rc_points.columns:
        rc_points["vbar_kms"] = compute_vbar_kms(rc_points)

    rows = []
    for galaxy, gdf in rc_points.sort_values(["galaxy", "r_kpc"]).groupby("galaxy", sort=True):
        stats = _beta_for_galaxy(gdf, args.g0, args.deep_threshold, args.min_deep)
        rows.append({"galaxy": galaxy, **stats})

    catalog = pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True)
    catalog["selected"] = catalog["n_deep"] >= args.min_deep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        catalog.to_parquet(out, index=False)
    else:
        catalog.to_csv(out, index=False)
    return catalog


if __name__ == "__main__":
    main()
