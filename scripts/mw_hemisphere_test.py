"""
scripts/mw_hemisphere_test.py — MW N/S galactic hemisphere asymmetry test.

Tests whether Milky Way Cepheid tracers in the northern galactic hemisphere
(lat_deg > 0) show systematically different mean circular velocity than
those in the southern hemisphere (lat_deg <= 0) for stars at galactocentric
radii R >= R_MIN_DEFAULT.

Physical interpretation
-----------------------
A significant hemisphere asymmetry in circular velocity at large radii could
indicate an environmental modulation of the MW outer rotation curve that breaks
north-south symmetry (e.g., due to the Large Magellanic Cloud wake or filament
alignment).

Expected results (from paper)
------------------------------
R >= 5 kpc: Δ ≈ 0.78 km/s, bootstrap 95 % CI ≈ [-0.24, 1.74] km/s

Usage
-----
    python scripts/mw_hemisphere_test.py

    python scripts/mw_hemisphere_test.py \\
        --csv data/mw_cepheids.csv \\
        --r-min 5 \\
        --out results/gaia \\
        --n-boot 2000 \\
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_MIN_DEFAULT: float = 5.0
N_BOOT_DEFAULT: int = 2000

CSV_DEFAULT = "data/mw_cepheids.csv"
OUT_DEFAULT = "results/gaia"

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_hemisphere_delta(
    df: pd.DataFrame,
    lat_col: str = "lat_deg",
    vc_col: str = "Vc_kms",
) -> dict:
    """Compute mean Vc difference between northern and southern hemisphere stars.

    Parameters
    ----------
    df      : DataFrame with lat_col and vc_col columns
    lat_col : column name for galactic latitude [deg]
    vc_col  : column name for circular velocity [km/s]

    Returns
    -------
    dict with keys: n_north, n_south, mean_north, mean_south, delta
        delta = mean_north - mean_south [km/s]
    """
    north = df[df[lat_col] > 0][vc_col].to_numpy(dtype=float)
    south = df[df[lat_col] <= 0][vc_col].to_numpy(dtype=float)

    mean_north = float(np.mean(north))
    mean_south = float(np.mean(south))
    return {
        "n_north":    int(len(north)),
        "n_south":    int(len(south)),
        "mean_north": mean_north,
        "mean_south": mean_south,
        "delta":      mean_north - mean_south,
    }


def bootstrap_hemisphere_delta(
    df: pd.DataFrame,
    lat_col: str = "lat_deg",
    vc_col: str = "Vc_kms",
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 42,
) -> dict:
    """Bootstrap the north-south Vc difference.

    Parameters
    ----------
    df      : DataFrame with lat_col and vc_col columns
    lat_col : galactic latitude column
    vc_col  : circular velocity column
    n_boot  : number of bootstrap resamples
    seed    : random seed

    Returns
    -------
    dict with keys: boot_median, ci_lo, ci_hi, n_boot
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    lat = df[lat_col].to_numpy(dtype=float)
    vc  = df[vc_col].to_numpy(dtype=float)

    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lat_b = lat[idx]
        vc_b  = vc[idx]
        north_b = vc_b[lat_b > 0]
        south_b = vc_b[lat_b <= 0]
        if len(north_b) == 0 or len(south_b) == 0:
            boot_deltas[i] = np.nan
        else:
            boot_deltas[i] = np.mean(north_b) - np.mean(south_b)

    valid = boot_deltas[~np.isnan(boot_deltas)]
    return {
        "boot_median": float(np.median(valid)),
        "ci_lo":       float(np.percentile(valid, 2.5)),
        "ci_hi":       float(np.percentile(valid, 97.5)),
        "n_boot":      n_boot,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the hemisphere asymmetry test and save results.

    Returns
    -------
    dict with keys: delta_result, bootstrap, r_min, out_path
    """
    parser = argparse.ArgumentParser(
        description="MW N/S hemisphere asymmetry test on outer rotation curve"
    )
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Input CSV path")
    parser.add_argument(
        "--r-min",
        type=float,
        default=R_MIN_DEFAULT,
        help="Minimum galactocentric radius [kpc]",
    )
    parser.add_argument("--out", default=OUT_DEFAULT, help="Output directory")
    parser.add_argument(
        "--n-boot", type=int, default=N_BOOT_DEFAULT, help="Bootstrap resamples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df_outer = df[df["R_kpc"] >= args.r_min].copy()

    delta_result = compute_hemisphere_delta(df_outer)
    boot_result  = bootstrap_hemisphere_delta(
        df_outer, n_boot=args.n_boot, seed=args.seed
    )

    stats = [
        ("r_min_kpc",         args.r_min),
        ("n_north",           delta_result["n_north"]),
        ("n_south",           delta_result["n_south"]),
        ("mean_vc_north_kms", delta_result["mean_north"]),
        ("mean_vc_south_kms", delta_result["mean_south"]),
        ("delta_kms",         delta_result["delta"]),
        ("boot_median_kms",   boot_result["boot_median"]),
        ("ci_lo_kms",         boot_result["ci_lo"]),
        ("ci_hi_kms",         boot_result["ci_hi"]),
    ]
    out_df = pd.DataFrame(stats, columns=["stat", "value"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mw_hemisphere_test.csv"
    out_df.to_csv(out_path, index=False)

    return {
        "delta_result": delta_result,
        "bootstrap":    boot_result,
        "r_min":        args.r_min,
        "out_path":     str(out_path),
    }


if __name__ == "__main__":
    result = main()
    dr = result["delta_result"]
    bt = result["bootstrap"]
    print(f"R >= {result['r_min']} kpc: N={dr['n_north']+dr['n_south']}")
    print(f"  North: mean Vc={dr['mean_north']:.2f} km/s (N={dr['n_north']})")
    print(f"  South: mean Vc={dr['mean_south']:.2f} km/s (N={dr['n_south']})")
    print(f"  Δ = {dr['delta']:.2f} km/s")
    print(f"  Bootstrap 95% CI: [{bt['ci_lo']:.2f}, {bt['ci_hi']:.2f}]")
