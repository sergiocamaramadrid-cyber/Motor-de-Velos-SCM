"""
scripts/mw_radial_scan.py — MW Cepheid radial scan for azimuthal velocity modulation.

Tests whether galactic longitude (azimuthal position) correlates with circular
velocity among Milky Way Cepheid tracers as a function of minimum galactocentric
radius.  A significant Spearman(lon_deg, log10(Vc_kms)) correlation at large
radii would indicate azimuthal environmental modulation of the MW rotation curve.

Physical interpretation
-----------------------
At small radii the MW rotation curve is dominated by the bar/bulge and spiral
arms; environmental effects (if any) are expected to emerge in the outer disc.
By raising the minimum radius R_cut, we probe progressively more outer regions
where environmental modulation should be most visible.

Expected results (from paper)
------------------------------
ρ ≈ -0.09 to -0.23 with p ≈ 1e-2 to 1e-4 for R_cut in [5, 20] kpc.

Usage
-----
    python scripts/mw_radial_scan.py

    python scripts/mw_radial_scan.py \\
        --csv data/mw_cepheids.csv \\
        --r-min 5 --r-max 20 --r-step 1 \\
        --out results/gaia
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_SCAN_MIN_DEFAULT: float = 5.0
R_SCAN_MAX_DEFAULT: float = 20.0
R_SCAN_STEP_DEFAULT: float = 1.0
R_MIN_N_DEFAULT: int = 30

CSV_DEFAULT = "data/mw_cepheids.csv"
OUT_DEFAULT = "results/gaia"

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_radial_spearman(
    df: pd.DataFrame,
    r_cut: float,
    lon_col: str = "lon_deg",
    vc_col: str = "Vc_kms",
) -> dict:
    """Compute Spearman(lon_deg, log10(Vc)) for stars with R_kpc >= r_cut.

    Parameters
    ----------
    df      : DataFrame with R_kpc, lon_col, and vc_col columns
    r_cut   : minimum galactocentric radius [kpc]
    lon_col : column name for galactic longitude
    vc_col  : column name for circular velocity [km/s]

    Returns
    -------
    dict with keys: r_cut_kpc, n, rho, pval
    """
    sub = df[df["R_kpc"] >= r_cut]
    n = len(sub)
    if n < 2:
        return {"r_cut_kpc": float(r_cut), "n": n, "rho": float("nan"), "pval": float("nan")}

    log_vc = np.log10(sub[vc_col].to_numpy(dtype=float))
    rho, pval = spearmanr(sub[lon_col].to_numpy(dtype=float), log_vc)
    return {
        "r_cut_kpc": float(r_cut),
        "n":         int(n),
        "rho":       float(rho),
        "pval":      float(pval),
    }


def scan_radii(
    df: pd.DataFrame,
    r_min: float,
    r_max: float,
    r_step: float,
    min_n: int,
) -> pd.DataFrame:
    """Scan minimum radii and collect Spearman correlations.

    Parameters
    ----------
    df     : Cepheid DataFrame
    r_min  : start of radius scan [kpc]
    r_max  : end of radius scan [kpc]
    r_step : step size [kpc]
    min_n  : minimum number of stars required per window

    Returns
    -------
    DataFrame with columns: r_cut_kpc, n, rho, pval
    """
    radii = np.arange(r_min, r_max + r_step / 2, r_step)
    rows = []
    for r_cut in radii:
        result = compute_radial_spearman(df, r_cut)
        if result["n"] >= min_n and not math.isnan(result["rho"]):
            rows.append(result)
    return pd.DataFrame(rows, columns=["r_cut_kpc", "n", "rho", "pval"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the MW radial scan and save results.

    Returns
    -------
    dict with keys: scan_df, r_scan_min, r_scan_max, out_path
    """
    parser = argparse.ArgumentParser(
        description="MW Cepheid radial scan for azimuthal velocity modulation"
    )
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Input CSV path")
    parser.add_argument(
        "--r-min", type=float, default=R_SCAN_MIN_DEFAULT, help="Minimum R_cut [kpc]"
    )
    parser.add_argument(
        "--r-max", type=float, default=R_SCAN_MAX_DEFAULT, help="Maximum R_cut [kpc]"
    )
    parser.add_argument(
        "--r-step", type=float, default=R_SCAN_STEP_DEFAULT, help="R_cut step [kpc]"
    )
    parser.add_argument("--out", default=OUT_DEFAULT, help="Output directory")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    scan_df = scan_radii(df, args.r_min, args.r_max, args.r_step, R_MIN_N_DEFAULT)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mw_radial_scan.csv"
    scan_df.to_csv(out_path, index=False)

    return {
        "scan_df":    scan_df,
        "r_scan_min": args.r_min,
        "r_scan_max": args.r_max,
        "out_path":   str(out_path),
    }


if __name__ == "__main__":
    result = main()
    print(result["scan_df"].to_string(index=False))
