"""
scripts/generate_f3_catalog.py — Physical F3 catalog measurement pipeline.

For each galaxy in the SPARC dataset the script fits the per-galaxy deep-regime
slope β in

    log10(g_obs) = β · log10(g_bar) + const

using only radial points in the deep regime (g_bar < deep_threshold × a0).

The result is the **F3 catalog**: one row per galaxy with β and associated
uncertainty.  On real SPARC LSB data the expected deep-regime value is β ≈ 0.5
(MOND / Motor-de-Velos deep form).  On synthetic flat-rotation-curve data both
g_obs and g_bar scale identically as V²/r, so β ≈ 1.0 by construction.

Usage
-----
Physical measurement (real SPARC data)::

    python scripts/generate_f3_catalog.py \\
        --data-dir data/SPARC \\
        --out results/f3_catalog_real.csv

Custom thresholds::

    python scripts/generate_f3_catalog.py \\
        --data-dir data/SPARC \\
        --out results/f3_catalog_real.csv \\
        --a0 1.2e-10 \\
        --deep-threshold 0.3 \\
        --min-deep-points 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.scm_analysis import (
    load_galaxy_table,
    load_rotation_curve,
    fit_galaxy,
    _CONV,
    _MIN_RADIUS_KPC,
)
from src.scm_models import v_baryonic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A0_DEFAULT = 1.2e-10
DEEP_THRESHOLD_DEFAULT = 0.3
MIN_DEEP_POINTS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Per-galaxy β measurement
# ---------------------------------------------------------------------------

def measure_galaxy_beta(
    rc: pd.DataFrame,
    upsilon_disk: float,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    min_deep_points: int = MIN_DEEP_POINTS_DEFAULT,
) -> dict:
    """Fit the deep-regime slope β for a single galaxy.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation curve with columns r, v_obs, v_gas, v_disk, v_bul (km/s).
    upsilon_disk : float
        Best-fit disk mass-to-light ratio from the SCM pipeline.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of *a0* that defines the deep regime.
    min_deep_points : int
        Minimum deep points required for a reliable fit.

    Returns
    -------
    dict with keys: beta, beta_err, intercept, r_value, p_value,
                    n_deep, n_total, reliable
    """
    r_arr = rc["r"].values
    v_obs_arr = rc["v_obs"].values
    vb_arr = v_baryonic(
        r_arr, rc["v_gas"].values, rc["v_disk"].values, rc["v_bul"].values,
        upsilon_disk=upsilon_disk, upsilon_bul=0.7,
    )
    g_bar_arr = vb_arr ** 2 / np.maximum(r_arr, _MIN_RADIUS_KPC) * _CONV
    g_obs_arr = v_obs_arr ** 2 / np.maximum(r_arr, _MIN_RADIUS_KPC) * _CONV

    valid = (g_bar_arr > 0) & (g_obs_arr > 0)
    g_bar_v = g_bar_arr[valid]
    g_obs_v = g_obs_arr[valid]

    deep_mask = g_bar_v < deep_threshold * a0
    n_total = int(valid.sum())
    n_deep = int(deep_mask.sum())

    if n_deep < 2:
        return {
            "beta": float("nan"),
            "beta_err": float("nan"),
            "intercept": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "n_deep": n_deep,
            "n_total": n_total,
            "reliable": False,
        }

    log_gbar = np.log10(g_bar_v[deep_mask])
    log_gobs = np.log10(g_obs_v[deep_mask])
    slope, intercept, r_value, p_value, stderr = linregress(log_gbar, log_gobs)

    return {
        "beta": float(slope),
        "beta_err": float(stderr),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "n_deep": n_deep,
        "n_total": n_total,
        "reliable": n_deep >= min_deep_points,
    }


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

def generate_f3_catalog(
    data_dir: str | Path,
    out: str | Path,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    min_deep_points: int = MIN_DEEP_POINTS_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate the F3 per-galaxy β catalog.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC data files.
    out : str or Path
        Output CSV path for the catalog.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of *a0* defining the deep regime.
    min_deep_points : int
        Minimum deep points for a reliable fit.
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Per-galaxy catalog with columns:
        galaxy, beta, beta_err, intercept, r_value, p_value,
        n_deep, n_total, reliable
    """
    data_dir = Path(data_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    galaxy_table = load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    records = []
    for name in galaxy_names:
        try:
            rc = load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found")
            continue

        fit = fit_galaxy(rc, a0=a0)
        meas = measure_galaxy_beta(
            rc,
            upsilon_disk=fit["upsilon_disk"],
            a0=a0,
            deep_threshold=deep_threshold,
            min_deep_points=min_deep_points,
        )
        meas["galaxy"] = name
        records.append(meas)
        if verbose:
            beta_str = f"{meas['beta']:.3f}" if not np.isnan(meas["beta"]) else "NaN"
            print(f"  {name}: β={beta_str}, n_deep={meas['n_deep']}")

    cols = [
        "galaxy", "beta", "beta_err", "intercept", "r_value", "p_value",
        "n_deep", "n_total", "reliable",
    ]
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=cols)
    else:
        df = df[cols]
    df = df.sort_values("galaxy").reset_index(drop=True)
    df.to_csv(out, index=False)

    if verbose:
        reliable = df["reliable"].sum() if not df.empty else 0
        print(f"\nF3 catalog written to {out}  ({len(df)} galaxies, "
              f"{reliable} with reliable β)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the F3 per-galaxy deep-regime slope catalog. "
            "On real SPARC data the expected β ≈ 0.5; "
            "on synthetic flat curves β ≈ 1.0."
        )
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing SPARC data (SPARC_Lelli2016c.csv + rotmod files).",
    )
    parser.add_argument(
        "--out", default="results/f3_catalog_real.csv",
        help="Output CSV path (default: results/f3_catalog_real.csv).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=f"Deep-regime threshold as fraction of a0 (default: {DEEP_THRESHOLD_DEFAULT}).",
    )
    parser.add_argument(
        "--min-deep-points", type=int, default=MIN_DEEP_POINTS_DEFAULT,
        dest="min_deep_points",
        help=f"Minimum deep points for a reliable fit (default: {MIN_DEEP_POINTS_DEFAULT}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Entry point: parse arguments and run catalog generation."""
    args = _parse_args(argv)
    return generate_f3_catalog(
        data_dir=args.data_dir,
        out=args.out,
        a0=args.a0,
        deep_threshold=args.deep_threshold,
        min_deep_points=args.min_deep_points,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
