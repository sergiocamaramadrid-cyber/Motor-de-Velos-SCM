"""
scripts/sparc_slope_tail.py — Outer-disk (tail) slope pipeline for SPARC galaxies.

For each SPARC rotation-curve file (``<Galaxy>_rotmod.dat``) the script fits
the log-log slope of the outer-disk velocity profile:

    slope_tail = d log10(V) / d log10(r)

using only radial points in the outer tail (r >= TAIL_FRAC × r_max).

In a perfectly flat rotation curve slope_tail = 0; negative values indicate
declining outer velocity (as expected in the SCM / Motor-de-Velos framework
relative to the baryonic reference slope of 0.5).  The slope is returned as
a plain OLS estimate via :func:`numpy.polyfit`.

Usage
-----
Default paths::

    python scripts/sparc_slope_tail.py

Custom paths::

    python scripts/sparc_slope_tail.py \\
        --data-dir data/SPARC/rotmod \\
        --out      results/slope_tail.csv \\
        --tail-frac 0.7 \\
        --min-points 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAIL_FRAC_DEFAULT: float = 0.7   # radii >= TAIL_FRAC * r_max are used
MIN_TAIL_POINTS: int = 4          # minimum points in tail for a valid fit
DATA_DIR_DEFAULT = "data/SPARC/rotmod"
OUTPUT_CSV_DEFAULT = "results/slope_tail.csv"

# Rotmod column positions (space-separated, no header)
_COL_R = 0
_COL_V = 1


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_slope_tail(r: np.ndarray, v: np.ndarray) -> float:
    """Fit the log-log outer-disk slope d log10(V) / d log10(r).

    Parameters
    ----------
    r : array_like
        Galactocentric radii (any consistent units, e.g. kpc).  Must be
        strictly positive.
    v : array_like
        Observed rotation velocities corresponding to *r* (km/s).  Must be
        strictly positive.

    Returns
    -------
    float
        OLS slope of log10(V) vs log10(r) over the supplied points.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    log_r = np.log10(r)
    log_v = np.log10(v)
    slope, _ = np.polyfit(log_r, log_v, 1)
    return float(slope)


def process_galaxy(
    file_path: str | Path,
    tail_frac: float = TAIL_FRAC_DEFAULT,
    min_points: int = MIN_TAIL_POINTS,
) -> float | None:
    """Compute the outer-disk slope for a single SPARC rotmod file.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``<Galaxy>_rotmod.dat`` file (space-separated, no header).
        Column 0 is radius (kpc), column 1 is observed velocity (km/s).
    tail_frac : float
        Radii >= *tail_frac* × r_max are included in the tail.
    min_points : int
        Minimum number of tail points required for a valid fit.

    Returns
    -------
    float or None
        The log-log slope, or *None* if fewer than *min_points* tail
        points are available or if the data cannot be read.
    """
    data = np.loadtxt(file_path)
    if data.ndim != 2 or data.shape[1] < 2:
        return None

    r = data[:, _COL_R]
    v = data[:, _COL_V]

    # Filter to physically valid rows only
    valid = (r > 0) & (v > 0)
    r = r[valid]
    v = v[valid]

    if len(r) == 0:
        return None

    r_max = np.max(r)
    mask = r >= tail_frac * r_max

    if np.sum(mask) < min_points:
        return None

    return compute_slope_tail(r[mask], v[mask])


def process_directory(
    data_dir: str | Path,
    output_csv: str | Path,
    tail_frac: float = TAIL_FRAC_DEFAULT,
    min_points: int = MIN_TAIL_POINTS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Process all ``*_rotmod.dat`` files in *data_dir* and save results.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC rotmod files.
    output_csv : str or Path
        Destination CSV with columns ``galaxy`` and ``slope_tail``.
    tail_frac : float
        Fraction of r_max defining the outer tail.
    min_points : int
        Minimum tail points required per galaxy.
    verbose : bool
        Print progress to stdout if True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``galaxy`` and ``slope_tail``.
    """
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith("_rotmod.dat"):
            continue
        galaxy = fname.replace("_rotmod.dat", "")
        path = data_dir / fname
        slope = process_galaxy(path, tail_frac=tail_frac, min_points=min_points)
        if slope is not None:
            results.append({"galaxy": galaxy, "slope_tail": slope})
            if verbose:
                print(f"  {galaxy}: slope_tail={slope:.4f}")
        else:
            if verbose:
                print(f"  [skip] {galaxy}: insufficient tail points")

    df = pd.DataFrame(results, columns=["galaxy", "slope_tail"])
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"Saved {len(df)} galaxies → {output_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the outer-disk log-log slope (slope_tail) "
            "for each SPARC rotmod galaxy."
        )
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR_DEFAULT, dest="data_dir",
        help=f"Directory with *_rotmod.dat files (default: {DATA_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUTPUT_CSV_DEFAULT,
        help=f"Output CSV path (default: {OUTPUT_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--tail-frac", type=float, default=TAIL_FRAC_DEFAULT, dest="tail_frac",
        help=(f"Radii >= tail_frac × r_max used for tail fit "
              f"(default: {TAIL_FRAC_DEFAULT})."),
    )
    parser.add_argument(
        "--min-points", type=int, default=MIN_TAIL_POINTS, dest="min_points",
        help=f"Minimum tail points required (default: {MIN_TAIL_POINTS}).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: parse arguments and run the slope-tail pipeline.

    Returns
    -------
    dict with keys:
        df           — resulting :class:`pandas.DataFrame`
        output_csv   — path to the written CSV (str)
        n_galaxies   — number of galaxies included
    """
    args = _parse_args(argv)
    df = process_directory(
        data_dir=args.data_dir,
        output_csv=args.out,
        tail_frac=args.tail_frac,
        min_points=args.min_points,
        verbose=not args.quiet,
    )
    return {
        "df": df,
        "output_csv": str(args.out),
        "n_galaxies": len(df),
    }


if __name__ == "__main__":
    main()
