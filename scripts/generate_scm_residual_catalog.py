"""
scripts/generate_scm_residual_catalog.py — Build per-galaxy residual catalog.

For each galaxy the script computes two summary quantities from the rotation
curve:

  f3_residual
      Median of log10(g_obs / g_bar) across all valid radial points.
      This captures the per-galaxy acceleration anomaly (positive values
      indicate excess observed acceleration over the baryonic prediction).

  v_last
      Observed rotation velocity (km/s) at the outermost valid radial point.
      This is a proxy for the asymptotic / flat-rotation velocity.

The output CSV (``results/scm_clean_with_residual.csv``) feeds the
Mann-Whitney U test in ``analyze_residual_by_v_last.py``, which tests
whether the F3 residual differs between galaxies with low and high ``v_last``.

Usage
-----
::

    python scripts/generate_scm_residual_catalog.py \\
        --data-dir data/SPARC \\
        --out results/scm_clean_with_residual.csv

Custom parameters::

    python scripts/generate_scm_residual_catalog.py \\
        --data-dir data/SPARC \\
        --out results/scm_clean_with_residual.csv \\
        --a0 1.2e-10 \\
        --min-points 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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

A0_DEFAULT = 1.2e-10        # characteristic acceleration (m/s²)
MIN_POINTS_DEFAULT = 3      # minimum valid radial points for a usable entry

OUTPUT_COLS = ["galaxy", "f3_residual", "v_last"]


# ---------------------------------------------------------------------------
# Per-galaxy computation
# ---------------------------------------------------------------------------

def compute_galaxy_residual(
    rc: pd.DataFrame,
    upsilon_disk: float,
    min_points: int = MIN_POINTS_DEFAULT,
) -> dict | None:
    """Compute ``f3_residual`` and ``v_last`` for a single galaxy.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation curve with columns r, v_obs, v_gas, v_disk, v_bul (km/s).
    upsilon_disk : float
        Best-fit disk mass-to-light ratio from the SCM pipeline.
    min_points : int
        Minimum number of valid (positive g_bar and g_obs) radial points
        required to produce an entry.  Returns ``None`` when fewer points
        are available.

    Returns
    -------
    dict or None
        Keys: ``f3_residual`` (float), ``v_last`` (float).
        ``None`` if the galaxy has insufficient valid data.
    """
    r_arr = rc["r"].values
    v_obs_arr = rc["v_obs"].values
    vb_arr = v_baryonic(
        r_arr,
        rc["v_gas"].values,
        rc["v_disk"].values,
        rc["v_bul"].values,
        upsilon_disk=upsilon_disk,
        upsilon_bul=0.7,
    )

    r_safe = np.maximum(r_arr, _MIN_RADIUS_KPC)
    g_bar_arr = vb_arr ** 2 / r_safe * _CONV
    g_obs_arr = v_obs_arr ** 2 / r_safe * _CONV

    valid = (g_bar_arr > 0) & (g_obs_arr > 0) & np.isfinite(g_bar_arr) & np.isfinite(g_obs_arr)

    if valid.sum() < min_points:
        return None

    log_ratio = np.log10(g_obs_arr[valid]) - np.log10(g_bar_arr[valid])
    f3_residual = float(np.median(log_ratio))

    # v_last: observed velocity at the outermost valid radial point
    valid_indices = np.where(valid)[0]
    last_idx = valid_indices[np.argmax(r_arr[valid_indices])]
    v_last = float(v_obs_arr[last_idx])

    return {"f3_residual": f3_residual, "v_last": v_last}


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

def generate_residual_catalog(
    data_dir: str | Path,
    out: str | Path,
    a0: float = A0_DEFAULT,
    min_points: int = MIN_POINTS_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate the per-galaxy F3 residual and v_last catalog.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC data files.
    out : str or Path
        Output CSV path.
    a0 : float
        Characteristic acceleration (m/s²), passed to ``fit_galaxy``.
    min_points : int
        Minimum valid radial points to include a galaxy.
    verbose : bool
        Print progress when True.

    Returns
    -------
    pd.DataFrame
        Per-galaxy catalog with columns: galaxy, f3_residual, v_last.
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
        entry = compute_galaxy_residual(rc, upsilon_disk=fit["upsilon_disk"],
                                        min_points=min_points)
        if entry is None:
            if verbose:
                print(f"  [skip] {name}: insufficient valid points")
            continue

        entry["galaxy"] = name
        records.append(entry)
        if verbose:
            print(f"  {name}: f3_residual={entry['f3_residual']:.4f}, "
                  f"v_last={entry['v_last']:.2f} km/s")

    if records:
        df = pd.DataFrame(records)[OUTPUT_COLS]
    else:
        df = pd.DataFrame(columns=OUTPUT_COLS)

    df = df.sort_values("galaxy").reset_index(drop=True)
    df.to_csv(out, index=False)

    if verbose:
        print(f"\nResidual catalog written to {out}  ({len(df)} galaxies)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-galaxy F3 residual catalog with f3_residual and "
            "v_last columns for downstream Mann-Whitney U analysis."
        )
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing SPARC data (SPARC_Lelli2016c.csv + rotmod files).",
    )
    parser.add_argument(
        "--out", default="results/scm_clean_with_residual.csv",
        help="Output CSV path (default: results/scm_clean_with_residual.csv).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--min-points", type=int, default=MIN_POINTS_DEFAULT,
        dest="min_points",
        help=f"Minimum valid radial points per galaxy (default: {MIN_POINTS_DEFAULT}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Entry point: parse arguments and run catalog generation."""
    args = _parse_args(argv)
    return generate_residual_catalog(
        data_dir=args.data_dir,
        out=args.out,
        a0=args.a0,
        min_points=args.min_points,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
