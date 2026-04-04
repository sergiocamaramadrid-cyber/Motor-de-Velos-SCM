"""
scripts/generate_scm_residual_catalog.py — Build per-galaxy residual catalog.

For each galaxy the script computes two summary quantities from the rotation
curve:

  a_residual
      Median of log10(g_obs / g_bar) across all valid radial points.
      This is the **acceleration-anomaly observable** (A_residual), distinct
      from the geometric deep-regime slope F3_geom (``friction_slope`` /
      ``beta`` in ``generate_f3_catalog.py``).  Positive values indicate
      excess observed acceleration over the baryonic prediction.

  v_last
      Observed rotation velocity (km/s) at the outermost valid radial point.
      This is a proxy for the asymptotic / flat-rotation velocity.

Naming convention
-----------------
``F3_geom``   — deep-regime log–log slope β = d log g_obs / d log g_bar
                (computed by ``generate_f3_catalog.py``, stored as
                ``friction_slope`` / ``beta``).
``A_residual`` — per-galaxy median of log10(g_obs / g_bar), the
                acceleration-anomaly offset (this script, column
                ``a_residual``).

Keeping the two observables under distinct names prevents conflation in
papers and downstream analyses.

The output CSV (``results/scm_clean_with_residual.csv``) feeds the
Mann-Whitney U test in ``analyze_residual_by_v_last.py``, which tests
whether the acceleration-anomaly residual differs between galaxies with
low and high ``v_last``.

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

Demo mode (no SPARC data required)::

    python scripts/generate_scm_residual_catalog.py --demo
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
DEMO_N_GALAXIES = 30        # number of galaxies in demo mode

OUTPUT_COLS = ["galaxy", "a_residual", "v_last"]


# ---------------------------------------------------------------------------
# Per-galaxy computation
# ---------------------------------------------------------------------------

def compute_galaxy_residual(
    rc: pd.DataFrame,
    upsilon_disk: float,
    min_points: int = MIN_POINTS_DEFAULT,
) -> dict | None:
    """Compute ``a_residual`` and ``v_last`` for a single galaxy.

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
        Keys: ``a_residual`` (float), ``v_last`` (float).
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
    a_residual = float(np.median(log_ratio))

    # v_last: observed velocity at the outermost valid radial point
    valid_indices = np.where(valid)[0]
    last_idx = valid_indices[np.argmax(r_arr[valid_indices])]
    v_last = float(v_obs_arr[last_idx])

    return {"a_residual": a_residual, "v_last": v_last}


# ---------------------------------------------------------------------------
# Demo catalog (no SPARC data required)
# ---------------------------------------------------------------------------

def generate_demo_catalog(
    out: str | Path,
    n_galaxies: int = DEMO_N_GALAXIES,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate a fully synthetic residual catalog for demonstration purposes.

    Uses ``numpy`` random numbers to produce plausible ``a_residual`` and
    ``v_last`` values without requiring any SPARC data files.  The output
    has the same schema as the real catalog so that
    ``analyze_residual_by_v_last.py`` can consume it directly.

    Parameters
    ----------
    out : str or Path
        Output CSV path.
    n_galaxies : int
        Number of synthetic galaxies (default: 30).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print a summary line when True.

    Returns
    -------
    pd.DataFrame
        Per-galaxy catalog with columns: galaxy, a_residual, v_last.
    """
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic v_last: uniform over a realistic SPARC-like range (km/s)
    v_last = rng.uniform(30.0, 300.0, n_galaxies)

    # Synthetic a_residual: slight positive correlation with v_last + noise
    a_residual = 0.001 * v_last + rng.normal(0.0, 0.15, n_galaxies)

    df = pd.DataFrame({
        "galaxy": [f"SYN{i:03d}" for i in range(n_galaxies)],
        "a_residual": a_residual,
        "v_last": v_last,
    })[OUTPUT_COLS].sort_values("galaxy").reset_index(drop=True)

    df.to_csv(out, index=False)
    if verbose:
        print(f"[demo] Synthetic catalog written to {out}  ({n_galaxies} galaxies)")
    return df


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
    """Generate the per-galaxy acceleration-anomaly residual and v_last catalog.

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
        Per-galaxy catalog with columns: galaxy, a_residual, v_last.
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
            print(f"  {name}: a_residual={entry['a_residual']:.4f}, "
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
            "Build a per-galaxy acceleration-anomaly catalog with a_residual and "
            "v_last columns for downstream Mann-Whitney U analysis."
        )
    )
    parser.add_argument(
        "--demo", action="store_true",
        help=(
            "Generate a synthetic demo catalog without requiring SPARC data. "
            "Ignores --data-dir."
        ),
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing SPARC data (SPARC_Lelli2016c.csv + rotmod files). "
             "Required unless --demo is used.",
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
    if args.demo:
        return generate_demo_catalog(
            out=args.out,
            verbose=not args.quiet,
        )
    if args.data_dir is None:
        raise SystemExit(
            "error: --data-dir is required (or use --demo for a synthetic catalog)."
        )
    return generate_residual_catalog(
        data_dir=args.data_dir,
        out=args.out,
        a0=args.a0,
        min_points=args.min_points,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
