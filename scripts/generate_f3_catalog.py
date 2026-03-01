"""
scripts/generate_f3_catalog.py — Generate the F3 per-galaxy friction-slope catalog.

The F3 catalog quantifies the per-galaxy deep-regime RAR slope (β, the
"friction slope") and its deviation from the Motor de Velos / MOND prediction
of β = 0.5.

For each galaxy with sufficient deep-regime radial points the script fits:

    log10(g_obs) = β · log10(g_bar) + const

in the deep regime (g_bar < deep_threshold × a0).  The resulting β values
(friction_slope), their standard errors (friction_slope_err), and a flag
(velo_inerte_flag = True when |β − 0.5| > 2 · friction_slope_err) are
written to results/f3_catalog.csv.

Usage
-----
From SPARC rotmod files::

    python scripts/generate_f3_catalog.py \\
        --data-dir data/SPARC \\
        --out results/f3_catalog.csv

With synthetic data (CI / unit-test mode)::

    python scripts/generate_f3_catalog.py \\
        --synthetic \\
        --n-galaxies 50 \\
        --out results/f3_catalog.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

KPC_TO_M = 3.085677581e16  # metres per kiloparsec (IAU 2012)
_CONV = 1e6 / KPC_TO_M     # (km/s)²/kpc → m/s²

A0_DEFAULT = 1.2e-10           # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT = 0.3   # g_bar/a0 fraction defining deep regime
EXPECTED_SLOPE = 0.5           # MOND / Motor de Velos prediction
MIN_DEEP_POINTS = 5            # minimum per-galaxy deep points for a reliable fit

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _v_baryonic(v_gas: np.ndarray, v_disk: np.ndarray, v_bul: np.ndarray,
                upsilon_disk: float, upsilon_bul: float = 0.7) -> np.ndarray:
    """Signed baryonic rotation velocity."""
    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + upsilon_bul * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _per_galaxy_friction_slope(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bul: np.ndarray,
    upsilon_disk: float = 1.0,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> tuple[float, float]:
    """Fit the deep-regime slope β for a single galaxy.

    Parameters
    ----------
    r : ndarray
        Galactocentric radii in kpc.
    v_obs : ndarray
        Observed rotation velocity in km/s.
    v_gas, v_disk, v_bul : ndarray
        Component velocities in km/s.
    upsilon_disk : float
        Disk mass-to-light ratio.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of a0 that defines the deep regime.

    Returns
    -------
    slope : float
        OLS slope β (NaN if fewer than MIN_DEEP_POINTS deep points).
    slope_err : float
        Standard error of the slope (NaN if fewer than 3 deep points).
    """
    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)
    g_bar = vb ** 2 / np.maximum(r, 1e-10) * _CONV      # m/s²
    g_obs = v_obs ** 2 / np.maximum(r, 1e-10) * _CONV   # m/s²

    deep_mask = (g_bar < deep_threshold * a0) & (g_bar > 0) & (g_obs > 0)
    n_deep = int(deep_mask.sum())

    if n_deep < MIN_DEEP_POINTS:
        return float("nan"), float("nan")

    log_gbar = np.log10(g_bar[deep_mask])
    log_gobs = np.log10(g_obs[deep_mask])

    slope, _, _, _, stderr = linregress(log_gbar, log_gobs)
    return float(slope), float(stderr)


# ---------------------------------------------------------------------------
# Catalog generation from SPARC rotmod files
# ---------------------------------------------------------------------------


def _load_galaxy_table(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for p in candidates:
        if p.exists():
            sep = "," if p.suffix == ".csv" else r"\s+"
            return pd.read_csv(p, sep=sep, comment="#")
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {data_dir}. "
        "Expected SPARC_Lelli2016c.csv or .mrt"
    )


def _load_rotation_curve(data_dir: Path, name: str) -> pd.DataFrame:
    candidates = [
        data_dir / f"{name}_rotmod.dat",
        data_dir / "raw" / f"{name}_rotmod.dat",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(
                p, sep=r"\s+", comment="#",
                names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
                       "SBdisk", "SBbul"],
            )
            return df[["r", "v_obs", "v_gas", "v_disk", "v_bul"]]
    raise FileNotFoundError(f"Rotation curve for {name} not found in {data_dir}")


def generate_from_data_dir(
    data_dir: Path,
    out_path: Path,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate the F3 catalog from SPARC rotmod files.

    Parameters
    ----------
    data_dir : Path
        Directory containing SPARC data.
    out_path : Path
        Destination CSV path.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        g_bar/a0 fraction that defines the deep regime.
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        The generated catalog DataFrame.
    """
    galaxy_table = _load_galaxy_table(data_dir)
    rows = []
    for name in galaxy_table["Galaxy"].tolist():
        try:
            rc = _load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotmod not found")
            continue

        slope, err = _per_galaxy_friction_slope(
            rc["r"].values,
            rc["v_obs"].values,
            rc["v_gas"].values,
            rc["v_disk"].values,
            rc["v_bul"].values,
            a0=a0,
            deep_threshold=deep_threshold,
        )
        if not np.isfinite(slope):
            if verbose:
                print(f"  [skip] {name}: insufficient deep-regime points")
            continue

        flag = bool(abs(slope - EXPECTED_SLOPE) > 2.0 * err) if np.isfinite(err) else False
        rows.append({
            "galaxy": name,
            "friction_slope": round(slope, 6),
            "friction_slope_err": round(err, 6) if np.isfinite(err) else float("nan"),
            "velo_inerte_flag": flag,
        })
        if verbose:
            print(f"  {name}: β={slope:.4f} ± {err:.4f}  flag={flag}")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nF3 catalog written to {out_path} ({len(df)} galaxies)")
    return df


# ---------------------------------------------------------------------------
# Synthetic catalog generation (for CI / testing)
# ---------------------------------------------------------------------------


def generate_synthetic(
    out_path: Path,
    n_galaxies: int = 50,
    seed: int = 42,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """Generate a synthetic F3 catalog using MOND-compliant simulated data.

    Each synthetic galaxy is assigned per-galaxy deep-regime radial points drawn
    from the expected MOND relation (β ≈ 0.5) plus Gaussian scatter, matching the
    scatter level seen in the SPARC LITTLE THINGS analysis.

    Parameters
    ----------
    out_path : Path
        Destination CSV path.
    n_galaxies : int
        Number of synthetic galaxies (default 50).
    seed : int
        Random seed for reproducibility.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        g_bar/a0 fraction for the deep regime.

    Returns
    -------
    pd.DataFrame
        The generated catalog.
    """
    rng = np.random.default_rng(seed)

    # Realistic per-galaxy scatter in β from SPARC-like analysis
    sigma_slope = 0.06        # galaxy-to-galaxy intrinsic scatter in β
    noise_log = 0.025         # per-point scatter in log(g_obs)
    n_deep_per_gal = 20       # deep-regime points per galaxy

    rows = []
    for i in range(n_galaxies):
        name = f"F3G{i + 1:03d}"

        # True β for this galaxy drawn from distribution centred on 0.5
        true_beta = rng.normal(EXPECTED_SLOPE, sigma_slope)

        # Simulate deep-regime (log g_bar, log g_obs) points
        g_bar_deep = rng.uniform(0.005 * a0, 0.29 * a0, n_deep_per_gal)
        log_gbar = np.log10(g_bar_deep)
        # log g_obs = β · log g_bar + 0.5·log(a0) + noise
        log_gobs = (true_beta * log_gbar
                    + 0.5 * np.log10(a0)
                    + rng.normal(0, noise_log, n_deep_per_gal))

        slope, _, _, _, stderr = linregress(log_gbar, log_gobs)
        slope = float(slope)
        stderr = float(stderr)

        flag = bool(abs(slope - EXPECTED_SLOPE) > 2.0 * stderr)
        rows.append({
            "galaxy": name,
            "friction_slope": round(slope, 6),
            "friction_slope_err": round(stderr, 6),
            "velo_inerte_flag": flag,
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the F3 per-galaxy friction-slope catalog "
            "(results/f3_catalog.csv)."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data-dir", metavar="DIR",
        help="Directory containing SPARC rotmod files and galaxy table.",
    )
    src.add_argument(
        "--synthetic", action="store_true",
        help="Generate a synthetic catalog (no real data required).",
    )
    parser.add_argument(
        "--n-galaxies", type=int, default=50, dest="n_galaxies",
        help="Number of synthetic galaxies (only with --synthetic, default 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthetic generation (default 42).",
    )
    parser.add_argument(
        "--out", default="results/f3_catalog.csv", metavar="FILE",
        help="Output CSV path (default: results/f3_catalog.csv).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=f"g_bar/a0 fraction for deep regime (default: {DEEP_THRESHOLD_DEFAULT}).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Generate and write the F3 catalog.  Returns the catalog DataFrame."""
    args = _parse_args(argv)
    out_path = Path(args.out)

    if args.synthetic:
        df = generate_synthetic(
            out_path,
            n_galaxies=args.n_galaxies,
            seed=args.seed,
            a0=args.a0,
            deep_threshold=args.deep_threshold,
        )
        if not args.quiet:
            n_flag = int(df["velo_inerte_flag"].sum())
            print(f"Synthetic F3 catalog: {len(df)} galaxies, "
                  f"{n_flag} flagged (velo_inerte).  Written to {out_path}")
    else:
        df = generate_from_data_dir(
            Path(args.data_dir),
            out_path,
            a0=args.a0,
            deep_threshold=args.deep_threshold,
            verbose=not args.quiet,
        )

    return df


if __name__ == "__main__":
    main()
