"""
scripts/generate_f3_catalog.py — Generate the F3 per-galaxy catalog.

Theory
------
For each galaxy the "friction slope" β_i is the OLS slope of:

    log10(g_obs) = β_i · log10(g_bar) + const.

restricted to deep-regime radial points (g_bar < deep_threshold × a0).

The velo-inerte flag is set to 1 when β_i is statistically consistent
with the MOND/deep-velos prediction β = 0.5 (i.e. |β_i − 0.5| ≤ 2 σ_i),
and 0 otherwise (including galaxies with insufficient deep-regime coverage).

Output columns
--------------
galaxy             — galaxy identifier
n_deep             — number of deep-regime radial points used
friction_slope     — OLS slope β_i  (NaN when n_deep < 2)
friction_slope_err — standard error of slope (NaN when n_deep < 3)
velo_inerte_flag   — 1 (consistent) / 0 (inconsistent or insufficient)

Usage
-----
    python scripts/generate_f3_catalog.py \\
        --data-dir data/SPARC \\
        --out results/f3_catalog_real.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A0_DEFAULT = 1.2e-10        # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT = 0.3
EXPECTED_SLOPE = 0.5
KPC_TO_M = 3.085677581e16
_CONV = 1e6 / KPC_TO_M      # (km/s)²/kpc → m/s²
_MIN_RADIUS_KPC = 1e-10

# Physical bounds for the disk mass-to-light ratio:
# 0.1 M_sun/L_sun (gas-dominated regime) to 5.0 M_sun/L_sun (old stellar pop.)
_UPSILON_DISK_MIN = 0.1
_UPSILON_DISK_MAX = 5.0


# ---------------------------------------------------------------------------
# Data loading (mirrors src/scm_analysis.py)
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
            return df[["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]]
    raise FileNotFoundError(
        f"Rotation curve for {name} not found in {data_dir}"
    )


# ---------------------------------------------------------------------------
# Per-galaxy baryonic velocity (mirrors src/scm_models.py)
# ---------------------------------------------------------------------------

def _v_baryonic(v_gas, v_disk, v_bul,
                upsilon_disk: float = 1.0,
                upsilon_bul: float = 0.7) -> np.ndarray:
    """Signed baryonic rotation velocity (km/s).

    Computes V_bar = sign(V²_bar) × √|V²_bar|  where
    V²_bar = V²_gas + Υ_disk V²_disk + Υ_bul V²_bul.
    The signed form preserves the direction of inward/outward forces.

    Parameters
    ----------
    v_gas, v_disk, v_bul : array_like
        Component velocities in km/s.
    upsilon_disk, upsilon_bul : float
        Mass-to-light ratios for disk and bulge (dimensionless).

    Returns
    -------
    ndarray
        Baryonic rotation velocity in km/s.
    """
    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + upsilon_bul * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


# ---------------------------------------------------------------------------
# Per-galaxy friction slope
# ---------------------------------------------------------------------------

def compute_friction_slope(rc: pd.DataFrame,
                            upsilon_disk: float = 1.0,
                            upsilon_bul: float = 0.7,
                            a0: float = A0_DEFAULT,
                            deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
                            ) -> dict:
    """Compute the per-galaxy friction slope in the deep regime.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve table with columns r, v_obs, v_gas, v_disk, v_bul.
    upsilon_disk : float
        Best-fit disk mass-to-light ratio.
    upsilon_bul : float
        Bulge mass-to-light ratio (fixed at 0.7 throughout the framework).
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of a0 below which a point is "deep".

    Returns
    -------
    dict with keys:
        n_deep             — int
        friction_slope     — float or nan
        friction_slope_err — float or nan
        velo_inerte_flag   — int (0 or 1)
    """
    r = np.asarray(rc["r"].values, dtype=float)
    v_obs = np.asarray(rc["v_obs"].values, dtype=float)
    v_gas = np.asarray(rc["v_gas"].values, dtype=float)
    v_disk = np.asarray(rc["v_disk"].values, dtype=float)
    v_bul = np.asarray(rc["v_bul"].values, dtype=float)

    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk, upsilon_bul)
    g_bar = vb ** 2 / np.maximum(r, _MIN_RADIUS_KPC) * _CONV  # m/s²
    g_obs = v_obs ** 2 / np.maximum(r, _MIN_RADIUS_KPC) * _CONV  # m/s²

    deep_mask = g_bar < deep_threshold * a0
    valid = deep_mask & (g_bar > 0) & (g_obs > 0)
    n_deep = int(valid.sum())

    if n_deep < 2:
        return {
            "n_deep": n_deep,
            "friction_slope": float("nan"),
            "friction_slope_err": float("nan"),
            "velo_inerte_flag": 0,
        }

    log_gbar = np.log10(g_bar[valid])
    log_gobs = np.log10(g_obs[valid])
    slope, _, _, _, stderr = linregress(log_gbar, log_gobs)
    slope = float(slope)
    stderr = float(stderr)

    # Flag consistent with β = 0.5 if |slope − 0.5| ≤ 2 × stderr
    consistent = int(abs(slope - EXPECTED_SLOPE) <= 2.0 * stderr)

    return {
        "n_deep": n_deep,
        "friction_slope": slope,
        "friction_slope_err": stderr,
        "velo_inerte_flag": consistent,
    }


# ---------------------------------------------------------------------------
# Upsilon fitting (scalar minimisation identical to src/scm_analysis.py)
# ---------------------------------------------------------------------------

def _fit_upsilon_disk(rc: pd.DataFrame, a0: float = A0_DEFAULT) -> float:
    """Return best-fit upsilon_disk minimising reduced chi-squared."""
    from scipy.optimize import minimize_scalar

    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = np.where(rc["v_obs_err"].values > 0,
                         rc["v_obs_err"].values, 1.0)
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    a0_kpc = a0 / _CONV  # (km/s)²/kpc

    def objective(ud: float) -> float:
        vb = _v_baryonic(v_gas, v_disk, v_bul, ud)
        v2 = vb * np.abs(vb) + a0_kpc * np.maximum(r, 0.0)
        vp = np.sign(v2) * np.sqrt(np.abs(v2))
        res = (v_obs - vp) / v_obs_err
        dof = max(len(r) - 2, 1)
        return float(np.sum(res ** 2) / dof)

    result = minimize_scalar(objective, bounds=(_UPSILON_DISK_MIN, _UPSILON_DISK_MAX), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Main catalog generation
# ---------------------------------------------------------------------------

def build_f3_catalog(data_dir: Path,
                     a0: float = A0_DEFAULT,
                     deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
                     verbose: bool = True) -> pd.DataFrame:
    """Build the F3 per-galaxy catalog.

    Parameters
    ----------
    data_dir : Path
        Directory containing SPARC data files.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of a0 defining the deep regime.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    pd.DataFrame
        Catalog with columns:
        galaxy, n_deep, friction_slope, friction_slope_err, velo_inerte_flag.
    """
    galaxy_table = _load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    records = []
    for name in galaxy_names:
        try:
            rc = _load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found",
                      file=sys.stderr)
            continue

        ud = _fit_upsilon_disk(rc, a0=a0)
        row = compute_friction_slope(rc, upsilon_disk=ud, a0=a0,
                                     deep_threshold=deep_threshold)
        row["galaxy"] = name
        records.append(row)
        if verbose:
            flag = row["velo_inerte_flag"]
            slope = row["friction_slope"]
            slope_s = f"{slope:.4f}" if not np.isnan(slope) else "nan"
            print(f"  {name}: friction_slope={slope_s}, flag={flag}",
                  file=sys.stderr)

    df = pd.DataFrame(records)[
        ["galaxy", "n_deep", "friction_slope", "friction_slope_err",
         "velo_inerte_flag"]
    ]
    df = df.sort_values("galaxy").reset_index(drop=True)
    df["n_deep"] = df["n_deep"].astype(int)
    df["velo_inerte_flag"] = df["velo_inerte_flag"].astype(int)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-galaxy F3 friction-slope catalog from SPARC data."
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="DIR",
        help="Directory containing SPARC rotmod files and galaxy table.",
    )
    parser.add_argument(
        "--out", required=True, metavar="FILE",
        help="Output CSV path (e.g. results/f3_catalog_real.csv).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(f"Fraction of a0 defining deep regime "
              f"(default: {DEEP_THRESHOLD_DEFAULT})."),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-galaxy progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run catalog generation and return the catalog DataFrame."""
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Provide a directory containing SPARC rotmod files."
        )

    df = build_f3_catalog(data_dir, a0=args.a0,
                          deep_threshold=args.deep_threshold,
                          verbose=not args.quiet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if not args.quiet:
        print(f"\nCatalog written to {out_path}  ({len(df)} galaxies)")
    return df


if __name__ == "__main__":
    main()
