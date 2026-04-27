"""
scripts/build_dataset.py — Build the SCM mass + distribution dataset.

Pipeline
--------
1. Parse SPARC rotmod files to extract rotation curves.
2. Compute outer-regime slope (r >= 0.7 * Rmax) per galaxy.
3. Merge with SPARC baryonic mass catalog.
4. Compute surface-density residual (Sigma_resid) as proxy for mass distribution.
5. Write clean sample to data/scm_mass_distribution_final_dataset.csv.

Usage
-----
With default paths::

    python scripts/build_dataset.py

Explicit options::

    python scripts/build_dataset.py \\
        --sparc-dir  data/sparc_rotmod \\
        --catalog    data/SPARC_Lelli2016c.mrt \\
        --out        data/scm_mass_distribution_final_dataset.csv \\
        --outer-frac 0.7 \\
        --min-points 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTER_FRAC = 0.7        # fraction of Rmax defining the outer regime
MIN_OUTER_POINTS = 4    # minimum points required in outer regime
HE_CORRECTION = 1.33    # helium correction factor for gas mass
UPSILON_DISK = 1.0      # stellar mass-to-light ratio (disk)
UPSILON_BULGE = 1.0     # stellar mass-to-light ratio (bulge)

# ---------------------------------------------------------------------------
# Rotmod parser
# ---------------------------------------------------------------------------

def parse_rotmod(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a SPARC rotmod file and return (r_kpc, v_obs_kms).

    Parameters
    ----------
    path : str or Path
        Path to a ``*_rotmod.dat`` file from the SPARC dataset.

    Returns
    -------
    r : np.ndarray
        Galactocentric radii in kpc.
    v : np.ndarray
        Observed rotation velocities in km/s.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        return np.array([]), np.array([])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def galaxy_name_from_path(path: str | Path) -> str:
    """Derive galaxy name from rotmod file path.

    Parameters
    ----------
    path : str or Path
        Path to a ``*_rotmod.dat`` file.

    Returns
    -------
    str
        Galaxy name (stem with ``_rotmod`` suffix removed).
    """
    stem = Path(path).stem
    return stem.replace("_rotmod", "")


# ---------------------------------------------------------------------------
# Outer-slope computation
# ---------------------------------------------------------------------------

def compute_outer_slope(
    r: np.ndarray,
    v: np.ndarray,
    outer_frac: float = OUTER_FRAC,
    min_points: int = MIN_OUTER_POINTS,
) -> dict:
    """Compute the log-slope of the rotation curve in the outer regime.

    Parameters
    ----------
    r : np.ndarray
        Radii in kpc.
    v : np.ndarray
        Observed velocities in km/s.
    outer_frac : float
        Fraction of Rmax above which the outer regime begins.
    min_points : int
        Minimum number of outer-regime points required for a valid fit.

    Returns
    -------
    dict with keys:
        slope_tail, Rmax, n_outer, outer_fit_ok
    """
    mask = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    r, v = r[mask], v[mask]

    if len(r) < 2:
        return {"slope_tail": np.nan, "Rmax": np.nan,
                "n_outer": 0, "outer_fit_ok": False}

    rmax = r.max()
    outer = r >= outer_frac * rmax

    if outer.sum() < min_points:
        return {"slope_tail": np.nan, "Rmax": rmax,
                "n_outer": int(outer.sum()), "outer_fit_ok": False}

    log_r = np.log10(r[outer])
    log_v = np.log10(v[outer])
    result = linregress(log_r, log_v)

    return {
        "slope_tail": float(result.slope),
        "Rmax": float(rmax),
        "n_outer": int(outer.sum()),
        "outer_fit_ok": True,
    }


# ---------------------------------------------------------------------------
# Baryonic mass and surface density
# ---------------------------------------------------------------------------

def compute_logMbar(row: pd.Series) -> float:
    """Compute log10 of total baryonic mass from SPARC catalog row.

    Parameters
    ----------
    row : pd.Series
        A row from the SPARC properties table. Expected columns:
        ``L36`` (luminosity at 3.6 µm in 10^9 Lsun),
        ``Mgas`` (HI gas mass in 10^9 Msun),
        ``BulgeFlag`` (1 if bulge present, else 0).

    Returns
    -------
    float
        log10(Mbar / Msun)
    """
    l36 = float(row.get("L36", 0)) * 1e9  # Lsun
    mgas = float(row.get("Mgas", 0)) * 1e9 * HE_CORRECTION  # Msun
    bulge_flag = int(row.get("BulgeFlag", 0))

    if bulge_flag:
        mstar = UPSILON_BULGE * l36
    else:
        mstar = UPSILON_DISK * l36

    mbar = mstar + mgas
    if mbar <= 0:
        return np.nan
    return float(np.log10(mbar))


def compute_sigma_resid(
    df: pd.DataFrame,
    logmbar_col: str = "logMbar",
    slope_col: str = "slope_tail",
) -> pd.Series:
    """Compute Sigma_resid: residual after regressing slope_tail on logMbar.

    This proxy captures mass-distribution effects beyond the global mass scale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``logmbar_col`` and ``slope_col`` columns.
    logmbar_col : str
        Column name for log baryonic mass.
    slope_col : str
        Column name for outer slope.

    Returns
    -------
    pd.Series
        Residuals from the OLS fit slope_tail ~ logMbar.
    """
    clean = df[[logmbar_col, slope_col]].dropna()
    result = linregress(clean[logmbar_col], clean[slope_col])
    predicted = result.slope * df[logmbar_col] + result.intercept
    return df[slope_col] - predicted


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    sparc_dir: str | Path,
    catalog: str | Path,
    out: str | Path,
    outer_frac: float = OUTER_FRAC,
    min_points: int = MIN_OUTER_POINTS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the SCM mass + distribution dataset from SPARC rotmod files.

    Parameters
    ----------
    sparc_dir : str or Path
        Directory containing ``*_rotmod.dat`` files.
    catalog : str or Path
        Path to SPARC mass catalog (CSV or MRT format).
    out : str or Path
        Output CSV path.
    outer_frac : float
        Outer-regime threshold as fraction of Rmax.
    min_points : int
        Minimum outer-regime points for a valid slope fit.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame
        The clean sample ready for analysis.
    """
    sparc_dir = Path(sparc_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rotmod_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    if verbose:
        print(f"Found {len(rotmod_files)} rotmod files in {sparc_dir}")

    # Parse catalog
    try:
        cat = pd.read_csv(catalog, comment="#")
    except Exception:
        cat = pd.DataFrame()

    records = []
    for fpath in rotmod_files:
        name = galaxy_name_from_path(fpath)
        r, v = parse_rotmod(fpath)
        stats = compute_outer_slope(r, v, outer_frac=outer_frac,
                                    min_points=min_points)
        rec = {"galaxy": name, **stats}

        # Merge catalog columns if available
        if not cat.empty and "Name" in cat.columns:
            row = cat[cat["Name"] == name]
            if not row.empty:
                row = row.iloc[0]
                rec["logMbar"] = compute_logMbar(row)
                rec["Mgas"] = float(row.get("Mgas", np.nan))
                rec["Vmax"] = float(row.get("Vmax", np.nan))

        records.append(rec)

    df = pd.DataFrame(records)
    n_valid = len(df)

    # Keep only galaxies with valid outer slope
    df = df[df["outer_fit_ok"].astype(bool)].copy()
    n_clean = len(df)

    if verbose:
        print(f"Valid galaxies parsed: {n_valid}")
        print(f"Final clean sample (outer_fit_ok): {n_clean}")

    # Compute Sigma_resid if logMbar available
    if "logMbar" in df.columns and df["logMbar"].notna().sum() >= 4:
        df["Sigma_resid"] = compute_sigma_resid(df)

    df.to_csv(out, index=False)
    if verbose:
        print(f"Dataset written to {out}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Build SCM mass + distribution dataset from SPARC rotmod files."
    )
    parser.add_argument(
        "--sparc-dir",
        default="data/sparc_rotmod",
        help="Directory with *_rotmod.dat files (default: data/sparc_rotmod)",
    )
    parser.add_argument(
        "--catalog",
        default="data/SPARC_Lelli2016c.mrt",
        help="SPARC mass catalog file (default: data/SPARC_Lelli2016c.mrt)",
    )
    parser.add_argument(
        "--out",
        default="data/scm_mass_distribution_final_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--outer-frac",
        type=float,
        default=OUTER_FRAC,
        help=f"Outer-regime threshold (default: {OUTER_FRAC})",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_OUTER_POINTS,
        help=f"Min outer-regime points (default: {MIN_OUTER_POINTS})",
    )
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args(argv)

    df = build_dataset(
        sparc_dir=args.sparc_dir,
        catalog=args.catalog,
        out=args.out,
        outer_frac=args.outer_frac,
        min_points=args.min_points,
        verbose=args.verbose,
    )

    return {
        "n_galaxies": len(df),
        "output": args.out,
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
