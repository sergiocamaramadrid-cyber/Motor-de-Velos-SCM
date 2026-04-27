"""
scripts/build_sparc_catalog.py — Build a processed SPARC galaxy catalog.

Reads SPARC rotmod files and the SPARC mass-properties table, computes per-galaxy
statistics (outer-regime slope, baryonic mass, structural parameters), and writes
a clean catalog to ``data/processed/sparc_catalog.csv``.

This catalog is the primary input for ``run_crtt.py`` and ``run_force_models.py``.

Pipeline
--------
1. Parse ``*_rotmod.dat`` files to extract (r, v_obs) curves.
2. Compute outer-slope at r ≥ OUTER_FRAC · Rmax (log-linear fit).
3. Merge with SPARC Lelli+2016 mass catalog.
4. Compute log baryonic mass (logMbar) with HE correction for gas.
5. Write ``data/processed/sparc_catalog.csv``.

Constants
---------
OUTER_FRAC = 0.7        outer-regime threshold (fraction of Rmax)
MIN_OUTER_POINTS = 4    minimum points required for a valid slope fit
HE_CORRECTION = 1.33    helium correction applied to HI gas mass
UPSILON_DISK = 1.0      stellar M/L ratio (disk)
UPSILON_BULGE = 1.0     stellar M/L ratio (bulge)

Usage
-----
With default paths::

    python scripts/build_sparc_catalog.py

Explicit options::

    python scripts/build_sparc_catalog.py \\
        --sparc-dir  data/raw/sparc_rotmod \\
        --catalog    data/raw/SPARC_Lelli2016c.mrt \\
        --out        data/processed/sparc_catalog.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTER_FRAC: float = 0.7
MIN_OUTER_POINTS: int = 4
HE_CORRECTION: float = 1.33
UPSILON_DISK: float = 1.0
UPSILON_BULGE: float = 1.0

_OUT_DEFAULT = "data/processed/sparc_catalog.csv"
_SPARC_DIR_DEFAULT = "data/raw/sparc_rotmod"
_CATALOG_DEFAULT = "data/raw/SPARC_Lelli2016c.mrt"


# ---------------------------------------------------------------------------
# Rotmod parsing
# ---------------------------------------------------------------------------

def galaxy_name_from_path(path: str | Path) -> str:
    """Return galaxy name from a rotmod file path (strips ``_rotmod`` suffix).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    str
    """
    return Path(path).stem.replace("_rotmod", "")


def parse_rotmod(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a SPARC ``*_rotmod.dat`` file and return (r_kpc, v_obs_kms).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    r : np.ndarray
        Galactocentric radii in kpc.
    v : np.ndarray
        Observed rotation velocities in km/s.
    """
    rows: list[tuple[float, float]] = []
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


# ---------------------------------------------------------------------------
# Outer-slope computation
# ---------------------------------------------------------------------------

def compute_outer_slope(
    r: np.ndarray,
    v: np.ndarray,
    outer_frac: float = OUTER_FRAC,
    min_points: int = MIN_OUTER_POINTS,
) -> dict:
    """Fit a log-linear slope to the outer regime of a rotation curve.

    Parameters
    ----------
    r : np.ndarray
        Radii in kpc.
    v : np.ndarray
        Observed velocities in km/s.
    outer_frac : float
        Fraction of Rmax defining the outer boundary (default 0.7).
    min_points : int
        Minimum outer-regime points required for a valid fit.

    Returns
    -------
    dict
        Keys: ``slope_tail``, ``Rmax``, ``Vmax``, ``n_outer``, ``outer_fit_ok``.
    """
    mask = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    r, v = r[mask], v[mask]

    base: dict = {
        "slope_tail": np.nan,
        "Rmax": np.nan,
        "Vmax": np.nan,
        "n_outer": 0,
        "outer_fit_ok": False,
    }
    if len(r) < 2:
        return base

    rmax = float(r.max())
    vmax = float(v.max())
    outer = r >= outer_frac * rmax
    n_outer = int(outer.sum())

    base["Rmax"] = rmax
    base["Vmax"] = vmax
    base["n_outer"] = n_outer

    if n_outer < min_points:
        return base

    result = linregress(np.log10(r[outer]), np.log10(v[outer]))
    base["slope_tail"] = float(result.slope)
    base["outer_fit_ok"] = True
    return base


# ---------------------------------------------------------------------------
# Baryonic mass
# ---------------------------------------------------------------------------

def compute_logMbar(
    row: pd.Series,
    upsilon_disk: float = UPSILON_DISK,
    upsilon_bulge: float = UPSILON_BULGE,
    he_correction: float = HE_CORRECTION,
) -> float:
    """Compute log10(Mbar / Msun) from a SPARC catalog row.

    Parameters
    ----------
    row : pd.Series
        Must contain ``L36`` (luminosity, 10^9 Lsun) and ``Mgas`` (10^9 Msun).
        Optionally ``BulgeFlag``.

    Returns
    -------
    float
        log10(Mbar / Msun), or np.nan if mass is non-positive.
    """
    l36 = float(row.get("L36", 0.0)) * 1e9
    mgas = float(row.get("Mgas", 0.0)) * 1e9 * he_correction
    bulge = int(row.get("BulgeFlag", 0))
    mstar = (upsilon_bulge if bulge else upsilon_disk) * l36
    mbar = mstar + mgas
    return float(np.log10(mbar)) if mbar > 0 else np.nan


# ---------------------------------------------------------------------------
# Main catalog builder
# ---------------------------------------------------------------------------

def build_sparc_catalog(
    sparc_dir: str | Path = _SPARC_DIR_DEFAULT,
    catalog: str | Path = _CATALOG_DEFAULT,
    out: str | Path = _OUT_DEFAULT,
    outer_frac: float = OUTER_FRAC,
    min_points: int = MIN_OUTER_POINTS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the processed SPARC catalog.

    Parameters
    ----------
    sparc_dir : str or Path
        Directory containing ``*_rotmod.dat`` files.
    catalog : str or Path
        Path to SPARC mass catalog (CSV with ``Name``, ``L36``, ``Mgas`` columns).
    out : str or Path
        Output CSV path.
    outer_frac : float
        Outer-regime threshold as fraction of Rmax.
    min_points : int
        Minimum outer-regime points for a valid slope fit.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Per-galaxy processed catalog.
    """
    sparc_dir = Path(sparc_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rotmod_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    if verbose:
        print(f"Found {len(rotmod_files)} rotmod files in {sparc_dir}")

    # Load mass catalog if present
    cat: pd.DataFrame = pd.DataFrame()
    cat_path = Path(catalog)
    if cat_path.exists():
        try:
            cat = pd.read_csv(cat_path, comment="#")
        except Exception as exc:
            if verbose:
                print(f"Warning: could not read catalog {catalog}: {exc}")

    records: list[dict] = []
    for fpath in rotmod_files:
        name = galaxy_name_from_path(fpath)
        r, v = parse_rotmod(fpath)
        stats = compute_outer_slope(r, v, outer_frac=outer_frac,
                                    min_points=min_points)
        rec: dict = {"galaxy": name, **stats}

        if not cat.empty and "Name" in cat.columns:
            row_match = cat[cat["Name"] == name]
            if not row_match.empty:
                row = row_match.iloc[0]
                rec["logMbar"] = compute_logMbar(row)
                rec["Mgas"] = float(row.get("Mgas", np.nan))
        records.append(rec)

    df = pd.DataFrame(records)
    n_total = len(df)
    df_clean = df[df["outer_fit_ok"].astype(bool)].copy()
    n_clean = len(df_clean)

    if verbose:
        print(f"Total galaxies parsed: {n_total}")
        print(f"Clean sample (outer_fit_ok): {n_clean}")

    df_clean.to_csv(out, index=False)
    if verbose:
        print(f"Catalog written to {out}")
    return df_clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Build processed SPARC catalog from rotmod files."
    )
    parser.add_argument("--sparc-dir", default=_SPARC_DIR_DEFAULT,
                        help="Directory with *_rotmod.dat files")
    parser.add_argument("--catalog", default=_CATALOG_DEFAULT,
                        help="SPARC mass catalog CSV")
    parser.add_argument("--out", default=_OUT_DEFAULT,
                        help="Output CSV path")
    parser.add_argument("--outer-frac", type=float, default=OUTER_FRAC,
                        help=f"Outer-regime fraction (default: {OUTER_FRAC})")
    parser.add_argument("--min-points", type=int, default=MIN_OUTER_POINTS,
                        help=f"Min outer points (default: {MIN_OUTER_POINTS})")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args(argv)

    df = build_sparc_catalog(
        sparc_dir=args.sparc_dir,
        catalog=args.catalog,
        out=args.out,
        outer_frac=args.outer_frac,
        min_points=args.min_points,
        verbose=args.verbose,
    )
    result = {"n": len(df), "output": args.out, "columns": list(df.columns)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
