#!/usr/bin/env python3
"""
build_hinge_sfr_inputs_from_sparc.py
=====================================
Convert SPARC rotation-curve files into the two CSVs expected by
``scripts/hinge_sfr_test.py``:

* ``profiles.csv``      — per radial point: galaxy, r_kpc, vbar_kms, rmax_kpc
* ``galaxy_table.csv``  — per galaxy:       galaxy, log_mbar, log_sfr, morph_bin

The script follows the **same file layout** as ``scripts/compare_nu_models.py``
so it works out-of-the-box once the SPARC data is downloaded.

SPARC data download
-------------------
Real data is *not* bundled in this repository.  Download from:
    http://astroweb.cwru.edu/SPARC/

Then place the files as::

    data/SPARC/
    ├── SPARC_Lelli2016c.mrt          (galaxy summary table, whitespace-delimited)
    │   or SPARC_Lelli2016c.csv       (comma-delimited alternative)
    └── raw/
        ├── NGC0300_rotmod.dat
        ├── NGC0891_rotmod.dat
        └── ...

Galaxy-table columns used
-------------------------
* ``Galaxy``  — galaxy name (must match rotmod filename stem)
* ``Mstar``   — stellar mass [10⁹ M☉] (from 3.6 µm luminosity × Υ*)
* ``MHI``     — HI gas mass  [10⁹ M☉]
* ``T``       — morphological Hubble type (integer, -3 … 10)

Baryonic mass
-------------
    M_bar = 1.33 * MHI  +  Mstar         [units: 10⁹ M☉]
    log_mbar = log10(M_bar * 1e9)         [log M☉]

(factor 1.33 accounts for helium)

SFR note
--------
SPARC does not include SFR measurements.  By default this script uses a
star-forming main-sequence proxy calibrated at z ≈ 0:

    log_sfr ≈ 0.76 * log_mbar − 7.64     [log M☉/yr]

(McGaugh 2017; Speagle et al. 2014 at z = 0 normalisation)

For a definitive test you should supply real SFR measurements via
``--sfr-table``.  The table must be a CSV with columns ``galaxy`` and
``log_sfr`` (or ``sfr`` in M☉/yr).

Morphology bins
---------------
* ``T ≥ 5``  → ``"late"``   (Sd, Sm, Im, irregulars)
* ``0 ≤ T < 5`` → ``"inter"``  (Sa, Sb, Sc)
* ``T < 0``  → ``"early"``  (S0, elliptical)

Usage
-----
Default paths::

    python scripts/build_hinge_sfr_inputs_from_sparc.py

Custom paths / output::

    python scripts/build_hinge_sfr_inputs_from_sparc.py \\
        --data-dir  data/SPARC \\
        --out-dir   data/hinge_sfr \\
        --sfr-table external/sfr_measurements.csv \\
        --min-pts   10 \\
        --quality   1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Main-sequence SFR proxy (McGaugh 2017 / Speagle 2014 z=0 calibration):
#   log_sfr = MS_SLOPE * log_mbar + MS_INTERCEPT
MS_SLOPE = 0.76
MS_INTERCEPT = -7.64

# Helium correction factor for gas mass: M_gas = HE_FACTOR * M_HI
HE_FACTOR = 1.33

# Minimum radial points required to compute F-proxies reliably
DEFAULT_MIN_PTS = 10

# Quality flags to keep (SPARC Q=1 best, Q=3 worst)
DEFAULT_QUALITY = (1, 2)

# SPARC Mstar / MHI are given in units of 10⁹ M☉; multiply to get M☉
_SPARC_MASS_UNIT = 1e9  # [M☉ per SPARC mass unit]

# Floor applied to linear SFR before taking log10 (avoids log(0))
_SFR_FLOOR = 1e-12  # M☉/yr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_galaxy_table(data_dir: Path) -> Path:
    """Return the path of the SPARC galaxy-summary file."""
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {data_dir}.  "
        "Expected SPARC_Lelli2016c.csv or .mrt under data/SPARC/.  "
        "See data/README.md for download instructions."
    )


def load_galaxy_table(data_dir: Path) -> pd.DataFrame:
    """Load the SPARC galaxy summary table."""
    p = _find_galaxy_table(data_dir)
    sep = "," if p.suffix == ".csv" else r"\s+"
    return pd.read_csv(p, sep=sep, comment="#")


def _find_rotmod(data_dir: Path, name: str) -> Path | None:
    """Return the rotmod file path for *name*, or None if absent."""
    candidates = [
        data_dir / f"{name}_rotmod.dat",
        data_dir / "raw" / f"{name}_rotmod.dat",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_rotmod(path: Path) -> pd.DataFrame:
    """Read a SPARC ``*_rotmod.dat`` file.

    SPARC rotmod columns (fixed order, whitespace-delimited):
        Rad   Vobs   e_Vobs   Vgas   Vdisk   Vbul   SBdisk   SBbul

    Returns a DataFrame with columns renamed to ``r``, ``v_gas``,
    ``v_disk``, ``v_bul``.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
               "SBdisk", "SBbul"],
    )
    return df


def compute_vbar(df: pd.DataFrame) -> np.ndarray:
    """Compute baryonic velocity in km/s from SPARC rotmod columns.

        v_bar = sqrt(v_gas² + v_disk² + v_bul²)

    Each component can be negative in SPARC to indicate retrograde gas/bulge;
    the absolute value is used here (standard practice).
    """
    vg = df["v_gas"].to_numpy(dtype=float)
    vd = df["v_disk"].to_numpy(dtype=float)
    vb = df["v_bul"].to_numpy(dtype=float)
    return np.sqrt(vg ** 2 + vd ** 2 + vb ** 2)


def morph_bin(t: float) -> str:
    """Map SPARC Hubble type *t* to a coarse morphology string."""
    if t >= 5:
        return "late"
    if t >= 0:
        return "inter"
    return "early"


def main_sequence_log_sfr(log_mbar: float) -> float:
    """Return the z≈0 main-sequence SFR proxy (log M☉/yr) from log_mbar."""
    return MS_SLOPE * log_mbar + MS_INTERCEPT


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_inputs(
    data_dir: Path,
    out_dir: Path,
    sfr_table: Path | None = None,
    min_pts: int = DEFAULT_MIN_PTS,
    quality_keep: tuple[int, ...] = DEFAULT_QUALITY,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build profiles and galaxy tables from SPARC data.

    Parameters
    ----------
    data_dir : Path
        Root directory containing SPARC files (``SPARC_Lelli2016c.csv`` and
        ``raw/`` subdirectory with rotmod files).
    out_dir : Path
        Where to write ``profiles.csv`` and ``galaxy_table.csv``.
    sfr_table : Path or None
        Optional CSV with real SFR measurements.  Must contain ``galaxy``
        and either ``log_sfr`` or ``sfr`` columns.
    min_pts : int
        Minimum number of radial points required to include a galaxy.
    quality_keep : tuple of int
        SPARC quality flags to accept (1 = best).  If the galaxy table
        has no ``Q`` column, all galaxies are accepted.
    verbose : bool
        Print progress messages.

    Returns
    -------
    (profiles_df, galaxy_df) : tuple of DataFrames
        The two DataFrames (also written to *out_dir*).

    Raises
    ------
    FileNotFoundError
        If the SPARC galaxy summary table is not found.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    galaxy_table = load_galaxy_table(data_dir)
    _log(f"SPARC galaxy table: {len(galaxy_table)} rows")

    # --- Load optional external SFR measurements --------------------------
    sfr_map: dict[str, float] = {}
    if sfr_table is not None:
        sfr_df = pd.read_csv(sfr_table)
        if "log_sfr" in sfr_df.columns:
            sfr_map = dict(zip(sfr_df["galaxy"], sfr_df["log_sfr"].astype(float)))
        elif "sfr" in sfr_df.columns:
            sfr_map = {
                g: float(np.log10(max(s, _SFR_FLOOR)))
                for g, s in zip(sfr_df["galaxy"], sfr_df["sfr"].astype(float))
            }
        else:
            raise ValueError(
                f"SFR table {sfr_table} must have a 'log_sfr' or 'sfr' column."
            )
        _log(f"External SFR measurements loaded: {len(sfr_map)} galaxies")

    profile_rows: list[dict] = []
    galaxy_rows: list[dict] = []
    n_skip_quality = 0
    n_skip_pts = 0
    n_skip_mass = 0
    n_skip_rotmod = 0

    for _, grow in galaxy_table.iterrows():
        name = str(grow["Galaxy"])

        # Quality filter
        if "Q" in galaxy_table.columns:
            q = int(grow.get("Q", 1))
            if q not in quality_keep:
                n_skip_quality += 1
                continue

        # Locate rotmod file
        rotmod_path = _find_rotmod(data_dir, name)
        if rotmod_path is None:
            n_skip_rotmod += 1
            _log(f"  [skip] {name}: rotmod not found")
            continue

        rc = load_rotmod(rotmod_path)
        if len(rc) < min_pts:
            n_skip_pts += 1
            _log(f"  [skip] {name}: only {len(rc)} radial points (< {min_pts})")
            continue

        r = rc["r"].to_numpy(dtype=float)
        vbar = compute_vbar(rc)
        rmax = float(np.nanmax(r))

        # Baryonic mass: M_bar = 1.33 * MHI + Mstar  [units: 1e9 M_sun]
        try:
            mstar = float(grow["Mstar"])
            mhi = float(grow["MHI"])
        except KeyError:
            n_skip_mass += 1
            _log(f"  [skip] {name}: Mstar or MHI column missing")
            continue

        mbar_1e9 = HE_FACTOR * mhi + mstar  # 10⁹ M☉
        if mbar_1e9 <= 0:
            n_skip_mass += 1
            _log(f"  [skip] {name}: M_bar ≤ 0")
            continue

        log_mbar = float(np.log10(mbar_1e9 * _SPARC_MASS_UNIT))  # log10(M☉)

        # Morphology
        try:
            t_val = float(grow["T"])
        except (KeyError, ValueError):
            t_val = 5.0  # default: late type if unknown
        mbin = morph_bin(t_val)

        # SFR: use real measurement if available, else main-sequence proxy
        if name in sfr_map:
            log_sfr = sfr_map[name]
        else:
            log_sfr = main_sequence_log_sfr(log_mbar)

        # Accumulate profile rows
        for ri, vi in zip(r, vbar):
            profile_rows.append({
                "galaxy": name,
                "r_kpc": round(float(ri), 6),
                "vbar_kms": round(float(vi), 6),
                "rmax_kpc": round(rmax, 6),
            })

        galaxy_rows.append({
            "galaxy": name,
            "log_mbar": round(log_mbar, 6),
            "log_sfr": round(log_sfr, 6),
            "morph_bin": mbin,
        })

    _PROFILES_COLS = ["galaxy", "r_kpc", "vbar_kms", "rmax_kpc"]
    _GALAXY_COLS = ["galaxy", "log_mbar", "log_sfr", "morph_bin"]

    profiles_df = (
        pd.DataFrame(profile_rows, columns=_PROFILES_COLS)
        if profile_rows
        else pd.DataFrame(columns=_PROFILES_COLS)
    )
    galaxy_df = (
        pd.DataFrame(galaxy_rows, columns=_GALAXY_COLS)
        if galaxy_rows
        else pd.DataFrame(columns=_GALAXY_COLS)
    )

    n_ok = len(galaxy_df)
    _log(
        f"\nSummary: {n_ok} galaxies written, "
        f"{n_skip_rotmod} skipped (rotmod missing), "
        f"{n_skip_pts} skipped (< {min_pts} pts), "
        f"{n_skip_quality} skipped (quality flag), "
        f"{n_skip_mass} skipped (mass data missing)."
    )
    _log(f"Profile rows: {len(profiles_df)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = out_dir / "profiles.csv"
    galaxy_path = out_dir / "galaxy_table.csv"
    profiles_df.to_csv(profiles_path, index=False)
    galaxy_df.to_csv(galaxy_path, index=False)
    _log(f"\nWrote: {profiles_path}")
    _log(f"Wrote: {galaxy_path}")

    if sfr_map:
        n_proxy = sum(1 for g in galaxy_df["galaxy"] if g not in sfr_map)
        if n_proxy:
            _log(
                f"\nNOTE: {n_proxy} galaxies used the main-sequence SFR proxy "
                "(no real SFR supplied).  Results with proxy SFR are indicative "
                "only.  Supply --sfr-table for a definitive test."
            )
    else:
        _log(
            "\nNOTE: All SFR values are from the main-sequence proxy "
            f"(log_sfr = {MS_SLOPE}*log_mbar + {MS_INTERCEPT}).  "
            "Supply --sfr-table for a definitive test."
        )

    return profiles_df, galaxy_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build profiles.csv and galaxy_table.csv from SPARC rotmod files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data/SPARC",
        help=(
            "Root directory containing SPARC_Lelli2016c.csv (or .mrt) "
            "and raw/ subdirectory with *_rotmod.dat files."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/hinge_sfr",
        help="Output directory for profiles.csv and galaxy_table.csv.",
    )
    parser.add_argument(
        "--sfr-table",
        default=None,
        help=(
            "CSV with real SFR measurements (columns: galaxy, log_sfr or sfr). "
            "When omitted, the z≈0 main-sequence proxy is used."
        ),
    )
    parser.add_argument(
        "--min-pts",
        type=int,
        default=DEFAULT_MIN_PTS,
        help="Minimum radial points per galaxy.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUALITY),
        help="SPARC quality flags to keep (1=best, 3=worst).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = _parse_args(argv)
    build_inputs(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        sfr_table=Path(args.sfr_table) if args.sfr_table else None,
        min_pts=args.min_pts,
        quality_keep=tuple(args.quality),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
