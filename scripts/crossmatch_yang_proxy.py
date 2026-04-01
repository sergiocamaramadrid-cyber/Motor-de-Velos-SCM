#!/usr/bin/env python3
"""
scripts/crossmatch_yang_proxy.py — YANG × SPARC environmental crossmatch.

For each SPARC galaxy the script computes two quantities and saves them to a
single CSV that ``generate_env_figure.py`` can read directly:

  delta_mass  (float)
      Large-scale stellar mass overdensity within a 3 Mpc sphere centred on the
      galaxy:  δ = ρ_local / ⟨ρ⟩ − 1.
      ρ_local = Σ M★ / V  (neighbours from the YANG catalog with log M★ > 9).
      ⟨ρ⟩ = mean of ρ_local across the full SPARC sample.
      Galaxies with no YANG neighbours within 3 Mpc are assigned δ ≈ −1 (void).

  rho_lag1  (float)
      Lag-1 Pearson autocorrelation of  Δ_F3(r) = log10 g_obs − log10 g_bar
      along each galaxy's rotation curve.  Requires at least 3 radial points.
      When SPARC rotation curves are unavailable the column falls back to the
      deep-regime slope β from the F3 catalog (``results/f3_catalog_real.csv``).

Outputs
-------
  results/delta_mass_yang_sparc.csv
      Columns: galaxy, delta_mass, rho_lag1

Prerequisites
-------------
  1.  YANG catalog:  data/yang/yang_catalog.csv
         Run: python scripts/fetch_yang_catalog.py --url <URL>
  2a. SPARC rotation curves in data/SPARC/raw/ (for rho_lag1 from kinematics)
         Run: python scripts/download_sparc_data.py
  2b. OR F3 catalog at results/f3_catalog_real.csv (beta-proxy fallback)
         Run: python scripts/generate_f3_catalog.py

Usage
-----
    python scripts/crossmatch_yang_proxy.py

    # Custom paths:
    python scripts/crossmatch_yang_proxy.py \\
        --yang   data/yang/yang_catalog.csv \\
        --sparc  data/SPARC \\
        --f3     results/f3_catalog_real.csv \\
        --out    results/delta_mass_yang_sparc.csv \\
        --radius 3.0

Notes
-----
* RA/Dec of SPARC galaxies: if the SPARC table contains RA/Dec columns they are
  used for a proper 3-D crossmatch.  Otherwise astroquery.ned is attempted
  (requires network + astroquery≥0.4).  If both are unavailable the crossmatch
  falls back to a 1-D distance-only redshift slice (width ±1.5 Mpc), which is
  a rough proxy but still captures large-scale structure trends.

* The YANG catalog must have at least  ra  and  dec  columns (normalised by
  ``fetch_yang_catalog.py``).  A  z  or  log_mstar  column improves accuracy.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_YANG = Path("data/yang/yang_catalog.csv")
_DEFAULT_SPARC = Path("data/SPARC")
_DEFAULT_F3 = Path("results/f3_catalog_real.csv")
_DEFAULT_OUT = Path("results/delta_mass_yang_sparc.csv")
_DEFAULT_RADIUS_MPC = 3.0        # sphere radius for delta_mass
_MSTAR_FLOOR_LOG = 9.0           # minimum log10(M*/M_sun) to include in sum
_H0 = 70.0                       # km/s/Mpc (for z → D conversion)
_C = 3e5                         # km/s
_DEG2RAD = np.pi / 180.0
_MIN_LAG1_POINTS = 3             # minimum RC points for rho_lag1
_CONV = 1e6 / 3.085677581e19     # (km/s)²/kpc → m/s²
# F3 beta proxy constants (deep-MOND prediction β = 0.5; rescale to [-1, +1])
_BETA_MOND = 0.5                 # expected deep-MOND value; maps to rho_lag1 = 0
_BETA_SCALE = 2.0                # 1.0 → +1, 0.0 → −1


# ---------------------------------------------------------------------------
# YANG catalog loading
# ---------------------------------------------------------------------------

def _load_yang(path: Path) -> pd.DataFrame:
    """Load and sanity-check the YANG catalog."""
    if not path.exists():
        raise FileNotFoundError(
            f"YANG catalog not found: {path}\n"
            "  → Run: python scripts/fetch_yang_catalog.py --url <URL>"
        )
    df = pd.read_csv(path, low_memory=False)
    # lower-case column names for robustness
    df.columns = [c.strip().lower() for c in df.columns]

    # Require at least positional columns (ra/dec or z)
    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError(
            f"YANG catalog at {path} is missing 'ra' and/or 'dec' columns. "
            "Re-run fetch_yang_catalog.py; it normalises column names."
        )

    # Ensure numeric types
    for col in ["ra", "dec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "z" in df.columns:
        df["z"] = pd.to_numeric(df["z"], errors="coerce")

    if "log_mstar" in df.columns:
        df["log_mstar"] = pd.to_numeric(df["log_mstar"], errors="coerce")
    elif "log_mhalo" in df.columns:
        df["log_mhalo"] = pd.to_numeric(df["log_mhalo"], errors="coerce")
        # coarse proxy: log M★ ≈ log M_halo − 1.5  (abundance matching)
        df["log_mstar"] = df["log_mhalo"] - 1.5

    df = df.dropna(subset=["ra", "dec"])
    print(f"Loaded YANG catalog: {len(df)} galaxies from {path}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPARC table loading
# ---------------------------------------------------------------------------

def _load_sparc_table(sparc_dir: Path) -> pd.DataFrame:
    """Load the SPARC galaxy table; returns DataFrame with Galaxy + D columns."""
    candidates = [
        sparc_dir / "SPARC_Lelli2016c.csv",
        sparc_dir / "SPARC_Lelli2016c.mrt",
        sparc_dir / "raw" / "SPARC_Lelli2016c.csv",
    ]
    for p in candidates:
        if p.exists():
            sep = "," if p.suffix == ".csv" else r"\s+"
            df = pd.read_csv(p, sep=sep, comment="#")
            print(f"Loaded SPARC table: {len(df)} galaxies from {p}")
            return df
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {sparc_dir}.\n"
        "  → Run: python scripts/download_sparc_data.py"
    )


def _get_sparc_coords(sparc_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns (galaxy, D_mpc, ra, dec).

    Tries (in order):
    1. RA/Dec columns already in sparc_df.
    2. astroquery.ned lookup by galaxy name.
    3. Returns None for RA/Dec (1-D fallback will be used).
    """
    galaxy_col = _find_col(sparc_df, ["Galaxy", "galaxy", "Name", "name"])
    d_col = _find_col(sparc_df, ["D", "d", "dist", "Dist", "D_Mpc"])

    if galaxy_col is None:
        raise ValueError("SPARC table lacks a 'Galaxy' column.")
    if d_col is None:
        raise ValueError("SPARC table lacks a distance 'D' column.")

    out = pd.DataFrame({
        "galaxy": sparc_df[galaxy_col].astype(str),
        "D_mpc":  pd.to_numeric(sparc_df[d_col], errors="coerce"),
    })

    # Check for existing RA/Dec columns
    ra_col = _find_col(sparc_df, ["RA", "ra", "RAJ2000", "RAdeg"])
    dec_col = _find_col(sparc_df, ["Dec", "dec", "DEJ2000", "DEdeg", "DE"])
    if ra_col and dec_col:
        out["ra"] = pd.to_numeric(sparc_df[ra_col], errors="coerce")
        out["dec"] = pd.to_numeric(sparc_df[dec_col], errors="coerce")
        n_coords = out[["ra", "dec"]].dropna().shape[0]
        print(f"  RA/Dec found in SPARC table ({n_coords} galaxies with coords).")
        return out

    # Try astroquery
    try:
        from astroquery.ned import Ned  # type: ignore
        print("  Querying NED for SPARC galaxy positions (this may take a moment)…")
        ras, decs = [], []
        for name in out["galaxy"]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tbl = Ned.query_object(name)
                ras.append(float(tbl["RA"][0]))
                decs.append(float(tbl["DEC"][0]))
            except Exception:
                ras.append(np.nan)
                decs.append(np.nan)
        out["ra"] = ras
        out["dec"] = decs
        n_coords = out[["ra", "dec"]].dropna().shape[0]
        print(f"  NED resolved {n_coords}/{len(out)} galaxy positions.")
    except ImportError:
        print(
            "  astroquery not available; using 1-D (distance-only) crossmatch "
            "fallback.  For a proper 3-D crossmatch install astroquery:\n"
            "    pip install astroquery",
            file=sys.stderr,
        )
        out["ra"] = np.nan
        out["dec"] = np.nan

    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# delta_mass computation
# ---------------------------------------------------------------------------

def _angular_sep_deg(ra1: float, dec1: float,
                     ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """Vectorised great-circle separation in degrees (haversine)."""
    ra1r = ra1 * _DEG2RAD
    dec1r = dec1 * _DEG2RAD
    ra2r = ra2 * _DEG2RAD
    dec2r = dec2 * _DEG2RAD
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = np.sin(ddec / 2) ** 2 + np.cos(dec1r) * np.cos(dec2r) * np.sin(dra / 2) ** 2
    return 2 * np.degrees(np.arcsin(np.clip(np.sqrt(a), 0, 1)))


def _compute_delta_mass(
    sparc_coords: pd.DataFrame,
    yang: pd.DataFrame,
    radius_mpc: float,
) -> pd.Series:
    """Compute per-galaxy delta_mass using YANG neighbours within radius_mpc.

    Returns a Series indexed by sparc_coords.index with delta_mass values.
    """
    # Stellar masses of YANG galaxies (in M_sun, not log).
    # Only include galaxies above the stellar mass floor to minimise
    # incompleteness, matching the definition in methods_delta_mass.md.
    if "log_mstar" in yang.columns:
        above_floor = yang["log_mstar"].fillna(-np.inf) >= _MSTAR_FLOOR_LOG
        yang_mstar = np.where(
            above_floor,
            10.0 ** yang["log_mstar"].where(above_floor, 0).values,
            0.0,
        )
    else:
        # Unknown masses: assume all equal to 10^10 M_sun
        yang_mstar = np.full(len(yang), 1e10)

    yang_ra = yang["ra"].values
    yang_dec = yang["dec"].values
    yang_z = yang["z"].values if "z" in yang.columns else None

    # Precompute YANG comoving distances once (independent of SPARC loop)
    yang_D: np.ndarray | None = (yang_z * _C / _H0) if yang_z is not None else None

    sphere_vol = (4.0 / 3.0) * np.pi * radius_mpc ** 3   # Mpc³

    rho_locals = []
    for _, row in sparc_coords.iterrows():
        D = row["D_mpc"]
        if np.isnan(D) or D <= 0:
            rho_locals.append(np.nan)
            continue

        has_coords = not (np.isnan(row.get("ra", np.nan)) or
                          np.isnan(row.get("dec", np.nan)))

        if has_coords:
            # Angular radius corresponding to radius_mpc at distance D
            # theta_max (deg) = radius_mpc / D * (180/pi)
            theta_max_deg = (radius_mpc / D) * (180.0 / np.pi)

            ang_sep = _angular_sep_deg(row["ra"], row["dec"], yang_ra, yang_dec)
            mask_sky = ang_sep < theta_max_deg

            # Line-of-sight distance filter (if redshifts available)
            if yang_D is not None:
                dlos = np.abs(yang_D - D)
                mask_los = dlos < radius_mpc
                mask = mask_sky & mask_los & np.isfinite(yang_mstar) & (yang_mstar > 0)
            else:
                mask = mask_sky & np.isfinite(yang_mstar) & (yang_mstar > 0)
        else:
            # 1-D fallback: distance slice only
            if yang_D is None:
                # Cannot match without any positional info
                rho_locals.append(np.nan)
                continue
            dlos = np.abs(yang_D - D)
            mask = (dlos < radius_mpc) & np.isfinite(yang_mstar) & (yang_mstar > 0)

        sum_mstar = float(yang_mstar[mask].sum()) if mask.any() else 0.0
        rho_locals.append(sum_mstar / sphere_vol)

    rho_arr = np.array(rho_locals, dtype=float)
    rho_mean = np.nanmean(rho_arr)

    if rho_mean == 0:
        # All galaxies are in voids relative to YANG: assign void floor
        delta = np.full(len(sparc_coords), -1.0)
    else:
        delta = rho_arr / rho_mean - 1.0
        # Galaxies truly without any neighbours get the void floor
        delta = np.where(rho_arr == 0, -1.0, delta)

    return pd.Series(delta, index=sparc_coords.index, name="delta_mass")


# ---------------------------------------------------------------------------
# rho_lag1 computation
# ---------------------------------------------------------------------------

def _compute_rho_lag1_from_rc(sparc_dir: Path, galaxy: str) -> float | None:
    """Compute lag-1 autocorrelation of Δ_F3(r) from a SPARC rotation curve."""
    raw_dir = sparc_dir / "raw"
    for rc_path in [sparc_dir / f"{galaxy}_rotmod.dat",
                    raw_dir / f"{galaxy}_rotmod.dat"]:
        if rc_path.exists():
            try:
                df = pd.read_csv(
                    rc_path, sep=r"\s+", comment="#",
                    names=["r", "v_obs", "v_obs_err", "v_gas",
                           "v_disk", "v_bul", "SBdisk", "SBbul"],
                )
                r = df["r"].values           # kpc
                v_obs = df["v_obs"].values   # km/s
                v_gas = df["v_gas"].values
                v_disk = df["v_disk"].values
                v_bul = df["v_bul"].values

                # g_obs = v_obs² / r  (converted to m/s²)
                with np.errstate(divide="ignore", invalid="ignore"):
                    g_obs = np.where(r > 0,
                                     v_obs ** 2 / r * _CONV, np.nan)
                    g_bar_sq = (v_gas ** 2 + v_disk ** 2 + v_bul ** 2)
                    g_bar = np.where(r > 0, g_bar_sq / r * _CONV, np.nan)

                    delta_f3 = np.log10(g_obs) - np.log10(g_bar)

                valid = np.isfinite(delta_f3)
                if valid.sum() < _MIN_LAG1_POINTS:
                    return None
                dv = delta_f3[valid]
                corr, _ = pearsonr(dv[:-1], dv[1:])
                return float(corr)
            except Exception:
                return None
    return None


def _compute_rho_lag1(sparc_coords: pd.DataFrame,
                      sparc_dir: Path,
                      f3_catalog: pd.DataFrame | None) -> pd.Series:
    """Return per-galaxy rho_lag1, using rotation curves then f3 fallback."""
    values = []
    source_counts = {"rc": 0, "f3_proxy": 0, "nan": 0}

    f3_beta: dict[str, float] = {}
    if f3_catalog is not None:
        gcol = _find_col(f3_catalog, ["galaxy", "Galaxy", "name", "Name"])
        bcol = _find_col(f3_catalog, ["beta", "Beta", "friction_slope",
                                      "deep_slope"])
        if gcol and bcol:
            for _, row in f3_catalog.iterrows():
                f3_beta[str(row[gcol])] = float(row[bcol])

    for _, row in sparc_coords.iterrows():
        galaxy = str(row["galaxy"])

        # Primary: rotation curve
        rho_lag1 = _compute_rho_lag1_from_rc(sparc_dir, galaxy)
        if rho_lag1 is not None:
            values.append(rho_lag1)
            source_counts["rc"] += 1
            continue

        # Fallback: beta proxy  (β ≈ 0.5 in deep MOND; rescale to [-1, 1])
        if galaxy in f3_beta:
            beta = f3_beta[galaxy]
            # rho_lag1 proxy: (beta − _BETA_MOND) * _BETA_SCALE; 0.5→0, 1.0→+1, 0→−1
            values.append(np.clip((beta - _BETA_MOND) * _BETA_SCALE, -1.0, 1.0))
            source_counts["f3_proxy"] += 1
            continue

        values.append(np.nan)
        source_counts["nan"] += 1

    print(
        f"  rho_lag1 sources — rotation curves: {source_counts['rc']}, "
        f"F3 beta proxy: {source_counts['f3_proxy']}, "
        f"missing: {source_counts['nan']}"
    )
    return pd.Series(values, index=sparc_coords.index, name="rho_lag1")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def crossmatch(
    yang_path: Path = _DEFAULT_YANG,
    sparc_dir: Path = _DEFAULT_SPARC,
    f3_path: Path = _DEFAULT_F3,
    out_path: Path = _DEFAULT_OUT,
    radius_mpc: float = _DEFAULT_RADIUS_MPC,
) -> pd.DataFrame:
    """Run the full YANG × SPARC crossmatch and return the result DataFrame."""

    # ---- Load YANG ------------------------------------------------------
    yang = _load_yang(yang_path)

    # ---- Load SPARC table -----------------------------------------------
    try:
        sparc_df = _load_sparc_table(sparc_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sparc_coords = _get_sparc_coords(sparc_df)
    sparc_coords = sparc_coords.dropna(subset=["D_mpc"])
    print(f"SPARC galaxies with valid distances: {len(sparc_coords)}")

    n_with_coords = sparc_coords[["ra", "dec"]].dropna().shape[0]
    if n_with_coords == 0:
        print(
            "  WARNING: No RA/Dec coordinates for SPARC galaxies. "
            "Using 1-D redshift-distance crossmatch (requires z in YANG catalog).",
            file=sys.stderr,
        )
        if "z" not in yang.columns:
            print(
                "ERROR: YANG catalog has no 'z' column either. "
                "Cannot perform any spatial crossmatch. "
                "Provide a YANG catalog with 'z' or install astroquery.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ---- Compute delta_mass ---------------------------------------------
    print(f"\nComputing delta_mass (radius = {radius_mpc} Mpc) …")
    delta_mass = _compute_delta_mass(sparc_coords, yang, radius_mpc)

    # ---- Load F3 catalog (optional) ------------------------------------
    f3_catalog: pd.DataFrame | None = None
    if f3_path.exists():
        try:
            f3_catalog = pd.read_csv(f3_path)
            print(f"Loaded F3 catalog: {len(f3_catalog)} rows from {f3_path}")
        except Exception as exc:
            print(f"  WARNING: could not load F3 catalog: {exc}", file=sys.stderr)

    # ---- Compute rho_lag1 ----------------------------------------------
    print("\nComputing rho_lag1 …")
    rho_lag1 = _compute_rho_lag1(sparc_coords, sparc_dir, f3_catalog)

    # ---- Assemble output ------------------------------------------------
    result = pd.DataFrame({
        "galaxy":     sparc_coords["galaxy"].values,
        "delta_mass": delta_mass.values,
        "rho_lag1":   rho_lag1.values,
    })
    before = len(result)
    result = result.dropna(subset=["delta_mass", "rho_lag1"])
    print(
        f"\nCross-match complete: {before} galaxies → {len(result)} "
        f"with both delta_mass and rho_lag1."
    )

    if len(result) == 0:
        print(
            "ERROR: No galaxies survived the crossmatch. "
            "Check that SPARC rotation curves or F3 catalog are available "
            "and that the YANG catalog covers the SPARC sky region.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Summary stats --------------------------------------------------
    from scipy.stats import spearmanr
    rho_sp, pval = spearmanr(result["delta_mass"], result["rho_lag1"])
    print(f"\n  N galaxies  : {len(result)}")
    print(f"  delta_mass  : mean={result['delta_mass'].mean():.3f}  "
          f"std={result['delta_mass'].std():.3f}")
    print(f"  rho_lag1    : mean={result['rho_lag1'].mean():.3f}  "
          f"std={result['rho_lag1'].std():.3f}")
    print(f"  Spearman ρ  : {rho_sp:.3f}  (p = {pval:.2e})")

    # ---- Save -----------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-match the YANG group catalog with the SPARC sample to "
            "compute per-galaxy delta_mass and rho_lag1. "
            "Output: results/delta_mass_yang_sparc.csv"
        )
    )
    parser.add_argument(
        "--yang", default=str(_DEFAULT_YANG),
        help=f"Path to the YANG catalog CSV (default: {_DEFAULT_YANG}).",
    )
    parser.add_argument(
        "--sparc", default=str(_DEFAULT_SPARC),
        help=f"Path to the SPARC data directory (default: {_DEFAULT_SPARC}).",
    )
    parser.add_argument(
        "--f3", default=str(_DEFAULT_F3),
        help=f"Path to F3 catalog CSV for rho_lag1 fallback (default: {_DEFAULT_F3}).",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT),
        help=f"Output CSV path (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--radius", type=float, default=_DEFAULT_RADIUS_MPC,
        help=f"Sphere radius in Mpc for delta_mass (default: {_DEFAULT_RADIUS_MPC}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    crossmatch(
        yang_path=Path(args.yang),
        sparc_dir=Path(args.sparc),
        f3_path=Path(args.f3),
        out_path=Path(args.out),
        radius_mpc=args.radius,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
