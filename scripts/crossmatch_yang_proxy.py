#!/usr/bin/env python3
"""
Cross-match SPARC galaxies with an external Yang-style environment catalog.

Inputs:
  - data/SPARC/sparc_basic.csv
      required columns: galaxy, ra, dec
  - data/environment/yang_catalog.fits or .csv
      required columns: RA, DEC
      optional columns for environment:
          DELTA_3MPC, DELTA_5MPC, delta, local_density
      fallback group columns:
          NGROUP, MGROUP, delta_log

Output:
  - results/delta_mass_yang_sparc.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u


SPARC_FILE = Path("data/SPARC/sparc_basic.csv")
YANG_FILE_CANDIDATES = [
    Path("data/environment/yang_catalog.fits"),
    Path("data/environment/yang_catalog.csv"),
    Path("data/environment/yang_group_catalog.fits"),
    Path("data/environment/yang_group_catalog.csv"),
]
OUTPUT = Path("results/delta_mass_yang_sparc.csv")
RADIUS_ARCSEC = 15.0


def detect_column_case_insensitive(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def find_existing_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".fits":
        return Table.read(path).to_pandas()
    return pd.read_csv(path)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not SPARC_FILE.exists():
        print(f"ERROR: SPARC file not found: {SPARC_FILE}")
        sys.exit(1)

    yang_file = find_existing_file(YANG_FILE_CANDIDATES)
    if yang_file is None:
        print("ERROR: No Yang catalog file found. Tried:")
        for p in YANG_FILE_CANDIDATES:
            print(" -", p)
        sys.exit(1)

    sparc = pd.read_csv(SPARC_FILE)
    required = ["galaxy", "ra", "dec"]
    missing = [c for c in required if c not in sparc.columns]
    if missing:
        print(f"ERROR: Missing SPARC columns: {missing}")
        sys.exit(1)

    yang = load_catalog(yang_file)

    ra_col = detect_column_case_insensitive(yang.columns, ["RA", "ra"])
    dec_col = detect_column_case_insensitive(yang.columns, ["DEC", "dec"])
    if ra_col is None or dec_col is None:
        print("ERROR: Yang catalog must contain RA and DEC columns.")
        print("Available columns:", list(yang.columns))
        sys.exit(1)

    c_sparc = SkyCoord(ra=sparc["ra"].values * u.deg, dec=sparc["dec"].values * u.deg)
    c_yang = SkyCoord(ra=yang[ra_col].values * u.deg, dec=yang[dec_col].values * u.deg)

    idx, sep, _ = c_sparc.match_to_catalog_sky(c_yang, nthneighbor=1)
    matched_mask = sep < RADIUS_ARCSEC * u.arcsec

    result = sparc.loc[matched_mask].copy().reset_index(drop=True)
    result["yang_idx"] = idx[matched_mask].astype(int)
    result["match_sep_arcsec"] = sep[matched_mask].arcsec

    n_matched = len(result)
    print(f"N galaxies  : {n_matched}")
    if n_matched == 0:
        print("ERROR: No matches within search radius.")
        sys.exit(1)

    density_candidates = ["DELTA_3MPC", "DELTA_5MPC", "delta", "local_density"]
    group_candidates = ["NGROUP", "MGROUP", "delta_log"]

    density_col = detect_column_case_insensitive(yang.columns, density_candidates)
    group_col = detect_column_case_insensitive(yang.columns, group_candidates)

    if density_col is not None:
        proxy_col = density_col
        proxy_mode = "density"
    elif group_col is not None:
        proxy_col = group_col
        proxy_mode = "group_proxy"
    else:
        print("ERROR: No suitable environmental proxy column found.")
        print("Available columns:", list(yang.columns))
        sys.exit(1)

    result["raw_proxy"] = yang.iloc[result["yang_idx"].values][proxy_col].values

    if proxy_mode == "density":
        result["delta_mass"] = result["raw_proxy"]
    else:
        median_val = np.nanmedian(result["raw_proxy"])
        if not np.isfinite(median_val) or median_val <= 0:
            print(f"ERROR: Cannot normalize group proxy; invalid median: {median_val!r}")
            sys.exit(1)
        result["delta_mass"] = (result["raw_proxy"] / median_val) - 1.0

    # Placeholder rho_lag1 for now: NaN unless already present in SPARC basic.
    if "rho_lag1" in sparc.columns:
        result["rho_lag1"] = sparc.loc[matched_mask, "rho_lag1"].values
        n_rot = int(result["rho_lag1"].notna().sum())
        n_beta = 0
    else:
        result["rho_lag1"] = np.nan
        n_rot = 0
        n_beta = 0

    delta_mean = float(np.nanmean(result["delta_mass"]))
    delta_std = float(np.nanstd(result["delta_mass"]))
    rho_mean = float(np.nanmean(result["rho_lag1"])) if result["rho_lag1"].notna().any() else np.nan
    rho_std = float(np.nanstd(result["rho_lag1"])) if result["rho_lag1"].notna().any() else np.nan

    valid = result[["delta_mass", "rho_lag1"]].dropna()
    if len(valid) >= 3:
        x_rank = valid["delta_mass"].rank().values
        y_rank = valid["rho_lag1"].rank().values
        spearman_rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
    else:
        spearman_rho = np.nan

    print(f"delta_mass  : mean={delta_mean:.6g}  std={delta_std:.6g}")
    print(f"rho_lag1    : mean={rho_mean:.6g}  std={rho_std:.6g}")
    print(f"Spearman ρ  : {spearman_rho}")
    print(f"rho_lag1 sources — rotation curves: {n_rot}, F3 beta proxy: {n_beta}, missing: {len(result) - n_rot - n_beta}")

    output = result[["galaxy", "delta_mass", "rho_lag1", "match_sep_arcsec"]].copy()
    output["proxy_source"] = proxy_col
    output["proxy_mode"] = proxy_mode
    output["match_radius_arcsec"] = RADIUS_ARCSEC
    output.to_csv(OUTPUT, index=False)

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
