#!/usr/bin/env python3
"""
Cross-match SPARC galaxies with the Yang et al. (SDSS) group catalog to obtain
an alternative environmental proxy.

Important:
- This is NOT the canonical delta_mass used in the paper.
- It provides an external robustness proxy derived from the Yang group catalog.
- If a density-like column exists (e.g. DELTA_3MPC), it is used directly.
- Otherwise, a pseudo-overdensity proxy is derived from NGROUP or MGROUP.

Inputs:
    data/SPARC/sparc_basic.csv
    data/environment/yang_group_catalog.fits  (or .csv)

Outputs:
    results/delta_mass_yang_sparc.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

# ===================== CONFIG =====================
SPARC_FILE = Path("data/SPARC/sparc_basic.csv")
YANG_FILE = Path("data/environment/yang_group_catalog.fits")
OUTPUT_CSV = Path("results/delta_mass_yang_sparc.csv")

RADIUS_ARCSEC = 15.0
# ==================================================


def detect_column_case_insensitive(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load SPARC
    try:
        sparc = pd.read_csv(SPARC_FILE)
    except FileNotFoundError:
        print(f"ERROR: SPARC file not found: {SPARC_FILE}")
        sys.exit(1)

    required_sparc = ["galaxy", "ra", "dec"]
    missing = [c for c in required_sparc if c not in sparc.columns]
    if missing:
        print(f"ERROR: SPARC file missing required columns: {missing}")
        sys.exit(1)

    print(f"Loaded {len(sparc)} SPARC galaxies.")

    # Load Yang catalog
    try:
        if YANG_FILE.suffix.lower() == ".fits":
            yang_tab = Table.read(YANG_FILE)
            yang = yang_tab.to_pandas()
        else:
            yang = pd.read_csv(YANG_FILE)
    except Exception as e:
        print(f"ERROR: Cannot read Yang catalog {YANG_FILE}: {e}")
        sys.exit(1)

    print(f"Loaded {len(yang)} entries from Yang catalog.")

    # Detect coordinate columns
    ra_col = detect_column_case_insensitive(yang.columns, ["RA"])
    dec_col = detect_column_case_insensitive(yang.columns, ["DEC"])

    if ra_col is None or dec_col is None:
        print("ERROR: Yang catalog must contain RA and DEC columns.")
        print("Available columns:", list(yang.columns))
        sys.exit(1)

    # Build coordinates
    c_sparc = SkyCoord(ra=sparc["ra"].values * u.deg, dec=sparc["dec"].values * u.deg)
    c_yang = SkyCoord(ra=yang[ra_col].values * u.deg, dec=yang[dec_col].values * u.deg)

    # Match
    idx, sep, _ = c_sparc.match_to_catalog_sky(c_yang, nthneighbor=1)
    matched_mask = sep < RADIUS_ARCSEC * u.arcsec

    result = sparc.loc[matched_mask].copy()
    result["yang_idx"] = idx[matched_mask]
    result["match_sep_arcsec"] = sep[matched_mask].arcsec

    n_total = len(sparc)
    n_matched = len(result)
    frac = n_matched / n_total if n_total > 0 else 0.0
    print(f"Matched {n_matched}/{n_total} galaxies ({frac:.1%}) within {RADIUS_ARCSEC:.1f} arcsec.")
    if n_matched > 0:
        print(f"Median match separation: {np.median(result['match_sep_arcsec']):.2f} arcsec")

    # Detect environmental proxy column
    density_candidates = ["DELTA_3MPC", "DELTA_5MPC", "delta", "local_density"]
    group_candidates = ["NGROUP", "MGROUP"]

    density_col = detect_column_case_insensitive(yang.columns, density_candidates)
    group_col = detect_column_case_insensitive(yang.columns, group_candidates)

    proxy_col = None
    proxy_mode = None

    if density_col is not None:
        proxy_col = density_col
        proxy_mode = "density"
        print(f"Using density-like column from Yang catalog: {proxy_col}")
    elif group_col is not None:
        proxy_col = group_col
        proxy_mode = "group_proxy"
        print(f"WARNING: No density-like column found. Using {proxy_col} as exploratory proxy.")
    else:
        print("ERROR: No suitable environmental proxy column found.")
        print("Available columns:", list(yang.columns))
        sys.exit(1)

    # Extract matched values (vectorized)
    result["raw_proxy"] = yang.iloc[result["yang_idx"].values][proxy_col].values

    # Build final proxy
    if proxy_mode == "density":
        result["delta_mass_yang"] = result["raw_proxy"]
    else:
        median_val = np.nanmedian(result["raw_proxy"])
        if not np.isfinite(median_val) or median_val == 0:
            print("ERROR: Cannot normalize group proxy; invalid median value.")
            sys.exit(1)
        result["delta_mass_yang"] = (result["raw_proxy"] / median_val) - 1.0
        print(f"Converted {proxy_col} to pseudo-overdensity using matched-sample median = {median_val:.4g}")

    output = result[["galaxy", "delta_mass_yang", "raw_proxy", "match_sep_arcsec"]].copy()
    output = output.rename(columns={"raw_proxy": proxy_col})
    output["proxy_source"] = proxy_col
    output["proxy_mode"] = proxy_mode

    output.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(output)} matched galaxies to {OUTPUT_CSV}")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
