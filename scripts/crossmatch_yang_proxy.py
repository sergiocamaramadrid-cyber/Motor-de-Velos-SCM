#!/usr/bin/env python3
"""
Cross-match SPARC galaxies with the Yang et al. (SDSS) group catalog
to obtain an alternative environmental proxy (delta_mass_yang).

IMPORTANT:
- This is NOT the canonical delta_mass used in the paper.
- It is an exploratory robustness proxy only.
- If a density-like column exists, it is used directly.
- Otherwise, a pseudo-overdensity is derived from group multiplicity.

USAGE:
    Ensure SPARC catalog with columns 'galaxy', 'ra', 'dec' exists at the path below.
    Ensure Yang catalog (FITS or CSV) with RA, DEC and density/group columns exists.
    Modify paths and RADIUS_ARCSEC if needed.
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
# Modify these paths to match your repository structure
SPARC_FILE = Path("data/SPARC/sparc_basic.csv")                 # must have 'galaxy', 'ra', 'dec'
YANG_FILE = Path("data/environment/yang_group_catalog.fits")    # adjust if .csv
OUTPUT = Path("results/delta_mass_yang_sparc.csv")

RADIUS_ARCSEC = 15.0  # search radius (arcseconds)
# =================================================


def detect_column_case_insensitive(columns, candidates):
    """Return the first candidate name found in columns (case-insensitive).

    Parameters
    ----------
    columns : iterable of str
        Available column names to search through.
    candidates : iterable of str
        Candidate names to look for, tried in order.

    Returns
    -------
    str or None
        The matching column name as it appears in *columns*, or None if not found.
    """
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Load SPARC
    if not SPARC_FILE.exists():
        print(f"ERROR: SPARC file not found: {SPARC_FILE}")
        print("Please generate the SPARC basic catalog with ra/dec first.")
        sys.exit(1)
    sparc = pd.read_csv(SPARC_FILE)

    required = ["galaxy", "ra", "dec"]
    missing = [c for c in required if c not in sparc.columns]
    if missing:
        print(f"ERROR: Missing columns in SPARC: {missing}")
        sys.exit(1)

    print(f"Loaded {len(sparc)} SPARC galaxies.")

    # Load Yang catalog
    if not YANG_FILE.exists():
        print(f"ERROR: Yang catalog file not found: {YANG_FILE}")
        sys.exit(1)

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

    # Cross-match
    c_sparc = SkyCoord(ra=sparc["ra"].values * u.deg, dec=sparc["dec"].values * u.deg)
    ra_col = detect_column_case_insensitive(yang.columns, ["RA", "ra"])
    dec_col = detect_column_case_insensitive(yang.columns, ["DEC", "dec"])

    if ra_col is None or dec_col is None:
        print("ERROR: Yang catalog must contain RA and DEC columns.")
        print("Available columns:", list(yang.columns))
        sys.exit(1)

    c_yang = SkyCoord(ra=yang[ra_col].values * u.deg, dec=yang[dec_col].values * u.deg)

    idx, sep, _ = c_sparc.match_to_catalog_sky(c_yang, nthneighbor=1)
    matched_mask = sep < RADIUS_ARCSEC * u.arcsec

    result = sparc[matched_mask].copy().reset_index(drop=True)
    result["yang_idx"] = idx[matched_mask].astype(int)
    result["match_sep_arcsec"] = sep[matched_mask].arcsec

    n_matched = len(result)
    print(f"Matched {n_matched}/{len(sparc)} galaxies ({n_matched/len(sparc)*100:.1f}%)")
    if n_matched > 0:
        print(f"Median match separation: {np.median(result['match_sep_arcsec']):.2f} arcsec")
    if n_matched == 0:
        print("ERROR: No SPARC galaxies matched the Yang catalog within the configured radius.")
        sys.exit(1)

    # Detect density/proxy column
    density_candidates = ["DELTA_3MPC", "DELTA_5MPC", "delta", "local_density"]
    group_candidates = ["NGROUP", "MGROUP", "Ngroup", "Mgroup", "delta_log"]

    density_col = detect_column_case_insensitive(yang.columns, density_candidates)
    group_col = detect_column_case_insensitive(yang.columns, group_candidates)

    if density_col is not None:
        proxy_col = density_col
        proxy_mode = "density"
        print(f"Using direct density column: {proxy_col}")
    elif group_col is not None:
        proxy_col = group_col
        proxy_mode = "group_proxy"
        print(f"WARNING: No density column found. Using {proxy_col} as exploratory proxy.")
    else:
        print("ERROR: No suitable proxy column found in Yang catalog.")
        print("Available columns:", list(yang.columns))
        sys.exit(1)

    # Extract values
    result["raw_proxy"] = yang.iloc[result["yang_idx"].values][proxy_col].values

    if proxy_mode == "density":
        result["delta_mass_yang"] = result["raw_proxy"]
    else:
        median_val = np.nanmedian(result["raw_proxy"])
        if not np.isfinite(median_val) or median_val <= 0:
            reason = "non-finite" if not np.isfinite(median_val) else ("zero" if median_val == 0 else "negative")
            print(f"ERROR: Cannot normalize group proxy (median is {reason}: {median_val!r}).")
            sys.exit(1)
        result["delta_mass_yang"] = (result["raw_proxy"] / median_val) - 1.0
        print(f"Converted to pseudo-overdensity using median = {median_val:.4g}")

    # Final output
    output = result[["galaxy", "delta_mass_yang", "raw_proxy", "match_sep_arcsec"]].copy()
    output = output.rename(columns={"raw_proxy": proxy_col})
    output["proxy_source"] = proxy_col
    output["proxy_mode"] = proxy_mode
    output["match_radius_arcsec"] = RADIUS_ARCSEC

    output.to_csv(OUTPUT, index=False)
    print(f"\n✅ Saved {len(output)} matched galaxies to {OUTPUT}")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
