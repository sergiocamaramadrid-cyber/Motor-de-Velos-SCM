#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_galaxy_catalog.py — Build the unified SPARC galaxy catalog.

Joins three real data sources:

1. **F3 catalog** (``--f3-catalog``) — per-galaxy β fits.
   Required columns: ``galaxy``, ``friction_slope`` or ``beta``, ``n_deep``.
2. **SPARC global table** (``--sparc-table``) — galaxy-level properties.
   Expected columns: ``Galaxy``, ``Inc``, ``L36``, ``MHI``, ``Re``.
3. **Environmental proxy** (``--env-catalog``) — Yang-group δ_mass proxy.
   Required columns: ``galaxy``, ``delta_mass_std`` (or ``delta_mass``).

Optionally, if a contract table is available (``--contract``) it is used to
compute ``Rmax`` as ``max(r_kpc)`` per galaxy; otherwise ``Re`` from the SPARC
global table is used as a fallback.

Output columns
--------------
galaxy_id, slope_tail, n_tail_points, inc_deg, logM, Rmax, env_proxy

Where:

* ``galaxy_id``     — galaxy name (join key from F3 catalog)
* ``slope_tail``    — friction slope β from F3 catalog
* ``n_tail_points`` — number of deep-regime points used to fit β
* ``inc_deg``       — inclination in degrees from SPARC global table
* ``logM``          — log10(M_bar / Msun) derived from L36 + MHI, or from
                      a ``log_M_bar`` / ``logMbar`` column if already present
* ``Rmax``          — maximum observed radius in kpc (from contract or Re)
* ``env_proxy``     — environmental density proxy δ_mass (Yang group catalog)

Missing values are left as ``NaN`` — nothing is invented.

Outputs
-------
data/galaxy_catalog.csv
    The unified per-galaxy catalog.
results/galaxy_catalog_build_summary.txt
    Completeness report (total galaxies, coverage per column, complete rows).

Usage
-----
::

    python scripts/build_galaxy_catalog.py \\
        --f3-catalog   results/f3_catalog_real.csv \\
        --sparc-table  data/SPARC/SPARC_Lelli2016c.csv \\
        --env-catalog  results/delta_mass_yang_sparc.csv

    python scripts/build_galaxy_catalog.py \\
        --f3-catalog   results/f3_catalog_real.csv \\
        --sparc-table  data/SPARC/SPARC_Lelli2016c.csv \\
        --env-catalog  results/delta_mass_yang_sparc.csv \\
        --contract     data/SPARC/sparc_contract.parquet \\
        --out-catalog  data/galaxy_catalog.csv \\
        --out-summary  results/galaxy_catalog_build_summary.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default mass-to-light ratio at 3.6 μm (McGaugh & Schombert 2015)
UPSILON_36: float = 0.5  # Msun / Lsun

# Helium correction factor for total gas mass
HE_CORRECTION: float = 1.33

# Units: L36 in 1e9 Lsun, MHI in 1e9 Msun → Msun after scaling
L36_UNIT: float = 1.0e9  # Lsun
MHI_UNIT: float = 1.0e9  # Msun

# Output columns in the required order
OUTPUT_COLS: list[str] = [
    "galaxy_id",
    "slope_tail",
    "n_tail_points",
    "inc_deg",
    "logM",
    "Rmax",
    "env_proxy",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_optional(path: Path | None, label: str) -> pd.DataFrame | None:
    """Read a CSV or Parquet file; return None if the file does not exist."""
    if path is None:
        print(f"  [skip] {label}: path not provided", file=sys.stderr)
        return None
    path = Path(path)
    if not path.exists():
        print(f"  [warn] {label}: file not found — {path}", file=sys.stderr)
        return None
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    print(f"  [ok]   {label}: {len(df)} rows from {path}")
    return df


# ---------------------------------------------------------------------------
# Column-name resolution helpers
# ---------------------------------------------------------------------------

def _resolve_beta(df: pd.DataFrame) -> pd.Series:
    """Return the β / friction_slope column, preferring 'friction_slope'."""
    for col in ("friction_slope", "beta"):
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def _resolve_n_deep(df: pd.DataFrame) -> pd.Series:
    """Return the n_deep / n_tail_points column."""
    for col in ("n_deep", "n_tail_points", "n_deep_points"):
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def _resolve_galaxy_key(df: pd.DataFrame) -> pd.Series:
    """Return the galaxy name column (handles 'galaxy' and 'Galaxy')."""
    for col in ("galaxy", "Galaxy", "name", "galname"):
        if col in df.columns:
            return df[col].astype(str).str.strip()
    raise ValueError(f"No galaxy-name column found in DataFrame. Columns: {list(df.columns)}")


def _resolve_env_proxy(df: pd.DataFrame) -> pd.Series:
    """Return the environmental proxy column."""
    for col in ("delta_mass_std", "delta_mass", "delta_mass_proxy", "env_proxy"):
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def _compute_logm(df_sparc: pd.DataFrame) -> pd.Series:
    """
    Compute log10(M_bar / Msun) from the SPARC global table.

    Priority:
    1. ``log_M_bar`` or ``logMbar`` if already present.
    2. Derived from ``L36`` (3.6 μm luminosity in 1e9 Lsun) and ``MHI``
       (HI mass in 1e9 Msun):
       M_bar = Upsilon_36 * L36 * 1e9  +  1.33 * MHI * 1e9  [Msun]
    """
    for col in ("log_M_bar", "logMbar", "logM_baryon", "logM"):
        if col in df_sparc.columns:
            return df_sparc[col]

    if "L36" in df_sparc.columns and "MHI" in df_sparc.columns:
        L36 = pd.to_numeric(df_sparc["L36"], errors="coerce").fillna(0.0)
        MHI = pd.to_numeric(df_sparc["MHI"], errors="coerce").fillna(0.0)
        M_bar = UPSILON_36 * L36 * L36_UNIT + HE_CORRECTION * MHI * MHI_UNIT
        logM = np.where(M_bar > 0, np.log10(M_bar), np.nan)
        return pd.Series(logM, index=df_sparc.index)

    return pd.Series(np.nan, index=df_sparc.index)


def _compute_rmax_from_contract(df_contract: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns ['galaxy', 'Rmax'] from contract table."""
    gal_col = _resolve_galaxy_key(df_contract)
    tmp = df_contract.copy()
    tmp["_galaxy"] = gal_col
    r_col = None
    for col in ("r_kpc", "r", "R_kpc"):
        if col in tmp.columns:
            r_col = col
            break
    if r_col is None:
        return pd.DataFrame(columns=["galaxy", "Rmax"])
    rmax = (
        tmp.groupby("_galaxy")[r_col]
        .max()
        .reset_index()
        .rename(columns={"_galaxy": "galaxy", r_col: "Rmax"})
    )
    return rmax


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def build_galaxy_catalog(
    f3_path: str | Path | None,
    sparc_path: str | Path | None,
    env_path: str | Path | None,
    contract_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the unified galaxy catalog.

    Parameters
    ----------
    f3_path : path-like or None
        F3 catalog CSV/Parquet.
    sparc_path : path-like or None
        SPARC global table CSV.
    env_path : path-like or None
        Yang environmental proxy CSV.
    contract_path : path-like or None
        Contract table for Rmax computation (CSV or Parquet).

    Returns
    -------
    pd.DataFrame
        Catalog with columns ``galaxy_id, slope_tail, n_tail_points,
        inc_deg, logM, Rmax, env_proxy``.  Missing values are NaN.
    """
    # ------------------------------------------------------------------
    # 1. F3 catalog — mandatory anchor
    # ------------------------------------------------------------------
    df_f3 = _read_optional(Path(f3_path) if f3_path else None, "F3 catalog")
    if df_f3 is None or df_f3.empty:
        raise ValueError(
            "F3 catalog is required and could not be loaded.  "
            "Provide a valid --f3-catalog path."
        )

    galaxy_ids = _resolve_galaxy_key(df_f3)
    catalog = pd.DataFrame({
        "galaxy_id": galaxy_ids.values,
        "slope_tail": _resolve_beta(df_f3).values,
        "n_tail_points": _resolve_n_deep(df_f3).values,
    })
    catalog["galaxy_id"] = catalog["galaxy_id"].astype(str)

    # ------------------------------------------------------------------
    # 2. SPARC global table — inc_deg, logM, optional Rmax fallback
    # ------------------------------------------------------------------
    df_sparc = _read_optional(Path(sparc_path) if sparc_path else None, "SPARC table")
    sparc_cols: dict[str, pd.Series] = {}
    if df_sparc is not None and not df_sparc.empty:
        sparc_gal = _resolve_galaxy_key(df_sparc)
        sparc_tmp = df_sparc.copy()
        sparc_tmp["_galaxy_id"] = sparc_gal.values

        inc_col = next((c for c in ("Inc", "inc", "Inc_deg", "inclination") if c in sparc_tmp.columns), None)
        sparc_tmp["_inc_deg"] = pd.to_numeric(sparc_tmp[inc_col], errors="coerce") if inc_col else np.nan

        sparc_tmp["_logM"] = _compute_logm(df_sparc).values

        # Re as Rmax fallback (effective radius in kpc)
        re_col = next((c for c in ("Re", "re", "r_eff") if c in sparc_tmp.columns), None)
        sparc_tmp["_Re"] = pd.to_numeric(sparc_tmp[re_col], errors="coerce") if re_col else np.nan

        sparc_agg = sparc_tmp.groupby("_galaxy_id").first().reset_index()

        catalog = catalog.merge(
            sparc_agg[["_galaxy_id", "_inc_deg", "_logM", "_Re"]].rename(
                columns={"_galaxy_id": "galaxy_id"}
            ),
            on="galaxy_id",
            how="left",
        )
        catalog.rename(columns={"_inc_deg": "inc_deg", "_logM": "logM", "_Re": "_Re_fallback"}, inplace=True)
    else:
        catalog["inc_deg"] = np.nan
        catalog["logM"] = np.nan
        catalog["_Re_fallback"] = np.nan

    # ------------------------------------------------------------------
    # 3. Contract table — Rmax = max(r_kpc) per galaxy
    # ------------------------------------------------------------------
    df_contract = _read_optional(Path(contract_path) if contract_path else None, "contract table")
    if df_contract is not None and not df_contract.empty:
        rmax_df = _compute_rmax_from_contract(df_contract)
        rmax_df = rmax_df.rename(columns={"galaxy": "galaxy_id"})
        catalog = catalog.merge(rmax_df, on="galaxy_id", how="left")
    else:
        catalog["Rmax"] = np.nan

    # Fill Rmax from Re if still missing
    re_avail = "_Re_fallback" in catalog.columns
    if re_avail:
        mask = catalog["Rmax"].isna()
        catalog.loc[mask, "Rmax"] = catalog.loc[mask, "_Re_fallback"]

    # ------------------------------------------------------------------
    # 4. Environmental proxy
    # ------------------------------------------------------------------
    df_env = _read_optional(Path(env_path) if env_path else None, "env proxy")
    if df_env is not None and not df_env.empty:
        env_gal = _resolve_galaxy_key(df_env)
        df_env = df_env.copy()
        df_env["_galaxy_id"] = env_gal.values
        df_env["_env_proxy"] = _resolve_env_proxy(df_env).values
        env_agg = df_env.groupby("_galaxy_id")["_env_proxy"].first().reset_index()
        env_agg.rename(columns={"_galaxy_id": "galaxy_id", "_env_proxy": "env_proxy"}, inplace=True)
        catalog = catalog.merge(env_agg, on="galaxy_id", how="left")
    else:
        catalog["env_proxy"] = np.nan

    # ------------------------------------------------------------------
    # 5. Drop internal helper columns and enforce output schema
    # ------------------------------------------------------------------
    drop_cols = [c for c in catalog.columns if c.startswith("_")]
    catalog.drop(columns=drop_cols, inplace=True)

    for col in OUTPUT_COLS:
        if col not in catalog.columns:
            catalog[col] = np.nan

    catalog = catalog[OUTPUT_COLS].copy()
    catalog.sort_values("galaxy_id").reset_index(drop=True, inplace=True)
    return catalog


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(catalog: pd.DataFrame, out_path: Path) -> str:
    """Write a completeness summary and return it as a string."""
    n_total = len(catalog)
    lines = [
        "galaxy_catalog_build_summary",
        "=" * 40,
        f"Total galaxies : {n_total}",
        "",
        "Column completeness (non-NaN rows):",
    ]
    for col in OUTPUT_COLS:
        n_ok = int(catalog[col].notna().sum())
        pct = 100 * n_ok / n_total if n_total > 0 else 0.0
        lines.append(f"  {col:<18s} {n_ok:4d} / {n_total}  ({pct:.1f}%)")

    n_complete = int(catalog[OUTPUT_COLS].notna().all(axis=1).sum())
    pct_complete = 100 * n_complete / n_total if n_total > 0 else 0.0
    lines += [
        "",
        f"Rows with ALL columns complete : {n_complete} / {n_total}  ({pct_complete:.1f}%)",
    ]

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/galaxy_catalog.csv from real SPARC data sources.\n\n"
            "Joins F3 catalog + SPARC global table + Yang env proxy.\n"
            "Missing values are left as NaN — nothing is invented."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--f3-catalog",
        default="results/f3_catalog_real.csv",
        metavar="FILE",
        help="F3 per-galaxy catalog CSV.  Columns: galaxy, friction_slope/beta, n_deep.  "
             "(default: results/f3_catalog_real.csv)",
    )
    parser.add_argument(
        "--sparc-table",
        default="data/SPARC/SPARC_Lelli2016c.csv",
        metavar="FILE",
        help="SPARC global galaxy table CSV.  Columns: Galaxy, Inc, L36, MHI, Re.  "
             "(default: data/SPARC/SPARC_Lelli2016c.csv)",
    )
    parser.add_argument(
        "--env-catalog",
        default="results/delta_mass_yang_sparc.csv",
        metavar="FILE",
        help="Yang environmental proxy CSV.  Columns: galaxy, delta_mass_std.  "
             "(default: results/delta_mass_yang_sparc.csv)",
    )
    parser.add_argument(
        "--contract",
        default=None,
        metavar="FILE",
        help="Contract table (CSV or Parquet) with r_kpc per galaxy for computing Rmax.  "
             "Optional; Re from SPARC table is used as fallback.",
    )
    parser.add_argument(
        "--out-catalog",
        default="data/galaxy_catalog.csv",
        metavar="FILE",
        help="Output CSV path.  (default: data/galaxy_catalog.csv)",
    )
    parser.add_argument(
        "--out-summary",
        default="results/galaxy_catalog_build_summary.txt",
        metavar="FILE",
        help="Output completeness summary path.  "
             "(default: results/galaxy_catalog_build_summary.txt)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print("Building galaxy catalog...")
    catalog = build_galaxy_catalog(
        f3_path=args.f3_catalog,
        sparc_path=args.sparc_table,
        env_path=args.env_catalog,
        contract_path=args.contract,
    )

    out_catalog = Path(args.out_catalog)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_catalog, index=False)
    print(f"\nCatalog written: {out_catalog}  ({len(catalog)} galaxies)")

    summary_text = write_summary(catalog, Path(args.out_summary))
    print(f"Summary written: {args.out_summary}\n")
    print(summary_text)


if __name__ == "__main__":
    main()
