"""
scripts/build_sparc_basic.py — Build the SPARC basic summary table.

Reads the SPARC galaxy table (``data/SPARC/SPARC_Lelli2016c.csv`` or ``.mrt``)
downloaded by ``scripts/download_sparc_data.py`` and produces a clean
``data/SPARC/sparc_basic.csv`` with one row per galaxy and the columns
required by downstream scripts (notably ``crossmatch_yang_proxy.py``):

    galaxy, ra, dec, D, Inc, Vflat, e_Vflat, L36, MHI, Q

RA / Dec resolution strategy (in order of preference):
  1. Columns already present in the SPARC table
     (``RAdeg``, ``DEdeg``, ``RA_J2000``, ``DEC_J2000``, ``_RA``, ``_DE``, …).
  2. ``astroquery.simbad`` name resolution (requires ``astroquery``).
  3. ``NaN`` with a warning — the file is still written so other columns are
     available; coordinates can be added later.

Usage
-----
    python scripts/build_sparc_basic.py --data data/SPARC --out data/SPARC/sparc_basic.csv

References
----------
Lelli, McGaugh & Schombert (2016), AJ 152, 157.
http://astroweb.cwru.edu/SPARC/
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

DEFAULT_DATA = Path("data/SPARC")
DEFAULT_OUT = Path("data/SPARC/sparc_basic.csv")

# Known SPARC column order when the MRT header is not parsed
_SPARC_COLS_ORDERED = [
    "Galaxy", "T", "D", "e_D", "Inc", "e_Inc",
    "L36", "e_L36", "Re", "MHI", "Vflat", "e_Vflat",
    "Q", "Ref",
]

# Candidate column names for RA and Dec (case-insensitive lookup applied)
_RA_CANDIDATES = ["RAdeg", "RA_J2000", "_RA", "RA", "ra"]
_DEC_CANDIDATES = ["DEdeg", "DE_J2000", "_DE", "Dec", "DEC", "dec"]

# Output columns (only those that exist in the source table are included)
_KEEP_COLS = ["galaxy", "ra", "dec", "D", "e_D", "Inc", "e_Inc",
              "Vflat", "e_Vflat", "L36", "e_L36", "MHI", "Re", "Q", "T"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ci_lookup(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _load_sparc_table(data_dir: Path) -> pd.DataFrame:
    """Load SPARC galaxy table from *data_dir*."""
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        sep = "," if path.suffix == ".csv" else r"\s+"
        df = pd.read_csv(path, sep=sep, comment="#")
        # If no header was detected assign known column order
        if "Galaxy" not in df.columns and len(df.columns) >= len(_SPARC_COLS_ORDERED):
            df.columns = _SPARC_COLS_ORDERED[:len(df.columns)]
        if "Galaxy" in df.columns:
            return df
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {data_dir}.\n"
        "Run:  python scripts/download_sparc_data.py --out data/SPARC"
    )


def _resolve_coords_simbad(galaxy_names: list[str]) -> pd.DataFrame:
    """Resolve RA/Dec via astroquery.simbad.  Returns DataFrame with columns
    galaxy, ra, dec.  Missing entries get NaN.
    """
    try:
        from astroquery.simbad import Simbad  # type: ignore
    except ImportError:
        print(
            "  [coords] astroquery not installed — cannot resolve from Simbad.\n"
            "  Install with:  pip install astroquery",
            file=sys.stderr,
        )
        return pd.DataFrame({"galaxy": galaxy_names, "ra": np.nan, "dec": np.nan})

    simbad = Simbad()
    simbad.add_votable_fields("ra(d)", "dec(d)")

    batch_size = 50
    rows = []
    for i in range(0, len(galaxy_names), batch_size):
        batch = galaxy_names[i : i + batch_size]
        print(f"  [simbad] resolving {i + 1}–{min(i + batch_size, len(galaxy_names))} / {len(galaxy_names)} …")
        try:
            result = simbad.query_objects(batch)
        except Exception as exc:
            print(f"  [simbad] batch {i // batch_size} failed: {exc}", file=sys.stderr)
            for name in batch:
                rows.append({"galaxy": name, "ra": np.nan, "dec": np.nan})
            continue

        if result is None:
            for name in batch:
                rows.append({"galaxy": name, "ra": np.nan, "dec": np.nan})
            continue

        result_df = result.to_pandas()
        # Simbad returns RA_d and DEC_d (or similar) when ra(d)/dec(d) is requested
        ra_col = _ci_lookup(list(result_df.columns), ["RA_d", "RA", "ra"])
        dec_col = _ci_lookup(list(result_df.columns), ["DEC_d", "DEC", "dec"])

        for j, name in enumerate(batch):
            if j < len(result_df) and ra_col and dec_col:
                rows.append({
                    "galaxy": name,
                    "ra": float(result_df[ra_col].iloc[j]),
                    "dec": float(result_df[dec_col].iloc[j]),
                })
            else:
                rows.append({"galaxy": name, "ra": np.nan, "dec": np.nan})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_sparc_basic(
    data_dir: str | Path = DEFAULT_DATA,
    out_path: str | Path = DEFAULT_OUT,
    resolve_coords: bool = True,
) -> pd.DataFrame:
    """Build the SPARC basic summary table.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the downloaded SPARC files.
    out_path : str or Path
        Destination CSV file.
    resolve_coords : bool
        If ``True`` and RA/Dec are not in the source table, attempt to resolve
        via astroquery.simbad.

    Returns
    -------
    pd.DataFrame
        The resulting summary table (also written to *out_path*).
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SPARC table from {data_dir} …")
    df = _load_sparc_table(data_dir)
    print(f"  {len(df)} galaxies found.")

    # Normalise the Galaxy column name to lowercase 'galaxy'
    df = df.rename(columns={"Galaxy": "galaxy"})

    # ------------------------------------------------------------------
    # RA / Dec: check if already present, else resolve
    # ------------------------------------------------------------------
    ra_col = _ci_lookup(list(df.columns), _RA_CANDIDATES)
    dec_col = _ci_lookup(list(df.columns), _DEC_CANDIDATES)

    if ra_col and dec_col:
        print(f"  Using coordinate columns: {ra_col}, {dec_col}")
        df = df.rename(columns={ra_col: "ra", dec_col: "dec"})
    elif resolve_coords:
        print("  RA/Dec not found in table — attempting Simbad resolution …")
        coords = _resolve_coords_simbad(df["galaxy"].tolist())
        n_resolved = int(coords["ra"].notna().sum())
        print(f"  Resolved {n_resolved}/{len(coords)} coordinates.")
        df = df.merge(coords, on="galaxy", how="left")
    else:
        print(
            "  WARNING: RA/Dec not found and --no-resolve requested. "
            "Coordinates will be NaN.",
            file=sys.stderr,
        )
        df["ra"] = np.nan
        df["dec"] = np.nan

    # ------------------------------------------------------------------
    # Select and reorder output columns
    # ------------------------------------------------------------------
    available = [c for c in _KEEP_COLS if c in df.columns]
    out_df = df[available].copy()

    # Report coordinate coverage
    n_total = len(out_df)
    n_with_coords = int(out_df["ra"].notna().sum()) if "ra" in out_df.columns else 0
    print(f"\nSPARC basic table: {n_total} galaxies, {n_with_coords} with coordinates.")

    if n_with_coords == 0:
        print(
            "WARNING: No RA/Dec coordinates available. "
            "crossmatch_yang_proxy.py will not be able to run.\n"
            "Re-run without --no-resolve, or add RA/Dec columns to the source table.",
            file=sys.stderr,
        )

    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/SPARC/sparc_basic.csv from the downloaded SPARC galaxy table. "
            "Resolves RA/Dec via astroquery.simbad if not present in the source file."
        )
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA), metavar="DIR",
        help=f"SPARC data directory (default: {DEFAULT_DATA}).",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), metavar="FILE",
        help=f"Output CSV file (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--no-resolve", action="store_true",
        help="Skip Simbad coordinate resolution; output NaN for missing RA/Dec.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        build_sparc_basic(
            data_dir=args.data,
            out_path=args.out,
            resolve_coords=not args.no_resolve,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
