"""
scripts/download_sparc_data.py — Download the SPARC dataset from the public archive.

Downloads from the official SPARC website (astroweb.cwru.edu):
  - SPARC_Lelli2016c.mrt  (galaxy table, ~20 KB)
  - <Galaxy>_rotmod.dat   (per-galaxy rotation curves, ~175 files, ~a few KB each)

The rotation curves are placed in ``<out>/raw/`` so they are found by
:func:`src.scm_analysis.load_rotation_curve`.  The galaxy table is saved
as ``<out>/SPARC_Lelli2016c.csv`` (CSV format) so it is found by
:func:`src.scm_analysis.load_galaxy_table`.

Usage
-----
    python scripts/download_sparc_data.py --out data/SPARC

References
----------
Lelli, McGaugh & Schombert (2016), AJ 152, 157.
http://astroweb.cwru.edu/SPARC/
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import socket
from pathlib import Path

import pandas as pd

SPARC_BASE = "https://astroweb.cwru.edu/SPARC"
TABLE_MRT_URL = f"{SPARC_BASE}/SPARC_Lelli2016c.mrt"
ROTMOD_URL = f"{SPARC_BASE}/Rotmod_LTG/{{galaxy}}_rotmod.dat"

_TIMEOUT = 30   # seconds per request
_RETRY_DELAY = 2  # seconds between retries

# Apply a global socket timeout so urllib.request.urlretrieve never hangs
socket.setdefaulttimeout(_TIMEOUT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download *url* to *dest*; return True on success."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except (urllib.error.URLError, OSError) as exc:
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                print(f"  [fail] {url}: {exc}", file=sys.stderr)
    return False


def _parse_galaxy_table(mrt_path: Path) -> pd.DataFrame:
    """Parse the SPARC galaxy table MRT file into a DataFrame.

    Tries ``pd.read_csv`` with ``comment='#'`` first (works when the MRT
    header lines are all prefixed with ``#``).  Falls back to a line-scan
    that locates the first non-header data line.

    Returns a DataFrame with at least a ``Galaxy`` column.
    """
    # Attempt 1: standard comment-skipping CSV parse
    try:
        df = pd.read_csv(mrt_path, sep=r"\s+", comment="#")
        if "Galaxy" in df.columns and len(df) > 5:
            return df
    except Exception:
        pass

    # Attempt 2: skip CDS MRT boilerplate and read fixed-width data
    lines = mrt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = []
    header_found = False
    column_names: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Skip empty lines and common CDS header markers
        if not stripped:
            continue
        if stripped.startswith(("=", "-", "J/", "B", "Title", "Authors",
                                 "Table", "Byte", "Note", "Ref", "Ack",
                                 "Description", "ADC")):
            continue
        if stripped.startswith("#"):
            # Try to extract column names from a # header line
            if not header_found:
                parts = stripped.lstrip("#").split()
                if parts and parts[0] in ("Galaxy", "Name"):
                    column_names = parts
                    header_found = True
            continue
        # Looks like data
        if stripped:
            data_lines.append(stripped.split())

    if data_lines:
        if column_names:
            # Only keep rows that have all required columns, then truncate
            ncols = len(column_names)
            rows = [r[:ncols] for r in data_lines if len(r) >= ncols]
            df = pd.DataFrame(rows, columns=column_names[:ncols])
        else:
            df = pd.DataFrame(data_lines)
            if len(df.columns) >= 12:
                # Known SPARC column order: Galaxy T D e_D Inc e_Inc L36 e_L36
                #                           Re MHI Vflat e_Vflat Q Ref
                col_names = [
                    "Galaxy", "T", "D", "e_D", "Inc", "e_Inc",
                    "L36", "e_L36", "Re", "MHI", "Vflat", "e_Vflat",
                    "Q", "Ref",
                ]
                df.columns = col_names[:len(df.columns)]
        if "Galaxy" in df.columns:
            return df

    raise ValueError(
        f"Could not parse galaxy table from {mrt_path}. "
        "Check that the file is a valid SPARC_Lelli2016c.mrt."
    )


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_sparc(out_dir: str | Path) -> bool:
    """Download SPARC data to *out_dir*.

    Parameters
    ----------
    out_dir : str or Path
        Destination directory.  Created if it does not exist.
        Rotation curves go into ``<out_dir>/raw/``.
        The galaxy table is saved as ``<out_dir>/SPARC_Lelli2016c.csv``.

    Returns
    -------
    bool
        ``True`` if all downloads succeeded, ``False`` otherwise.
    """
    out = Path(out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Galaxy table
    # ------------------------------------------------------------------
    mrt_path = out / "SPARC_Lelli2016c.mrt"
    csv_path = out / "SPARC_Lelli2016c.csv"

    if csv_path.exists():
        print(f"Galaxy table already present: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"Downloading galaxy table → {mrt_path} …")
        if not _download_file(TABLE_MRT_URL, mrt_path):
            print("ERROR: Could not download the SPARC galaxy table.", file=sys.stderr)
            return False

        print("Parsing galaxy table …")
        try:
            df = _parse_galaxy_table(mrt_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return False

        # Ensure required columns exist with canonical names
        if "Galaxy" not in df.columns:
            print("ERROR: 'Galaxy' column not found in table.", file=sys.stderr)
            return False

        # Save as CSV so load_galaxy_table() picks it up with sep=","
        df.to_csv(csv_path, index=False)
        print(f"Galaxy table saved as {csv_path}  ({len(df)} galaxies)")

    galaxy_names: list[str] = df["Galaxy"].dropna().tolist()
    print(f"Galaxies in table: {len(galaxy_names)}")

    # ------------------------------------------------------------------
    # 2. Rotation curves
    # ------------------------------------------------------------------
    print(f"\nDownloading rotation curves → {raw} …")
    ok = 0
    fail = 0
    skipped = 0

    for name in galaxy_names:
        dest = raw / f"{name}_rotmod.dat"
        if dest.exists():
            skipped += 1
            ok += 1
            continue
        url = ROTMOD_URL.format(galaxy=name)
        if _download_file(url, dest, retries=3):
            ok += 1
            print(f"  [ok]   {name}")
        else:
            fail += 1
        time.sleep(0.05)  # polite rate limiting

    total = len(galaxy_names)
    print(
        f"\nRotation curves: {ok - skipped} downloaded, {skipped} already present, "
        f"{fail} failed  (total {total})"
    )

    if fail > 0:
        print(
            f"WARNING: {fail} rotation curve(s) could not be downloaded. "
            "Those galaxies will be skipped by generate_f3_catalog.py.",
            file=sys.stderr,
        )

    return True  # partial download still allows the pipeline to run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the SPARC dataset (Lelli+2016) from astroweb.cwru.edu. "
            "Downloads the galaxy table and per-galaxy rotation curves."
        )
    )
    parser.add_argument(
        "--out", default="data/SPARC",
        help="Destination directory (default: data/SPARC).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ok = download_sparc(args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
