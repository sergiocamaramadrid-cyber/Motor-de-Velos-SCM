"""
scripts/download_sparc_data.py — Download the SPARC dataset from the public archive.

Downloads from the official SPARC website (astroweb.cwru.edu) with automatic
fallback to the Zenodo long-term archive (DOI 10.5281/zenodo.16284118):

  - SPARC_Lelli2016c.mrt  (galaxy table, ~20 KB)
  - <Galaxy>_rotmod.dat   (per-galaxy rotation curves, ~175 files, ~a few KB each)
  - Rotmod_LTG.zip        (Zenodo fallback — full rotation-curve bundle)

The rotation curves are placed in ``<out>/raw/`` so they are found by
:func:`src.scm_analysis.load_rotation_curve`.  The galaxy table is saved
as ``<out>/SPARC_Lelli2016c.csv`` (CSV format) so it is found by
:func:`src.scm_analysis.load_galaxy_table`.

Usage
-----
    python scripts/download_sparc_data.py --out data/SPARC

    # Force Zenodo source even when CWRU is reachable:
    python scripts/download_sparc_data.py --out data/SPARC --source zenodo

References
----------
Lelli, McGaugh & Schombert (2016), AJ 152, 157.
http://astroweb.cwru.edu/SPARC/
https://doi.org/10.5281/zenodo.16284118
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import socket
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
SPARC_BASE = "https://astroweb.cwru.edu/SPARC"
TABLE_MRT_URL = f"{SPARC_BASE}/SPARC_Lelli2016c.mrt"
ROTMOD_URL = f"{SPARC_BASE}/Rotmod_LTG/{{galaxy}}_rotmod.dat"
ROTMOD_ZIP_URL = f"{SPARC_BASE}/Rotmod_LTG.zip"

# Zenodo long-term archive (DOI 10.5281/zenodo.16284118)
ZENODO_BASE = "https://zenodo.org/records/16284118/files"
ZENODO_TABLE_URL = f"{ZENODO_BASE}/SPARC_Lelli2016c.mrt"
ZENODO_ZIP_URL = f"{ZENODO_BASE}/Rotmod_LTG.zip"

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
# Zenodo zip helpers
# ---------------------------------------------------------------------------

def _download_bytes(url: str, retries: int = 3) -> bytes | None:
    """Download *url* entirely into memory; return bytes or None on failure."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                print(f"  [fail] {url}: {exc}", file=sys.stderr)
    return None


def _extract_zip_to(zip_bytes: bytes, dest_dir: Path) -> list[str]:
    """Extract a zip archive (given as bytes) into *dest_dir*.

    Returns the list of member names that were extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            # Skip directories and manifest files
            if member.endswith("/"):
                continue
            name = Path(member).name
            if not name:
                continue
            target = dest_dir / name
            if not target.exists():
                target.write_bytes(zf.read(member))
            extracted.append(name)
    return extracted


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_sparc(out_dir: str | Path, source: str = "auto") -> bool:
    """Download SPARC data to *out_dir*.

    Parameters
    ----------
    out_dir : str or Path
        Destination directory.  Created if it does not exist.
        Rotation curves go into ``<out_dir>/raw/``.
        The galaxy table is saved as ``<out_dir>/SPARC_Lelli2016c.csv``.
    source : {"auto", "cwru", "zenodo"}
        Data source.  ``"auto"`` tries CWRU first, then Zenodo.
        ``"cwru"`` only tries astroweb.cwru.edu.
        ``"zenodo"`` only tries the Zenodo archive
        (DOI 10.5281/zenodo.16284118).

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
        table_downloaded = False

        # Try CWRU first (unless Zenodo is forced)
        if source in ("auto", "cwru"):
            print(f"Downloading galaxy table from CWRU → {mrt_path} …")
            table_downloaded = _download_file(TABLE_MRT_URL, mrt_path)

        # Fall back to Zenodo
        if not table_downloaded and source in ("auto", "zenodo"):
            print(f"Trying Zenodo fallback for galaxy table …")
            table_downloaded = _download_file(ZENODO_TABLE_URL, mrt_path)

        if not table_downloaded:
            print(
                "ERROR: Could not download the SPARC galaxy table from any source.\n"
                "  Primary: " + TABLE_MRT_URL + "\n"
                "  Zenodo:  " + ZENODO_TABLE_URL + "\n"
                "Run manually: python scripts/download_sparc_data.py --out data/SPARC",
                file=sys.stderr,
            )
            return False

        print("Parsing galaxy table …")
        try:
            df = _parse_galaxy_table(mrt_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return False

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
    already_have = {p.stem.replace("_rotmod", "") for p in raw.glob("*_rotmod.dat")}
    need = [n for n in galaxy_names if n not in already_have]

    if not need:
        print(f"\nAll {len(galaxy_names)} rotation curves already present in {raw}")
        return True

    print(f"\nDownloading {len(need)} rotation curve(s) → {raw} …")

    # ------------------------------------------------------------------
    # 2a. Try bulk zip download first (faster than 175 individual requests)
    # ------------------------------------------------------------------
    zip_ok = False
    for zip_url, label in [
        (ROTMOD_ZIP_URL, "CWRU"),
        (ZENODO_ZIP_URL, "Zenodo"),
    ]:
        if source == "cwru" and label == "Zenodo":
            continue
        if source == "zenodo" and label == "CWRU":
            continue
        print(f"  Trying bulk zip from {label}: {zip_url} …")
        zip_bytes = _download_bytes(zip_url, retries=2)
        if zip_bytes is not None:
            extracted = _extract_zip_to(zip_bytes, raw)
            print(f"  Extracted {len(extracted)} file(s) from zip.")
            zip_ok = True
            break

    if zip_ok:
        # Check what's still missing after the zip extraction
        still_missing = [
            n for n in galaxy_names
            if not (raw / f"{n}_rotmod.dat").exists()
        ]
        if still_missing:
            print(
                f"  {len(still_missing)} file(s) not found in zip; "
                "attempting individual downloads …"
            )
        else:
            print(f"\nAll rotation curves obtained from zip.")
            return True
        need = still_missing

    # ------------------------------------------------------------------
    # 2b. Individual file downloads (fallback / supplement)
    # ------------------------------------------------------------------
    ok_count = 0
    fail_count = 0

    for name in need:
        dest = raw / f"{name}_rotmod.dat"
        if dest.exists():
            ok_count += 1
            continue

        # Try CWRU then Zenodo per file
        downloaded = False
        for url, label in [
            (ROTMOD_URL.format(galaxy=name), "CWRU"),
            (f"{ZENODO_BASE}/{name}_rotmod.dat", "Zenodo"),
        ]:
            if source == "cwru" and label == "Zenodo":
                continue
            if source == "zenodo" and label == "CWRU":
                continue
            if _download_file(url, dest, retries=2):
                downloaded = True
                break

        if downloaded:
            ok_count += 1
            print(f"  [ok]   {name}")
        else:
            fail_count += 1
        time.sleep(0.05)  # polite rate limiting

    total = len(galaxy_names)
    print(
        f"\nRotation curves: {ok_count} downloaded/present, "
        f"{fail_count} failed  (total {total})"
    )

    if fail_count > 0:
        print(
            f"WARNING: {fail_count} rotation curve(s) could not be downloaded. "
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
            "Download the SPARC dataset (Lelli+2016) from astroweb.cwru.edu "
            "with automatic fallback to Zenodo (DOI 10.5281/zenodo.16284118). "
            "Downloads the galaxy table and per-galaxy rotation curves."
        )
    )
    parser.add_argument(
        "--out", default="data/SPARC",
        help="Destination directory (default: data/SPARC).",
    )
    parser.add_argument(
        "--source", choices=["auto", "cwru", "zenodo"], default="auto",
        help=(
            "Data source: 'auto' tries CWRU then Zenodo (default), "
            "'cwru' uses only astroweb.cwru.edu, "
            "'zenodo' uses only the Zenodo archive."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ok = download_sparc(args.out, source=args.source)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
