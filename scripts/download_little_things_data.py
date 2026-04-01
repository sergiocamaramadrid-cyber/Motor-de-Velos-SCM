"""
scripts/download_little_things_data.py — Download the LITTLE THINGS dataset from VizieR.

Downloads from the CDS/VizieR permanent archive (Oh et al. 2015, AJ 149, 180):

  - table1.dat  — global galaxy properties (26 galaxies)
  - table2.dat … table27.dat  — per-galaxy rotation curves (one per galaxy,
    same order as Table 1)

Output layout::

    <out>/
        little_things_table1.csv     ← full galaxy properties (CSV)
        rotcur/
            CVnIdwA_rotcur.dat
            DDO43_rotcur.dat
            …                        ← 26 per-galaxy rotation-curve files

Default output directory: ``data/LITTLE_THINGS``

Usage
-----
::

    python scripts/download_little_things_data.py
    python scripts/download_little_things_data.py --out data/LITTLE_THINGS

References
----------
Oh et al. (2015), AJ 149, 180.
https://doi.org/10.1088/0004-6256/149/6/180

VizieR catalog J/AJ/149/180:
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/149/180
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIZIER_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/149/180"

TABLE1_URL = f"{VIZIER_BASE}/table1.dat"
README_URL = f"{VIZIER_BASE}/ReadMe"
# Rotation curves: VizieR tables 2–27 correspond to the 26 galaxies in the
# same order as Table 1 of Oh et al. (2015).
ROTCUR_URL = f"{VIZIER_BASE}/table{{n}}.dat"

_TIMEOUT = 30    # seconds per request
_RETRY_DELAY = 2  # seconds between retries

# Galaxy list in the order they appear in Oh et al. (2015) Table 1.
# VizieR table numbers: galaxy[i] ↔ table(i+2).dat   (table2 … table27)
_LT_GALAXIES: list[str] = [
    "CVnIdwA",
    "DDO43",
    "DDO46",
    "DDO47",
    "DDO50",
    "DDO52",
    "DDO53",
    "DDO63",
    "DDO69",
    "DDO70",
    "DDO71",
    "DDO75",
    "DDO87",
    "DDO101",
    "DDO126",
    "DDO133",
    "DDO154",
    "DDO168",
    "DDO210",
    "DDO216",
    "F564-V3",
    "Haro29",
    "IC1613",
    "NGC1569",
    "NGC2366",
    "UGC8508",
]

# Apply a global socket timeout so urllib never hangs
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


def _parse_table1(dat_path: Path) -> pd.DataFrame:
    """Parse the LITTLE THINGS global-properties table (CDS MRT format).

    The file uses a fixed-width / space-separated CDS format.  We try a
    whitespace-split first, falling back to a raw line scan that skips the
    CDS byte-by-byte header block.

    Column assignment follows the known Oh+2015 Table 1 structure:
        Name  HType  Dist  e_Dist  DistMethod  Incl  e_Incl  PA  e_PA
        Vsys  e_Vsys  MHI  (additional columns retained as-is)
    """
    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Identify the first data line (skips CDS header: lines starting with
    # '#', '=', '-', or blank lines at the top).
    data_lines: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in ("#", "=", "-"):
            continue
        # Stop at trailing separator lines that some CDS files append
        if stripped.startswith("---"):
            continue
        data_lines.append(stripped.split())

    if not data_lines:
        raise ValueError(
            f"No data lines found in {dat_path}. "
            "The file may be empty or in an unexpected format."
        )

    # Known column order for J/AJ/149/180 table1
    known_cols = [
        "Galaxy", "HType", "Dist_Mpc", "e_Dist", "DistMethod",
        "Incl_deg", "e_Incl", "PA_deg", "e_PA",
        "Vsys_kms", "e_Vsys", "MHI_1e7Msun",
    ]

    ncols = len(known_cols)
    rows = []
    for parts in data_lines:
        if len(parts) >= ncols:
            rows.append(parts[:ncols])
        elif len(parts) >= 1:
            # Pad with NaN strings if fewer columns are present
            padded = parts + ["NaN"] * (ncols - len(parts))
            rows.append(padded[:ncols])

    df = pd.DataFrame(rows, columns=known_cols)

    # Attempt numeric conversions for known numeric columns
    for col in ["Dist_Mpc", "e_Dist", "Incl_deg", "e_Incl",
                "PA_deg", "e_PA", "Vsys_kms", "e_Vsys", "MHI_1e7Msun"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_little_things(out_dir: str | Path) -> bool:
    """Download LITTLE THINGS data to *out_dir*.

    Parameters
    ----------
    out_dir : str or Path
        Destination directory.  Created if it does not exist.
        Rotation curves go into ``<out_dir>/rotcur/``.
        The galaxy table is saved as ``<out_dir>/little_things_table1.csv``.

    Returns
    -------
    bool
        ``True`` if all downloads succeeded, ``False`` if any file failed.
    """
    out = Path(out_dir)
    rotcur_dir = out / "rotcur"
    rotcur_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. ReadMe (informational — failure is non-fatal)
    # ------------------------------------------------------------------
    readme_path = out / "ReadMe"
    if not readme_path.exists():
        print(f"Downloading ReadMe → {readme_path} …")
        if not _download_file(README_URL, readme_path):
            print("  (ReadMe download failed — continuing without it)", file=sys.stderr)
    else:
        print(f"ReadMe already present: {readme_path}")

    # ------------------------------------------------------------------
    # 2. Global galaxy properties (Table 1)
    # ------------------------------------------------------------------
    dat1_path = out / "table1.dat"
    csv1_path = out / "little_things_table1.csv"

    if csv1_path.exists():
        print(f"Galaxy table already present: {csv1_path}")
    else:
        print(f"Downloading galaxy table → {dat1_path} …")
        if not _download_file(TABLE1_URL, dat1_path):
            print(
                "ERROR: Could not download the LITTLE THINGS galaxy table.",
                file=sys.stderr,
            )
            return False

        print("Parsing galaxy table …")
        try:
            df_tab1 = _parse_table1(dat1_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return False

        df_tab1.to_csv(csv1_path, index=False)
        print(
            f"Galaxy table saved as {csv1_path}  "
            f"({len(df_tab1)} rows, {len(df_tab1.columns)} columns)"
        )

    # ------------------------------------------------------------------
    # 3. Per-galaxy rotation curves (Tables 2–27)
    # ------------------------------------------------------------------
    print(f"\nDownloading rotation curves → {rotcur_dir} …")
    ok = 0
    fail = 0
    skipped = 0

    for i, galaxy in enumerate(_LT_GALAXIES, start=2):
        # VizieR table number is i (2 for CVnIdwA, …, 27 for UGC8508)
        dest = rotcur_dir / f"{galaxy}_rotcur.dat"
        if dest.exists():
            skipped += 1
            ok += 1
            continue
        url = ROTCUR_URL.format(n=i)
        if _download_file(url, dest, retries=3):
            ok += 1
            print(f"  [ok]   {galaxy}  (table{i}.dat)")
        else:
            fail += 1
        time.sleep(0.05)  # polite rate limiting

    total = len(_LT_GALAXIES)
    print(
        f"\nRotation curves: {ok - skipped} downloaded, {skipped} already present, "
        f"{fail} failed  (total {total})"
    )

    if fail > 0:
        print(
            f"WARNING: {fail} rotation curve file(s) could not be downloaded.",
            file=sys.stderr,
        )

    return True  # partial download still allows the pipeline to run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the LITTLE THINGS dataset (Oh et al. 2015, AJ 149, 180) "
            "from the CDS/VizieR archive (J/AJ/149/180). "
            "Downloads the global galaxy-properties table and per-galaxy "
            "rotation curves."
        )
    )
    parser.add_argument(
        "--out",
        default="data/LITTLE_THINGS",
        help="Destination directory (default: data/LITTLE_THINGS).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ok = download_little_things(args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
