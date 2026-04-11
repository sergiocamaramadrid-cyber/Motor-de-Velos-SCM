"""
scripts/download_iate_group_catalog.py — Download the IATE FoF+Halo Group Catalog.

Downloads from the IATE CONICET catalog server::

    https://catalogs.iate.conicet.unc.edu.ar/fofandhalo/FINAL_Group.dat

The file is a whitespace-separated ASCII table.  Comment lines (starting
with ``#``) are parsed to extract column names when present.  The result
is saved as ``data/iate/iate_group_catalog.csv`` (CSV format).

Usage
-----
    python scripts/download_iate_group_catalog.py
    python scripts/download_iate_group_catalog.py --out data/iate/iate_group_catalog.csv

References
----------
Rodríguez, F. & Merchán, M. (2020), A&A, 636, A61.
https://catalogs.iate.conicet.unc.edu.ar/fofandhalo/
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path
from typing import IO

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_URL = (
    "https://catalogs.iate.conicet.unc.edu.ar/fofandhalo/FINAL_Group.dat"
)
DEFAULT_OUT = Path("data/iate/iate_group_catalog.csv")

_TIMEOUT = 60   # seconds per request
_RETRY_DELAY = 3  # seconds between retries

# Fallback column names used when the file has no header comments.
# Order matches the standard IATE FoF+Halo group catalog format
# (Rodríguez & Merchán 2020).
_FALLBACK_COLUMNS = [
    "GroupID",
    "RA_deg",
    "Dec_deg",
    "z",
    "N_members",
    "sigma_v_kms",
    "log_Mh_Msun",
    "R200_Mpc",
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_text(url: str, retries: int = 3) -> str:
    """Download *url* and return content as a string; raise on failure."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(
        f"Could not download {url} after {retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_dat(content: str | IO[str]) -> pd.DataFrame:
    """Parse the FINAL_Group.dat content into a DataFrame.

    Supports two header conventions:

    1. **Named-column header**: one or more ``#``-prefixed lines where the
       last such line before data contains the column names.  Example::

           # GroupID RA_deg Dec_deg z N_members sigma_v_kms log_Mh_Msun R200_Mpc
           1  150.23  2.45  0.081  5  312.4  13.2  0.92

    2. **No header**: data starts immediately; fallback column names are
       applied based on the known catalog format.

    Parameters
    ----------
    content : str or file-like
        Raw text content of FINAL_Group.dat, or a file-like object.

    Returns
    -------
    pd.DataFrame
        Parsed catalog with typed numeric columns.

    Raises
    ------
    ValueError
        If the content is empty or cannot be parsed.
    """
    if hasattr(content, "read"):
        text = content.read()
    else:
        text = content

    lines = text.splitlines()
    if not lines:
        raise ValueError("FINAL_Group.dat content is empty.")

    # Separate comment lines from data lines
    comment_lines: list[str] = []
    data_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_lines.append(stripped.lstrip("#").strip())
        else:
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(
            "FINAL_Group.dat has no data lines (only comments or blank lines)."
        )

    # Try to derive column names from comment lines.  The last comment line
    # whose token count matches the data column count is used as the header.
    n_cols = len(data_lines[0].split())
    column_names: list[str] = []
    for cline in reversed(comment_lines):
        parts = cline.split()
        if len(parts) == n_cols:
            column_names = parts
            break

    # Parse data using pandas (fast, handles varying whitespace)
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        names=column_names if column_names else None,
    )

    # Apply fallback names if still no column names
    if not column_names:
        if len(df.columns) == len(_FALLBACK_COLUMNS):
            df.columns = _FALLBACK_COLUMNS
        else:
            df.columns = [f"col{i}" for i in range(len(df.columns))]

    # Coerce all columns to numeric where possible; keep original values for
    # any column that converts entirely to NaN (i.e., truly non-numeric).
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    for col in df.columns:
        if not numeric_df[col].isna().all():
            df[col] = numeric_df[col]

    return df


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download_iate_group_catalog(out_path: str | Path) -> pd.DataFrame:
    """Download the IATE FoF+Halo group catalog and save as CSV.

    Parameters
    ----------
    out_path : str or Path
        Destination CSV file.  Parent directories are created as needed.

    Returns
    -------
    pd.DataFrame
        The parsed catalog.

    Raises
    ------
    RuntimeError
        If the download fails after all retries.
    ValueError
        If the downloaded file cannot be parsed.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"Catalog already present: {out}")
        return pd.read_csv(out)

    print(f"Downloading IATE group catalog → {CATALOG_URL} …")
    content = _download_text(CATALOG_URL)

    print("Parsing catalog …")
    df = parse_dat(content)

    df.to_csv(out, index=False)
    print(f"Saved {len(df)} groups → {out}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the IATE FoF+Halo galaxy group catalog "
            "(Rodríguez & Merchán 2020) from catalogs.iate.conicet.unc.edu.ar."
        )
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Destination CSV file (default: {DEFAULT_OUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry-point for the IATE group catalog downloader.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    dict
        Keys: ``out_path`` (str), ``n_groups`` (int), ``columns`` (list).
    """
    args = _parse_args(argv)
    try:
        df = download_iate_group_catalog(args.out)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    return {
        "out_path": args.out,
        "n_groups": len(df),
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    main()
