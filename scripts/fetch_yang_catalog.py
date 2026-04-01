#!/usr/bin/env python3
"""
scripts/fetch_yang_catalog.py — Download and normalise the YANG group catalog.

YANG = Yang et al. (2007, 2012) SDSS galaxy group catalog.
A public copy is available through CDS/VizieR or the SDSS DR7 value-added
catalogs.

The script downloads the file at the given URL (CSV, ASCII whitespace, or FITS),
normalises column names, and saves a clean copy to ``data/yang/yang_catalog.csv``.

If the catalog is already present locally the script prints ``OK`` and exits 0
without re-downloading.

Expected output columns (best-effort; absent columns are left out):
    galaxy_id  ra  dec  z  log_mstar  log_mhalo

Usage
-----
    python scripts/fetch_yang_catalog.py --url <DIRECT_URL>

    # Force re-download even when local copy exists:
    python scripts/fetch_yang_catalog.py --url <DIRECT_URL> --force

Options
-------
--url       Direct HTTP/FTP/file URL to the YANG catalog.
--out       Destination file (default: data/yang/yang_catalog.csv).
--force     Re-download and overwrite even if local copy already exists.
--timeout   Per-request timeout in seconds (default: 60).
--retries   Number of download retries (default: 3).
"""

from __future__ import annotations

import argparse
import re
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

_DEFAULT_OUT = Path("data/yang/yang_catalog.csv")
_TIMEOUT = 60
_RETRY_DELAY = 2

# Column-name heuristics: (canonical_name, list_of_regex_patterns)
_COL_MAP: list[tuple[str, list[str]]] = [
    ("galaxy_id", [r"^galaxy[_\s]?id$", r"^id$", r"^ngal$", r"^objid$", r"^groupid$",
                   r"^group[_\s]?id$"]),
    ("ra",        [r"^ra[_\s]?(j2000|deg)?$", r"^raj2000$", r"^alpha$"]),
    ("dec",       [r"^dec[_\s]?(j2000|deg)?$", r"^dej2000$", r"^delta$", r"^de$"]),
    ("z",         [r"^z$", r"^z_spec$", r"^zspec$", r"^redshift$", r"^z_grp$"]),
    ("log_mstar", [r"^log\s*m\s*star$", r"^logms$", r"^log_?m_?star$", r"^logmstar$",
                   r"^log_?mstellar$", r"^logm\*$", r"^log_?m\*$"]),
    ("log_mhalo", [r"^log\s*m\s*h(alo)?$", r"^logmh$", r"^log_?m_?halo$",
                   r"^logmhalo$", r"^log_?mh$"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, timeout: int, retries: int) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    for attempt in range(retries):
        try:
            socket.setdefaulttimeout(timeout)
            urllib.request.urlretrieve(url, dest)
            return True
        except (urllib.error.URLError, OSError) as exc:
            if attempt < retries - 1:
                wait = _RETRY_DELAY * (attempt + 1)
                print(f"  [retry {attempt+1}/{retries}] {exc} — waiting {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [fail] {url}: {exc}", file=sys.stderr)
    return False


def _normalise_colname(name: str) -> str | None:
    """Map a raw column name to a canonical name, or return None if no match."""
    cleaned = name.strip().lower()
    for canonical, patterns in _COL_MAP:
        for pat in patterns:
            if re.fullmatch(pat, cleaned):
                return canonical
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename matched columns to canonical names; preserve unrecognised extras."""
    rename: dict[str, str] = {}
    canonical_seen: set[str] = set()
    for col in df.columns:
        canon = _normalise_colname(col)
        if canon and canon not in canonical_seen:
            rename[col] = canon
            canonical_seen.add(canon)
    return df.rename(columns=rename)


def _load_raw(path: Path) -> pd.DataFrame:
    """Try to load path as FITS, then CSV, then whitespace-delimited ASCII."""
    suffix = path.suffix.lower()

    # ---- FITS -----------------------------------------------------------
    if suffix in (".fits", ".fit", ".fts"):
        try:
            from astropy.io import fits as _fits
            with _fits.open(path) as hdul:
                for hdu in hdul[1:]:
                    if hasattr(hdu, "columns") and hdu.data is not None:
                        return pd.DataFrame(hdu.data.byteswap().newbyteorder())
        except ImportError:
            print("WARNING: astropy not installed; cannot read FITS. "
                  "Install it with: pip install astropy", file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: FITS read failed: {exc}", file=sys.stderr)

    # ---- CSV / ASCII ----------------------------------------------------
    errors: list[str] = []
    for sep in (",", r"\s+", "\t", ";"):
        try:
            df = pd.read_csv(path, sep=sep, comment="#", engine="python")
            if len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError(
        f"Could not parse {path} as CSV or whitespace-delimited ASCII. "
        f"Errors: {errors}"
    )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def fetch_yang_catalog(
    url: str,
    out: Path = _DEFAULT_OUT,
    force: bool = False,
    timeout: int = _TIMEOUT,
    retries: int = 3,
) -> Path:
    """Download the YANG catalog from *url* and save normalised CSV to *out*.

    Parameters
    ----------
    url : str
        Direct URL to the catalog file.
    out : Path
        Destination CSV path.
    force : bool
        Re-download even if *out* already exists.
    timeout, retries :
        Download parameters.

    Returns
    -------
    Path
        Path to the saved CSV.

    Raises
    ------
    SystemExit
        On download or parse failure.
    """
    out = Path(out)

    # ---- Already present? -----------------------------------------------
    if out.exists() and not force:
        df = pd.read_csv(out, nrows=2)
        n = sum(1 for _ in open(out)) - 1  # rough row count
        print(f"OK — YANG catalog already present: {out}  ({n} rows, "
              f"cols: {list(df.columns)})")
        return out

    # ---- Download -------------------------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dest = out.parent / ("_raw_yang" + Path(url.rstrip("/")).suffix)
    if not raw_dest.suffix:
        raw_dest = raw_dest.with_suffix(".dat")

    print(f"Downloading YANG catalog → {raw_dest} …")
    ok = _download(url, raw_dest, timeout, retries)
    if not ok:
        print("ERROR: Could not download the YANG catalog.", file=sys.stderr)
        sys.exit(1)

    # ---- Parse ----------------------------------------------------------
    print("Parsing …")
    try:
        df = _load_raw(raw_dest)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Raw shape: {df.shape}  raw columns: {list(df.columns)}")

    # ---- Normalise columns ----------------------------------------------
    df = _normalise_columns(df)
    print(f"  Normalised columns: {list(df.columns)}")

    # ---- Basic sanity checks --------------------------------------------
    needed = {"ra", "dec"}
    missing = needed - set(df.columns)
    if missing:
        print(
            f"WARNING: Could not detect columns {missing} in YANG catalog. "
            "Cross-match will be limited. Check the column names in the raw file.",
            file=sys.stderr,
        )

    # ---- Save -----------------------------------------------------------
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and normalise the YANG group catalog (Yang et al. 2007/2012). "
            "Saves a clean CSV to data/yang/yang_catalog.csv."
        )
    )
    parser.add_argument(
        "--url", required=True,
        help="Direct URL to the YANG catalog file (CSV, ASCII, or FITS).",
    )
    parser.add_argument(
        "--out", default=str(_DEFAULT_OUT),
        help=f"Destination CSV path (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the local copy already exists.",
    )
    parser.add_argument(
        "--timeout", type=int, default=_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {_TIMEOUT}).",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Number of download retries (default: 3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    fetch_yang_catalog(
        url=args.url,
        out=Path(args.out),
        force=args.force,
        timeout=args.timeout,
        retries=args.retries,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
