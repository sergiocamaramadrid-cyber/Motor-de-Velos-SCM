"""
scripts/fetch_yang_catalog.py — Download the Yang SDSS group catalog.

The Yang et al. (2007, ApJ 671, 153) SDSS group catalog is used as an
external large-scale-structure (environment) proxy in the SCM pipeline
via ``scripts/crossmatch_yang_proxy.py``.

This script is intentionally conservative:

* If the catalog file already exists locally it exits immediately (exit 0).
* If ``--url`` is supplied it attempts the download from that URL.
* If ``--url`` is omitted it prints instructions and exits with exit code 1.

No URL is hard-coded because the primary host (gax.sjtu.edu.cn) experiences
periodic outages and no secondary mirror URL has been independently verified
as stable.  Once you locate a direct FITS/CSV link, pass it with ``--url``.

Usage
-----
    # Check / download
    python scripts/fetch_yang_catalog.py --url https://example.com/yang_dr7.fits

    # Check only (will exit 1 if the file is absent)
    python scripts/fetch_yang_catalog.py

    # Override output path
    python scripts/fetch_yang_catalog.py --url <URL> --out data/environment/yang.fits

References
----------
Yang et al. (2007) ApJ 671, 153.  arXiv:0707.4640
https://gax.sjtu.edu.cn/data/Group.html  (may be unavailable)
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import socket
from pathlib import Path

_TIMEOUT = 60        # seconds per request
_RETRY_DELAY = 3     # seconds between retries
_CHUNK = 65_536      # bytes per read chunk for progress display

socket.setdefaulttimeout(_TIMEOUT)

DEFAULT_OUT = Path("data/environment/yang_group_catalog.fits")

_INSTRUCTIONS = """\
Yang group catalog not found at: {out}

To download it, run:
    python scripts/fetch_yang_catalog.py --url <DIRECT_FITS_OR_CSV_URL>

Known sources (check availability):
  1. gax.sjtu.edu.cn/data/Group.html   (primary; may be offline)
  2. VizieR J/ApJ/671/153              (use astroquery.vizier; see docs below)
  3. NYU VAGC mirror                   (search "Yang SDSS group FITS" online)

astroquery alternative (run from Python):
    from astroquery.vizier import Vizier
    from pathlib import Path
    OUT = Path("{out}")
    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs("J/ApJ/671/153")
    if catalogs:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        catalogs[0].write(str(OUT), format="fits", overwrite=True)
        print("Saved:", OUT)

Once the file is present, run:
    python scripts/crossmatch_yang_proxy.py
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download *url* to *dest*; return True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(retries):
        try:
            print(f"  Connecting … (attempt {attempt + 1}/{retries})")
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(
                                f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB"
                                f"  ({pct:.0f}%)",
                                end="",
                                flush=True,
                            )
                print()  # newline after progress
            tmp.rename(dest)
            return True
        except (urllib.error.URLError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                wait = _RETRY_DELAY * (attempt + 1)
                print(
                    f"  [warn] attempt {attempt + 1} failed: {exc}  "
                    f"(retrying in {wait}s …)",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(f"  [fail] {url}: {exc}", file=sys.stderr)
    return False


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def fetch_yang_catalog(out: Path, url: str | None, retries: int = 3) -> int:
    """Ensure *out* exists, downloading from *url* if necessary.

    Returns
    -------
    int
        0 on success, 1 if the file is absent and no URL was supplied,
        2 on download failure.
    """
    if out.exists():
        size_mb = out.stat().st_size / 1e6
        print(f"OK: catalog already present → {out}  ({size_mb:.1f} MB)")
        return 0

    if url is None:
        print(_INSTRUCTIONS.format(out=out))
        return 1

    print(f"Downloading Yang group catalog from:\n  {url}")
    print(f"  → {out}")
    if _download_file(url, out, retries=retries):
        size_mb = out.stat().st_size / 1e6
        print(f"Download complete: {out}  ({size_mb:.1f} MB)")
        return 0

    print(
        f"ERROR: download failed after {retries} attempts.\n"
        "Check the URL and your internet connection.",
        file=sys.stderr,
    )
    return 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Yang et al. (2007) SDSS group catalog. "
            "Exits 0 if the file is already present, 1 if absent and no URL "
            "was given, 2 on download failure."
        )
    )
    parser.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help=(
            "Direct URL to a FITS or CSV file.  If omitted, the script "
            "checks for an existing local copy and prints instructions."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        metavar="PATH",
        help=f"Output path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        metavar="N",
        help="Number of download attempts (default: 3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rc = fetch_yang_catalog(
        out=Path(args.out),
        url=args.url,
        retries=args.retries,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
