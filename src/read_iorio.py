"""
read_iorio.py
-------------
Adapter for loading SPARC/Iorio rotation-curve data files into a
pandas DataFrame.

Column-index map (0-based)
--------------------------
Mandatory columns (present in every file):
  0  R       – galactocentric radius           [kpc]
  1  Vobs    – observed rotation velocity       [km/s]
  2  errV    – uncertainty on Vobs             [km/s]
  3  Vgas    – gas rotation contribution       [km/s]
  4  Vdisk   – stellar-disk contribution       [km/s]
  5  Vbul    – stellar-bulge contribution      [km/s]

Optional extra columns (present in some extended files):
  6  sigma_V – line-of-sight velocity dispersion [km/s]
  7  SBdisk  – disk surface brightness          [L_sun/pc^2]
  8  SBbul   – bulge surface brightness         [L_sun/pc^2]

Supported delimiters: whitespace (default), tab ('\\t'), comma (',').
The adapter auto-detects the delimiter when ``delimiter='auto'``.
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

#: Mandatory columns and their 0-based indices in the raw data table.
MANDATORY_COLS: dict[str, int] = {
    "R": 0,
    "Vobs": 1,
    "errV": 2,
    "Vgas": 3,
    "Vdisk": 4,
    "Vbul": 5,
}

#: Optional extra columns (index → name).  Added when present.
OPTIONAL_COLS: dict[int, str] = {
    6: "sigma_V",
    7: "SBdisk",
    8: "SBbul",
}

#: Minimum number of columns a valid file must have.
MIN_COLS = len(MANDATORY_COLS)


# ---------------------------------------------------------------------------
# Delimiter auto-detection
# ---------------------------------------------------------------------------

def _detect_delimiter(first_data_line: str) -> str:
    """Return the most likely delimiter found in *first_data_line*.

    Priority: tab > comma > whitespace (fallback).
    """
    if "\t" in first_data_line:
        return "\t"
    if "," in first_data_line:
        return ","
    return r"\s+"


def _first_data_line(filepath: str | Path) -> str:
    """Return the first non-comment, non-empty line of *filepath*."""
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


# ---------------------------------------------------------------------------
# Header / column validation
# ---------------------------------------------------------------------------

def validate_header(df: pd.DataFrame, galaxy_name: str = "") -> None:
    """Raise ``ValueError`` if *df* is missing mandatory columns.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`read_galaxy`.
    galaxy_name:
        Optional label used in error messages.
    """
    missing = [col for col in MANDATORY_COLS if col not in df.columns]
    if missing:
        tag = f" [{galaxy_name}]" if galaxy_name else ""
        raise ValueError(
            f"Parsed data{tag} is missing mandatory column(s): {missing}. "
            "Check that the column-index map in read_iorio.py matches the file layout."
        )

    # Basic sanity checks on values
    if (df["R"] < 0).any():
        logger.warning("%s: negative radius values detected.", galaxy_name or "unknown")
    if (df["errV"] <= 0).any():
        logger.warning(
            "%s: non-positive velocity errors detected – these rows may be flagged.",
            galaxy_name or "unknown",
        )


# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------

def read_galaxy(
    filepath: str | Path,
    delimiter: str = "auto",
    comment: str = "#",
    extra_cols: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """Read a single SPARC/Iorio rotation-curve file.

    Parameters
    ----------
    filepath:
        Path to the data file.
    delimiter:
        Column separator.  Use ``'auto'`` (default) to detect automatically.
    comment:
        Character that marks comment / header lines.
    extra_cols:
        Override for :data:`OPTIONAL_COLS`.  Mapping of column index → column name
        for any additional columns beyond the mandatory ones.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns as defined in :data:`MANDATORY_COLS` plus any
        detected optional columns.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the parsed table has fewer columns than :data:`MIN_COLS`.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if delimiter == "auto":
        first_line = _first_data_line(filepath)
        delimiter = _detect_delimiter(first_line)

    sep = delimiter if delimiter != r"\s+" else None
    df_raw = pd.read_csv(
        filepath,
        sep=sep if sep is not None else r"\s+",
        comment=comment,
        header=None,
        engine="python",
    )

    n_cols = df_raw.shape[1]
    if n_cols < MIN_COLS:
        raise ValueError(
            f"{filepath.name}: expected at least {MIN_COLS} columns, got {n_cols}. "
            "Verify the file layout and the delimiter."
        )

    # Assign mandatory column names
    col_names: list[str] = [""] * n_cols
    for name, idx in MANDATORY_COLS.items():
        col_names[idx] = name

    # Assign optional column names
    opt = extra_cols if extra_cols is not None else OPTIONAL_COLS
    for idx, name in opt.items():
        if idx < n_cols:
            col_names[idx] = name

    # Fill any remaining unnamed columns with generic labels
    for i, name in enumerate(col_names):
        if not name:
            col_names[i] = f"col_{i}"

    df_raw.columns = col_names

    galaxy_name = filepath.stem
    logger.info(
        "Loaded %s: %d rows, %d columns (%s).",
        galaxy_name,
        len(df_raw),
        n_cols,
        ", ".join(col_names),
    )

    validate_header(df_raw, galaxy_name)
    return df_raw


# ---------------------------------------------------------------------------
# Batch reader
# ---------------------------------------------------------------------------

def read_batch(
    filepaths: list[str | Path],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Read multiple galaxy files and return a dict keyed by galaxy name.

    Parameters
    ----------
    filepaths:
        List of paths to data files.
    **kwargs:
        Forwarded to :func:`read_galaxy`.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for fp in filepaths:
        fp = Path(fp)
        try:
            df = read_galaxy(fp, **kwargs)
            results[fp.stem] = df
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s: %s", fp.name, exc)
            errors.append(f"{fp.name}: {exc}")

    if errors:
        logger.warning("%d file(s) could not be read:\n  %s", len(errors), "\n  ".join(errors))
    logger.info("Successfully read %d / %d galaxies.", len(results), len(filepaths))
    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _delimiter_arg(raw: str) -> str:
    if raw == "tab":
        return "\t"
    return raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.read_iorio",
        description="Read and validate SPARC/Iorio rotation-curve files.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("file", nargs="?", metavar="FILE",
                             help="Single rotation-curve file to read and validate.")
    input_group.add_argument("--batch", metavar="LIST_FILE",
                             help="Text file with one galaxy data-file path per line.")
    input_group.add_argument("--glob", dest="glob_pattern", metavar="PATTERN",
                             help="Glob pattern for galaxy files.")

    parser.add_argument("--delimiter", default="auto")
    parser.add_argument("--out", metavar="DIR", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    delimiter = _delimiter_arg(args.delimiter)
    if args.file:
        filepaths = [Path(args.file)]
    elif args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            sys.exit(f"Batch list file not found: {batch_file}")
        filepaths = [
            Path(line.strip())
            for line in batch_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        matched = sorted(_glob.glob(args.glob_pattern))
        if not matched:
            sys.exit(f"No files matched pattern: {args.glob_pattern}")
        filepaths = [Path(p) for p in matched]

    galaxies = read_batch(filepaths, delimiter=delimiter)

    summary_rows = []
    for name, df in galaxies.items():
        summary_rows.append({
            "galaxy": name,
            "n_points": len(df),
            "R_max_kpc": float(df["R"].max()),
            "Vobs_max_kms": float(df["Vobs"].max()),
            "has_sigma_V": "sigma_V" in df.columns,
            "status": "OK",
        })
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"{name}_parsed.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
