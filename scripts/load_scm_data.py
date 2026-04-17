"""
scripts/load_scm_data.py — Load and inspect the three SCM result catalogs.

Reads ``SCM_spectral_catalog.csv``, ``SCM_summary.csv``, and
``SCM_peaks.csv`` from a data directory, prints a summary (shapes, column
names, first rows), and optionally exports all three tables to a single
Excel workbook.

Usage
-----
Load and print summary::

    python scripts/load_scm_data.py --data-dir /path/to/scm/results

With Excel export::

    python scripts/load_scm_data.py \\
        --data-dir /path/to/scm/results \\
        --excel out/SCM_catalogs.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_FILES: dict[str, str] = {
    "spectral_catalog": "SCM_spectral_catalog.csv",
    "summary_catalog": "SCM_summary.csv",
    "peaks_catalog": "SCM_peaks.csv",
}

_SEP = "=" * 60


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_catalogs(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the three SCM catalogs from *data_dir*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``SCM_spectral_catalog.csv``,
        ``SCM_summary.csv``, and ``SCM_peaks.csv``.

    Returns
    -------
    dict
        Keys are ``spectral_catalog``, ``summary_catalog``,
        ``peaks_catalog``; values are DataFrames.

    Raises
    ------
    FileNotFoundError
        If any of the three catalog files is missing.
    """
    data_dir = Path(data_dir)
    catalogs: dict[str, pd.DataFrame] = {}
    for key, filename in CATALOG_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {path}\n"
                f"Expected '{filename}' inside '{data_dir}'."
            )
        catalogs[key] = pd.read_csv(path)
    return catalogs


def print_summary(catalogs: dict[str, pd.DataFrame]) -> None:
    """Print shape, column names, and first rows for each catalog."""
    for name, df in catalogs.items():
        print(_SEP)
        print(f"  {name}")
        print(f"  Shape  : {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(df.head().to_string(index=False))
    print(_SEP)


def export_excel(
    catalogs: dict[str, pd.DataFrame],
    out_path: str | Path,
) -> Path:
    """Write all catalogs to a single Excel workbook.

    Parameters
    ----------
    catalogs : dict[str, DataFrame]
        Mapping from sheet name to DataFrame.
    out_path : str or Path
        Destination ``.xlsx`` file path.  Parent directories are created
        automatically if they do not exist.

    Returns
    -------
    Path
        Resolved path of the written workbook.

    Raises
    ------
    ImportError
        If *openpyxl* is not installed.
    """
    try:
        import openpyxl  # availability check only; engine is passed to ExcelWriter
        del openpyxl
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. "
            "Install it with: pip install openpyxl"
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in catalogs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n  Excel workbook written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect the three SCM result catalogs."
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="DIR",
        help=(
            "Directory containing SCM_spectral_catalog.csv, "
            "SCM_summary.csv, and SCM_peaks.csv."
        ),
    )
    parser.add_argument(
        "--excel", default=None, metavar="FILE",
        help="If provided, export all catalogs to this .xlsx file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Load SCM catalogs, print a summary, and optionally export to Excel.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments (uses ``sys.argv`` when *None*).

    Returns
    -------
    dict
        ``catalogs`` — mapping of name → DataFrame,
        ``data_dir``  — resolved data directory path (str),
        ``excel_path``— path of written workbook (str) or *None*.
    """
    args = _parse_args(argv)
    catalogs = load_catalogs(args.data_dir)
    print_summary(catalogs)

    excel_path: str | None = None
    if args.excel:
        written = export_excel(catalogs, args.excel)
        excel_path = str(written)

    return {
        "catalogs": catalogs,
        "data_dir": str(Path(args.data_dir).resolve()),
        "excel_path": excel_path,
    }


if __name__ == "__main__":
    main()
