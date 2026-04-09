"""
scripts/build_galaxy_catalog_env.py — Assemble the master SPARC environment
catalog used by the SCM environmental correlation analysis.

Three source tables are merged on the ``galaxy`` key:

* **SPARC summary table** (``data/sparc_basic.csv``) — galaxy name and stellar
  mass (Mstar, M_sun).
* **Outer-disk slope catalog** (``results/slope_tail.csv``) — outer-disk
  velocity slope produced by ``sparc_slope_tail.py``.
* **Environmental proxy table** (``data/env_proxy.csv``) — the per-galaxy
  environmental proxy value (``env_proxy``).

The merged table receives a derived column ``logM = log10(Mstar)`` and is
written to ``data/galaxy_catalog_env.csv``.  This file is the default input
for ``plot_env_mass_scan.py``.

Usage
-----
Default paths::

    python scripts/build_galaxy_catalog_env.py

Custom paths::

    python scripts/build_galaxy_catalog_env.py \\
        --sparc-csv   data/sparc_basic.csv \\
        --slopes-csv  results/slope_tail.csv \\
        --env-csv     data/env_proxy.csv \\
        --out         data/galaxy_catalog_env.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARC_CSV_DEFAULT = "data/sparc_basic.csv"
SLOPES_CSV_DEFAULT = "results/slope_tail.csv"
ENV_CSV_DEFAULT = "data/env_proxy.csv"
OUTPUT_CSV_DEFAULT = "data/galaxy_catalog_env.csv"

# Required columns in each source table
_SPARC_REQUIRED = {"galaxy", "Mstar"}
_SLOPES_REQUIRED = {"galaxy", "slope_tail"}
_ENV_REQUIRED = {"galaxy", "env_proxy"}

# Output column set (guaranteed after build_catalog succeeds)
OUTPUT_COLUMNS = ["galaxy", "Mstar", "slope_tail", "env_proxy", "logM"]


# ---------------------------------------------------------------------------
# Individual loaders (validate columns on entry)
# ---------------------------------------------------------------------------

def _load_csv(path: str | Path, required: set[str], label: str) -> pd.DataFrame:
    """Load a CSV and validate that *required* columns are present.

    Parameters
    ----------
    path : str or Path
        File to load.
    required : set of str
        Column names that must be present.
    label : str
        Human-readable description used in error messages.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )
    return df


def load_sparc(sparc_csv: str | Path) -> pd.DataFrame:
    """Load the SPARC summary table (must have columns ``galaxy``, ``Mstar``)."""
    return _load_csv(
        sparc_csv, _SPARC_REQUIRED,
        "SPARC summary CSV"
    )


def load_slopes(slopes_csv: str | Path) -> pd.DataFrame:
    """Load the slope-tail catalog (must have columns ``galaxy``, ``slope_tail``)."""
    return _load_csv(
        slopes_csv, _SLOPES_REQUIRED,
        "Slope-tail CSV"
    )


def load_env(env_csv: str | Path) -> pd.DataFrame:
    """Load the environmental proxy table (must have columns ``galaxy``, ``env_proxy``)."""
    return _load_csv(
        env_csv, _ENV_REQUIRED,
        "Environmental proxy CSV"
    )


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_catalog(
    sparc_csv: str | Path = SPARC_CSV_DEFAULT,
    slopes_csv: str | Path = SLOPES_CSV_DEFAULT,
    env_csv: str | Path = ENV_CSV_DEFAULT,
    out_path: str | Path = OUTPUT_CSV_DEFAULT,
) -> pd.DataFrame:
    """Merge the three source tables and write the master catalog CSV.

    The merge strategy is inner join on ``galaxy`` for both steps, so only
    galaxies present in **all three** tables appear in the output.  The derived
    column ``logM = log10(Mstar)`` is appended before writing.

    Parameters
    ----------
    sparc_csv : str or Path
        SPARC summary table (columns: ``galaxy``, ``Mstar``).
    slopes_csv : str or Path
        Outer-disk slope catalog (columns: ``galaxy``, ``slope_tail``).
    env_csv : str or Path
        Environmental proxy table (columns: ``galaxy``, ``env_proxy``).
    out_path : str or Path
        Destination CSV file.  Parent directories are created automatically.

    Returns
    -------
    pd.DataFrame
        The assembled catalog (same content written to *out_path*).

    Raises
    ------
    FileNotFoundError
        If any of the three input files does not exist.
    ValueError
        If required columns are missing from any input file.
    """
    df_sparc = load_sparc(sparc_csv)
    df_slopes = load_slopes(slopes_csv)
    df_env = load_env(env_csv)

    df = df_sparc.merge(df_slopes, on="galaxy", how="inner")
    df = df.merge(df_env, on="galaxy", how="inner")

    df["logM"] = np.log10(df["Mstar"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the master SPARC environment catalog "
            "(data/galaxy_catalog_env.csv) from three source tables."
        )
    )
    parser.add_argument(
        "--sparc-csv", default=SPARC_CSV_DEFAULT, dest="sparc_csv",
        help=f"SPARC summary CSV (default: {SPARC_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--slopes-csv", default=SLOPES_CSV_DEFAULT, dest="slopes_csv",
        help=f"Outer-disk slope catalog CSV (default: {SLOPES_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--env-csv", default=ENV_CSV_DEFAULT, dest="env_csv",
        help=f"Environmental proxy CSV (default: {ENV_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUTPUT_CSV_DEFAULT, dest="out",
        help=f"Output CSV path (default: {OUTPUT_CSV_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: merge tables and write the master catalog.

    Returns
    -------
    dict with keys:
        catalog    — pd.DataFrame (the assembled catalog)
        n          — int (number of rows in the catalog)
        out_path   — str (absolute path to the written CSV)
    """
    args = _parse_args(argv)

    df = build_catalog(
        sparc_csv=args.sparc_csv,
        slopes_csv=args.slopes_csv,
        env_csv=args.env_csv,
        out_path=args.out,
    )

    out = Path(args.out)
    print(f"Master catalog ready: {out}  ({len(df)} galaxies)")

    return {
        "catalog": df,
        "n": len(df),
        "out_path": str(out.resolve()),
    }


if __name__ == "__main__":
    main()
