"""
scripts/build_env_real_input.py — Build the crossmatched env-real input table.

Merges three source catalogs:

1. **F3 catalog** (``--f3-catalog``, default ``data/f3_catalog_real.csv``)
   Required columns: ``galaxy``, ``F3``
   Produces: ``galaxy_name`` (normalised), ``delta_f3`` = F3 − 0.5

2. **SPARC global table** (``--sparc-basic``, default ``data/sparc_basic.csv``)
   Required columns: ``Galaxy``, ``L36``, ``MHI``
   Produces: ``galaxy_name`` (normalised), ``logM`` = log10(0.5·L36·10⁹ + 1.33·MHI·10⁹)

3. **Chae environment catalog** (``--chae-env``, default ``data/chae_env.csv``)
   Required columns: ``galaxy_name``, ``e_env``

Galaxy names are normalised to upper-case with spaces and hyphens removed so
that the three catalogs can be inner-joined on a common key.

Output
------
``--out`` (default ``data/env_real_input.csv``) — four-column CSV:
``galaxy_name``, ``delta_f3``, ``logM``, ``e_env``

Usage
-----
::

    python scripts/build_env_real_input.py

    python scripts/build_env_real_input.py \\
        --f3-catalog  data/f3_catalog_real.csv \\
        --sparc-basic data/sparc_basic.csv \\
        --chae-env    data/chae_env.csv \\
        --out         data/env_real_input.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_F3_CATALOG = "data/f3_catalog_real.csv"
DEFAULT_SPARC_BASIC = "data/sparc_basic.csv"
DEFAULT_CHAE_ENV = "data/chae_env.csv"
DEFAULT_OUT = "data/env_real_input.csv"

F3_REQUIRED = ["galaxy", "F3"]
SPARC_REQUIRED = ["Galaxy", "L36", "MHI"]
CHAE_REQUIRED = ["galaxy_name", "e_env"]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def clean_name(x: object) -> str:
    """Normalise a galaxy name to upper-case, no spaces or hyphens.

    Parameters
    ----------
    x : object
        Raw galaxy name (any type accepted, converted via ``str``).

    Returns
    -------
    str
        Upper-case string with spaces and hyphens removed.
    """
    return str(x).strip().upper().replace(" ", "").replace("-", "")


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_f3_catalog(path: str | Path) -> pd.DataFrame:
    """Load and validate the F3 catalog.

    Parameters
    ----------
    path : str | Path
        CSV with at least ``galaxy`` and ``F3`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy_name`` (normalised), ``delta_f3`` = F3 − 0.5.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    df = pd.read_csv(path)
    missing = [c for c in F3_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"F3 catalog missing columns: {missing}")

    return pd.DataFrame({
        "galaxy_name": df["galaxy"].apply(clean_name),
        "delta_f3": pd.to_numeric(df["F3"], errors="coerce") - 0.5,
    })


def load_sparc_basic(path: str | Path) -> pd.DataFrame:
    """Load and validate the SPARC global table.

    Stellar mass proxy is computed as::

        logM = log10(0.5 * L36 * 1e9 + 1.33 * MHI * 1e9)

    Parameters
    ----------
    path : str | Path
        CSV with at least ``Galaxy``, ``L36``, and ``MHI`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy_name`` (normalised), ``logM``.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    df = pd.read_csv(path)
    missing = [c for c in SPARC_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"SPARC table missing columns: {missing}")

    L36 = pd.to_numeric(df["L36"], errors="coerce")
    MHI = pd.to_numeric(df["MHI"], errors="coerce")
    mass = 0.5 * L36 * 1e9 + 1.33 * MHI * 1e9

    return pd.DataFrame({
        "galaxy_name": df["Galaxy"].apply(clean_name),
        "logM": np.log10(mass),
    })


def load_chae_env(path: str | Path) -> pd.DataFrame:
    """Load and validate the Chae environment catalog.

    Parameters
    ----------
    path : str | Path
        CSV with at least ``galaxy_name`` and ``e_env`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy_name`` (normalised), ``e_env``.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    df = pd.read_csv(path)
    missing = [c for c in CHAE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Chae env catalog missing columns: {missing}")

    return pd.DataFrame({
        "galaxy_name": df["galaxy_name"].apply(clean_name),
        "e_env": pd.to_numeric(df["e_env"], errors="coerce"),
    })


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_crossmatch(
    f3: pd.DataFrame,
    sparc: pd.DataFrame,
    chae: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join the three catalogs on normalised ``galaxy_name``.

    Rows with any NaN in the output columns are dropped.

    Parameters
    ----------
    f3 : pd.DataFrame
        Output of :func:`load_f3_catalog`.
    sparc : pd.DataFrame
        Output of :func:`load_sparc_basic`.
    chae : pd.DataFrame
        Output of :func:`load_chae_env`.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy_name``, ``delta_f3``, ``logM``, ``e_env``.
        Index reset to 0-based consecutive integers.
    """
    df = f3.merge(sparc, on="galaxy_name", how="inner")
    df = df.merge(chae, on="galaxy_name", how="inner")
    df = df.dropna(subset=["galaxy_name", "delta_f3", "logM", "e_env"])
    df = df.reset_index(drop=True)
    return df[["galaxy_name", "delta_f3", "logM", "e_env"]]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_output(df: pd.DataFrame, out_path: str | Path) -> None:
    """Write the crossmatched table to *out_path*.

    Parent directories are created if they do not exist.

    Parameters
    ----------
    df : pd.DataFrame
        Crossmatched table.
    out_path : str | Path
        Destination CSV path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge F3, SPARC, and Chae environment catalogs into "
            "the env-real analysis input table."
        )
    )
    parser.add_argument(
        "--f3-catalog",
        default=DEFAULT_F3_CATALOG,
        metavar="FILE",
        help=f"F3 catalog CSV (default: {DEFAULT_F3_CATALOG}).",
    )
    parser.add_argument(
        "--sparc-basic",
        default=DEFAULT_SPARC_BASIC,
        metavar="FILE",
        help=f"SPARC global table CSV (default: {DEFAULT_SPARC_BASIC}).",
    )
    parser.add_argument(
        "--chae-env",
        default=DEFAULT_CHAE_ENV,
        metavar="FILE",
        help=f"Chae environment catalog CSV (default: {DEFAULT_CHAE_ENV}).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        metavar="FILE",
        help=f"Output CSV path (default: {DEFAULT_OUT}).",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    f3_catalog: str | None = None,
    sparc_basic: str | None = None,
    chae_env: str | None = None,
    out: str | None = None,
) -> pd.DataFrame:
    """Entry point: load, merge, save, and print summary.

    Can be called either with a CLI-style argument list::

        main(["--f3-catalog", "f3.csv", "--sparc-basic", "sparc.csv",
              "--chae-env", "chae.csv", "--out", "merged.csv"])

    or with keyword arguments::

        main(f3_catalog="f3.csv", sparc_basic="sparc.csv",
             chae_env="chae.csv", out="merged.csv")

    Keyword arguments take precedence over *argv* for any parameter they
    specify.  Parameters not provided via keywords fall back to *argv* /
    the argparse defaults.
    """
    args = _parse_args([] if argv is None and any(
        v is not None for v in (f3_catalog, sparc_basic, chae_env, out)
    ) else argv)
    if f3_catalog is not None:
        args.f3_catalog = f3_catalog
    if sparc_basic is not None:
        args.sparc_basic = sparc_basic
    if chae_env is not None:
        args.chae_env = chae_env
    if out is not None:
        args.out = out

    f3 = load_f3_catalog(args.f3_catalog)
    sparc = load_sparc_basic(args.sparc_basic)
    chae = load_chae_env(args.chae_env)

    df = build_crossmatch(f3, sparc, chae)
    save_output(df, args.out)

    print(f"OK: {args.out}")
    print(f"N = {len(df)}")
    print(df.head())
    return df


if __name__ == "__main__":
    main()
