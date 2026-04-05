"""
scripts/build_env_real_input.py — Build the environment–real analysis input CSV.

Merges three data sources into a single analysis-ready CSV suitable for
downstream environmental-correlation analyses (e.g. ``analyze_env_real_merged.py``):

  1. **F3 catalog** — per-galaxy deep-regime slope β (columns: ``galaxy``, ``F3``
     or ``beta``).
  2. **SPARC basic table** — photometric and HI data (columns: ``Galaxy``,
     ``L36``, ``MHI``).
  3. **Chae environment catalog** — environmental proxy (columns:
     ``galaxy_name``, ``e_env``; optionally ``e_env_err``).

Output columns
--------------
galaxy_name
    Normalised galaxy identifier (lower-case, spaces stripped).
delta_f3
    β − 0.5  (deviation from the MOND deep-regime expectation).
logM
    log₁₀(0.5 × L₃₆ × 10⁹  +  1.33 × M_HI × 10⁹)
    — log₁₀ of the baryonic mass proxy in solar masses.
e_env
    Environment proxy from Chae et al.
e_env_err  (optional)
    Uncertainty on ``e_env``, included when present in the Chae input.

Usage
-----
CLI::

    python scripts/build_env_real_input.py \\
        --f3-catalog  results/f3_catalog_real.csv \\
        --sparc-basic data/sparc_table1.csv \\
        --chae-env    data/chae_env.csv \\
        --out         results/env_real/sparc_f3_chae_merged.csv

Programmatic (keyword API)::

    from scripts.build_env_real_input import main as build_main

    build_main(
        f3_catalog="path/to/f3_catalog.csv",
        sparc_basic="path/to/sparc_table1.csv",
        chae_env="path/to/chae_env.csv",
        out="results/env_real/sparc_f3_chae_merged.csv",
    )
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: MOND deep-regime reference slope; delta_f3 = F3 − MOND_REF
MOND_REF: float = 0.5

#: Stellar mass-to-light ratio for 3.6 µm band (Schombert & McGaugh 2014)
UPSILON_36: float = 0.5

#: HI-to-neutral-gas correction factor (accounts for He and metals)
ALPHA_HI: float = 1.33


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def clean_name(name: str) -> str:
    """Return a normalised galaxy name for cross-catalogue matching.

    Transformation applied (in order):
    1. Strip leading/trailing whitespace.
    2. Convert to lower-case.
    3. Remove all whitespace, hyphens, underscores, and dots
       (common variant separators in galaxy names).

    Parameters
    ----------
    name : str
        Raw galaxy name as it appears in a catalogue.

    Returns
    -------
    str
        Normalised name string.
    """
    s = str(name).strip().lower()
    s = re.sub(r"[-_.\s]", "", s)
    return s


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_f3_catalog(path: str | Path) -> pd.DataFrame:
    """Load the F3 per-galaxy slope catalog.

    Accepts either a ``beta`` column (from ``generate_f3_catalog.py``) or an
    ``F3`` column (alternate naming used in some analyses).  The result always
    carries a column named ``F3``.

    Required source columns (one of): ``galaxy`` + ``F3``  |  ``galaxy`` + ``beta``

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``galaxy`` (str) and ``F3`` (float) columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"F3 catalog not found: {path}")
    df = pd.read_csv(path)
    if "galaxy" not in df.columns:
        raise ValueError(
            f"F3 catalog {path} must contain a 'galaxy' column; "
            f"found: {list(df.columns)}"
        )
    if "F3" not in df.columns and "beta" not in df.columns:
        raise ValueError(
            f"F3 catalog {path} must contain an 'F3' or 'beta' column; "
            f"found: {list(df.columns)}"
        )
    if "F3" not in df.columns:
        df = df.copy()
        df["F3"] = df["beta"]
    return df[["galaxy", "F3"]].copy()


def load_sparc_basic(path: str | Path) -> pd.DataFrame:
    """Load the SPARC basic photometric / HI table.

    Required source columns: ``Galaxy``, ``L36``, ``MHI``.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Galaxy`` (str), ``L36`` (float), ``MHI`` (float).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SPARC basic table not found: {path}")
    df = pd.read_csv(path)
    required = {"Galaxy", "L36", "MHI"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"SPARC basic table {path} missing columns: {sorted(missing)}; "
            f"found: {list(df.columns)}"
        )
    return df[["Galaxy", "L36", "MHI"]].copy()


def load_chae_env(path: str | Path) -> pd.DataFrame:
    """Load the Chae environment catalog.

    Required source columns: ``galaxy_name``, ``e_env``.
    Optional source column: ``e_env_err`` (retained when present).

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``galaxy_name``, ``e_env`` and (if present)
        ``e_env_err``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chae env catalog not found: {path}")
    df = pd.read_csv(path)
    required = {"galaxy_name", "e_env"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Chae env catalog {path} missing columns: {sorted(missing)}; "
            f"found: {list(df.columns)}"
        )
    cols = ["galaxy_name", "e_env"]
    if "e_env_err" in df.columns:
        cols.append("e_env_err")
    return df[cols].copy()


# ---------------------------------------------------------------------------
# Mass proxy
# ---------------------------------------------------------------------------

def compute_logM(L36: np.ndarray, MHI: np.ndarray) -> np.ndarray:
    """Compute the log₁₀ baryonic mass proxy.

    Formula::

        M_bar = UPSILON_36 × L36 × 10⁹  +  ALPHA_HI × MHI × 10⁹
        logM  = log₁₀(M_bar)

    where ``L36`` is the 3.6 µm luminosity in units of 10⁹ L_sun and
    ``MHI`` is the HI mass in units of 10⁹ M_sun.

    Parameters
    ----------
    L36 : array_like
        3.6 µm luminosity (10⁹ L_sun units, as stored in SPARC table).
    MHI : array_like
        HI mass (10⁹ M_sun units, as stored in SPARC table).

    Returns
    -------
    np.ndarray
        log₁₀ of the baryonic mass in solar masses.
    """
    L36 = np.asarray(L36, dtype=float)
    MHI = np.asarray(MHI, dtype=float)
    M_bar = UPSILON_36 * L36 * 1e9 + ALPHA_HI * MHI * 1e9
    return np.log10(np.maximum(M_bar, 1.0))


# ---------------------------------------------------------------------------
# Merge pipeline
# ---------------------------------------------------------------------------

def merge_catalogs(
    df_f3: pd.DataFrame,
    df_sparc: pd.DataFrame,
    df_chae: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join the three catalogs on normalised galaxy names.

    Name normalisation is performed via :func:`clean_name` before merging so
    that minor typographic differences (case, hyphens, spaces) do not cause
    spurious mismatches.

    Parameters
    ----------
    df_f3 : pd.DataFrame
        Output of :func:`load_f3_catalog`.
    df_sparc : pd.DataFrame
        Output of :func:`load_sparc_basic`.
    df_chae : pd.DataFrame
        Output of :func:`load_chae_env`.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        ``galaxy_name``, ``delta_f3``, ``logM``, ``e_env``
        (plus ``e_env_err`` when present in *df_chae*).
    """
    f3 = df_f3.copy()
    sparc = df_sparc.copy()
    chae = df_chae.copy()

    f3["_key"] = f3["galaxy"].map(clean_name)
    sparc["_key"] = sparc["Galaxy"].map(clean_name)
    chae["_key"] = chae["galaxy_name"].map(clean_name)

    # F3 ⊕ SPARC (inner join on _key)
    merged = pd.merge(f3, sparc, on="_key", how="inner")

    # ⊕ Chae (inner join on _key)
    merged = pd.merge(merged, chae, on="_key", how="inner")

    # Derived columns
    merged["delta_f3"] = merged["F3"] - MOND_REF
    merged["logM"] = compute_logM(merged["L36"].values, merged["MHI"].values)
    merged["galaxy_name"] = merged["_key"]

    out_cols = ["galaxy_name", "delta_f3", "logM", "e_env"]
    if "e_env_err" in merged.columns:
        out_cols.append("e_env_err")

    return merged[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------

def build_env_real_input(
    f3_catalog: str | Path,
    sparc_basic: str | Path,
    chae_env: str | Path,
    out: str | Path,
) -> pd.DataFrame:
    """Full pipeline: load → merge → save.

    Parameters
    ----------
    f3_catalog : str or Path
        F3 per-galaxy slope catalog CSV.
    sparc_basic : str or Path
        SPARC basic photometric / HI table CSV.
    chae_env : str or Path
        Chae environment catalog CSV.
    out : str or Path
        Output CSV path.  Parent directory is created if necessary.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame written to *out*.
    """
    df_f3 = load_f3_catalog(f3_catalog)
    df_sparc = load_sparc_basic(sparc_basic)
    df_chae = load_chae_env(chae_env)

    df_out = merge_catalogs(df_f3, df_sparc, df_chae)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)

    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge F3 catalog + SPARC basic table + Chae env catalog into "
            "a single analysis-ready CSV (galaxy_name, delta_f3, logM, e_env)."
        )
    )
    parser.add_argument(
        "--f3-catalog", dest="f3_catalog", default=None,
        help="F3 per-galaxy slope catalog CSV (columns: galaxy, F3 or beta).",
    )
    parser.add_argument(
        "--sparc-basic", dest="sparc_basic", default=None,
        help="SPARC basic table CSV (columns: Galaxy, L36, MHI).",
    )
    parser.add_argument(
        "--chae-env", dest="chae_env", default=None,
        help="Chae environment catalog CSV (columns: galaxy_name, e_env).",
    )
    parser.add_argument(
        "--out", default="results/env_real/sparc_f3_chae_merged.csv",
        help=(
            "Output CSV path "
            "(default: results/env_real/sparc_f3_chae_merged.csv)."
        ),
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
    """Entry point for CLI and programmatic use.

    Keyword arguments take precedence over parsed *argv* values.  When any
    keyword argument is supplied and *argv* is ``None``, argparse receives an
    empty list (``[]``) so that ``sys.argv`` is not inadvertently consumed.

    Parameters
    ----------
    argv : list[str] or None
        Command-line argument list (passed to :func:`argparse.parse_args`).
        Pass ``[]`` or omit to use only keyword arguments.
    f3_catalog, sparc_basic, chae_env, out : str or None
        Keyword overrides for the corresponding CLI options.

    Returns
    -------
    pd.DataFrame
        Merged output DataFrame.

    Raises
    ------
    ValueError
        If any required path is still ``None`` after merging argv + kwargs.
    """
    kwargs_provided = any(
        x is not None for x in [f3_catalog, sparc_basic, chae_env, out]
    )
    if kwargs_provided and argv is None:
        argv = []

    args = _parse_args(argv)

    f3_path = f3_catalog if f3_catalog is not None else args.f3_catalog
    sparc_path = sparc_basic if sparc_basic is not None else args.sparc_basic
    chae_path = chae_env if chae_env is not None else args.chae_env
    out_path = out if out is not None else args.out

    missing = [
        name
        for name, val in [
            ("--f3-catalog", f3_path),
            ("--sparc-basic", sparc_path),
            ("--chae-env", chae_path),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Required argument(s) not provided: {', '.join(missing)}"
        )

    return build_env_real_input(f3_path, sparc_path, chae_path, out_path)


if __name__ == "__main__":
    main()
