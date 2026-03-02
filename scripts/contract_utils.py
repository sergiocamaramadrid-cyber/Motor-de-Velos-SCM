"""
scripts/contract_utils.py — Shared data-contract utilities for SCM pipeline v2.

Provides:
  - REQUIRED_COLS: minimum column set for a valid SCM contract table.
  - validate_contract(df): raises ValueError on missing columns.
  - read_table(path): read CSV or Parquet into a DataFrame.
  - compute_vbar_kms(df): derive vbar_kms from component velocity columns when
    the column is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Contract definition
# ---------------------------------------------------------------------------

REQUIRED_COLS: list[str] = [
    "galaxy",
    "r_kpc",
    "vobs_kms",
    "vobs_err_kms",
    "vbar_kms",
]

# Component columns used to derive vbar_kms when it is missing.
_VBAR_COMPONENTS: list[str] = ["vgas_kms", "vdisk_kms", "vbul_kms"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def validate_contract(df: pd.DataFrame) -> None:
    """Raise ValueError if *df* is missing any column from REQUIRED_COLS."""
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data contract violation — missing columns: {sorted(missing)}"
        )


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    path:
        File path.  Parquet is assumed when the suffix is ``.parquet`` or
        ``.pq``; CSV otherwise.

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def compute_vbar_kms(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with ``vbar_kms`` populated.

    If ``vbar_kms`` already exists, the DataFrame is returned unchanged.
    Otherwise it is derived as the quadrature sum of the component columns
    ``vgas_kms``, ``vdisk_kms``, and ``vbul_kms`` (any missing component is
    treated as zero).

    Parameters
    ----------
    df:
        Input DataFrame (not modified in-place).

    Returns
    -------
    pd.DataFrame with ``vbar_kms`` present.
    """
    if "vbar_kms" in df.columns:
        return df

    df = df.copy()
    v2 = np.zeros(len(df))
    for col in _VBAR_COMPONENTS:
        if col in df.columns:
            v2 += df[col].fillna(0.0).to_numpy() ** 2
    df["vbar_kms"] = np.sqrt(v2)
    return df
