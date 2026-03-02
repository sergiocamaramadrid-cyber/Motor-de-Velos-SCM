"""
scripts/contract_utils.py — Shared utilities for SCM contract-based pipelines.

Provides:
  read_table(path)         — Format-agnostic table loader (CSV or Parquet).
  compute_vbar_kms(row)    — Baryonic velocity (km/s) via quadrature from
                             v_gas, v_disk, and optional v_bul columns.

These routines are reused by both the SPARC and BIG-SPARC ingestion paths
and by any downstream catalog generator that reads contract-compliant tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SCM data-contract column names
# ---------------------------------------------------------------------------

#: Minimum set of columns that every ingestor's output must contain.
CONTRACT_COLUMNS: list[str] = [
    "galaxy",
    "r_kpc",
    "vobs_kms",
    "vobs_err_kms",
    "vbar_kms",
]


# ---------------------------------------------------------------------------
# read_table — format-agnostic loader
# ---------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    """Load a tabular data file regardless of its on-disk format.

    Supported formats (selected by file extension):
      * ``.parquet`` — Apache Parquet (requires *pyarrow* ≥ 14.0.1)
      * anything else — plain CSV (``pd.read_csv``)

    Parameters
    ----------
    path : str or Path
        Path to the data file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# compute_vbar_kms — baryonic velocity via quadrature
# ---------------------------------------------------------------------------

def compute_vbar_kms(
    v_gas: float | np.ndarray,
    v_disk: float | np.ndarray,
    v_bul: float | np.ndarray | None = None,
) -> np.ndarray:
    """Compute the total baryonic circular velocity via quadrature.

    The baryonic velocity is defined as:

        V_bar = sign(v_disk) · √( v_gas² + v_disk² + v_bul² )

    where each component enters as its *signed* value (the sign of v_disk
    propagates to the total, following the SPARC convention).

    Parameters
    ----------
    v_gas : array_like
        Gas contribution to the circular velocity (km/s).  May be negative
        (falling rotation curve at large radii) but is squared internally.
    v_disk : array_like
        Disk stellar contribution (km/s).  Its sign is used to set the sign
        of the returned V_bar.
    v_bul : array_like or None
        Bulge contribution (km/s).  Defaults to zero when *None*.

    Returns
    -------
    ndarray
        V_bar in km/s, same shape as the broadcast of the inputs.
    """
    v_gas = np.asarray(v_gas, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    v_bul = np.zeros_like(v_gas) if v_bul is None else np.asarray(v_bul, dtype=float)

    vbar_sq = v_gas**2 + v_disk**2 + v_bul**2
    sign = np.sign(v_disk)
    # When v_disk == 0 treat sign as +1 to avoid zero output for non-zero gas/bul
    sign = np.where(sign == 0.0, 1.0, sign)
    return sign * np.sqrt(vbar_sq)


# ---------------------------------------------------------------------------
# validate_contract — check that a DataFrame satisfies the column contract
# ---------------------------------------------------------------------------

def validate_contract(df: pd.DataFrame, source: str = "<unknown>") -> None:
    """Raise *ValueError* if *df* is missing any required contract column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    source : str
        Descriptive label used in the error message (e.g. a file path).

    Raises
    ------
    ValueError
        If any column listed in :data:`CONTRACT_COLUMNS` is absent.
    """
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Contract violation in '{source}': missing columns {missing}"
        )
