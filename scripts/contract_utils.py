"""
contract_utils.py

Shared utilities for the SCM internal data contract:
- ensure_dir: create output directories
- validate_galaxies_df: validate galaxies.parquet schema
- validate_rc_points_df: validate rc_points.parquet schema
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


GALAXIES_REQUIRED = ["galaxy_id"]
RC_POINTS_REQUIRED = ["galaxy_id", "r_kpc", "vrot_kms"]


@dataclass
class ValidationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)


def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_galaxies_df(df: pd.DataFrame) -> ValidationResult:
    """Validate that *df* meets the galaxies contract (galaxy_id required)."""
    result = ValidationResult()
    for col in GALAXIES_REQUIRED:
        if col not in df.columns:
            result.ok = False
            result.errors.append(f"galaxies: missing required column '{col}'")
    if result.ok and df["galaxy_id"].isnull().any():
        result.ok = False
        result.errors.append("galaxies: 'galaxy_id' contains null values")
    return result


def validate_rc_points_df(df: pd.DataFrame) -> ValidationResult:
    """Validate that *df* meets the rc_points contract."""
    result = ValidationResult()
    for col in RC_POINTS_REQUIRED:
        if col not in df.columns:
            result.ok = False
            result.errors.append(f"rc_points: missing required column '{col}'")
    if result.ok and df["galaxy_id"].isnull().any():
        result.ok = False
        result.errors.append("rc_points: 'galaxy_id' contains null values")
    return result


def compute_vbar_kms(df_rc: pd.DataFrame) -> pd.Series:
    """Compute vbar_kms from components (vstar_kms, vgas_kms) in quadrature.

    Returns *df_rc['vbar_kms']* if it already exists, otherwise combines
    available components: vbar = sqrt(vstar² + vgas²).
    """
    if "vbar_kms" in df_rc.columns:
        return df_rc["vbar_kms"].astype(float)

    v2 = np.zeros(len(df_rc), dtype=float)
    found = False
    for c in ["vstar_kms", "vgas_kms"]:
        if c in df_rc.columns:
            found = True
            v2 += np.square(df_rc[c].astype(float).to_numpy())

    if not found:
        raise ValueError(
            "Cannot compute vbar_kms: no vbar_kms and no component columns "
            "(vstar_kms, vgas_kms) present."
        )
    return pd.Series(np.sqrt(v2), index=df_rc.index, name="vbar_kms")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file based on its extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv, .parquet, or .pq)")
