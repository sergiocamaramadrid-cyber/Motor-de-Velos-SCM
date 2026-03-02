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
