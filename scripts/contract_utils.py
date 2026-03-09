"""
scripts/contract_utils.py — Shared utilities for SCM contract-based pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_COLUMNS: list[str] = [
    "galaxy",
    "r_kpc",
    "vobs_kms",
    "vobs_err_kms",
    "vbar_kms",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_vbar_kms(
    v_gas: float | np.ndarray,
    v_disk: float | np.ndarray,
    v_bul: float | np.ndarray | None = None,
) -> np.ndarray:
    v_gas = np.asarray(v_gas, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    if v_bul is None:
        v_bul = 0.0
    v_bul = np.asarray(v_bul, dtype=float)

    return np.asarray(np.sqrt(v_gas**2 + v_disk**2 + v_bul**2), dtype=float)


def validate_contract(df: pd.DataFrame, source: str = "<unknown>") -> None:
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Contract violation in '{source}': missing columns {missing}")
