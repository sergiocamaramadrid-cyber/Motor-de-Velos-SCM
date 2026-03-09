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
    if v_bul is None:
        v_bul = 0.0

    vg = np.asarray(v_gas, dtype=float)
    vd = np.asarray(v_disk, dtype=float)
    vb = np.asarray(v_bul, dtype=float)
    out = np.sqrt(np.clip(vg**2 + vd**2 + vb**2, a_min=0.0, a_max=None))
    return out


def validate_contract(df: pd.DataFrame, source: str = "<unknown>") -> None:
    missing = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Contract violation in '{source}': missing columns {missing}")
