from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Missing table: {table_path}")
    if table_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def compute_vbar_kms(df: pd.DataFrame) -> pd.Series:
    return np.sqrt(
        np.clip(
            df["vgas_kms"].to_numpy(dtype=float) ** 2
            + df["vdisk_kms"].to_numpy(dtype=float) ** 2
            + df["vbul_kms"].to_numpy(dtype=float) ** 2,
            a_min=0.0,
            a_max=None,
        )
    )


def validate_contract(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")
