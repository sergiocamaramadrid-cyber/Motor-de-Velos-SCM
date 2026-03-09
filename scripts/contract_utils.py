from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_COLUMNS = [
    "galaxy",
    "source_file",
    "n_points_curve",
    "rmax_kpc",
    "vmax_obs_kms",
    "tail_frac",
    "n_tail_points",
    "F3_SCM",
    "delta_f3",
    "beta",
    "n_beta_points",
    "logSigmaHI_out",
    "logSigmaHI_out_proxy",
    "quality_flag_tail_ok",
    "quality_flag_beta_ok",
    "contract_version",
]

LEGACY_CONTRACT_COLUMNS = [
    "galaxy",
    "r_kpc",
    "vobs_kms",
    "vobs_err_kms",
    "vbar_kms",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_contract(df: pd.DataFrame, source: str = "<unknown>") -> pd.DataFrame:
    missing_master = [c for c in CONTRACT_COLUMNS if c not in df.columns]
    if not missing_master:
        return df.loc[:, CONTRACT_COLUMNS].copy()

    missing_legacy = [c for c in LEGACY_CONTRACT_COLUMNS if c not in df.columns]
    if not missing_legacy:
        return df.loc[:, LEGACY_CONTRACT_COLUMNS].copy()

    raise ValueError(
        f"Contrato roto en '{source}'. Faltan columnas master: {missing_master}; "
        f"faltan columnas legacy: {missing_legacy}"
    )


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
    return np.asarray(out, dtype=float)
