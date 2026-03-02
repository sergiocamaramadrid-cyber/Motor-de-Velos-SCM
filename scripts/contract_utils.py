#!/usr/bin/env python3
"""
contract_utils.py

Define y valida el contrato interno del Framework para catálogos masivos
(BIG-SPARC, SPARC convertido, futuros surveys).

Tablas:
- galaxies: 1 fila por galaxia
- rc_points: múltiples filas por galaxia (curva de rotación / puntos)

Unidades:
- r_kpc en kpc
- vrot_kms en km/s
- vbar_kms en km/s (componente bariónica equivalente; si no existe, no se puede calcular g_bar)

El core de SCM calcula:
g_obs = (vrot^2)/r
g_bar = (vbar^2)/r
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


GALAXY_REQUIRED_COLS = [
    "galaxy_id",
]

RC_REQUIRED_COLS = [
    "galaxy_id",
    "r_kpc",
    "vrot_kms",
]

# Para calcular g_bar necesitamos vbar_kms (o componentes que se sumen en cuadratura).
RC_GBAR_OPTIONAL_COLS = [
    "vbar_kms",     # preferido
    "vstar_kms",    # opcional
    "vgas_kms",     # opcional
]


@dataclass(frozen=True)
class ContractValidationResult:
    ok: bool
    errors: List[str]


def _missing_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]


def validate_galaxies_df(df: pd.DataFrame) -> ContractValidationResult:
    errors: List[str] = []

    missing = _missing_cols(df, GALAXY_REQUIRED_COLS)
    if missing:
        errors.append(f"galaxies missing columns: {missing}")

    if "galaxy_id" in df.columns:
        if df["galaxy_id"].isna().any():
            errors.append("galaxies has NaN galaxy_id")
        if df["galaxy_id"].duplicated().any():
            errors.append("galaxies has duplicated galaxy_id (must be unique per row)")

    return ContractValidationResult(ok=(len(errors) == 0), errors=errors)


def validate_rc_points_df(df: pd.DataFrame) -> ContractValidationResult:
    errors: List[str] = []

    missing = _missing_cols(df, RC_REQUIRED_COLS)
    if missing:
        errors.append(f"rc_points missing columns: {missing}")

    if "galaxy_id" in df.columns and df["galaxy_id"].isna().any():
        errors.append("rc_points has NaN galaxy_id")

    for c in ["r_kpc", "vrot_kms"]:
        if c in df.columns:
            if (df[c].isna()).any():
                errors.append(f"rc_points has NaN in {c}")
            if (df[c] <= 0).any():
                errors.append(f"rc_points has non-positive values in {c} (must be > 0)")

    has_vbar = ("vbar_kms" in df.columns)
    has_components = ("vstar_kms" in df.columns) or ("vgas_kms" in df.columns)

    if not has_vbar and not has_components:
        errors.append(
            "rc_points lacks vbar_kms and lacks (vstar_kms/vgas_kms). "
            "Need vbar_kms OR components to compute g_bar."
        )

    return ContractValidationResult(ok=(len(errors) == 0), errors=errors)


def compute_vbar_kms(df_rc: pd.DataFrame) -> pd.Series:
    """
    Compute vbar_kms:
    - If vbar_kms exists, use it.
    - Else use quadrature of available components (vstar_kms, vgas_kms).
    """
    if "vbar_kms" in df_rc.columns:
        return df_rc["vbar_kms"].astype(float)

    v2 = np.zeros(len(df_rc), dtype=float)
    found = False

    for c in ["vstar_kms", "vgas_kms"]:
        if c in df_rc.columns:
            found = True
            v = df_rc[c].astype(float).to_numpy()
            v2 += np.square(v)

    if not found:
        raise ValueError("Cannot compute vbar_kms: no vbar_kms and no component columns present.")

    return pd.Series(np.sqrt(v2), index=df_rc.index, name="vbar_kms")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    """
    Read CSV or Parquet based on extension.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")
