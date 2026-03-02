from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def compute_vbar_kms(rc_points: pd.DataFrame) -> pd.Series:
    required = {"vgas_kms", "vdisk_kms", "vbul_kms"}
    missing = required - set(rc_points.columns)
    if missing:
        raise ValueError(f"Missing component columns for vbar_kms: {sorted(missing)}")
    return np.sqrt(
        rc_points["vgas_kms"] ** 2
        + rc_points["vdisk_kms"] ** 2
        + rc_points["vbul_kms"] ** 2
    )


def validate_contract(galaxies: pd.DataFrame, rc_points: pd.DataFrame) -> None:
    gal_missing = {"galaxy"} - set(galaxies.columns)
    if gal_missing:
        raise ValueError(f"galaxies table missing required columns: {sorted(gal_missing)}")

    required_rc = {"galaxy", "r_kpc", "v_obs_kms"}
    rc_missing = required_rc - set(rc_points.columns)
    if rc_missing:
        raise ValueError(f"rc_points table missing required columns: {sorted(rc_missing)}")

    if "vbar_kms" not in rc_points.columns:
        comp_missing = {"vgas_kms", "vdisk_kms", "vbul_kms"} - set(rc_points.columns)
        if comp_missing:
            raise ValueError(
                "rc_points must include vbar_kms or all component columns: "
                f"{sorted(comp_missing)}"
            )

    galaxies_set = set(galaxies["galaxy"].dropna().astype(str))
    rc_set = set(rc_points["galaxy"].dropna().astype(str))
    extra = sorted(rc_set - galaxies_set)
    if extra:
        raise ValueError(f"rc_points contains galaxies missing in galaxies table: {extra[:5]}")
