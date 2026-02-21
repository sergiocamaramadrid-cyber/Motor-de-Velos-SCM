"""I/O utilities — load SPARC-style rotation-curve CSV files."""

import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

#: Required columns in every rotation-curve CSV
REQUIRED_COLUMNS = {"r", "Vobs", "eVobs", "Vdisk", "Vgas"}

#: Optional columns (set to zeros when absent)
OPTIONAL_COLUMNS = {"Vbul": 0.0}


def load_rotation_curve(
    path: Union[str, Path],
    min_points: int = 4,
) -> pd.DataFrame:
    """Load a SPARC-style rotation-curve CSV and validate its contents.

    Expected columns (case-sensitive)
    ----------------------------------
    r      : galactocentric radius [kpc]
    Vobs   : observed rotation velocity [km s⁻¹]
    eVobs  : measurement uncertainty on Vobs [km s⁻¹]
    Vdisk  : stellar-disk contribution at Υ_disk = 1 [km s⁻¹]
    Vgas   : gas contribution (face value) [km s⁻¹]
    Vbul   : bulge contribution at Υ_bul = 1 [km s⁻¹]  (optional, default 0)

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    min_points : int, optional
        Minimum number of data rows required (default 4).

    Returns
    -------
    df : pd.DataFrame
        Validated DataFrame with at least the required columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing, data are non-finite, or there are
        fewer than *min_points* rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rotation-curve file not found: {path}")

    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{path.name}' is missing required columns: {sorted(missing)}"
        )

    # Fill optional columns with defaults when absent
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    # Drop rows with non-finite values in key columns
    key_cols = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)
    df = df.dropna(subset=key_cols)
    finite_mask = np.all(np.isfinite(df[key_cols].values), axis=1)
    df = df[finite_mask].reset_index(drop=True)

    if len(df) < min_points:
        raise ValueError(
            f"CSV '{path.name}' has only {len(df)} valid rows "
            f"(minimum required: {min_points})."
        )

    # Positive-definite sanity checks
    if (df["r"] <= 0).any():
        raise ValueError("Column 'r' must be strictly positive.")
    if (df["eVobs"] <= 0).any():
        raise ValueError("Column 'eVobs' must be strictly positive.")

    return df


def galaxy_name_from_path(path: Union[str, Path]) -> str:
    """Return the galaxy name derived from the CSV filename (stem)."""
    return Path(path).stem
