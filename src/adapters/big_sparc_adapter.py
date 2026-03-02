"""
big_sparc_adapter.py — BIG-SPARC multi-format adapter.

BIG-SPARC (~4000 galaxies) may arrive in several formats:

    A) Many individual *_rotmod.dat files (standard SPARC format)
    B) An .mrt (AAS machine-readable table) or .csv master catalog
    C) A single mega-catalog parquet/CSV already assembled

This adapter auto-detects the format from the input path and normalises
every format into the SCM internal contract (galaxies.parquet + rc_points.parquet).

All formats share the same column-alias map and baryonic-velocity derivation
logic as sparc_adapter.py.  Parallel ingestion is supported for format A via
:mod:`concurrent.futures` (stdlib).

Public API
----------
    detect_format(path)           — returns 'rotmod_dir' | 'catalog_file'
    ingest_big_sparc(path)        — auto-dispatch to the right reader
    ingest_rotmod_dir(...)        — delegates to sparc_adapter.ingest_sparc_dir
    ingest_catalog_file(...)      — reads single .mrt/.csv/.parquet table
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .sparc_adapter import CANDIDATE_COLS, _pick_col, ingest_sparc_dir

import numpy as np

# ---------------------------------------------------------------------------
# Additional BIG-SPARC column aliases (super-set of SPARC)
# ---------------------------------------------------------------------------

_BIG_SPARC_EXTRA: Dict[str, List[str]] = {
    "galaxy_id": [
        "Galaxy", "galaxy", "Name", "name", "ID", "id",
        "galaxy_id", "GalName",
    ],
    "r":         ["Rad", "R", "r", "r_kpc", "radius", "Radius"],
    "vobs":      ["Vobs", "Vrot", "vobs", "vrot", "Vcirc", "vcirc"],
    "e_vobs":    ["eVobs", "eVrot", "dVobs", "evobs", "err_vobs"],
    "vgas":      ["Vgas", "vgas", "V_gas"],
    "vdisk":     ["Vdisk", "vdisk", "V_disk", "Vstars", "vstars"],
    "vbul":      ["Vbul", "vbul", "V_bul", "Vbulge"],
    "vbar":      ["Vbar", "vbar", "V_bar"],
}


def detect_format(path: Path) -> str:
    """Detect the BIG-SPARC input format.

    Parameters
    ----------
    path : Path
        Either a directory (→ 'rotmod_dir') or a file (→ 'catalog_file').

    Returns
    -------
    str
        ``'rotmod_dir'`` or ``'catalog_file'``.
    """
    path = Path(path)
    if path.is_dir():
        return "rotmod_dir"
    if path.is_file():
        return "catalog_file"
    raise FileNotFoundError(f"Path does not exist: {path}")


def _rc_from_wide(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a wide-format BIG-SPARC catalog table to the SCM contract.

    Assumes one row = one (galaxy, radius) point.  The galaxy-ID column is
    matched using `_BIG_SPARC_EXTRA['galaxy_id']` aliases.
    """
    alias = {**CANDIDATE_COLS, **_BIG_SPARC_EXTRA}

    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    c_gid  = pick(_BIG_SPARC_EXTRA["galaxy_id"])
    c_r    = pick(alias["r"])
    c_vobs = pick(alias["vobs"])
    c_e    = pick(alias["e_vobs"])
    c_vgas = pick(alias["vgas"])
    c_vdisk = pick(alias["vdisk"])
    c_vbul  = pick(alias["vbul"])
    c_vbar  = pick(alias["vbar"])

    if c_gid is None:
        raise ValueError(
            "Cannot find galaxy-ID column.  "
            f"Tried: {_BIG_SPARC_EXTRA['galaxy_id']}.  "
            f"Columns present: {list(df.columns)}"
        )
    if c_r is None or c_vobs is None:
        raise ValueError(
            "Cannot find radius/velocity columns in catalog table.  "
            f"Columns present: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "galaxy_id": df[c_gid].astype(str),
        "r_kpc":     pd.to_numeric(df[c_r],    errors="coerce"),
        "vrot_kms":  pd.to_numeric(df[c_vobs], errors="coerce"),
    })

    if c_e is not None:
        out["vrot_err_kms"] = pd.to_numeric(df[c_e], errors="coerce")

    if c_vbar is not None:
        out["vbar_kms"] = pd.to_numeric(df[c_vbar], errors="coerce")
    else:
        if c_vgas is not None:
            out["vgas_kms"] = pd.to_numeric(df[c_vgas], errors="coerce")
        v2_star = None
        if c_vdisk is not None:
            vd = pd.to_numeric(df[c_vdisk], errors="coerce")
            v2_star = vd ** 2 if v2_star is None else v2_star + vd ** 2
        if c_vbul is not None:
            vb = pd.to_numeric(df[c_vbul], errors="coerce")
            v2_star = vb ** 2 if v2_star is None else v2_star + vb ** 2
        if v2_star is not None:
            out["vstar_kms"] = np.sqrt(v2_star)

    galaxies = (
        pd.DataFrame({"galaxy_id": out["galaxy_id"].unique()})
        .sort_values("galaxy_id")
        .reset_index(drop=True)
    )
    rc = out.sort_values(["galaxy_id", "r_kpc"]).reset_index(drop=True)
    return galaxies, rc


def ingest_catalog_file(
    path: Path,
    sep: str = r"\s+",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest a single BIG-SPARC catalog file (.mrt, .csv, or .parquet).

    Parameters
    ----------
    path : Path
        Path to the catalog file.  Supported extensions:
        ``.parquet``, ``.csv``, ``.txt``, ``.mrt`` (whitespace-separated).
    sep : str
        Column separator for text files (default: whitespace regex).

    Returns
    -------
    (galaxies_df, rc_df)
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".csv",):
        df = pd.read_csv(path, comment="#")
    else:
        # .mrt, .txt, or unknown — try whitespace-separated
        df = pd.read_csv(path, sep=sep, comment="#", engine="python")

    if df.empty:
        raise ValueError(f"Empty catalog file: {path}")

    return _rc_from_wide(df)


def ingest_big_sparc(
    path: Path,
    pattern: str = "*_rotmod.dat",
    workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Auto-detect format and ingest BIG-SPARC data.

    Parameters
    ----------
    path : Path
        Directory of rotmod files **or** single catalog file.
    pattern : str
        Glob pattern used when *path* is a directory (default ``*_rotmod.dat``).
    workers : int
        Parallel workers for rotmod-directory ingestion (default 1).

    Returns
    -------
    (galaxies_df, rc_df)
        Contract-compliant DataFrames.
    """
    fmt = detect_format(path)
    if fmt == "rotmod_dir":
        return ingest_sparc_dir(path, pattern=pattern, workers=workers)
    return ingest_catalog_file(path)
