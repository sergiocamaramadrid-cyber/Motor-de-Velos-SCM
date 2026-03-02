"""
sparc_adapter.py — SPARC *_rotmod.dat → SCM contract adapter.

Wraps the ingestion logic from ``scripts/ingest_sparc_contract.py`` into a
reusable, importable module.  Supports parallel ingestion over many files
using :mod:`concurrent.futures.ProcessPoolExecutor` (stdlib, no extra deps).

Public API
----------
    read_rotmod(path)             — parse one *_rotmod.dat → (galaxy_id, rc_df)
    ingest_sparc_dir(sparc_dir)   — read all *_rotmod.dat → (galaxies_df, rc_df)

Contract output columns
-----------------------
    galaxies_df : galaxy_id
    rc_df       : galaxy_id, r_kpc, vrot_kms, [vrot_err_kms],
                  [vgas_kms], [vstar_kms] or [vbar_kms]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is importable (for scripts.contract_utils)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Column alias map — same as ingest_sparc_contract.py
# ---------------------------------------------------------------------------

CANDIDATE_COLS: Dict[str, List[str]] = {
    "r":     ["Rad", "R", "r", "radius", "rad_kpc", "r_kpc"],
    "vobs":  ["Vobs", "Vrot", "vrot", "v_obs", "vrot_kms"],
    "e_vobs": ["eVobs", "evobs", "eVrot", "vrot_err", "vrot_err_kms"],
    "vgas":  ["Vgas", "vgas", "v_gas", "vgas_kms"],
    "vdisk": ["Vdisk", "vdisk", "v_disk", "vdisk_kms"],
    "vbul":  ["Vbul", "vbul", "v_bul", "vbul_kms"],
    "vbar":  ["Vbar", "vbar", "v_bar", "vbar_kms"],
}


def _pick_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def _parse_raw(path: Path) -> pd.DataFrame:
    """Read whitespace-separated rotmod file, skipping comment lines."""
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty or unreadable file: {path}")
    return df


def read_rotmod(path: Path) -> Tuple[str, pd.DataFrame]:
    """Parse a single *_rotmod.dat file into the SCM rc_points contract.

    Parameters
    ----------
    path : Path
        Path to a ``*_rotmod.dat`` file.

    Returns
    -------
    (galaxy_id, rc_df)
        ``galaxy_id`` is the file stem without ``_rotmod``.
        ``rc_df`` conforms to the SCM rc_points contract.

    Raises
    ------
    ValueError
        If required columns (radius, observed velocity) are missing.
    """
    path = Path(path)
    galaxy_id = path.stem.replace("_rotmod", "")
    raw = _parse_raw(path)

    c_r    = _pick_col(raw, CANDIDATE_COLS["r"])
    c_vobs = _pick_col(raw, CANDIDATE_COLS["vobs"])
    c_e    = _pick_col(raw, CANDIDATE_COLS["e_vobs"])
    c_vgas = _pick_col(raw, CANDIDATE_COLS["vgas"])
    c_vdisk = _pick_col(raw, CANDIDATE_COLS["vdisk"])
    c_vbul = _pick_col(raw, CANDIDATE_COLS["vbul"])
    c_vbar = _pick_col(raw, CANDIDATE_COLS["vbar"])

    if c_r is None or c_vobs is None:
        raise ValueError(
            f"{galaxy_id}: missing radius/velocity column. "
            f"Columns found: {list(raw.columns)}"
        )

    out = pd.DataFrame({
        "galaxy_id": galaxy_id,
        "r_kpc":     pd.to_numeric(raw[c_r],    errors="coerce"),
        "vrot_kms":  pd.to_numeric(raw[c_vobs], errors="coerce"),
    })

    if c_e is not None:
        out["vrot_err_kms"] = pd.to_numeric(raw[c_e], errors="coerce")

    if c_vbar is not None:
        out["vbar_kms"] = pd.to_numeric(raw[c_vbar], errors="coerce")
    else:
        if c_vgas is not None:
            out["vgas_kms"] = pd.to_numeric(raw[c_vgas], errors="coerce")
        v2_star = None
        if c_vdisk is not None:
            vd = pd.to_numeric(raw[c_vdisk], errors="coerce")
            v2_star = vd ** 2 if v2_star is None else v2_star + vd ** 2
        if c_vbul is not None:
            vb = pd.to_numeric(raw[c_vbul], errors="coerce")
            v2_star = vb ** 2 if v2_star is None else v2_star + vb ** 2
        if v2_star is not None:
            out["vstar_kms"] = np.sqrt(v2_star)

    return galaxy_id, out


def _worker(path_str: str) -> Tuple[str, str]:
    """Top-level function required for pickling in ProcessPoolExecutor."""
    gid, rc = read_rotmod(Path(path_str))
    return gid, rc.to_json(orient="split")


def ingest_sparc_dir(
    sparc_dir: Path,
    pattern: str = "*_rotmod.dat",
    workers: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read all *_rotmod.dat files from *sparc_dir* into the SCM contract.

    Parameters
    ----------
    sparc_dir : Path
        Directory containing ``*_rotmod.dat`` files.
    pattern : str
        Glob pattern for input files (default ``"*_rotmod.dat"``).
    workers : int
        Number of parallel worker processes (default 1 = serial).
        Set to ``os.cpu_count()`` for maximum parallelism.

    Returns
    -------
    (galaxies_df, rc_df)
        ``galaxies_df`` has one row per galaxy (column: ``galaxy_id``).
        ``rc_df`` has all radial points conforming to the SCM rc_points contract.

    Raises
    ------
    SystemExit
        If no files match *pattern*.
    """
    sparc_dir = Path(sparc_dir)
    files = sorted(sparc_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched '{pattern}' in {sparc_dir}")

    rc_frames: List[pd.DataFrame] = []
    gal_ids: List[str] = []

    if workers > 1:
        file_strs = [str(f) for f in files]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, s): s for s in file_strs}
            for future in as_completed(futures):
                gid, rc_json = future.result()
                gal_ids.append(gid)
                rc_frames.append(pd.read_json(rc_json, orient="split"))
    else:
        for f in files:
            gid, rc = read_rotmod(f)
            gal_ids.append(gid)
            rc_frames.append(rc)

    df_gal = (
        pd.DataFrame({"galaxy_id": gal_ids})
        .sort_values("galaxy_id")
        .reset_index(drop=True)
    )
    df_rc = (
        pd.concat(rc_frames, ignore_index=True)
        .sort_values(["galaxy_id", "r_kpc"])
        .reset_index(drop=True)
    )

    return df_gal, df_rc
