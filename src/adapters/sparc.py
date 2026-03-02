"""
sparc.py — Class-based SPARC adapter.

Converts a directory of SPARC ``*_rotmod.dat`` files into the SCM internal
contract (galaxies_df + rc_points_df) via the :class:`SPARCAdapter` class.

The column-resolution and parsing logic mirrors ``sparc_adapter.py``; this
module exposes it through the :class:`Adapter` Protocol defined in ``base.py``
so the unified ``ingest`` CLI can dispatch to it by name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.contract_utils import validate_galaxies_df, validate_rc_points_df
from .base import IngestConfig, add_metadata


# ---------------------------------------------------------------------------
# Column alias map (identical to sparc_adapter.py)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def _read_rotmod(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")


def _to_rc_contract(galaxy_id: str, df: pd.DataFrame) -> pd.DataFrame:
    c_r    = _pick_col(df, CANDIDATE_COLS["r"])
    c_vobs = _pick_col(df, CANDIDATE_COLS["vobs"])
    c_e    = _pick_col(df, CANDIDATE_COLS["e_vobs"])
    c_vgas = _pick_col(df, CANDIDATE_COLS["vgas"])
    c_vdisk = _pick_col(df, CANDIDATE_COLS["vdisk"])
    c_vbul  = _pick_col(df, CANDIDATE_COLS["vbul"])
    c_vbar  = _pick_col(df, CANDIDATE_COLS["vbar"])

    if c_r is None or c_vobs is None:
        raise ValueError(
            f"{galaxy_id}: missing Rad/Vobs-like columns. "
            f"cols={list(df.columns)}"
        )

    out = pd.DataFrame({
        "galaxy_id": galaxy_id,
        "r_kpc":     pd.to_numeric(df[c_r],    errors="coerce"),
        "vrot_kms":  pd.to_numeric(df[c_vobs], errors="coerce"),
    })

    if c_e is not None:
        out["vrot_err_kms"] = pd.to_numeric(df[c_e], errors="coerce")

    if c_vbar is not None:
        out["vbar_kms"] = pd.to_numeric(df[c_vbar], errors="coerce")
        return out

    if c_vgas is not None:
        out["vgas_kms"] = pd.to_numeric(df[c_vgas], errors="coerce")

    v2_star = None
    if c_vdisk is not None:
        vdisk = pd.to_numeric(df[c_vdisk], errors="coerce")
        v2_star = vdisk ** 2 if v2_star is None else v2_star + vdisk ** 2
    if c_vbul is not None:
        vbul = pd.to_numeric(df[c_vbul], errors="coerce")
        v2_star = vbul ** 2 if v2_star is None else v2_star + vbul ** 2

    if v2_star is not None:
        out["vstar_kms"] = np.sqrt(v2_star)

    return out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SPARCAdapter:
    """Adapter for standard SPARC *_rotmod.dat files."""

    name: str = "sparc"

    def ingest(
        self,
        input_path: Path,
        config: IngestConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest all ``*_rotmod.dat`` files in *input_path*.

        Parameters
        ----------
        input_path : Path
            Directory containing ``*_rotmod.dat`` files.
        config : IngestConfig
            Ingestion configuration (survey name, instrument, etc.).

        Returns
        -------
        (galaxies_df, rc_points_df)
            Contract-compliant DataFrames.  ``galaxies_df`` includes
            ``survey`` and ``instrument`` metadata columns.

        Raises
        ------
        FileNotFoundError
            If no ``*_rotmod.dat`` files are found in *input_path*.
        ValueError
            If the contract validation fails.
        """
        files = sorted(Path(input_path).glob("*_rotmod.dat"))
        if not files:
            raise FileNotFoundError(f"No *_rotmod.dat files found in {input_path}")

        rc_frames: List[pd.DataFrame] = []
        gal_rows: List[Dict[str, object]] = []

        for f in files:
            galaxy_id = f.name.replace("_rotmod.dat", "")
            df = _read_rotmod(f)
            rc_frames.append(_to_rc_contract(galaxy_id, df))
            gal_rows.append({"galaxy_id": galaxy_id})

        df_gal = (
            pd.DataFrame(gal_rows)
            .sort_values("galaxy_id")
            .reset_index(drop=True)
        )
        df_gal = add_metadata(
            df_gal,
            survey=config.survey if config.survey != "unknown" else "SPARC",
            instrument=config.instrument,
        )

        df_rc = (
            pd.concat(rc_frames, ignore_index=True)
            .sort_values(["galaxy_id", "r_kpc"])
            .reset_index(drop=True)
        )

        # Validate against contract (galaxy_id-only subset to match validator)
        v1 = validate_galaxies_df(df_gal[["galaxy_id"]])
        v2 = validate_rc_points_df(df_rc)

        if not v1.ok or not v2.ok:
            raise ValueError(
                "Contract validation failed: "
                + "; ".join(v1.errors + v2.errors)
            )

        return df_gal, df_rc
