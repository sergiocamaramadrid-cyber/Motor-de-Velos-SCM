#!/usr/bin/env python3
"""
ingest_sparc_contract.py

Convierte SPARC (archivos *_rotmod.dat) al contrato interno SCM:
- galaxies.parquet (1 fila por galaxia)
- rc_points.parquet (puntos de curva por galaxia)

Entrada:
- Directorio con archivos SPARC *_rotmod.dat (p.ej. data/SPARC/)

Salida:
- out_dir/galaxies.parquet
- out_dir/rc_points.parquet

Mapeo estándar SPARC:
- r_kpc   <- Rad
- vrot_kms <- Vobs
- vrot_err_kms <- eVobs (si existe)
- vgas_kms <- Vgas (si existe)
- vstar_kms <- sqrt(Vdisk^2 + Vbul^2) (si existe alguno)
- vbar_kms se podrá derivar después por contract_utils (o aquí si quisieras)

Robustez:
- Ignora líneas comentadas (#)
- Tolera columnas extra
- Requiere al menos: Rad, Vobs y alguna forma de bariónico (Vgas/Vdisk/Vbul o Vbar)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so 'scripts.contract_utils' resolves
# whether this file is run as a script or imported as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.contract_utils import ensure_dir, validate_galaxies_df, validate_rc_points_df


CANDIDATE_COLS = {
    "r": ["Rad", "R", "r", "radius", "rad_kpc", "r_kpc"],
    "vobs": ["Vobs", "Vrot", "vrot", "v_obs", "vrot_kms"],
    "e_vobs": ["eVobs", "evobs", "eVrot", "vrot_err", "vrot_err_kms"],
    "vgas": ["Vgas", "vgas", "v_gas", "vgas_kms"],
    "vdisk": ["Vdisk", "vdisk", "v_disk", "vdisk_kms"],
    "vbul": ["Vbul", "vbul", "v_bul", "vbul_kms"],
    "vbar": ["Vbar", "vbar", "v_bar", "vbar_kms"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparc-dir", required=True, help="Directory containing SPARC *_rotmod.dat files")
    ap.add_argument("--out-dir", required=True, help="Output directory for contract parquet files")
    ap.add_argument("--pattern", default="*_rotmod.dat", help="Glob pattern for rotmod files")
    return ap.parse_args()


def _pick_col(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def _read_rotmod(path: Path) -> pd.DataFrame:
    """
    Lee un *_rotmod.dat con robustez:
    - ignora comentarios con '#'
    - requiere header (SPARC normalmente lo trae)
    """
    # engine="python" por compatibilidad con separadores irregulares
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty or unreadable rotmod file: {path}")
    return df


def _to_rc_contract(galaxy_id: str, df: pd.DataFrame) -> pd.DataFrame:
    c_r = _pick_col(df, CANDIDATE_COLS["r"])
    c_vobs = _pick_col(df, CANDIDATE_COLS["vobs"])
    c_e = _pick_col(df, CANDIDATE_COLS["e_vobs"])
    c_vgas = _pick_col(df, CANDIDATE_COLS["vgas"])
    c_vdisk = _pick_col(df, CANDIDATE_COLS["vdisk"])
    c_vbul = _pick_col(df, CANDIDATE_COLS["vbul"])
    c_vbar = _pick_col(df, CANDIDATE_COLS["vbar"])

    if c_r is None or c_vobs is None:
        raise ValueError(
            f"{galaxy_id}: Missing required columns (need Rad and Vobs-like). "
            f"Columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "galaxy_id": galaxy_id,
        "r_kpc": pd.to_numeric(df[c_r], errors="coerce"),
        "vrot_kms": pd.to_numeric(df[c_vobs], errors="coerce"),
    })

    if c_e is not None:
        out["vrot_err_kms"] = pd.to_numeric(df[c_e], errors="coerce")

    # Prefer vbar if present (already combined)
    if c_vbar is not None:
        out["vbar_kms"] = pd.to_numeric(df[c_vbar], errors="coerce")
    else:
        # components
        if c_vgas is not None:
            out["vgas_kms"] = pd.to_numeric(df[c_vgas], errors="coerce")
        # stellar as quadrature disk+bulge where available
        v2_star = None
        if c_vdisk is not None:
            vdisk = pd.to_numeric(df[c_vdisk], errors="coerce")
            v2_star = vdisk ** 2 if v2_star is None else (v2_star + vdisk ** 2)
        if c_vbul is not None:
            vbul = pd.to_numeric(df[c_vbul], errors="coerce")
            v2_star = vbul ** 2 if v2_star is None else (v2_star + vbul ** 2)
        if v2_star is not None:
            out["vstar_kms"] = np.sqrt(v2_star)

    return out


def main() -> None:
    args = parse_args()
    sparc_dir = Path(args.sparc_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    files = sorted(sparc_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {sparc_dir}")

    rc_frames: List[pd.DataFrame] = []
    gal_rows: List[Dict[str, object]] = []

    for f in files:
        galaxy_id = f.name.replace("_rotmod.dat", "")
        df = _read_rotmod(f)
        rc = _to_rc_contract(galaxy_id, df)

        rc_frames.append(rc)
        gal_rows.append({"galaxy_id": galaxy_id})

    df_gal = pd.DataFrame(gal_rows).sort_values("galaxy_id").reset_index(drop=True)
    df_rc = (
        pd.concat(rc_frames, ignore_index=True)
        .sort_values(["galaxy_id", "r_kpc"])
        .reset_index(drop=True)
    )

    # Validate against contract
    v1 = validate_galaxies_df(df_gal)
    v2 = validate_rc_points_df(df_rc)

    if not v1.ok or not v2.ok:
        print("❌ Contract validation failed.")
        for e in v1.errors + v2.errors:
            print(" -", e)
        raise SystemExit(2)

    out_gal = out_dir / "galaxies.parquet"
    out_rc = out_dir / "rc_points.parquet"
    df_gal.to_parquet(out_gal, index=False)
    df_rc.to_parquet(out_rc, index=False)

    print("✅ SPARC → Contract OK")
    print(" - galaxies:", out_gal)
    print(" - rc_points:", out_rc)
    print(f" - N_galaxies: {len(df_gal)}")
    print(f" - N_points:   {len(df_rc)}")


if __name__ == "__main__":
    main()
