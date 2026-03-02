#!/usr/bin/env python3
"""
generate_f3_catalog_from_contract.py

Genera el catálogo F3 (deep-slope + velo-inerte) a partir del contrato
interno SCM (galaxies.parquet + rc_points.parquet).

Columnas de salida por galaxia:
  galaxy_id         — identificador
  friction_slope    — pendiente log(g_obs) vs log(g_bar) en régimen profundo
  friction_slope_err — error estándar de friction_slope
  n_deep            — nº de puntos en régimen profundo (g_bar < deep_frac × a0)
  velo_inerte_flag  — True si n_deep ≥ 2 y friction_slope ∈ [0.35, 0.65]

Uso
---
  python scripts/generate_f3_catalog_from_contract.py \\
    --data-dir data/SPARC/processed_contract \\
    --out results/f3_catalog_sparc_from_contract.parquet

Requisitos del contrato de entrada
-----------------------------------
  galaxies.parquet : columnas mínimas  → galaxy_id
  rc_points.parquet: columnas mínimas  → galaxy_id, r_kpc, vrot_kms
                     más uno de:        vbar_kms  ó  (vgas_kms + vstar_kms)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

# Ensure repo root resolves imports whether run as script or module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.contract_utils import ensure_dir

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KPC_TO_M: float = 3.085677581e19          # 1 kpc in metres
_CONV: float = 1e6 / KPC_TO_M             # km²/s² → m/s² (divide by r_kpc separately)
_A0_DEFAULT: float = 1.2e-10              # characteristic acceleration (m/s²)
_DEEP_THRESHOLD: float = 0.3             # deep regime: g_bar < threshold × a0
                                          # (0.3 × a0 ≈ 0.3 × 1.2e-10 m/s²; consistent
                                          #  with McGaugh+2016 deep-MOND definition)
_MIN_R_KPC: float = 1e-10                # guard against r ≈ 0

# Slope range considered consistent with MOND/velos deep prediction (β ≈ 0.5)
_VELO_INERTE_SLOPE_LO: float = 0.35
_VELO_INERTE_SLOPE_HI: float = 0.65


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate F3 (deep-slope / velo-inerte) catalog from SCM contract."
    )
    ap.add_argument("--data-dir", required=True,
                    help="Directory containing galaxies.parquet and rc_points.parquet")
    ap.add_argument("--out", required=True,
                    help="Output parquet path, e.g. results/f3_catalog.parquet")
    ap.add_argument("--a0", type=float, default=_A0_DEFAULT,
                    help=f"Characteristic acceleration in m/s² (default {_A0_DEFAULT})")
    ap.add_argument("--deep-threshold", type=float, default=_DEEP_THRESHOLD,
                    help=f"Deep-regime threshold as fraction of a0 (default {_DEEP_THRESHOLD})")
    ap.add_argument("--min-deep", type=int, default=2,
                    help="Minimum number of deep-regime points required to fit the slope (default 2)")
    return ap.parse_args()


def _g_bar(row_group: pd.DataFrame) -> Optional[pd.Series]:
    """Compute baryonic centripetal acceleration in m/s² for each row."""
    r = row_group["r_kpc"].values.astype(float)
    safe_r = np.maximum(r, _MIN_R_KPC)

    if "vbar_kms" in row_group.columns:
        vb = row_group["vbar_kms"].values.astype(float)
    elif "vgas_kms" in row_group.columns and "vstar_kms" in row_group.columns:
        vgas = row_group["vgas_kms"].values.astype(float)
        vstar = row_group["vstar_kms"].values.astype(float)
        vb = np.sqrt(np.maximum(vgas ** 2 + vstar ** 2, 0.0))
    else:
        return None  # insufficient baryonic information

    return pd.Series(vb ** 2 / safe_r * _CONV, index=row_group.index)


def _compute_galaxy_stats(
    gid: str,
    group: pd.DataFrame,
    a0: float,
    deep_threshold: float,
    min_deep: int = 2,
) -> Dict[str, object]:
    """Return per-galaxy F3 metrics dict."""
    base: Dict[str, object] = {
        "galaxy_id": gid,
        "friction_slope": float("nan"),
        "friction_slope_err": float("nan"),
        "n_deep": 0,
        "velo_inerte_flag": False,
    }

    g_bar_vals = _g_bar(group)
    if g_bar_vals is None:
        return base

    vrot = group["vrot_kms"].values.astype(float)
    r = group["r_kpc"].values.astype(float)
    safe_r = np.maximum(r, _MIN_R_KPC)
    g_obs_vals = vrot ** 2 / safe_r * _CONV

    valid = (g_bar_vals.values > 0) & (g_obs_vals > 0)
    if valid.sum() < 2:
        return base

    log_gb = np.log10(g_bar_vals.values[valid])
    log_go = np.log10(g_obs_vals[valid])

    deep_mask = g_bar_vals.values[valid] < deep_threshold * a0
    n_deep = int(deep_mask.sum())
    base["n_deep"] = n_deep

    if n_deep >= min_deep:
        slope, _, _, _, stderr = linregress(log_gb[deep_mask], log_go[deep_mask])
        base["friction_slope"] = float(slope)
        base["friction_slope_err"] = float(stderr)
        base["velo_inerte_flag"] = bool(
            _VELO_INERTE_SLOPE_LO <= slope <= _VELO_INERTE_SLOPE_HI
        )

    return base


def build_f3_catalog(
    data_dir: Path,
    a0: float = _A0_DEFAULT,
    deep_threshold: float = _DEEP_THRESHOLD,
    min_deep: int = 2,
) -> pd.DataFrame:
    """Load contract and build the F3 per-galaxy catalog."""
    gal_path = data_dir / "galaxies.parquet"
    rc_path = data_dir / "rc_points.parquet"

    for p in (gal_path, rc_path):
        if not p.exists():
            raise FileNotFoundError(f"Contract file not found: {p}")

    df_gal = pd.read_parquet(gal_path)
    df_rc = pd.read_parquet(rc_path)

    rows: List[Dict[str, object]] = []
    for gid in df_gal["galaxy_id"]:
        group = df_rc[df_rc["galaxy_id"] == gid].copy()
        rows.append(_compute_galaxy_stats(gid, group, a0, deep_threshold, min_deep))

    result = pd.DataFrame(rows)
    # Stable column order
    result = result[["galaxy_id", "friction_slope", "friction_slope_err",
                     "n_deep", "velo_inerte_flag"]]
    result["n_deep"] = result["n_deep"].astype(int)
    result["friction_slope"] = result["friction_slope"].astype(float)
    result["friction_slope_err"] = result["friction_slope_err"].astype(float)
    result["velo_inerte_flag"] = result["velo_inerte_flag"].astype(bool)
    return result.sort_values("galaxy_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    ensure_dir(out_path.parent)
    catalog = build_f3_catalog(data_dir, a0=args.a0, deep_threshold=args.deep_threshold,
                               min_deep=args.min_deep)

    catalog.to_parquet(out_path, index=False)

    print("✅ F3 catalog OK")
    print(f" - N_galaxies:       {len(catalog)}")
    print(f" - velo_inerte:      {catalog['velo_inerte_flag'].sum()} / {len(catalog)}")
    n_slope = catalog["friction_slope"].notna().sum()
    if n_slope:
        print(f" - friction_slope μ: {catalog['friction_slope'].mean():.3f}")
        print(f" - n_deep total:     {catalog['n_deep'].sum()}")
    print(f" - output:           {out_path}")


if __name__ == "__main__":
    main()
