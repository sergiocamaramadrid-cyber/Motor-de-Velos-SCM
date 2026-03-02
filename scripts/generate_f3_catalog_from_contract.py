#!/usr/bin/env python3
"""
generate_f3_catalog_from_contract.py

Genera catálogo β (friction_slope) desde el contrato interno:
- data_dir/galaxies.parquet
- data_dir/rc_points.parquet

Uso:
python scripts/generate_f3_catalog_from_contract.py \
  --data-dir data/BIG_SPARC/processed \
  --out results/f3_catalog_big_sparc.parquet \
  --a0 1.2e-10 \
  --deep-frac 0.3 \
  --min-deep 4

Notas:
- Calcula g_obs = (vrot^2)/r
- Calcula g_bar = (vbar^2)/r
- Selecciona puntos deep: g_bar < deep_frac * a0
- Ajusta OLS: log10(g_obs) = beta * log10(g_bar) + c
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from scripts.contract_utils import ensure_dir, read_table


KPC_TO_M = 3.085677581e19
KMS_TO_MS = 1e3


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with galaxies.parquet and rc_points.parquet")
    ap.add_argument("--out", required=True, help="Output catalog (csv|parquet)")
    ap.add_argument("--a0", type=float, default=1.2e-10, help="Acceleration scale a0 (m/s^2)")
    ap.add_argument("--deep-frac", type=float, default=0.3, help="Deep regime threshold: g_bar < deep_frac * a0")
    ap.add_argument("--min-deep", type=int, default=4, help="Minimum deep points to fit beta")
    return ap.parse_args()


def compute_g(v_kms: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    v = v_kms * KMS_TO_MS
    r = r_kpc * KPC_TO_M
    return (v * v) / r


def fit_beta(log_gbar: np.ndarray, log_gobs: np.ndarray) -> Tuple[float, float, float, float]:
    """
    OLS slope + stderr + r_value + p_value from scipy.stats.linregress
    """
    res = stats.linregress(log_gbar, log_gobs)
    return float(res.slope), float(res.stderr), float(res.rvalue), float(res.pvalue)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    gal_path = data_dir / "galaxies.parquet"
    rc_path = data_dir / "rc_points.parquet"

    df_gal = read_table(gal_path)
    df_rc = read_table(rc_path)

    # Ensure required columns exist
    for c in ["galaxy_id"]:
        if c not in df_gal.columns:
            raise SystemExit(f"Missing {c} in galaxies")
    for c in ["galaxy_id", "r_kpc", "vrot_kms", "vbar_kms"]:
        if c not in df_rc.columns:
            raise SystemExit(f"Missing {c} in rc_points")

    # groupby per galaxy
    rows = []
    a0 = float(args.a0)
    deep_thr = float(args.deep_frac) * a0
    min_deep = int(args.min_deep)

    for gid, g in df_rc.groupby("galaxy_id", sort=True):
        r_kpc = g["r_kpc"].to_numpy(dtype=float)
        vrot = g["vrot_kms"].to_numpy(dtype=float)
        vbar = g["vbar_kms"].to_numpy(dtype=float)

        # basic sanity
        ok = (
            np.isfinite(r_kpc) & np.isfinite(vrot) & np.isfinite(vbar)
            & (r_kpc > 0) & (vrot > 0) & (vbar > 0)
        )
        r_kpc = r_kpc[ok]
        vrot = vrot[ok]
        vbar = vbar[ok]

        n_total = int(len(r_kpc))
        if n_total < 2:
            rows.append(
                dict(
                    galaxy_id=gid,
                    n_total=n_total,
                    n_deep=0,
                    friction_slope=np.nan,
                    friction_slope_err=np.nan,
                    r_value=np.nan,
                    p_value=np.nan,
                    velo_inerte_flag=np.nan,
                )
            )
            continue

        g_obs = compute_g(vrot, r_kpc)
        g_bar = compute_g(vbar, r_kpc)

        deep = g_bar < deep_thr
        n_deep = int(np.sum(deep))

        if n_deep < min_deep:
            rows.append(
                dict(
                    galaxy_id=gid,
                    n_total=n_total,
                    n_deep=n_deep,
                    friction_slope=np.nan,
                    friction_slope_err=np.nan,
                    r_value=np.nan,
                    p_value=np.nan,
                    velo_inerte_flag=np.nan,
                )
            )
            continue

        log_gobs = np.log10(g_obs[deep])
        log_gbar = np.log10(g_bar[deep])

        beta, beta_err, r_value, p_value = fit_beta(log_gbar, log_gobs)

        # FEU/SCM semantic flag: consistent with beta=0.5 within 2σ
        velo_flag = 1 if abs(beta - 0.5) <= 2.0 * beta_err else 0

        rows.append(
            dict(
                galaxy_id=gid,
                n_total=n_total,
                n_deep=n_deep,
                friction_slope=beta,
                friction_slope_err=beta_err,
                r_value=r_value,
                p_value=p_value,
                velo_inerte_flag=velo_flag,
            )
        )

    df_out = pd.DataFrame(rows).sort_values("galaxy_id").reset_index(drop=True)

    # write
    if out_path.suffix.lower() == ".csv":
        df_out.to_csv(out_path, index=False)
    else:
        df_out.to_parquet(out_path, index=False)

    print("✅ Catalog generated:", out_path)
    print(" - N rows:", len(df_out))
    print(" - N valid slopes:", int(np.isfinite(df_out["friction_slope"]).sum()))


if __name__ == "__main__":
    main()
