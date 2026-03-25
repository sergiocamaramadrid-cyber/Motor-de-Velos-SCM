#!/usr/bin/env python3
"""
run_h3_magellanic.py – Pilot H3 para Nubes de Magallanes
Mide persistencia radial (lag-1 / lag-2) y roughness en perfiles de anillos.
LMC como caso base ordenado, SMC como stress test perturbado.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_lag_autocorr(series: pd.Series, lag: int = 1) -> float:
    """Autocorrelación de orden lag."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < lag + 3:
        return np.nan
    return float(clean.autocorr(lag=lag))


def compute_roughness(series: pd.Series) -> float:
    """Roughness = MAD de las diferencias entre anillos consecutivos."""
    clean = pd.to_numeric(series, errors="coerce").dropna().values
    if len(clean) < 3:
        return np.nan
    diffs = np.diff(clean)
    return float(np.median(np.abs(diffs - np.median(diffs))))


def prepare_profile(df: pd.DataFrame, min_ring_points: int = 20, r_min_frac: float = 0.5) -> pd.DataFrame:
    """Prepara y filtra el perfil radial."""
    df = df.copy()

    # Conversión segura
    for col in ["r_kpc", "delta_F3", "n_points_ring", "v_rot_kms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Limpieza
    df = df.dropna(subset=["r_kpc", "delta_F3", "n_points_ring", "v_rot_kms"])
    df = df[df["r_kpc"] > 0]
    df = df[df["n_points_ring"] >= min_ring_points]

    # Zona externa (opcional pero recomendado)
    if "Rmax_kpc" in df.columns:
        df = df[df["r_kpc"] >= r_min_frac * df["Rmax_kpc"]]

    df = df.sort_values("r_kpc").reset_index(drop=True)
    return df


def analyze_cloud(df: pd.DataFrame, min_rings: int = 8) -> dict:
    """Analiza un perfil de una galaxia."""
    galaxy = df["galaxy"].iloc[0] if not df.empty else "unknown"

    if len(df) < min_rings:
        return {
            "galaxy": galaxy,
            "n_rings": len(df),
            "rho_lag1": np.nan,
            "rho_lag2": np.nan,
            "roughness": np.nan,
            "r_min_kpc": np.nan,
            "r_max_kpc": np.nan,
            "status": "too_few_rings"
        }

    rho1 = compute_lag_autocorr(df["delta_F3"], lag=1)
    rho2 = compute_lag_autocorr(df["delta_F3"], lag=2)
    rough = compute_roughness(df["delta_F3"])

    return {
        "galaxy": galaxy,
        "n_rings": int(len(df)),
        "rho_lag1": rho1,
        "rho_lag2": rho2,
        "roughness": rough,
        "r_min_kpc": float(df["r_kpc"].min()),
        "r_max_kpc": float(df["r_kpc"].max()),
        "status": "ok"
    }


def main():
    parser = argparse.ArgumentParser(description="H3 Pilot - Nubes de Magallanes")
    parser.add_argument("--input", required=True,
                        help="CSV con anillos cinemáticos (lmc_kinematic_rings.csv o combinado)")
    parser.add_argument("--outdir", required=True, help="Directorio de salida")
    parser.add_argument("--min-ring-points", type=int, default=20)
    parser.add_argument("--min-rings", type=int, default=8)
    parser.add_argument("--r-min-frac", type=float, default=0.5,
                        help="Fracción mínima de Rmax para zona externa")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    summaries = []
    for gal, group in df.groupby("galaxy"):
        prof = prepare_profile(group,
                               min_ring_points=args.min_ring_points,
                               r_min_frac=args.r_min_frac)

        # Guardar perfil limpio
        prof.to_csv(outdir / f"{gal.lower()}_clean_profile.csv", index=False)

        # Análisis
        summary = analyze_cloud(prof, min_rings=args.min_rings)
        summaries.append(summary)

    # Resultados finales
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outdir / "h3_magellanic_summary.csv", index=False)

    print("\n=== RESULTADOS H3 - NUBES DE MAGALLANES ===")
    print(summary_df.to_string(index=False))
    print(f"\nResultados guardados en: {outdir}")


if __name__ == "__main__":
    main()
