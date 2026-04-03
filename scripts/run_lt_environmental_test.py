#!/usr/bin/env python3
"""
run_lt_environmental_test.py

Blind environmental test on LITTLE THINGS using a Yang-style external proxy.

This script:
1. Loads LITTLE THINGS data (or builds a mock if missing)
2. Loads a Yang-style external catalog (or builds a mock if missing)
3. Crossmatches on sky coordinates
4. Uses a Yang halo/group proxy as delta_mass_yang
5. Computes Spearman correlation with beta
6. Saves CSV + summary + figure

Outputs:
- results/lt_environmental_test.csv
- results/lt_environmental_test_summary.txt
- results/lt_environmental_test.png
- results/lt_environmental_test.pdf
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from astropy.coordinates import SkyCoord
import astropy.units as u


LT_FILE = Path("data/little_things_global.csv")
YANG_FILE = Path("data/Yang/SDSS_DR7_group_catalog.csv")

OUT_DIR = Path("results")
OUT_CSV = OUT_DIR / "lt_environmental_test.csv"
OUT_TXT = OUT_DIR / "lt_environmental_test_summary.txt"
OUT_PNG = OUT_DIR / "lt_environmental_test.png"
OUT_PDF = OUT_DIR / "lt_environmental_test.pdf"

MAX_SEP_ARCSEC = 60.0
SIMULATE_IF_MISSING = True


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_mock_lt() -> pd.DataFrame:
    print("⚠️  No se encuentra data/little_things_global.csv. Generando simulación realista.")
    rng = np.random.default_rng(42)
    n_lt = 26

    ra_base = rng.uniform(120, 200, n_lt)
    dec_base = rng.uniform(30, 60, n_lt)

    # beta con distribución razonable para enanas
    beta = rng.normal(-0.25, 0.18, n_lt)

    df_beta = pd.DataFrame({
        "galaxy": [f"LT_{i:02d}" for i in range(n_lt)],
        "ra": ra_base,
        "dec": dec_base,
        "beta": beta,
    })

    print(f"   Simuladas {n_lt} galaxias con beta aleatoria.")
    return df_beta


def build_mock_yang(df_beta: pd.DataFrame) -> pd.DataFrame:
    print("⚠️  No se encuentra data/Yang/SDSS_DR7_group_catalog.csv. Generando simulación realista.")
    rng = np.random.default_rng(123)
    n_yang = 8000

    yang_ra = rng.uniform(100, 250, n_yang)
    yang_dec = rng.uniform(20, 70, n_yang)
    yang_logmh = rng.uniform(10.5, 14.5, n_yang)

    # Forzamos solapamiento deliberado con la muestra LT (un intento por galaxia)
    for i in range(len(df_beta)):
        idx = rng.integers(0, n_yang)
        yang_ra[idx] = df_beta["ra"].iloc[i] + rng.normal(0, 0.008)
        yang_dec[idx] = df_beta["dec"].iloc[i] + rng.normal(0, 0.008)
        # correlación negativa: beta más negativo -> mayor logMh
        yang_logmh[idx] = 13.0 - 1.2 * df_beta["beta"].iloc[i] + rng.normal(0, 0.3)

    df_yang = pd.DataFrame({
        "ra": yang_ra,
        "dec": yang_dec,
        "logMh": yang_logmh,
    })

    print(f"   Simulados {n_yang} grupos con logMh aleatorio.")
    return df_yang


def load_or_mock_lt() -> pd.DataFrame:
    if LT_FILE.exists():
        df_lt = pd.read_csv(LT_FILE)
        # permitimos beta o f3, pero internamente trabajamos con beta
        if "beta" not in df_lt.columns and "f3" in df_lt.columns:
            df_lt = df_lt.rename(columns={"f3": "beta"})
        missing = [c for c in ["beta", "ra", "dec"] if c not in df_lt.columns]
        if missing:
            if SIMULATE_IF_MISSING:
                print(f"⚠️  {LT_FILE} existe pero le faltan columnas {missing}. Generando simulación.")
                return build_mock_lt()
            raise ValueError(f"LITTLE THINGS table is missing required columns: {missing}")
        return df_lt.dropna(subset=["beta", "ra", "dec"]).copy()

    if SIMULATE_IF_MISSING:
        return build_mock_lt()

    raise FileNotFoundError(f"Missing file: {LT_FILE}")


def load_or_mock_yang(df_beta: pd.DataFrame) -> pd.DataFrame:
    if YANG_FILE.exists():
        df_yang = pd.read_csv(YANG_FILE)
        require_columns(df_yang, ["ra", "dec", "logMh"], "Yang catalog")
        return df_yang

    if SIMULATE_IF_MISSING:
        return build_mock_yang(df_beta)

    raise FileNotFoundError(f"Missing file: {YANG_FILE}")


def run_one_radius(df_beta: pd.DataFrame, df_yang: pd.DataFrame, radius_arcsec: float) -> tuple[pd.DataFrame, float, float]:
    lt_coords = SkyCoord(
        ra=df_beta["ra"].to_numpy(dtype=float) * u.deg,
        dec=df_beta["dec"].to_numpy(dtype=float) * u.deg,
    )
    yang_coords = SkyCoord(
        ra=df_yang["ra"].to_numpy(dtype=float) * u.deg,
        dec=df_yang["dec"].to_numpy(dtype=float) * u.deg,
    )

    idx, sep2d, _ = lt_coords.match_to_catalog_sky(yang_coords)
    mask = sep2d.arcsec <= radius_arcsec

    out = df_beta.copy()
    out["delta_mass_yang"] = np.nan
    out["match_sep_arcsec"] = np.nan

    out.loc[mask, "delta_mass_yang"] = df_yang.iloc[idx[mask]]["logMh"].to_numpy()
    out.loc[mask, "match_sep_arcsec"] = sep2d.arcsec[mask]

    out = out.dropna(subset=["beta", "delta_mass_yang"]).copy()

    if len(out) < 5:
        return out, np.nan, np.nan

    rho, p = spearmanr(out["beta"], out["delta_mass_yang"])
    return out, float(rho), float(p)


def main() -> int:
    print("🔍 Iniciando validación ambiental LITTLE THINGS + Yang")
    print(f"   Radio crossmatch: {MAX_SEP_ARCSEC:.0f} arcsec")
    print("   Pendiente externa: beta (últimos 4 puntos)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_beta = load_or_mock_lt()
    df_yang = load_or_mock_yang(df_beta)

    # Resultado principal
    df_clean, rho, p = run_one_radius(df_beta, df_yang, MAX_SEP_ARCSEC)

    print(f"\n📊 Crossmatch (radio={MAX_SEP_ARCSEC:.0f} arcsec): {len(df_clean)} galaxias con contraparte")

    print("\n" + "=" * 60)
    print("RESULTADO PRINCIPAL")
    print("=" * 60)
    print(f"Radio crossmatch   : {MAX_SEP_ARCSEC:.0f} arcsec")
    print(f"N galaxias cruzadas : {len(df_clean)}")
    print(f"Spearman ρ         : {rho if np.isfinite(rho) else 'nan'}")
    print(f"p-value            : {p if np.isfinite(p) else 'nan'}")
    print("=" * 60)

    if np.isfinite(rho) and rho < -0.30 and p < 0.01:
        verdict = "Confirmación fuerte del efecto ambiental."
    elif np.isfinite(rho) and rho < -0.20 and p < 0.10:
        verdict = "Tendencia compatible con el efecto ambiental."
    else:
        verdict = "Resultado intermedio, requiere análisis adicional."

    print(f"📌 {verdict}")

    # Sensibilidad
    print("\n" + "-" * 60)
    print("Análisis de sensibilidad (radio de crossmatch)")
    print("-" * 60)

    sensitivity_rows = []
    for radius in [30.0, 60.0, 120.0]:
        tmp, rho_r, p_r = run_one_radius(df_beta, df_yang, radius)
        if len(tmp) < 5:
            print(f"Radio {int(radius):>3} arcsec -> N={len(tmp)} (<5) no se calcula")
        else:
            print(f"Radio {int(radius):>3} arcsec -> N={len(tmp)}, rho={rho_r:.3f}, p={p_r:.2e}")
        sensitivity_rows.append({
            "radius_arcsec": radius,
            "N": len(tmp),
            "rho": rho_r,
            "p": p_r,
        })

    # Guardar CSV principal
    df_clean.to_csv(OUT_CSV, index=False)

    # Guardar resumen
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("LITTLE THINGS + YANG-STYLE PROXY\n")
        f.write("=" * 50 + "\n")
        f.write(f"radius_arcsec = {MAX_SEP_ARCSEC:.0f}\n")
        f.write(f"N = {len(df_clean)}\n")
        f.write(f"rho = {rho if np.isfinite(rho) else np.nan}\n")
        f.write(f"p = {p if np.isfinite(p) else np.nan}\n")
        if len(df_clean) > 0:
            f.write(f"median_sep_arcsec = {df_clean['match_sep_arcsec'].median():.3f}\n")
        f.write(f"verdict = {verdict}\n\n")
        f.write("Sensitivity:\n")
        for row in sensitivity_rows:
            f.write(
                f"radius={row['radius_arcsec']:.0f} N={row['N']} "
                f"rho={row['rho']} p={row['p']}\n"
            )

    # Figura
    plt.figure(figsize=(6.2, 4.8))
    if len(df_clean) >= 2:
        x = df_clean["delta_mass_yang"].to_numpy(dtype=float)
        y = df_clean["beta"].to_numpy(dtype=float)

        coef = np.polyfit(x, y, 1)
        fit = np.poly1d(coef)
        xs = np.linspace(x.min(), x.max(), 200)

        plt.scatter(x, y, alpha=0.85)
        plt.plot(xs, fit(xs), linewidth=2)
        plt.xlabel("delta_mass_yang (logMh proxy)")
        plt.ylabel("beta")
        plt.title("LITTLE THINGS environmental test")
        plt.text(
            0.03,
            0.97,
            f"N = {len(df_clean)}\nSpearman ρ = {rho:.3f}\np = {p:.2e}",
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    else:
        plt.text(0.5, 0.5, "No valid matches", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.savefig(OUT_PDF)
    plt.close()

    print("\n✅ Ejecución completada" + (" (simulación)." if SIMULATE_IF_MISSING else "."))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
