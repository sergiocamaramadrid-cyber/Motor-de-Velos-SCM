#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ is None or __package__ == "":
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from scripts.contract_utils import (
        CONTRACT_COLUMNS,
        compute_vbar_kms,
        read_table,
        validate_contract,
    )
else:
    from .contract_utils import (
        CONTRACT_COLUMNS,
        compute_vbar_kms,
        read_table,
        validate_contract,
    )

A0_SI = 1.2e-10
KPC_TO_M = 3.085677581491367e19
KMS_TO_MS = 1.0e3


def fail(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def check_local_sparc_data(data_root: Path, rotmod_subdir: str = "rotmod") -> tuple[Path, list[Path]]:
    if not data_root.exists():
        fail(f"No existe data_root: {data_root}")

    rotmod_dir = data_root / rotmod_subdir
    if not rotmod_dir.exists():
        fail(
            f"No existe directorio de curvas: {rotmod_dir}\n"
            f"Estructura esperada:\n"
            f"  {data_root}/rotmod/*.dat"
        )

    dat_files = sorted(rotmod_dir.glob("*.dat"))
    if not dat_files:
        fail(f"No se encontraron archivos .dat en {rotmod_dir}")

    return rotmod_dir, dat_files


def _read_numeric_lines(path: Path) -> np.ndarray:
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"No hay datos numéricos utilizables en {path}")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Número inconsistente de columnas en {path}: {sorted(widths)}")

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Formato no soportado en {path}: shape={arr.shape}")

    return arr


def read_rotmod_curve(path: Path) -> pd.DataFrame:
    arr = _read_numeric_lines(path)
    ncol = arr.shape[1]

    df = pd.DataFrame(
        {
            "r_kpc": arr[:, 0],
            "v_obs_kms": arr[:, 1],
            "e_v_obs_kms": arr[:, 2],
        }
    )

    df["v_gas_kms"] = arr[:, 3] if ncol >= 4 else np.nan
    df["v_disk_kms"] = arr[:, 4] if ncol >= 5 else np.nan
    df["v_bulge_kms"] = arr[:, 5] if ncol >= 6 else np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_kpc", "v_obs_kms"])
    df = df[df["r_kpc"] > 0].copy()

    if len(df) < 4:
        raise ValueError(f"Curva insuficiente tras limpieza en {path}")

    return df.reset_index(drop=True)


def weighted_linear_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[mask]
    y = y[mask]
    w = w[mask]

    if x.size < 2:
        return np.nan

    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    cov = np.average((x - xw) * (y - yw), weights=w)
    var = np.average((x - xw) ** 2, weights=w)

    if not np.isfinite(var) or var <= 0:
        return np.nan

    return float(cov / var)


def compute_f3_scm(
    df: pd.DataFrame,
    tail_frac: float = 0.7,
    min_points_tail: int = 4,
    v_floor_kms: float = 5.0,
) -> tuple[float, int, float]:
    r = df["r_kpc"].to_numpy(dtype=float)
    v = df["v_obs_kms"].to_numpy(dtype=float)
    e = df["e_v_obs_kms"].to_numpy(dtype=float)

    rmax = np.nanmax(r)
    mask = (r >= tail_frac * rmax) & np.isfinite(r) & np.isfinite(v) & (v > v_floor_kms)

    rt = r[mask]
    vt = v[mask]
    et = e[mask]

    if rt.size < min_points_tail:
        return np.nan, int(rt.size), float(rmax)

    x = np.log10(rt)
    y = np.log10(vt)

    rel = np.where(np.isfinite(et) & (et > 0), et / vt, np.nan)
    w = np.where(np.isfinite(rel) & (rel > 0), 1.0 / (rel**2), 1.0)

    slope = weighted_linear_slope(x, y, w)
    return slope, int(rt.size), float(rmax)


def acceleration_from_curve(r_kpc: np.ndarray, v_kms: np.ndarray) -> np.ndarray:
    r_m = np.asarray(r_kpc, dtype=float) * KPC_TO_M
    v_ms = np.asarray(v_kms, dtype=float) * KMS_TO_MS
    with np.errstate(divide="ignore", invalid="ignore"):
        return (v_ms**2) / r_m


def compute_beta_from_curve(
    df: pd.DataFrame,
    beta_gbar_max: float = 0.3,
    min_points_beta: int = 4,
    v_floor_kms: float = 5.0,
) -> tuple[float, int]:
    r = df["r_kpc"].to_numpy(dtype=float)
    v_obs = df["v_obs_kms"].to_numpy(dtype=float)
    v_bar = compute_vbar_kms(df["v_gas_kms"], df["v_disk_kms"], df["v_bulge_kms"])

    g_obs = acceleration_from_curve(r, v_obs)
    g_bar = acceleration_from_curve(r, v_bar)

    mask = (
        np.isfinite(r)
        & np.isfinite(v_obs)
        & np.isfinite(v_bar)
        & (v_obs > v_floor_kms)
        & (v_bar > 0)
        & np.isfinite(g_obs)
        & np.isfinite(g_bar)
        & (g_obs > 0)
        & (g_bar > 0)
        & (g_bar < beta_gbar_max * A0_SI)
    )

    if mask.sum() < min_points_beta:
        return np.nan, int(mask.sum())

    x = np.log10(g_bar[mask])
    y = np.log10(g_obs[mask])
    slope = weighted_linear_slope(x, y)

    return slope, int(mask.sum())


def estimate_logsigma_hi_out(df: pd.DataFrame, tail_frac: float = 0.7) -> float:
    r = df["r_kpc"].to_numpy(dtype=float)
    vgas = np.nan_to_num(df["v_gas_kms"].to_numpy(dtype=float), nan=0.0)

    if r.size == 0 or np.nanmax(r) <= 0:
        return np.nan

    rmax = np.nanmax(r)
    mask = (r >= tail_frac * rmax) & np.isfinite(r) & np.isfinite(vgas) & (vgas > 0)

    if mask.sum() < 3:
        return np.nan

    proxy = (vgas[mask] ** 2) / r[mask]
    proxy = proxy[np.isfinite(proxy) & (proxy > 0)]
    if proxy.size == 0:
        return np.nan

    return float(np.log10(np.median(proxy)))


def build_master_row(path: Path, tail_frac: float, beta_gbar_max: float) -> dict:
    df = read_rotmod_curve(path)

    galaxy = path.stem
    n_points = int(len(df))
    rmax_kpc = float(np.nanmax(df["r_kpc"].to_numpy(dtype=float)))
    vmax_obs_kms = float(np.nanmax(df["v_obs_kms"].to_numpy(dtype=float)))

    f3_scm, n_tail, rmax_kpc_check = compute_f3_scm(df, tail_frac=tail_frac)
    beta, n_beta = compute_beta_from_curve(df, beta_gbar_max=beta_gbar_max)
    delta_f3 = float(f3_scm - 0.5) if np.isfinite(f3_scm) else np.nan
    logsigma_hi_out = estimate_logsigma_hi_out(df, tail_frac=tail_frac)

    row = {
        "galaxy": galaxy,
        "source_file": path.name,
        "n_points_curve": n_points,
        "rmax_kpc": rmax_kpc,
        "vmax_obs_kms": vmax_obs_kms,
        "tail_frac": float(tail_frac),
        "n_tail_points": int(n_tail),
        "F3_SCM": float(f3_scm) if np.isfinite(f3_scm) else np.nan,
        "delta_f3": float(delta_f3) if np.isfinite(delta_f3) else np.nan,
        "beta": float(beta) if np.isfinite(beta) else np.nan,
        "n_beta_points": int(n_beta),
        "logSigmaHI_out": float(logsigma_hi_out) if np.isfinite(logsigma_hi_out) else np.nan,
        "logSigmaHI_out_proxy": float(logsigma_hi_out) if np.isfinite(logsigma_hi_out) else np.nan,
        "quality_flag_tail_ok": bool(np.isfinite(f3_scm) and n_tail >= 4),
        "quality_flag_beta_ok": bool(np.isfinite(beta) and n_beta >= 4),
        "contract_version": "SPARC_MASTER_v1.0",
    }

    if not math.isclose(rmax_kpc, rmax_kpc_check, rel_tol=0.0, abs_tol=1e-10):
        warn(f"Inconsistencia menor en rmax para {galaxy}")

    return row


def build_master_table(dat_files: list[Path], tail_frac: float, beta_gbar_max: float) -> pd.DataFrame:
    rows: list[dict] = []

    for i, path in enumerate(dat_files, start=1):
        try:
            rows.append(build_master_row(path, tail_frac=tail_frac, beta_gbar_max=beta_gbar_max))
        except Exception as exc:
            warn(f"[{i}/{len(dat_files)}] Saltando {path.name}: {exc}")

    if not rows:
        fail("No se pudo construir ninguna fila del CSV maestro.")

    df = pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True)
    return validate_contract(df)


def build_sanity_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_galaxies": [int(len(df))],
            "n_tail_ok": [int(df["quality_flag_tail_ok"].sum())],
            "n_beta_ok": [int(df["quality_flag_beta_ok"].sum())],
            "median_F3_SCM": [float(df["F3_SCM"].median(skipna=True))],
            "median_delta_f3": [float(df["delta_f3"].median(skipna=True))],
            "median_beta": [float(df["beta"].median(skipna=True))],
            "median_logSigmaHI_out": [float(df["logSigmaHI_out"].median(skipna=True))],
            "median_logSigmaHI_out_proxy": [float(df["logSigmaHI_out_proxy"].median(skipna=True))],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye sparc_175_master.csv a partir de curvas SPARC locales.")
    parser.add_argument("--data-root", type=Path, required=True, help="Raíz local de datos SPARC.")
    parser.add_argument("--rotmod-subdir", type=str, default="rotmod", help="Subdirectorio con .dat.")
    parser.add_argument("--out", type=Path, required=True, help="Ruta CSV maestro de salida.")
    parser.add_argument("--sanity-out", type=Path, default=None, help="Ruta opcional de sanity summary.")
    parser.add_argument("--tail-frac", type=float, default=0.7, help="Fracción de Rmax para definir cola externa.")
    parser.add_argument("--beta-gbar-max", type=float, default=0.3, help="Umbral profundo: g_bar < x * a0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.tail_frac < 1.0):
        fail("--tail-frac debe estar entre 0 y 1.")
    if args.beta_gbar_max <= 0:
        fail("--beta-gbar-max debe ser positivo.")

    _, dat_files = check_local_sparc_data(args.data_root, rotmod_subdir=args.rotmod_subdir)
    info(f"Curvas detectadas: {len(dat_files)}")

    master = build_master_table(
        dat_files=dat_files,
        tail_frac=args.tail_frac,
        beta_gbar_max=args.beta_gbar_max,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.out, index=False)
    info(f"CSV maestro escrito en: {args.out}")

    sanity = build_sanity_summary(master)
    sanity_out = args.sanity_out or args.out.with_name("sparc_175_master_sanity.csv")
    sanity.to_csv(sanity_out, index=False)
    info(f"Sanity summary escrito en: {sanity_out}")

    print("\n=== RESUMEN ===")
    print(sanity.to_string(index=False))
    print("\nColumnas del contrato:")
    for col in CONTRACT_COLUMNS:
        print(f" - {col}")


if __name__ == "__main__":
    main()
