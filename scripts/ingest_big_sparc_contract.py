#!/usr/bin/env python3
"""
ingest_big_sparc_contract.py

Construye un CSV maestro limpio para el bloque BIG-SPARC / SCM-Motor de Velos
usando curvas SPARC crudas (rotmod/*.dat).

Además, mantiene un modo legado de ingesta de tablas BIG-SPARC (galaxies +
rc_points) para compatibilidad con el pipeline previo.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.contract_utils import CONTRACT_COLUMNS
from scripts.contract_utils import compute_vbar_kms as compute_vbar_kms_legacy
from scripts.contract_utils import read_table, validate_contract

A0_SI = 1.2e-10
KPC_TO_M = 3.085677581491367e19
KMS_TO_MS = 1.0e3
DEFAULT_SANITY_FILENAME = "sparc_175_master_sanity.csv"


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def fail(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


# ---------------------------------------------------------------------
# Lectura SPARC rotmod/*.dat
# ---------------------------------------------------------------------
def check_local_sparc_data(data_root: Path, rotmod_subdir: str = "rotmod") -> tuple[Path, list[Path]]:
    if not data_root.exists():
        fail(f"No existe data_root: {data_root}")

    rotmod_dir = data_root / rotmod_subdir
    if not rotmod_dir.exists():
        fail(
            f"No existe directorio de curvas: {rotmod_dir}\n"
            f"Estructura esperada:\n  {data_root}/rotmod/*.dat"
        )

    dat_files = sorted(rotmod_dir.glob("*.dat"))
    if not dat_files:
        fail(
            f"No se encontraron archivos .dat en {rotmod_dir}\n"
            "Asegúrate de haber descargado las curvas SPARC locales."
        )

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


# ---------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------
def compute_vbar_kms(v_gas_kms: Iterable[float], v_disk_kms: Iterable[float], v_bulge_kms: Iterable[float]) -> np.ndarray:
    v_gas = np.nan_to_num(np.asarray(v_gas_kms, dtype=float), nan=0.0)
    v_disk = np.nan_to_num(np.asarray(v_disk_kms, dtype=float), nan=0.0)
    v_bulge = np.nan_to_num(np.asarray(v_bulge_kms, dtype=float), nan=0.0)
    return np.sqrt(np.clip(v_gas**2 + v_disk**2 + v_bulge**2, a_min=0.0, a_max=None))


def weighted_linear_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)

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


def compute_f3_scm(df: pd.DataFrame, tail_frac: float = 0.7, min_points_tail: int = 4, v_floor_kms: float = 5.0) -> tuple[float, int, float]:
    """Compute F3_SCM as slope dlog10(V_obs)/dlog10(r) in the outer tail."""
    r = df["r_kpc"].to_numpy(dtype=float)
    v = df["v_obs_kms"].to_numpy(dtype=float)
    e = df["e_v_obs_kms"].to_numpy(dtype=float)

    rmax = np.nanmax(r)
    tail_mask = (r >= tail_frac * rmax) & np.isfinite(r) & np.isfinite(v) & (v > v_floor_kms)

    rt = r[tail_mask]
    vt = v[tail_mask]
    et = e[tail_mask]

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
        g = (v_ms**2) / r_m
    return g


def compute_beta_from_curve(df: pd.DataFrame, beta_gbar_max: float = 0.3, min_points_beta: int = 4, v_floor_kms: float = 5.0) -> tuple[float, int]:
    """Compute beta as slope dlog10(g_obs)/dlog10(g_bar) in deep regime."""
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
    return weighted_linear_slope(x, y, None), int(mask.sum())


def estimate_logsigma_hi_out(df: pd.DataFrame, tail_frac: float = 0.7) -> float:
    """Return a reproducible proxy: median log10(v_gas^2 / r) in outer tail."""
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


# ---------------------------------------------------------------------
# Contrato maestro
# ---------------------------------------------------------------------
def build_master_row(path: Path, tail_frac: float, beta_gbar_max: float) -> dict:
    df = read_rotmod_curve(path)

    galaxy = path.stem
    n_points = int(len(df))
    rmax_kpc = float(np.nanmax(df["r_kpc"].to_numpy(dtype=float)))
    vmax_obs_kms = float(np.nanmax(df["v_obs_kms"].to_numpy(dtype=float)))

    f3_scm, n_tail, rmax_kpc_check = compute_f3_scm(df=df, tail_frac=tail_frac, min_points_tail=4, v_floor_kms=5.0)
    beta, n_beta = compute_beta_from_curve(df=df, beta_gbar_max=beta_gbar_max, min_points_beta=4, v_floor_kms=5.0)
    delta_f3 = float(f3_scm - 0.5) if np.isfinite(f3_scm) else np.nan
    logsigma_hi_out = estimate_logsigma_hi_out(df, tail_frac=tail_frac)

    logsigma_value = float(logsigma_hi_out) if np.isfinite(logsigma_hi_out) else np.nan

    row = {
        "galaxy": galaxy,
        "source_file": str(path.name),
        "n_points_curve": n_points,
        "rmax_kpc": rmax_kpc,
        "vmax_obs_kms": vmax_obs_kms,
        "tail_frac": float(tail_frac),
        "n_tail_points": int(n_tail),
        "F3_SCM": float(f3_scm) if np.isfinite(f3_scm) else np.nan,
        "delta_f3": float(delta_f3) if np.isfinite(delta_f3) else np.nan,
        "beta": float(beta) if np.isfinite(beta) else np.nan,
        "n_beta_points": int(n_beta),
        # Keep both names for compatibility while migrating downstream scripts
        # to the canonical `logSigmaHI_out`.
        "logSigmaHI_out": logsigma_value,
        "logSigmaHI_out_proxy": logsigma_value,
        "quality_flag_tail_ok": bool(np.isfinite(f3_scm) and n_tail >= 4),
        "quality_flag_beta_ok": bool(np.isfinite(beta) and n_beta >= 4),
        "contract_version": "SPARC_MASTER_v1.0",
    }

    if not math.isclose(rmax_kpc, rmax_kpc_check, rel_tol=0.0, abs_tol=1e-10):
        warn(f"Inconsistencia menor en rmax para {galaxy}")

    return row


def enforce_contract(df: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "galaxy",
        "source_file",
        "n_points_curve",
        "rmax_kpc",
        "vmax_obs_kms",
        "tail_frac",
        "n_tail_points",
        "F3_SCM",
        "delta_f3",
        "beta",
        "n_beta_points",
        "logSigmaHI_out",
        "logSigmaHI_out_proxy",
        "quality_flag_tail_ok",
        "quality_flag_beta_ok",
        "contract_version",
    ]
    missing = [c for c in ordered_cols if c not in df.columns]
    if missing:
        fail(f"Contrato roto. Faltan columnas: {missing}")
    return df.loc[:, ordered_cols].copy()


def build_master_table(dat_files: list[Path], tail_frac: float, beta_gbar_max: float) -> pd.DataFrame:
    rows: list[dict] = []
    for i, path in enumerate(dat_files, start=1):
        try:
            rows.append(build_master_row(path, tail_frac=tail_frac, beta_gbar_max=beta_gbar_max))
        except Exception as exc:
            warn(f"[{i}/{len(dat_files)}] Saltando {path.name}: {exc}")

    if not rows:
        fail("No se pudo construir ninguna fila del CSV maestro.")

    df = pd.DataFrame(rows)
    df = df.sort_values("galaxy").reset_index(drop=True)
    return enforce_contract(df)


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


def ingest_from_rotmod(data_root: Path, out: Path, sanity_out: Path | None = None, tail_frac: float = 0.7, beta_gbar_max: float = 0.3, rotmod_subdir: str = "rotmod") -> pd.DataFrame:
    _, dat_files = check_local_sparc_data(data_root, rotmod_subdir=rotmod_subdir)
    info(f"Curvas detectadas: {len(dat_files)}")

    master = build_master_table(dat_files=dat_files, tail_frac=tail_frac, beta_gbar_max=beta_gbar_max)

    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False)
    info(f"CSV maestro escrito en: {out}")

    sanity = build_sanity_summary(master)
    final_sanity = sanity_out or out.with_name(DEFAULT_SANITY_FILENAME)
    final_sanity.parent.mkdir(parents=True, exist_ok=True)
    sanity.to_csv(final_sanity, index=False)
    info(f"Sanity summary escrito en: {final_sanity}")

    return master


# ---------------------------------------------------------------------
# Modo legado (compatibilidad con tests/pipeline previo)
# ---------------------------------------------------------------------
def _check_cols(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in '{source}': {missing}")


def ingest(galaxies_path: Path, rc_points_path: Path, out_dir: Path) -> pd.DataFrame:
    galaxies = read_table(galaxies_path)
    rc = read_table(rc_points_path)

    _check_cols(galaxies, ["galaxy"], galaxies_path)
    _check_cols(rc, ["galaxy", "r_kpc", "vobs_kms", "vobs_err_kms"], rc_points_path)

    if "vbar_kms" not in rc.columns:
        missing_comp = [c for c in ["v_gas", "v_disk"] if c not in rc.columns]
        if missing_comp:
            raise ValueError(
                f"'vbar_kms' not found in {rc_points_path} and component columns {missing_comp} are also missing — cannot derive baryonic velocity."
            )
        v_bul = rc["v_bul"].values if "v_bul" in rc.columns else None
        rc = rc.copy()
        rc["vbar_kms"] = compute_vbar_kms_legacy(rc["v_gas"].values, rc["v_disk"].values, v_bul)

    valid_galaxies = set(galaxies["galaxy"].unique())
    rc_filtered = rc[rc["galaxy"].isin(valid_galaxies)].copy()
    if rc_filtered.empty:
        raise ValueError(
            "After filtering rc_points by the galaxies table the result is empty. "
            "Check that galaxy keys match between both files (including case and whitespace)."
        )

    out_df = rc_filtered[CONTRACT_COLUMNS].copy()
    validate_contract(out_df, source=str(rc_points_path))
    out_df = out_df.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "big_sparc_contract.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestor BIG-SPARC/SPARC contract builder.")

    # Nuevo modo (recomendado)
    parser.add_argument("--data-root", type=Path, default=None, help="Raíz local de datos SPARC (debe contener rotmod/*.dat).")
    parser.add_argument("--rotmod-subdir", type=str, default="rotmod", help="Subdirectorio dentro de data-root con archivos .dat.")
    parser.add_argument("--out", type=Path, required=True, help="Salida principal: CSV maestro (modo nuevo) o directorio (modo legado).")
    parser.add_argument("--sanity-out", type=Path, default=None, help="Ruta opcional para CSV de sanity summary (modo nuevo).")
    parser.add_argument("--tail-frac", type=float, default=0.7, help="Fracción de Rmax para definir cola externa (modo nuevo).")
    parser.add_argument("--beta-gbar-max", type=float, default=0.3, help="Umbral profundo: g_bar < beta_gbar_max * a0 (modo nuevo).")

    # Modo legado
    parser.add_argument("--galaxies", type=Path, default=None, help="Path a tabla galaxies (modo legado).")
    parser.add_argument("--rc-points", dest="rc_points", type=Path, default=None, help="Path a tabla rc_points (modo legado).")

    args = parser.parse_args(argv)

    using_new = args.data_root is not None
    using_legacy = args.galaxies is not None or args.rc_points is not None

    if using_new and using_legacy:
        fail("No mezclar modo nuevo (--data-root) con modo legado (--galaxies/--rc-points).")

    if not using_new and not (args.galaxies and args.rc_points):
        fail("Debes usar modo nuevo (--data-root) o modo legado (--galaxies y --rc-points).")

    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.data_root is not None:
        if not (0.0 < args.tail_frac < 1.0):
            fail("--tail-frac debe estar entre 0 y 1.")
        if args.beta_gbar_max <= 0:
            fail("--beta-gbar-max debe ser positivo.")

        master = ingest_from_rotmod(
            data_root=args.data_root,
            out=args.out,
            sanity_out=args.sanity_out,
            tail_frac=args.tail_frac,
            beta_gbar_max=args.beta_gbar_max,
            rotmod_subdir=args.rotmod_subdir,
        )
        sanity = build_sanity_summary(master)

        print("\n=== RESUMEN ===")
        print(sanity.to_string(index=False))
        print("\nColumnas del contrato:")
        for col in master.columns:
            print(f" - {col}")
        return

    out_df = ingest(args.galaxies, args.rc_points, args.out)
    print(
        f"Ingested {len(out_df['galaxy'].unique())} galaxies, {len(out_df)} rotation-curve points -> {args.out / 'big_sparc_contract.parquet'}"
    )


if __name__ == "__main__":
    main()
