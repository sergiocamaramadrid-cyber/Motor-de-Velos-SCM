"""
Prepare a BIG-SPARC catalog CSV for run_big_sparc_veil_test.py.

This utility normalizes either:
  1) A table that already contains galaxy/g_obs/g_bar, or
  2) A contract-style table with galaxy/r_kpc/vobs_kms/vbar_kms

into a CSV with the columns expected by the veil test runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from scripts.contract_utils import read_table
except ImportError:
    if __name__ == "__main__":
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from scripts.contract_utils import read_table
    else:
        raise


_REQ_DIRECT = {"galaxy", "g_obs", "g_bar"}
_REQ_CONTRACT = {"galaxy", "r_kpc", "vobs_kms", "vbar_kms"}
_KILOPARSEC_TO_METERS = 3.085677581491367e19
_GRAVITATIONAL_CONSTANT = 6.67430e-11
_SOLAR_MASS_KG = 1.98847e30
_PARSEC_TO_METERS = 3.085677581491367e16
_KG_M2_TO_MSUN_PC2 = (_PARSEC_TO_METERS**2) / _SOLAR_MASS_KG


def _compute_accel_from_contract(df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQ_CONTRACT - set(df.columns)
    if missing:
        raise ValueError(
            "Input is missing required columns for conversion. "
            f"Need either {_REQ_DIRECT} or {_REQ_CONTRACT}; missing {sorted(missing)}."
        )

    out = df.copy()
    radius_m = out["r_kpc"].astype(float).to_numpy() * _KILOPARSEC_TO_METERS
    vobs_m_per_s = out["vobs_kms"].astype(float).to_numpy() * 1_000.0
    vbar_m_per_s = out["vbar_kms"].astype(float).to_numpy() * 1_000.0

    valid = (
        np.isfinite(radius_m)
        & np.isfinite(vobs_m_per_s)
        & np.isfinite(vbar_m_per_s)
        & (radius_m > 0.0)
    )
    if not np.any(valid):
        raise ValueError(
            "No valid rows available to compute accelerations: require finite "
            "r_kpc, finite velocities, and r_kpc > 0."
        )

    g_obs = np.full(len(out), np.nan, dtype=float)
    g_bar = np.full(len(out), np.nan, dtype=float)
    g_obs[valid] = (vobs_m_per_s[valid] ** 2) / radius_m[valid]
    g_bar[valid] = (vbar_m_per_s[valid] ** 2) / radius_m[valid]

    out["g_obs"] = g_obs
    out["g_bar"] = g_bar
    return out


def prepare_catalog(input_path: Path, out_path: Path) -> pd.DataFrame:
    df = read_table(input_path)

    if not _REQ_DIRECT.issubset(df.columns):
        df = _compute_accel_from_contract(df)

    cols = ["galaxy", "g_obs", "g_bar"]
    for opt in ("logMbar", "logSigmaHI_out"):
        if opt in df.columns:
            cols.append(opt)

    out = df[cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["galaxy", "g_obs", "g_bar"])
    out = out[(out["g_obs"] > 0.0) & (out["g_bar"] > 0.0)]
    out = out.sort_values(["galaxy"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _read_rotmod_for_catalog(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if raw.shape[1] >= 6:
        # SPARC standard layout: r, vobs, err, vgas, vdisk, vbul, ...
        cols = {"r_kpc": 0, "v_obs": 1, "v_gas": 3, "v_disk": 4, "v_bul": 5}
    elif raw.shape[1] == 5:
        # Simplified layout: r, vobs, vgas, vdisk, vbul
        cols = {"r_kpc": 0, "v_obs": 1, "v_gas": 2, "v_disk": 3, "v_bul": 4}
    else:
        raise ValueError(
            f"Unsupported rotmod format in {path}: expected 5 or >=6 columns, got {raw.shape[1]}"
        )

    return pd.DataFrame({name: raw.iloc[:, idx].astype(float) for name, idx in cols.items()})


def _derive_env_columns(rc: pd.DataFrame) -> tuple[float, float]:
    valid = rc.replace([np.inf, -np.inf], np.nan).dropna(subset=["r_kpc", "v_gas", "v_disk", "v_bul"])
    valid = valid[valid["r_kpc"] > 0.0]
    if valid.empty:
        return float("nan"), float("nan")

    r_m = valid["r_kpc"].to_numpy(dtype=float) * _KILOPARSEC_TO_METERS
    vgas = valid["v_gas"].to_numpy(dtype=float) * 1_000.0
    vdisk = valid["v_disk"].to_numpy(dtype=float) * 1_000.0
    vbul = valid["v_bul"].to_numpy(dtype=float) * 1_000.0

    g_bar = (vgas**2 + vdisk**2 + vbul**2) / r_m
    g_gas = (vgas**2) / r_m
    i_out = int(np.argmax(r_m))
    mbar_kg = g_bar[i_out] * (r_m[i_out] ** 2) / _GRAVITATIONAL_CONSTANT
    sigma_gas_kg_m2 = g_gas[i_out] / (2.0 * np.pi * _GRAVITATIONAL_CONSTANT)
    sigma_gas_msun_pc2 = sigma_gas_kg_m2 * _KG_M2_TO_MSUN_PC2

    log_mbar = float(np.log10(mbar_kg / _SOLAR_MASS_KG)) if mbar_kg > 0 else float("nan")
    log_sigma_hi_out = float(np.log10(sigma_gas_msun_pc2)) if sigma_gas_msun_pc2 > 0 else float("nan")
    return log_mbar, log_sigma_hi_out


def prepare_catalog_from_sparc_dir(
    sparc_dir: Path,
    out_path: Path,
    galaxies: list[str] | None = None,
) -> pd.DataFrame:
    all_rotmods = {
        p.name.replace("_rotmod.dat", ""): p
        for base in (sparc_dir, sparc_dir / "raw")
        for p in base.glob("*_rotmod.dat")
    }
    if not all_rotmods:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {sparc_dir} or {sparc_dir / 'raw'}")

    target_galaxies = galaxies if galaxies else sorted(all_rotmods)
    rows = []
    for galaxy in target_galaxies:
        path = all_rotmods.get(galaxy)
        if path is None:
            continue

        rc = _read_rotmod_for_catalog(path)
        rc["galaxy"] = galaxy

        radius_m = rc["r_kpc"].to_numpy(dtype=float) * _KILOPARSEC_TO_METERS
        v_obs = rc["v_obs"].to_numpy(dtype=float) * 1_000.0
        v_gas = rc["v_gas"].to_numpy(dtype=float) * 1_000.0
        v_disk = rc["v_disk"].to_numpy(dtype=float) * 1_000.0
        v_bul = rc["v_bul"].to_numpy(dtype=float) * 1_000.0

        valid = np.isfinite(radius_m) & np.isfinite(v_obs) & np.isfinite(v_gas) & np.isfinite(v_disk) & np.isfinite(v_bul) & (radius_m > 0.0)
        if not np.any(valid):
            continue

        g_obs = (v_obs[valid] ** 2) / radius_m[valid]
        g_bar = (v_gas[valid] ** 2 + v_disk[valid] ** 2 + v_bul[valid] ** 2) / radius_m[valid]
        log_mbar, log_sigma_hi_out = _derive_env_columns(rc.loc[valid, ["r_kpc", "v_gas", "v_disk", "v_bul"]])

        rows.append(
            pd.DataFrame(
                {
                    "galaxy": galaxy,
                    "g_obs": g_obs,
                    "g_bar": g_bar,
                    "logMbar": log_mbar,
                    "logSigmaHI_out": log_sigma_hi_out,
                }
            )
        )

    if not rows:
        raise ValueError("No valid rows were generated from provided SPARC rotmod files.")

    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["galaxy", "g_obs", "g_bar"])
    out = out[(out["g_obs"] > 0.0) & (out["g_bar"] > 0.0)]
    out = out.sort_values(["galaxy"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data/big_sparc_catalog.csv for the BIG-SPARC veil test."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Input CSV/Parquet table.")
    source.add_argument(
        "--sparc-dir",
        help="Directory containing SPARC *_rotmod.dat files (optionally in ./raw).",
    )
    parser.add_argument(
        "--galaxies",
        help="Optional comma-separated galaxy names to include when --sparc-dir is used.",
    )
    parser.add_argument(
        "--out",
        default="data/big_sparc_catalog.csv",
        help="Output CSV path (default: data/big_sparc_catalog.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.input:
        out = prepare_catalog(Path(args.input), Path(args.out))
    else:
        galaxy_list = None
        if args.galaxies:
            galaxy_list = [name.strip() for name in args.galaxies.split(",") if name.strip()]
        out = prepare_catalog_from_sparc_dir(Path(args.sparc_dir), Path(args.out), galaxies=galaxy_list)
    print(f"Wrote {len(out)} rows to {Path(args.out)}")


if __name__ == "__main__":
    main()
