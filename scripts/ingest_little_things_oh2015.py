#!/usr/bin/env python3
"""
Ingesta reproducible de LITTLE THINGS (Oh+2015, J/AJ/149/180)
para el Framework SCM-Motor de Velos.

Entradas esperadas:
- data/LITTLE_THINGS_Oh2015/table1.dat
- data/LITTLE_THINGS_Oh2015/table2.dat
- data/LITTLE_THINGS_Oh2015/rotdmbar.dat

Salidas:
- results/LITTLE_THINGS_Oh2015/little_things_galaxy_table.csv
- results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_float(x: str) -> float:
    x = x.strip().replace(",", ".")
    if not x:
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def _parse_str(x: str) -> str:
    return x.strip()


def _read_slice(line: str, start: int, end: int) -> str:
    return line[start:end] if len(line) >= end else line[start:]


def _likely_fixed_width(values: dict[str, float]) -> bool:
    finite_count = sum(np.isfinite(v) for v in values.values())
    return finite_count >= 3


def _fallback_galaxy_from_tokens(tokens: list[str]) -> str:
    if len(tokens) >= 2 and tokens[0].isalpha() and any(c.isdigit() for c in tokens[1]):
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0] if tokens else ""


def read_table1(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            row = {
                "galaxy": _parse_str(_read_slice(line, 0, 8)),
                "dist_mpc": _parse_float(_read_slice(line, 32, 36)),
                "vsys_kms": _parse_float(_read_slice(line, 37, 43)),
                "vsys_err_kms": _parse_float(_read_slice(line, 44, 47)),
                "pa_deg": _parse_float(_read_slice(line, 48, 53)),
                "pa_err_deg": _parse_float(_read_slice(line, 54, 58)),
                "incl_deg": _parse_float(_read_slice(line, 59, 63)),
                "incl_err_deg": _parse_float(_read_slice(line, 64, 68)),
                "vmag_abs": _parse_float(_read_slice(line, 69, 74)),
                "oh_12log": _parse_float(_read_slice(line, 75, 78)),
                "oh_err": _parse_float(_read_slice(line, 79, 83)),
                "logsfr_ha": _parse_float(_read_slice(line, 84, 89)),
                "logsfr_ha_err": _parse_float(_read_slice(line, 90, 94)),
                "logsfr_fuv": _parse_float(_read_slice(line, 95, 100)),
                "logsfr_fuv_err": _parse_float(_read_slice(line, 101, 105)),
            }

            if not _likely_fixed_width(
                {
                    "dist_mpc": row["dist_mpc"],
                    "vsys_kms": row["vsys_kms"],
                    "pa_deg": row["pa_deg"],
                    "incl_deg": row["incl_deg"],
                }
            ):
                tokens = line.split()
                row["galaxy"] = _fallback_galaxy_from_tokens(tokens)

            rows.append(row)
    return pd.DataFrame(rows)


def read_table2(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            row = {
                "galaxy": _parse_str(_read_slice(line, 0, 8)),
                "rmax_kpc": _parse_float(_read_slice(line, 9, 13)),
                "r03_kpc": _parse_float(_read_slice(line, 14, 18)),
                "v_rmax_kms": _parse_float(_read_slice(line, 19, 24)),
                "viso_rmax_kms": _parse_float(_read_slice(line, 25, 30)),
                "rmax_over_hibeam": _parse_float(_read_slice(line, 31, 36)),
                "z0_kpc": _parse_float(_read_slice(line, 37, 41)),
                "c_nfw": _parse_float(_read_slice(line, 42, 46)),
                "c_nfw_err": _parse_float(_read_slice(line, 47, 53)),
                "c_m07": _parse_float(_read_slice(line, 54, 58)),
                "v200_kms": _parse_float(_read_slice(line, 59, 65)),
                "v200_err_kms": _parse_float(_read_slice(line, 66, 73)),
                "v200m07_kms": _parse_float(_read_slice(line, 74, 79)),
                "v200m07_err_kms": _parse_float(_read_slice(line, 80, 84)),
                "rc_kpc": _parse_float(_read_slice(line, 85, 89)),
                "rc_err_kpc": _parse_float(_read_slice(line, 90, 95)),
                "rho0_1e3_msun_pc3": _parse_float(_read_slice(line, 96, 103)),
                "rho0_err_1e3_msun_pc3": _parse_float(_read_slice(line, 104, 110)),
                "alphamin": _parse_float(_read_slice(line, 111, 116)),
                "alphamin_err": _parse_float(_read_slice(line, 117, 122)),
                "alphamin_flag": _parse_str(_read_slice(line, 123, 124)),
                "alpha36": _parse_float(_read_slice(line, 125, 130)),
                "alpha36_err": _parse_float(_read_slice(line, 131, 136)),
                "alpha36_flag": _parse_str(_read_slice(line, 137, 138)),
                "mgas_1e7_msun": _parse_float(_read_slice(line, 139, 145)),
                "mstar_k_1e7_msun": _parse_float(_read_slice(line, 146, 151)),
                "mstar_sed_1e7_msun": _parse_float(_read_slice(line, 152, 157)),
                "logmdyn": _parse_float(_read_slice(line, 158, 163)),
                "logm200": _parse_float(_read_slice(line, 164, 171)),
            }

            if not _likely_fixed_width(
                {
                    "rmax_kpc": row["rmax_kpc"],
                    "r03_kpc": row["r03_kpc"],
                    "v_rmax_kms": row["v_rmax_kms"],
                    "z0_kpc": row["z0_kpc"],
                }
            ):
                tokens = line.split()
                row["galaxy"] = _fallback_galaxy_from_tokens(tokens)

            rows.append(row)
    return pd.DataFrame(rows)


def read_rotdmbar(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            galaxy = _parse_str(_read_slice(line, 0, 8))
            data_type = _parse_str(_read_slice(line, 9, 14))
            r03_kpc = _parse_float(_read_slice(line, 15, 23))
            v03_kms = _parse_float(_read_slice(line, 24, 34))
            r_scaled = _parse_float(_read_slice(line, 35, 44))
            v_scaled = _parse_float(_read_slice(line, 45, 54))
            ev_scaled = _parse_float(_read_slice(line, 55, 63))

            if data_type.lower() not in {"data", "model"}:
                tokens = line.split()
                type_idx = next(
                    (i for i, tok in enumerate(tokens) if tok.lower() in {"data", "model"}),
                    None,
                )
                if type_idx is not None:
                    galaxy = " ".join(tokens[:type_idx])
                    data_type = tokens[type_idx]
                    numeric = tokens[type_idx + 1 :]
                    if len(numeric) >= 5:
                        r03_kpc = _parse_float(numeric[0])
                        v03_kms = _parse_float(numeric[1])
                        r_scaled = _parse_float(numeric[2])
                        v_scaled = _parse_float(numeric[3])
                        ev_scaled = _parse_float(numeric[4])

            rows.append(
                {
                    "galaxy": galaxy,
                    "data_type": data_type,
                    "r03_kpc": r03_kpc,
                    "v03_kms": v03_kms,
                    "r_scaled": r_scaled,
                    "v_scaled": v_scaled,
                    "ev_scaled": ev_scaled,
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["data_type"].str.lower() == "data"].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def build_outputs(data_root: Path, out_root: Path) -> None:
    table1_path = data_root / "table1.dat"
    table2_path = data_root / "table2.dat"
    rotdmbar_path = data_root / "rotdmbar.dat"

    missing = [p for p in [table1_path, table2_path, rotdmbar_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos requeridos:\n" + "\n".join(str(p) for p in missing)
        )

    out_root.mkdir(parents=True, exist_ok=True)

    t1 = read_table1(table1_path)
    t2 = read_table2(table2_path)
    rot = read_rotdmbar(rotdmbar_path)

    galaxy = pd.merge(t1, t2, on="galaxy", how="outer", validate="one_to_one")
    galaxy = galaxy.sort_values("galaxy").reset_index(drop=True)

    galaxy.to_csv(out_root / "little_things_galaxy_table.csv", index=False)
    rot.to_csv(out_root / "little_things_rotcurves.csv", index=False)

    summary = {
        "n_galaxies": int(galaxy["galaxy"].nunique()),
        "n_rotcurve_points": int(len(rot)),
        "n_rotcurve_galaxies": int(rot["galaxy"].nunique()),
        "output_dir": str(out_root),
    }
    pd.Series(summary).to_json(out_root / "ingest_summary.json", indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/LITTLE_THINGS_Oh2015"),
        help="Directorio con table1.dat, table2.dat y rotdmbar.dat",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/LITTLE_THINGS_Oh2015"),
        help="Directorio de salida",
    )
    args = parser.parse_args()
    build_outputs(args.data_root, args.out_root)


if __name__ == "__main__":
    main()
