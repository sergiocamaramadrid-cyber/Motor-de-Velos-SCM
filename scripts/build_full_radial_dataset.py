#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build full SPARC radial dataset by merging:
- *_rotmod.dat files
- *.dens files
- SPARC_Lelli2016c.mrt metadata

Output:
    data/sparc_full_radial.csv
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

KPC_TO_M = 3.085677581e19
EPS = 1e-30


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_metadata(mrt_path: str) -> pd.DataFrame:
    path = Path(mrt_path)
    if path.suffix.lower() == ".mrt":
        tab = Table.read(path, format="ascii.mrt")
        df = tab.to_pandas()
    else:
        df = pd.read_csv(path)

    galaxy_col = None
    for c in df.columns:
        if c.lower() in {"galaxy", "name", "gal"}:
            galaxy_col = c
            break
    if galaxy_col is None:
        raise ValueError("No se encontró columna identificadora de galaxia en el metadata.")

    df = df.rename(columns={galaxy_col: "galaxy"})
    df["galaxy"] = df["galaxy"].astype(str).str.strip()
    return df


def _decoded_member_lines(raw_bytes: bytes) -> list[str]:
    raw = raw_bytes.decode("utf-8", errors="ignore")
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def read_rotmod_from_zip(zip_path: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith("_rotmod.dat")]

        for member in members:
            galaxy = Path(member).name.replace("_rotmod.dat", "")
            with zf.open(member) as fh:
                lines = _decoded_member_lines(fh.read())
            if not lines:
                continue

            data = np.loadtxt(io.StringIO("\n".join(lines)))
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 6:
                continue

            r = data[:, 0]
            vobs = data[:, 1]
            evobs = data[:, 2]
            vgas = data[:, 3]
            vdisk = data[:, 4]
            vbul = data[:, 5]

            vbar = np.sqrt(np.maximum(vgas**2 + vdisk**2 + vbul**2, 0.0))
            r_m = np.maximum(r * KPC_TO_M, EPS)
            gobs = (vobs * 1000.0) ** 2 / r_m
            gbar = (vbar * 1000.0) ** 2 / r_m

            for i in range(len(r)):
                if not np.isfinite(r[i]) or r[i] <= 0:
                    continue
                rows.append(
                    {
                        "galaxy": galaxy,
                        "r": float(r[i]),
                        "Vobs": float(vobs[i]),
                        "eVobs": float(evobs[i]),
                        "Vgas": float(vgas[i]),
                        "Vdisk": float(vdisk[i]),
                        "Vbul": float(vbul[i]),
                        "Vbar": float(vbar[i]),
                        "gobs": float(gobs[i]),
                        "gbar": float(gbar[i]),
                    }
                )

    return pd.DataFrame(rows)


def _guess_dens_columns(df: pd.DataFrame) -> tuple[str, str]:
    r_col = None
    sb_col = None
    for c in df.columns:
        cl = c.lower()
        if r_col is None and ("rad" in cl or cl == "r"):
            r_col = c
        if sb_col is None and (
            "sb" in cl or "surf" in cl or "mu" in cl or "dens" in cl or "sigma" in cl
        ):
            sb_col = c

    if r_col is None:
        r_col = df.columns[0]
    if sb_col is None:
        sb_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return r_col, sb_col


def read_dens_zip(zip_path: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".dens")]

        for member in members:
            galaxy = Path(member).name.replace(".dens", "")
            with zf.open(member) as fh:
                lines = _decoded_member_lines(fh.read())
            if not lines:
                continue

            try:
                data = np.genfromtxt(
                    io.StringIO("\n".join(lines)),
                    names=True,
                    dtype=None,
                    encoding=None,
                )
                if getattr(data, "dtype", None) is None or data.dtype.names is None:
                    raise ValueError("density table missing named columns")
                names = list(data.dtype.names)
                def _is_numeric_name(token: str) -> bool:
                    try:
                        float(token)
                        return True
                    except ValueError:
                        return False
                if all(_is_numeric_name(n) for n in names):
                    raise ValueError("numeric first row interpreted as header")
                if np.shape(data) == ():
                    df = pd.DataFrame([tuple(data.tolist())], columns=names)
                else:
                    df = pd.DataFrame(data)
                if df.empty:
                    raise ValueError("empty density table")
            except Exception:
                df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", header=None, engine="python")
                if df.empty or df.shape[1] < 2:
                    continue
                # Drop a potential text header line if present.
                df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
                if df.empty or df.shape[1] < 2:
                    continue
                df = df.iloc[:, :2].copy()
                df.columns = ["r", "SB"]

            r_col, sb_col = _guess_dens_columns(df)
            for _, row in df.iterrows():
                r = row[r_col]
                sb = row[sb_col]
                if not np.isfinite(r) or not np.isfinite(sb) or r <= 0 or sb <= 0:
                    continue
                rows.append({"galaxy": galaxy, "r": float(r), "SB": float(sb)})

    return pd.DataFrame(rows)


def merge_radial(rot_df: pd.DataFrame, dens_df: pd.DataFrame) -> pd.DataFrame:
    if dens_df.empty:
        out = rot_df.copy()
        out["SB"] = np.maximum(out["Vdisk"] ** 2, 1e-6)
        return out

    merged_rows: list[pd.DataFrame] = []
    for galaxy, g_rot in rot_df.groupby("galaxy", sort=False):
        g_rot = g_rot.sort_values("r")
        g_dens = dens_df[dens_df["galaxy"] == galaxy].sort_values("r")

        if g_dens.empty:
            tmp = g_rot.copy()
            tmp["SB"] = np.maximum(tmp["Vdisk"] ** 2, 1e-6)
            merged_rows.append(tmp)
            continue

        dens_r = g_dens["r"].to_numpy(dtype=float)
        dens_sb = g_dens["SB"].to_numpy(dtype=float)
        tmp = g_rot.copy()
        tmp["SB"] = np.interp(
            tmp["r"].to_numpy(dtype=float),
            dens_r,
            dens_sb,
            left=np.nan,
            right=np.nan,
        )
        tmp["SB"] = tmp["SB"].fillna(np.maximum(tmp["Vdisk"] ** 2, 1e-6))
        merged_rows.append(tmp)

    return pd.concat(merged_rows, ignore_index=True)


def attach_metadata(radial_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    meta = meta_df.copy()
    rename_map: dict[str, str] = {}
    for c in meta.columns:
        cl = c.lower()
        if cl in {"type", "t"}:
            rename_map[c] = "type"
        elif cl in {"rdisk", "rd", "r_d"}:
            rename_map[c] = "Rdisk"
        elif cl in {"incl", "inclination", "inc"}:
            rename_map[c] = "inclination"
    meta = meta.rename(columns=rename_map)

    keep = ["galaxy"] + [c for c in ["type", "Mbar", "logMbar", "logM", "Rdisk", "inclination"] if c in meta.columns]
    meta = meta[[c for c in keep if c in meta.columns]].drop_duplicates(subset=["galaxy"])

    return radial_df.merge(meta, on="galaxy", how="left")


def compute_f3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gbar_safe = np.maximum(out["gbar"].to_numpy(dtype=float), EPS)
    out["F3"] = (out["gobs"].to_numpy(dtype=float) - gbar_safe) / gbar_safe
    return out


def build_full_radial_dataset(
    rotmod_zip: str,
    ltg_dens_zip: str,
    etg_dens_zip: str,
    metadata_mrt: str,
    output: str,
) -> pd.DataFrame:
    ensure_dir(os.path.dirname(output) or ".")

    meta_df = load_metadata(metadata_mrt)
    rot_df = read_rotmod_from_zip(rotmod_zip)
    ltg_df = read_dens_zip(ltg_dens_zip)
    etg_df = read_dens_zip(etg_dens_zip)
    dens_df = pd.concat([ltg_df, etg_df], ignore_index=True)

    radial_df = merge_radial(rot_df, dens_df)
    radial_df = attach_metadata(radial_df, meta_df)
    radial_df = compute_f3(radial_df)

    radial_df = radial_df.replace([np.inf, -np.inf], np.nan)
    radial_df = radial_df.dropna(subset=["galaxy", "r", "gobs", "gbar", "SB", "F3"])
    radial_df = radial_df[radial_df["r"] > 0].copy()
    radial_df = radial_df.sort_values(["galaxy", "r"]).reset_index(drop=True)

    radial_df.to_csv(output, index=False)
    return radial_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full SPARC radial dataset.")
    parser.add_argument("--rotmod-zip", required=True)
    parser.add_argument("--ltg-dens-zip", required=True)
    parser.add_argument("--etg-dens-zip", required=True)
    parser.add_argument("--metadata-mrt", required=True)
    parser.add_argument("--output", default="data/sparc_full_radial.csv")
    args = parser.parse_args()

    radial_df = build_full_radial_dataset(
        rotmod_zip=args.rotmod_zip,
        ltg_dens_zip=args.ltg_dens_zip,
        etg_dens_zip=args.etg_dens_zip,
        metadata_mrt=args.metadata_mrt,
        output=args.output,
    )

    print(f"\nSaved: {args.output}")
    print(f"Rows: {len(radial_df)}")
    print(f"Galaxies: {radial_df['galaxy'].nunique()}")
    print(f"NaN total: {int(radial_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
