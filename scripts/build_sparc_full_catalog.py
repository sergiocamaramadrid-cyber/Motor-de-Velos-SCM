#!/usr/bin/env python3
"""
Build a full SPARC catalog ready for BIG-SPARC veil tests.

Output columns:
    galaxy, r_kpc, g_obs, g_bar, logMbar, logSigmaHI_out
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_URL = "https://zenodo.org/records/16284118/files/Rotmod_LTG.zip?download=1"
MRT_URL = "https://zenodo.org/records/16284118/files/SPARC_Lelli2016c.mrt?download=1"

KM_TO_M = 1_000.0
KPC_TO_M = 3.085677581e19
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7
MRT_HEADER_LINES = 60
MIN_ROTMOD_COLUMNS = 5


def norm_name(value: str) -> str:
    return str(value).strip().upper().replace(" ", "")


def _find_existing_rotmod_files(data_root: Path, repo_root: Path, rot_dir: Path | None = None) -> list[Path]:
    candidates = []
    search_roots = [data_root, repo_root / "data"]
    if rot_dir is not None:
        search_roots.append(rot_dir)
    seen_roots: set[Path] = set()
    for base in search_roots:
        resolved_base = base.resolve()
        if resolved_base in seen_roots:
            continue
        seen_roots.add(resolved_base)
        if base.exists():
            candidates.extend(base.rglob("*_rotmod.dat"))
    unique = sorted({path.resolve() for path in candidates})
    return unique


def _find_existing_master_table(data_root: Path, repo_root: Path) -> Path | None:
    candidates = [
        data_root / "SPARC_Lelli2016c.mrt",
        repo_root / "data" / "SPARC_Lelli2016c.mrt",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def download_file(url: str, outpath: Path, timeout: int = 60, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                outpath.write_bytes(response.read())
            return
        except TimeoutError as exc:
            last_error = exc
            print(f"[WARN] Download attempt {attempt}/{retries} timed out for {url}")
        except urllib.error.URLError as exc:
            last_error = exc
            print(f"[WARN] Download attempt {attempt}/{retries} failed for {url}: network error {exc}")
    raise RuntimeError(f"Unable to download {url} after {retries} attempts") from last_error


def download_and_extract_zip(url: str, zip_path: Path, extract_dir: Path) -> None:
    download_file(url, zip_path, timeout=120, retries=3)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def load_master_table(mrt_file: Path) -> pd.DataFrame:
    # Fixed-width schema from SPARC_Lelli2016c.mrt data rows.
    col_widths = [11, 2, 6, 5, 2, 4, 4, 7, 7, 5, 6, 5, 6, 7, 5, 5, 5, 3, 14]
    names = [
        "Galaxy", "T", "D", "e_D", "f_D", "Inc", "e_Inc", "L_3.6", "e_L_3.6",
        "Reff", "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat",
        "Q", "Ref",
    ]

    df = pd.read_fwf(
        mrt_file,
        widths=col_widths,
        names=names,
        skiprows=MRT_HEADER_LINES,
        na_values=["...", "....", "....."],
    )
    df = df[df["Galaxy"].notna()].copy()
    df["Galaxy"] = df["Galaxy"].astype(str).str.strip()
    df["Galaxy_norm"] = df["Galaxy"].apply(norm_name)
    return df


def add_master_derived_columns(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()

    df["Mstar_1e9"] = UPSILON_DISK * pd.to_numeric(df["L_3.6"], errors="coerce")
    df["Mgas_1e9"] = 1.33 * pd.to_numeric(df["MHI"], errors="coerce")
    df["Mbar_1e9"] = df["Mstar_1e9"] + df["Mgas_1e9"]

    df["logMbar"] = np.nan
    mask_mbar = df["Mbar_1e9"] > 0
    df.loc[mask_mbar, "logMbar"] = np.log10(df.loc[mask_mbar, "Mbar_1e9"]) + 9.0

    rhi = pd.to_numeric(df["RHI"], errors="coerce")
    mhi = pd.to_numeric(df["MHI"], errors="coerce")
    mask_hi = (rhi > 0) & (mhi > 0)
    df["SigmaHI_out"] = np.nan
    df.loc[mask_hi, "SigmaHI_out"] = (mhi[mask_hi] * 1e9) / (np.pi * (rhi[mask_hi] ** 2) * 1e6)

    df["logSigmaHI_out"] = np.nan
    mask_sigma = df["SigmaHI_out"] > 0
    df.loc[mask_sigma, "logSigmaHI_out"] = np.log10(df.loc[mask_sigma, "SigmaHI_out"])

    return df


def process_rotmod(file_path: Path, galaxy_params: dict[str, dict[str, float]]) -> pd.DataFrame:
    galaxy = file_path.name.replace("_rotmod.dat", "")
    galaxy_norm = norm_name(galaxy)
    try:
        data = np.loadtxt(file_path, comments="#")
    except Exception as exc:
        print(f"[WARN] Could not read {file_path}: {exc}")
        return pd.DataFrame()

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < MIN_ROTMOD_COLUMNS:
        print(
            f"[WARN] File has too few columns: {file_path} "
            f"(found {data.shape[1]}, expected at least {MIN_ROTMOD_COLUMNS})"
        )
        return pd.DataFrame()

    r = data[:, 0]
    vobs = data[:, 1]
    # SPARC rotmod files commonly come as:
    #   6+ cols: r, vobs, vobs_err, vgas, vdisk, vbul, ...
    #   5 cols : r, vobs, vgas, vdisk, vbul
    if data.shape[1] > 5:
        vgas = data[:, 3]
        vdisk = data[:, 4]
        vbul = data[:, 5]
    else:
        vgas = data[:, 2]
        vdisk = data[:, 3]
        vbul = data[:, 4]

    vdisk = vdisk * np.sqrt(UPSILON_DISK)
    vbul = vbul * np.sqrt(UPSILON_BULGE)

    mask = (
        np.isfinite(r)
        & np.isfinite(vobs)
        & np.isfinite(vgas)
        & np.isfinite(vdisk)
        & np.isfinite(vbul)
        & (r > 0)
    )
    if not np.any(mask):
        return pd.DataFrame()

    r = r[mask]
    r_m = r * KPC_TO_M
    vobs_ms = vobs[mask] * KM_TO_M
    vgas_ms = vgas[mask] * KM_TO_M
    vdisk_ms = vdisk[mask] * KM_TO_M
    vbul_ms = vbul[mask] * KM_TO_M

    g_obs = (vobs_ms**2) / r_m
    g_bar = (vgas_ms**2 + vdisk_ms**2 + vbul_ms**2) / r_m

    out = pd.DataFrame({"galaxy": galaxy, "r_kpc": r, "g_obs": g_obs, "g_bar": g_bar})
    if galaxy_norm in galaxy_params:
        out["logMbar"] = galaxy_params[galaxy_norm]["logMbar"]
        out["logSigmaHI_out"] = galaxy_params[galaxy_norm]["logSigmaHI_out"]
    else:
        out["logMbar"] = np.nan
        out["logSigmaHI_out"] = np.nan
        print(f"[WARN] Galaxy not found in master table: {galaxy} (normalized: {galaxy_norm})")
    return out


def build_catalog(data_root: Path, out_csv: Path) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parent.parent
    data_root.mkdir(parents=True, exist_ok=True)
    rot_dir = data_root / "rotmod"
    zip_path = data_root / "Rotmod_LTG.zip"
    mrt_path = _find_existing_master_table(data_root, repo_root)
    files = _find_existing_rotmod_files(data_root, repo_root, rot_dir=rot_dir)

    if not files:
        print(f"Downloading and extracting {ZIP_URL}")
        download_and_extract_zip(ZIP_URL, zip_path, rot_dir)
        files = _find_existing_rotmod_files(data_root, repo_root, rot_dir=rot_dir)
    else:
        print(f"Using {len(files)} existing rotmod files found in repository data paths")

    if mrt_path is None:
        mrt_path = data_root / "SPARC_Lelli2016c.mrt"
        print(f"Downloading {MRT_URL}")
        download_file(MRT_URL, mrt_path, timeout=60, retries=3)
    else:
        print(f"Using existing master table: {mrt_path}")

    master_df = add_master_derived_columns(load_master_table(mrt_path))
    galaxy_params = master_df.set_index("Galaxy_norm")[["logMbar", "logSigmaHI_out"]].to_dict(orient="index")

    if not files:
        raise FileNotFoundError(
            f"No *_rotmod.dat files found under {data_root}, {repo_root / 'data'}, or {rot_dir}"
        )

    rows = [df for df in (process_rotmod(path, galaxy_params) for path in files) if not df.empty]
    if not rows:
        raise RuntimeError("No valid rotation-curve files were processed.")

    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["g_obs", "g_bar"])
    out = out[(out["g_obs"] > 0) & (out["g_bar"] > 0)].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full SPARC CSV for BIG-SPARC veil test.")
    parser.add_argument("--data-root", default="data/SPARC", help="SPARC data directory (default: data/SPARC).")
    parser.add_argument("--out", default="data/SPARC/sparc_full.csv", help="Output CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out = build_catalog(Path(args.data_root), Path(args.out))
    print(f"CSV generated: {Path(args.out)}")
    print(f"Rows: {len(out)}")
    print(f"Unique galaxies: {out['galaxy'].nunique()}")
    print(f"Columns: {list(out.columns)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
