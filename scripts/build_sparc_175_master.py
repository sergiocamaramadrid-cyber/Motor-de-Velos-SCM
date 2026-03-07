from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


REQUIRED_COLUMNS = [
    "galaxy",
    "deep_slope",
    "delta_f3",
    "n_tail_points",
    "tail_r_min",
    "tail_r_max",
    "logSigmaHI_out",
    "logMstar",
    "logRd",
    "inclination",
]

REQUIRED_SUPPORT_FILES = (
    "MassModels_Lelli2016c.mrt",
    "RAR.mrt",
    "RARbins.mrt",
    "CDR_Lelli2016b.mrt",
)

METADATA_TABLE_CANDIDATES = (
    "SPARC_Lelli2016c.mrt",
    "SPARC_Lelli2016c.csv",
    "sparc_master_catalog.csv",
)


def _norm_galaxy(value: object) -> str:
    return str(value).strip().upper().replace(" ", "")


def _resolve_metadata_dir(sparc_dir: Path) -> Path:
    metadata_dir = sparc_dir / "metadata"
    return metadata_dir if metadata_dir.exists() else sparc_dir


def _require_support_files(metadata_dir: Path) -> None:
    missing = [name for name in REQUIRED_SUPPORT_FILES if not (metadata_dir / name).exists()]
    if missing:
        expected = ", ".join(REQUIRED_SUPPORT_FILES)
        raise FileNotFoundError(
            "Missing SPARC support table(s): "
            f"{', '.join(missing)}. Expected in {metadata_dir}: {expected}"
        )


def _find_metadata_table(sparc_dir: Path, metadata_dir: Path) -> Path:
    for name in METADATA_TABLE_CANDIDATES:
        for base in (metadata_dir, sparc_dir):
            candidate = base / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Could not find a metadata table with stellar/HI columns. "
        f"Tried {list(METADATA_TABLE_CANDIDATES)} in {metadata_dir} and {sparc_dir}."
    )


def _load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    galaxy_col = "Galaxy" if "Galaxy" in meta.columns else "galaxy" if "galaxy" in meta.columns else None
    if galaxy_col is None:
        raise ValueError(f"Metadata table {path} must include 'Galaxy' or 'galaxy'.")

    mhi_col = "MHI" if "MHI" in meta.columns else "mhi" if "mhi" in meta.columns else None
    rhi_col = "RHI" if "RHI" in meta.columns else "rhi" if "rhi" in meta.columns else None
    l36_col = "L_3.6" if "L_3.6" in meta.columns else "L36" if "L36" in meta.columns else None
    rd_col = "Rdisk" if "Rdisk" in meta.columns else "rdisk" if "rdisk" in meta.columns else None
    inc_col = "Inc" if "Inc" in meta.columns else "inclination" if "inclination" in meta.columns else None

    missing = [
        name
        for name, col in [
            ("MHI", mhi_col),
            ("RHI", rhi_col),
            ("L_3.6/L36", l36_col),
            ("Rdisk", rd_col),
            ("Inc/inclination", inc_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Metadata table {path} is missing required columns: {missing}"
        )

    out = pd.DataFrame(
        {
            "galaxy": meta[galaxy_col].astype(str).str.strip(),
            "MHI": pd.to_numeric(meta[mhi_col], errors="coerce"),
            "RHI": pd.to_numeric(meta[rhi_col], errors="coerce"),
            "L_3.6": pd.to_numeric(meta[l36_col], errors="coerce"),
            "Rdisk": pd.to_numeric(meta[rd_col], errors="coerce"),
            "inclination": pd.to_numeric(meta[inc_col], errors="coerce"),
        }
    )
    out["galaxy_norm"] = out["galaxy"].map(_norm_galaxy)

    sigma = (out["MHI"] * 1e9) / (np.pi * (out["RHI"] ** 2) * 1e6)
    out["logSigmaHI_out"] = np.where(sigma > 0, np.log10(sigma), np.nan)
    out["logMstar"] = np.where(out["L_3.6"] > 0, np.log10(0.5 * out["L_3.6"]) + 9.0, np.nan)
    out["logRd"] = np.where(out["Rdisk"] > 0, np.log10(out["Rdisk"]), np.nan)

    return out[["galaxy_norm", "logSigmaHI_out", "logMstar", "logRd", "inclination"]].drop_duplicates("galaxy_norm")


def _iter_rotmod_files(rotmod_dir: Path) -> list[Path]:
    if not rotmod_dir.exists():
        raise FileNotFoundError(f"Missing rotmod directory: {rotmod_dir}")
    files = sorted(rotmod_dir.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"No .dat files found in {rotmod_dir}")
    return files


def _slope_from_tail(path: Path, tail_points: int, min_tail_points: int) -> dict | None:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        return None

    r = data[:, 0]
    v = data[:, 1]
    mask = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    if mask.sum() < min_tail_points:
        return None

    r = r[mask]
    v = v[mask]
    idx = np.argsort(r)
    r = r[idx]
    v = v[idx]

    n_tail = min(tail_points, len(r))
    if n_tail < min_tail_points:
        return None
    r_tail = r[-n_tail:]
    v_tail = v[-n_tail:]
    if np.unique(r_tail).size < 2:
        return None

    slope, *_ = linregress(np.log10(r_tail), np.log10(v_tail))
    return {
        "deep_slope": float(slope),
        "delta_f3": float(slope + 0.5),
        "n_tail_points": int(n_tail),
        "tail_r_min": float(np.min(r_tail)),
        "tail_r_max": float(np.max(r_tail)),
    }


def build_sparc_175_master(
    sparc_dir: Path,
    out_path: Path,
    *,
    tail_points: int = 5,
    min_tail_points: int = 3,
) -> pd.DataFrame:
    if tail_points < min_tail_points:
        raise ValueError("tail_points must be >= min_tail_points")
    if min_tail_points < 3:
        raise ValueError("min_tail_points must be >= 3")

    metadata_dir = _resolve_metadata_dir(sparc_dir)
    _require_support_files(metadata_dir)
    metadata_table = _find_metadata_table(sparc_dir, metadata_dir)
    meta = _load_metadata(metadata_table)
    meta_map = meta.set_index("galaxy_norm").to_dict(orient="index")

    rows: list[dict] = []
    for rotmod in _iter_rotmod_files(sparc_dir / "rotmod"):
        galaxy = rotmod.stem.replace("_rotmod", "")
        slope_data = _slope_from_tail(rotmod, tail_points=tail_points, min_tail_points=min_tail_points)
        if slope_data is None:
            continue
        props = meta_map.get(_norm_galaxy(galaxy), {})
        row = {
            "galaxy": galaxy,
            **slope_data,
            "logSigmaHI_out": props.get("logSigmaHI_out", np.nan),
            "logMstar": props.get("logMstar", np.nan),
            "logRd": props.get("logRd", np.nan),
            "inclination": props.get("inclination", np.nan),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[REQUIRED_COLUMNS].sort_values("galaxy").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SPARC 175 master table for ΔF3 vs external HI analysis."
    )
    parser.add_argument(
        "--sparc-dir",
        default="data/SPARC",
        help="SPARC directory containing metadata/ and rotmod/.",
    )
    parser.add_argument(
        "--out",
        default="data/sparc_175_master.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tail-points",
        type=int,
        default=5,
        help="Number of outer points used for deep_slope fit (default: 5).",
    )
    parser.add_argument(
        "--min-tail-points",
        type=int,
        default=3,
        help="Minimum required tail points per galaxy (default: 3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out = build_sparc_175_master(
        sparc_dir=Path(args.sparc_dir),
        out_path=Path(args.out),
        tail_points=args.tail_points,
        min_tail_points=args.min_tail_points,
    )

    print(f"Written: {args.out}")
    print(f"Rows: {len(out)}")
    if not out.empty:
        print(out.describe(include="all"))
        print(out.isna().sum())


if __name__ == "__main__":
    main()
