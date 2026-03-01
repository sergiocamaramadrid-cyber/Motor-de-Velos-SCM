#!/usr/bin/env python3
"""
compute_f3_catalog.py
Compute F3_SCM for all rotation-curve files in SPARC and LITTLE THINGS folders.

F3_SCM is defined as the outer log-log slope:
    F3_SCM = d log10(Vobs) / d log10(r)   evaluated at r >= frac * Rmax

Usage:
  python scripts/compute_f3_catalog.py \\
    --sparc-dir data/SPARC/Rotmod \\
    --lt-dir data/LITTLE_THINGS/Rotmod \\
    --out results/f3_catalog.csv \\
    --outer-fracs 0.6 0.7 0.8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class F3Row:
    galaxy: str
    source: str
    outer_frac: float
    F3_SCM: float
    F3_SCM_err: float
    R2: float
    n_all: int
    n_used: int
    rmin_used_kpc: float
    rmax_kpc: float
    status: str   # ok|warn|fail
    note: str
    file: str


def _discover_rotmod_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob("*_rotmod.dat"))


def _read_r_vobs_any_format(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly read r and Vobs from whitespace files.
    Assumption: first two numeric columns are r_kpc and Vobs (true for both
    SPARC and LITTLE THINGS rotmod formats).
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    if df.shape[1] < 2:
        raise ValueError(f"Expected >=2 columns, got {df.shape[1]}")

    r = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(r) & np.isfinite(v)
    r = r[mask]
    v = v[mask]
    return r, v


def _ols_slope_err_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    OLS for y = a + b x. Returns (b, se_b, r2).
    """
    n = x.size
    if n < 2:
        return np.nan, np.nan, np.nan

    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)

    if n <= 2:
        se_b = np.nan
    else:
        mse = ss_res / (n - 2)
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        se_b = np.nan if sxx == 0 else float(np.sqrt(mse / sxx))

    return b, se_b, r2


def compute_f3_for_file(path: Path, source: str, outer_frac: float) -> F3Row:
    galaxy = path.stem.replace("_rotmod", "")
    note_parts: List[str] = []

    try:
        r, v = _read_r_vobs_any_format(path)
    except Exception as e:
        return F3Row(
            galaxy=galaxy, source=source, outer_frac=outer_frac,
            F3_SCM=np.nan, F3_SCM_err=np.nan, R2=np.nan,
            n_all=0, n_used=0, rmin_used_kpc=np.nan, rmax_kpc=np.nan,
            status="fail", note=f"read_error:{type(e).__name__}", file=str(path),
        )

    # Filter non-physical values before log
    mask_phys = (r > 0) & (v > 0)
    r = r[mask_phys]
    v = v[mask_phys]
    n_all = int(r.size)

    if n_all < 3:
        return F3Row(
            galaxy=galaxy, source=source, outer_frac=outer_frac,
            F3_SCM=np.nan, F3_SCM_err=np.nan, R2=np.nan,
            n_all=n_all, n_used=0, rmin_used_kpc=np.nan, rmax_kpc=np.nan,
            status="fail", note="too_few_points_after_clean", file=str(path),
        )

    rmax = float(np.max(r))
    rcut = outer_frac * rmax
    mask_ext = r >= rcut
    r_ext = r[mask_ext]
    v_ext = v[mask_ext]
    n_used = int(r_ext.size)

    if n_used < 3:
        # Fallback to last 3 points in radius
        idx = np.argsort(r)[-3:]
        r_ext = r[idx]
        v_ext = v[idx]
        n_used = 3
        note_parts.append("fallback_last3")

    log_r = np.log10(r_ext)
    log_v = np.log10(v_ext)

    slope, se, r2 = _ols_slope_err_r2(log_r, log_v)

    status = "ok"
    if not np.isfinite(slope):
        status = "fail"
        note_parts.append("nan_slope")
    else:
        if np.isfinite(r2) and r2 < 0.2:
            status = "warn"
            note_parts.append("low_r2")
        if np.isfinite(se) and se > 0.2:
            status = "warn"
            note_parts.append("high_err")

    note = ";".join(note_parts) if note_parts else "ok"

    return F3Row(
        galaxy=galaxy, source=source, outer_frac=float(outer_frac),
        F3_SCM=float(slope),
        F3_SCM_err=float(se) if np.isfinite(se) else np.nan,
        R2=float(r2) if np.isfinite(r2) else np.nan,
        n_all=n_all, n_used=n_used,
        rmin_used_kpc=float(np.min(r_ext)),
        rmax_kpc=rmax,
        status=status, note=note, file=str(path),
    )


def process_dirs(
    sparc_dir: Optional[Path],
    lt_dir: Optional[Path],
    outer_fracs: Iterable[float],
) -> pd.DataFrame:
    rows: List[dict] = []

    sources = []
    if sparc_dir is not None:
        sources.append(("SPARC", sparc_dir))
    if lt_dir is not None:
        sources.append(("LITTLE_THINGS", lt_dir))

    for src_name, d in sources:
        files = _discover_rotmod_files(d)
        for f in files:
            for frac in outer_fracs:
                res = compute_f3_for_file(f, source=src_name, outer_frac=float(frac))
                rows.append(res.__dict__)

    cols = [
        "source", "galaxy", "outer_frac",
        "F3_SCM", "F3_SCM_err", "R2",
        "n_all", "n_used", "rmin_used_kpc", "rmax_kpc",
        "status", "note", "file",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[cols].sort_values(["source", "galaxy", "outer_frac"]).reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute F3_SCM catalog from SPARC and/or LITTLE THINGS rotmod files."
    )
    p.add_argument("--sparc-dir", type=str, default=None,
                   help="Dir with SPARC *_rotmod.dat files")
    p.add_argument("--lt-dir", type=str, default=None,
                   help="Dir with LITTLE THINGS *_rotmod.dat files")
    p.add_argument("--out", type=str, default="results/f3_catalog.csv",
                   help="Output CSV path")
    p.add_argument(
        "--outer-fracs",
        nargs="*",
        type=float,
        default=[0.7],
        help="Outer region thresholds as fractions of Rmax (e.g. 0.6 0.7 0.8)",
    )
    args = p.parse_args(argv)

    sparc_dir = Path(args.sparc_dir) if args.sparc_dir else None
    lt_dir = Path(args.lt_dir) if args.lt_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = process_dirs(sparc_dir, lt_dir, outer_fracs=args.outer_fracs)
    df.to_csv(out_path, index=False)

    print(f"[compute_f3_catalog] wrote: {out_path}")
    if df.empty:
        print("No files processed (check directories).")
        return 2

    valid = df["F3_SCM"].notna().sum()
    fails = (df["status"] == "fail").sum()
    warns = (df["status"] == "warn").sum()
    print(f"Total rows (galaxy×threshold): {len(df)}")
    print(f"Valid F3_SCM: {valid} | warn: {warns} | fail: {fails}")
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
