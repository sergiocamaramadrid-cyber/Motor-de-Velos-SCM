#!/usr/bin/env python3
"""
scripts/pilot_f3_test.py — Pilot F3 test for LITTLE THINGS / SPARC-style rotmod files.

Computes:
    F3 = d log10(Vobs) / d log10(r)
using only outer radii r >= 0.7 * Rmax.

Outputs:
    results/F3_values.csv (by default)

Expected SPARC rotmod format (whitespace-delimited):
    r_kpc  Vobs  errV  Vgas  Vdisk  Vbul  Vbar
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROT_MOD_COLS = ["r_kpc", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "Vbar"]

PILOT_GALAXIES: List[str] = ["DDO69", "DDO70", "DDO75", "DDO210"]


@dataclass(frozen=True)
class F3Result:
    galaxy: str
    file: str
    F3: float
    F3_err: float
    R2: float
    n_all: int
    n_used: int
    rmin_used_kpc: float
    rmax_kpc: float
    status: str  # "ok" | "warn" | "fail"
    note: str


def _read_rotmod(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=ROT_MOD_COLS,
        engine="python",
    )
    # Coerce numeric; drop non-numeric rows defensively
    for c in ROT_MOD_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["r_kpc", "Vobs"])
    return df


def _linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns (slope, slope_err, r2) for y = a + b x.
    """
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")

    X = np.vstack([np.ones_like(x), x]).T
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])

    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0 else (1.0 - ss_res / ss_tot)

    # slope standard error
    n = x.size
    if n <= 2:
        slope_err = float("nan")
    else:
        mse = ss_res / (n - 2)
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        slope_err = float("nan") if sxx == 0 else math.sqrt(mse / sxx)

    return slope, float(slope_err), r2


def compute_f3_from_file(path: Path, outer_frac: float = 0.7) -> F3Result:
    galaxy = path.stem.replace("_rotmod", "")
    note_parts: List[str] = []

    if not path.exists():
        return F3Result(
            galaxy=galaxy,
            file=str(path),
            F3=float("nan"),
            F3_err=float("nan"),
            R2=float("nan"),
            n_all=0,
            n_used=0,
            rmin_used_kpc=float("nan"),
            rmax_kpc=float("nan"),
            status="fail",
            note="file_not_found",
        )

    df = _read_rotmod(path)

    # Basic cleaning
    df = df[(df["r_kpc"] > 0) & (df["Vobs"] > 0)]
    n_all = int(df.shape[0])

    if n_all < 3:
        return F3Result(
            galaxy=galaxy,
            file=str(path),
            F3=float("nan"),
            F3_err=float("nan"),
            R2=float("nan"),
            n_all=n_all,
            n_used=0,
            rmin_used_kpc=float("nan"),
            rmax_kpc=float("nan"),
            status="fail",
            note="too_few_points_after_clean",
        )

    r = df["r_kpc"].to_numpy(dtype=float)
    v = df["Vobs"].to_numpy(dtype=float)

    rmax = float(np.max(r))
    rcut = outer_frac * rmax
    mask = r >= rcut

    r_ext = r[mask]
    v_ext = v[mask]
    n_used = int(r_ext.size)

    if n_used < 3:
        # not enough outer points; fall back to last 3 points (still outer-ish)
        idx = np.argsort(r)[-3:]
        r_ext = r[idx]
        v_ext = v[idx]
        n_used = 3
        note_parts.append("fallback_last3")

    log_r = np.log10(r_ext)
    log_v = np.log10(v_ext)

    slope, slope_err, r2 = _linregress(log_r, log_v)

    status = "ok"
    if not np.isfinite(slope) or n_used < 3:
        status = "fail"
    elif (np.isfinite(r2) and r2 < 0.2) or (np.isfinite(slope_err) and slope_err > 0.2):
        status = "warn"
        note_parts.append("low_r2_or_high_err")

    return F3Result(
        galaxy=galaxy,
        file=str(path),
        F3=float(slope),
        F3_err=float(slope_err) if np.isfinite(slope_err) else float("nan"),
        R2=float(r2) if np.isfinite(r2) else float("nan"),
        n_all=n_all,
        n_used=n_used,
        rmin_used_kpc=float(np.min(r_ext)),
        rmax_kpc=rmax,
        status=status,
        note=";".join(note_parts) if note_parts else "ok",
    )


def compute_f3_for_galaxies(
    sparc_dir: Path,
    galaxies: Iterable[str],
    outer_frac: float = 0.7,
) -> pd.DataFrame:
    rows = []
    for g in galaxies:
        path = sparc_dir / f"{g}_rotmod.dat"
        res = compute_f3_from_file(path, outer_frac=outer_frac)
        rows.append(res.__dict__)
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pilot F3 test (outer slope) from SPARC rotmod files."
    )
    parser.add_argument(
        "--sparc-dir",
        type=str,
        default="data/SPARC/Rotmod",
        help="Directory containing *_rotmod.dat files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/F3_values.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--outer-frac",
        type=float,
        default=0.7,
        help="Use radii r >= outer_frac * Rmax for the fit.",
    )
    parser.add_argument(
        "--galaxies",
        nargs="*",
        default=PILOT_GALAXIES,
        help="Galaxy names matching rotmod filenames (without _rotmod.dat).",
    )

    args = parser.parse_args(argv)

    sparc_dir = Path(args.sparc_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = compute_f3_for_galaxies(
        sparc_dir=sparc_dir,
        galaxies=args.galaxies,
        outer_frac=float(args.outer_frac),
    )

    # Stable column order
    cols = [
        "galaxy",
        "F3",
        "F3_err",
        "R2",
        "n_all",
        "n_used",
        "rmin_used_kpc",
        "rmax_kpc",
        "status",
        "note",
        "file",
    ]
    df = df[cols]

    df.to_csv(out_path, index=False)

    # Minimal terminal summary
    print(f"[pilot_f3_test] wrote: {out_path}")
    print(df[["galaxy", "F3", "F3_err", "R2", "n_used", "status", "note"]].to_string(index=False))

    # Non-zero exit if any fail
    if (df["status"] == "fail").any():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
