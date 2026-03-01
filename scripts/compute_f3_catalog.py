"""
scripts/compute_f3_catalog.py — Automated F3_SCM catalog for SPARC + LITTLE THINGS.

Computes the observable:

    F_{3,SCM} = d log10(V_obs) / d log10(r) |_{r >= f * R_max}

for every rotation-curve file found under the data directory.

Supports multiple outer-fraction thresholds (--outer-fracs) to assess the
stability of F3_SCM with respect to the definition of the outer region.

Output
------
results/f3_catalog.csv (or --out PATH) with columns:

  source       — data source label (e.g. SPARC, LT_OH2015)
  galaxy       — galaxy identifier (filename stem without format suffix)
  outer_frac   — fraction of R_max used as the outer-region boundary
  F3_SCM       — logarithmic slope d log10(V_obs) / d log10(r)
  F3_SCM_err   — standard error of the slope (OLS)
  R2           — coefficient of determination
  n_all        — total number of radial points in the rotation curve
  n_used       — points used (r >= outer_frac * R_max with V_obs > 0)
  status       — "ok" or a skip reason
  note         — human-readable explanation when status != "ok"
  file         — path to the source file (relative to cwd at call time)

Usage
-----
::

    python scripts/compute_f3_catalog.py

    python scripts/compute_f3_catalog.py \\
        --data-dir data/raw \\
        --out results/f3_catalog.csv \\
        --outer-fracs 0.6 0.7 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTER_FRACS_DEFAULT: list[float] = [0.7]
MIN_POINTS_DEFAULT: int = 3

# Canonical output columns (contract)
CATALOG_COLS: list[str] = [
    "source",
    "galaxy",
    "outer_frac",
    "F3_SCM",
    "F3_SCM_err",
    "R2",
    "n_all",
    "n_used",
    "status",
    "note",
    "file",
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_files(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Return a list of (source_label, galaxy_name, file_path) tuples.

    Supported file types:
    - ``*_rotmod.dat`` — SPARC white-space-separated format
    - ``*_rot.csv``    — LITTLE THINGS CSV format
    """
    found: list[tuple[str, str, Path]] = []

    for path in sorted(data_dir.rglob("*_rotmod.dat")):
        galaxy = path.name.replace("_rotmod.dat", "")
        found.append(("SPARC", galaxy, path))

    for path in sorted(data_dir.rglob("*_rot.csv")):
        galaxy = path.name.replace("_rot.csv", "")
        found.append(("LT_OH2015", galaxy, path))

    return found


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _load_sparc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a SPARC rotmod.dat file.

    Parameters
    ----------
    path : Path
        Path to the ``*_rotmod.dat`` file.

    Returns
    -------
    r : ndarray
        Radii in kpc.
    v_obs : ndarray
        Observed rotation velocities in km/s.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
               "SBdisk", "SBbul"],
        usecols=[0, 1],
    )
    return df["r"].values, df["v_obs"].values


def _load_lt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a LITTLE THINGS ``*_rot.csv`` file.

    Parameters
    ----------
    path : Path
        Path to the ``*_rot.csv`` file.

    Returns
    -------
    r : ndarray
        Radii in kpc.
    v_obs : ndarray
        Observed rotation velocities in km/s.
    """
    df = pd.read_csv(path)
    r_col = "r_kpc" if "r_kpc" in df.columns else df.columns[0]
    v_col = "Vobs_kms" if "Vobs_kms" in df.columns else (
        "Vbary_kms" if "Vbary_kms" in df.columns else df.columns[1]
    )
    return df[r_col].values, df[v_col].values


def _load_rotation_curve(
    source: str, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the correct loader based on *source* label."""
    if source == "SPARC":
        return _load_sparc(path)
    return _load_lt(path)


# ---------------------------------------------------------------------------
# F3_SCM computation
# ---------------------------------------------------------------------------

def compute_f3(
    r: np.ndarray,
    v_obs: np.ndarray,
    outer_frac: float,
    min_points: int = MIN_POINTS_DEFAULT,
) -> dict:
    """Compute F3_SCM for a single rotation curve at one outer_frac threshold.

    Parameters
    ----------
    r : ndarray
        Radii in kpc (must be > 0 for log-space regression).
    v_obs : ndarray
        Observed rotation velocities in km/s (must be > 0 for log-space).
    outer_frac : float
        Fraction of R_max that defines the outer region.
    min_points : int
        Minimum number of valid points required to attempt the fit.

    Returns
    -------
    dict
        Keys: F3_SCM, F3_SCM_err, R2, n_all, n_used, status, note.
    """
    n_all = len(r)
    result: dict = {
        "F3_SCM": float("nan"),
        "F3_SCM_err": float("nan"),
        "R2": float("nan"),
        "n_all": n_all,
        "n_used": 0,
        "status": "ok",
        "note": "",
    }

    if n_all == 0:
        result["status"] = "skip_no_data"
        result["note"] = "rotation curve is empty"
        return result

    r_max = float(np.max(r))
    r_threshold = outer_frac * r_max

    outer_mask = r >= r_threshold
    # Require positive velocities and radii for log regression
    valid_mask = outer_mask & (v_obs > 0) & (r > 0)
    n_used = int(valid_mask.sum())
    result["n_used"] = n_used

    if n_used < min_points:
        if not outer_mask.any():
            result["status"] = "skip_no_outer_points"
            result["note"] = (
                f"no points at r >= {outer_frac} * R_max = {r_threshold:.3f} kpc"
            )
        elif n_used == 0:
            result["status"] = "skip_no_valid_points"
            result["note"] = (
                "all outer-region points have non-positive v_obs or r"
            )
        else:
            result["status"] = "skip_few_points"
            result["note"] = (
                f"only {n_used} valid outer point(s); need >= {min_points}"
            )
        return result

    log_r = np.log10(r[valid_mask])
    log_v = np.log10(v_obs[valid_mask])

    slope, _intercept, r_value, _p_value, stderr = linregress(log_r, log_v)

    result["F3_SCM"] = float(slope)
    result["F3_SCM_err"] = float(stderr)
    result["R2"] = float(r_value ** 2)
    return result


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_catalog(
    data_dir: Path,
    outer_fracs: list[float] = OUTER_FRACS_DEFAULT,
    min_points: int = MIN_POINTS_DEFAULT,
) -> pd.DataFrame:
    """Build the full F3_SCM catalog.

    Parameters
    ----------
    data_dir : Path
        Root directory to search for rotation-curve files.
    outer_fracs : list[float]
        Outer-fraction thresholds to evaluate.
    min_points : int
        Minimum outer-region points required for a fit.

    Returns
    -------
    pd.DataFrame
        Catalog with columns defined by :data:`CATALOG_COLS`.
    """
    files = _discover_files(data_dir)
    rows: list[dict] = []

    for source, galaxy, path in files:
        try:
            r, v_obs = _load_rotation_curve(source, path)
        except Exception as exc:  # noqa: BLE001
            for frac in outer_fracs:
                rows.append({
                    "source": source,
                    "galaxy": galaxy,
                    "outer_frac": frac,
                    "F3_SCM": float("nan"),
                    "F3_SCM_err": float("nan"),
                    "R2": float("nan"),
                    "n_all": 0,
                    "n_used": 0,
                    "status": "skip_load_error",
                    "note": str(exc),
                    "file": str(path),
                })
            continue

        for frac in outer_fracs:
            metrics = compute_f3(r, v_obs, frac, min_points=min_points)
            rows.append({
                "source": source,
                "galaxy": galaxy,
                "outer_frac": frac,
                **metrics,
                "file": str(path),
            })

    df = pd.DataFrame(rows, columns=CATALOG_COLS)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compute the F3_SCM catalog: d log10(V_obs)/d log10(r) "
            "in the outer region of each rotation curve."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(repo_root / "data" / "raw"),
        metavar="DIR",
        help="Root directory to search for rotation-curve files "
             "(default: data/raw).",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "results" / "f3_catalog.csv"),
        metavar="FILE",
        help="Output CSV file path (default: results/f3_catalog.csv).",
    )
    parser.add_argument(
        "--outer-fracs",
        nargs="+",
        type=float,
        default=OUTER_FRACS_DEFAULT,
        dest="outer_fracs",
        metavar="F",
        help=(
            "Outer-fraction threshold(s) f such that r >= f * R_max defines "
            "the outer region (default: 0.7).  Multiple values allowed."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_POINTS_DEFAULT,
        dest="min_points",
        metavar="N",
        help=(
            f"Minimum number of outer-region points required for a fit "
            f"(default: {MIN_POINTS_DEFAULT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    catalog = build_catalog(
        data_dir,
        outer_fracs=args.outer_fracs,
        min_points=args.min_points,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    n_ok = int((catalog["status"] == "ok").sum())
    n_total = len(catalog)
    sep = "=" * 65
    print(sep)
    print("  Motor de Velos SCM — F3_SCM Catalog")
    print(sep)
    print(f"  Data dir      : {data_dir}")
    print(f"  Outer fracs   : {args.outer_fracs}")
    print(f"  Min points    : {args.min_points}")
    print(f"  Total rows    : {n_total}")
    print(f"  Status 'ok'   : {n_ok}")
    print(f"  Output        : {out_path}")
    print(sep)


if __name__ == "__main__":
    main()
