#!/usr/bin/env python3
"""
scripts/build_sample_csv.py — Assemble the LITTLE THINGS sample catalog.

Merges per-galaxy observational data from three sources:

1. ``data/little_things_global.csv`` (required)
   Columns used: ``galaxy_id``, ``logM``, ``log_j`` (environmental proxy).

2. ``results/blind_test_lt/predictions.csv`` (optional)
   Provides ``residual_btfr`` → renamed ``delta_f3`` and used as
   ``slope_tail`` (outer-disk BTFR deviation, a proxy for the
   deep-regime friction slope β).

3. ``data/raw/lt_oh2015/`` directory (optional)
   One ``<galaxy>_rot.csv`` per galaxy with columns ``r_kpc``, ``Vbary_kms``.
   Provides ``Rmax_kpc`` = max(r_kpc) for each galaxy where data exist.

Output
------
``results/lt_sample_catalog.csv`` with columns:

  galaxy, logM, delta_mass_std, slope_tail, Rmax_kpc, delta_f3

``delta_mass_std`` is the z-score standardisation of ``log_j`` across the
sample.  ``slope_tail`` equals ``delta_f3`` (BTFR residual) when
per-galaxy friction-slope measurements are unavailable.

Usage
-----
::

    python scripts/build_sample_csv.py
    python scripts/build_sample_csv.py --lt-global data/little_things_global.csv \\
        --predictions results/blind_test_lt/predictions.csv \\
        --rot-dir data/raw/lt_oh2015 \\
        --out results/lt_sample_catalog.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LT_GLOBAL = _REPO_ROOT / "data" / "little_things_global.csv"
_DEFAULT_PREDICTIONS = (
    _REPO_ROOT / "results" / "blind_test_lt" / "predictions.csv"
)
_DEFAULT_ROT_DIR = _REPO_ROOT / "data" / "raw" / "lt_oh2015"
_DEFAULT_OUT = _REPO_ROOT / "results" / "lt_sample_catalog.csv"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_lt_global(path: Path | str) -> pd.DataFrame:
    """Load *little_things_global.csv* and return a clean DataFrame.

    Required columns: ``galaxy_id``, ``logM``, ``log_j``.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy`` (str), ``logM`` (float), ``log_j`` (float).

    Raises
    ------
    ValueError
        If required columns are missing or the file has fewer than 2 rows.
    """
    df = pd.read_csv(Path(path))
    required = {"galaxy_id", "logM", "log_j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"lt_global CSV missing columns: {sorted(missing)}")
    if len(df) < 2:
        raise ValueError("lt_global CSV must have at least 2 data rows")
    df = df.rename(columns={"galaxy_id": "galaxy"})
    return df[["galaxy", "logM", "log_j"]].copy()


def load_predictions(path: Path | str) -> pd.DataFrame:
    """Load *predictions.csv* from the blind test and return delta_f3.

    Required columns: ``galaxy_id``, ``residual_btfr``.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy`` (str), ``delta_f3`` (float).
    """
    df = pd.read_csv(Path(path))
    required = {"galaxy_id", "residual_btfr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions CSV missing columns: {sorted(missing)}")
    df = df.rename(columns={"galaxy_id": "galaxy", "residual_btfr": "delta_f3"})
    return df[["galaxy", "delta_f3"]].copy()


def collect_rmax(rot_dir: Path | str) -> dict[str, float]:
    """Scan *rot_dir* for ``<galaxy>_rot.csv`` files and return max radius.

    Each file must contain a ``r_kpc`` column.  Files that cannot be parsed
    are silently skipped.

    Parameters
    ----------
    rot_dir:
        Directory containing per-galaxy rotation-curve CSVs.

    Returns
    -------
    dict[str, float]
        Mapping ``{galaxy_name: Rmax_kpc}``.
    """
    rot_dir = Path(rot_dir)
    result: dict[str, float] = {}
    if not rot_dir.is_dir():
        return result
    for csv_file in sorted(rot_dir.glob("*_rot.csv")):
        galaxy = csv_file.stem.replace("_rot", "")
        try:
            df = pd.read_csv(csv_file)
            if "r_kpc" not in df.columns:
                continue
            rmax = float(df["r_kpc"].max())
            result[galaxy] = rmax
        except Exception:
            continue
    return result


def standardise(series: pd.Series) -> pd.Series:
    """Z-score standardise a numeric Series; returns NaN if std == 0."""
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.full(len(series), np.nan), index=series.index)
    return (series - mu) / sigma


def build_catalog(
    lt_global_path: Path | str = _DEFAULT_LT_GLOBAL,
    predictions_path: Path | str | None = _DEFAULT_PREDICTIONS,
    rot_dir: Path | str | None = _DEFAULT_ROT_DIR,
    out_path: Path | str = _DEFAULT_OUT,
) -> pd.DataFrame:
    """Build and write the sample catalog CSV.

    Parameters
    ----------
    lt_global_path:
        Path to ``data/little_things_global.csv``.
    predictions_path:
        Path to ``results/blind_test_lt/predictions.csv``.  If ``None`` or
        the file does not exist, ``delta_f3`` and ``slope_tail`` are ``NaN``.
    rot_dir:
        Directory with per-galaxy ``*_rot.csv`` files.  If ``None`` or the
        directory does not exist, ``Rmax_kpc`` is ``NaN`` for all galaxies.
    out_path:
        Destination CSV path.  Parent directory is created if necessary.

    Returns
    -------
    pd.DataFrame
        The assembled catalog (also written to *out_path*).
    """
    # 1. Base table
    base = load_lt_global(lt_global_path)

    # 2. Standardise environmental proxy
    base["delta_mass_std"] = standardise(base["log_j"])

    # 3. Merge BTFR residuals (delta_f3 / slope_tail)
    pred_path = Path(predictions_path) if predictions_path else None
    if pred_path is not None and pred_path.exists():
        preds = load_predictions(pred_path)
        base = base.merge(preds, on="galaxy", how="left")
    else:
        base["delta_f3"] = np.nan

    # slope_tail = delta_f3 (best available per-galaxy outer-disk measurement)
    base["slope_tail"] = base["delta_f3"]

    # 4. Add Rmax from rotation curve files
    rmax_map = collect_rmax(rot_dir) if rot_dir is not None else {}
    base["Rmax_kpc"] = base["galaxy"].map(rmax_map)

    # 5. Select and order output columns
    catalog = base[
        ["galaxy", "logM", "delta_mass_std", "slope_tail", "Rmax_kpc", "delta_f3"]
    ].copy()

    # 6. Write
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False, float_format="%.4f")
    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LITTLE THINGS sample catalog CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lt-global",
        default=str(_DEFAULT_LT_GLOBAL),
        metavar="PATH",
        help="Path to data/little_things_global.csv",
    )
    parser.add_argument(
        "--predictions",
        default=str(_DEFAULT_PREDICTIONS),
        metavar="PATH",
        help="Path to blind-test predictions CSV (optional)",
    )
    parser.add_argument(
        "--rot-dir",
        default=str(_DEFAULT_ROT_DIR),
        metavar="DIR",
        help="Directory with *_rot.csv rotation-curve files (optional)",
    )
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        metavar="PATH",
        help="Output CSV path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    catalog = build_catalog(
        lt_global_path=args.lt_global,
        predictions_path=args.predictions,
        rot_dir=args.rot_dir,
        out_path=args.out,
    )
    n_full = int(catalog["slope_tail"].notna().sum())
    n_rmax = int(catalog["Rmax_kpc"].notna().sum())
    print(
        f"Wrote {len(catalog)} galaxies → {args.out}  "
        f"(slope_tail: {n_full}/{len(catalog)}, "
        f"Rmax_kpc: {n_rmax}/{len(catalog)})"
    )
    return catalog


if __name__ == "__main__":
    main(sys.argv[1:])
