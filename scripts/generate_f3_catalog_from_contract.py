"""
scripts/generate_f3_catalog_from_contract.py — F3 (flat-fast-faint) catalog
generator from a contract-compliant rotation-curve table.

Reads a Parquet (or CSV) file that satisfies the SCM data contract
(columns: galaxy, r_kpc, vobs_kms, vobs_err_kms, vbar_kms) and produces a
per-galaxy summary catalog that flags each galaxy according to the *F3*
criterion:

  F3 = (Vflat > vflat_min) AND (Mbar < mbar_max) AND (deep_slope < 0.6)

where *deep_slope* is the slope of the power-law fit
``log10(Vobs) ~ slope · log10(Vbar)`` restricted to data points where
``Vbar < vbar_deep_threshold`` km/s (i.e. the "deep-MOND" regime).

Parameters
----------
--input FILE
    Path to the contract-compliant table (Parquet or CSV).
--out DIR
    Output directory.  Writes ``f3_catalog.csv`` (default: current dir).
--vflat-min FLOAT
    Minimum flat velocity threshold (km/s) for the F3 flag (default: 80).
--mbar-max FLOAT
    Maximum baryonic mass (log10 M_sun) for the F3 flag (default: 10.5).
--min-deep INT
    Minimum number of "deep-regime" points required to attempt a slope fit.
    Galaxies with fewer points are assigned ``deep_slope = NaN`` and
    are **not** flagged as F3.  Previously this was hardcoded to 3; exposing
    it makes the deep-slope fit policy explicit and reproducible (default: 3).
--vbar-deep FLOAT
    Vbar threshold below which a point is considered "deep-MOND" (km/s).
    Default: 50.

Usage
-----
::

    python scripts/generate_f3_catalog_from_contract.py \\
        --input data/BIG-SPARC/processed/big_sparc_contract.parquet \\
        --out results/f3

    # Stricter deep-slope policy (require at least 5 deep points):
    python scripts/generate_f3_catalog_from_contract.py \\
        --input data/BIG-SPARC/processed/big_sparc_contract.parquet \\
        --out results/f3 \\
        --min-deep 5
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow both ``python -m scripts.generate_f3_catalog_from_contract`` and
# ``python scripts/generate_f3_catalog_from_contract.py`` invocations.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.contract_utils import read_table, validate_contract

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

_VFLAT_MIN_DEFAULT: float = 80.0       # km/s
_MBAR_MAX_DEFAULT: float = 10.5        # log10(M_sun)
_MIN_DEEP_DEFAULT: int = 3             # minimum deep-regime points for slope fit
_VBAR_DEEP_DEFAULT: float = 50.0       # km/s  (deep-MOND threshold on Vbar)

# ---------------------------------------------------------------------------
# Per-galaxy computation
# ---------------------------------------------------------------------------

def _compute_galaxy_stats(
    sub: pd.DataFrame,
    min_deep: int,
    vbar_deep: float,
) -> dict:
    """Compute per-galaxy statistics from a subset of the contract table.

    Parameters
    ----------
    sub : pd.DataFrame
        Rows for a single galaxy (must contain contract columns).
    min_deep : int
        Minimum number of deep-regime points to attempt a slope fit.
    vbar_deep : float
        Vbar threshold (km/s) below which a point is "deep-MOND".

    Returns
    -------
    dict
        Keys: galaxy, n_points, vflat_kms, log_mbar, deep_slope, deep_n.
    """
    galaxy = sub["galaxy"].iloc[0]
    n_points = len(sub)

    # Flat velocity: median of the outermost 20 % of radii.
    # Fall back to the full sample median if no points clear the threshold.
    r_thresh = sub["r_kpc"].quantile(0.80)
    outer = sub[sub["r_kpc"] >= r_thresh]
    vflat = float(outer["vobs_kms"].median()) if len(outer) > 0 else float(sub["vobs_kms"].median())

    # Baryonic mass proxy: max Vbar² → Mbar ∝ Vbar^4 (BTFR); we store
    # log10(Vbar_max²) as a mass proxy in internal units
    vbar_max = float(sub["vbar_kms"].abs().max())
    # Use Vbar_max^4 as a dimensionless mass proxy (avoids requiring distance)
    log_mbar_proxy = 4.0 * np.log10(max(vbar_max, 1e-6))

    # Deep-slope fit
    deep_mask = sub["vbar_kms"].abs() < vbar_deep
    deep_pts = sub[deep_mask]
    deep_n = int(deep_mask.sum())

    deep_slope: float = float("nan")
    if deep_n >= min_deep:
        # clip to 1e-6 km/s before log10 to avoid log10(0) for near-zero velocities
        log_vbar = np.log10(deep_pts["vbar_kms"].abs().clip(lower=1e-6).values)
        log_vobs = np.log10(deep_pts["vobs_kms"].abs().clip(lower=1e-6).values)
        # Least-squares slope through origin (no intercept) in log-log space
        if np.std(log_vbar) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(log_vbar, log_vobs, 1)
            deep_slope = float(coeffs[0])

    return {
        "galaxy": galaxy,
        "n_points": n_points,
        "vflat_kms": round(vflat, 3),
        "log_mbar_proxy": round(log_mbar_proxy, 4),
        "deep_n": deep_n,
        "deep_slope": round(deep_slope, 4) if not np.isnan(deep_slope) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main catalog generation
# ---------------------------------------------------------------------------

def generate_catalog(
    input_path: Path,
    out_dir: Path,
    vflat_min: float = _VFLAT_MIN_DEFAULT,
    mbar_max: float = _MBAR_MAX_DEFAULT,
    min_deep: int = _MIN_DEEP_DEFAULT,
    vbar_deep: float = _VBAR_DEEP_DEFAULT,
) -> pd.DataFrame:
    """Generate the F3 catalog from a contract-compliant table.

    Parameters
    ----------
    input_path : Path
        Path to the contract-compliant table (Parquet or CSV).
    out_dir : Path
        Output directory.  Writes ``f3_catalog.csv``.
    vflat_min : float
        Minimum flat velocity (km/s) for the F3 flag.
    mbar_max : float
        Maximum baryonic mass proxy (log10 scale) for the F3 flag.
    min_deep : int
        Minimum deep-regime points required to attempt a slope fit.
    vbar_deep : float
        Vbar threshold (km/s) defining the deep-MOND regime.

    Returns
    -------
    pd.DataFrame
        Per-galaxy F3 catalog.
    """
    df = read_table(input_path)
    validate_contract(df, source=str(input_path))

    rows = []
    for galaxy, sub in df.groupby("galaxy", sort=True):
        stats = _compute_galaxy_stats(sub, min_deep=min_deep, vbar_deep=vbar_deep)
        rows.append(stats)

    catalog = pd.DataFrame(rows)

    # F3 flag
    has_slope = catalog["deep_slope"].notna()
    catalog["f3_flag"] = (
        (catalog["vflat_kms"] >= vflat_min)
        & (catalog["log_mbar_proxy"] <= mbar_max)
        & has_slope
        & (catalog["deep_slope"] < 0.6)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "f3_catalog.csv"
    catalog.to_csv(out_path, index=False)

    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the F3 (flat-fast-faint) catalog from a "
            "contract-compliant rotation-curve table."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Contract-compliant table (Parquet or CSV).",
    )
    parser.add_argument(
        "--out",
        default=".",
        metavar="DIR",
        help="Output directory for f3_catalog.csv (default: current dir).",
    )
    parser.add_argument(
        "--vflat-min",
        type=float,
        default=_VFLAT_MIN_DEFAULT,
        dest="vflat_min",
        metavar="FLOAT",
        help=f"Minimum Vflat (km/s) for F3 flag (default: {_VFLAT_MIN_DEFAULT}).",
    )
    parser.add_argument(
        "--mbar-max",
        type=float,
        default=_MBAR_MAX_DEFAULT,
        dest="mbar_max",
        metavar="FLOAT",
        help=(
            f"Maximum baryonic mass proxy (log10) for F3 flag "
            f"(default: {_MBAR_MAX_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--min-deep",
        type=int,
        default=_MIN_DEEP_DEFAULT,
        dest="min_deep",
        metavar="INT",
        help=(
            f"Minimum deep-regime points for slope fit (default: {_MIN_DEEP_DEFAULT}). "
            "Galaxies with fewer points get deep_slope=NaN and are excluded from F3."
        ),
    )
    parser.add_argument(
        "--vbar-deep",
        type=float,
        default=_VBAR_DEEP_DEFAULT,
        dest="vbar_deep",
        metavar="FLOAT",
        help=(
            f"Vbar threshold (km/s) defining deep-MOND regime "
            f"(default: {_VBAR_DEEP_DEFAULT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    catalog = generate_catalog(
        input_path=Path(args.input),
        out_dir=Path(args.out),
        vflat_min=args.vflat_min,
        mbar_max=args.mbar_max,
        min_deep=args.min_deep,
        vbar_deep=args.vbar_deep,
    )
    n_f3 = int(catalog["f3_flag"].sum())
    print(
        f"F3 catalog: {len(catalog)} galaxies, {n_f3} flagged as F3 "
        f"(vflat_min={args.vflat_min}, mbar_max={args.mbar_max}, "
        f"min_deep={args.min_deep})"
    )
    print(f"Written to: {Path(args.out) / 'f3_catalog.csv'}")


if __name__ == "__main__":
    main()
