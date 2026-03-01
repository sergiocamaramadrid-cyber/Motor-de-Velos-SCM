"""
scripts/generate_f3_catalog.py — Per-galaxy deep-regime friction-slope (β) catalog.

Theory
------
In the deep-MOND / deep-velos regime (g_bar ≪ a0):

    g_obs ≈ sqrt(g_bar · a0)

In log-space this is a straight line:

    log10(g_obs) = β · log10(g_bar) + const

with expected slope β = 0.5.  This script estimates β per galaxy via OLS
regression restricted to deep-regime points (g_bar < deep_threshold × a0).

Base-60 hierarchical scale coordinate
--------------------------------------
The sexagesimal (base-60) system naturally aligns with astronomical angular and
rotational units (degrees → arcminutes → arcseconds).  A structural coordinate

    S_SCM = log_60(r / r0)

converts galactic radii into discrete hierarchy levels, where r0 is a reference
radius (default: 1 kpc).  This is computed for each galaxy as the base-60
logarithm of the median deep-regime radius.

    hierarchy_level(x, x0) = ln(x / x0) / ln(60)

velo_inerte_flag semantics
--------------------------
  1   → statistically consistent with β = 0.5 within 2σ  (|β − 0.5| ≤ 2·stderr)
  0   → statistically inconsistent at ≥ 2σ               (|β − 0.5| > 2·stderr)
  NaN → insufficient deep-regime data (fewer than 2 points)

Output columns
--------------
  galaxy             — galaxy identifier
  n_total            — total radial points for this galaxy
  n_deep             — deep-regime points used in the regression
  friction_slope     — OLS slope β  (NaN if n_deep < 2)
  friction_slope_stderr — standard error of β  (NaN if n_deep < 2)
  velo_inerte_flag   — 1 / 0 / NaN  (see above)
  hierarchy_scm      — S_SCM = log_60(r_median_deep / r0_kpc)
                       (NaN if r_kpc column absent or n_deep < 1)

Usage
-----
::

    python scripts/generate_f3_catalog.py

    python scripts/generate_f3_catalog.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --out  results/f3_catalog_real.csv \\
        --a0   1.2e-10 \\
        --deep-threshold 0.3 \\
        --r0-kpc 1.0
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

A0_DEFAULT: float = 1.2e-10          # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT: float = 0.3  # g_bar < threshold × a0  → deep regime
EXPECTED_SLOPE: float = 0.5          # MOND / deep-velos prediction
BASE_60: float = 60.0                # sexagesimal base
R0_KPC_DEFAULT: float = 1.0          # reference radius for S_SCM (kpc)
CSV_DEFAULT = "results/universal_term_comparison_full.csv"
OUT_DEFAULT = "results/f3_catalog_real.csv"

# ---------------------------------------------------------------------------
# Base-60 hierarchy level
# ---------------------------------------------------------------------------


def hierarchy_level(x: "np.ndarray | float", x0: float) -> "np.ndarray | float":
    """Compute the base-60 hierarchical scale coordinate S = log_60(x / x0).

    This converts a physical quantity *x* into a discrete hierarchy level
    relative to the reference *x0*, using the sexagesimal base that underlies
    astronomical angular units (degrees → arcminutes → arcseconds) and the
    time system (hours → minutes → seconds).

    Parameters
    ----------
    x : array_like or float
        Physical values (must be positive).
    x0 : float
        Reference value (same units as *x*, must be positive).

    Returns
    -------
    np.ndarray or float
        S = log_60(x / x0) = ln(x / x0) / ln(60).
        Returns NaN for non-positive inputs.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(x / x0) / np.log(BASE_60)
    return float(result) if result.ndim == 0 else result

# ---------------------------------------------------------------------------
# Core per-galaxy computation
# ---------------------------------------------------------------------------


def compute_galaxy_beta(
    log_gbar: np.ndarray,
    log_gobs: np.ndarray,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> dict:
    """Estimate the deep-regime friction slope β for a single galaxy.

    Parameters
    ----------
    log_gbar : array_like
        log10 of baryonic centripetal acceleration (m/s²) per radial point.
    log_gobs : array_like
        log10 of observed centripetal acceleration (m/s²) per radial point.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of a0 below which a point is "deep": g_bar < threshold × a0.

    Returns
    -------
    dict with keys:
        n_total              — total radial points for this galaxy
        n_deep               — deep-regime points used in regression
        friction_slope       — OLS slope β  (NaN if n_deep < 2)
        friction_slope_stderr— standard error of β  (NaN if n_deep < 2)
        velo_inerte_flag     — 1.0 / 0.0 / NaN
    """
    log_gbar = np.asarray(log_gbar, dtype=float)
    log_gobs = np.asarray(log_gobs, dtype=float)

    g_bar = 10.0 ** log_gbar
    deep_mask = g_bar < deep_threshold * a0
    n_total = len(log_gbar)
    n_deep = int(deep_mask.sum())

    if n_deep < 2:
        return {
            "n_total": n_total,
            "n_deep": n_deep,
            "friction_slope": float("nan"),
            "friction_slope_stderr": float("nan"),
            "velo_inerte_flag": float("nan"),
        }

    slope, _intercept, _r, _p, stderr = linregress(
        log_gbar[deep_mask], log_gobs[deep_mask]
    )
    slope = float(slope)
    stderr = float(stderr)

    flag = 1.0 if abs(slope - EXPECTED_SLOPE) <= 2.0 * stderr else 0.0

    return {
        "n_total": n_total,
        "n_deep": n_deep,
        "friction_slope": slope,
        "friction_slope_stderr": stderr,
        "velo_inerte_flag": flag,
    }


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_catalog(
    df: pd.DataFrame,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    r0_kpc: float = R0_KPC_DEFAULT,
) -> pd.DataFrame:
    """Build a per-galaxy β catalog from a per-radial-point DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``galaxy``, ``log_g_bar``, ``log_g_obs``.
        Optional column ``r_kpc`` is used to compute ``hierarchy_scm``.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Deep-regime threshold as a fraction of a0.
    r0_kpc : float
        Reference radius for the base-60 scale coordinate S_SCM (kpc).
        Default: 1.0 kpc.

    Returns
    -------
    pd.DataFrame
        One row per galaxy with columns:
        galaxy, n_total, n_deep, friction_slope,
        friction_slope_stderr, velo_inerte_flag, hierarchy_scm.
    """
    required = {"galaxy", "log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing}.\n"
            "Regenerate with an updated run_pipeline() that emits per-radial-point rows."
        )

    has_r_kpc = "r_kpc" in df.columns

    rows = []
    for galaxy, grp in df.groupby("galaxy", sort=True):
        result = compute_galaxy_beta(
            grp["log_g_bar"].to_numpy(),
            grp["log_g_obs"].to_numpy(),
            a0=a0,
            deep_threshold=deep_threshold,
        )
        result["galaxy"] = galaxy

        # Compute S_SCM = log_60(r_median_deep / r0_kpc)
        if has_r_kpc:
            g_bar = 10.0 ** grp["log_g_bar"].to_numpy()
            deep_mask = g_bar < deep_threshold * a0
            r_deep = grp["r_kpc"].to_numpy()[deep_mask]
            if len(r_deep) >= 1 and np.median(r_deep) > 0:
                result["hierarchy_scm"] = hierarchy_level(np.median(r_deep), r0_kpc)
            else:
                result["hierarchy_scm"] = float("nan")
        else:
            result["hierarchy_scm"] = float("nan")

        rows.append(result)

    catalog = pd.DataFrame(rows, columns=[
        "galaxy", "n_total", "n_deep",
        "friction_slope", "friction_slope_stderr", "velo_inerte_flag",
        "hierarchy_scm",
    ])
    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-galaxy deep-regime friction-slope (β) catalog.\n"
            "Reads a per-radial-point CSV and outputs one row per galaxy."
        )
    )
    parser.add_argument(
        "--csv", default=CSV_DEFAULT,
        help=f"Per-radial-point input CSV (default: {CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUT_DEFAULT,
        help=f"Output catalog CSV path (default: {OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(
            f"Fraction of a0 defining deep regime "
            f"(default: {DEEP_THRESHOLD_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--r0-kpc", type=float, default=R0_KPC_DEFAULT,
        dest="r0_kpc",
        help=(
            f"Reference radius for the base-60 scale coordinate S_SCM (kpc). "
            f"(default: {R0_KPC_DEFAULT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run the catalog generator and write output CSV.

    Returns the catalog DataFrame so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )

    df = pd.read_csv(csv_path)
    catalog = build_catalog(df, a0=args.a0, deep_threshold=args.deep_threshold,
                            r0_kpc=args.r0_kpc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    n_total = len(catalog)
    n_valid = int(catalog["friction_slope"].apply(np.isfinite).sum())
    n_consistent = int((catalog["velo_inerte_flag"] == 1).sum())
    n_inconsistent = int((catalog["velo_inerte_flag"] == 0).sum())
    n_nan = int(catalog["velo_inerte_flag"].isna().sum())

    print(f"\n=== F3 CATALOG GENERATED ===")
    print(f"Input  : {csv_path}")
    print(f"Output : {out_path}")
    print(f"Total galaxies      : {n_total}")
    print(f"With valid β        : {n_valid}")
    print(f"Consistent (flag=1) : {n_consistent}")
    print(f"Inconsistent (flag=0): {n_inconsistent}")
    print(f"Insufficient (NaN)  : {n_nan}")

    return catalog


if __name__ == "__main__":
    main()
