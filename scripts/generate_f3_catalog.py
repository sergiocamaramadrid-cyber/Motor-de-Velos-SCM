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

Usage
-----
::

    python scripts/generate_f3_catalog.py

    python scripts/generate_f3_catalog.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --out  results/f3_catalog_real.csv \\
        --a0   1.2e-10 \\
        --deep-threshold 0.3
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
CSV_DEFAULT = "results/universal_term_comparison_full.csv"
OUT_DEFAULT = "results/f3_catalog_real.csv"

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
) -> pd.DataFrame:
    """Build a per-galaxy β catalog from a per-radial-point DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``galaxy``, ``log_g_bar``, ``log_g_obs``.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Deep-regime threshold as a fraction of a0.

    Returns
    -------
    pd.DataFrame
        One row per galaxy with columns:
        galaxy, n_total, n_deep, friction_slope,
        friction_slope_stderr, velo_inerte_flag.
    """
    required = {"galaxy", "log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing}.\n"
            "Regenerate with an updated run_pipeline() that emits per-radial-point rows."
        )

    rows = []
    for galaxy, grp in df.groupby("galaxy", sort=True):
        result = compute_galaxy_beta(
            grp["log_g_bar"].to_numpy(),
            grp["log_g_obs"].to_numpy(),
            a0=a0,
            deep_threshold=deep_threshold,
        )
        result["galaxy"] = galaxy
        rows.append(result)

    catalog = pd.DataFrame(rows, columns=[
        "galaxy", "n_total", "n_deep",
        "friction_slope", "friction_slope_stderr", "velo_inerte_flag",
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
    catalog = build_catalog(df, a0=args.a0, deep_threshold=args.deep_threshold)

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
