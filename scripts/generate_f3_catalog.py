"""
scripts/generate_f3_catalog.py — Per-galaxy deep-regime friction-slope catalog.

Theory
------
In the deep-MOND / deep-velos regime (g_bar ≪ a0):

    g_obs ≈ sqrt(g_bar · a0)

In log-space this gives slope β = 0.5.  A per-galaxy OLS fit:

    log10(g_obs) = β · log10(g_bar) + const

restricted to points with g_bar < threshold × a0 yields a *friction slope*
β that probes whether each galaxy individually follows the deep-velos relation.

Output columns
--------------
galaxy              : galaxy identifier
n_total             : total radial points for the galaxy
n_deep              : number of points with g_bar < threshold × a0
friction_slope      : OLS slope β (NaN when n_deep < MIN_DEEP_POINTS)
friction_slope_err  : standard error of β (NaN when n_deep < MIN_DEEP_POINTS)
velo_inerte_flag    : 1 if |β − 0.5| ≤ 2·σ_β, else 0 (NaN if slope is NaN)

Usage
-----
With default paths::

    python scripts/generate_f3_catalog.py

Explicit options::

    python scripts/generate_f3_catalog.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --g0   1.2e-10 \\
        --deep-threshold 0.3 \\
        --out  results/f3_catalog_real.csv
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

G0_DEFAULT = 1.2e-10          # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT = 0.3  # fraction of a0 defining "deep regime"
EXPECTED_SLOPE = 0.5          # MOND / deep-velos prediction
MIN_DEEP_POINTS = 10          # minimum for a meaningful per-galaxy regression
CSV_DEFAULT = "results/universal_term_comparison_full.csv"
OUT_DEFAULT = "results/f3_catalog_real.csv"

# ---------------------------------------------------------------------------
# Core per-galaxy computation
# ---------------------------------------------------------------------------


def fit_galaxy_slope(log_gbar: np.ndarray, log_gobs: np.ndarray,
                     g0: float = G0_DEFAULT,
                     deep_threshold: float = DEEP_THRESHOLD_DEFAULT) -> dict:
    """Fit the deep-regime friction slope β for a single galaxy.

    Parameters
    ----------
    log_gbar : array_like
        log10 of baryonic acceleration per radial point.
    log_gobs : array_like
        log10 of observed acceleration per radial point.
    g0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of g0 below which a point is "deep".

    Returns
    -------
    dict with keys:
        n_total            — total radial points
        n_deep             — deep-regime points
        friction_slope     — OLS slope β (nan if n_deep < MIN_DEEP_POINTS)
        friction_slope_err — standard error of β (nan if n_deep < MIN_DEEP_POINTS)
        velo_inerte_flag   — 1 if |β−0.5| ≤ 2·σ_β, else 0 (nan if slope nan)
    """
    log_gbar = np.asarray(log_gbar, dtype=float)
    log_gobs = np.asarray(log_gobs, dtype=float)

    g_bar = 10.0 ** log_gbar
    deep_mask = g_bar < deep_threshold * g0
    n_total = len(log_gbar)
    n_deep = int(deep_mask.sum())

    if n_deep < MIN_DEEP_POINTS:
        return {
            "n_total": n_total,
            "n_deep": n_deep,
            "friction_slope": float("nan"),
            "friction_slope_err": float("nan"),
            "velo_inerte_flag": float("nan"),
        }

    slope, _intercept, _r, _p, stderr = linregress(
        log_gbar[deep_mask], log_gobs[deep_mask]
    )
    flag = 1 if abs(slope - EXPECTED_SLOPE) <= 2.0 * stderr else 0

    return {
        "n_total": n_total,
        "n_deep": n_deep,
        "friction_slope": float(slope),
        "friction_slope_err": float(stderr),
        "velo_inerte_flag": float(flag),
    }


def build_catalog(df: pd.DataFrame,
                  g0: float = G0_DEFAULT,
                  deep_threshold: float = DEEP_THRESHOLD_DEFAULT) -> pd.DataFrame:
    """Build the per-galaxy catalog from a per-radial-point DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: galaxy, log_g_bar, log_g_obs.
    g0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of g0 defining deep regime.

    Returns
    -------
    pd.DataFrame with one row per galaxy and columns:
        galaxy, n_total, n_deep, friction_slope, friction_slope_err,
        velo_inerte_flag
    """
    required = {"galaxy", "log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}."
        )

    rows = []
    for galaxy, gdf in df.groupby("galaxy", sort=False):
        result = fit_galaxy_slope(
            gdf["log_g_bar"].to_numpy(),
            gdf["log_g_obs"].to_numpy(),
            g0=g0,
            deep_threshold=deep_threshold,
        )
        result["galaxy"] = galaxy
        rows.append(result)

    catalog = pd.DataFrame(rows, columns=[
        "galaxy", "n_total", "n_deep",
        "friction_slope", "friction_slope_err", "velo_inerte_flag",
    ])
    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-galaxy deep-regime friction-slope catalog "
            "(F3 catalog)."
        )
    )
    parser.add_argument(
        "--csv", default=CSV_DEFAULT,
        help=f"Per-radial-point CSV (default: {CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--data-dir", default=None, dest="data_dir",
        help=(
            "If provided, run the full pipeline on this data directory "
            "and write universal_term_comparison_full.csv to --out-dir "
            "before generating the catalog."
        ),
    )
    parser.add_argument(
        "--g0", type=float, default=G0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {G0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(
            f"Fraction of g0 defining deep regime "
            f"(default: {DEEP_THRESHOLD_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--out", default=OUT_DEFAULT,
        help=f"Output CSV path (default: {OUT_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Generate the F3 per-galaxy catalog and write it to CSV.

    Returns the catalog DataFrame so callers can inspect it programmatically.
    """
    args = _parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first,\n"
            "or use --data-dir to specify a data directory."
        )

    df = pd.read_csv(csv_path)
    catalog = build_catalog(df, g0=args.g0, deep_threshold=args.deep_threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    n_analyzed = catalog["friction_slope"].notna().sum()
    print(
        f"F3 catalog written to {out_path}\n"
        f"  N total rows     : {len(catalog)}\n"
        f"  N analyzed (non-NaN friction_slope): {n_analyzed}"
    )
    return catalog


if __name__ == "__main__":
    main()
