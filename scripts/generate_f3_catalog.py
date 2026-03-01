"""
scripts/generate_f3_catalog.py — Per-galaxy deep-regime friction slope (β) catalog.

Theory
------
In the deep-MOND / deep-velos regime (g_bar ≪ a0):

    g_obs ≈ sqrt(g_bar · a0)

In log-space this is a straight line with expected slope β = 0.5:

    log10(g_obs) = β · log10(g_bar) + const

This script groups the per-radial-point CSV (produced by scm_analysis) by galaxy,
fits the deep-regime slope β for each galaxy, and writes a per-galaxy catalog.

Output columns
--------------
galaxy              — galaxy identifier
n_total             — total radial points for this galaxy
n_deep              — deep-regime points (g_bar < deep_threshold × g0)
friction_slope      — fitted β (NaN when n_deep < 2)
friction_slope_err  — standard error of β
r_value             — Pearson r of the deep-regime fit
p_value             — two-tailed p-value of the slope
velo_inerte_flag    — Consistency flag with the MOND/deep-velos prediction:
                      1 (True)  → fitted β is consistent with β = 0.5 within 2σ
                      0 (False) → fitted β deviates from β = 0.5 by ≥ 2σ
                      NaN       → insufficient deep-regime points to fit β

Usage
-----
::

    python scripts/generate_f3_catalog.py

    python scripts/generate_f3_catalog.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --g0   1.2e-10 \\
        --deep-threshold 0.3 \\
        --out  results/f3_catalog.csv
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

G0_DEFAULT: float = 1.2e-10          # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT: float = 0.3  # fraction of g0
EXPECTED_SLOPE: float = 0.5          # MOND / deep-velos prediction
CSV_DEFAULT: str = "results/universal_term_comparison_full.csv"
OUT_DEFAULT: str = "results/f3_catalog.csv"

# ---------------------------------------------------------------------------
# Core per-galaxy fitter
# ---------------------------------------------------------------------------


def fit_galaxy_slope(
    log_gbar: np.ndarray,
    log_gobs: np.ndarray,
    g0: float = G0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> dict:
    """Fit the deep-regime slope β for a single galaxy.

    Parameters
    ----------
    log_gbar : array_like
        log10 of baryonic acceleration per radial point.
    log_gobs : array_like
        log10 of observed acceleration per radial point.
    g0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Points with g_bar < deep_threshold × g0 are "deep regime".

    Returns
    -------
    dict with keys:
        n_total, n_deep, friction_slope, friction_slope_err,
        r_value, p_value, velo_inerte_flag

        velo_inerte_flag semantics:
            1 (True)  — fitted β is consistent with β = 0.5 within 2σ
                        (|friction_slope − 0.5| ≤ 2 × friction_slope_err)
            0 (False) — fitted β deviates from β = 0.5 by ≥ 2σ
            NaN       — insufficient deep-regime points to compute stderr
                        (n_deep < 2 or stderr = 0)
    """
    log_gbar = np.asarray(log_gbar, dtype=float)
    log_gobs = np.asarray(log_gobs, dtype=float)

    g_bar = 10.0 ** log_gbar
    deep_mask = g_bar < deep_threshold * g0
    n_total = len(log_gbar)
    n_deep = int(deep_mask.sum())

    nan_result: dict = {
        "n_total": n_total,
        "n_deep": n_deep,
        "friction_slope": float("nan"),
        "friction_slope_err": float("nan"),
        "r_value": float("nan"),
        "p_value": float("nan"),
        "velo_inerte_flag": float("nan"),
    }

    if n_deep < 2:
        return nan_result

    slope, _intercept, r_value, p_value, stderr = linregress(
        log_gbar[deep_mask], log_gobs[deep_mask]
    )

    # velo_inerte_flag: consistency with the MOND prediction β = 0.5 within 2σ
    # 1 (True)  → |slope − 0.5| ≤ 2·stderr  (consistent)
    # 0 (False) → |slope − 0.5| > 2·stderr  (deviant)
    # NaN       → stderr ≤ 0 or non-finite   (undetermined)
    flag: float
    if np.isfinite(stderr) and stderr > 0:
        flag = 1.0 if abs(slope - EXPECTED_SLOPE) <= 2.0 * stderr else 0.0
    else:
        flag = float("nan")

    return {
        "n_total": n_total,
        "n_deep": n_deep,
        "friction_slope": float(slope),
        "friction_slope_err": float(stderr),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "velo_inerte_flag": flag,
    }


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_f3_catalog(
    df: pd.DataFrame,
    g0: float = G0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """Build the per-galaxy F3 catalog from a per-radial-point DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: galaxy, log_g_bar, log_g_obs.
    g0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Deep-regime threshold as a fraction of g0.

    Returns
    -------
    pd.DataFrame
        Per-galaxy catalog sorted by galaxy name.
    """
    required = {"galaxy", "log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    records = []
    for galaxy, gdf in df.groupby("galaxy", sort=True):
        row = fit_galaxy_slope(
            gdf["log_g_bar"].to_numpy(),
            gdf["log_g_obs"].to_numpy(),
            g0=g0,
            deep_threshold=deep_threshold,
        )
        row["galaxy"] = galaxy
        records.append(row)

    catalog = pd.DataFrame(records, columns=[
        "galaxy", "n_total", "n_deep",
        "friction_slope", "friction_slope_err",
        "r_value", "p_value", "velo_inerte_flag",
    ])
    return catalog.sort_values("galaxy").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-galaxy deep-regime friction slope (β) catalog "
            "from a SPARC per-radial-point CSV."
        )
    )
    parser.add_argument(
        "--csv", default=CSV_DEFAULT,
        help=f"Per-radial-point input CSV (default: {CSV_DEFAULT}).",
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
        help=f"Output catalog CSV path (default: {OUT_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run the catalog generator and return the catalog DataFrame."""
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )

    df = pd.read_csv(csv_path)
    required = {"galaxy", "log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}.\n"
            "Regenerate with an updated run_pipeline() that emits per-radial-point rows."
        )

    catalog = build_f3_catalog(df, g0=args.g0, deep_threshold=args.deep_threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)
    print(f"F3 catalog written to {out_path}  ({len(catalog)} galaxies)")

    return catalog


if __name__ == "__main__":
    main()
