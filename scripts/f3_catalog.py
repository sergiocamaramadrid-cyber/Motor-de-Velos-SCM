"""
scripts/f3_catalog.py — Per-galaxy friction slope (pendiente de fricción) catalog.

The F3 catalog provides a per-galaxy statistical diagnostic by computing the
deep-regime slope β of the Radial Acceleration Relation (RAR):

    log10(g_obs) = β · log10(g_bar) + const

In the deep-velos / deep-MOND limit the expected value is β = 0.5.
Galaxies where β deviates significantly from 0.5 are flagged with
``velo_inerte_flag = True``, indicating that the Velo Inerte (inert-veil)
pressure term may be responsible for a structural anomaly in that galaxy's
deep-regime behaviour.

Input
-----
Either a pre-computed ``universal_term_comparison_full.csv`` produced by
``src.scm_analysis.run_pipeline()`` (default) or a raw SPARC data directory.

Output
------
``results/f3_catalog.csv`` (by default) with columns:

    galaxy            — galaxy identifier
    n_points          — total number of radial points for this galaxy
    n_deep            — number of deep-regime points (g_bar < threshold × a0)
    deep_frac         — n_deep / n_points
    friction_slope    — OLS slope β in deep regime (NaN if n_deep < 2)
    friction_slope_err— standard error of friction_slope (NaN if n_deep < 3)
    delta_from_mond   — friction_slope − 0.5
    r2_deep           — Pearson r² in deep regime (NaN if n_deep < 2)
    velo_inerte_flag  — True when n_deep ≥ MIN_DEEP_POINTS and
                        |delta_from_mond| > ANOMALY_SIGMA × friction_slope_err

Summary statistics printed to stdout:
    - Mean and median friction slope (all galaxies with valid slope)
    - Count and list of Velo Inerte anomaly galaxies

Usage
-----
Default (reads results/universal_term_comparison_full.csv)::

    python scripts/f3_catalog.py

Explicit options::

    python scripts/f3_catalog.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --out  results/f3_catalog.csv \\
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
DEEP_THRESHOLD_DEFAULT: float = 0.3  # g_bar / a0 < threshold → deep regime
MIN_DEEP_POINTS: int = 10            # minimum deep points for reliable slope
ANOMALY_SIGMA: float = 2.0           # |Δβ| > ANOMALY_SIGMA × stderr → flag
EXPECTED_SLOPE: float = 0.5          # MOND / deep-velos prediction

CSV_DEFAULT = "results/universal_term_comparison_full.csv"
OUT_DEFAULT = "results/f3_catalog.csv"
_SEP = "=" * 64

# Required columns in the input per-radial-point CSV
_REQUIRED_COLS = {"galaxy", "log_g_bar", "log_g_obs", "g_bar"}


# ---------------------------------------------------------------------------
# Core per-galaxy computation
# ---------------------------------------------------------------------------

def per_galaxy_friction_slope(
    group: pd.DataFrame,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> dict:
    """Compute the friction slope β for a single galaxy.

    Parameters
    ----------
    group : pd.DataFrame
        Subset of the per-radial-point CSV for one galaxy.
        Must contain columns ``log_g_bar``, ``log_g_obs``, and ``g_bar``.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of *a0* that defines the deep regime.

    Returns
    -------
    dict
        Keys: ``n_points``, ``n_deep``, ``deep_frac``,
        ``friction_slope``, ``friction_slope_err``, ``delta_from_mond``,
        ``r2_deep``, ``velo_inerte_flag``.
    """
    log_gbar = group["log_g_bar"].values
    log_gobs = group["log_g_obs"].values
    g_bar = group["g_bar"].values

    n_total = len(log_gbar)
    deep_mask = g_bar < deep_threshold * a0
    n_deep = int(deep_mask.sum())
    deep_frac = float(n_deep / max(n_total, 1))

    if n_deep < 2:
        return {
            "n_points": n_total,
            "n_deep": n_deep,
            "deep_frac": deep_frac,
            "friction_slope": float("nan"),
            "friction_slope_err": float("nan"),
            "delta_from_mond": float("nan"),
            "r2_deep": float("nan"),
            "velo_inerte_flag": False,
        }

    x_deep = log_gbar[deep_mask]
    y_deep = log_gobs[deep_mask]

    # Guard against degenerate case: all deep log_g_bar values are identical
    # (zero variance), which makes the linear regression undefined.
    if x_deep.max() == x_deep.min():
        return {
            "n_points": n_total,
            "n_deep": n_deep,
            "deep_frac": deep_frac,
            "friction_slope": float("nan"),
            "friction_slope_err": float("nan"),
            "delta_from_mond": float("nan"),
            "r2_deep": float("nan"),
            "velo_inerte_flag": False,
        }

    slope, _intercept, r_value, _p_value, stderr = linregress(x_deep, y_deep)
    slope = float(slope)
    stderr = float(stderr)
    delta = slope - EXPECTED_SLOPE
    r2 = float(r_value ** 2)

    # Velo Inerte anomaly: statistically significant deviation from β = 0.5
    # and enough deep points to make the regression meaningful.
    velo_inerte_flag = (
        n_deep >= MIN_DEEP_POINTS
        and not np.isnan(stderr)
        and stderr > 0.0
        and abs(delta) > ANOMALY_SIGMA * stderr
    )

    return {
        "n_points": n_total,
        "n_deep": n_deep,
        "deep_frac": deep_frac,
        "friction_slope": slope,
        "friction_slope_err": stderr,
        "delta_from_mond": delta,
        "r2_deep": r2,
        "velo_inerte_flag": bool(velo_inerte_flag),
    }


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_f3_catalog(
    csv_path: Path,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """Build the F3 per-galaxy friction slope catalog.

    Parameters
    ----------
    csv_path : Path
        Per-radial-point CSV (``universal_term_comparison_full.csv``).
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Deep-regime threshold (fraction of a0).

    Returns
    -------
    pd.DataFrame
        F3 catalog with one row per galaxy.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' "
            "first to generate universal_term_comparison_full.csv."
        )

    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}.\n"
            "Regenerate with an updated run_pipeline() that emits per-radial-point rows."
        )

    records = []
    for galaxy_name, group in df.groupby("galaxy", sort=True):
        row = per_galaxy_friction_slope(group, a0=a0, deep_threshold=deep_threshold)
        row["galaxy"] = galaxy_name
        records.append(row)

    catalog = pd.DataFrame(records, columns=[
        "galaxy", "n_points", "n_deep", "deep_frac",
        "friction_slope", "friction_slope_err", "delta_from_mond",
        "r2_deep", "velo_inerte_flag",
    ])
    catalog["n_points"] = catalog["n_points"].astype(int)
    catalog["n_deep"] = catalog["n_deep"].astype(int)
    catalog["velo_inerte_flag"] = catalog["velo_inerte_flag"].astype(bool)

    return catalog


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def format_summary(catalog: pd.DataFrame, a0: float, threshold: float,
                   csv_path: str, out_path: str) -> list[str]:
    """Format the F3 catalog summary report as a list of lines."""
    valid = catalog["friction_slope"].dropna()
    n_flagged = int(catalog["velo_inerte_flag"].sum())
    flagged_galaxies = catalog.loc[catalog["velo_inerte_flag"], "galaxy"].tolist()

    lines = [
        _SEP,
        "  Motor de Velos SCM — F3 Friction Slope Catalog",
        _SEP,
        f"  Input CSV      : {csv_path}",
        f"  Output         : {out_path}",
        f"  a0             : {a0:.2e} m/s²",
        f"  Deep threshold : g_bar < {threshold} × a0",
        f"  Min deep pts   : {MIN_DEEP_POINTS}",
        f"  Anomaly σ      : |Δβ| > {ANOMALY_SIGMA} × stderr",
        "",
        f"  Total galaxies       : {len(catalog)}",
        f"  With valid slope     : {len(valid)}",
        f"  No deep-regime data  : {len(catalog) - len(valid)}",
    ]
    if len(valid):
        lines += [
            "",
            f"  Friction slope (β):",
            f"    Mean             : {valid.mean():.4f}",
            f"    Median           : {valid.median():.4f}",
            f"    Std dev          : {valid.std():.4f}",
            f"    Min              : {valid.min():.4f}",
            f"    Max              : {valid.max():.4f}",
            f"    Expected (MOND)  : {EXPECTED_SLOPE:.4f}",
        ]
    lines += [
        "",
        f"  Velo Inerte anomalies: {n_flagged}",
    ]
    if flagged_galaxies:
        lines.append(f"    Flagged: {', '.join(flagged_galaxies)}")
    lines.append(_SEP)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the F3 per-galaxy friction slope catalog "
            "for the Motor de Velos SCM analysis."
        )
    )
    parser.add_argument(
        "--csv", default=CSV_DEFAULT, metavar="FILE",
        help=f"Per-radial-point input CSV (default: {CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUT_DEFAULT, metavar="FILE",
        help=f"Output catalog path (default: {OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(f"Fraction of a0 defining deep regime "
              f"(default: {DEEP_THRESHOLD_DEFAULT})."),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Generate the F3 catalog and print the summary report.

    Returns the catalog DataFrame so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    catalog = build_f3_catalog(csv_path, a0=args.a0, deep_threshold=args.deep_threshold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    report_lines = format_summary(
        catalog, args.a0, args.deep_threshold, str(csv_path), str(out_path)
    )
    for line in report_lines:
        print(line)

    print(f"\n  F3 catalog written to {out_path}  ({len(catalog)} rows)")
    return catalog


if __name__ == "__main__":
    main()
