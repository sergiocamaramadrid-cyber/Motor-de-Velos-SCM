"""
scm_analysis.py — SCM v0.2 analysis pipeline (Motor de Velos)

Produces the SCM v0.2 telemetry log and the binned-residuals diagnostics CSV.

Telemetry format
----------------
[SCM v0.2] g0_hat=<value_SI>  g0_err=<value_SI>
[INFO] g_bar quantiles:  p10=<v>  p25=<v>  p50=<v>  p75=<v>  p90=<v>
[WARN] Optimal g0 touches parameter bound — results may be unreliable.

Output file
-----------
results/diagnostics/residuals_binned_v02.csv
    bin_center_log10, mean_residual, std_residual, n_points

Usage
-----
    python -m src.scm_analysis [--data-dir DATA_DIR] [--out OUT_DIR]

If DATA_DIR is omitted (or the SPARC files are absent) the pipeline falls back
to a synthetic dataset so that the telemetry and CSV are always produced.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np

from .scm_models import fit_g0

# ---------------------------------------------------------------------------
# Logging setup — plain format so tags like [SCM v0.2] stay unambiguous
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERSION = "0.2"
_RESIDUALS_CSV = "residuals_binned_v02.csv"
_N_BINS = 10  # number of log10(g_bar) bins for the residuals table


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_sparc(data_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load g_bar and g_obs arrays from the SPARC processed directory.

    Returns None if the expected file is not found.
    """
    candidate = data_dir / "processed" / "sparc_gbar_gobs.csv"
    if not candidate.is_file():
        return None

    g_bar_list, g_obs_list = [], []
    with open(candidate, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                gb = float(row["g_bar"])
                go = float(row["g_obs"])
            except (KeyError, ValueError):
                continue
            if gb > 0 and go > 0:
                g_bar_list.append(gb)
                g_obs_list.append(go)

    if len(g_bar_list) < 2:
        return None

    return np.array(g_bar_list), np.array(g_obs_list)


def _synthetic_dataset(rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset that approximates the RAR scatter.

    Uses the fiducial g0 = 1.2e-10 m/s^2 and adds 0.11 dex log-normal scatter
    representative of the SPARC sample.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    g0_true = 1.2e-10  # m s^-2
    # 2693 points ~ full SPARC size; log10(g_bar) uniform in [-13, -8.5]
    n = 2693
    log10_g_bar = rng.uniform(-13.0, -8.5, size=n)
    g_bar = 10.0 ** log10_g_bar

    # True model prediction
    x = np.sqrt(g_bar / g0_true)
    g_obs_true = g_bar / (1.0 - np.exp(-x))

    # Add observational scatter (0.11 dex, Lelli+2017)
    scatter = rng.normal(0.0, 0.11, size=n)
    g_obs = g_obs_true * 10.0 ** scatter

    return g_bar, g_obs


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------

def _print_telemetry(g0_hat: float, g0_err: float, g_bar: np.ndarray, at_bound: bool) -> None:
    """Emit the standard SCM v0.2 telemetry lines."""
    logger.info(
        "[SCM v0.2] g0_hat=%.4e  g0_err=%.4e  (m s^-2)",
        g0_hat,
        g0_err,
    )

    qs = np.quantile(g_bar, [0.10, 0.25, 0.50, 0.75, 0.90])
    logger.info(
        "[INFO] g_bar quantiles:  p10=%.3e  p25=%.3e  p50=%.3e  p75=%.3e  p90=%.3e  (m s^-2)",
        *qs,
    )

    if at_bound:
        logger.warning(
            "[WARN] Optimal g0 touches parameter bound — results may be unreliable."
        )


# ---------------------------------------------------------------------------
# Residuals CSV
# ---------------------------------------------------------------------------

def _write_residuals_csv(
    g_bar: np.ndarray,
    residuals: np.ndarray,
    out_dir: Path,
    n_bins: int = _N_BINS,
) -> Path:
    """Bin residuals in log10(g_bar) and write the diagnostics CSV.

    Parameters
    ----------
    g_bar:
        Baryonic acceleration array [m s^-2].
    residuals:
        Log10 residuals (log10 g_obs / g_pred) per data point.
    out_dir:
        Directory in which to write the CSV.
    n_bins:
        Number of equal-width bins in log10(g_bar).

    Returns
    -------
    Path to the written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / _RESIDUALS_CSV

    log10_g_bar = np.log10(g_bar)
    edges = np.linspace(log10_g_bar.min(), log10_g_bar.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict] = []
    for i in range(n_bins):
        mask = (log10_g_bar >= edges[i]) & (log10_g_bar < edges[i + 1])
        # Include the right edge in the last bin
        if i == n_bins - 1:
            mask = (log10_g_bar >= edges[i]) & (log10_g_bar <= edges[i + 1])
        n_pts = int(mask.sum())
        if n_pts == 0:
            rows.append(
                {
                    "bin_center_log10": f"{centers[i]:.4f}",
                    "mean_residual": "nan",
                    "std_residual": "nan",
                    "n_points": "0",
                }
            )
        else:
            rows.append(
                {
                    "bin_center_log10": f"{centers[i]:.4f}",
                    "mean_residual": f"{residuals[mask].mean():.6f}",
                    "std_residual": f"{residuals[mask].std(ddof=1) if n_pts > 1 else 0.0:.6f}",
                    "n_points": str(n_pts),
                }
            )

    fieldnames = ["bin_center_log10", "mean_residual", "std_residual", "n_points"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


# ---------------------------------------------------------------------------
# Reprint helper (Notarial de Residuos)
# ---------------------------------------------------------------------------

def reprint_residuals(csv_path: Path) -> None:
    """Print the full residuals table from *csv_path* to stdout."""
    if not csv_path.is_file():
        logger.error("[ERROR] Residuals file not found: %s", csv_path)
        return

    with open(csv_path, newline="", encoding="utf-8") as fh:
        content = fh.read()

    print("\n=== Notarial de Residuos — Reprint ===")
    print(f"File: {csv_path}\n")
    print(content)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(data_dir: Path, out_dir: Path) -> Path:
    """Execute the full SCM v0.2 analysis.

    Parameters
    ----------
    data_dir:
        Root directory for SPARC data (can be absent — synthetic data used).
    out_dir:
        Root output directory.  Diagnostics go to ``out_dir/diagnostics/``.

    Returns
    -------
    Path to the written residuals CSV.
    """
    # 1. Load data
    dataset = _load_sparc(data_dir)
    if dataset is None:
        logger.info("[INFO] SPARC data not found at %s — using synthetic dataset.", data_dir)
        g_bar, g_obs = _synthetic_dataset()
    else:
        g_bar, g_obs = dataset
        logger.info("[INFO] Loaded %d data points from SPARC.", len(g_bar))

    # 2. Fit g0
    fit = fit_g0(g_bar, g_obs)

    # 3. Telemetry
    _print_telemetry(fit["g0_hat"], fit["g0_err"], g_bar, fit["at_bound"])

    # 4. Save residuals CSV
    diag_dir = out_dir / "diagnostics"
    csv_path = _write_residuals_csv(g_bar, fit["residuals"], diag_dir)
    logger.info("[INFO] Residuals saved to %s", csv_path)

    # 5. Reprint table (Notarial de Residuos)
    reprint_residuals(csv_path)

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scm_analysis",
        description="SCM v0.2 analysis — Motor de Velos",
    )
    parser.add_argument(
        "--data-dir",
        default="data/SPARC",
        help="Root directory for SPARC data (default: data/SPARC).",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output root directory (default: results).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_analysis(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
