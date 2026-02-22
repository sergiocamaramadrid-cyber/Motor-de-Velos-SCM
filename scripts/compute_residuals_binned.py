"""
compute_residuals_binned.py
----------------------------
SCM v0.2 diagnostic: compute binned RAR residuals and fit g0_hat.

For each galaxy in the analysis results, loads the raw rotation-curve data,
computes baryonic and observed centripetal accelerations, fits the RAR
acceleration scale g0_hat globally, then bins the log10 residuals in
log10(g_bar) space.

Usage
-----
    python scripts/compute_residuals_binned.py \\
        --csv results/universal_term_comparison_full.csv \\
        --out results/diagnostics/

    python scripts/compute_residuals_binned.py \\
        --csv results/universal_term_comparison_full.csv \\
        --data-dir data/SPARC \\
        --n-bins 10 \\
        --out results/diagnostics/

Outputs
-------
results/diagnostics/residuals_binned_v02.csv
    Columns:
        g_bar_center    – bin centre in m/s² (linear)
        median_residual – median of log10(g_obs) − log10(g_rar) [dex]
        mad_residual    – median absolute deviation of residuals [dex]
        count           – number of data points in bin
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

# Allow running as `python scripts/compute_residuals_binned.py` from repo root
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.read_iorio import read_batch
from src.scm_models import v_baryon, g_from_v, rar_g_obs, fit_g0_rar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def collect_rar_data(
    galaxies: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute g_bar and g_obs arrays (m/s²) across all galaxies.

    Parameters
    ----------
    galaxies:
        Dict mapping galaxy name → DataFrame (from read_batch).

    Returns
    -------
    (g_bar, g_obs) arrays of shape (N_total,) in m/s².
    """
    all_g_bar: list[np.ndarray] = []
    all_g_obs: list[np.ndarray] = []

    for name, df in galaxies.items():
        r = df["R"].to_numpy()
        vobs = df["Vobs"].to_numpy()
        vgas = df["Vgas"].to_numpy()
        vdisk = df["Vdisk"].to_numpy()
        vbul = df["Vbul"].to_numpy()

        vb = v_baryon(vgas, vdisk, vbul)

        # Only include points where both velocities and radius are physically valid
        mask = (r > 0) & (vobs > 0) & (np.abs(vb) > 0) & np.isfinite(r) & np.isfinite(vobs)
        if mask.sum() == 0:
            logger.warning("%s: no valid data points, skipping.", name)
            continue

        g_bar = g_from_v(np.abs(vb[mask]), r[mask])
        g_obs = g_from_v(vobs[mask], r[mask])

        # Keep only physically positive accelerations
        valid = (g_bar > 0) & (g_obs > 0)
        if valid.sum() == 0:
            logger.warning("%s: all accelerations non-positive, skipping.", name)
            continue

        all_g_bar.append(g_bar[valid])
        all_g_obs.append(g_obs[valid])

        logger.debug("%s: %d valid RAR points.", name, valid.sum())

    if not all_g_bar:
        raise ValueError("No valid RAR data points found in any galaxy.")

    return np.concatenate(all_g_bar), np.concatenate(all_g_obs)


def compute_binned_residuals(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_hat: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin RAR log10 residuals in equal-width log10(g_bar) bins.

    Parameters
    ----------
    g_bar, g_obs:
        Baryonic and observed accelerations in m/s².
    g0_hat:
        Best-fit g0 used to compute the RAR prediction.
    n_bins:
        Number of bins.

    Returns
    -------
    pd.DataFrame with columns g_bar_center, median_residual, mad_residual, count.
    """
    log_gb = np.log10(g_bar)
    residuals = np.log10(g_obs) - np.log10(rar_g_obs(g_bar, g0_hat))

    edges = np.linspace(log_gb.min(), log_gb.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    median_stat, _, bin_num = binned_statistic(
        log_gb, residuals, statistic="median", bins=edges
    )
    count_stat, _, _ = binned_statistic(
        log_gb, residuals, statistic="count", bins=edges
    )

    mad_stat = np.full(n_bins, np.nan)
    for i in range(n_bins):
        pts = residuals[bin_num == (i + 1)]
        if len(pts) >= 2:
            mad_stat[i] = float(np.median(np.abs(pts - median_stat[i])))

    df = pd.DataFrame({
        "g_bar_center":    10.0 ** centres,
        "median_residual": median_stat,
        "mad_residual":    mad_stat,
        "count":           count_stat.astype(int),
    })
    return df.dropna(subset=["median_residual"]).reset_index(drop=True)


def _resolve_data_dir(csv_path: Path, data_dir: str | None) -> Path:
    """Return the raw-data directory, searching common locations if not given."""
    if data_dir:
        p = Path(data_dir)
        if p.is_dir():
            return p
        # Try relative to csv file
        p2 = csv_path.parent.parent / data_dir
        if p2.is_dir():
            return p2
        sys.exit(f"Data directory not found: {data_dir}")

    # Default search order
    candidates = [
        Path("data/SPARC"),
        csv_path.parent.parent / "data" / "SPARC",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    sys.exit(
        "Could not find raw data directory.  "
        "Pass --data-dir explicitly, e.g. --data-dir data/SPARC"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/compute_residuals_binned.py",
        description="SCM v0.2: compute binned RAR residuals and fit g0_hat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/compute_residuals_binned.py \\\n"
            "      --csv results/universal_term_comparison_full.csv \\\n"
            "      --out results/diagnostics/\n"
        ),
    )
    parser.add_argument(
        "--csv", metavar="FILE", required=True,
        help="Path to universal_term_comparison_full.csv.",
    )
    parser.add_argument(
        "--data-dir", metavar="DIR", default=None,
        help="Directory with raw *.txt galaxy files (default: data/SPARC).",
    )
    parser.add_argument(
        "--out", metavar="DIR", default="results/diagnostics/",
        help="Output directory (default: results/diagnostics/).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10, metavar="N",
        help="Number of log10(g_bar) bins (default: 10).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    summary = pd.read_csv(csv_path)
    if "galaxy" not in summary.columns:
        sys.exit("CSV must contain a 'galaxy' column.")

    # Filter to successfully fitted galaxies
    if "veredicto" in summary.columns:
        ok = summary[summary["veredicto"] == "OK"]["galaxy"].tolist()
    else:
        ok = summary["galaxy"].tolist()

    if not ok:
        sys.exit("No OK galaxies found in the CSV.")

    # Locate raw data files
    data_dir = _resolve_data_dir(csv_path, args.data_dir)
    logger.info("Raw data directory: %s", data_dir)

    # Locate raw data files
    data_dir_listing = {p.stem: p for p in data_dir.glob("*.txt")}
    filepaths = []
    for gal in ok:
        base = gal.replace("_rotmod", "")
        for stem, path in data_dir_listing.items():
            if stem == gal or stem.startswith(base):
                filepaths.append(path)
    filepaths = sorted(set(filepaths))

    if not filepaths:
        sys.exit(
            f"No *.txt files found in {data_dir} matching the galaxies in {csv_path}.\n"
            "Pass --data-dir pointing to the directory with the raw rotation-curve files."
        )

    logger.info("Loading %d galaxy file(s) …", len(filepaths))
    galaxies = read_batch(filepaths)
    if not galaxies:
        sys.exit("Failed to load any galaxy data.")

    # ── Collect all g_bar / g_obs data points ──────────────────────────────
    g_bar, g_obs = collect_rar_data(galaxies)
    logger.info("Total RAR data points: %d", len(g_bar))

    # ── Fit g0_hat globally ─────────────────────────────────────────────────
    fit = fit_g0_rar(g_bar, g_obs)
    g0_hat = fit["g0_hat"]
    rms = fit["rms_dex"]
    at_bound = fit["at_bound"]

    print(f"[SCM v0.2] g0_hat = {g0_hat:.4e} m/s²")
    print(f"[SCM v0.2] rms_dex = {rms:.4f} dex  |  N_pts = {fit['n_pts']}")
    if at_bound:
        print("[SCM v0.2] WARNING: g0_hat is at or near an optimisation bound — "
              "check units or broaden bounds.")

    # ── Compute binned residuals ────────────────────────────────────────────
    bins_df = compute_binned_residuals(g_bar, g_obs, g0_hat, n_bins=args.n_bins)

    # ── Save output ─────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "residuals_binned_v02.csv"
    bins_df.to_csv(out_path, index=False)
    logger.info("Binned residuals written to %s", out_path)

    # ── Print summary ───────────────────────────────────────────────────────
    print(f"\nbins_effective: {len(bins_df)}")
    print(f"g_bar_center_min: {bins_df['g_bar_center'].min():.4e} m/s²")
    print(f"g_bar_center_max: {bins_df['g_bar_center'].max():.4e} m/s²")
    print(
        bins_df[["g_bar_center", "median_residual", "mad_residual", "count"]]
        .to_string(index=False)
    )

    # ── Deep-regime diagnosis ───────────────────────────────────────────────
    deep_bins = bins_df[bins_df["g_bar_center"] < 0.1 * g0_hat]
    if not deep_bins.empty:
        deep_med = deep_bins["median_residual"].median()
        collapsed = abs(deep_med) < 0.05
        print(f"\n[SCM v0.2] Deep-regime median residual: {deep_med:+.4f} dex")
        print(f"[SCM v0.2] Deep-regime collapse: {'YES ✓' if collapsed else 'NO ✗'}")
    else:
        print("\n[SCM v0.2] No bins in deep regime (g_bar < 0.1 × g0_hat) — need lower-acceleration data.")


if __name__ == "__main__":
    main()
