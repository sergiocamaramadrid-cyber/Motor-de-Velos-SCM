"""
scripts/build_sparc_full_catalog.py — Build the full SPARC + SCM fit catalog.

Downloads the SPARC dataset (Lelli et al. 2016) if not already present, runs
the Motor de Velos SCM pipeline on every galaxy, measures the per-galaxy
deep-regime slope β (F3 catalog), and merges all results into a single CSV
file suitable for downstream analysis and the run_big_sparc_veil_test.py
validation script.

Outputs
-------
``data/SPARC/sparc_full.csv``
    One row per galaxy with columns from the galaxy table (D, Inc, L36,
    Vflat, …) plus SCM fit results (upsilon_disk, chi2_reduced, n_points,
    M_bar_BTFR_Msun) and deep-regime slope statistics (beta, beta_err,
    n_deep, n_total, reliable, friction_slope, friction_slope_err).

Usage
-----
Default (downloads to data/SPARC, writes data/SPARC/sparc_full.csv)::

    python scripts/build_sparc_full_catalog.py

Custom paths::

    python scripts/build_sparc_full_catalog.py \\
        --data-dir data/SPARC \\
        --out data/SPARC/sparc_full.csv

Skip download (use existing data)::

    python scripts/build_sparc_full_catalog.py --skip-download

References
----------
Lelli, McGaugh & Schombert (2016), AJ 152, 157.
http://astroweb.cwru.edu/SPARC/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_sparc_data import download_sparc
from scripts.generate_f3_catalog import measure_galaxy_beta
from src.scm_analysis import (
    load_galaxy_table,
    load_rotation_curve,
    fit_galaxy,
)
from src.scm_models import baryonic_tully_fisher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A0_DEFAULT = 1.2e-10
DEEP_THRESHOLD_DEFAULT = 0.3
MIN_DEEP_POINTS_DEFAULT = 5

# Galaxy-table columns to carry through into the merged catalog when present
_GALAXY_TABLE_COLS = [
    "T", "D", "e_D", "Inc", "e_Inc",
    "L36", "e_L36", "Re", "MHI",
    "Vflat", "e_Vflat", "Q",
]


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_full_catalog(
    data_dir: str | Path = "data/SPARC",
    out: str | Path = "data/SPARC/sparc_full.csv",
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    min_deep_points: int = MIN_DEEP_POINTS_DEFAULT,
    skip_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download SPARC data and build the full merged per-galaxy catalog.

    Parameters
    ----------
    data_dir : str or Path
        SPARC data directory (galaxy table + ``raw/`` rotation curves).
    out : str or Path
        Output CSV path.
    a0 : float
        Characteristic velos acceleration (m/s²).
    deep_threshold : float
        Deep-regime threshold as fraction of *a0*.
    min_deep_points : int
        Minimum deep points for a reliable β measurement.
    skip_download : bool
        If True, skip the download step and use whatever data is present.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Full per-galaxy catalog (also written to *out*).
    """
    data_dir = Path(data_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download SPARC data
    # ------------------------------------------------------------------
    if skip_download:
        if verbose:
            print("=== Step 1: Skipping download (using existing SPARC data) ===")
    else:
        if verbose:
            print("=== Step 1: Downloading SPARC data ===")
        ok = download_sparc(data_dir)
        if not ok:
            print(
                "WARNING: Some files could not be downloaded. "
                "Proceeding with available data.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # 2. Load galaxy table
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Step 2: Loading galaxy table ===")
    galaxy_table = load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()
    if verbose:
        print(f"Galaxies in table: {len(galaxy_names)}")

    # ------------------------------------------------------------------
    # 3. Per-galaxy SCM fit + β measurement
    # ------------------------------------------------------------------
    if verbose:
        print("\n=== Step 3: Running SCM fits and measuring deep-regime slope β ===")

    records = []
    for name in galaxy_names:
        try:
            rc = load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found", file=sys.stderr)
            continue

        fit = fit_galaxy(rc, a0=a0)
        beta_meas = measure_galaxy_beta(
            rc,
            upsilon_disk=fit["upsilon_disk"],
            a0=a0,
            deep_threshold=deep_threshold,
            min_deep_points=min_deep_points,
        )

        galaxy_row = galaxy_table[galaxy_table["Galaxy"] == name].iloc[0]

        record: dict = {"galaxy": name}

        # Carry through galaxy-table columns that are present
        for col in _GALAXY_TABLE_COLS:
            if col in galaxy_table.columns:
                record[col] = galaxy_row.get(col, np.nan)

        # SCM fit results
        record["upsilon_disk"] = fit["upsilon_disk"]
        record["chi2_reduced"] = fit["chi2_reduced"]
        record["n_points"] = fit["n_points"]

        # Baryonic Tully-Fisher prediction
        try:
            v_flat = float(record.get("Vflat", np.nan))
        except (TypeError, ValueError):
            v_flat = float("nan")
        record["M_bar_BTFR_Msun"] = (
            float(baryonic_tully_fisher(v_flat, a0=a0))
            if np.isfinite(v_flat)
            else float("nan")
        )

        # Deep-regime slope (F3 / β)
        record.update({
            "beta": beta_meas["beta"],
            "beta_err": beta_meas["beta_err"],
            "n_deep": beta_meas["n_deep"],
            "n_total": beta_meas["n_total"],
            "reliable": beta_meas["reliable"],
            "friction_slope": beta_meas["beta"],
            "friction_slope_err": beta_meas["beta_err"],
        })

        records.append(record)

        if verbose:
            beta_str = (
                f"{beta_meas['beta']:.3f}"
                if not np.isnan(beta_meas["beta"])
                else "NaN"
            )
            print(
                f"  {name}: chi2={fit['chi2_reduced']:.2f}, "
                f"β={beta_str}, n_deep={beta_meas['n_deep']}"
            )

    # ------------------------------------------------------------------
    # 4. Assemble, sort, and save
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    df = df.sort_values("galaxy").reset_index(drop=True)
    df.to_csv(out, index=False)

    if verbose:
        n_reliable = int(df["reliable"].sum()) if "reliable" in df.columns else 0
        print(f"\nFull catalog written to {out}")
        print(f"  {len(df)} galaxies fitted, {n_reliable} with reliable β")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full SPARC + Motor de Velos SCM fit catalog. "
            "Downloads SPARC data if needed and merges per-galaxy SCM fits "
            "with F3 deep-regime slope measurements into sparc_full.csv."
        )
    )
    parser.add_argument(
        "--data-dir", default="data/SPARC",
        help="SPARC data directory (default: data/SPARC).",
    )
    parser.add_argument(
        "--out", default="data/SPARC/sparc_full.csv",
        help="Output CSV path (default: data/SPARC/sparc_full.csv).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=f"Deep-regime threshold as fraction of a0 (default: {DEEP_THRESHOLD_DEFAULT}).",
    )
    parser.add_argument(
        "--min-deep-points", type=int, default=MIN_DEEP_POINTS_DEFAULT,
        dest="min_deep_points",
        help=f"Minimum deep points for reliable β (default: {MIN_DEEP_POINTS_DEFAULT}).",
    )
    parser.add_argument(
        "--skip-download", action="store_true", dest="skip_download",
        help="Skip downloading and use existing SPARC data.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = _parse_args(argv)
    build_full_catalog(
        data_dir=args.data_dir,
        out=args.out,
        a0=args.a0,
        deep_threshold=args.deep_threshold,
        min_deep_points=args.min_deep_points,
        skip_download=args.skip_download,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
