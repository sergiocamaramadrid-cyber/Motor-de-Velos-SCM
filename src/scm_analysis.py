"""
scm_analysis.py — Pipeline for running the Motor de Velos SCM analysis.

Entry point
-----------
Run as a module::

    python -m src.scm_analysis --data-dir data/SPARC --out results/

Or import and call :func:`run_pipeline` programmatically.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .scm_models import (
    v_total,
    chi2_reduced,
    baryonic_tully_fisher,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_galaxy_table(data_dir):
    """Load the SPARC galaxy table (SPARC_Lelli2016c.mrt or .csv).

    The function tries several common file-name variants.  If none are found
    it raises :class:`FileNotFoundError`.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing SPARC data files.

    Returns
    -------
    pd.DataFrame
        Galaxy-level properties with at least the columns:
        ``['Galaxy', 'D', 'Inc', 'L36', 'Vflat', 'e_Vflat']``.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for path in candidates:
        if path.exists():
            sep = "," if path.suffix == ".csv" else r"\s+"
            df = pd.read_csv(path, sep=sep, comment="#")
            return df
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {data_dir}. "
        "Expected SPARC_Lelli2016c.csv or .mrt"
    )


def load_rotation_curve(data_dir, galaxy_name):
    """Load the rotation curve data for a single galaxy.

    Parameters
    ----------
    data_dir : str or Path
        Root directory.
    galaxy_name : str
        Galaxy identifier matching a file ``<galaxy_name>_rotmod.dat`` inside
        ``data_dir/raw/`` or ``data_dir/``.

    Returns
    -------
    pd.DataFrame
        Columns: ``['r', 'v_obs', 'v_obs_err', 'v_gas', 'v_disk', 'v_bul']``
        (velocities in km/s, radii in kpc).
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{galaxy_name}_rotmod.dat",
        data_dir / "raw" / f"{galaxy_name}_rotmod.dat",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(
                path,
                sep=r"\s+",
                comment="#",
                names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
                       "SBdisk", "SBbul"],
            )
            return df[["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]]
    raise FileNotFoundError(
        f"Rotation curve for {galaxy_name} not found in {data_dir}"
    )


# ---------------------------------------------------------------------------
# Per-galaxy fitting
# ---------------------------------------------------------------------------

def fit_galaxy(rc, a0=1.2e-10):
    """Fit upsilon_disk for a single galaxy minimising chi-squared.

    The bulge mass-to-light ratio is fixed at 0.7 (standard assumption).

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data (output of :func:`load_rotation_curve`).
    a0 : float
        Velos characteristic acceleration.

    Returns
    -------
    dict
        Keys: ``upsilon_disk``, ``chi2_reduced``, ``n_points``.
    """
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    def objective(ud):
        vp = v_total(r, v_gas, v_disk, v_bul,
                     upsilon_disk=ud, upsilon_bul=0.7, a0=a0)
        return chi2_reduced(v_obs, v_obs_err, vp)

    result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
    best_ud = float(result.x)
    best_chi2 = float(result.fun)
    return {
        "upsilon_disk": best_ud,
        "chi2_reduced": best_chi2,
        "n_points": len(r),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_dir, out_dir, a0=1.2e-10, verbose=True):
    """Execute the full Motor de Velos SCM analysis pipeline.

    1. Load the SPARC galaxy table.
    2. For each galaxy, load the rotation curve and fit upsilon_disk.
    3. Compute the residual chi-squared and the baryonic Tully-Fisher relation.
    4. Write summary files to *out_dir*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC data.
    out_dir : str or Path
        Output directory (created if necessary).
    a0 : float
        Characteristic velos acceleration (m/s²).
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Galaxy-level results table.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    galaxy_table = load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    records = []
    for name in galaxy_names:
        try:
            rc = load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found", file=sys.stderr)
            continue

        fit = fit_galaxy(rc, a0=a0)
        row = galaxy_table[galaxy_table["Galaxy"] == name].iloc[0]

        v_flat = float(row.get("Vflat", np.nan))
        m_bar_pred = baryonic_tully_fisher(v_flat, a0=a0) if np.isfinite(v_flat) else np.nan

        records.append({
            "galaxy": name,
            "upsilon_disk": fit["upsilon_disk"],
            "chi2_reduced": fit["chi2_reduced"],
            "n_points": fit["n_points"],
            "Vflat_kms": v_flat,
            "M_bar_BTFR_Msun": m_bar_pred,
        })
        if verbose:
            print(f"  {name}: chi2={fit['chi2_reduced']:.2f}, ud={fit['upsilon_disk']:.2f}")

    results_df = pd.DataFrame(records)

    # Stable column order, sort, and explicit types to avoid machine-to-machine diffs
    results_df = results_df[[
        "galaxy", "upsilon_disk", "chi2_reduced",
        "n_points", "Vflat_kms", "M_bar_BTFR_Msun",
    ]]
    results_df["n_points"] = results_df["n_points"].astype(int)
    for col in ("upsilon_disk", "chi2_reduced", "Vflat_kms", "M_bar_BTFR_Msun"):
        results_df[col] = results_df[col].astype(float)
    results_df = results_df.sort_values("galaxy").reset_index(drop=True)

    # --- Write per-galaxy summary (compact audit table for downstream scripts) ---
    per_galaxy_path = out_dir / "per_galaxy_summary.csv"
    results_df.to_csv(per_galaxy_path, index=False)

    # --- Write universal term comparison CSV (full dataset; currently per-galaxy) ---
    # When per-radial-point comparison data is available this file will diverge
    # from per_galaxy_summary.csv and contain one row per radial point.
    csv_path = out_dir / "universal_term_comparison_full.csv"
    results_df.to_csv(csv_path, index=False)

    # --- Write executive summary ---
    _write_executive_summary(results_df, out_dir / "executive_summary.txt")

    # --- Write top-10 LaTeX table ---
    _write_top10_latex(results_df, out_dir / "top10_universal.tex")

    if verbose:
        print(f"\nResults written to {out_dir}")

    return results_df


def _write_executive_summary(df, path):
    """Write a plain-text executive summary of the pipeline results."""
    lines = ["SCM Pipeline — Executive Summary"]
    lines.append(f"N_galaxies: {len(df)}")
    if "chi2_reduced" in df.columns and len(df):
        lines.append(f"chi2_reduced median: {df['chi2_reduced'].median():.4f}")
    if "upsilon_disk" in df.columns and len(df):
        lines.append(f"upsilon_disk median: {df['upsilon_disk'].median():.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_top10_latex(df, path):
    """Write a LaTeX table of the 10 best-fit galaxies."""
    if df.empty:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("% No data\n")
        return

    top10 = df.nsmallest(10, "chi2_reduced")[
        ["galaxy", "chi2_reduced", "upsilon_disk", "Vflat_kms"]
    ]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Top-10 best-fit galaxies (Motor de Velos SCM)}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Galaxy & $\chi^2_\nu$ & $\Upsilon_\star^{\rm disk}$ & $V_{\rm flat}$ (km/s) \\",
        r"\hline",
    ]
    for _, row in top10.iterrows():
        vflat = f"{row['Vflat_kms']:.1f}" if np.isfinite(row["Vflat_kms"]) else "---"
        lines.append(
            f"{row['galaxy']} & {row['chi2_reduced']:.2f} & "
            f"{row['upsilon_disk']:.2f} & {vflat} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Motor de Velos SCM analysis pipeline"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing SPARC data"
    )
    parser.add_argument(
        "--out", default="results/", help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--a0", type=float, default=1.2e-10,
        help="Characteristic velos acceleration in m/s² (default: 1.2e-10)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out,
        a0=args.a0,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
