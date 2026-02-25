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
    KPC_TO_M,
    v_baryonic,
    v_total,
    chi2_reduced,
    baryonic_tully_fisher,
)

# Convert (km/s)²/kpc → m/s² (used for per-point g_bar / g_obs)
_CONV = 1e6 / KPC_TO_M

# Guard against division by zero for near-zero radii (well below physical kpc)
_MIN_RADIUS_KPC = 1e-10


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
    compare_rows = []  # per-radial-point rows for universal_term_comparison_full.csv
    for name in galaxy_names:
        try:
            rc = load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found", file=sys.stderr)
            continue

        fit = fit_galaxy(rc, a0=a0)
        row = galaxy_table[galaxy_table["Galaxy"] == name].iloc[0]

        # Per-radial-point baryonic and observed accelerations for RAR/deep-slope
        r_arr = rc["r"].values
        v_obs_arr = rc["v_obs"].values
        vb_arr = v_baryonic(
            r_arr, rc["v_gas"].values, rc["v_disk"].values, rc["v_bul"].values,
            upsilon_disk=fit["upsilon_disk"], upsilon_bul=0.7,
        )
        g_bar_arr = vb_arr ** 2 / np.maximum(r_arr, _MIN_RADIUS_KPC) * _CONV  # m/s²
        g_obs_arr = v_obs_arr ** 2 / np.maximum(r_arr, _MIN_RADIUS_KPC) * _CONV  # m/s²
        valid = (g_bar_arr > 0) & (g_obs_arr > 0)
        for k in range(len(r_arr)):
            if valid[k]:
                compare_rows.append({
                    "galaxy": name,
                    "r_kpc": float(r_arr[k]),
                    "g_bar": float(g_bar_arr[k]),
                    "g_obs": float(g_obs_arr[k]),
                    "log_g_bar": float(np.log10(g_bar_arr[k])),
                    "log_g_obs": float(np.log10(g_obs_arr[k])),
                })

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

    # --- Write universal term comparison CSV (per radial point for RAR/deep-slope) ---
    # Columns: galaxy, r_kpc, g_bar, g_obs, log_g_bar, log_g_obs
    compare_df = pd.DataFrame(compare_rows)
    csv_path = out_dir / "universal_term_comparison_full.csv"
    compare_df.to_csv(csv_path, index=False)

    # --- Write global statistics CSV ---
    _write_sparc_global(results_df, compare_df, out_dir / "sparc_global.csv")

    # --- Write per-galaxy audit CSV (audit/sparc_global.csv) ---
    audit_df = _build_sparc_global_df(results_df, compare_df)
    export_sparc_global(audit_df, out_dir / "audit" / "sparc_global.csv")

    # --- Write executive summary ---
    _write_executive_summary(results_df, out_dir / "executive_summary.txt")

    # --- Write top-10 LaTeX table ---
    _write_top10_latex(results_df, out_dir / "top10_universal.tex")

    if verbose:
        print(f"\nResults written to {out_dir}")

    return results_df


def export_sparc_global(df, out_csv):
    """Exporta el CSV global para auditoría OOS (scripts/audit_scm.py).

    Requiere (columnas mínimas):

    - galaxy_id (str)
    - logM      (float)  = log10(Mbar / Msun)
    - log_gbar  (float)  = log10(gbar / (m s⁻²))
    - log_j     (float)  = log10(j_* / (kpc km s⁻¹))
    - v_obs     (float)  = log10(v_obs / (km s⁻¹))  ← in log

    Parameters
    ----------
    df : pd.DataFrame
        Per-galaxy audit table with the required columns.
    out_csv : str or Path
        Destination path.  Parent directories are created as needed.

    Raises
    ------
    ValueError
        If any required column is absent from *df*.
    """
    required = ["galaxy_id", "logM", "log_gbar", "log_j", "v_obs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"export_sparc_global: faltan columnas {missing}")

    out = df[required].copy()
    out["galaxy_id"] = out["galaxy_id"].astype(str)

    for c in ["logM", "log_gbar", "log_j", "v_obs"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = (out
           .replace([np.inf, -np.inf], np.nan)
           .dropna(subset=required)
           .reset_index(drop=True))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _build_sparc_global_df(results_df, compare_df):
    """Build the per-galaxy audit DataFrame expected by :func:`export_sparc_global`.

    Derived quantities
    ------------------
    logM      = log10(M_bar_BTFR_Msun)
    log_gbar  = median log10(g_bar) over all radial points of the galaxy
    log_j     = log10(median specific angular momentum j_* = r * v_obs [kpc km/s])
                where v_obs per radial point is recovered from g_obs:
                v_obs = sqrt(g_obs * r / _CONV)
    v_obs     = log10(Vflat_kms)  — flat rotation velocity (proxy for asymptotic v_obs)

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of the pipeline (columns: galaxy, upsilon_disk, …, Vflat_kms,
        M_bar_BTFR_Msun).
    compare_df : pd.DataFrame
        Per-radial-point data (columns: galaxy, r_kpc, g_bar, g_obs,
        log_g_bar, log_g_obs).

    Returns
    -------
    pd.DataFrame
        Columns: galaxy_id, logM, log_gbar, log_j, v_obs.
    """
    rows = []
    compare_grouped = (compare_df.groupby("galaxy")
                       if not compare_df.empty else {})

    for _, res_row in results_df.iterrows():
        name = res_row["galaxy"]
        m_bar = res_row["M_bar_BTFR_Msun"]
        vflat = res_row["Vflat_kms"]

        logM = float(np.log10(m_bar)) if (np.isfinite(m_bar) and m_bar > 0) else np.nan
        v_obs_log = (float(np.log10(vflat))
                     if (np.isfinite(vflat) and vflat > 0) else np.nan)

        # Per-radial-point quantities for this galaxy
        if not compare_df.empty and name in compare_df["galaxy"].values:
            gal_pts = compare_df[compare_df["galaxy"] == name]
            log_gbar = float(gal_pts["log_g_bar"].median())

            r_arr = gal_pts["r_kpc"].values
            g_obs_arr = gal_pts["g_obs"].values
            # v_obs[km/s] = sqrt(g_obs[m/s²] * r[kpc] / _CONV)
            v_obs_pt = np.sqrt(np.maximum(g_obs_arr * r_arr / _CONV, 0.0))
            j_arr = r_arr * v_obs_pt
            valid_j = j_arr[j_arr > 0]
            j_med = float(np.median(valid_j)) if len(valid_j) else np.nan
            log_j = float(np.log10(j_med)) if (np.isfinite(j_med) and j_med > 0) else np.nan
        else:
            log_gbar = np.nan
            log_j = np.nan

        rows.append({
            "galaxy_id": name,
            "logM": logM,
            "log_gbar": log_gbar,
            "log_j": log_j,
            "v_obs": v_obs_log,
        })

    return pd.DataFrame(rows)


def _write_sparc_global(results_df, compare_df, path):
    """Write a single-row global statistics CSV (sparc_global.csv).

    Columns
    -------
    n_galaxies          : number of galaxies processed.
    n_radial_points     : total radial points across all galaxies.
    chi2_reduced_median : median reduced chi-squared across galaxies.
    chi2_reduced_mean   : mean reduced chi-squared.
    upsilon_disk_median : median best-fit disk mass-to-light ratio.
    upsilon_disk_mean   : mean best-fit disk mass-to-light ratio.
    Vflat_kms_median    : median flat rotation velocity (km/s).
    log_g_bar_median    : median log10(g_bar) over all radial points.
    log_g_obs_median    : median log10(g_obs) over all radial points.
    """
    n_gal = len(results_df)
    n_pts = int(compare_df.shape[0]) if not compare_df.empty else 0

    chi2_med = float(results_df["chi2_reduced"].median()) if n_gal else float("nan")
    chi2_mean = float(results_df["chi2_reduced"].mean()) if n_gal else float("nan")
    ud_med = float(results_df["upsilon_disk"].median()) if n_gal else float("nan")
    ud_mean = float(results_df["upsilon_disk"].mean()) if n_gal else float("nan")
    vflat_col = results_df["Vflat_kms"].dropna()
    vflat_med = float(vflat_col.median()) if len(vflat_col) else float("nan")

    lg_bar_med = float(compare_df["log_g_bar"].median()) if n_pts else float("nan")
    lg_obs_med = float(compare_df["log_g_obs"].median()) if n_pts else float("nan")

    row = {
        "n_galaxies": n_gal,
        "n_radial_points": n_pts,
        "chi2_reduced_median": chi2_med,
        "chi2_reduced_mean": chi2_mean,
        "upsilon_disk_median": ud_med,
        "upsilon_disk_mean": ud_mean,
        "Vflat_kms_median": vflat_med,
        "log_g_bar_median": lg_bar_med,
        "log_g_obs_median": lg_obs_med,
    }
    pd.DataFrame([row]).to_csv(path, index=False)


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
