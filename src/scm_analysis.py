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
from scipy.stats import linregress

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

# Fiducial characteristic acceleration (m/s²) — same value used in scm_models defaults
_A0_DEFAULT = 1.2e-10


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

    # --- Write deep-slope test CSV (derived directly from compare_df) ---
    _write_deep_slope_csv(compare_df, out_dir / "deep_slope_test.csv", a0=a0)

    # --- Write sensitivity analysis CSV (a0 grid scan) ---
    # Imported here to avoid a circular import (sensitivity imports from scm_analysis).
    from .sensitivity import run_sensitivity  # noqa: PLC0415
    run_sensitivity(data_dir, out_dir, verbose=verbose)

    # --- Write audit artefacts (VIF, stability, quality status, features) ---
    _write_audit_artifacts(compare_df, out_dir / "audit", a0=a0)

    # --- Write executive summary ---
    _write_executive_summary(results_df, out_dir / "executive_summary.txt")

    # --- Write top-10 LaTeX table ---
    _write_top10_latex(results_df, out_dir / "top10_universal.tex")

    if verbose:
        print(f"\nResults written to {out_dir}")

    return results_df


def _write_deep_slope_csv(compare_df, path, a0=_A0_DEFAULT, deep_threshold=0.3):
    """Compute and write the deep-regime slope test CSV.

    Parameters
    ----------
    compare_df : pd.DataFrame
        Per-radial-point table with columns ``log_g_bar`` and ``log_g_obs``.
    path : Path
        Destination file.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of *a0* that defines the deep regime: a radial point is
        "deep" if g_bar < deep_threshold × a0 (e.g. 0.3 means 30% of a0).
    """
    if compare_df.empty or not {"log_g_bar", "log_g_obs"}.issubset(compare_df.columns):
        pd.DataFrame(columns=[
            "n_total", "n_deep", "deep_frac", "slope", "intercept",
            "stderr", "r_value", "p_value", "delta_from_mond", "log_g0_pred",
        ]).to_csv(path, index=False)
        return

    log_gbar = compare_df["log_g_bar"].values
    log_gobs = compare_df["log_g_obs"].values
    g_bar = 10.0 ** log_gbar
    deep_mask = g_bar < deep_threshold * a0
    n_total = len(log_gbar)
    n_deep = int(deep_mask.sum())

    if n_deep < 2:
        result = {
            "n_total": n_total, "n_deep": n_deep,
            "deep_frac": float(n_deep / max(n_total, 1)),
            "slope": float("nan"), "intercept": float("nan"),
            "stderr": float("nan"), "r_value": float("nan"),
            "p_value": float("nan"), "delta_from_mond": float("nan"),
            "log_g0_pred": float("nan"),
        }
    else:
        slope, intercept, r_value, p_value, stderr = linregress(
            log_gbar[deep_mask], log_gobs[deep_mask]
        )
        result = {
            "n_total": n_total,
            "n_deep": n_deep,
            "deep_frac": float(n_deep / max(n_total, 1)),
            "slope": float(slope),
            "intercept": float(intercept),
            "stderr": float(stderr),
            "r_value": float(r_value),
            "p_value": float(p_value),
            "delta_from_mond": float(slope - 0.5),
            "log_g0_pred": float(2.0 * intercept),
        }
    pd.DataFrame([result]).to_csv(path, index=False)


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

def _write_audit_artifacts(compare_df, audit_dir, a0=_A0_DEFAULT):
    """Write audit artefacts: VIF table, stability metrics, quality status, features.

    Parameters
    ----------
    compare_df : pd.DataFrame
        Per-radial-point table with at least ``log_g_bar`` and ``galaxy`` columns
        (output of :func:`run_pipeline`).
    audit_dir : str or Path
        Destination directory (created if necessary).
    a0 : float
        Characteristic acceleration (m/s²) used as the hinge threshold.
    """
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    log_a0 = np.log10(a0)

    # ------------------------------------------------------------------
    # audit_features.csv — summary statistics for each modelling feature
    # ------------------------------------------------------------------
    feature_rows = []
    for col, label in [("log_g_bar", "log_g_bar"), ("log_g_obs", "log_g_obs")]:
        if col in compare_df.columns and len(compare_df):
            vals = compare_df[col].values
            feature_rows.append({
                "feature": label,
                "n_points": len(vals),
                "mean": round(float(vals.mean()), 6),
                "std": round(float(vals.std()), 6),
                "min": round(float(vals.min()), 6),
                "max": round(float(vals.max()), 6),
            })
    if "log_g_bar" in compare_df.columns and len(compare_df):
        hv = np.maximum(compare_df["log_g_bar"].values - log_a0, 0.0)
        feature_rows.append({
            "feature": "hinge",
            "n_points": len(hv),
            "mean": round(float(hv.mean()), 6),
            "std": round(float(hv.std()), 6),
            "min": round(float(hv.min()), 6),
            "max": round(float(hv.max()), 6),
        })
    pd.DataFrame(feature_rows).to_csv(audit_dir / "audit_features.csv", index=False)

    # ------------------------------------------------------------------
    # vif_table.csv — VIF for each predictor in the hinge regression
    # Design matrix: [intercept, log_g_bar, hinge=max(log_g_bar-log_a0, 0)]
    # ------------------------------------------------------------------
    vif_rows = []
    if "log_g_bar" in compare_df.columns and len(compare_df) >= 3:
        log_gbar = compare_df["log_g_bar"].values
        hinge = np.maximum(log_gbar - log_a0, 0.0)
        X = np.column_stack([np.ones(len(log_gbar)), log_gbar, hinge])
        for j, fname in enumerate(("intercept", "log_g_bar", "hinge")):
            y_j = X[:, j]
            rest = np.delete(X, j, axis=1)
            ss_tot = float(np.sum((y_j - y_j.mean()) ** 2))
            if ss_tot < 1e-15:
                vif = float("inf")
            else:
                coef = np.linalg.lstsq(rest, y_j, rcond=None)[0]
                ss_res = float(np.sum((y_j - rest @ coef) ** 2))
                r2 = min(max(1.0 - ss_res / ss_tot, 0.0), 1.0 - 1e-10)
                vif = 1.0 / (1.0 - r2)
            vif_rows.append({"feature": fname, "vif": round(float(vif), 4)})
    pd.DataFrame(vif_rows).to_csv(audit_dir / "vif_table.csv", index=False)

    # ------------------------------------------------------------------
    # stability_metrics.csv — condition number and key diagnostics
    # ------------------------------------------------------------------
    hinge_vif = next(
        (r["vif"] for r in vif_rows if r["feature"] == "hinge"), float("nan")
    )
    kappa = float("nan")
    if "log_g_bar" in compare_df.columns and len(compare_df) >= 3:
        log_gbar = compare_df["log_g_bar"].values
        hinge = np.maximum(log_gbar - log_a0, 0.0)
        Xf = np.column_stack([log_gbar, hinge])
        std = Xf.std(axis=0)
        std = np.where(std < 1e-15, 1.0, std)
        Xn = (Xf - Xf.mean(axis=0)) / std
        sv = np.linalg.svd(Xn, compute_uv=False)
        kappa = round(float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"), 4)

    n_gal = int(compare_df["galaxy"].nunique()) if "galaxy" in compare_df.columns else 0
    stability_rows = [
        {"metric": "kappa", "value": kappa},
        {"metric": "hinge_vif", "value": round(float(hinge_vif), 4)},
        {"metric": "n_points", "value": len(compare_df)},
        {"metric": "n_galaxies", "value": n_gal},
    ]
    pd.DataFrame(stability_rows).to_csv(audit_dir / "stability_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # quality_status.txt — PASS / WARNING verdict
    # ------------------------------------------------------------------
    hinge_vif_val = float(hinge_vif) if not np.isnan(float(hinge_vif)) else float("nan")
    kappa_val = float(kappa) if not np.isnan(float(kappa)) else float("nan")
    if (
        not np.isnan(hinge_vif_val)
        and not np.isnan(kappa_val)
        and hinge_vif_val < 5.0
        and kappa_val < 30.0
    ):
        status = "PASS"
    else:
        status = "WARNING"
    quality_lines = [
        f"quality_status: {status}",
        f"hinge_vif: {hinge_vif_val:.4f}",
        f"kappa: {kappa_val:.4f}",
        "threshold_hinge_vif: 5.0",
        "threshold_kappa: 30.0",
    ]
    (audit_dir / "quality_status.txt").write_text(
        "\n".join(quality_lines) + "\n", encoding="utf-8"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Motor de Velos SCM analysis pipeline"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing SPARC data"
    )
    # Accept both --out (legacy) and --outdir (canonical) for output directory.
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--out", default=None, help="Output directory (falls back to results/ if neither --out nor --outdir is given)"
    )
    out_group.add_argument(
        "--outdir", default=None, help="Output directory (alias for --out; preferred canonical form)"
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
    out_dir = args.outdir if args.outdir is not None else (args.out or "results/")
    run_pipeline(
        data_dir=args.data_dir,
        out_dir=out_dir,
        a0=args.a0,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
