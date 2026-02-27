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
import json
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


def load_pressure_calibration(path="data/calibration/local_group_xi_calibration.json"):
    """Load the empirical ξ pressure calibration from the Local Group sample.

    Parameters
    ----------
    path : str or Path
        Path to the calibration JSON file.  Defaults to the repository-relative
        location ``data/calibration/local_group_xi_calibration.json``.

    Returns
    -------
    dict
        Full calibration data as parsed from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the calibration file does not exist at *path*.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path) as f:
        return json.load(f)


def estimate_xi_from_sfr(log_sfr):
    """Estimate the ξ (xi) pressure parameter from the decimal log of the SFR.

    Uses the empirically calibrated SFR relation derived from the Local Group
    sample (v0.6.1):

        ξ = 1.33 + 0.21 × log₁₀(SFR)

    The result is clamped to the validated range [1.28, 1.42].

    Parameters
    ----------
    log_sfr : float
        log₁₀ of the star-formation rate in M_sun yr⁻¹.

    Returns
    -------
    float
        Estimated ξ value, clamped to [1.28, 1.42].
    """
    # Constants from data/calibration/local_group_xi_calibration.json sfr_model
    intercept = 1.33
    slope = 0.21
    xi = intercept + slope * log_sfr
    # Clamp to the validated range from xi_statistics in the calibration file
    xi = max(1.28, min(1.42, xi))
    return xi


# ---------------------------------------------------------------------------
# Per-galaxy fitting
# ---------------------------------------------------------------------------

def load_custom_rotation_curve(path):
    """Load a simplified rotation curve from a plain-text file.

    The expected format is three whitespace-separated columns with an optional
    ``#``-prefixed header line::

        # radius_kpc  velocity_kms  error_kms
        0.5  55.1  2.2
        1.0  78.4  2.1
        ...

    The function tolerates both space and tab separators and skips any line
    starting with ``#``.

    Parameters
    ----------
    path : str or Path
        Path to the rotation-curve text file.

    Returns
    -------
    pd.DataFrame
        Columns: ``['r', 'v_obs', 'v_obs_err', 'v_gas', 'v_disk', 'v_bul']``
        (velocities in km/s, radii in kpc).  Gas and bulge velocity components
        are not available in the simplified format and are therefore set to
        zero (``v_gas = v_bul = 0``); ``v_disk`` is set equal to ``v_obs`` so
        that ``fit_galaxy`` uses the full observed velocity as the disk baseline.
        The caller should be aware that kinematic decomposition is unavailable.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file does not contain at least the three required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom rotation curve not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["r", "v_obs", "v_obs_err"],
        usecols=[0, 1, 2],
    )
    if df.empty or len(df.columns) < 3:
        raise ValueError(
            f"Custom rotation curve file {path} must contain at least three "
            "columns: radius_kpc, velocity_kms, error_kms"
        )
    # Synthetic zero-valued velocity components (no baryonic decomposition available)
    df["v_gas"] = 0.0
    df["v_disk"] = df["v_obs"].copy()
    df["v_bul"] = 0.0
    return df[["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]]


def run_custom_galaxy(name, custom_data, out_dir, a0=_A0_DEFAULT,
                      detect_pressure_injectors=False, audit_mode=None,
                      verbose=True):
    """Run the SCM analysis for a single galaxy from a custom rotation curve.

    This is the entry point for the M81 Group and other non-SPARC datasets that
    supply a simplified three-column rotation curve (radius, velocity, error).

    Parameters
    ----------
    name : str
        Galaxy identifier (used in output file names and summary fields).
    custom_data : str or Path
        Path to the simplified rotation curve file understood by
        :func:`load_custom_rotation_curve`.
    out_dir : str or Path
        Output directory (created if necessary).
    a0 : float
        Characteristic acceleration (m/s²).
    detect_pressure_injectors : bool
        If ``True``, annotate the output with a pressure-injector detection
        flag based on the calibrated ξ range.
    audit_mode : str or None
        Audit intensity label (e.g. ``"high-pressure"``).  Stored in the
        output audit summary as metadata; does not change numerical results.
    verbose : bool
        Print progress if ``True``.

    Returns
    -------
    dict
        Result dict with keys: ``galaxy``, ``upsilon_disk``, ``chi2_reduced``,
        ``n_points``, ``xi_estimated``, ``pressure_injector_detected``,
        ``audit_mode``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = load_custom_rotation_curve(custom_data)
    fit = fit_galaxy(rc, a0=a0)

    # ξ estimate: use default from config; SFR not available for custom data
    xi_estimated = 1.37  # global empirical calibration centre

    pressure_injector_detected = False
    if detect_pressure_injectors:
        # Flag galaxies whose fit quality is significantly worse than expected
        # (χ²_ν > 2.0 indicates the baryonic model cannot explain the kinematics,
        # consistent with an external pressure contribution).
        pressure_injector_detected = bool(fit["chi2_reduced"] > 2.0)

    result = {
        "galaxy": name,
        "upsilon_disk": fit["upsilon_disk"],
        "chi2_reduced": fit["chi2_reduced"],
        "n_points": fit["n_points"],
        "xi_estimated": xi_estimated,
        "pressure_injector_detected": pressure_injector_detected,
        "audit_mode": audit_mode or "standard",
    }

    if verbose:
        flag = " [PRESSURE INJECTOR]" if pressure_injector_detected else ""
        print(f"  {name}: chi2={fit['chi2_reduced']:.2f}, "
              f"ud={fit['upsilon_disk']:.2f}, xi={xi_estimated:.2f}{flag}")

    # Write per-galaxy summary CSV
    pd.DataFrame([result]).to_csv(out_dir / f"{name}_result.csv", index=False)

    # Write audit summary JSON
    audit = {
        "xi_calibration": {
            "version": "v0.6.1",
            "model": "xi = 1.33 + 0.21 log10(SFR)",
            "range": [1.28, 1.42],
        },
        "custom_run": {
            "galaxy": name,
            "custom_data": str(custom_data),
            "detect_pressure_injectors": detect_pressure_injectors,
            "audit_mode": audit_mode or "standard",
        },
    }
    _write_audit_summary(pd.DataFrame([result]), audit,
                         out_dir / "audit_summary.json")

    return result


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

    # --- Write executive summary ---
    _write_executive_summary(results_df, out_dir / "executive_summary.txt")

    # --- Write top-10 LaTeX table ---
    _write_top10_latex(results_df, out_dir / "top10_universal.tex")

    # --- Write audit summary (xi calibration traceability) ---
    audit = {}
    audit["xi_calibration"] = {
        "version": "v0.6.1",
        "model": "xi = 1.33 + 0.21 log10(SFR)",
        "range": [1.28, 1.42],
    }
    _write_audit_summary(results_df, audit, out_dir / "audit_summary.json")

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


def _write_audit_summary(df, audit, path):
    """Write a JSON audit summary combining pipeline statistics and audit metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy-level results (output of :func:`run_pipeline`).
    audit : dict
        Audit metadata dict, e.g. containing ``xi_calibration`` entry.
    path : Path
        Destination JSON file.
    """
    summary = dict(audit)
    summary["pipeline_stats"] = {
        "n_galaxies": int(len(df)),
        "chi2_reduced_median": (
            float(df["chi2_reduced"].median()) if "chi2_reduced" in df.columns and len(df) else None
        ),
        "upsilon_disk_median": (
            float(df["upsilon_disk"].median()) if "upsilon_disk" in df.columns and len(df) else None
        ),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


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
        "--data-dir", default=None, help="Directory containing SPARC data"
    )
    parser.add_argument(
        "--out", "--outdir", dest="out", default="results/",
        help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--a0", type=float, default=1.2e-10,
        help="Characteristic velos acceleration in m/s² (default: 1.2e-10)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output"
    )
    # Custom single-galaxy mode (M81 Group / non-SPARC datasets)
    parser.add_argument(
        "--custom-data", default=None,
        help="Path to a simplified rotation curve file (radius_kpc, velocity_kms, error_kms)"
    )
    parser.add_argument(
        "--target-galaxy", default=None,
        help="Galaxy name to use when running with --custom-data"
    )
    parser.add_argument(
        "--detect-pressure-injectors", action="store_true",
        help="Enable pressure-injector detection heuristic"
    )
    parser.add_argument(
        "--audit-mode", default=None,
        help="Audit intensity label stored in the output audit summary "
             "(e.g. 'high-pressure', 'standard')"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.custom_data is not None:
        # Single-galaxy mode: custom rotation curve
        name = args.target_galaxy or Path(args.custom_data).stem
        run_custom_galaxy(
            name=name,
            custom_data=args.custom_data,
            out_dir=args.out,
            a0=args.a0,
            detect_pressure_injectors=args.detect_pressure_injectors,
            audit_mode=args.audit_mode,
            verbose=not args.quiet,
        )
    else:
        if args.data_dir is None:
            import sys as _sys
            print("error: --data-dir is required unless --custom-data is supplied",
                  file=_sys.stderr)
            _sys.exit(1)
        run_pipeline(
            data_dir=args.data_dir,
            out_dir=args.out,
            a0=args.a0,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
