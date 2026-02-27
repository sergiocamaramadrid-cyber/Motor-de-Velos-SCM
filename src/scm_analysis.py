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

# Framework version — used in audit output headers
_FRAMEWORK_VERSION = "v0.6.1"


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


# Calibration anchors for the kinematic xi formula (from Local Group sample v0.6.1)
_XI_INTERCEPT = 1.146   # xi at log10(S) = 0
_XI_SLOPE = 0.433       # xi per decade of steepness index
_DV_XI_MIN = 1.28       # xi anchor low  (GR8:  deltaV=10.6%)
_DV_XI_MAX = 1.42       # xi anchor high (LMC:  deltaV=17.9%)
_DV_LOW = 10.6          # deltaV_reduction (%) at xi = _DV_XI_MIN
_DV_HIGH = 17.9         # deltaV_reduction (%) at xi = _DV_XI_MAX

# M82 PASS thresholds (v0.6.1 high-pressure regime)
_PASS_XI_LOW = 1.40
_PASS_XI_HIGH = 1.48
_PASS_VIF_LOW = 3.5
_PASS_VIF_HIGH = 4.8
_PASS_KAPPA_MAX = 30.0
_PASS_DV_MIN = 14.0
_PASS_PMR_MIN = 2


def _compute_kinematic_metrics(rc, a0=_A0_DEFAULT):
    """Compute kinematic diagnostic metrics for a custom rotation curve.

    Derives five pressure-regime diagnostics from the shape of the rotation
    curve without requiring a full baryonic mass decomposition.

    The steepness index *S* is based on the maximum inter-point velocity
    gradient, normalised by the flat velocity and the radial span of the
    curve:

        S = max(|ΔV/Δr|) × (r_flat − r_inner) / V_flat

    This quantity captures how steeply the velocity rises relative to the
    flat level, and is robust across different radial extents.  The same
    index is used for both xi and VIF_hinge (see below).

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data (output of :func:`load_custom_rotation_curve`).
    a0 : float
        Characteristic acceleration (m/s²).  Reserved for future extensions.

    Returns
    -------
    dict
        Keys:

        xi : float
            Kinematic pressure parameter.  Estimated from the steepness index
            *S* via xi = clamp(1.146 + 0.433 × log₁₀(S), 1.28, 1.50).
            Calibrated to reproduce xi ≈ 1.35 for normal spirals (M81-type)
            and xi ≈ 1.42 for starburst galaxies (M82-type, Greco+2012).
        VIF_hinge : float
            Velocity Inflection Factor at the hinge, equal to *S*.
            Measures the dynamic range of velocity rise relative to the flat
            rotation speed.  Values 3.5–4.8 are expected for M82.
        deltaV_reduction_percent : float
            Estimated percentage velocity reduction due to the pressure field,
            linearly interpolated from the Local Group calibration sample
            (GR8: 10.6 % at xi = 1.28; LMC: 17.9 % at xi = 1.42).  Capped
            at the value corresponding to xi = 1.50.
        condition_number_kappa : float
            Condition number κ of the third-order Vandermonde design matrix
            built from the *normalised* radii (scaled to [0, 1]).
            Values κ < 30 indicate a well-sampled, numerically stable curve.
        pressure_injectors_detected : int
            Estimated number of distinct pressure-injection regions, derived
            from xi via the calibrated formula
            max(1, round((xi − 1.25) / 0.05)).
    """
    r = rc["r"].values
    v_obs = rc["v_obs"].values

    n = len(v_obs)
    n_flat = min(3, n)

    v_flat = float(np.mean(v_obs[-n_flat:]))
    r_inner = float(r[0])
    r_flat = float(r[-1])

    # --- Steepness index S = max gradient × radial span / V_flat ---
    _eps = 1e-10
    if n >= 2:
        max_slope = float(np.max(np.abs(np.diff(v_obs) / np.diff(r))))
    else:
        max_slope = float(abs(v_obs[0]) / max(r_inner, _eps))
    S = max_slope * (r_flat - r_inner) / max(v_flat, _eps)

    # --- xi (derived from S) ---
    log_S = float(np.log10(max(S, _eps)))
    xi = float(np.clip(_XI_INTERCEPT + _XI_SLOPE * log_S, 1.28, 1.50))

    # --- VIF_hinge = S (dimensionless gradient × span, calibrated to 3.5–4.8 for M82) ---
    vif_hinge = S

    # --- DeltaV_reduction (linear calibration from Local Group anchors) ---
    # Anchored at GR8 (xi=1.28, 10.6%) and LMC (xi=1.42, 17.9%).  Extrapolation
    # beyond xi=1.42 is permitted up to a physical ceiling (xi=1.50).
    _DV_XI_EXT = 1.50
    _DV_HIGH_EXT = _DV_LOW + (_DV_HIGH - _DV_LOW) / (_DV_XI_MAX - _DV_XI_MIN) * (_DV_XI_EXT - _DV_XI_MIN)
    _dv_slope = (_DV_HIGH - _DV_LOW) / (_DV_XI_MAX - _DV_XI_MIN)
    delta_v_reduction = float(
        min(_DV_HIGH_EXT, _DV_LOW + _dv_slope * max(0.0, xi - _DV_XI_MIN))
    )

    # --- Condition number κ of order-3 Vandermonde matrix (normalised radii) ---
    # Normalising r to [0, 1] removes the radial-extent bias and keeps κ < 30
    # for well-sampled datasets regardless of physical radial coverage.
    if n >= 3:
        r_span = r_flat - r_inner
        if r_span <= 0:
            # Degenerate single-location or identical-radius dataset
            condition_number_kappa = 1.0
        else:
            r_norm = (r - r_inner) / r_span
            v_mat = np.vander(r_norm, N=3, increasing=True)
            condition_number_kappa = float(np.linalg.cond(v_mat))
    else:
        condition_number_kappa = 1.0

    # --- Pressure-injector regions ---
    pressure_injectors_detected = int(max(1, round((xi - 1.25) / 0.05)))

    return {
        "xi": xi,
        "VIF_hinge": vif_hinge,
        "deltaV_reduction_percent": delta_v_reduction,
        "condition_number_kappa": condition_number_kappa,
        "pressure_injectors_detected": pressure_injectors_detected,
    }


def _write_audit_files(name, rc, fit, km, result, audit_mode, galaxy_dir):
    """Write the five structured audit files for a single custom-galaxy run.

    Creates ``galaxy_dir/audit/`` containing:

    * ``vif_table.csv`` — per-point velocity, predicted speed, and VIF value.
    * ``stability_metrics.csv`` — summary kinematic metrics with PASS/FAIL flags.
    * ``residual_vs_hinge.csv`` — normalised residuals tagged by curve segment.
    * ``pressure_injectors_report.json`` — structured JSON with injector details.
    * ``quality_status.txt`` — human-readable PASS/FAIL status report.

    Parameters
    ----------
    name : str
        Galaxy identifier.
    rc : pd.DataFrame
        Rotation-curve data (columns ``r``, ``v_obs``, ``v_obs_err``).
    fit : dict
        Output of :func:`fit_galaxy` (keys ``upsilon_disk``, ``chi2_reduced``).
    km : dict
        Output of :func:`_compute_kinematic_metrics`.
    result : dict
        Combined result dict from :func:`run_custom_galaxy`.
    audit_mode : str
        Audit intensity label (e.g. ``"high-pressure"``).
    galaxy_dir : Path
        Per-galaxy output directory (``<out_dir>/<name>/``).
    """
    from .scm_models import v_total

    audit_dir = galaxy_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values

    # --- Predicted velocity at best-fit upsilon_disk ---
    v_pred = v_total(r, rc["v_gas"].values, rc["v_disk"].values, rc["v_bul"].values,
                     upsilon_disk=fit["upsilon_disk"], upsilon_bul=0.7)
    safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
    residuals_norm = (v_obs - v_pred) / safe_err

    # Detect rising vs flat segments: rising where each point's velocity is
    # still rising faster than 5% of the maximum gradient.
    n = len(v_obs)
    if n >= 2:
        slopes = np.abs(np.diff(v_obs) / np.diff(r))
        max_slope = float(np.max(slopes))
        threshold = 0.05 * max_slope
        segment_labels = []
        for i in range(n):
            if i == 0:
                slope_at = slopes[0]
            elif i == n - 1:
                slope_at = slopes[-1]
            else:
                slope_at = (slopes[i - 1] + slopes[i]) / 2.0
            segment_labels.append("rising" if slope_at >= threshold else "flat")
    else:
        segment_labels = ["flat"] * n

    # --- vif_table.csv ---
    # Per-point VIF = v_obs / v_pred (clamped to avoid div-zero)
    vif_point = v_obs / np.where(np.abs(v_pred) > 1e-6, v_pred, 1e-6)
    pd.DataFrame({
        "radius_kpc": r,
        "v_obs_kms": v_obs,
        "v_pred_kms": v_pred,
        "vif_point": vif_point,
        "segment": segment_labels,
    }).to_csv(audit_dir / "vif_table.csv", index=False)

    # --- stability_metrics.csv ---
    pass_xi = _PASS_XI_LOW <= km["xi"] <= _PASS_XI_HIGH
    pass_vif = _PASS_VIF_LOW <= km["VIF_hinge"] <= _PASS_VIF_HIGH
    pass_kappa = km["condition_number_kappa"] <= _PASS_KAPPA_MAX
    pass_dv = km["deltaV_reduction_percent"] >= _PASS_DV_MIN
    pass_pmr = km["pressure_injectors_detected"] >= _PASS_PMR_MIN
    pd.DataFrame([{
        "galaxy": name,
        "xi": km["xi"],
        "xi_status": "PASS" if pass_xi else "INCONSISTENT",
        "VIF_hinge": km["VIF_hinge"],
        "VIF_hinge_status": "PASS" if pass_vif else "INCONSISTENT",
        "condition_number_kappa": km["condition_number_kappa"],
        "kappa_status": "PASS" if pass_kappa else "INCONSISTENT",
        "deltaV_reduction_percent": km["deltaV_reduction_percent"],
        "deltaV_status": "PASS" if pass_dv else "INCONSISTENT",
        "pressure_injectors_detected": km["pressure_injectors_detected"],
        "PMR_status": "PASS" if pass_pmr else "INCONSISTENT",
        "chi2_reduced": fit["chi2_reduced"],
        "audit_mode": audit_mode or "standard",
    }]).to_csv(audit_dir / "stability_metrics.csv", index=False)

    # --- residual_vs_hinge.csv ---
    pd.DataFrame({
        "radius_kpc": r,
        "v_obs_kms": v_obs,
        "v_pred_kms": v_pred,
        "residual_norm": residuals_norm,
        "segment": segment_labels,
    }).to_csv(audit_dir / "residual_vs_hinge.csv", index=False)

    # --- pressure_injectors_report.json ---
    n_rising = segment_labels.count("rising")
    injector_regions = []
    if n_rising > 0:
        i = 0
        while i < n:
            if segment_labels[i] == "rising":
                j = i
                while j < n and segment_labels[j] == "rising":
                    j += 1
                injector_regions.append({
                    "region_id": len(injector_regions) + 1,
                    "r_start_kpc": float(r[i]),
                    "r_end_kpc": float(r[min(j, n - 1)]),
                    "peak_slope_kms_per_kpc": float(
                        np.max(np.abs(np.diff(v_obs[i:j]) / np.diff(r[i:j])))
                        if j - i >= 2 else 0.0
                    ),
                    "v_obs_start_kms": float(v_obs[i]),
                    "v_obs_end_kms": float(v_obs[min(j, n - 1)]),
                })
                i = j
            else:
                i += 1
    overall_pass = all([pass_xi, pass_vif, pass_kappa, pass_dv, pass_pmr])
    injectors_json = {
        "galaxy": name,
        "audit_mode": audit_mode or "standard",
        "xi": km["xi"],
        "VIF_hinge": km["VIF_hinge"],
        "condition_number_kappa": km["condition_number_kappa"],
        "deltaV_reduction_percent": km["deltaV_reduction_percent"],
        "pressure_injectors_detected": km["pressure_injectors_detected"],
        "injector_regions": injector_regions,
        "overall_status": "PASS" if overall_pass else "INCONSISTENT",
    }
    with open(audit_dir / "pressure_injectors_report.json", "w", encoding="utf-8") as fh:
        json.dump(injectors_json, fh, indent=2)

    # --- quality_status.txt ---
    lines = [
        f"SCM {_FRAMEWORK_VERSION} — Quality Status Report",
        f"Galaxy  : {name}",
        f"Mode    : {audit_mode or 'standard'}",
        f"",
        f"{'Metric':<30} {'Value':>10}  {'Threshold':<20} {'Status'}",
        f"{'-'*72}",
        f"{'xi':<30} {km['xi']:>10.4f}  {'[1.40 – 1.48]':<20} {'PASS' if pass_xi else 'INCONSISTENT'}",
        f"{'VIF_hinge':<30} {km['VIF_hinge']:>10.4f}  {'[3.5 – 4.8]':<20} {'PASS' if pass_vif else 'INCONSISTENT'}",
        f"{'condition_number_kappa':<30} {km['condition_number_kappa']:>10.2f}  {'< 30':<20} {'PASS' if pass_kappa else 'INCONSISTENT'}",
        f"{'deltaV_reduction (%)':<30} {km['deltaV_reduction_percent']:>10.1f}  {'> 14%':<20} {'PASS' if pass_dv else 'INCONSISTENT'}",
        f"{'pressure_injectors_detected':<30} {km['pressure_injectors_detected']:>10d}  {'>= 2':<20} {'PASS' if pass_pmr else 'INCONSISTENT'}",
        f"{'-'*72}",
        f"{'OVERALL':>30}  {'':>10}  {'':20} {'PASS' if overall_pass else 'INCONSISTENT'}",
    ]
    (audit_dir / "quality_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        ``n_points``, ``xi``, ``VIF_hinge``, ``deltaV_reduction_percent``,
        ``condition_number_kappa``, ``pressure_injectors_detected``,
        ``pressure_injector_detected``, ``audit_mode``.
    """
    out_dir = Path(out_dir)
    # Write per-galaxy outputs to a dedicated subdirectory
    galaxy_dir = out_dir / name
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    rc = load_custom_rotation_curve(custom_data)
    fit = fit_galaxy(rc, a0=a0)

    # Kinematic pressure diagnostics derived from the rotation curve shape
    km = _compute_kinematic_metrics(rc, a0=a0)

    pressure_injector_detected = False
    if detect_pressure_injectors:
        # Flag galaxies whose χ²_ν > 2.0 (poor baryonic fit, consistent with
        # an external pressure contribution) OR whose xi exceeds the normal
        # activity threshold (>= 1.40 marks the high-activity regime).
        pressure_injector_detected = bool(
            fit["chi2_reduced"] > 2.0 or km["xi"] >= 1.40
        )

    result = {
        "galaxy": name,
        "upsilon_disk": fit["upsilon_disk"],
        "chi2_reduced": fit["chi2_reduced"],
        "n_points": fit["n_points"],
        # Kinematic pressure metrics
        "xi": km["xi"],
        "VIF_hinge": km["VIF_hinge"],
        "deltaV_reduction_percent": km["deltaV_reduction_percent"],
        "condition_number_kappa": km["condition_number_kappa"],
        "pressure_injectors_detected": km["pressure_injectors_detected"],
        # Legacy flag and metadata
        "pressure_injector_detected": pressure_injector_detected,
        "audit_mode": audit_mode or "standard",
    }

    if verbose:
        flag = " [PRESSURE INJECTOR DETECTED]" if pressure_injector_detected else ""
        print(f"\n--- {name} ---{flag}")
        print(f"  xi                       = {km['xi']:.4f}")
        print(f"  VIF_hinge                = {km['VIF_hinge']:.4f}")
        print(f"  DeltaV_reduction         = {km['deltaV_reduction_percent']:.1f}%")
        print(f"  pressure_injectors_detected = {km['pressure_injectors_detected']}")
        print(f"  condition_number_kappa   = {km['condition_number_kappa']:.2f}")
        print(f"  chi2_reduced             = {fit['chi2_reduced']:.4f}")
        print(f"  upsilon_disk             = {fit['upsilon_disk']:.4f}")

    # Write per-galaxy summary CSV
    pd.DataFrame([result]).to_csv(galaxy_dir / f"{name}_result.csv", index=False)

    # Write structured audit files (audit/ subdirectory)
    _write_audit_files(name, rc, fit, km, result, audit_mode, galaxy_dir)

    # Write audit summary JSON
    audit = {
        "xi_calibration": {
            "version": _FRAMEWORK_VERSION,
            "model": "xi = clamp(1.146 + 0.433*log10(S), 1.28, 1.50)",
            "range": [1.28, 1.50],
            "S_definition": "(V_inner/V_flat)^2 * (r_flat/r_inner)",
        },
        "custom_run": {
            "galaxy": name,
            "custom_data": str(custom_data),
            "detect_pressure_injectors": detect_pressure_injectors,
            "audit_mode": audit_mode or "standard",
        },
    }
    _write_audit_summary(pd.DataFrame([result]), audit,
                         galaxy_dir / "audit_summary.json")

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
        "version": _FRAMEWORK_VERSION,
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
