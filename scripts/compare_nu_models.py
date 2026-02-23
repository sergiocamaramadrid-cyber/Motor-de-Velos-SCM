"""
scripts/compare_nu_models.py — Compare ν (nu) interpolation models.

Four models are evaluated:

  velos       — Motor de Velos baseline:
                    V_total² = V_bar² + V_velos²  (V_velos² = a0_kpc · r)
  simple      — RAR simple form [Famaey & McGaugh 2012]:
                    ν(x) = 1 / (1 − exp(−√x))
  standard    — RAR standard form [McGaugh 1999]:
                    ν(x) = ½ · (1 + √(1 + 4/x))
  exp_linear  — AQUAL exponential form:
                    ν(x) = 1 / √(1 − exp(−x))

where x = g_bar / a0 = (V_bar²/r) · CONV / a0
and   CONV = 1 km²/s²/kpc → m/s²  = 1e6 / KPC_TO_M.

For all ν models, V_total = √(ν(x)) · |V_bar|.

Deep-regime definition (Option A): a radial point is "deep" if
  g_bar / a0 < deep_threshold   (default 0.3, physical criterion)
The threshold can be tuned via --deep-threshold.

LSB detection (Option B): a galaxy is flagged as low-surface-brightness (LSB)
if the median stellar surface-brightness SBdisk column is below --lsb-threshold
(default 1.0 L_sun/pc²).  Separate statistics are reported for LSB galaxies.

Deep-slope diagnostic (Option C): for deep-regime points the script fits
  log10(g_obs) = slope × log10(g_bar) + intercept
The expected MOND value is slope = 0.5.  Deviations signal structural failure.

Deep-collapse flag: the deep-regime median normalised residual exceeds 2.0,
i.e., the model systematically fails in the sub-Newtonian regime.

Outputs written to --out:
  compare_nu_models.csv   — per-model statistics table
  compare_nu_models.log   — full run log (also printed to stdout)

Usage
-----
With raw rotmod files::

    python scripts/compare_nu_models.py \\
        --data-dir data/SPARC \\
        --out results/diagnostics/compare_nu_models_175

With custom deep threshold (Option A)::

    python scripts/compare_nu_models.py \\
        --data-dir data/SPARC \\
        --out results/diagnostics/compare_nu_models_175 \\
        --deep-threshold 0.3

With a pre-computed per-galaxy CSV (limited comparison)::

    python scripts/compare_nu_models.py \\
        --csv results/universal_term_comparison_full.csv \\
        --out results/diagnostics/compare_nu_models_175
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

KPC_TO_M = 3.085677581e16  # metres per kiloparsec (IAU 2012)
# Convert (km/s)²/kpc → m/s²
_CONV = 1e6 / KPC_TO_M  # ≈ 3.241e-14  m s⁻² / [(km/s)² kpc⁻¹]

# Fiducial characteristic acceleration (m/s²)
A0_DEFAULT = 1.2e-10

# Default deep-regime threshold: g_bar/a0 < DEEP_THRESHOLD_DEFAULT
# (Option A: physical criterion, not quantile-based)
DEEP_THRESHOLD_DEFAULT = 0.3

# Default LSB surface-brightness threshold (Option B)
# Galaxies with median SBdisk < LSB_THRESHOLD_DEFAULT L_sun/pc² are flagged LSB.
LSB_THRESHOLD_DEFAULT = 1.0  # L_sun/pc²

# ---------------------------------------------------------------------------
# ν interpolation functions  (all accept array_like x = g_bar/a0 ≥ 0)
# ---------------------------------------------------------------------------

def nu_simple(x: np.ndarray) -> np.ndarray:
    """ν(x) = 1 / (1 − exp(−√x))  [Famaey & McGaugh 2012 'simple']."""
    x = np.asarray(x, dtype=float)
    safe = np.maximum(x, 1e-10)
    return 1.0 / (1.0 - np.exp(-np.sqrt(safe)))


def nu_standard(x: np.ndarray) -> np.ndarray:
    """ν(x) = ½ · (1 + √(1 + 4/x))  [McGaugh 1999 'standard']."""
    x = np.asarray(x, dtype=float)
    safe = np.maximum(x, 1e-10)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / safe))


def nu_exp_linear(x: np.ndarray) -> np.ndarray:
    """ν(x) = 1 / √(1 − exp(−x))  [AQUAL exponential]."""
    x = np.asarray(x, dtype=float)
    safe = np.maximum(x, 1e-10)
    denom = np.maximum(1.0 - np.exp(-safe), 1e-20)
    return 1.0 / np.sqrt(denom)


# Registry: name → callable
NU_MODELS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "simple": nu_simple,
    "standard": nu_standard,
    "exp_linear": nu_exp_linear,
}

# ---------------------------------------------------------------------------
# Per-galaxy velocity predictors
# ---------------------------------------------------------------------------

def _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk: float,
                upsilon_bul: float = 0.7) -> np.ndarray:
    """Signed baryonic velocity accounting for mass-to-light ratios."""
    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + upsilon_bul * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


def v_pred_nu(r, v_gas, v_disk, v_bul, upsilon_disk: float,
              nu_func, a0: float = A0_DEFAULT) -> np.ndarray:
    """Predict V_total using a ν-model RAR interpolation.

    Parameters
    ----------
    r : array_like
        Radii in kpc.
    v_gas, v_disk, v_bul : array_like
        Component velocities in km/s.
    upsilon_disk : float
        Disk mass-to-light ratio.
    nu_func : callable
        ν(x) function, x = g_bar/a0.
    a0 : float
        Characteristic acceleration in m/s².

    Returns
    -------
    ndarray
        Predicted rotation velocity in km/s.
    """
    r = np.asarray(r, dtype=float)
    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)
    # Baryonic centripetal acceleration: g_bar = V_bar² / r [converted to m/s²]
    g_bar = vb ** 2 / np.maximum(r, 1e-10) * _CONV  # m/s²
    x = g_bar / a0
    nu = nu_func(x)
    v2 = nu * vb * np.abs(vb)  # preserves sign, |V_pred|² = ν·|V_bar|²
    return np.sign(v2) * np.sqrt(np.abs(v2))


def v_pred_velos(r, v_gas, v_disk, v_bul, upsilon_disk: float,
                 a0: float = A0_DEFAULT) -> np.ndarray:
    """Predict V_total using the Motor de Velos baseline model.

    V_total² = V_bar² + V_velos²    where V_velos² = (a0 · CONV⁻¹) · r  [kpc → km/s]

    Parameters
    ----------
    r : array_like
        Radii in kpc.
    v_gas, v_disk, v_bul : array_like
        Component velocities in km/s.
    upsilon_disk : float
        Disk mass-to-light ratio.
    a0 : float
        Characteristic acceleration in m/s².

    Returns
    -------
    ndarray
        Predicted rotation velocity in km/s.
    """
    r = np.asarray(r, dtype=float)
    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)
    vb2 = vb * np.abs(vb)
    # V_velos² = a0 [m/s²] / CONV [(m/s²) per (km/s)²/kpc] × r [kpc]
    a0_kpc = a0 / _CONV  # (km/s)² / kpc
    vv2 = a0_kpc * np.maximum(r, 0.0)
    v2 = vb2 + vv2
    return np.sign(v2) * np.sqrt(np.abs(v2))


# ---------------------------------------------------------------------------
# Log-likelihood helpers
# ---------------------------------------------------------------------------

def log_likelihood(v_obs: np.ndarray, v_obs_err: np.ndarray,
                   v_pred: np.ndarray) -> float:
    """Gaussian log-likelihood.

    LL = −½ Σ [(v_obs − v_pred)² / σ² + ln(2π σ²)]
    """
    safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
    residuals2 = (v_obs - v_pred) ** 2 / safe_err ** 2
    ll = -0.5 * np.sum(residuals2 + np.log(2 * np.pi * safe_err ** 2))
    return float(ll)


def aicc(ll: float, k: int, n: int) -> float:
    """AICc = −2·LL + 2k + 2k(k+1)/(n−k−1)."""
    aic = -2.0 * ll + 2.0 * k
    correction = 2.0 * k * (k + 1) / max(n - k - 1, 1)
    return aic + correction


# ---------------------------------------------------------------------------
# Per-galaxy fitting
# ---------------------------------------------------------------------------

def _fit_upsilon(rc: pd.DataFrame, pred_fn) -> tuple[float, float]:
    """Minimise χ² over upsilon_disk ∈ [0.1, 5.0] for the given predictor.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data with columns r, v_obs, v_obs_err, v_gas, v_disk, v_bul.
    pred_fn : callable(r, v_gas, v_disk, v_bul, upsilon_disk) → ndarray

    Returns
    -------
    upsilon_disk : float
        Best-fit mass-to-light ratio.
    ll : float
        Log-likelihood at the best-fit upsilon_disk.
    """
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    def objective(ud):
        vp = pred_fn(r, v_gas, v_disk, v_bul, ud)
        # Minimise negative log-likelihood
        return -log_likelihood(v_obs, v_obs_err, vp)

    result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
    best_ud = float(result.x)
    best_ll = -float(result.fun)
    return best_ud, best_ll


# ---------------------------------------------------------------------------
# Deep-regime diagnostics
# ---------------------------------------------------------------------------

def deep_regime_stats(rc: pd.DataFrame, upsilon_disk: float,
                      pred_fn, a0: float = A0_DEFAULT,
                      deep_threshold: float = DEEP_THRESHOLD_DEFAULT) -> dict:
    """Compute deep-regime fraction, residuals, rms_dex, and slope diagnostic.

    Option A: deep threshold is a physical criterion, not a quantile.
    Option C: slope of log10(g_obs) vs log10(g_bar) in the deep regime.

    A radial point is "deep" if g_bar / a0 < deep_threshold.
    Expected MOND slope in the deep regime = 0.5.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data.
    upsilon_disk : float
        Best-fit mass-to-light ratio.
    pred_fn : callable
        Velocity predictor (same signature as in _fit_upsilon).
    a0 : float
        Characteristic acceleration.
    deep_threshold : float
        Threshold x = g_bar/a0 below which a point is in the deep regime.
        Default 0.3 (Option A).

    Returns
    -------
    dict with keys:
        deep_frac     — fraction of radial points in the deep regime.
        deep_res      — median |normalised residual| in the deep regime (nan if none).
        shallow_res   — median |normalised residual| outside the deep regime.
        rms_dex       — RMS of log10(V_obs/V_pred) over all points.
        rms_dex_deep  — same, restricted to deep-regime points.
        deep_slope    — OLS slope log10(g_obs) vs log10(g_bar) in deep regime
                        (Option C; nan if < 2 deep points).
        deep_slope_err — standard error of the slope (nan if < 3 points).
    """
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)
    g_bar = vb ** 2 / np.maximum(r, 1e-10) * _CONV  # m/s²
    x = g_bar / a0
    deep_mask = x < deep_threshold
    deep_frac = float(np.mean(deep_mask))

    vp = pred_fn(r, v_gas, v_disk, v_bul, upsilon_disk)
    safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
    norm_res = np.abs((v_obs - vp) / safe_err)

    # rms_dex: RMS of log10(V_obs / V_pred) — standard RAR diagnostic
    # Floors prevent log(0) and division-by-zero for near-zero velocities.
    _VEL_FLOOR = 1e-10  # km/s — well below any physical velocity
    safe_vp = np.where(np.abs(vp) > 0, np.abs(vp), _VEL_FLOOR)
    log_ratio = np.log10(np.maximum(np.abs(v_obs), _VEL_FLOOR) / safe_vp)
    rms_dex = float(np.sqrt(np.mean(log_ratio ** 2)))
    rms_dex_deep = (float(np.sqrt(np.mean(log_ratio[deep_mask] ** 2)))
                    if deep_mask.any() else float("nan"))

    deep_res = float(np.median(norm_res[deep_mask])) if deep_mask.any() else float("nan")
    shallow_mask = ~deep_mask
    shallow_res = (float(np.median(norm_res[shallow_mask]))
                   if shallow_mask.any() else float("nan"))

    # Option C: slope of log10(g_obs) vs log10(g_bar) in deep regime
    # g_obs = V_obs² / r (converted to m/s²)
    deep_slope = float("nan")
    deep_slope_err = float("nan")
    if deep_mask.sum() >= 2:
        g_obs_deep = v_obs[deep_mask] ** 2 / np.maximum(r[deep_mask], 1e-10) * _CONV
        g_bar_deep = g_bar[deep_mask]
        # Only use points with positive g_obs and g_bar
        valid = (g_obs_deep > 0) & (g_bar_deep > 0)
        if valid.sum() >= 2:
            lg_obs = np.log10(g_obs_deep[valid])
            lg_bar = np.log10(g_bar_deep[valid])
            # OLS: slope = cov(x,y)/var(x)
            lg_bar_c = lg_bar - lg_bar.mean()
            slope = float(np.dot(lg_bar_c, lg_obs - lg_obs.mean()) /
                          np.dot(lg_bar_c, lg_bar_c))
            deep_slope = slope
            if valid.sum() >= 3:
                y_pred_ols = slope * lg_bar + (lg_obs.mean() - slope * lg_bar.mean())
                sse = float(np.sum((lg_obs - y_pred_ols) ** 2))
                ssx = float(np.dot(lg_bar_c, lg_bar_c))
                deep_slope_err = float(np.sqrt(sse / (valid.sum() - 2) / ssx))

    return {
        "deep_frac": deep_frac,
        "deep_res": deep_res,
        "shallow_res": shallow_res,
        "rms_dex": rms_dex,
        "rms_dex_deep": rms_dex_deep,
        "deep_slope": deep_slope,
        "deep_slope_err": deep_slope_err,
    }


# ---------------------------------------------------------------------------
# Full --data-dir analysis
# ---------------------------------------------------------------------------

def _load_galaxy_table(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for p in candidates:
        if p.exists():
            sep = "," if p.suffix == ".csv" else r"\s+"
            return pd.read_csv(p, sep=sep, comment="#")
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {data_dir}. "
        "Expected SPARC_Lelli2016c.csv or .mrt"
    )


def _load_rotation_curve(data_dir: Path, name: str) -> pd.DataFrame:
    candidates = [
        data_dir / f"{name}_rotmod.dat",
        data_dir / "raw" / f"{name}_rotmod.dat",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(
                p, sep=r"\s+", comment="#",
                names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
                       "SBdisk", "SBbul"],
            )
            cols = ["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]
            if "SBdisk" in df.columns:
                cols.append("SBdisk")
            return df[cols]
    raise FileNotFoundError(f"Rotation curve for {name} not found in {data_dir}")


def run_data_dir_comparison(
    data_dir: Path,
    out_dir: Path,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    lsb_threshold: float = LSB_THRESHOLD_DEFAULT,
    log_lines: list[str] | None = None,
) -> tuple[pd.DataFrame, str, Callable]:
    """Run the full per-rotmod ν model comparison.

    Parameters
    ----------
    data_dir : Path
        Directory with SPARC rotmod files.
    out_dir : Path
        Output directory for results.
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        g_bar/a0 threshold for deep regime (Option A, default 0.3).
    lsb_threshold : float
        Median SBdisk threshold (L_sun/pc²) for LSB classification (Option B).
    log_lines : list[str] | None
        If provided, log messages are appended here (in addition to stdout).

    Returns
    -------
    tuple of (DataFrame, winner str, log callable)
    """

    def _log(msg: str) -> None:
        print(msg)
        if log_lines is not None:
            log_lines.append(msg)

    galaxy_table = _load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    # Build predictors for each model
    _PredFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
    models: dict[str, _PredFn] = {
        "velos": lambda r, vg, vd, vb, ud: v_pred_velos(r, vg, vd, vb, ud, a0=a0),
    }
    for mname, nu_fn in NU_MODELS.items():
        # Closure capture: bind nu_fn and a0
        models[mname] = (
            lambda r, vg, vd, vb, ud, _fn=nu_fn:
            v_pred_nu(r, vg, vd, vb, ud, _fn, a0=a0)
        )

    # Accumulators: per model
    total_ll: dict[str, float] = {m: 0.0 for m in models}
    total_n: dict[str, int] = {m: 0 for m in models}
    total_k: dict[str, int] = {m: 0 for m in models}  # one free param per galaxy
    deep_fracs: dict[str, list[float]] = {m: [] for m in models}
    deep_res_all: dict[str, list[float]] = {m: [] for m in models}
    shallow_res_all: dict[str, list[float]] = {m: [] for m in models}
    rms_dex_all: dict[str, list[float]] = {m: [] for m in models}
    rms_dex_deep_all: dict[str, list[float]] = {m: [] for m in models}
    rms_dex_lsb_all: dict[str, list[float]] = {m: [] for m in models}
    deep_slope_all: dict[str, list[float]] = {m: [] for m in models}
    n_processed = 0
    n_lsb = 0

    for name in galaxy_names:
        try:
            rc = _load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            _log(f"  [skip] {name}: rotmod not found")
            continue

        n_pts = len(rc)
        n_processed += 1

        # Option B: LSB flag based on median stellar surface brightness
        is_lsb = False
        if "SBdisk" in rc.columns:
            sb_median = float(np.nanmedian(rc["SBdisk"].values))
            is_lsb = sb_median < lsb_threshold
        if is_lsb:
            n_lsb += 1

        for mname, pred_fn in models.items():
            ud, ll = _fit_upsilon(rc, pred_fn)
            total_ll[mname] += ll
            total_n[mname] += n_pts
            total_k[mname] += 1  # one upsilon_disk per galaxy
            ds = deep_regime_stats(rc, ud, pred_fn, a0=a0,
                                   deep_threshold=deep_threshold)
            deep_fracs[mname].append(ds["deep_frac"])
            if not np.isnan(ds["deep_res"]):
                deep_res_all[mname].append(ds["deep_res"])
            if not np.isnan(ds["shallow_res"]):
                shallow_res_all[mname].append(ds["shallow_res"])
            rms_dex_all[mname].append(ds["rms_dex"])
            if not np.isnan(ds["rms_dex_deep"]):
                rms_dex_deep_all[mname].append(ds["rms_dex_deep"])
            if is_lsb:
                rms_dex_lsb_all[mname].append(ds["rms_dex"])
            if not np.isnan(ds["deep_slope"]):
                deep_slope_all[mname].append(ds["deep_slope"])

    _log(f"\n  Galaxies processed: {n_processed}  |  LSB galaxies: {n_lsb}"
         f"  |  deep_threshold: {deep_threshold}")

    # Build comparison table
    records = []
    for mname in models:
        ll = total_ll[mname]
        n = total_n[mname]
        k = total_k[mname]
        aic_c = aicc(ll, k, n)
        med_deep_frac = float(np.median(deep_fracs[mname])) if deep_fracs[mname] else 0.0
        med_deep_res = (float(np.median(deep_res_all[mname]))
                        if deep_res_all[mname] else float("nan"))
        med_shallow_res = (float(np.median(shallow_res_all[mname]))
                           if shallow_res_all[mname] else float("nan"))
        med_rms_dex = (float(np.median(rms_dex_all[mname]))
                       if rms_dex_all[mname] else float("nan"))
        med_rms_dex_deep = (float(np.median(rms_dex_deep_all[mname]))
                            if rms_dex_deep_all[mname] else float("nan"))
        med_rms_dex_lsb = (float(np.median(rms_dex_lsb_all[mname]))
                           if rms_dex_lsb_all[mname] else float("nan"))
        med_deep_slope = (float(np.median(deep_slope_all[mname]))
                          if deep_slope_all[mname] else float("nan"))
        deep_regime = med_deep_frac > 0.0
        deep_collapse = (not np.isnan(med_deep_res)
                         and not np.isnan(med_shallow_res)
                         and med_shallow_res > 0
                         and med_deep_res > 2.0 * med_shallow_res)
        records.append({
            "model": mname,
            "LL": ll,
            "AICc": aic_c,
            "N_galaxies": n_processed,
            "N_lsb": n_lsb,
            "N_points": n,
            "k": k,
            "deep_threshold": deep_threshold,
            "rms_dex": med_rms_dex,
            "rms_dex_deep": med_rms_dex_deep,
            "rms_dex_lsb": med_rms_dex_lsb,
            "deep_slope": med_deep_slope,
            "deep_frac_median": med_deep_frac,
            "deep_res_median": med_deep_res,
            "shallow_res_median": med_shallow_res,
            "deep_regime": deep_regime,
            "deep_collapse": deep_collapse,
        })

    df = pd.DataFrame(records)
    best_aicc = df["AICc"].min()
    df["delta_AICc"] = df["AICc"] - best_aicc
    winner = df.loc[df["AICc"].idxmin(), "model"]

    # Write comparison CSV to out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "compare_nu_models.csv", index=False)

    return df, winner, _log


def run_csv_comparison(csv_path: Path, out_dir: Path,
                       log_lines: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Limited comparison using a pre-computed per-galaxy CSV.

    Since individual rotation curves are unavailable, this mode uses the
    chi2_reduced values from the CSV as a proxy for each model's fit quality.
    The ν models cannot be individually refit; they are assessed theoretically
    (asymptotic correctness) and the existing velos chi2 is reported.

    Returns
    -------
    pd.DataFrame
        Per-model comparison table.
    str
        Winner label.
    """

    def _log(msg: str) -> None:
        print(msg)
        if log_lines is not None:
            log_lines.append(msg)

    pg = pd.read_csv(csv_path)
    required = {"galaxy", "chi2_reduced", "n_points"}
    missing = required - set(pg.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    n_gal = len(pg)
    total_n = int(pg["n_points"].sum())
    # Free parameters = 1 upsilon_disk per galaxy
    k = n_gal

    # LL proxy from chi2_reduced: LL ≈ -0.5 * Σ chi2_reduced_i * dof_i
    # dof = n_points - 2 (2 free params per galaxy)
    dof_per = (pg["n_points"] - 2).clip(lower=1)
    ll_proxy = -0.5 * float((pg["chi2_reduced"] * dof_per).sum())
    aic_c_velos = aicc(ll_proxy, k, total_n)

    records = []

    # velos model (has measured chi2)
    records.append({
        "model": "velos",
        "LL": ll_proxy,
        "AICc": aic_c_velos,
        "N_galaxies": n_gal,
        "N_lsb": 0,
        "N_points": total_n,
        "k": k,
        "deep_threshold": DEEP_THRESHOLD_DEFAULT,
        "rms_dex": float("nan"),
        "rms_dex_deep": float("nan"),
        "rms_dex_lsb": float("nan"),
        "deep_slope": float("nan"),
        "deep_frac_median": float("nan"),
        "deep_res_median": float("nan"),
        "shallow_res_median": float("nan"),
        "deep_regime": True,
        "deep_collapse": False,
        "note": "measured chi2_reduced from CSV",
    })

    # For ν models: cannot refit without rotmods — mark as N/A
    for mname in NU_MODELS:
        records.append({
            "model": mname,
            "LL": float("nan"),
            "AICc": float("nan"),
            "N_galaxies": n_gal,
            "N_lsb": 0,
            "N_points": total_n,
            "k": k,
            "deep_threshold": DEEP_THRESHOLD_DEFAULT,
            "rms_dex": float("nan"),
            "rms_dex_deep": float("nan"),
            "rms_dex_lsb": float("nan"),
            "deep_slope": float("nan"),
            "deep_frac_median": float("nan"),
            "deep_res_median": float("nan"),
            "shallow_res_median": float("nan"),
            "deep_regime": True,
            "deep_collapse": False,
            "note": "requires --data-dir for refitting",
        })

    df = pd.DataFrame(records)
    df["delta_AICc"] = float("nan")
    df.loc[df["model"] == "velos", "delta_AICc"] = 0.0
    winner = "velos (only model with measured chi2; rerun with --data-dir)"
    _log(f"\n  NOTE: CSV mode — ν model refitting requires --data-dir.")
    _log(f"  Galaxies in CSV: {n_gal}  |  Total radial points: {total_n}")
    return df, winner


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_SEP = "=" * 72


def _format_report(df: pd.DataFrame, winner: str, a0: float,
                   mode: str) -> list[str]:
    """Return list of lines for the report."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — ν Model Comparison Report",
        _SEP,
        f"  Mode         : {mode}",
        f"  a0           : {a0:.2e} m/s²",
        f"  N_galaxies   : {int(df['N_galaxies'].iloc[0])}",
        f"  N_lsb        : {int(df['N_lsb'].iloc[0])}",
        f"  N_points     : {int(df['N_points'].iloc[0])}",
        f"  deep_thresh  : {df['deep_threshold'].iloc[0]:.2f} × a0",
        "",
        f"  {'Model':<12} {'LL':>14} {'AICc':>14} {'ΔAICc':>8} "
        f"{'rms_dex':>8} {'deep_slope':>11} {'deep_collapse':>14}",
        "  " + "-" * 87,
    ]
    for _, row in df.sort_values("delta_AICc").iterrows():
        ll_s = f"{row['LL']:.2f}" if not np.isnan(row["LL"]) else "N/A"
        aicc_s = f"{row['AICc']:.2f}" if not np.isnan(row["AICc"]) else "N/A"
        daicc_s = f"{row['delta_AICc']:.2f}" if not np.isnan(row["delta_AICc"]) else "N/A"
        rms_s = f"{row['rms_dex']:.4f}" if not np.isnan(row["rms_dex"]) else "N/A"
        slope_s = (f"{row['deep_slope']:.3f} (expected 0.5)"
                   if "deep_slope" in row.index and not np.isnan(row["deep_slope"])
                   else "N/A")
        dc = "YES (⚠)" if row["deep_collapse"] else "no"
        lines.append(
            f"  {row['model']:<12} {ll_s:>14} {aicc_s:>14} {daicc_s:>8} "
            f"{rms_s:>8} {slope_s:>11} {dc:>14}"
        )
    lines += [
        "  " + "-" * 87,
        f"  Winner: {winner}",
        _SEP,
    ]
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ν (MOND interpolation) models against SPARC data."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data-dir", metavar="DIR",
        help="Directory containing SPARC rotmod files and galaxy table.",
    )
    src.add_argument(
        "--csv", metavar="FILE",
        help="Pre-computed per-galaxy summary CSV (limited comparison).",
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(f"g_bar/a0 threshold for deep regime (Option A, default: "
              f"{DEEP_THRESHOLD_DEFAULT})."),
    )
    parser.add_argument(
        "--lsb-threshold", type=float, default=LSB_THRESHOLD_DEFAULT,
        dest="lsb_threshold",
        help=(f"Median SBdisk threshold for LSB classification in L_sun/pc² "
              f"(Option B, default: {LSB_THRESHOLD_DEFAULT})."),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    if args.data_dir:
        mode = f"data-dir ({args.data_dir})"
        df, winner, _log = run_data_dir_comparison(
            Path(args.data_dir), out_dir, a0=args.a0,
            deep_threshold=args.deep_threshold,
            lsb_threshold=args.lsb_threshold,
            log_lines=log_lines,
        )
    else:
        mode = f"csv ({args.csv})"
        df, winner = run_csv_comparison(
            Path(args.csv), out_dir, log_lines=log_lines
        )

    # Format and print report
    report_lines = _format_report(df, winner, args.a0, mode)
    for line in report_lines:
        print(line)
        log_lines.append(line)

    # Write CSV
    csv_out = out_dir / "compare_nu_models.csv"
    df.to_csv(csv_out, index=False)

    # Write log
    log_out = out_dir / "compare_nu_models.log"
    log_out.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"\n  Results written to {out_dir}")


if __name__ == "__main__":
    main()
