"""
scm_models.py
-------------
Definición de modelos SCM (Supply-Chain / causalidad) y funciones de ajuste
aplicados a curvas de rotación del catálogo SPARC/Iorio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar


# ---------------------------------------------------------------------------
# Modelo NFW (Navarro-Frenk-White)
# ---------------------------------------------------------------------------

def v_nfw(r: np.ndarray, v200: float, c: float, r200: float | None = None) -> np.ndarray:
    """Velocidad de rotación de un halo NFW.

    Parameters
    ----------
    r:    Radio galactocéntrico [kpc].
    v200: Velocidad circular a r200 [km/s].
    c:    Parámetro de concentración (adimensional).
    r200: Radio virial [kpc].  Si es *None* se usa el último punto de *r* como
          aproximación (válida sólo si los datos alcanzan r200).

    Returns
    -------
    np.ndarray  Velocidad de rotación [km/s].
    """
    r200_eff = r200 if r200 is not None else float(r[-1])
    if r200_eff == 0:
        raise ValueError("r200 no puede ser cero.")
    x = c * r / r200_eff
    f_nfw = np.log(1.0 + x) - x / (1.0 + x)
    f_c   = np.log(1.0 + c) - c / (1.0 + c)
    return v200 * np.sqrt(f_nfw / (f_c * (r / r200_eff)))


# ---------------------------------------------------------------------------
# Modelo isotérmico (Pseudo-Isothermal Halo – PISO)
# ---------------------------------------------------------------------------

def v_piso(r: np.ndarray, v_inf: float, r_c: float) -> np.ndarray:
    """Velocidad de rotación de un halo pseudo-isotérmico.

    Parameters
    ----------
    r:     Radio galactocéntrico [kpc].
    v_inf: Velocidad asintótica [km/s].
    r_c:   Radio de núcleo [kpc].

    Returns
    -------
    np.ndarray  Velocidad de rotación [km/s].
    """
    return v_inf * np.sqrt(1.0 - (r_c / r) * np.arctan(r / r_c))


# ---------------------------------------------------------------------------
# Velocidad barionica total
# ---------------------------------------------------------------------------

def v_baryon(
    vgas: np.ndarray,
    vdisk: np.ndarray,
    vbul: np.ndarray,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    """Contribución bariónica total a la curva de rotación.

    Parameters
    ----------
    vgas, vdisk, vbul: Contribuciones individuales [km/s].
    upsilon_disk, upsilon_bul: Relaciones masa-luminosidad del disco y bulbo.

    Returns
    -------
    np.ndarray  Velocidad bariónica total [km/s].
    """
    v2 = (
        np.sign(vgas)  * vgas**2
        + upsilon_disk * np.sign(vdisk) * vdisk**2
        + upsilon_bul  * np.sign(vbul)  * vbul**2
    )
    return np.sign(v2) * np.sqrt(np.abs(v2))


# ---------------------------------------------------------------------------
# Ajuste de modelos
# ---------------------------------------------------------------------------

def fit_piso(df: pd.DataFrame) -> dict:
    """Ajusta el modelo PISO a los datos de una galaxia.

    Parameters
    ----------
    df: DataFrame con columnas R, Vobs, errV (de read_iorio).

    Returns
    -------
    dict con claves: v_inf, r_c, chi2_red, success.
    """
    r    = df["R"].to_numpy()
    vobs = df["Vobs"].to_numpy()
    err  = df["errV"].to_numpy()

    try:
        popt, _ = curve_fit(
            v_piso,
            r, vobs,
            p0=[vobs.max(), r.mean()],
            sigma=err,
            absolute_sigma=True,
            bounds=([0.0, 1e-3], [1e4, 1e3]),
            maxfev=10_000,
        )
        v_model = v_piso(r, *popt)
        chi2 = np.sum(((vobs - v_model) / err) ** 2)
        dof  = max(len(vobs) - 2, 1)
        return {"v_inf": popt[0], "r_c": popt[1], "chi2_red": chi2 / dof, "success": True}
    except (RuntimeError, ValueError) as exc:
        return {"v_inf": np.nan, "r_c": np.nan, "chi2_red": np.nan, "success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Término universal SCM
# ---------------------------------------------------------------------------

def scm_universal_term(df: pd.DataFrame) -> float:
    """Calcula el término universal del Framework SCM.

    El término se define como la media cuadrática de (Vobs - V_baryon) / errV,
    proporcionando una medida de cuánta materia oscura requiere la curva
    de rotación más allá de la componente bariónica.

    Parameters
    ----------
    df: DataFrame con columnas R, Vobs, errV, Vgas, Vdisk, Vbul.

    Returns
    -------
    float  Término universal (adimensional).
    """
    vb = v_baryon(
        df["Vgas"].to_numpy(),
        df["Vdisk"].to_numpy(),
        df["Vbul"].to_numpy(),
    )
    residuals = (df["Vobs"].to_numpy() - vb) / df["errV"].to_numpy()
    return float(np.sqrt(np.mean(residuals**2)))


# ---------------------------------------------------------------------------
# RAR (Radial Acceleration Relation) helpers — SCM v0.2
# ---------------------------------------------------------------------------

#: Conversion factor: (km/s)² / kpc → m/s²
KPC_TO_MS2: float = 1e6 / 3.0857e19  # ≈ 3.241e-14

#: Floor value for y in rar_nu to avoid log/sqrt of zero (numerically safe).
_Y_FLOOR: float = 1e-30

#: Fraction-of-bound tolerance for declaring a fit has hit a boundary.
_BOUND_TOLERANCE: float = 0.01

#: Default g0 initial guess (m/s²), consistent with McGaugh et al. 2016
G0_DEFAULT: float = 1.2e-10


def g_from_v(v_kms: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    """Centripetal acceleration g = V² / R in m/s².

    Parameters
    ----------
    v_kms : array-like
        Circular velocity in km/s.
    r_kpc : array-like
        Galactocentric radius in kpc.

    Returns
    -------
    np.ndarray
        Centripetal acceleration in m/s².
    """
    return np.asarray(v_kms, dtype=float) ** 2 / np.asarray(r_kpc, dtype=float) * KPC_TO_MS2


def rar_nu(y: np.ndarray) -> np.ndarray:
    """McGaugh+2016 interpolating function ν(y) = 1 / (1 − exp(−√y)).

    Parameters
    ----------
    y : array-like
        Dimensionless ratio g_bar / g0.  Must be ≥ 0.

    Returns
    -------
    np.ndarray
        ν values (≥ 1).
    """
    safe_y = np.maximum(np.asarray(y, dtype=float), _Y_FLOOR)
    return 1.0 / (1.0 - np.exp(-np.sqrt(safe_y)))


def rar_g_obs(g_bar: np.ndarray, g0: float) -> np.ndarray:
    """RAR predicted g_obs = g_bar × ν(g_bar / g0).

    Parameters
    ----------
    g_bar : array-like
        Baryonic centripetal acceleration in m/s² (must be > 0).
    g0 : float
        Characteristic acceleration scale in m/s².

    Returns
    -------
    np.ndarray
        Predicted observed acceleration in m/s².
    """
    g_bar = np.asarray(g_bar, dtype=float)
    return g_bar * rar_nu(g_bar / g0)


def fit_g0_rar(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_bounds: tuple[float, float] = (1e-12, 1e-9),
) -> dict:
    """Fit the global RAR acceleration scale g0 by minimising log10 residuals.

    Uses bounded scalar optimisation over log10(g0) for numerical stability.

    Parameters
    ----------
    g_bar : array-like
        Baryonic centripetal accelerations in m/s² (must be > 0).
    g_obs : array-like
        Observed centripetal accelerations in m/s² (must be > 0).
    g0_bounds : tuple of float, optional
        Search bounds for g0 in m/s².

    Returns
    -------
    dict with keys:
        ``g0_hat``    — best-fit g0 (m/s²)
        ``rms_dex``   — RMS of log10 residuals at the best-fit g0
        ``n_pts``     — number of valid data points used
        ``at_bound``  — True if the fit converged at or very near a bound
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    mask = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar) & np.isfinite(g_obs)
    gb = g_bar[mask]
    go = g_obs[mask]
    log_go = np.log10(go)

    log_bounds = (np.log10(g0_bounds[0]), np.log10(g0_bounds[1]))

    def objective(log10_g0: float) -> float:
        g0 = 10.0 ** log10_g0
        pred = rar_g_obs(gb, g0)
        valid = pred > 0
        if valid.sum() < 3:
            return 1e10
        return float(np.sum((log_go[valid] - np.log10(pred[valid])) ** 2))

    result = minimize_scalar(objective, bounds=log_bounds, method="bounded")
    g0_hat = 10.0 ** result.x
    rms = float(np.sqrt(result.fun / max(mask.sum(), 1)))

    # Check if solution is within _BOUND_TOLERANCE of either search boundary
    rel_lo = abs(g0_hat - g0_bounds[0]) / g0_bounds[0]
    rel_hi = abs(g0_hat - g0_bounds[1]) / g0_bounds[1]
    at_bound = bool(rel_lo < _BOUND_TOLERANCE or rel_hi < _BOUND_TOLERANCE)

    return {
        "g0_hat": g0_hat,
        "rms_dex": rms,
        "n_pts": int(mask.sum()),
        "at_bound": at_bound,
    }
