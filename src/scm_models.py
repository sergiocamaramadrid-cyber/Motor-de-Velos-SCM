"""
scm_models.py
-------------
Definición de modelos SCM (Supply-Chain / causalidad) y funciones de ajuste
aplicados a curvas de rotación del catálogo SPARC/Iorio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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
