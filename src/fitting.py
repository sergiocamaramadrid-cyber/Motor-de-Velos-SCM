"""Local (per-galaxy) fitting of Baseline-SCM and Universal-SCM models.

The core public function is :func:`fit_p0_local`.

Both models have k = 2 free parameters so the AICc penalty terms cancel
and ΔAICc = χ²(universal) − χ²(baseline).  A negative ΔAICc indicates
that the Universal-SCM is preferred over the Baseline-SCM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .scm_models import (
    A0_DEFAULT,
    aicc,
    chi2_gaussian,
    v_model_baseline,
    v_model_universal,
)
from .io_utils import galaxy_name_from_path, load_rotation_curve

#: Number of free parameters for each model (must be equal for fair comparison)
K_PARAMS: int = 2


# ---------------------------------------------------------------------------
# Internal cost functions
# ---------------------------------------------------------------------------

def _cost_baseline(
    params,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
) -> float:
    """χ² for Baseline-SCM; returns a large penalty for invalid parameters."""
    upsilon, log10_a0 = params
    if upsilon <= 0.0 or not np.isfinite(log10_a0):
        return 1e20
    v_mod = v_model_baseline(r, v_disk, v_gas, upsilon, log10_a0)
    if not np.all(np.isfinite(v_mod)):
        return 1e20
    return chi2_gaussian(v_obs, v_mod, v_err)


def _cost_universal(
    params,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    a0_fixed: float,
) -> float:
    """χ² for Universal-SCM; returns a large penalty for invalid parameters."""
    upsilon, log10_v_ext = params
    if upsilon <= 0.0 or not np.isfinite(log10_v_ext):
        return 1e20
    v_mod = v_model_universal(r, v_disk, v_gas, upsilon, log10_v_ext, a0_fixed)
    if not np.all(np.isfinite(v_mod)):
        return 1e20
    return chi2_gaussian(v_obs, v_mod, v_err)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_p0_local(
    csv_path: Union[str, Path],
    p0_baseline: Tuple[float, float] = (0.5, np.log10(A0_DEFAULT)),
    p0_universal: Tuple[float, float] = (0.5, 1.5),
    a0_fixed: float = A0_DEFAULT,
    nelder_options: dict | None = None,
) -> dict:
    """Fit Baseline-SCM and Universal-SCM models locally to one galaxy.

    Both models share k = 2 free parameters:

    * Baseline-SCM   : (Υ_disk, log₁₀ a₀)
    * Universal-SCM  : (Υ_disk, log₁₀ V_ext)   with a₀ fixed at *a0_fixed*

    The Universal-SCM adds a constant external velocity V_ext from the
    Motor-de-Velos cosmological fluid:
        V_total² = V_rar(Υ, a₀_fixed)² + V_ext²

    Parameters
    ----------
    csv_path : str or Path
        Path to the galaxy rotation-curve CSV file.
        Required columns: ``r``, ``Vobs``, ``eVobs``, ``Vdisk``, ``Vgas``.
    p0_baseline : (float, float), optional
        Initial parameters for Baseline-SCM: ``(Υ_disk, log₁₀ a₀)``.
        Defaults to ``(0.5, log₁₀(A0_DEFAULT))``.
    p0_universal : (float, float), optional
        Initial parameters for Universal-SCM: ``(Υ_disk, log₁₀ V_ext)``.
        Defaults to ``(0.5, 1.5)`` — i.e. V_ext ≈ 31.6 km s⁻¹.
    a0_fixed : float, optional
        Fixed Motor-de-Velos acceleration scale [(km s⁻¹)² kpc⁻¹] used by
        the Universal-SCM model.  Defaults to :data:`A0_DEFAULT`.
    nelder_options : dict, optional
        Extra keyword arguments forwarded to the Nelder-Mead optimiser.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        ``galaxy``
            Name derived from the CSV filename.
        ``n_points``
            Number of radial data points *n*.
        ``k_params``
            Number of free parameters per model *k* (= 2).
        ``chi2_baseline``
            Best-fit χ² for Baseline-SCM.
        ``chi2_universal``
            Best-fit χ² for Universal-SCM.
        ``aicc_baseline``
            AICc for Baseline-SCM.
        ``aicc_universal``
            AICc for Universal-SCM.
        ``delta_aicc``
            **ΔAICc = AICc(universal) − AICc(baseline).**
            Negative values indicate the Universal-SCM is preferred.
        ``params_baseline``
            Best-fit ``(Υ_disk, log₁₀ a₀)`` for Baseline-SCM.
        ``params_universal``
            Best-fit ``(Υ_disk, log₁₀ V_ext)`` for Universal-SCM.
        ``fit_success_baseline``
            Whether the Baseline-SCM optimiser converged.
        ``fit_success_universal``
            Whether the Universal-SCM optimiser converged.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If the CSV is missing required columns or has too few data points.
    """
    opts = {
        "method": "Nelder-Mead",
        "options": {"maxiter": 20_000, "xatol": 1e-7, "fatol": 1e-7},
    }
    if nelder_options:
        opts["options"].update(nelder_options)

    df = load_rotation_curve(csv_path)
    r = df["r"].values
    v_obs = df["Vobs"].values
    v_err = df["eVobs"].values
    v_disk = df["Vdisk"].values
    v_gas = df["Vgas"].values
    n = len(r)

    # --- Baseline-SCM fit ---
    res_bl = minimize(
        _cost_baseline,
        list(p0_baseline),
        args=(r, v_obs, v_err, v_disk, v_gas),
        **opts,
    )
    chi2_bl = res_bl.fun
    aicc_bl = aicc(chi2_bl, n, K_PARAMS)

    # --- Universal-SCM fit ---
    res_un = minimize(
        _cost_universal,
        list(p0_universal),
        args=(r, v_obs, v_err, v_disk, v_gas, a0_fixed),
        **opts,
    )
    chi2_un = res_un.fun
    aicc_un = aicc(chi2_un, n, K_PARAMS)

    delta = aicc_un - aicc_bl

    return {
        "galaxy": galaxy_name_from_path(csv_path),
        "n_points": n,
        "k_params": K_PARAMS,
        "chi2_baseline": chi2_bl,
        "chi2_universal": chi2_un,
        "aicc_baseline": aicc_bl,
        "aicc_universal": aicc_un,
        "delta_aicc": delta,
        "params_baseline": tuple(res_bl.x),
        "params_universal": tuple(res_un.x),
        "fit_success_baseline": bool(res_bl.success),
        "fit_success_universal": bool(res_un.success),
    }
