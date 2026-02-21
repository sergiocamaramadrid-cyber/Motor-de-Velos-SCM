"""Motor de Velos – Fluid Condensation Model (SCM).

Source package exposing model functions and analysis utilities.
"""

from .scm_models import (
    G_CONV,
    A0_DEFAULT,
    H0_DEFAULT,
    G_SI,
    M_SUN_KG,
    M_PER_KPC,
    nu_rar,
    g_obs_baseline,
    g_obs_universal,
    v_model_baseline,
    v_model_universal,
    gaussian_ll,
    aicc,
    chi2_gaussian,
    scm_model_accel,
    nll_gauss_accel,
    aicc_from_nll,
    rar_model_accel,
    nll_rar_accel,
    nfw_g_dm,
    nfw_model_accel,
    nll_nfw_accel,
)
from .fitting import fit_p0_local
from .io_utils import load_rotation_curve

__all__ = [
    "G_CONV",
    "A0_DEFAULT",
    "H0_DEFAULT",
    "G_SI",
    "M_SUN_KG",
    "M_PER_KPC",
    "nu_rar",
    "g_obs_baseline",
    "g_obs_universal",
    "v_model_baseline",
    "v_model_universal",
    "gaussian_ll",
    "aicc",
    "chi2_gaussian",
    "scm_model_accel",
    "nll_gauss_accel",
    "aicc_from_nll",
    "rar_model_accel",
    "nll_rar_accel",
    "nfw_g_dm",
    "nfw_model_accel",
    "nll_nfw_accel",
    "fit_p0_local",
    "load_rotation_curve",
]
