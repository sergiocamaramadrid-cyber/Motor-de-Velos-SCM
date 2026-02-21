"""Motor de Velos – Fluid Condensation Model (SCM).

Source package exposing model functions and analysis utilities.
"""

from .scm_models import (
    G_CONV,
    A0_DEFAULT,
    H0_DEFAULT,
    nu_rar,
    g_obs_baseline,
    g_obs_universal,
    v_model_baseline,
    v_model_universal,
    aicc,
    chi2_gaussian,
    scm_model_accel,
    nll_gauss_accel,
    aicc_from_nll,
)
from .fitting import fit_p0_local
from .io_utils import load_rotation_curve

__all__ = [
    "G_CONV",
    "A0_DEFAULT",
    "H0_DEFAULT",
    "nu_rar",
    "g_obs_baseline",
    "g_obs_universal",
    "v_model_baseline",
    "v_model_universal",
    "aicc",
    "chi2_gaussian",
    "scm_model_accel",
    "nll_gauss_accel",
    "aicc_from_nll",
    "fit_p0_local",
    "load_rotation_curve",
]
