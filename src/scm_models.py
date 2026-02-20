"""SCM (Fluid Condensation Model / Motor de Velos) — model definitions.

Two competing models are implemented with equal free-parameter count k = 2:

Baseline SCM
    Parameters: (Upsilon_disk, log10_a0)
    Physics: RAR-like formula  g_obs = g_bary * nu(g_bary / a0)
    where nu(x) = 1 / (1 − exp(−sqrt(x)))  (McGaugh+ 2016 interpolation)
    Predicted circular velocity:
        V_baseline(r) = sqrt(r * g_bary * nu(g_bary/a0))

Universal SCM
    Parameters: (Upsilon_disk, log10_V_ext)
    Physics: same RAR formula with a0 fixed at A0_DEFAULT, plus a universal
    external velocity field V_ext from the Motor-de-Velos cosmological fluid:
        V_model(r)^2 = V_rar(r; Upsilon, a0_fixed)^2 + V_ext^2
    V_ext (km s⁻¹) is a constant velocity contribution arising from the
    universal background pressure.  Its cosmological origin is parameterized
    via H0 and a fixed environmental radius r_env (both fixed constants), so
    the free parameter is log10(V_ext) and the total free-parameter count
    remains k = 2.

Unit conventions used throughout this module
    Radii                 : kpc
    Velocities            : km s⁻¹
    Accelerations         : (km s⁻¹)² kpc⁻¹
    Masses                : M☉
    G_CONV = G in kpc (km s⁻¹)² M☉⁻¹  ≈ 4.302 × 10⁻⁶
    A0_DEFAULT ≈ 3703 (km s⁻¹)² kpc⁻¹  ≡ 1.2 × 10⁻¹⁰ m s⁻²
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (in internal units)
# ---------------------------------------------------------------------------

#: Gravitational constant G in kpc (km/s)^2 M_sun^-1
G_CONV: float = 4.302e-6

#: Default Motor-de-Velos / MOND acceleration scale in (km/s)^2 kpc^-1
#: Corresponds to a0 = 1.2e-10 m s^-2.
#:   1 (km/s)^2/kpc = 3.241e-14 m s^-2  =>  a0 = 1.2e-10 / 3.241e-14 ≈ 3703
A0_DEFAULT: float = 3703.0

#: Hubble constant in km s^-1 Mpc^-1 (used for universal-term normalisation)
H0_DEFAULT: float = 70.0


# ---------------------------------------------------------------------------
# Interpolation function
# ---------------------------------------------------------------------------

def nu_rar(x: np.ndarray) -> np.ndarray:
    """Motor-de-Velos / RAR interpolation function ν(x).

    ν(x) = 1 / (1 − exp(−√x))

    In the deep-MOND limit (x → 0):  ν ≈ 1/√x  → g_obs ≈ √(g_bary · a0)
    In the Newtonian limit (x → ∞): ν → 1      → g_obs ≈ g_bary

    Parameters
    ----------
    x : array_like
        Dimensionless ratio g_bary / a0.  Must be > 0.

    Returns
    -------
    nu : ndarray
        Interpolation factor ≥ 1.
    """
    x = np.asarray(x, dtype=float)
    sqrtx = np.sqrt(np.clip(x, 1e-30, None))
    return 1.0 / (1.0 - np.exp(-sqrtx))


# ---------------------------------------------------------------------------
# Internal RAR helper
# ---------------------------------------------------------------------------

def _v_rar(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    upsilon: float,
    a0: float,
) -> np.ndarray:
    """RAR circular velocity [km s⁻¹] (internal helper).

    V_rar = sqrt(r · g_bary · ν(g_bary / a0))
    """
    v_bary2 = upsilon * v_disk ** 2 + v_gas ** 2
    g_bary = v_bary2 / r
    g_obs = g_bary * nu_rar(g_bary / a0)
    return np.sqrt(np.abs(r * g_obs))


# ---------------------------------------------------------------------------
# Public model predictors
# ---------------------------------------------------------------------------

def g_obs_baseline(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    upsilon: float,
    log10_a0: float,
) -> np.ndarray:
    """Baseline-SCM observed centripetal acceleration [(km s⁻¹)² kpc⁻¹].

    g_obs = g_bary · ν(g_bary / a0)

    Parameters
    ----------
    r : array_like
        Galactocentric radii [kpc].
    v_disk : array_like
        Disk circular velocity at Υ_disk = 1 [km s⁻¹].
    v_gas : array_like
        Gas circular velocity (face value) [km s⁻¹].
    upsilon : float
        Stellar mass-to-light ratio (free parameter).
    log10_a0 : float
        log₁₀ of the SCM acceleration scale [(km s⁻¹)² kpc⁻¹].

    Returns
    -------
    g : ndarray
        Modelled centripetal acceleration [(km s⁻¹)² kpc⁻¹].
    """
    r = np.asarray(r, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    v_gas = np.asarray(v_gas, dtype=float)

    a0 = 10.0 ** log10_a0
    v_bary2 = upsilon * v_disk ** 2 + v_gas ** 2
    g_bary = v_bary2 / r
    return g_bary * nu_rar(g_bary / a0)


def g_obs_universal(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    upsilon: float,
    log10_v_ext: float,
    a0_fixed: float = A0_DEFAULT,
) -> np.ndarray:
    """Universal-SCM effective centripetal acceleration [(km s⁻¹)² kpc⁻¹].

    g_eff(r) = [V_rar^2 + V_ext^2] / r

    where V_ext = 10^log10_v_ext is the constant Motor-de-Velos external
    velocity field, and V_rar is the baseline RAR circular velocity.

    Parameters
    ----------
    r : array_like
        Galactocentric radii [kpc].
    v_disk : array_like
        Disk circular velocity at Υ_disk = 1 [km s⁻¹].
    v_gas : array_like
        Gas circular velocity [km s⁻¹].
    upsilon : float
        Stellar mass-to-light ratio (free parameter).
    log10_v_ext : float
        log₁₀ of the external velocity V_ext [km s⁻¹] (free parameter).
    a0_fixed : float, optional
        Fixed Motor-de-Velos acceleration scale [(km s⁻¹)² kpc⁻¹].

    Returns
    -------
    g : ndarray
        Effective centripetal acceleration [(km s⁻¹)² kpc⁻¹].
    """
    r = np.asarray(r, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    v_gas = np.asarray(v_gas, dtype=float)

    v_ext = 10.0 ** log10_v_ext
    v_rar = _v_rar(r, v_disk, v_gas, upsilon, a0_fixed)
    return (v_rar ** 2 + v_ext ** 2) / r


# ---------------------------------------------------------------------------
# Circular-velocity predictors
# ---------------------------------------------------------------------------

def v_model_baseline(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    upsilon: float,
    log10_a0: float,
) -> np.ndarray:
    """Baseline-SCM predicted circular velocity [km s⁻¹].

    V_model = sqrt(r · g_obs_baseline)
    """
    g = g_obs_baseline(r, v_disk, v_gas, upsilon, log10_a0)
    return np.sqrt(np.abs(r) * np.abs(g))


def v_model_universal(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    upsilon: float,
    log10_v_ext: float,
    a0_fixed: float = A0_DEFAULT,
) -> np.ndarray:
    """Universal-SCM predicted circular velocity [km s⁻¹].

    V_model = sqrt(V_rar^2 + V_ext^2)
    """
    r = np.asarray(r, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    v_gas = np.asarray(v_gas, dtype=float)

    v_ext = 10.0 ** log10_v_ext
    v_rar = _v_rar(r, v_disk, v_gas, upsilon, a0_fixed)
    return np.sqrt(v_rar ** 2 + v_ext ** 2)


# ---------------------------------------------------------------------------
# Information criteria
# ---------------------------------------------------------------------------

def chi2_gaussian(
    v_obs: np.ndarray,
    v_model: np.ndarray,
    v_err: np.ndarray,
) -> float:
    """Gaussian χ² = Σ [(V_obs − V_model) / σ]²."""
    return float(np.sum(((v_obs - v_model) / v_err) ** 2))


def aicc(chi2: float, n_data: int, k_params: int) -> float:
    """Corrected Akaike Information Criterion.

    AICc = 2k − 2 ln L + 2k(k+1)/(n−k−1)

    With Gaussian errors and σ known, −2 ln L ≡ χ², so:
    AICc = χ² + 2k + 2k(k+1)/(n−k−1)

    Parameters
    ----------
    chi2 : float
        Gaussian χ² of the model.
    n_data : int
        Number of data points n.
    k_params : int
        Number of free parameters k.

    Returns
    -------
    float
        AICc value.
    """
    aic = chi2 + 2.0 * k_params
    if n_data - k_params - 1 <= 0:
        raise ValueError(
            f"AICc correction is undefined for n={n_data}, k={k_params} "
            "(n − k − 1 ≤ 0). Need more data points."
        )
    correction = 2.0 * k_params * (k_params + 1) / (n_data - k_params - 1)
    return aic + correction
