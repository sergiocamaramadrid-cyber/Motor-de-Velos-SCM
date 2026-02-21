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


def gaussian_ll(y, yhat, sigma=1.0) -> float:
    """Gaussian log-likelihood with normalisation term (to be *maximised*).

    ln L = −½ Σ [ (y − ŷ)² / σ² + ln(2π σ²) ]

    This is the canonical Gaussian log-likelihood **including the
    normalisation constant** ``ln(2π σ²)``.  All competing models (SCM,
    RAR, NFW) must use this identical function so that their log-likelihoods
    are directly comparable.

    Parameters
    ----------
    y : array_like
        Observed values.
    yhat : array_like
        Model-predicted values; must have the same shape as *y*.
    sigma : float or array_like, optional
        Measurement uncertainty (σ).  A scalar value applies the same σ to
        all data points; a vector applies per-point uncertainties.
        Defaults to 1.0 (unit variance).

    Returns
    -------
    ll : float
        Log-likelihood value (typically negative; larger values indicate a
        better fit).  Use ``-gaussian_ll(...)`` to obtain the NLL for
        minimisation.

    Notes
    -----
    The AICc is computed as ``AICc = -2·ll + 2k + 2k(k+1)/(n−k−1)``; this
    is equivalent to calling :func:`aicc_from_nll` with ``nll = -ll``.
    """
    y    = np.asarray(y,    dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if np.isscalar(sigma):
        s2 = float(sigma) ** 2
        return float(-0.5 * np.sum(((y - yhat) ** 2) / s2 + np.log(2.0 * np.pi * s2)))
    else:
        sigma = np.asarray(sigma, dtype=float)
        s2 = sigma ** 2
        return float(-0.5 * np.sum(((y - yhat) ** 2) / s2 + np.log(2.0 * np.pi * s2)))


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


# ---------------------------------------------------------------------------
# Acceleration-space SCM model (OOS / train-test framework)
# ---------------------------------------------------------------------------

def scm_model_accel(
    g_bar: np.ndarray,
    a0: float,
    beta: float,
) -> np.ndarray:
    """SCM acceleration correction term.

    Computes the non-Newtonian part of the observed centripetal acceleration
    as a function of the baryonic acceleration alone:

        scm(g_bar; a0, beta) = a0 · (√(1 + (g_bar/a0)^beta) − 1)

    The total predicted acceleration is then:
        g_pred = g_bar + scm(g_bar; a0, beta)

    Limits
    ------
    * Deep-MOND (g_bar ≪ a0, beta=1):  scm ≈ g_bar/2  (first-order Taylor)
    * Newtonian  (g_bar ≫ a0, beta=1):  scm ≈ √(g_bar·a0)  (≪ g_bar)

    Parameters
    ----------
    g_bar : array_like
        Baryonic centripetal acceleration [any consistent unit].
    a0 : float
        Motor-de-Velos / MOND characteristic acceleration scale.
    beta : float
        Shape exponent (= 1 recovers the standard MOND simple interpolation).

    Returns
    -------
    scm : ndarray
        SCM correction term in the same units as *g_bar*.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    ratio = np.clip(g_bar / a0, 0.0, None) ** beta
    return a0 * (np.sqrt(1.0 + ratio) - 1.0)


def nll_gauss_accel(
    params,
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g_err=None,
) -> float:
    """Full Gaussian negative log-likelihood for the acceleration-space SCM.

    The model prediction is:
        g_pred = g_bar + scm_model_accel(g_bar, a0, beta)

    The NLL is:
        NLL = 0.5 · Σ [(g_obs − g_pred)² / σ² + ln(2π σ²)]

    where σ is taken from *g_err* when provided, or set to 1 otherwise.
    Non-finite or non-positive σ values are replaced by 1.

    Parameters
    ----------
    params : (float, float)
        ``(a0, beta)`` — the two free parameters of the SCM model.
    g_bar : array_like
        Baryonic centripetal accelerations.
    g_obs : array_like
        Observed centripetal accelerations.
    g_err : array_like or None, optional
        Observational uncertainties on *g_obs*.  When ``None`` all σ = 1.

    Returns
    -------
    nll : float
        Negative log-likelihood value (to be minimised).  Returns ``1e100``
        for physically invalid parameters (a0 ≤ 0 or beta ≤ 0).
    """
    a0, beta = params
    if a0 <= 0.0 or beta <= 0.0:
        return 1e100
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    g_pred = g_bar + scm_model_accel(g_bar, a0, beta)
    resid = g_obs - g_pred
    if g_err is not None:
        sigma = np.asarray(g_err, dtype=float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    else:
        sigma = np.ones_like(resid)
    return float(0.5 * np.sum((resid / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2)))


def aicc_from_nll(nll: float, k: int, n: int) -> float:
    """AICc computed from a negative log-likelihood value.

    AICc = 2k + 2·NLL + 2k(k+1)/(n−k−1)

    The small-sample correction term is included whenever n > k + 1;
    otherwise only the standard AIC = 2k + 2·NLL is returned.

    Parameters
    ----------
    nll : float
        Negative log-likelihood of the model (to be minimised).
    k : int
        Number of free parameters.
    n : int
        Number of data points.

    Returns
    -------
    float
        AICc value.
    """
    aic = 2.0 * k + 2.0 * nll
    if n > k + 1:
        aic += (2.0 * k ** 2 + 2.0 * k) / (n - k - 1)
    return aic


# ---------------------------------------------------------------------------
# RAR model in acceleration space (k = 1)
# ---------------------------------------------------------------------------

#: SI gravitational constant [m³ kg⁻¹ s⁻²]
G_SI: float = 6.674e-11

#: Solar mass [kg]
M_SUN_KG: float = 1.989e30

#: Metres per kiloparsec
M_PER_KPC: float = 3.0857e19


def _sanitize_sigma(g_err) -> np.ndarray:
    """Return a safe σ array from *g_err*, replacing non-finite/non-positive
    values with 1.0.  Accepts ``None`` (returns scalar 1.0)."""
    if g_err is None:
        return 1.0
    sigma = np.asarray(g_err, dtype=float)
    return np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)


def rar_model_accel(
    g_bar: np.ndarray,
    g_dagger: float,
) -> np.ndarray:
    """McGaugh+2016 RAR predicted centripetal acceleration.

    g_obs = g_bar · ν(g_bar / g†)

    where the interpolation function is the same as :func:`nu_rar`:
        ν(x) = 1 / (1 − e^{−√x})

    Parameters
    ----------
    g_bar : array_like
        Baryonic centripetal acceleration [m s⁻²].
    g_dagger : float
        RAR characteristic acceleration scale [m s⁻²].  k = 1.

    Returns
    -------
    g_obs : ndarray
        Predicted centripetal acceleration [m s⁻²].
    """
    g_bar = np.asarray(g_bar, dtype=float)
    return g_bar * nu_rar(g_bar / g_dagger)


def nll_rar_accel(
    params,
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g_err=None,
) -> float:
    """NLL for the RAR model (k = 1).

    Uses :func:`gaussian_ll` internally so the log-likelihood is directly
    comparable with the SCM and NFW models.

    Parameters
    ----------
    params : (float,)
        ``(g_dagger,)`` — single free parameter of the RAR.
    g_bar : array_like
        Baryonic centripetal accelerations [m s⁻²].
    g_obs : array_like
        Observed centripetal accelerations [m s⁻²].
    g_err : array_like or None, optional
        Observational uncertainties.  ``None`` → σ = 1.

    Returns
    -------
    nll : float
        Negative log-likelihood (1e100 for g_dagger ≤ 0).
    """
    (g_dagger,) = params
    if g_dagger <= 0.0:
        return 1e100
    g_pred = rar_model_accel(g_bar, g_dagger)
    return float(-gaussian_ll(g_obs, g_pred, _sanitize_sigma(g_err)))


# ---------------------------------------------------------------------------
# NFW dark-matter model in acceleration space (k = 2)
# ---------------------------------------------------------------------------

def nfw_g_dm(
    r: np.ndarray,
    log10_rho_s: float,
    log10_r_s: float,
    G_grav: float = G_SI,
) -> np.ndarray:
    """NFW dark-matter centripetal acceleration at radius *r*.

    The NFW density profile is:
        ρ(r) = ρ_s / ((r/r_s) · (1 + r/r_s)²)

    The enclosed dark-matter mass is:
        M_NFW(<r) = 4π ρ_s r_s³ [ln(1 + r/r_s) − r/r_s / (1 + r/r_s)]

    The DM centripetal acceleration is:
        g_dm(r) = G · M_NFW(<r) / r²

    Parameters
    ----------
    r : array_like
        Galactocentric radii [m].
    log10_rho_s : float
        log₁₀ of the NFW characteristic density ρ_s [kg m⁻³].
    log10_r_s : float
        log₁₀ of the NFW scale radius r_s [m].
    G_grav : float, optional
        Gravitational constant [m³ kg⁻¹ s⁻²].  Defaults to :data:`G_SI`.

    Returns
    -------
    g_dm : ndarray
        Dark-matter centripetal acceleration [m s⁻²] at each radius.
    """
    r     = np.asarray(r,    dtype=float)
    rho_s = 10.0 ** log10_rho_s   # kg m^-3
    r_s   = 10.0 ** log10_r_s     # m
    x     = r / r_s
    # Enclosed NFW mass
    m_nfw = 4.0 * np.pi * rho_s * r_s ** 3 * (np.log1p(x) - x / (1.0 + x))
    m_nfw = np.maximum(m_nfw, 0.0)  # physical floor
    return G_grav * m_nfw / r ** 2


def nfw_model_accel(
    g_bar: np.ndarray,
    m_bar: np.ndarray,
    log10_rho_s: float,
    log10_r_s: float,
    G_grav: float = G_SI,
) -> np.ndarray:
    """NFW total centripetal acceleration prediction [m s⁻²].

    The effective galactocentric radius is derived from the enclosed
    baryonic mass and baryonic acceleration via g_bar = G·m_bar/r²:

        r_eff = √(G · m_bar / g_bar)

    The total predicted acceleration is then:

        g_obs = g_bar + g_dm(r_eff; ρ_s, r_s)

    Parameters
    ----------
    g_bar : array_like
        Baryonic centripetal accelerations [m s⁻²].
    m_bar : array_like
        Enclosed baryonic masses [kg].
    log10_rho_s : float
        log₁₀(ρ_s) [kg m⁻³] — NFW free parameter (k = 1).
    log10_r_s : float
        log₁₀(r_s) [m]      — NFW free parameter (k = 2).
    G_grav : float, optional
        Gravitational constant.

    Returns
    -------
    g_obs : ndarray
        Predicted centripetal acceleration [m s⁻²].
    """
    g_bar = np.asarray(g_bar, dtype=float)
    m_bar = np.asarray(m_bar, dtype=float)
    r_eff = np.sqrt(G_grav * m_bar / g_bar)
    return g_bar + nfw_g_dm(r_eff, log10_rho_s, log10_r_s, G_grav)


def nll_nfw_accel(
    params,
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    m_bar: np.ndarray,
    g_err=None,
    G_grav: float = G_SI,
) -> float:
    """NLL for the NFW dark-matter model (k = 2).

    Uses :func:`gaussian_ll` internally so the log-likelihood is directly
    comparable with the SCM and RAR models.

    Parameters
    ----------
    params : (float, float)
        ``(log10_rho_s, log10_r_s)``.
    g_bar : array_like
        Baryonic centripetal accelerations [m s⁻²].
    g_obs : array_like
        Observed centripetal accelerations [m s⁻²].
    m_bar : array_like
        Enclosed baryonic masses [kg].
    g_err : array_like or None, optional
        Observational uncertainties.
    G_grav : float, optional
        Gravitational constant.

    Returns
    -------
    nll : float
        Negative log-likelihood (1e100 for unphysical parameters).
    """
    log10_rho_s, log10_r_s = params
    g_pred = nfw_model_accel(g_bar, m_bar, log10_rho_s, log10_r_s, G_grav)
    if not np.all(np.isfinite(g_pred)) or np.any(g_pred <= 0):
        return 1e100
    return float(-gaussian_ll(g_obs, g_pred, _sanitize_sigma(g_err)))
