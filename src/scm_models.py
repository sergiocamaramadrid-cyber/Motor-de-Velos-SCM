"""
scm_models.py — Model definitions for the Motor de Velos SCM framework.

The Motor de Velos model proposes that galactic rotation is driven by a
universal pressure term (V_velos) in addition to the baryonic contributions.
The total rotation velocity is modelled as:

    V_total² = V_gas² + V_disk² + V_bul² + V_velos²

where V_velos represents the contribution from the universal veil pressure field.
"""

import numpy as np

# 1 kiloparsec in metres (IAU 2012)
KPC_TO_M = 3.085677581e16


def v_baryonic(r, v_gas, v_disk, v_bul, upsilon_disk=1.0, upsilon_bul=1.0):
    """Compute the total baryonic rotation velocity at radii *r*.

    Parameters
    ----------
    r : array_like
        Galactocentric radii (kpc).
    v_gas : array_like
        Gas component velocity (km/s), same length as *r*.
    v_disk : array_like
        Stellar-disk component velocity (km/s), scaled by sqrt(upsilon_disk).
    v_bul : array_like
        Bulge component velocity (km/s), scaled by sqrt(upsilon_bul).
    upsilon_disk : float
        Mass-to-light ratio for the disk (dimensionless).
    upsilon_bul : float
        Mass-to-light ratio for the bulge (dimensionless).

    Returns
    -------
    ndarray
        Baryonic rotation velocity in km/s.
    """
    r = np.asarray(r, dtype=float)
    v_gas = np.asarray(v_gas, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    v_bul = np.asarray(v_bul, dtype=float)

    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + upsilon_bul * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


def v_velos(r, a0=1.2e-10, G=4.302e-3):
    """Compute the universal veil pressure velocity contribution.

    This term encodes the acceleration imparted by the Motor de Velos
    pressure field.  The functional form mirrors the deep-MOND limit:

        V_velos² = sqrt(G * M_bar * a0)

    but here parameterised directly as a centripetal velocity offset that
    grows as sqrt(a0) at large radii.

    Parameters
    ----------
    r : array_like
        Galactocentric radii (kpc).
    a0 : float
        Characteristic acceleration of the veil pressure field (m/s²).
        Default: 1.2e-10 m/s² (empirically motivated).
    G : float
        Gravitational constant in units of (km/s)² pc / M_sun.
        Default: 4.302e-3 (pc M_sun⁻¹ (km/s)²).

    Returns
    -------
    ndarray
        Velos velocity contribution in km/s.
    """
    r = np.asarray(r, dtype=float)
    # Convert a0 from m/s² to km/s² per kpc
    a0_kpc = a0 * 1e-3 / KPC_TO_M  # km/s² per kpc
    return np.sqrt(np.maximum(a0_kpc * r, 0.0))


def v_total(r, v_gas, v_disk, v_bul,
            upsilon_disk=1.0, upsilon_bul=1.0,
            a0=1.2e-10, include_velos=True):
    """Compute the total predicted rotation velocity.

    Parameters
    ----------
    r : array_like
        Galactocentric radii (kpc).
    v_gas, v_disk, v_bul : array_like
        Component velocities in km/s.
    upsilon_disk : float
        Disk mass-to-light ratio.
    upsilon_bul : float
        Bulge mass-to-light ratio.
    a0 : float
        Velos characteristic acceleration (m/s²).
    include_velos : bool
        Whether to include the V_velos term (default True).

    Returns
    -------
    ndarray
        Total rotation velocity in km/s.
    """
    vb = v_baryonic(r, v_gas, v_disk, v_bul, upsilon_disk, upsilon_bul)
    vb2 = vb * np.abs(vb)
    if include_velos:
        vv = v_velos(r, a0=a0)
        vv2 = vv ** 2
    else:
        vv2 = 0.0
    v2 = vb2 + vv2
    return np.sign(v2) * np.sqrt(np.abs(v2))


def residuals(v_obs, v_obs_err, v_pred):
    """Compute normalised residuals between observed and predicted velocities.

    Parameters
    ----------
    v_obs : array_like
        Observed rotation velocities (km/s).
    v_obs_err : array_like
        Uncertainties on *v_obs* (km/s).
    v_pred : array_like
        Model-predicted rotation velocities (km/s).

    Returns
    -------
    ndarray
        Normalised residuals (v_obs - v_pred) / v_obs_err.
    """
    v_obs = np.asarray(v_obs, dtype=float)
    v_obs_err = np.asarray(v_obs_err, dtype=float)
    v_pred = np.asarray(v_pred, dtype=float)
    safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
    return (v_obs - v_pred) / safe_err


def chi2_reduced(v_obs, v_obs_err, v_pred, n_free=2):
    """Compute the reduced chi-squared statistic.

    Parameters
    ----------
    v_obs : array_like
        Observed velocities.
    v_obs_err : array_like
        Velocity uncertainties.
    v_pred : array_like
        Predicted velocities.
    n_free : int
        Number of free parameters in the fit.

    Returns
    -------
    float
        Reduced chi-squared value.
    """
    res = residuals(v_obs, v_obs_err, v_pred)
    n = len(np.asarray(v_obs))
    dof = max(n - n_free, 1)
    return float(np.sum(res ** 2) / dof)


def baryonic_tully_fisher(v_flat, a0=1.2e-10, G=4.302e-3):
    """Predict baryonic mass from the flat rotation velocity via the BTFR.

    M_bar = V_flat^4 / (G * a0)

    Parameters
    ----------
    v_flat : float or array_like
        Flat (asymptotic) rotation velocity in km/s.
    a0 : float
        Characteristic acceleration (m/s²).
    G : float
        Gravitational constant in (km/s)² pc / M_sun.

    Returns
    -------
    float or ndarray
        Predicted baryonic mass in solar masses.
    """
    v_flat = np.asarray(v_flat, dtype=float)
    # Convert a0 to (km/s)² / kpc
    a0_kms2_kpc = a0 * 1e-3 / KPC_TO_M
    # G in (km/s)² kpc / M_sun → 4.302e-3 * 1e-3 kpc / M_sun
    G_kpc = G * 1e-3  # (km/s)² kpc / M_sun
    return v_flat ** 4 / (G_kpc * a0_kms2_kpc)
