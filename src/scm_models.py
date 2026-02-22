"""
scm_models.py — Modelos SCM para la Relación de Aceleración Radial (RAR).

Implementa el motor de ajuste para la relación g_obs = g_bar * nu(g_bar / g0)
sobre el dataset SPARC, con diagnósticos de sesgo por bins en régimen profundo.

Versión v0.2b: bounds_log10_g0 ampliados a (-16.0, -8.0) para permitir que el
solver alcance el óptimo real (anterior lower bound en -12 era demasiado restrictivo).
"""

import numpy as np
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Bounds para el parámetro g0 en escala log10
# ---------------------------------------------------------------------------
# v0.1: bounds implícitamente restringidos ~ (-12, -8)
# v0.2: bounds_log10_g0 = (-12.0, -8.0)  — g0_hat tocaba el lower bound
# v0.2b: ampliamos el lower bound para liberar el óptimo real
BOUNDS_LOG10_G0 = (-16.0, -8.0)

# Tolerancia (en unidades de log10) para declarar que g0_hat toca el lower bound
LOWER_BOUND_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Función de transición ν(y)
# ---------------------------------------------------------------------------

def nu(y: np.ndarray) -> np.ndarray:
    """Función de transición estándar ν(y) = 1 / (1 - exp(-sqrt(y))).

    Relaciona la aceleración observada con la barionica:
        g_obs = g_bar * nu(g_bar / g0)

    Parameters
    ----------
    y : array-like
        Cociente adimensional y = g_bar / g0  (debe ser > 0).

    Returns
    -------
    np.ndarray
        Valores de ν(y) >= 1.
    """
    y = np.asarray(y, dtype=float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(y)))


def g_obs_model(g_bar: np.ndarray, g0: float) -> np.ndarray:
    """Predicción del modelo para g_obs dada g_bar y el parámetro de escala g0.

    Parameters
    ----------
    g_bar : array-like
        Aceleración bariónica [m/s^2].
    g0 : float
        Aceleración de escala característica [m/s^2].  g0 > 0.

    Returns
    -------
    np.ndarray
        Aceleración observada predicha por el modelo.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    y = g_bar / g0
    return g_bar * nu(y)


# ---------------------------------------------------------------------------
# Residuos en escala log10
# ---------------------------------------------------------------------------

def log10_residuals(g_obs: np.ndarray, g_bar: np.ndarray, g0: float) -> np.ndarray:
    """Residuos en log10: log10(g_obs) - log10(g_obs_model).

    Parameters
    ----------
    g_obs : array-like
        Aceleración observada [m/s^2].
    g_bar : array-like
        Aceleración bariónica [m/s^2].
    g0 : float
        Aceleración de escala [m/s^2].

    Returns
    -------
    np.ndarray
        Residuos log10-escala.
    """
    g_obs = np.asarray(g_obs, dtype=float)
    g_bar = np.asarray(g_bar, dtype=float)
    pred = g_obs_model(g_bar, g0)
    return np.log10(g_obs) - np.log10(pred)


# ---------------------------------------------------------------------------
# Función de coste y optimización
# ---------------------------------------------------------------------------

def _cost_rms(log10_g0: float, g_obs: np.ndarray, g_bar: np.ndarray) -> float:
    """RMS de residuos log10 para un valor dado de log10(g0)."""
    g0 = 10.0 ** log10_g0
    resid = log10_residuals(g_obs, g_bar, g0)
    return float(np.sqrt(np.mean(resid ** 2)))


def fit_g0(
    g_obs: np.ndarray,
    g_bar: np.ndarray,
    bounds_log10_g0: tuple[float, float] = BOUNDS_LOG10_G0,
) -> dict:
    """Ajusta g0 minimizando el RMS de residuos log10.

    Parameters
    ----------
    g_obs : array-like
        Aceleración observada [m/s^2].
    g_bar : array-like
        Aceleración bariónica [m/s^2].
    bounds_log10_g0 : tuple[float, float]
        Límites inferior y superior para log10(g0).
        Por defecto BOUNDS_LOG10_G0 = (-16.0, -8.0).

    Returns
    -------
    dict
        Resultado con claves:
        - ``g0_hat``        : valor óptimo de g0 [m/s^2]
        - ``log10_g0_hat``  : valor óptimo de log10(g0)
        - ``rms``           : RMS de residuos en el óptimo
        - ``lower_bound``   : límite inferior de g0 usado
        - ``upper_bound``   : límite superior de g0 usado
        - ``at_lower_bound``: True si g0_hat toca el lower bound (señal de alerta)
    """
    g_obs = np.asarray(g_obs, dtype=float)
    g_bar = np.asarray(g_bar, dtype=float)

    lo, hi = bounds_log10_g0
    result = minimize_scalar(
        _cost_rms,
        bounds=(lo, hi),
        method="bounded",
        args=(g_obs, g_bar),
    )

    log10_g0_hat = float(result.x)
    g0_hat = 10.0 ** log10_g0_hat
    lower_bound = 10.0 ** lo
    upper_bound = 10.0 ** hi
    at_lower_bound = abs(log10_g0_hat - lo) < LOWER_BOUND_TOLERANCE

    return {
        "g0_hat": g0_hat,
        "log10_g0_hat": log10_g0_hat,
        "rms": float(result.fun),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "at_lower_bound": at_lower_bound,
    }


# ---------------------------------------------------------------------------
# Diagnósticos por bins en régimen profundo
# ---------------------------------------------------------------------------

def bin_residuals(
    g_obs: np.ndarray,
    g_bar: np.ndarray,
    g0: float,
    n_bins: int = 10,
) -> list[dict]:
    """Calcula residuos medianos por bin de g_bar (régimen profundo a alto).

    Parameters
    ----------
    g_obs : array-like
        Aceleración observada [m/s^2].
    g_bar : array-like
        Aceleración bariónica [m/s^2].
    g0 : float
        Aceleración de escala [m/s^2].
    n_bins : int
        Número de bins logarítmicos.

    Returns
    -------
    list[dict]
        Lista de dicts con ``g_bar_center``, ``median_residual``, ``n_points``.
    """
    g_obs = np.asarray(g_obs, dtype=float)
    g_bar = np.asarray(g_bar, dtype=float)

    resid = log10_residuals(g_obs, g_bar, g0)
    log_gbar = np.log10(g_bar)
    bins = np.linspace(log_gbar.min(), log_gbar.max(), n_bins + 1)

    result = []
    for i in range(n_bins):
        mask = (log_gbar >= bins[i]) & (log_gbar < bins[i + 1])
        if mask.sum() == 0:
            continue
        center = 10.0 ** (0.5 * (bins[i] + bins[i + 1]))
        result.append(
            {
                "g_bar_center": float(center),
                "median_residual": float(np.median(resid[mask])),
                "n_points": int(mask.sum()),
            }
        )
    return result


def quantiles_g_bar(g_bar: np.ndarray) -> dict:
    """Cuantiles q10, q50, q90 de g_bar para diagnóstico de escala.

    Parameters
    ----------
    g_bar : array-like
        Aceleración bariónica [m/s^2].

    Returns
    -------
    dict
        Con claves ``q10``, ``q50``, ``q90``.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    return {
        "q10": float(np.percentile(g_bar, 10)),
        "q50": float(np.percentile(g_bar, 50)),
        "q90": float(np.percentile(g_bar, 90)),
    }


def print_fit_summary(fit_result: dict, bins: list[dict], g_bar: np.ndarray) -> None:
    """Imprime el resumen de ajuste al estilo SCM v0.2b.

    Parameters
    ----------
    fit_result : dict
        Resultado de :func:`fit_g0`.
    bins : list[dict]
        Salida de :func:`bin_residuals`.
    g_bar : array-like
        Aceleración bariónica [m/s^2] (para cuantiles).
    """
    q = quantiles_g_bar(g_bar)
    lb_flag = " ⚠️  AT LOWER BOUND" if fit_result["at_lower_bound"] else ""
    print(
        f"[SCM v0.2b] "
        f"g0_hat={fit_result['g0_hat']:.4e}  "
        f"log10_g0={fit_result['log10_g0_hat']:.3f}  "
        f"rms={fit_result['rms']:.4f}"
        f"{lb_flag}"
    )
    print(
        f"  lower_bound={fit_result['lower_bound']:.4e}  "
        f"upper_bound={fit_result['upper_bound']:.4e}"
    )
    print(
        f"  g_bar quantiles: q10={q['q10']:.2e}  q50={q['q50']:.2e}  q90={q['q90']:.2e}"
    )
    if bins:
        first = bins[0]
        print(
            f"  first bin: g_bar_center={first['g_bar_center']:.2e}  "
            f"median_residual={first['median_residual']:.3f}  "
            f"n={first['n_points']}"
        )
