"""
galaxy_regime_classifier.py — Automatic pressure-regime classification for
the Motor de Velos SCM framework (v0.6.1).

Three dynamic regimes are distinguished based on the empirically calibrated
pressure-injection parameter ξ (xi):

    Low activity   : ξ < 1.34   — stable, quiescent galaxies (e.g. DDO 161)
    Normal activity: 1.34 ≤ ξ < 1.40 — baseline (e.g. M31, M33, NGC 6822)
    High activity  : ξ ≥ 1.40   — strong pressure injection (e.g. LMC)

The global default remains xi_default = 1.37 (normal activity).
LMC (ξ ≈ 1.42) is a special high-activity case, not the baseline.
"""


def classify_pressure_regime(xi_value: float) -> str:
    """Classify a galaxy's dynamic pressure regime from its ξ (xi) value.

    Parameters
    ----------
    xi_value : float
        Empirically derived pressure-injection parameter ξ for a galaxy.

    Returns
    -------
    str
        One of ``"high_activity"``, ``"normal_activity"``, or
        ``"low_activity"``.
    """
    if xi_value >= 1.40:
        return "high_activity"
    elif xi_value >= 1.34:
        return "normal_activity"
    else:
        return "low_activity"
