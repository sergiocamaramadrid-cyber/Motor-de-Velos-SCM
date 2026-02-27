"""
galaxy_regime_classifier.py — Automatic pressure-regime classifier for the
Motor de Velos SCM framework.

Classifies a galaxy's dynamic pressure regime based on its measured ξ
(xi) value, using the empirically validated thresholds from v0.6.1:

    Regime          ξ range        Example galaxies
    low_activity    1.30–1.33      DDO 161
    normal_activity 1.34–1.39      M31, M33, NGC 6822   (baseline)
    high_activity   1.40–1.42      LMC

The global default remains xi_default = 1.37 (normal_activity baseline).
LMC-like high-activity cases are treated as special, not the baseline.
"""


def classify_pressure_regime(xi_value: float) -> str:
    """Classify a galaxy's dynamic pressure regime from its ξ value.

    Parameters
    ----------
    xi_value : float
        Measured ξ (xi) parameter for the galaxy.

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
