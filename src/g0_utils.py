import math
from typing import Tuple


def g0_touches_bounds(
    g0_hat: float,
    g0_bounds: Tuple[float, float],
    tol_log10: float = 0.05,
) -> Tuple[bool, bool]:
    """Check whether g0_hat is within tol_log10 log10-units of its bounds.

    Parameters
    ----------
    g0_hat:
        Estimated value of g0 (must be positive).
    g0_bounds:
        Tuple (lo, hi) with positive lower and upper bounds.
    tol_log10:
        Tolerance in log10 units. A bound is considered "touched" when
        the log10 distance between g0_hat and that bound is less than
        this value.

    Returns
    -------
    touch_lo, touch_hi : (bool, bool)
        touch_lo is True when g0_hat is within tol_log10 of the lower bound.
        touch_hi is True when g0_hat is within tol_log10 of the upper bound.
    """
    lo, hi = g0_bounds
    if lo <= 0 or hi <= 0 or g0_hat <= 0:
        raise ValueError(
            "g0_hat and both bounds must be strictly positive for log10 comparison; "
            f"got g0_hat={g0_hat}, lo={lo}, hi={hi}"
        )
    touch_lo = (math.log10(g0_hat) - math.log10(lo)) < tol_log10
    touch_hi = (math.log10(hi) - math.log10(g0_hat)) < tol_log10
    return touch_lo, touch_hi
