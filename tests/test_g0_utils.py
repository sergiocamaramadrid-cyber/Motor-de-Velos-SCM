import math
import pytest
from src.g0_utils import g0_touches_bounds


G0_BOUNDS = (1e-4, 1e4)


def test_touch_lower():
    # g0_hat very close to lower bound → touch_lo True
    touch_lo, touch_hi = g0_touches_bounds(1e-4, G0_BOUNDS)
    assert touch_lo is True
    assert touch_hi is False


def test_touch_upper():
    # g0_hat very close to upper bound → touch_hi True
    touch_lo, touch_hi = g0_touches_bounds(1e4, G0_BOUNDS)
    assert touch_lo is False
    assert touch_hi is True


def test_interior_no_touch():
    # g0_hat well inside bounds → neither bound touched
    touch_lo, touch_hi = g0_touches_bounds(1.0, G0_BOUNDS)
    assert touch_lo is False
    assert touch_hi is False


def test_both_touched_when_bounds_tight():
    # Very tight bounds where g0_hat is exactly the midpoint in log10 space
    # and tol_log10 is large enough to cover both sides
    bounds = (1.0, 10.0)  # log10 width = 1
    g0_hat = math.sqrt(10)  # midpoint in log10: log10 distance = 0.5 from each side
    touch_lo, touch_hi = g0_touches_bounds(g0_hat, bounds, tol_log10=0.6)
    assert touch_lo is True
    assert touch_hi is True


def test_custom_tol_log10():
    # With tight tolerance, a value 0.1 log10-units from lower bound is not touching
    # log10(1e-4 * 10**0.1) - log10(1e-4) = 0.1, which is NOT < 0.05
    lo = 1e-4
    g0_hat = lo * (10 ** 0.1)  # 0.1 log10-units above lo
    touch_lo, touch_hi = g0_touches_bounds(g0_hat, G0_BOUNDS, tol_log10=0.05)
    assert touch_lo is False


def test_default_tol_lo_touch():
    # Value 0.03 log10-units above lower bound should touch lo (0.03 < 0.05)
    lo = 1e-4
    g0_hat = lo * (10 ** 0.03)
    touch_lo, touch_hi = g0_touches_bounds(g0_hat, G0_BOUNDS)
    assert touch_lo is True
    assert touch_hi is False


def test_default_tol_hi_touch():
    # Value 0.03 log10-units below upper bound should touch hi
    hi = 1e4
    g0_hat = hi / (10 ** 0.03)
    touch_lo, touch_hi = g0_touches_bounds(g0_hat, G0_BOUNDS)
    assert touch_lo is False
    assert touch_hi is True


def test_returns_tuple_of_bools():
    result = g0_touches_bounds(1.0, G0_BOUNDS)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], bool)


def test_raises_on_non_positive_g0_hat():
    with pytest.raises(ValueError):
        g0_touches_bounds(0.0, G0_BOUNDS)
    with pytest.raises(ValueError):
        g0_touches_bounds(-1.0, G0_BOUNDS)


def test_raises_on_non_positive_bounds():
    with pytest.raises(ValueError):
        g0_touches_bounds(1.0, (0.0, 1e4))
    with pytest.raises(ValueError):
        g0_touches_bounds(1.0, (1e-4, 0.0))
