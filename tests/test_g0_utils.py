"""Tests for src/g0_utils.py — g0_touches_bounds helper."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.g0_utils import g0_touches_bounds


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_g0_touches_bounds_basic():
    bounds = (1e-16, 1e-8)

    lo, hi = g0_touches_bounds(1e-16, bounds, tol_log10=0.05)
    assert lo is True and hi is False

    lo, hi = g0_touches_bounds(1e-8, bounds, tol_log10=0.05)
    assert lo is False and hi is True


# ---------------------------------------------------------------------------
# Non-finite / non-positive g0_hat — safe mode (no raise)
# ---------------------------------------------------------------------------

def test_g0_touches_bounds_nonfinite_safe():
    bounds = (1e-16, 1e-8)
    lo, hi = g0_touches_bounds(float("nan"), bounds)
    assert (lo, hi) == (False, False)

    lo, hi = g0_touches_bounds(0.0, bounds)
    assert (lo, hi) == (False, False)


# ---------------------------------------------------------------------------
# Invalid bounds — fail-hard (ValueError)
# ---------------------------------------------------------------------------

def test_g0_touches_bounds_invalid_bounds():
    try:
        g0_touches_bounds(1e-12, (0.0, 1e-8))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Additional edge-case coverage
# ---------------------------------------------------------------------------

def test_no_touch_in_interior():
    """g0_hat far from both bounds returns (False, False)."""
    lo, hi = 1e-16, 1e-8
    mid = 1e-12  # log10 midpoint of (-16, -8)
    tl, tu = g0_touches_bounds(mid, (lo, hi))
    assert tl is False
    assert tu is False


def test_nonfinite_inf_returns_false_false():
    bounds = (1e-16, 1e-8)
    assert g0_touches_bounds(float("inf"), bounds) == (False, False)
    assert g0_touches_bounds(float("-inf"), bounds) == (False, False)


def test_negative_g0_hat_returns_false_false():
    bounds = (1e-16, 1e-8)
    assert g0_touches_bounds(-1e-10, bounds) == (False, False)


def test_invalid_bounds_lo_negative_raises():
    with pytest.raises(ValueError, match="Invalid bounds"):
        g0_touches_bounds(1e-12, (-1e-16, 1e-8))


def test_invalid_bounds_lo_ge_hi_raises():
    with pytest.raises(ValueError, match="Invalid bounds"):
        g0_touches_bounds(1e-12, (1e-8, 1e-16))


def test_invalid_bounds_nonfinite_raises():
    with pytest.raises(ValueError, match="Invalid bounds"):
        g0_touches_bounds(1e-12, (float("nan"), 1e-8))
    with pytest.raises(ValueError, match="Invalid bounds"):
        g0_touches_bounds(1e-12, (1e-16, float("inf")))


def test_custom_tol_wider():
    """A wider tol_log10 catches a value that narrow tol misses."""
    lo, hi = 1e-16, 1e-8
    # 0.1 decades above lo = 1e-16 → 10^(-16+0.1) ≈ 1.259e-16
    g0 = 10 ** (-16 + 0.1)
    tl_narrow, _ = g0_touches_bounds(g0, (lo, hi), tol_log10=0.05)
    tl_wide, _ = g0_touches_bounds(g0, (lo, hi), tol_log10=0.15)
    assert tl_narrow is False
    assert tl_wide is True


def test_returns_bools():
    """Return values are strict Python bools, not numpy bools."""
    tl, tu = g0_touches_bounds(1e-12, (1e-16, 1e-8))
    assert type(tl) is bool
    assert type(tu) is bool
