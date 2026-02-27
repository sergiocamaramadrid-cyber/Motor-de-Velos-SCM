"""Unit tests for src/galaxy_regime_classifier.py."""

import pytest

from src.galaxy_regime_classifier import classify_pressure_regime


class TestClassifyPressureRegime:
    def test_high_activity_exact_threshold(self):
        assert classify_pressure_regime(1.40) == "high_activity"

    def test_high_activity_above_threshold(self):
        assert classify_pressure_regime(1.42) == "high_activity"

    def test_normal_activity_exact_lower_bound(self):
        assert classify_pressure_regime(1.34) == "normal_activity"

    def test_normal_activity_default(self):
        # xi_default = 1.37 must be normal_activity (baseline)
        assert classify_pressure_regime(1.37) == "normal_activity"

    def test_normal_activity_upper_bound(self):
        assert classify_pressure_regime(1.39) == "normal_activity"

    def test_low_activity_just_below_normal(self):
        assert classify_pressure_regime(1.33) == "low_activity"

    def test_low_activity_well_below(self):
        assert classify_pressure_regime(1.30) == "low_activity"

    def test_example_lmc_is_high_activity(self):
        # LMC: ξ ≈ 1.41 → high_activity (special case, not baseline)
        assert classify_pressure_regime(1.41) == "high_activity"

    def test_example_m31_is_normal_activity(self):
        # M31: ξ ≈ 1.37 → normal_activity
        assert classify_pressure_regime(1.37) == "normal_activity"

    def test_example_ddo161_is_low_activity(self):
        # DDO 161: ξ ≈ 1.30 → low_activity
        assert classify_pressure_regime(1.30) == "low_activity"

    def test_return_type_is_str(self):
        for xi in (1.20, 1.35, 1.45):
            assert isinstance(classify_pressure_regime(xi), str)
