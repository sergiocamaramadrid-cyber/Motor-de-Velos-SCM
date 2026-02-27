"""Unit tests for src/galaxy_regime_classifier.py."""

import pytest

from src.galaxy_regime_classifier import classify_pressure_regime


class TestClassifyPressureRegime:
    def test_high_activity_exact_threshold(self):
        assert classify_pressure_regime(1.40) == "high_activity"

    def test_high_activity_above_threshold(self):
        assert classify_pressure_regime(1.42) == "high_activity"

    def test_high_activity_just_below_starburst(self):
        # 1.44 is still high_activity (starburst_extreme starts at 1.45)
        assert classify_pressure_regime(1.44) == "high_activity"

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
        # LMC: ξ ≈ 1.41 → high_activity
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

    # --- starburst_extreme (new in v0.6.1 — M82 confirmation) ---

    def test_starburst_extreme_exact_threshold(self):
        # xi = 1.45 is exactly the starburst_extreme boundary
        assert classify_pressure_regime(1.45) == "starburst_extreme"

    def test_starburst_extreme_m82_value(self):
        # M82: xi = 1.48 (highest observed, confirmed v0.6.1)
        assert classify_pressure_regime(1.48) == "starburst_extreme"

    def test_starburst_extreme_upper_clamp(self):
        # xi = 1.50 (formula upper clamp) → starburst_extreme
        assert classify_pressure_regime(1.50) == "starburst_extreme"

    def test_starburst_extreme_not_reached_at_1_44(self):
        # 1.44 is still high_activity (starburst_extreme boundary is 1.45)
        assert classify_pressure_regime(1.44) == "high_activity"
