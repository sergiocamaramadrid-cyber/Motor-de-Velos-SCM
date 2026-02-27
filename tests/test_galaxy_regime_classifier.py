"""Unit tests for src/galaxy_regime_classifier.py."""

import pytest

from src.galaxy_regime_classifier import classify_pressure_regime


class TestClassifyPressureRegime:
    # --- high_activity ---

    def test_lmc_is_high_activity(self):
        assert classify_pressure_regime(1.42) == "high_activity"

    def test_exactly_at_high_threshold(self):
        assert classify_pressure_regime(1.40) == "high_activity"

    def test_above_high_threshold(self):
        assert classify_pressure_regime(1.50) == "high_activity"

    # --- normal_activity ---

    def test_default_xi_is_normal(self):
        # xi_default = 1.37
        assert classify_pressure_regime(1.37) == "normal_activity"

    def test_m31_is_normal(self):
        assert classify_pressure_regime(1.36) == "normal_activity"

    def test_lower_boundary_of_normal(self):
        assert classify_pressure_regime(1.34) == "normal_activity"

    def test_upper_boundary_of_normal(self):
        # 1.39 is still within normal (< 1.40)
        assert classify_pressure_regime(1.39) == "normal_activity"

    # --- low_activity ---

    def test_ddo161_is_low_activity(self):
        assert classify_pressure_regime(1.30) == "low_activity"

    def test_just_below_normal_boundary(self):
        # 1.3399... should be low_activity (< 1.34)
        assert classify_pressure_regime(1.3399) == "low_activity"

    def test_very_low_xi(self):
        assert classify_pressure_regime(0.50) == "low_activity"
