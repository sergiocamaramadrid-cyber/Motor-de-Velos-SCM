"""
tests/test_scm_fallback_modulation.py — Tests for scripts/scm_fallback_modulation.py.

Covers:
  1. Unit conversion correctness (PC_TO_CM, KMS_TO_CMS)
  2. Numerical output for the default parameters
  3. Numerical output for the M31-2014-DS1 reference scenario
  4. Monotonicity: higher energy → higher β_mod
  5. Monotonicity: larger radius → smaller β_mod (dominant via r³ and decay)
  6. Decay factor bounds (0 < decay ≤ 1)
  7. Modulation positivity
  8. β_mod ≥ BETA_MOND for all positive inputs
  9. Zero-decay limit (decay_scale_pc → ∞ ⇒ decay → 1)
  10. Input validation (non-positive parameters raise ValueError)
  11. CLI: default run exits 0 and prints expected fields
  12. CLI: custom parameters produce the correct β_mod
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent

from scripts.scm_fallback_modulation import (
    BETA_MOND,
    KMS_TO_CMS,
    PC_TO_CM,
    fallback_modulation,
    main,
)


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_pc_to_cm(self):
        # IAU 2012: 1 pc = 3.085677581e18 cm (exact by definition)
        assert abs(PC_TO_CM - 3.085677581e18) / 3.085677581e18 < 1e-10

    def test_kms_to_cms(self):
        assert KMS_TO_CMS == 1.0e5

    def test_beta_mond(self):
        assert BETA_MOND == 0.5


# ---------------------------------------------------------------------------
# 2. Default parameters
# ---------------------------------------------------------------------------

class TestDefaultParameters:
    def setup_method(self):
        self.beta_mod, self.mod, self.dec = fallback_modulation()

    def test_returns_tuple_of_three(self):
        result = fallback_modulation()
        assert len(result) == 3

    def test_modulation_positive(self):
        assert self.mod > 0.0

    def test_decay_in_0_1(self):
        assert 0.0 < self.dec <= 1.0

    def test_beta_mod_at_least_half(self):
        assert self.beta_mod >= BETA_MOND

    def test_decay_default(self):
        # r=5 pc, scale=10 pc → exp(-0.5) ≈ 0.6065
        expected = math.exp(-0.5)
        assert abs(self.dec - expected) < 1e-10

    def test_beta_mod_formula(self):
        beta, mod, dec = fallback_modulation(
            energy_erg=1e32, r_pc=5.0, v_kms=20.0,
            rho_gcm3=1e-21, decay_scale_pc=10.0,
        )
        r_cm = 5.0 * PC_TO_CM
        v_cms = 20.0 * KMS_TO_CMS
        expected_mod = 1e32 / (1e-21 * v_cms**2 * r_cm**3)
        expected_dec = math.exp(-0.5)
        expected_beta = 0.5 + expected_mod * expected_dec
        assert abs(beta - expected_beta) / abs(expected_beta) < 1e-12
        assert abs(mod - expected_mod) / abs(expected_mod) < 1e-12
        assert abs(dec - expected_dec) < 1e-12


# ---------------------------------------------------------------------------
# 3. M31-2014-DS1 reference scenario
# ---------------------------------------------------------------------------

class TestM31Reference:
    """Reference scenario: E=1e46 erg, r=0.1 pc, v=50 km/s, ρ=1e-24 g/cm³."""

    def setup_method(self):
        self.beta, self.mod, self.dec = fallback_modulation(
            energy_erg=1e46,
            r_pc=0.1,
            v_kms=50.0,
            rho_gcm3=1e-24,
            decay_scale_pc=10.0,
        )

    def test_modulation_large(self):
        # With CGS-consistent units the modulation is ~1.35e4 (dominates)
        assert self.mod > 1e3

    def test_decay_near_one(self):
        # r=0.1 pc, scale=10 pc → exp(-0.01) ≈ 0.9900
        expected = math.exp(-0.01)
        assert abs(self.dec - expected) < 1e-10

    def test_beta_mod_large(self):
        assert self.beta > 1e3

    def test_reference_modulation_value(self):
        r_cm = 0.1 * PC_TO_CM
        v_cms = 50.0 * KMS_TO_CMS
        expected_mod = 1e46 / (1e-24 * v_cms**2 * r_cm**3)
        assert abs(self.mod - expected_mod) / abs(expected_mod) < 1e-12


# ---------------------------------------------------------------------------
# 4. Monotonicity — energy
# ---------------------------------------------------------------------------

class TestMonotonicityEnergy:
    def test_higher_energy_higher_beta(self):
        beta_lo, _, _ = fallback_modulation(energy_erg=1e30)
        beta_hi, _, _ = fallback_modulation(energy_erg=1e35)
        assert beta_hi > beta_lo

    def test_modulation_proportional_to_energy(self):
        _, mod1, _ = fallback_modulation(energy_erg=1e30)
        _, mod2, _ = fallback_modulation(energy_erg=1e31)
        assert abs(mod2 / mod1 - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. Monotonicity — radius
# ---------------------------------------------------------------------------

class TestMonotonicityRadius:
    def test_larger_radius_lower_beta(self):
        # Both the r³ denominator and the exponential decay suppress modulation
        beta_small, _, _ = fallback_modulation(r_pc=1.0)
        beta_large, _, _ = fallback_modulation(r_pc=50.0)
        assert beta_small > beta_large

    def test_decay_decreases_with_radius(self):
        _, _, dec1 = fallback_modulation(r_pc=1.0, decay_scale_pc=10.0)
        _, _, dec2 = fallback_modulation(r_pc=5.0, decay_scale_pc=10.0)
        assert dec1 > dec2


# ---------------------------------------------------------------------------
# 6 & 7. Decay and modulation bounds
# ---------------------------------------------------------------------------

class TestBounds:
    @pytest.mark.parametrize("r_pc", [0.01, 0.5, 1.0, 5.0, 20.0, 100.0])
    def test_decay_in_unit_interval(self, r_pc):
        _, _, dec = fallback_modulation(r_pc=r_pc, decay_scale_pc=10.0)
        assert 0.0 < dec <= 1.0

    @pytest.mark.parametrize("energy", [1e20, 1e30, 1e40, 1e50])
    def test_modulation_positive(self, energy):
        _, mod, _ = fallback_modulation(energy_erg=energy)
        assert mod > 0.0


# ---------------------------------------------------------------------------
# 8. β_mod ≥ BETA_MOND
# ---------------------------------------------------------------------------

class TestBetaModBaseline:
    @pytest.mark.parametrize("energy,r,v,rho", [
        (1e20, 1.0, 10.0, 1e-20),
        (1e32, 5.0, 20.0, 1e-21),
        (1e46, 0.1, 50.0, 1e-24),
        (1e50, 100.0, 100.0, 1e-26),
    ])
    def test_beta_at_least_mond(self, energy, r, v, rho):
        beta, _, _ = fallback_modulation(energy_erg=energy, r_pc=r, v_kms=v, rho_gcm3=rho)
        assert beta >= BETA_MOND


# ---------------------------------------------------------------------------
# 9. Infinite decay scale → decay ≈ 1
# ---------------------------------------------------------------------------

class TestDecayLimit:
    def test_large_decay_scale_decay_near_one(self):
        _, _, dec = fallback_modulation(r_pc=5.0, decay_scale_pc=1e12)
        assert abs(dec - 1.0) < 1e-6

    def test_small_decay_scale_decay_near_zero(self):
        _, _, dec = fallback_modulation(r_pc=100.0, decay_scale_pc=0.01)
        assert dec < 1e-4


# ---------------------------------------------------------------------------
# 10. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize("kwargs,field", [
        ({"energy_erg": 0.0},    "energy_erg"),
        ({"energy_erg": -1e30},  "energy_erg"),
        ({"r_pc": 0.0},          "r_pc"),
        ({"r_pc": -1.0},         "r_pc"),
        ({"v_kms": 0.0},         "v_kms"),
        ({"v_kms": -5.0},        "v_kms"),
        ({"rho_gcm3": 0.0},      "rho_gcm3"),
        ({"rho_gcm3": -1e-24},   "rho_gcm3"),
        ({"decay_scale_pc": 0.0},  "decay_scale_pc"),
        ({"decay_scale_pc": -1.0}, "decay_scale_pc"),
    ])
    def test_raises_value_error(self, kwargs, field):
        with pytest.raises(ValueError, match=field):
            fallback_modulation(**kwargs)


# ---------------------------------------------------------------------------
# 11. CLI — default run
# ---------------------------------------------------------------------------

class TestCLIDefault:
    def test_default_run_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "scripts/scm_fallback_modulation.py"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0

    def test_default_output_contains_beta(self):
        result = subprocess.run(
            [sys.executable, "scripts/scm_fallback_modulation.py"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert "β modificado" in result.stdout

    def test_default_output_contains_modulation(self):
        result = subprocess.run(
            [sys.executable, "scripts/scm_fallback_modulation.py"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert "modulación" in result.stdout

    def test_default_output_contains_decay(self):
        result = subprocess.run(
            [sys.executable, "scripts/scm_fallback_modulation.py"],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert "decay factor" in result.stdout


# ---------------------------------------------------------------------------
# 12. CLI — custom parameters match Python API
# ---------------------------------------------------------------------------

class TestCLICustom:
    def test_custom_params_match_api(self):
        result = subprocess.run(
            [
                sys.executable, "scripts/scm_fallback_modulation.py",
                "--energy", "1e32",
                "--radius", "5.0",
                "--velocity", "20.0",
                "--density", "1e-21",
                "--decay-scale", "10.0",
            ],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        beta_api, _, _ = fallback_modulation(
            energy_erg=1e32, r_pc=5.0, v_kms=20.0,
            rho_gcm3=1e-21, decay_scale_pc=10.0,
        )
        # The CLI prints beta with 6 decimal places; check the value appears
        assert f"{beta_api:.6f}" in result.stdout

    def test_m31_reference_via_cli(self):
        result = subprocess.run(
            [
                sys.executable, "scripts/scm_fallback_modulation.py",
                "--energy", "1e46",
                "--radius", "0.1",
                "--velocity", "50.0",
                "--density", "1e-24",
                "--decay-scale", "10.0",
            ],
            capture_output=True, text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0
        beta_api, _, _ = fallback_modulation(
            energy_erg=1e46, r_pc=0.1, v_kms=50.0,
            rho_gcm3=1e-24, decay_scale_pc=10.0,
        )
        assert f"{beta_api:.6f}" in result.stdout
