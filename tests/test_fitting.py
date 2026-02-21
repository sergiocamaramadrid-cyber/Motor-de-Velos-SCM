"""Unit tests for src/fitting.py — fit_p0_local and supporting functions."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repository root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scm_models import (
    A0_DEFAULT,
    aicc,
    chi2_gaussian,
    nu_rar,
    v_model_baseline,
    v_model_universal,
)
from src.fitting import fit_p0_local, _cost_baseline, _cost_universal, K_PARAMS
from src.io_utils import load_rotation_curve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
GALAXY_CSV = DATA_DIR / "GXY_D13.8_V144_SCM_01.csv"


@pytest.fixture
def galaxy_data():
    """Load the example galaxy rotation curve."""
    return load_rotation_curve(GALAXY_CSV)


# ---------------------------------------------------------------------------
# Tests: nu_rar
# ---------------------------------------------------------------------------

class TestNuRar:
    def test_newtonian_limit(self):
        """nu(x) → 1 as x → ∞."""
        x = np.array([1e6])
        assert abs(nu_rar(x)[0] - 1.0) < 1e-4

    def test_deep_mond_limit(self):
        """nu(x) → 1/sqrt(x) as x → 0."""
        x = np.array([1e-8])
        expected = 1.0 / np.sqrt(1e-8)
        assert abs(nu_rar(x)[0] - expected) / expected < 0.01

    def test_nu_positive(self):
        """nu is always ≥ 1."""
        x = np.logspace(-6, 4, 50)
        assert np.all(nu_rar(x) >= 1.0 - 1e-9)

    def test_scalar_input(self):
        """Scalar input returns scalar-like array."""
        result = nu_rar(np.array([1.0]))
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Tests: chi2_gaussian and aicc
# ---------------------------------------------------------------------------

class TestChi2AndAICc:
    def test_chi2_perfect_fit(self):
        v = np.array([100.0, 120.0, 130.0])
        assert chi2_gaussian(v, v, np.ones_like(v)) == pytest.approx(0.0)

    def test_chi2_known_value(self):
        v_obs   = np.array([100.0])
        v_model = np.array([105.0])
        v_err   = np.array([5.0])
        assert chi2_gaussian(v_obs, v_model, v_err) == pytest.approx(1.0)

    def test_aicc_formula(self):
        # AICc = chi2 + 2k + 2k(k+1)/(n-k-1)
        chi2, n, k = 10.0, 14, 2
        expected = 10.0 + 4.0 + 2*2*3/(14-2-1)
        assert aicc(chi2, n, k) == pytest.approx(expected)

    def test_aicc_undefined(self):
        """AICc is undefined when n - k - 1 ≤ 0."""
        with pytest.raises(ValueError):
            aicc(10.0, n_data=3, k_params=2)


# ---------------------------------------------------------------------------
# Tests: velocity models
# ---------------------------------------------------------------------------

class TestVelocityModels:
    r    = np.array([1.0, 5.0, 10.0])
    vdisk = np.array([80.0, 120.0, 80.0])
    vgas  = np.array([20.0, 30.0, 20.0])

    def test_baseline_positive(self):
        v = v_model_baseline(self.r, self.vdisk, self.vgas, 0.5, math.log10(A0_DEFAULT))
        assert np.all(v > 0)

    def test_universal_ge_rar(self):
        """Universal model with V_ext > 0 always ≥ baseline with same a0."""
        upsilon = 0.5
        log10_a0 = math.log10(A0_DEFAULT)
        v_bl = v_model_baseline(self.r, self.vdisk, self.vgas, upsilon, log10_a0)
        v_un = v_model_universal(self.r, self.vdisk, self.vgas, upsilon,
                                  1.5, A0_DEFAULT)   # V_ext ≈ 31.6 km/s
        assert np.all(v_un >= v_bl)

    def test_universal_vext_zero_equals_rar(self):
        """V_ext → 0 makes universal model equal to RAR (at fixed a0)."""
        upsilon = 0.5
        log10_a0 = math.log10(A0_DEFAULT)
        v_rar = v_model_baseline(self.r, self.vdisk, self.vgas, upsilon, log10_a0)
        # V_ext = 10^-10 ≈ 0
        v_un = v_model_universal(self.r, self.vdisk, self.vgas, upsilon,
                                  -10.0, A0_DEFAULT)
        np.testing.assert_allclose(v_un, v_rar, rtol=1e-4)

    def test_baseline_upsilon_scaling(self):
        """Higher Upsilon → higher baryonic contribution → higher velocity."""
        log10_a0 = math.log10(A0_DEFAULT)
        v_low  = v_model_baseline(self.r, self.vdisk, self.vgas, 0.2, log10_a0)
        v_high = v_model_baseline(self.r, self.vdisk, self.vgas, 1.5, log10_a0)
        assert np.all(v_high >= v_low)


# ---------------------------------------------------------------------------
# Tests: fit_p0_local
# ---------------------------------------------------------------------------

class TestFitP0Local:
    def test_returns_expected_keys(self):
        result = fit_p0_local(GALAXY_CSV)
        expected_keys = {
            "galaxy", "n_points", "k_params",
            "chi2_baseline", "chi2_universal",
            "aicc_baseline", "aicc_universal",
            "delta_aicc",
            "params_baseline", "params_universal",
            "fit_success_baseline", "fit_success_universal",
        }
        assert expected_keys.issubset(result.keys())

    def test_k_params_equals_two(self):
        result = fit_p0_local(GALAXY_CSV)
        assert result["k_params"] == K_PARAMS == 2

    def test_galaxy_name(self):
        result = fit_p0_local(GALAXY_CSV)
        assert result["galaxy"] == "GXY_D13.8_V144_SCM_01"

    def test_n_points(self):
        result = fit_p0_local(GALAXY_CSV)
        assert result["n_points"] == 14

    def test_aicc_consistency(self):
        """delta_aicc must equal aicc_universal − aicc_baseline."""
        result = fit_p0_local(GALAXY_CSV)
        diff = result["aicc_universal"] - result["aicc_baseline"]
        assert abs(result["delta_aicc"] - diff) < 1e-6

    def test_universal_preferred(self):
        """Universal-SCM should be preferred (ΔAICc < 0) for this galaxy."""
        result = fit_p0_local(GALAXY_CSV)
        assert result["delta_aicc"] < 0, (
            f"Expected ΔAICc < 0 (Universal preferred), got {result['delta_aicc']:.4f}"
        )

    def test_chi2_reasonable(self):
        """chi2 should be in [0, 5*n] for a reasonable fit."""
        result = fit_p0_local(GALAXY_CSV)
        n = result["n_points"]
        assert 0 <= result["chi2_baseline"] <= 5 * n
        assert 0 <= result["chi2_universal"] <= 5 * n

    def test_params_baseline_shape(self):
        result = fit_p0_local(GALAXY_CSV)
        assert len(result["params_baseline"]) == K_PARAMS

    def test_params_universal_shape(self):
        result = fit_p0_local(GALAXY_CSV)
        assert len(result["params_universal"]) == K_PARAMS

    def test_upsilon_positive(self):
        """Best-fit Υ must be positive for both models."""
        result = fit_p0_local(GALAXY_CSV)
        assert result["params_baseline"][0] > 0
        assert result["params_universal"][0] > 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            fit_p0_local("data/nonexistent_galaxy.csv")

    def test_custom_p0(self):
        """Custom initial parameters should still converge to a valid result."""
        import math
        result = fit_p0_local(
            GALAXY_CSV,
            p0_baseline=(1.0, math.log10(A0_DEFAULT)),
            p0_universal=(1.0, 1.8),
        )
        assert np.isfinite(result["delta_aicc"])


# ---------------------------------------------------------------------------
# Tests: load_rotation_curve
# ---------------------------------------------------------------------------

class TestLoadRotationCurve:
    def test_loads_successfully(self):
        df = load_rotation_curve(GALAXY_CSV)
        assert len(df) == 14

    def test_required_columns_present(self):
        df = load_rotation_curve(GALAXY_CSV)
        for col in ["r", "Vobs", "eVobs", "Vdisk", "Vgas"]:
            assert col in df.columns

    def test_all_finite(self):
        df = load_rotation_curve(GALAXY_CSV)
        for col in ["r", "Vobs", "eVobs", "Vdisk", "Vgas"]:
            assert np.all(np.isfinite(df[col].values))

    def test_positive_r(self):
        df = load_rotation_curve(GALAXY_CSV)
        assert (df["r"] > 0).all()

    def test_positive_evobs(self):
        df = load_rotation_curve(GALAXY_CSV)
        assert (df["eVobs"] > 0).all()
