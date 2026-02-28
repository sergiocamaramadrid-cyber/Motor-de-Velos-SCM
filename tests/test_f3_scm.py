"""
tests/test_f3_scm.py — Deterministic unit tests for compute_f3_scm() and
the pilot_f3_test pipeline.

The F3_SCM observable is defined as::

    F_{3,SCM} = d(log V_obs) / d(log r)  |_{r >= 0.7 * R_max}

All tests use fully synthetic, deterministic data so no SPARC download is
required and results are reproducible across environments.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_models import compute_f3_scm


# ---------------------------------------------------------------------------
# Helper factories for synthetic rotation curves
# ---------------------------------------------------------------------------

def _flat_curve(n=20, r_max=15.0, v_flat=150.0):
    """Return (r, v_obs) for a perfectly flat rotation curve."""
    r = np.linspace(0.5, r_max, n)
    v = np.full(n, v_flat)
    return r, v


def _power_law_curve(n=20, r_max=15.0, v0=150.0, alpha=0.1):
    """Return (r, v_obs) for a power-law curve V = v0 * (r/r_max)^alpha."""
    r = np.linspace(0.5, r_max, n)
    v = v0 * (r / r_max) ** alpha
    return r, v


# ---------------------------------------------------------------------------
# compute_f3_scm — return-value contract
# ---------------------------------------------------------------------------

class TestComputeF3ScmContract:
    def test_returns_dict_with_required_keys(self):
        r, v = _flat_curve()
        result = compute_f3_scm(r, v)
        assert isinstance(result, dict)
        for key in ("f3_scm", "n_outer", "r_min_outer", "r_max"):
            assert key in result, f"Missing key: {key}"

    def test_n_outer_is_int(self):
        r, v = _flat_curve()
        result = compute_f3_scm(r, v)
        assert isinstance(result["n_outer"], int)

    def test_r_max_matches_input(self):
        r, v = _flat_curve(r_max=20.0)
        result = compute_f3_scm(r, v)
        assert result["r_max"] == pytest.approx(20.0, rel=1e-6)

    def test_r_min_outer_geq_frac_r_max(self):
        r, v = _flat_curve(n=30, r_max=15.0)
        result = compute_f3_scm(r, v, r_max_frac=0.7)
        assert result["r_min_outer"] >= 0.7 * result["r_max"] - 1e-9


# ---------------------------------------------------------------------------
# compute_f3_scm — physical correctness
# ---------------------------------------------------------------------------

class TestComputeF3ScmPhysics:
    def test_flat_curve_slope_near_zero(self):
        """A flat rotation curve must give F3_SCM ≈ 0."""
        r, v = _flat_curve(n=50, v_flat=200.0)
        result = compute_f3_scm(r, v)
        assert abs(result["f3_scm"]) < 0.01

    def test_rising_curve_positive_slope(self):
        """A rising outer profile must give F3_SCM > 0."""
        r, v = _power_law_curve(n=50, alpha=0.3)
        result = compute_f3_scm(r, v)
        assert result["f3_scm"] > 0.0

    def test_declining_curve_negative_slope(self):
        """A declining outer profile must give F3_SCM < 0."""
        r, v = _power_law_curve(n=50, alpha=-0.3)
        result = compute_f3_scm(r, v)
        assert result["f3_scm"] < 0.0

    def test_power_law_slope_exact(self):
        """For V = v0*(r/r_max)^alpha, the slope must equal alpha exactly."""
        alpha = 0.25
        r, v = _power_law_curve(n=100, r_max=20.0, alpha=alpha)
        result = compute_f3_scm(r, v, r_max_frac=0.5)
        assert result["f3_scm"] == pytest.approx(alpha, abs=1e-6)

    def test_slope_independent_of_v_scale(self):
        """Multiplying V by a constant must not change the slope."""
        r, v = _power_law_curve(n=50, alpha=0.15)
        r1 = compute_f3_scm(r, v)
        r2 = compute_f3_scm(r, 3.7 * v)
        assert r1["f3_scm"] == pytest.approx(r2["f3_scm"], rel=1e-6)

    def test_slope_independent_of_r_scale(self):
        """Rescaling r by a constant must not change the log-log slope."""
        r, v = _power_law_curve(n=50, alpha=0.20)
        r1 = compute_f3_scm(r, v)
        r2 = compute_f3_scm(10.0 * r, v)
        assert r1["f3_scm"] == pytest.approx(r2["f3_scm"], rel=1e-6)


# ---------------------------------------------------------------------------
# compute_f3_scm — edge cases / robustness
# ---------------------------------------------------------------------------

class TestComputeF3ScmEdgeCases:
    def test_fewer_than_two_points_returns_nan(self):
        result = compute_f3_scm(np.array([5.0]), np.array([100.0]))
        assert math.isnan(result["f3_scm"])
        assert result["n_outer"] == 0

    def test_empty_arrays_returns_nan(self):
        result = compute_f3_scm(np.array([]), np.array([]))
        assert math.isnan(result["f3_scm"])

    def test_all_zero_velocities_returns_nan(self):
        r = np.linspace(1.0, 10.0, 10)
        v = np.zeros(10)
        result = compute_f3_scm(r, v)
        assert math.isnan(result["f3_scm"])

    def test_single_outer_point_returns_nan(self):
        """r_max_frac so high that only one outer point qualifies."""
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.full(5, 100.0)
        # r_max = 5.0, frac=0.99 → only r=5.0 qualifies
        result = compute_f3_scm(r, v, r_max_frac=0.99)
        assert math.isnan(result["f3_scm"])
        assert result["n_outer"] == 1

    def test_negative_radii_ignored(self):
        """Negative radii must not affect the computation."""
        r_good = np.linspace(1.0, 15.0, 20)
        v_good = np.full(20, 150.0)
        r_bad = np.concatenate([[-5.0, 0.0], r_good])
        v_bad = np.concatenate([[100.0, 100.0], v_good])
        r1 = compute_f3_scm(r_good, v_good)
        r2 = compute_f3_scm(r_bad, v_bad)
        assert r1["f3_scm"] == pytest.approx(r2["f3_scm"], abs=1e-6)

    def test_r_max_frac_zero_uses_all_points(self):
        """r_max_frac=0 should include all valid points."""
        r, v = _power_law_curve(n=40, alpha=0.2)
        result = compute_f3_scm(r, v, r_max_frac=0.0)
        assert result["n_outer"] == 40


# ---------------------------------------------------------------------------
# n_outer sanity
# ---------------------------------------------------------------------------

class TestComputeF3ScmNOuter:
    def test_n_outer_increases_with_smaller_frac(self):
        r, v = _flat_curve(n=50)
        n7 = compute_f3_scm(r, v, r_max_frac=0.7)["n_outer"]
        n5 = compute_f3_scm(r, v, r_max_frac=0.5)["n_outer"]
        assert n5 >= n7

    def test_n_outer_equals_outer_region_count(self):
        r = np.linspace(1.0, 10.0, 20)
        v = np.full(20, 120.0)
        result = compute_f3_scm(r, v, r_max_frac=0.7)
        expected = int((r >= 0.7 * r.max()).sum())
        assert result["n_outer"] == expected


# ---------------------------------------------------------------------------
# Pilot script integration — uses synthetic rotmod files
# ---------------------------------------------------------------------------

@pytest.fixture()
def rotmod_dir(tmp_path):
    """Synthetic SPARC rotmod directory with three galaxies."""
    rng = np.random.default_rng(99)
    galaxies = {
        "NGC_FLAT": (0.0, 180.0),    # flat → F3 ≈ 0
        "NGC_RISE": (0.2, 150.0),    # rising
        "NGC_FALL": (-0.2, 200.0),   # falling
    }
    for name, (alpha, v0) in galaxies.items():
        r = np.linspace(0.5, 15.0, 25)
        v = v0 * (r / r.max()) ** alpha + rng.normal(0, 0.5, 25)
        v = np.maximum(v, 1.0)  # keep positive
        df = pd.DataFrame({
            "Rad": r, "Vobs": v,
            "errV": np.full(25, 5.0),
            "Vgas": 0.3 * v, "Vdisk": 0.7 * v,
            "Vbul": np.zeros(25), "SBdisk": np.zeros(25), "SBbul": np.zeros(25),
        })
        df.to_csv(tmp_path / f"{name}_rotmod.dat", sep=" ", index=False, header=False)
    return tmp_path


class TestPilotF3Test:
    def test_run_pilot_returns_dataframe(self, rotmod_dir):
        from scripts.pilot_f3_test import run_pilot
        df = run_pilot(rotmod_dir, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_run_pilot_processes_all_galaxies(self, rotmod_dir):
        from scripts.pilot_f3_test import run_pilot
        df = run_pilot(rotmod_dir, verbose=False)
        assert len(df) == 3

    def test_run_pilot_expected_columns(self, rotmod_dir):
        from scripts.pilot_f3_test import run_pilot
        df = run_pilot(rotmod_dir, verbose=False)
        for col in ("galaxy", "f3_scm", "n_outer", "r_min_outer", "r_max"):
            assert col in df.columns

    def test_run_pilot_writes_csv(self, rotmod_dir, tmp_path):
        from scripts.pilot_f3_test import run_pilot
        out = tmp_path / "f3_out"
        run_pilot(rotmod_dir, out_dir=out, verbose=False)
        assert (out / "f3_scm_results.csv").exists()

    def test_run_pilot_writes_summary_log(self, rotmod_dir, tmp_path):
        from scripts.pilot_f3_test import run_pilot
        out = tmp_path / "f3_out"
        run_pilot(rotmod_dir, out_dir=out, verbose=False)
        log_path = out / "f3_scm_summary.log"
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8")
        assert "F3_SCM" in text

    def test_run_pilot_missing_dir_raises(self, tmp_path):
        from scripts.pilot_f3_test import run_pilot
        with pytest.raises(FileNotFoundError):
            run_pilot(tmp_path / "nonexistent", verbose=False)

    def test_run_pilot_empty_dir_raises(self, tmp_path):
        from scripts.pilot_f3_test import run_pilot
        with pytest.raises(FileNotFoundError):
            run_pilot(tmp_path, verbose=False)

    def test_run_pilot_n_outer_is_int_dtype(self, rotmod_dir):
        from scripts.pilot_f3_test import run_pilot
        df = run_pilot(rotmod_dir, verbose=False)
        assert pd.api.types.is_integer_dtype(df["n_outer"])

    def test_run_pilot_flat_galaxy_near_zero(self, rotmod_dir):
        from scripts.pilot_f3_test import run_pilot
        df = run_pilot(rotmod_dir, verbose=False)
        flat_row = df[df["galaxy"] == "NGC_FLAT"]
        assert len(flat_row) == 1
        assert abs(float(flat_row["f3_scm"].iloc[0])) < 0.1
