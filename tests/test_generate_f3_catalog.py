"""
tests/test_generate_f3_catalog.py — Tests for scripts/generate_f3_catalog.py.

Uses synthetic data; no real SPARC download required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.generate_f3_catalog import (
    generate_synthetic,
    _per_galaxy_friction_slope,
    main,
    EXPECTED_SLOPE,
    A0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
    MIN_DEEP_POINTS,
)


# ---------------------------------------------------------------------------
# _per_galaxy_friction_slope unit tests
# ---------------------------------------------------------------------------

class TestPerGalaxyFrictionSlope:
    """Unit tests for the per-galaxy deep-regime slope fitter."""

    def _make_mond_rc(self, n_pts: int = 30, beta: float = 0.5,
                     noise: float = 0.0, seed: int = 0):
        """Synthetic rotation curve where deep-regime slope = beta."""
        rng = np.random.default_rng(seed)
        a0 = A0_DEFAULT
        threshold = DEEP_THRESHOLD_DEFAULT

        # Choose g_bar values in the deep regime and convert to rotation-curve arrays
        g_bar = rng.uniform(0.005 * a0, 0.28 * a0, n_pts)
        # g_obs from the MOND-like relation with planted slope
        g_obs = 10 ** (beta * np.log10(g_bar) + 0.5 * np.log10(a0)
                       + rng.normal(0, noise, n_pts))

        # Invert g_bar = V_bar² / r  →  use unit radius so V_bar = sqrt(g_bar / _CONV)
        from scripts.generate_f3_catalog import _CONV
        r = np.ones(n_pts)  # 1 kpc
        v_bar = np.sqrt(g_bar / _CONV)
        v_obs = np.sqrt(g_obs / _CONV)

        return r, v_obs, v_bar, np.zeros(n_pts), np.zeros(n_pts)

    def test_returns_nan_when_too_few_points(self):
        """Slope must be NaN when fewer than MIN_DEEP_POINTS deep points exist."""
        # Place all points in the Newtonian regime (g_bar >> a0)
        from scripts.generate_f3_catalog import _CONV
        r = np.ones(30)
        v_bar_large = np.full(30, 300.0)   # km/s — Newtonian regime
        v_obs = np.full(30, 300.0)
        slope, err = _per_galaxy_friction_slope(
            r, v_obs, v_bar_large, np.zeros(30), np.zeros(30), a0=A0_DEFAULT,
        )
        assert np.isnan(slope)
        assert np.isnan(err)

    def test_recovers_mond_slope(self):
        """Should recover β ≈ 0.5 for pure MOND data."""
        r, v_obs, v_gas, v_disk, v_bul = self._make_mond_rc(n_pts=50, beta=0.5, noise=0.0)
        slope, err = _per_galaxy_friction_slope(r, v_obs, v_gas, v_disk, v_bul)
        assert slope == pytest.approx(0.5, abs=0.01)

    def test_recovers_planted_slope(self):
        """Should recover a planted slope of 0.44."""
        r, v_obs, v_gas, v_disk, v_bul = self._make_mond_rc(n_pts=60, beta=0.44, noise=0.001)
        slope, err = _per_galaxy_friction_slope(r, v_obs, v_gas, v_disk, v_bul)
        assert slope == pytest.approx(0.44, abs=0.03)

    def test_error_is_nonneg_finite(self):
        """Slope standard error must be non-negative and finite."""
        r, v_obs, v_gas, v_disk, v_bul = self._make_mond_rc(n_pts=40, beta=0.5, noise=0.005)
        slope, err = _per_galaxy_friction_slope(r, v_obs, v_gas, v_disk, v_bul)
        assert np.isfinite(err)
        assert err >= 0.0


# ---------------------------------------------------------------------------
# generate_synthetic tests
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    def test_output_has_required_columns(self, tmp_path):
        out = tmp_path / "cat.csv"
        df = generate_synthetic(out, n_galaxies=10, seed=0)
        for col in ["galaxy", "friction_slope", "friction_slope_err", "velo_inerte_flag"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_n_galaxies_matches_request(self, tmp_path):
        out = tmp_path / "cat.csv"
        df = generate_synthetic(out, n_galaxies=20, seed=1)
        assert len(df) == 20

    def test_csv_written(self, tmp_path):
        out = tmp_path / "cat.csv"
        generate_synthetic(out, n_galaxies=5, seed=2)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 5

    def test_mean_slope_near_half(self, tmp_path):
        """Mean friction_slope of the synthetic ensemble should be close to 0.5."""
        out = tmp_path / "cat.csv"
        df = generate_synthetic(out, n_galaxies=200, seed=42)
        mean_beta = df["friction_slope"].mean()
        assert mean_beta == pytest.approx(EXPECTED_SLOPE, abs=0.05)

    def test_velo_inerte_flag_is_bool(self, tmp_path):
        out = tmp_path / "cat.csv"
        df = generate_synthetic(out, n_galaxies=10, seed=3)
        assert df["velo_inerte_flag"].dtype == bool

    def test_slope_err_positive(self, tmp_path):
        out = tmp_path / "cat.csv"
        df = generate_synthetic(out, n_galaxies=15, seed=4)
        assert (df["friction_slope_err"] > 0).all()

    def test_reproducible_with_same_seed(self, tmp_path):
        out1 = tmp_path / "cat1.csv"
        out2 = tmp_path / "cat2.csv"
        df1 = generate_synthetic(out1, n_galaxies=20, seed=99)
        df2 = generate_synthetic(out2, n_galaxies=20, seed=99)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# CLI (main) tests
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_synthetic_mode_creates_csv(self, tmp_path):
        out = tmp_path / "f3_catalog.csv"
        df = main(["--synthetic", "--n-galaxies", "15", "--out", str(out)])
        assert out.exists()
        assert len(df) == 15

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--data-dir", str(tmp_path / "nonexistent"), "--out", "/tmp/x.csv"])

    def test_custom_n_galaxies(self, tmp_path):
        out = tmp_path / "cat.csv"
        df = main(["--synthetic", "--n-galaxies", "7", "--out", str(out)])
        assert len(df) == 7
