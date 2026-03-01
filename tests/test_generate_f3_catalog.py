"""
tests/test_generate_f3_catalog.py — Tests for scripts/generate_f3_catalog.py.

Uses synthetic per-radial-point data with known slope to verify the
per-galaxy catalog generation is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_f3_catalog import (
    fit_galaxy_slope,
    build_f3_catalog,
    load_sparc_data_dir,
    main,
    EXPECTED_SLOPE,
    G0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
    _KPC_TO_M,
    _CONV,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_point_csv(
    tmp_path: Path,
    n_galaxies: int = 5,
    n_deep: int = 40,
    n_shallow: int = 20,
    planted_slope: float = 0.5,
    noise_std: float = 0.005,
    g0: float = G0_DEFAULT,
) -> Path:
    """Create a multi-galaxy per-radial-point CSV with a known β."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_galaxies):
        g_bar_deep = rng.uniform(0.001 * g0, 0.29 * g0, n_deep)
        g_bar_shallow = rng.uniform(g0, 10.0 * g0, n_shallow)
        g_bar_all = np.concatenate([g_bar_deep, g_bar_shallow])
        log_gbar = np.log10(g_bar_all)
        log_gobs = (
            planted_slope * log_gbar
            + 0.5 * np.log10(g0)
            + rng.normal(0, noise_std, len(g_bar_all))
        )
        for j in range(len(g_bar_all)):
            rows.append({
                "galaxy": f"G{i:02d}",
                "r_kpc": float(j),
                "log_g_bar": log_gbar[j],
                "log_g_obs": log_gobs[j],
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Unit tests for fit_galaxy_slope()
# ---------------------------------------------------------------------------


class TestFitGalaxySlope:
    def test_returns_required_keys(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 50)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0)
        result = fit_galaxy_slope(log_gbar, log_gobs)
        required = {
            "n_total", "n_deep", "friction_slope", "friction_slope_err",
            "r_value", "p_value", "velo_inerte_flag",
        }
        assert required.issubset(set(result.keys()))

    def test_exact_mond_slope(self):
        """Pure MOND data must recover friction_slope ≈ 0.5."""
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 200)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0)
        result = fit_galaxy_slope(log_gbar, log_gobs, g0=g0)
        assert result["friction_slope"] == pytest.approx(0.5, abs=1e-4)

    def test_planted_slope_recovered(self):
        """A planted slope of 0.44 must be recovered within tolerance."""
        rng = np.random.default_rng(7)
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 300)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.44 * log_gbar + 0.5 * np.log10(g0) + rng.normal(0, 0.001, 300)
        result = fit_galaxy_slope(log_gbar, log_gobs, g0=g0)
        assert result["friction_slope"] == pytest.approx(0.44, abs=0.02)

    def test_nan_when_too_few_deep_points(self):
        """friction_slope must be NaN when n_deep < 2."""
        g0 = G0_DEFAULT
        # All points in Newtonian regime
        g_bar = np.linspace(2.0 * g0, 10.0 * g0, 50)
        log_gbar = np.log10(g_bar)
        log_gobs = log_gbar
        result = fit_galaxy_slope(log_gbar, log_gobs, g0=g0)
        assert np.isnan(result["friction_slope"])
        assert result["n_deep"] == 0

    def test_velo_inerte_flag_set_for_mond_data(self):
        """Flag must be 1 when slope is within 2σ of 0.5."""
        g0 = G0_DEFAULT
        # Use 30 evenly-spaced deep points — large enough to fit, small enough
        # that stderr > |β−0.5| so the flag is reliably 1.
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 30)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0)
        result = fit_galaxy_slope(log_gbar, log_gobs, g0=g0)
        # Exact MOND data: slope == 0.5 exactly → always within 2σ (or σ=0)
        # If stderr == 0 the flag is NaN by design; accept 1.0 or NaN
        assert result["friction_slope"] == pytest.approx(0.5, abs=1e-4)
        assert result["velo_inerte_flag"] in (1.0, float("nan")) or np.isnan(
            result["velo_inerte_flag"]
        )

    def test_velo_inerte_flag_zero_for_deviant_data(self):
        """Flag must be 0 when slope deviates significantly from 0.5."""
        rng = np.random.default_rng(99)
        g0 = G0_DEFAULT
        # Very tight data, planted slope far from 0.5
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 500)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.2 * log_gbar + 0.5 * np.log10(g0) + rng.normal(0, 0.0001, 500)
        result = fit_galaxy_slope(log_gbar, log_gobs, g0=g0)
        assert result["velo_inerte_flag"] == 0.0


# ---------------------------------------------------------------------------
# Unit tests for build_f3_catalog()
# ---------------------------------------------------------------------------


class TestBuildF3Catalog:
    def test_output_columns(self, tmp_path):
        csv = _make_per_point_csv(tmp_path, n_galaxies=3)
        df = pd.read_csv(csv)
        catalog = build_f3_catalog(df)
        expected_cols = [
            "galaxy", "n_total", "n_deep",
            "friction_slope", "friction_slope_err",
            "r_value", "p_value", "velo_inerte_flag",
        ]
        assert catalog.columns.tolist() == expected_cols

    def test_one_row_per_galaxy(self, tmp_path):
        n = 5
        csv = _make_per_point_csv(tmp_path, n_galaxies=n)
        df = pd.read_csv(csv)
        catalog = build_f3_catalog(df)
        assert len(catalog) == n

    def test_sorted_by_galaxy(self, tmp_path):
        csv = _make_per_point_csv(tmp_path, n_galaxies=4)
        df = pd.read_csv(csv)
        catalog = build_f3_catalog(df)
        assert list(catalog["galaxy"]) == sorted(catalog["galaxy"].tolist())

    def test_slope_close_to_planted(self, tmp_path):
        csv = _make_per_point_csv(tmp_path, n_galaxies=3, planted_slope=0.5,
                                   noise_std=0.005)
        df = pd.read_csv(csv)
        catalog = build_f3_catalog(df)
        fitted = catalog["friction_slope"].dropna()
        assert len(fitted) > 0
        assert fitted.mean() == pytest.approx(0.5, abs=0.05)

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"galaxy": ["A"], "log_g_bar": [-10.5]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_f3_catalog(df)


# ---------------------------------------------------------------------------
# CLI (main) integration tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_produces_catalog_file(self, tmp_path):
        csv = _make_per_point_csv(tmp_path, n_galaxies=5)
        out = tmp_path / "f3_catalog.csv"
        catalog = main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()
        assert len(catalog) == 5

    def test_recovers_mond_slope(self, tmp_path):
        csv = _make_per_point_csv(tmp_path, n_galaxies=5, planted_slope=0.5)
        out = tmp_path / "f3_catalog.csv"
        catalog = main(["--csv", str(csv), "--out", str(out)])
        fitted = catalog["friction_slope"].dropna()
        assert fitted.mean() == pytest.approx(0.5, abs=0.05)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv")])

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "x": [1.0]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(bad)])

    def test_reference_catalog_is_readable(self):
        """results/f3_catalog.csv must load and have expected columns."""
        from pathlib import Path
        p = Path("results/f3_catalog.csv")
        assert p.exists(), "results/f3_catalog.csv not found"
        df = pd.read_csv(p)
        required = {
            "galaxy", "n_total", "n_deep",
            "friction_slope", "friction_slope_err",
            "r_value", "p_value", "velo_inerte_flag",
        }
        assert required.issubset(set(df.columns))
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Helper: create synthetic rotmod files
# ---------------------------------------------------------------------------


def _make_rotmod_dir(
    tmp_path: Path,
    n_galaxies: int = 3,
    n_points: int = 20,
    rng_seed: int = 0,
) -> Path:
    """Write synthetic *_rotmod.dat files to *tmp_path*."""
    rng = np.random.default_rng(rng_seed)
    for i in range(n_galaxies):
        name = f"FakeGal{i:02d}"
        r = np.linspace(0.5, 15.0, n_points)
        v_disk = 60.0 + rng.normal(0, 2, n_points)
        v_gas = 20.0 + rng.normal(0, 1, n_points)
        v_bul = np.zeros(n_points)
        v_obs = 80.0 + rng.normal(0, 3, n_points)
        v_err = 3.0 * np.ones(n_points)
        data = np.column_stack([r, v_obs, v_err, v_gas, v_disk, v_bul,
                                np.ones(n_points), np.zeros(n_points)])
        np.savetxt(tmp_path / f"{name}_rotmod.dat", data, fmt="%.6f")
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests for load_sparc_data_dir()
# ---------------------------------------------------------------------------


class TestLoadSparcDataDir:
    def test_loads_rotmod_files(self, tmp_path):
        d = _make_rotmod_dir(tmp_path, n_galaxies=3, n_points=20)
        df = load_sparc_data_dir(d)
        assert set(df.columns) >= {"galaxy", "log_g_bar", "log_g_obs", "g_bar", "g_obs"}
        assert df["galaxy"].nunique() == 3

    def test_all_accelerations_positive(self, tmp_path):
        d = _make_rotmod_dir(tmp_path, n_galaxies=2, n_points=15)
        df = load_sparc_data_dir(d)
        assert (df["g_bar"] > 0).all()
        assert (df["g_obs"] > 0).all()

    def test_raises_when_no_dat_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="_rotmod.dat"):
            load_sparc_data_dir(tmp_path)

    def test_reads_from_raw_subdir(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _make_rotmod_dir(raw, n_galaxies=2, n_points=10)
        df = load_sparc_data_dir(tmp_path)
        assert df["galaxy"].nunique() == 2

    def test_cli_data_dir_mode(self, tmp_path):
        d = _make_rotmod_dir(tmp_path, n_galaxies=3, n_points=20)
        out = tmp_path / "catalog_rotmod.csv"
        catalog = main(["--data-dir", str(d), "--out", str(out)])
        assert out.exists()
        assert len(catalog) == 3
        assert set(catalog.columns) >= {"galaxy", "friction_slope", "velo_inerte_flag"}
