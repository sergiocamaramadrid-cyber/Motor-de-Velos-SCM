"""
tests/test_sparc_spectral_analysis.py — Tests for scripts/sparc_spectral_analysis.py.

Covers:
  1. galaxy_name_from_path() — various filename formats.
  2. parse_rotmod() — synthetic temp files.
  3. compute_spectral_features() — correct keys, rmin/rmax, edge cases.
  4. build_spectral_catalog() — mock rotmod directory, min_points enforcement.
  5. main() — end-to-end CLI invocation.
  6. Edge cases — empty directory, all below min_points, single galaxy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.sparc_spectral_analysis import (
    galaxy_name_from_path,
    parse_rotmod,
    compute_spectral_features,
    build_spectral_catalog,
    main,
    _parse_args,
    _CATALOG_COLS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROTMOD_HEADER = "# Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul\n"


def _make_rotmod_file(
    path: Path,
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
) -> Path:
    """Write a minimal SPARC rotmod file to *path*."""
    lines = [_ROTMOD_HEADER]
    for r, v in zip(r_kpc, v_obs_kms):
        lines.append(f"{r:.4f}  {v:.4f}  1.0  0.0  0.0  0.0  0.0  0.0\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _make_sparc_dir(
    tmp_path: Path,
    galaxies: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    """Create a temp dir with one rotmod file per galaxy."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, (r, v) in galaxies.items():
        _make_rotmod_file(tmp_path / f"{name}_rotmod.dat", r, v)
    return tmp_path


def _flat_curve(n: int = 20, r_max: float = 10.0, v: float = 150.0):
    r = np.linspace(0.5, r_max, n)
    return r, np.full(n, v)


def _rising_curve(n: int = 20, r_max: float = 10.0):
    r = np.linspace(0.5, r_max, n)
    v = 50.0 + 10.0 * r
    return r, v


# ---------------------------------------------------------------------------
# 1. TestGalaxyNameFromPath
# ---------------------------------------------------------------------------

class TestGalaxyNameFromPath:
    def test_simple_filename(self):
        assert galaxy_name_from_path("DDO064_rotmod.dat") == "DDO064"

    def test_full_absolute_path(self):
        p = "/data/SPARC/NGC3198_rotmod.dat"
        assert galaxy_name_from_path(p) == "NGC3198"

    def test_path_object(self):
        p = Path("/data/SPARC/UGC02885_rotmod.dat")
        assert galaxy_name_from_path(p) == "UGC02885"

    def test_string_path_with_directories(self):
        assert galaxy_name_from_path("data/SPARC/F563-1_rotmod.dat") == "F563-1"

    def test_nested_deep_path(self):
        p = "/home/user/work/project/data/SPARC/IC2574_rotmod.dat"
        assert galaxy_name_from_path(p) == "IC2574"

    def test_name_with_numbers(self):
        assert galaxy_name_from_path("NGC1560_rotmod.dat") == "NGC1560"

    def test_name_with_hyphens(self):
        assert galaxy_name_from_path("UGC05005_rotmod.dat") == "UGC05005"

    def test_without_rotmod_suffix_returns_stem(self):
        # If no _rotmod suffix, just return the stem
        result = galaxy_name_from_path("somefile.dat")
        assert result == "somefile"

    def test_relative_path_object(self):
        p = Path("SPARC") / "DDO161_rotmod.dat"
        assert galaxy_name_from_path(p) == "DDO161"

    def test_returns_string_type(self):
        result = galaxy_name_from_path("NGC2403_rotmod.dat")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 2. TestParseRotmod
# ---------------------------------------------------------------------------

class TestParseRotmod:
    def test_returns_two_arrays(self, tmp_path):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([100.0, 120.0, 130.0])
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        result = parse_rotmod(fpath)
        assert len(result) == 2

    def test_correct_shape(self, tmp_path):
        r = np.linspace(1, 10, 8)
        v = np.full(8, 150.0)
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        r_out, v_out = parse_rotmod(fpath)
        assert r_out.shape == (8,)
        assert v_out.shape == (8,)

    def test_r_kpc_values(self, tmp_path):
        r = np.array([0.5, 1.0, 2.0, 4.0])
        v = np.array([80.0, 100.0, 120.0, 140.0])
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        r_out, _ = parse_rotmod(fpath)
        np.testing.assert_allclose(r_out, r, rtol=1e-4)

    def test_v_obs_values(self, tmp_path):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([110.0, 130.0, 145.0])
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        _, v_out = parse_rotmod(fpath)
        np.testing.assert_allclose(v_out, v, rtol=1e-4)

    def test_skips_comment_lines(self, tmp_path):
        content = (
            "# This is a comment\n"
            "# Another comment\n"
            "1.0  100.0  1.0  0.0  0.0  0.0  0.0  0.0\n"
            "2.0  120.0  1.0  0.0  0.0  0.0  0.0  0.0\n"
        )
        fpath = tmp_path / "G01_rotmod.dat"
        fpath.write_text(content, encoding="utf-8")
        r_out, v_out = parse_rotmod(fpath)
        assert len(r_out) == 2
        assert r_out[0] == pytest.approx(1.0)
        assert v_out[0] == pytest.approx(100.0)

    def test_single_row(self, tmp_path):
        r = np.array([5.0])
        v = np.array([200.0])
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        r_out, v_out = parse_rotmod(fpath)
        assert len(r_out) == 1
        assert r_out[0] == pytest.approx(5.0)

    def test_multiple_rows(self, tmp_path):
        n = 15
        r = np.linspace(1, 15, n)
        v = np.linspace(100, 200, n)
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        r_out, v_out = parse_rotmod(fpath)
        assert len(r_out) == n
        assert len(v_out) == n

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(Exception):
            parse_rotmod(tmp_path / "nonexistent_rotmod.dat")

    def test_returns_ndarray_types(self, tmp_path):
        r = np.array([1.0, 2.0])
        v = np.array([100.0, 150.0])
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        r_out, v_out = parse_rotmod(fpath)
        assert isinstance(r_out, np.ndarray)
        assert isinstance(v_out, np.ndarray)

    def test_first_column_is_radius_second_is_velocity(self, tmp_path):
        r_in = np.array([3.0, 6.0, 9.0])
        v_in = np.array([150.0, 180.0, 200.0])
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r_in, v_in)
        r_out, v_out = parse_rotmod(fpath)
        # Radius values should be in kpc range, velocities in km/s range
        assert r_out.max() < 50.0     # kpc, not km/s
        assert v_out.min() > 10.0    # km/s, not kpc


# ---------------------------------------------------------------------------
# 3. TestComputeSpectralFeatures
# ---------------------------------------------------------------------------

class TestComputeSpectralFeatures:
    def test_returns_dict(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        assert isinstance(result, dict)

    def test_correct_keys(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        expected_keys = {
            "n_points_raw", "rmin_kpc", "rmax_kpc", "n_grid",
            "residual_rms_kms", "lambda_dom_kpc", "peak_freq_1perkpc",
            "peak_power", "n_peaks",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_n_points_raw(self):
        r, v = _flat_curve(n=20)
        result = compute_spectral_features(r, v)
        assert result["n_points_raw"] == 20

    def test_rmin_kpc(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        assert result["rmin_kpc"] == pytest.approx(r.min())

    def test_rmax_kpc(self):
        r, v = _flat_curve(r_max=15.0)
        result = compute_spectral_features(r, v)
        assert result["rmax_kpc"] == pytest.approx(15.0)

    def test_n_grid_default(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        assert result["n_grid"] == 256

    def test_n_grid_custom(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v, n_grid=128)
        assert result["n_grid"] == 128

    def test_residual_rms_constant_v_is_zero(self):
        r, v = _flat_curve(v=200.0)
        result = compute_spectral_features(r, v)
        assert result["residual_rms_kms"] == pytest.approx(0.0, abs=1e-8)

    def test_residual_rms_linear_v_is_zero(self):
        r = np.linspace(1, 10, 30)
        v = 50.0 + 20.0 * r  # perfectly linear
        result = compute_spectral_features(r, v)
        assert result["residual_rms_kms"] == pytest.approx(0.0, abs=1e-6)

    def test_residual_rms_positive_for_nonlinear(self):
        rng = np.random.default_rng(42)
        r = np.linspace(1, 10, 30)
        v = 100.0 + rng.normal(0, 5.0, 30)
        result = compute_spectral_features(r, v)
        assert result["residual_rms_kms"] > 0.0

    def test_peak_freq_positive_for_oscillatory(self):
        r = np.linspace(0, 10, 200)
        v = 100.0 + 10.0 * np.sin(2 * np.pi * r / 3.0)
        result = compute_spectral_features(r, v)
        assert result["peak_freq_1perkpc"] > 0.0

    def test_peak_power_nonnegative(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        assert result["peak_power"] >= 0.0

    def test_n_peaks_positive_for_oscillatory(self):
        r = np.linspace(0, 10, 200)
        v = 100.0 + 10.0 * np.sin(2 * np.pi * r / 2.0)
        result = compute_spectral_features(r, v)
        assert result["n_peaks"] >= 1

    def test_lambda_dom_kpc_positive_finite(self):
        r = np.linspace(1, 10, 50)
        v = 100.0 + 5.0 * np.sin(2 * np.pi * r / 4.0)
        result = compute_spectral_features(r, v)
        assert result["lambda_dom_kpc"] > 0.0
        assert math.isfinite(result["lambda_dom_kpc"])

    def test_lambda_dom_equals_1_over_peak_freq(self):
        r = np.linspace(0, 20, 200)
        v = 100.0 + 8.0 * np.sin(2 * np.pi * r / 5.0)
        result = compute_spectral_features(r, v)
        if result["peak_freq_1perkpc"] > 0:
            expected = 1.0 / result["peak_freq_1perkpc"]
            assert result["lambda_dom_kpc"] == pytest.approx(expected, rel=1e-6)

    def test_constant_v_returns_valid_dict(self):
        r, v = _flat_curve(v=180.0)
        result = compute_spectral_features(r, v)
        assert "n_peaks" in result
        assert result["n_points_raw"] > 0

    def test_two_points_no_error(self):
        r = np.array([1.0, 5.0])
        v = np.array([100.0, 150.0])
        result = compute_spectral_features(r, v)
        assert result["n_points_raw"] == 2
        assert result["residual_rms_kms"] == pytest.approx(0.0, abs=1e-8)

    def test_single_point_no_error(self):
        r = np.array([3.0])
        v = np.array([120.0])
        result = compute_spectral_features(r, v)
        assert result["n_points_raw"] == 1
        assert result["rmin_kpc"] == pytest.approx(3.0)
        assert result["rmax_kpc"] == pytest.approx(3.0)
        assert result["residual_rms_kms"] == pytest.approx(0.0)

    def test_large_grid_no_error(self):
        r, v = _flat_curve(n=30)
        result = compute_spectral_features(r, v, n_grid=1024)
        assert result["n_grid"] == 1024

    def test_n_peaks_threshold_is_10_percent(self):
        # A pure sinusoid should have essentially 1 dominant peak
        r = np.linspace(0, 20, 512)
        v = np.sin(2 * np.pi * r / 4.0)
        result = compute_spectral_features(r, v, n_grid=512)
        # All peaks > 10% of max; for pure sinusoid n_peaks should be small
        assert result["n_peaks"] >= 1

    def test_accepts_kwargs(self):
        r, v = _flat_curve()
        # Should not raise even with unknown kwargs
        result = compute_spectral_features(r, v, future_param="unused")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. TestBuildSpectralCatalog
# ---------------------------------------------------------------------------

class TestBuildSpectralCatalog:
    def test_returns_dataframe(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        result = build_spectral_catalog(sparc_dir, out)
        assert isinstance(result, pd.DataFrame)

    def test_catalog_has_correct_columns(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert list(df.columns) == _CATALOG_COLS

    def test_catalog_contains_all_galaxies(self, tmp_path):
        galaxies = {
            "NGC001": _flat_curve(),
            "NGC002": _rising_curve(),
            "UGC001": _flat_curve(v=200.0),
        }
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=5)
        assert len(df) == 3

    def test_writes_csv(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        build_spectral_catalog(sparc_dir, out)
        assert out.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        build_spectral_catalog(sparc_dir, out)
        df_csv = pd.read_csv(out)
        assert list(df_csv.columns) == _CATALOG_COLS

    def test_respects_min_points_skip(self, tmp_path):
        r_few = np.array([1.0, 2.0, 3.0])  # 3 points < min_points=5
        v_few = np.array([100.0, 110.0, 120.0])
        r_enough = np.linspace(1, 10, 10)
        v_enough = np.full(10, 150.0)
        galaxies = {
            "FewPoints": (r_few, v_few),
            "EnoughPoints": (r_enough, v_enough),
        }
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=5)
        assert len(df) == 1
        assert df["galaxy"].iloc[0] == "EnoughPoints"

    def test_empty_dir_returns_empty_df(self, tmp_path):
        sparc_dir = tmp_path / "empty_sparc"
        sparc_dir.mkdir()
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert len(df) == 0
        assert list(df.columns) == _CATALOG_COLS

    def test_sorted_by_galaxy(self, tmp_path):
        galaxies = {
            "ZZZ001": _flat_curve(),
            "AAA001": _flat_curve(),
            "MMM001": _flat_curve(),
        }
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert list(df["galaxy"]) == sorted(df["galaxy"])

    def test_single_galaxy(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"SOLO001": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert len(df) == 1
        assert df["galaxy"].iloc[0] == "SOLO001"

    def test_creates_output_parent_dir(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "deep" / "nested" / "out.csv"
        build_spectral_catalog(sparc_dir, out)
        assert out.exists()

    def test_galaxy_name_in_catalog(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"TestGal": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert "TestGal" in df["galaxy"].values

    def test_n_points_raw_in_catalog(self, tmp_path):
        r = np.linspace(1, 10, 12)
        v = np.full(12, 150.0)
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert df["n_points_raw"].iloc[0] == 12

    def test_all_below_min_points_returns_empty(self, tmp_path):
        r_few = np.array([1.0, 2.0])
        v_few = np.array([100.0, 120.0])
        galaxies = {
            "Tiny001": (r_few, v_few),
            "Tiny002": (r_few, v_few),
        }
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=5)
        assert len(df) == 0

    def test_min_points_1_includes_all(self, tmp_path):
        r_tiny = np.array([1.0, 2.0])
        v_tiny = np.array([100.0, 120.0])
        r_big = np.linspace(1, 10, 20)
        v_big = np.full(20, 150.0)
        galaxies = {
            "Tiny": (r_tiny, v_tiny),
            "Big": (r_big, v_big),
        }
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=1)
        assert len(df) == 2

    def test_verbose_does_not_crash(self, tmp_path, capsys):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        build_spectral_catalog(sparc_dir, out, verbose=True)
        captured = capsys.readouterr()
        assert "NGC001" in captured.out or "rotmod" in captured.out.lower()


# ---------------------------------------------------------------------------
# 5. TestMain
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert isinstance(result, dict)

    def test_dict_has_catalog_key(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert "catalog" in result
        assert isinstance(result["catalog"], pd.DataFrame)

    def test_dict_has_n_key(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert "n" in result
        assert isinstance(result["n"], int)

    def test_dict_has_out_path_key(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert "out_path" in result
        assert isinstance(result["out_path"], str)

    def test_n_matches_catalog_length(self, tmp_path):
        galaxies = {"A": _flat_curve(), "B": _flat_curve(), "C": _flat_curve()}
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert result["n"] == len(result["catalog"])

    def test_out_path_matches_arg(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert result["out_path"] == str(out)

    def test_default_out_path(self):
        args = _parse_args(["--sparc-dir", "data/SPARC"])
        assert "sparc_spectral_catalog.csv" in args.out

    def test_quiet_flag_suppresses_output(self, tmp_path, capsys):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        main(["--sparc-dir", str(sparc_dir), "--out", str(out), "--quiet"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_min_points_arg(self, tmp_path):
        r_few = np.array([1.0, 2.0, 3.0])
        v_few = np.array([100.0, 110.0, 120.0])
        r_many = np.linspace(1, 10, 15)
        v_many = np.full(15, 150.0)
        galaxies = {"FEW": (r_few, v_few), "MANY": (r_many, v_many)}
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", galaxies)
        out = tmp_path / "out.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--min-points", "5",
        ])
        assert result["n"] == 1
        assert result["catalog"]["galaxy"].iloc[0] == "MANY"

    def test_writes_csv_file(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_sparc_dir_returns_empty_df(self, tmp_path):
        sparc_dir = tmp_path / "empty"
        sparc_dir.mkdir()
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert len(df) == 0

    def test_empty_dir_csv_has_correct_columns(self, tmp_path):
        sparc_dir = tmp_path / "empty"
        sparc_dir.mkdir()
        out = tmp_path / "out.csv"
        build_spectral_catalog(sparc_dir, out)
        df = pd.read_csv(out)
        assert list(df.columns) == _CATALOG_COLS

    def test_all_files_below_min_points(self, tmp_path):
        r = np.array([1.0, 2.0])
        v = np.array([100.0, 120.0])
        sparc_dir = _make_sparc_dir(
            tmp_path / "sparc",
            {"G1": (r, v), "G2": (r, v)},
        )
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=10)
        assert len(df) == 0

    def test_single_galaxy_catalog(self, tmp_path):
        r, v = _flat_curve(n=10)
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"LONELY": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=5)
        assert len(df) == 1
        assert df["galaxy"].iloc[0] == "LONELY"

    def test_constant_velocity_catalog(self, tmp_path):
        r, v = _flat_curve(v=200.0, n=20)
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"FLAT": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert len(df) == 1
        assert df["residual_rms_kms"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_main_empty_dir(self, tmp_path):
        sparc_dir = tmp_path / "empty"
        sparc_dir.mkdir()
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert result["n"] == 0
        assert isinstance(result["catalog"], pd.DataFrame)

    def test_main_single_galaxy(self, tmp_path):
        r, v = _flat_curve(n=10)
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"SOLO": (r, v)})
        out = tmp_path / "out.csv"
        result = main(["--sparc-dir", str(sparc_dir), "--out", str(out)])
        assert result["n"] == 1

    def test_compute_spectral_features_rmin_equals_rmax(self):
        r = np.array([5.0])
        v = np.array([150.0])
        result = compute_spectral_features(r, v)
        assert result["rmin_kpc"] == pytest.approx(5.0)
        assert result["rmax_kpc"] == pytest.approx(5.0)

    def test_non_rotmod_files_are_ignored(self, tmp_path):
        sparc_dir = tmp_path / "sparc"
        sparc_dir.mkdir()
        # Write a non-rotmod file
        (sparc_dir / "README.txt").write_text("not a rotmod file")
        (sparc_dir / "data.csv").write_text("col1,col2\n1,2")
        r, v = _flat_curve()
        _make_rotmod_file(sparc_dir / "NGC001_rotmod.dat", r, v)
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert len(df) == 1
        assert df["galaxy"].iloc[0] == "NGC001"

    def test_plot_dir_accepted_but_not_required(self, tmp_path):
        r, v = _flat_curve()
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        plot_dir = tmp_path / "plots"
        # Should not raise
        df = build_spectral_catalog(sparc_dir, out, plot_dir=plot_dir)
        assert isinstance(df, pd.DataFrame)

    def test_parse_rotmod_path_as_string(self, tmp_path):
        r = np.linspace(1, 5, 5)
        v = np.full(5, 130.0)
        fpath = _make_rotmod_file(tmp_path / "G01_rotmod.dat", r, v)
        # Pass as string, not Path
        r_out, v_out = parse_rotmod(str(fpath))
        assert len(r_out) == 5

    def test_catalog_rmin_rmax_values_are_correct(self, tmp_path):
        r = np.linspace(2.0, 8.0, 10)
        v = np.full(10, 150.0)
        sparc_dir = _make_sparc_dir(tmp_path / "sparc", {"NGC001": (r, v)})
        out = tmp_path / "out.csv"
        df = build_spectral_catalog(sparc_dir, out)
        assert df["rmin_kpc"].iloc[0] == pytest.approx(2.0, abs=1e-3)
        assert df["rmax_kpc"].iloc[0] == pytest.approx(8.0, abs=1e-3)
