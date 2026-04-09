"""
tests/test_sparc_slope_tail.py — Tests for scripts/sparc_slope_tail.py.

Uses synthetic rotmod files with known slope_tail values so the suite runs
without any real SPARC download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.sparc_slope_tail import (
    compute_slope_tail,
    process_galaxy,
    process_directory,
    main,
    TAIL_FRAC_DEFAULT,
    MIN_TAIL_POINTS,
    DATA_DIR_DEFAULT,
    OUTPUT_CSV_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rotmod(path: Path, r: np.ndarray, v: np.ndarray) -> None:
    """Write a minimal rotmod.dat file (space-separated, no header)."""
    # Additional required columns filled with zeros (v_obs_err, v_gas, v_disk,
    # v_bul, SBdisk, SBbul)
    n = len(r)
    zeros = np.zeros(n)
    data = np.column_stack([r, v, zeros, zeros, zeros, zeros, zeros, zeros])
    np.savetxt(path, data)


def _make_rotmod_dir(tmp_path: Path, n_galaxies: int = 5,
                     n_pts: int = 30, planted_slope: float = 0.0,
                     rng_seed: int = 42) -> Path:
    """Create a synthetic rotmod directory with a known slope_tail.

    Velocities follow V(r) = V0 * r^planted_slope so that log-log slope
    equals planted_slope in the outer tail.
    """
    rng = np.random.default_rng(rng_seed)
    root = tmp_path / "rotmod"
    root.mkdir()
    for i in range(n_galaxies):
        name = f"SYN{i:03d}"
        r = np.linspace(1.0, 20.0, n_pts)
        v0 = rng.uniform(100.0, 300.0)
        noise = rng.normal(0, 0.001 * v0, n_pts)
        v = v0 * (r ** planted_slope) + noise
        _write_rotmod(root / f"{name}_rotmod.dat", r, v)
    return root


# ---------------------------------------------------------------------------
# Unit tests: compute_slope_tail
# ---------------------------------------------------------------------------

class TestComputeSlopeTail:
    def test_flat_curve_slope_near_zero(self):
        """Flat rotation curve → slope ≈ 0."""
        r = np.linspace(1.0, 20.0, 100)
        v = np.full(100, 150.0)
        slope = compute_slope_tail(r, v)
        assert slope == pytest.approx(0.0, abs=1e-6)

    def test_known_positive_slope(self):
        """V ∝ r^0.5 → slope = 0.5 exactly."""
        r = np.logspace(0, 1.3, 200)
        v = 100.0 * r ** 0.5
        slope = compute_slope_tail(r, v)
        assert slope == pytest.approx(0.5, abs=1e-8)

    def test_known_negative_slope(self):
        """V ∝ r^(-0.2) → slope = -0.2 exactly."""
        r = np.logspace(0, 1.3, 200)
        v = 100.0 * r ** (-0.2)
        slope = compute_slope_tail(r, v)
        assert slope == pytest.approx(-0.2, abs=1e-8)

    def test_returns_float(self):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([100.0, 100.0, 100.0])
        result = compute_slope_tail(r, v)
        assert isinstance(result, float)

    def test_two_points_gives_exact_slope(self):
        """With exactly two points the log-log slope is uniquely determined."""
        r = np.array([5.0, 10.0])
        v = np.array([200.0, 100.0])   # V halves as r doubles → slope = -1
        slope = compute_slope_tail(r, v)
        assert slope == pytest.approx(-1.0, abs=1e-8)

    def test_single_galaxy_regression(self):
        """Regression guard: planted slope of -0.15 recovered within tolerance."""
        rng = np.random.default_rng(0)
        r = np.linspace(5.0, 20.0, 50)
        v = 150.0 * (r ** -0.15) + rng.normal(0, 0.1, 50)
        slope = compute_slope_tail(r, v)
        assert slope == pytest.approx(-0.15, abs=0.02)


# ---------------------------------------------------------------------------
# Unit tests: process_galaxy
# ---------------------------------------------------------------------------

class TestProcessGalaxy:
    def test_returns_float_for_valid_file(self, tmp_path):
        r = np.linspace(1.0, 20.0, 30)
        v = np.full(30, 150.0)
        p = tmp_path / "G001_rotmod.dat"
        _write_rotmod(p, r, v)
        result = process_galaxy(p)
        assert isinstance(result, float)

    def test_flat_curve_returns_near_zero(self, tmp_path):
        r = np.linspace(1.0, 20.0, 40)
        v = np.full(40, 200.0)
        p = tmp_path / "G002_rotmod.dat"
        _write_rotmod(p, r, v)
        result = process_galaxy(p)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_uses_outer_tail_only(self, tmp_path):
        """Slope computed from r >= 0.7*r_max, not the full curve."""
        r = np.linspace(1.0, 10.0, 100)
        # inner 70%: rising, outer 30%: flat
        v = np.where(r < 7.0, 100.0 * r / 7.0, 100.0)
        p = tmp_path / "G003_rotmod.dat"
        _write_rotmod(p, r, v)
        result = process_galaxy(p)
        assert result == pytest.approx(0.0, abs=0.05)

    def test_returns_none_when_too_few_tail_points(self, tmp_path):
        """Only 3 tail points with min_points=4 → None."""
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.full(5, 100.0)
        p = tmp_path / "G004_rotmod.dat"
        _write_rotmod(p, r, v)
        # tail_frac=0.9 → only r >= 4.5 qualifies → 1 point
        result = process_galaxy(p, tail_frac=0.9, min_points=4)
        assert result is None

    def test_custom_tail_frac(self, tmp_path):
        """tail_frac parameter is respected."""
        rng = np.random.default_rng(11)
        r = np.linspace(1.0, 20.0, 60)
        v = 100.0 * r ** (-0.1) + rng.normal(0, 0.05, 60)
        p = tmp_path / "G005_rotmod.dat"
        _write_rotmod(p, r, v)
        result = process_galaxy(p, tail_frac=0.5)
        assert result is not None
        assert isinstance(result, float)

    def test_min_points_parameter(self, tmp_path):
        """min_points=2 allows files with small tails."""
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.full(5, 100.0)
        p = tmp_path / "G006_rotmod.dat"
        _write_rotmod(p, r, v)
        # tail_frac=0.9 → r >= 4.5 → 1 point → still None (need >= 2)
        assert process_galaxy(p, tail_frac=0.9, min_points=2) is None
        # tail_frac=0.7 → r >= 3.5 → 2 points → valid with min_points=2
        result = process_galaxy(p, tail_frac=0.7, min_points=2)
        assert result is not None

    def test_skips_zero_r_rows(self, tmp_path):
        """Rows with r <= 0 or v <= 0 are filtered before tail selection."""
        # Use enough points so that after filtering the tail still has >= 4
        r = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        v = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0, 100.0])
        p = tmp_path / "G007_rotmod.dat"
        _write_rotmod(p, r, v)
        # After filtering r=0/v=0: r=[1..10], r_max=10, tail r>=7 → 4 points ✓
        result = process_galaxy(p)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_planted_slope_recovered(self, tmp_path):
        """Planted slope of -0.2 recovered within tolerance."""
        rng = np.random.default_rng(3)
        r = np.linspace(1.0, 20.0, 80)
        v = 200.0 * (r ** -0.2) + rng.normal(0, 0.2, 80)
        p = tmp_path / "G008_rotmod.dat"
        _write_rotmod(p, r, v)
        result = process_galaxy(p)
        assert result == pytest.approx(-0.2, abs=0.05)


# ---------------------------------------------------------------------------
# Unit tests: process_directory
# ---------------------------------------------------------------------------

class TestProcessDirectory:
    def test_creates_csv(self, tmp_path):
        root = _make_rotmod_dir(tmp_path)
        out = tmp_path / "out" / "slope_tail.csv"
        process_directory(root, out, verbose=False)
        assert out.exists()

    def test_returns_dataframe(self, tmp_path):
        root = _make_rotmod_dir(tmp_path)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert list(df.columns) == ["galaxy", "slope_tail"]

    def test_n_galaxies_matches(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=5)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert len(df) == 5

    def test_ignores_non_rotmod_files(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3)
        # Add a file that should be ignored
        (root / "README.txt").write_text("ignore me")
        (root / "galaxy_table.csv").write_text("also ignore")
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert len(df) == 3

    def test_flat_curve_slope_near_zero(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=4, planted_slope=0.0)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert (df["slope_tail"].abs() < 0.05).all()

    def test_negative_planted_slope(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=4, planted_slope=-0.15,
                                n_pts=80)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert (df["slope_tail"] < 0).all()

    def test_positive_planted_slope(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3, planted_slope=0.3,
                                n_pts=80)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert (df["slope_tail"] > 0).all()

    def test_skips_galaxies_with_few_tail_points(self, tmp_path):
        """Galaxies with too few tail points are excluded from output."""
        root = tmp_path / "rotmod"
        root.mkdir()
        # Galaxy with enough points
        r_ok = np.linspace(1.0, 20.0, 30)
        v_ok = np.full(30, 150.0)
        _write_rotmod(root / "OK001_rotmod.dat", r_ok, v_ok)
        # Galaxy with only 2 points total → 0 tail points (tail_frac=0.7)
        r_few = np.array([1.0, 2.0])
        v_few = np.array([100.0, 100.0])
        _write_rotmod(root / "FEW001_rotmod.dat", r_few, v_few)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, tail_frac=0.7, min_points=4,
                               verbose=False)
        assert len(df) == 1
        assert df.iloc[0]["galaxy"] == "OK001"

    def test_creates_output_directory(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=2)
        out = tmp_path / "new_dir" / "subdir" / "slope_tail.csv"
        process_directory(root, out, verbose=False)
        assert out.exists()

    def test_csv_roundtrip(self, tmp_path):
        """Written CSV can be read back and matches the returned DataFrame."""
        root = _make_rotmod_dir(tmp_path, n_galaxies=4)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        df_read = pd.read_csv(out)
        pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                      df_read.reset_index(drop=True))

    def test_galaxy_names_match_filenames(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3)
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        expected = {f"SYN{i:03d}" for i in range(3)}
        assert set(df["galaxy"]) == expected

    def test_empty_dir_produces_empty_df(self, tmp_path):
        root = tmp_path / "empty_rotmod"
        root.mkdir()
        out = tmp_path / "slope_tail.csv"
        df = process_directory(root, out, verbose=False)
        assert len(df) == 0
        assert list(df.columns) == ["galaxy", "slope_tail"]

    def test_custom_tail_frac_parameter(self, tmp_path):
        """Changing tail_frac changes the slope estimate."""
        root = tmp_path / "rotmod"
        root.mkdir()
        # Curve: rising inner part, declining outer part
        r = np.linspace(1.0, 20.0, 100)
        v = np.where(r < 10.0, 100.0 + 5.0 * r, 250.0 - 5.0 * (r - 10.0))
        v = np.clip(v, 1.0, None)
        _write_rotmod(root / "G001_rotmod.dat", r, v)
        out1 = tmp_path / "s1.csv"
        out2 = tmp_path / "s2.csv"
        df1 = process_directory(root, out1, tail_frac=0.5, min_points=4,
                                verbose=False)
        df2 = process_directory(root, out2, tail_frac=0.9, min_points=4,
                                verbose=False)
        # Different fractions should give different slopes
        assert df1.iloc[0]["slope_tail"] != pytest.approx(
            df2.iloc[0]["slope_tail"], abs=1e-6)


# ---------------------------------------------------------------------------
# CLI tests: main
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        assert isinstance(result, dict)

    def test_dict_keys(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=2)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        assert {"df", "output_csv", "n_galaxies"}.issubset(result.keys())

    def test_n_galaxies_in_result(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=4)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        assert result["n_galaxies"] == 4

    def test_output_csv_written(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3)
        out = tmp_path / "slope_tail.csv"
        main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        assert out.exists()

    def test_output_csv_path_in_result(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=2)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        assert result["output_csv"] == str(out)

    def test_custom_tail_frac_cli(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out),
                       "--tail-frac", "0.8", "--quiet"])
        assert result["n_galaxies"] > 0

    def test_custom_min_points_cli(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=3, n_pts=50)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out),
                       "--min-points", "2", "--quiet"])
        assert result["n_galaxies"] == 3

    def test_df_in_result_is_dataframe(self, tmp_path):
        root = _make_rotmod_dir(tmp_path, n_galaxies=2)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        assert isinstance(result["df"], pd.DataFrame)

    def test_regression_slope_range(self, tmp_path):
        """Flat curves produce slopes very close to zero."""
        root = _make_rotmod_dir(tmp_path, n_galaxies=5, planted_slope=0.0,
                                n_pts=60)
        out = tmp_path / "slope_tail.csv"
        result = main(["--data-dir", str(root), "--out", str(out), "--quiet"])
        df = result["df"]
        assert (df["slope_tail"].abs() < 0.05).all()


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_tail_frac_default_value(self):
        assert TAIL_FRAC_DEFAULT == pytest.approx(0.7)

    def test_min_tail_points_value(self):
        assert MIN_TAIL_POINTS == 4

    def test_data_dir_default_is_string(self):
        assert isinstance(DATA_DIR_DEFAULT, str)

    def test_output_csv_default_is_string(self):
        assert isinstance(OUTPUT_CSV_DEFAULT, str)
