"""
tests/test_compute_f3_catalog.py — Tests for scripts/compute_f3_catalog.py.

Covers:
  - power-law rotation curve with known slope
  - CLI invocation with --outer-fracs
  - multi-threshold (long-format) output
  - physical filter (r<=0, Vobs<=0 rows discarded)
  - fallback to last-3-points when outer region has fewer than 3 points
  - empty directory handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.compute_f3_catalog import (
    compute_f3_for_file,
    process_dirs,
    main,
    _ols_slope_err_r2,
    _read_r_vobs_any_format,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rotmod(path: Path, r: np.ndarray, v: np.ndarray) -> None:
    """Write a minimal two-column *_rotmod.dat file."""
    with open(path, "w") as fh:
        for ri, vi in zip(r, v):
            fh.write(f"{ri:.6f} {vi:.6f}\n")


def _power_law_curve(n: int = 30, slope: float = 0.0, v0: float = 150.0) -> tuple:
    """Return (r, v) where v = v0 * r^slope (log-log slope = slope)."""
    r = np.linspace(1.0, 20.0, n)
    v = v0 * r ** slope
    return r, v


# ---------------------------------------------------------------------------
# _ols_slope_err_r2
# ---------------------------------------------------------------------------

class TestOlsSlopeErrR2:
    def test_flat_curve_slope_zero(self):
        x = np.linspace(0, 1, 10)
        y = np.ones(10) * 5.0
        slope, se, r2 = _ols_slope_err_r2(x, y)
        assert slope == pytest.approx(0.0, abs=1e-10)

    def test_known_slope(self):
        x = np.linspace(0.0, 1.0, 50)
        y = 2.0 * x + 1.0
        slope, se, r2 = _ols_slope_err_r2(x, y)
        assert slope == pytest.approx(2.0, abs=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-8)

    def test_too_few_points_returns_nan(self):
        x = np.array([1.0])
        y = np.array([2.0])
        slope, se, r2 = _ols_slope_err_r2(x, y)
        assert np.isnan(slope)
        assert np.isnan(se)
        assert np.isnan(r2)


# ---------------------------------------------------------------------------
# _read_r_vobs_any_format
# ---------------------------------------------------------------------------

class TestReadRVobs:
    def test_reads_two_columns(self, tmp_path):
        path = tmp_path / "NGC0000_rotmod.dat"
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([100.0, 110.0, 120.0])
        _write_rotmod(path, r, v)
        r_out, v_out = _read_r_vobs_any_format(path)
        np.testing.assert_allclose(r_out, r)
        np.testing.assert_allclose(v_out, v)

    def test_skips_comment_lines(self, tmp_path):
        path = tmp_path / "NGC0001_rotmod.dat"
        path.write_text("# header comment\n1.0 100.0\n2.0 110.0\n3.0 120.0\n")
        r_out, v_out = _read_r_vobs_any_format(path)
        assert len(r_out) == 3

    def test_ignores_extra_columns(self, tmp_path):
        path = tmp_path / "NGC0002_rotmod.dat"
        path.write_text("1.0 100.0 45.0 90.0\n2.0 110.0 50.0 95.0\n3.0 120.0 55.0 100.0\n")
        r_out, v_out = _read_r_vobs_any_format(path)
        np.testing.assert_allclose(r_out, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(v_out, [100.0, 110.0, 120.0])

    def test_raises_if_fewer_than_two_columns(self, tmp_path):
        path = tmp_path / "bad_rotmod.dat"
        path.write_text("1.0\n2.0\n3.0\n")
        with pytest.raises(ValueError, match="Expected >=2 columns"):
            _read_r_vobs_any_format(path)


# ---------------------------------------------------------------------------
# compute_f3_for_file — power-law (known slope)
# ---------------------------------------------------------------------------

class TestComputeF3ForFile:
    def test_flat_curve_slope_near_zero(self, tmp_path):
        """Flat rotation curve → log-log slope ≈ 0."""
        path = tmp_path / "NGC_flat_rotmod.dat"
        r, v = _power_law_curve(slope=0.0)
        _write_rotmod(path, r, v)
        row = compute_f3_for_file(path, source="SPARC", outer_frac=0.7)
        assert row.status in ("ok", "warn")
        assert row.F3_SCM == pytest.approx(0.0, abs=0.05)

    def test_rising_curve_positive_slope(self, tmp_path):
        """Rising power-law v ~ r^0.5 → F3 ≈ 0.5."""
        path = tmp_path / "NGC_rise_rotmod.dat"
        r, v = _power_law_curve(slope=0.5, n=50)
        _write_rotmod(path, r, v)
        row = compute_f3_for_file(path, source="SPARC", outer_frac=0.7)
        assert row.status in ("ok", "warn")
        assert row.F3_SCM == pytest.approx(0.5, abs=0.05)

    def test_physical_filter_removes_nonphysical(self, tmp_path):
        """Rows with r<=0 or v<=0 must be discarded."""
        path = tmp_path / "NGC_bad_rotmod.dat"
        r = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        v = np.array([100.0, -5.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        _write_rotmod(path, r, v)
        row = compute_f3_for_file(path, source="SPARC", outer_frac=0.7)
        # n_all should reflect only physical points (8 valid points)
        assert row.n_all == 8
        assert row.status != "fail"

    def test_too_few_points_returns_fail(self, tmp_path):
        """File with only 2 physical points → status=fail."""
        path = tmp_path / "NGC_tiny_rotmod.dat"
        _write_rotmod(path, np.array([1.0, 2.0]), np.array([100.0, 110.0]))
        row = compute_f3_for_file(path, source="SPARC", outer_frac=0.7)
        assert row.status == "fail"
        assert np.isnan(row.F3_SCM)

    def test_fallback_last3_when_outer_too_sparse(self, tmp_path):
        """If outer region < 3 points, fallback to last-3 and note it."""
        path = tmp_path / "NGC_sparse_rotmod.dat"
        # 10 points but frac=0.99 will give only 1 outer point
        r = np.linspace(1.0, 10.0, 10)
        v = np.ones(10) * 150.0
        _write_rotmod(path, r, v)
        row = compute_f3_for_file(path, source="SPARC", outer_frac=0.99)
        assert "fallback_last3" in row.note
        assert row.n_used == 3

    def test_read_error_returns_fail_row(self, tmp_path):
        """Non-existent file → status=fail with read_error note."""
        path = tmp_path / "NOEXIST_rotmod.dat"
        row = compute_f3_for_file(path, source="SPARC", outer_frac=0.7)
        assert row.status == "fail"
        assert "read_error" in row.note


# ---------------------------------------------------------------------------
# process_dirs — multi-threshold (long format)
# ---------------------------------------------------------------------------

class TestProcessDirs:
    def _make_sparc_dir(self, tmp_path: Path, n_galaxies: int = 3) -> Path:
        d = tmp_path / "sparc"
        d.mkdir()
        for i in range(n_galaxies):
            path = d / f"NGC{i:04d}_rotmod.dat"
            r, v = _power_law_curve(slope=0.0, n=20)
            _write_rotmod(path, r, v)
        return d

    def test_long_format_rows_per_galaxy_per_frac(self, tmp_path):
        """Output has n_galaxies × n_fracs rows."""
        d = self._make_sparc_dir(tmp_path, n_galaxies=3)
        fracs = [0.6, 0.7, 0.8]
        df = process_dirs(sparc_dir=d, lt_dir=None, outer_fracs=fracs)
        assert len(df) == 3 * 3  # 3 galaxies × 3 thresholds

    def test_output_columns_present(self, tmp_path):
        d = self._make_sparc_dir(tmp_path)
        df = process_dirs(sparc_dir=d, lt_dir=None, outer_fracs=[0.7])
        expected = [
            "source", "galaxy", "outer_frac",
            "F3_SCM", "F3_SCM_err", "R2",
            "n_all", "n_used", "rmin_used_kpc", "rmax_kpc",
            "status", "note", "file",
        ]
        assert df.columns.tolist() == expected

    def test_sorted_by_source_galaxy_frac(self, tmp_path):
        d = self._make_sparc_dir(tmp_path, n_galaxies=4)
        fracs = [0.8, 0.6, 0.7]
        df = process_dirs(sparc_dir=d, lt_dir=None, outer_fracs=fracs)
        # outer_frac must be sorted ascending within each galaxy
        for _, grp in df.groupby("galaxy"):
            assert list(grp["outer_frac"]) == sorted(grp["outer_frac"].tolist())

    def test_empty_dir_returns_empty_df(self, tmp_path):
        d = tmp_path / "empty_sparc"
        d.mkdir()
        df = process_dirs(sparc_dir=d, lt_dir=None, outer_fracs=[0.7])
        assert df.empty

    def test_nonexistent_dir_returns_empty_df(self, tmp_path):
        d = tmp_path / "nonexistent"
        df = process_dirs(sparc_dir=d, lt_dir=None, outer_fracs=[0.7])
        assert df.empty

    def test_lt_source_label(self, tmp_path):
        lt_dir = tmp_path / "lt"
        lt_dir.mkdir()
        path = lt_dir / "WLM_rotmod.dat"
        r, v = _power_law_curve(slope=0.0, n=20)
        _write_rotmod(path, r, v)
        df = process_dirs(sparc_dir=None, lt_dir=lt_dir, outer_fracs=[0.7])
        assert (df["source"] == "LITTLE_THINGS").all()


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestCLI:
    def _make_sparc_dir(self, tmp_path: Path, n_galaxies: int = 5) -> Path:
        d = tmp_path / "sparc"
        d.mkdir()
        for i in range(n_galaxies):
            path = d / f"G{i:04d}_rotmod.dat"
            r, v = _power_law_curve(slope=0.0, n=20)
            _write_rotmod(path, r, v)
        return d

    def test_cli_creates_csv(self, tmp_path):
        d = self._make_sparc_dir(tmp_path)
        out = tmp_path / "out" / "f3.csv"
        rc = main(["--sparc-dir", str(d), "--out", str(out), "--outer-fracs", "0.7"])
        assert out.exists()
        assert rc == 0

    def test_cli_multi_threshold(self, tmp_path):
        d = self._make_sparc_dir(tmp_path, n_galaxies=2)
        out = tmp_path / "out2" / "f3.csv"
        main(["--sparc-dir", str(d), "--out", str(out), "--outer-fracs", "0.6", "0.7", "0.8"])
        df = pd.read_csv(out)
        # 2 galaxies × 3 thresholds
        assert len(df) == 2 * 3
        assert set(df["outer_frac"].unique()) == {0.6, 0.7, 0.8}

    def test_cli_no_dirs_returns_empty_csv(self, tmp_path):
        out = tmp_path / "empty.csv"
        rc = main(["--out", str(out)])
        assert out.exists()
        # Empty output should still have column headers
        df = pd.read_csv(out)
        assert df.empty
        assert "source" in df.columns
        assert rc == 2

    def test_cli_default_outer_frac(self, tmp_path):
        d = self._make_sparc_dir(tmp_path, n_galaxies=2)
        out = tmp_path / "default_frac.csv"
        main(["--sparc-dir", str(d), "--out", str(out)])
        df = pd.read_csv(out)
        assert list(df["outer_frac"].unique()) == [0.7]
