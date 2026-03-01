"""
tests/test_compute_f3_catalog.py — Unit and integration tests for
scripts/compute_f3_catalog.py.

Tests cover:
  - compute_f3: correct slope, edge cases (too few points, no data, neg vel)
  - build_catalog: correct columns, multi-frac, skip behaviour
  - CLI main: produces output file, honours --outer-fracs
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compute_f3_catalog import (
    CATALOG_COLS,
    MIN_POINTS_DEFAULT,
    OUTER_FRACS_DEFAULT,
    build_catalog,
    compute_f3,
    main,
    _discover_files,
    _load_sparc,
    _load_lt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sparc_dir(tmp_path):
    """Minimal synthetic SPARC rotmod dataset (2 galaxies)."""
    for name, v_flat in [("AAA0001", 150.0), ("BBB0002", 200.0)]:
        r = np.linspace(0.5, 10.0, 15)
        v = np.full(15, v_flat) + np.linspace(0, 5, 15)
        rows = np.column_stack([
            r,
            v,
            np.full(15, 5.0),   # v_obs_err
            0.3 * v,            # v_gas
            0.8 * v,            # v_disk
            np.zeros(15),       # v_bul
            np.zeros(15),       # SBdisk
            np.zeros(15),       # SBbul
        ])
        np.savetxt(tmp_path / f"{name}_rotmod.dat", rows)
    return tmp_path


@pytest.fixture()
def lt_dir(tmp_path):
    """Minimal synthetic LITTLE THINGS rot.csv dataset (2 galaxies)."""
    for name, v_flat in [("DDO001", 30.0), ("DDO002", 45.0)]:
        r = np.linspace(0.2, 3.0, 10)
        v = np.full(10, v_flat) + np.linspace(0, 3, 10)
        df = pd.DataFrame({"r_kpc": r, "Vbary_kms": v})
        df.to_csv(tmp_path / f"{name}_rot.csv", index=False)
    return tmp_path


@pytest.fixture()
def mixed_dir(tmp_path):
    """Dataset with both SPARC and LITTLE THINGS files."""
    # One SPARC file
    r = np.linspace(0.5, 8.0, 12)
    v = np.full(12, 120.0)
    rows = np.column_stack([r, v, np.full(12, 5.0), 0.3 * v, 0.8 * v,
                             np.zeros(12), np.zeros(12), np.zeros(12)])
    np.savetxt(tmp_path / "NGC1111_rotmod.dat", rows)

    # One LITTLE THINGS file
    sub = tmp_path / "lt_oh2015"
    sub.mkdir()
    r2 = np.linspace(0.3, 4.0, 8)
    v2 = np.full(8, 55.0)
    df = pd.DataFrame({"r_kpc": r2, "Vbary_kms": v2})
    df.to_csv(sub / "DDO555_rot.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# _discover_files
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_finds_sparc_files(self, sparc_dir):
        files = _discover_files(sparc_dir)
        sources = [s for s, _, _ in files]
        assert "SPARC" in sources

    def test_finds_lt_files(self, lt_dir):
        files = _discover_files(lt_dir)
        sources = [s for s, _, _ in files]
        assert "LT_OH2015" in sources

    def test_galaxy_name_extraction_sparc(self, sparc_dir):
        files = _discover_files(sparc_dir)
        galaxies = {g for s, g, _ in files if s == "SPARC"}
        assert "AAA0001" in galaxies

    def test_galaxy_name_extraction_lt(self, lt_dir):
        files = _discover_files(lt_dir)
        galaxies = {g for _, g, _ in files}
        assert "DDO001" in galaxies

    def test_empty_dir_returns_empty(self, tmp_path):
        assert _discover_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestLoadSparc:
    def test_returns_two_arrays(self, sparc_dir):
        path = next(sparc_dir.glob("*_rotmod.dat"))
        r, v = _load_sparc(path)
        assert isinstance(r, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert len(r) == len(v) == 15

    def test_r_positive(self, sparc_dir):
        path = next(sparc_dir.glob("*_rotmod.dat"))
        r, _ = _load_sparc(path)
        assert (r > 0).all()


class TestLoadLT:
    def test_returns_two_arrays(self, lt_dir):
        path = next(lt_dir.glob("*_rot.csv"))
        r, v = _load_lt(path)
        assert isinstance(r, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert len(r) == len(v) == 10


# ---------------------------------------------------------------------------
# compute_f3
# ---------------------------------------------------------------------------

class TestComputeF3:
    def test_flat_curve_slope_near_zero(self):
        """Perfectly flat rotation curve → slope ≈ 0."""
        r = np.linspace(1.0, 10.0, 20)
        v = np.full(20, 100.0)
        result = compute_f3(r, v, outer_frac=0.7)
        assert result["status"] == "ok"
        assert abs(result["F3_SCM"]) < 1e-6

    def test_power_law_curve_recovers_slope(self):
        """V ∝ r^α → F3_SCM should recover α exactly."""
        r = np.linspace(0.5, 10.0, 50)
        alpha = 0.5
        v = 50.0 * (r ** alpha)
        result = compute_f3(r, v, outer_frac=0.6)
        assert result["status"] == "ok"
        assert abs(result["F3_SCM"] - alpha) < 1e-6

    def test_r2_unity_for_exact_power_law(self):
        """Exact power law → R² = 1."""
        r = np.linspace(1.0, 8.0, 30)
        v = 80.0 * (r ** 0.3)
        result = compute_f3(r, v, outer_frac=0.7)
        assert abs(result["R2"] - 1.0) < 1e-10

    def test_skip_few_points(self):
        """Fewer outer points than min_points → skip_few_points status."""
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([80.0, 90.0, 95.0, 98.0, 100.0])
        # outer_frac=0.99 → only 1 point qualifies (r=5.0)
        result = compute_f3(r, v, outer_frac=0.99, min_points=3)
        assert result["status"] in ("skip_few_points", "skip_no_outer_points")

    def test_skip_empty_curve(self):
        """Empty arrays → skip_no_data."""
        result = compute_f3(np.array([]), np.array([]), outer_frac=0.7)
        assert result["status"] == "skip_no_data"

    def test_skip_negative_velocities(self):
        """All outer velocities <= 0 → skip_no_valid_points."""
        r = np.linspace(1.0, 5.0, 10)
        v = np.full(10, -10.0)
        result = compute_f3(r, v, outer_frac=0.6, min_points=2)
        assert result["status"] == "skip_no_valid_points"

    def test_n_all_and_n_used(self):
        """n_all = total points; n_used = outer valid points."""
        r = np.linspace(1.0, 10.0, 20)
        v = np.full(20, 100.0)
        result = compute_f3(r, v, outer_frac=0.7)
        assert result["n_all"] == 20
        # Points with r >= 0.7 * 10 = 7.0 — approximately 6 points
        assert 1 <= result["n_used"] <= 20

    def test_f3_err_non_negative(self):
        """Standard error of slope must be non-negative."""
        r = np.linspace(1.0, 10.0, 20)
        v = 80.0 * (r ** 0.2) + np.random.default_rng(0).normal(0, 1, 20)
        result = compute_f3(r, v, outer_frac=0.6)
        if result["status"] == "ok":
            assert result["F3_SCM_err"] >= 0.0


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    def test_output_columns(self, sparc_dir):
        df = build_catalog(sparc_dir)
        assert list(df.columns) == CATALOG_COLS

    def test_row_count_single_frac(self, sparc_dir):
        """Two galaxies × 1 frac = 2 rows."""
        df = build_catalog(sparc_dir, outer_fracs=[0.7])
        assert len(df) == 2

    def test_row_count_multi_frac(self, sparc_dir):
        """Two galaxies × 3 fracs = 6 rows."""
        df = build_catalog(sparc_dir, outer_fracs=[0.6, 0.7, 0.8])
        assert len(df) == 6

    def test_outer_frac_column_values(self, sparc_dir):
        fracs = [0.6, 0.8]
        df = build_catalog(sparc_dir, outer_fracs=fracs)
        assert set(df["outer_frac"].unique()) == set(fracs)

    def test_source_label_sparc(self, sparc_dir):
        df = build_catalog(sparc_dir)
        assert (df["source"] == "SPARC").all()

    def test_source_label_lt(self, lt_dir):
        df = build_catalog(lt_dir)
        assert (df["source"] == "LT_OH2015").all()

    def test_mixed_sources(self, mixed_dir):
        df = build_catalog(mixed_dir)
        assert set(df["source"].unique()) == {"SPARC", "LT_OH2015"}

    def test_status_ok_for_valid_data(self, sparc_dir):
        df = build_catalog(sparc_dir, outer_fracs=[0.7])
        assert (df["status"] == "ok").all(), df[df["status"] != "ok"]

    def test_file_column_non_empty(self, sparc_dir):
        df = build_catalog(sparc_dir)
        assert df["file"].notna().all()
        assert (df["file"] != "").all()

    def test_empty_dir_returns_empty_df(self, tmp_path):
        df = build_catalog(tmp_path)
        assert len(df) == 0
        assert list(df.columns) == CATALOG_COLS

    def test_f3_finite_for_flat_curve(self, sparc_dir):
        """Flat rotation curves should produce finite F3_SCM."""
        df = build_catalog(sparc_dir)
        ok_rows = df[df["status"] == "ok"]
        assert ok_rows["F3_SCM"].notna().all()
        assert np.isfinite(ok_rows["F3_SCM"].values).all()

    def test_real_lt_data(self):
        """Smoke test against the actual LITTLE THINGS rotmod files."""
        real_dir = Path(__file__).parent.parent / "data" / "raw" / "lt_oh2015"
        if not real_dir.exists():
            pytest.skip("lt_oh2015 directory not found")
        df = build_catalog(real_dir, outer_fracs=[0.7])
        assert len(df) >= 1
        assert list(df.columns) == CATALOG_COLS


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

class TestCLIMain:
    def test_main_creates_output_file(self, sparc_dir, tmp_path):
        out = tmp_path / "catalog.csv"
        main([
            "--data-dir", str(sparc_dir),
            "--out", str(out),
        ])
        assert out.exists()

    def test_main_output_readable(self, sparc_dir, tmp_path):
        out = tmp_path / "catalog.csv"
        main([
            "--data-dir", str(sparc_dir),
            "--out", str(out),
        ])
        df = pd.read_csv(out)
        assert list(df.columns) == CATALOG_COLS

    def test_main_multi_fracs(self, sparc_dir, tmp_path):
        out = tmp_path / "catalog_multi.csv"
        main([
            "--data-dir", str(sparc_dir),
            "--out", str(out),
            "--outer-fracs", "0.6", "0.7", "0.8",
        ])
        df = pd.read_csv(out)
        assert set(df["outer_frac"].unique()) == {0.6, 0.7, 0.8}

    def test_main_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main([
                "--data-dir", str(tmp_path / "nonexistent"),
                "--out", str(tmp_path / "out.csv"),
            ])

    def test_main_creates_parent_dirs(self, sparc_dir, tmp_path):
        out = tmp_path / "nested" / "deep" / "catalog.csv"
        main([
            "--data-dir", str(sparc_dir),
            "--out", str(out),
        ])
        assert out.exists()
