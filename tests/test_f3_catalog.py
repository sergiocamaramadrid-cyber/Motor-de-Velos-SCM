"""
tests/test_f3_catalog.py — Tests for generate_f3_catalog.py and f3_catalog_analysis.py.

Covers:
  - Unit tests for compute_galaxy_beta()
  - Unit tests for build_catalog()
  - Unit tests for analyze_catalog()
  - CLI tests for generate_f3_catalog.main()
  - CLI tests for f3_catalog_analysis.main()
  - End-to-end pipeline validation
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress as _linregress

from scripts.generate_f3_catalog import (
    A0_DEFAULT,
    BASE_60,
    DEEP_THRESHOLD_DEFAULT,
    EXPECTED_SLOPE,
    R0_KPC_DEFAULT,
    build_catalog,
    compute_galaxy_beta,
    hierarchy_level,
    main as generate_main,
)
from scripts.f3_catalog_analysis import (
    CATALOG_DEFAULT,
    analyze_catalog,
    main as analysis_main,
    print_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mond_per_point_csv(
    tmp_path: Path,
    n_galaxies: int = 5,
    n_deep: int = 30,
    n_shallow: int = 20,
    planted_slope: float = 0.5,
    noise_std: float = 0.005,
    a0: float = A0_DEFAULT,
    rng_seed: int = 0,
) -> Path:
    """Create a synthetic per-radial-point CSV with a known deep-regime slope."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for gal_idx in range(n_galaxies):
        g_bar_deep = rng.uniform(0.001 * a0, 0.29 * a0, n_deep)
        g_bar_shallow = rng.uniform(a0, 5.0 * a0, n_shallow)
        g_bar_all = np.concatenate([g_bar_deep, g_bar_shallow])
        log_gbar = np.log10(g_bar_all)
        log_gobs = (
            planted_slope * log_gbar
            + 0.5 * np.log10(a0)
            + rng.normal(0, noise_std, len(g_bar_all))
        )
        for j in range(len(g_bar_all)):
            rows.append({
                "galaxy": f"SYN{gal_idx:02d}",
                "r_kpc": float(j + 1),
                "g_bar": g_bar_all[j],
                "g_obs": 10.0 ** log_gobs[j],
                "log_g_bar": log_gbar[j],
                "log_g_obs": log_gobs[j],
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


def _make_f3_catalog_csv(
    tmp_path: Path,
    n_consistent: int = 10,
    n_inconsistent: int = 3,
    n_nan: int = 2,
    rng_seed: int = 42,
    with_hierarchy: bool = True,
) -> Path:
    """Create a synthetic per-galaxy f3 catalog CSV."""
    rng = np.random.default_rng(rng_seed)
    a0 = A0_DEFAULT
    rows = []

    for i in range(n_consistent):
        n_deep = 20
        g_bar = rng.uniform(0.001 * a0, 0.29 * a0, n_deep)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(a0) + rng.normal(0, 0.01, n_deep)
        slope, _, _, _, stderr = _linregress(log_gbar, log_gobs)
        flag = 1.0 if abs(slope - 0.5) <= 2.0 * stderr else 0.0
        r_med = rng.uniform(1.0, 10.0)
        row = {
            "galaxy": f"CON{i:03d}", "n_total": 40, "n_deep": n_deep,
            "friction_slope": float(slope),
            "friction_slope_stderr": float(stderr),
            "velo_inerte_flag": flag,
        }
        if with_hierarchy:
            row["hierarchy_scm"] = float(hierarchy_level(r_med, R0_KPC_DEFAULT))
        rows.append(row)

    for i in range(n_inconsistent):
        n_deep = 30
        g_bar = rng.uniform(0.001 * a0, 0.29 * a0, n_deep)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.75 * log_gbar + 0.5 * np.log10(a0) + rng.normal(0, 0.001, n_deep)
        slope, _, _, _, stderr = _linregress(log_gbar, log_gobs)
        flag = 1.0 if abs(slope - 0.5) <= 2.0 * stderr else 0.0
        r_med = rng.uniform(1.0, 10.0)
        row = {
            "galaxy": f"INC{i:03d}", "n_total": 40, "n_deep": n_deep,
            "friction_slope": float(slope),
            "friction_slope_stderr": float(stderr),
            "velo_inerte_flag": flag,
        }
        if with_hierarchy:
            row["hierarchy_scm"] = float(hierarchy_level(r_med, R0_KPC_DEFAULT))
        rows.append(row)

    for i in range(n_nan):
        row = {
            "galaxy": f"NAN{i:03d}", "n_total": 3, "n_deep": 0,
            "friction_slope": float("nan"),
            "friction_slope_stderr": float("nan"),
            "velo_inerte_flag": float("nan"),
        }
        if with_hierarchy:
            row["hierarchy_scm"] = float("nan")
        rows.append(row)

    cols = [
        "galaxy", "n_total", "n_deep",
        "friction_slope", "friction_slope_stderr", "velo_inerte_flag",
    ]
    if with_hierarchy:
        cols.append("hierarchy_scm")
    catalog = pd.DataFrame(rows, columns=cols)
    p = tmp_path / "f3_catalog.csv"
    catalog.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Unit tests: hierarchy_level
# ---------------------------------------------------------------------------

class TestHierarchyLevel:
    def test_base_value_gives_zero(self):
        """hierarchy_level(x0, x0) must be exactly 0."""
        assert hierarchy_level(1.0, 1.0) == pytest.approx(0.0)

    def test_one_level_up_gives_one(self):
        """hierarchy_level(60 * x0, x0) must be exactly 1."""
        assert hierarchy_level(60.0, 1.0) == pytest.approx(1.0)

    def test_two_levels_up_gives_two(self):
        """hierarchy_level(3600 * x0, x0) must be exactly 2."""
        assert hierarchy_level(3600.0, 1.0) == pytest.approx(2.0)

    def test_fractional_level(self):
        """hierarchy_level(sqrt(60), 1) == 0.5."""
        assert hierarchy_level(60.0 ** 0.5, 1.0) == pytest.approx(0.5)

    def test_negative_level_below_reference(self):
        """hierarchy_level(1/60, 1) == -1."""
        assert hierarchy_level(1.0 / 60.0, 1.0) == pytest.approx(-1.0)

    def test_array_input(self):
        """hierarchy_level accepts arrays and returns the right shape."""
        x = np.array([1.0, 60.0, 3600.0])
        result = hierarchy_level(x, 1.0)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-12)

    def test_non_positive_input_gives_nan(self):
        """Non-positive x should produce NaN (log of non-positive is undefined)."""
        assert math.isnan(hierarchy_level(0.0, 1.0))
        assert math.isnan(hierarchy_level(-1.0, 1.0))

    def test_base_60_constant_used(self):
        """Result must equal log(x/x0) / log(60)."""
        x, x0 = 120.0, 2.0
        expected = np.log(x / x0) / np.log(BASE_60)
        assert hierarchy_level(x, x0) == pytest.approx(expected)

    def test_custom_reference(self):
        """Scaling x0 should shift S by a constant."""
        s1 = hierarchy_level(10.0, 1.0)
        s2 = hierarchy_level(10.0, 2.0)
        expected_shift = hierarchy_level(2.0, 1.0)
        assert s1 - s2 == pytest.approx(expected_shift)


# ---------------------------------------------------------------------------
# Unit tests: compute_galaxy_beta
# ---------------------------------------------------------------------------

class TestComputeGalaxyBeta:
    def test_returns_required_keys(self):
        a0 = A0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * a0), np.log10(0.29 * a0), 50)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(a0)
        result = compute_galaxy_beta(log_gbar, log_gobs)
        required = {
            "n_total", "n_deep", "friction_slope",
            "friction_slope_stderr", "velo_inerte_flag",
        }
        assert required.issubset(set(result.keys()))

    def test_exact_mond_slope_recovers_half(self):
        """MOND data with small noise must recover slope ≈ 0.5 and flag = 1."""
        rng = np.random.default_rng(17)
        a0 = A0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * a0), np.log10(0.29 * a0), 200)
        log_gbar = np.log10(g_bar)
        # Small but non-zero noise so that linregress produces a finite stderr
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(a0) + rng.normal(0, 0.005, 200)
        result = compute_galaxy_beta(log_gbar, log_gobs, a0=a0)
        assert result["friction_slope"] == pytest.approx(0.5, abs=0.02)
        assert result["velo_inerte_flag"] == 1.0

    def test_nan_when_zero_deep_points(self):
        """Slope must be NaN when no deep-regime points exist."""
        a0 = A0_DEFAULT
        g_bar = np.linspace(2.0 * a0, 10.0 * a0, 50)
        log_gbar = np.log10(g_bar)
        log_gobs = log_gbar
        result = compute_galaxy_beta(log_gbar, log_gobs, a0=a0)
        assert math.isnan(result["friction_slope"])
        assert math.isnan(result["velo_inerte_flag"])
        assert result["n_deep"] == 0

    def test_nan_when_one_deep_point(self):
        """Slope must be NaN when only one deep-regime point (can't fit a line)."""
        a0 = A0_DEFAULT
        # One deep point + many shallow points
        g_bar = np.array([0.1 * a0] + [5.0 * a0] * 20)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(a0)
        result = compute_galaxy_beta(log_gbar, log_gobs, a0=a0)
        assert math.isnan(result["friction_slope"])
        assert result["n_deep"] == 1

    def test_inconsistent_slope_gives_flag_zero(self):
        """A strongly deviant slope should yield velo_inerte_flag = 0."""
        a0 = A0_DEFAULT
        rng = np.random.default_rng(99)
        g_bar = rng.uniform(0.001 * a0, 0.29 * a0, 100)
        log_gbar = np.log10(g_bar)
        # Plant slope = 0.75, very low noise → far from 0.5
        log_gobs = 0.75 * log_gbar + 0.5 * np.log10(a0) + rng.normal(0, 0.001, 100)
        result = compute_galaxy_beta(log_gbar, log_gobs, a0=a0)
        assert result["velo_inerte_flag"] == 0.0

    def test_n_deep_matches_threshold(self):
        """n_deep must match the count of points with g_bar < threshold × a0."""
        a0 = A0_DEFAULT
        threshold = 0.3
        rng = np.random.default_rng(7)
        g_bar = rng.uniform(0.0, 2.0 * a0, 100)
        expected = int((g_bar < threshold * a0).sum())
        log_gbar = np.log10(np.maximum(g_bar, 1e-40))
        log_gobs = 0.5 * log_gbar
        result = compute_galaxy_beta(log_gbar, log_gobs, a0=a0, deep_threshold=threshold)
        assert result["n_deep"] == expected

    def test_n_total_equals_input_length(self):
        a0 = A0_DEFAULT
        g_bar = np.linspace(0.001 * a0, 5.0 * a0, 75)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar
        result = compute_galaxy_beta(log_gbar, log_gobs, a0=a0)
        assert result["n_total"] == 75


# ---------------------------------------------------------------------------
# Unit tests: build_catalog
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    def test_output_has_expected_columns(self, tmp_path):
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=3)
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        expected_cols = {
            "galaxy", "n_total", "n_deep",
            "friction_slope", "friction_slope_stderr", "velo_inerte_flag",
        }
        assert expected_cols.issubset(set(catalog.columns))

    def test_one_row_per_galaxy(self, tmp_path):
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=6)
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        assert len(catalog) == 6
        assert catalog["galaxy"].nunique() == 6

    def test_mond_galaxies_yield_slope_near_half(self, tmp_path):
        csv = _make_mond_per_point_csv(
            tmp_path, n_galaxies=5, n_deep=50, planted_slope=0.5, noise_std=0.005
        )
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        valid = catalog.dropna(subset=["friction_slope"])
        assert len(valid) > 0
        assert valid["friction_slope"].mean() == pytest.approx(0.5, abs=0.05)

    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"galaxy": ["A"], "g_bar": [1e-11]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_catalog(bad_df)

    def test_nan_flag_for_no_deep_regime(self, tmp_path):
        """Galaxies with all shallow points must have NaN flag."""
        # Make a CSV where all g_bar >> a0
        a0 = A0_DEFAULT
        rng = np.random.default_rng(3)
        g_bar = rng.uniform(2.0 * a0, 10.0 * a0, 50)
        log_gbar = np.log10(g_bar)
        log_gobs = log_gbar
        df = pd.DataFrame({
            "galaxy": ["SHALLOW"] * 50,
            "r_kpc": np.arange(50, dtype=float),
            "g_bar": g_bar,
            "g_obs": 10.0 ** log_gobs,
            "log_g_bar": log_gbar,
            "log_g_obs": log_gobs,
        })
        catalog = build_catalog(df)
        assert len(catalog) == 1
        assert math.isnan(catalog.iloc[0]["velo_inerte_flag"])

    def test_hierarchy_scm_column_present(self, tmp_path):
        """build_catalog must always return a hierarchy_scm column."""
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=3)
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        assert "hierarchy_scm" in catalog.columns

    def test_hierarchy_scm_finite_when_r_kpc_present(self, tmp_path):
        """hierarchy_scm must be finite for deep-regime galaxies that have r_kpc."""
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=3, n_deep=30)
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        valid = catalog.dropna(subset=["friction_slope"])
        assert valid["hierarchy_scm"].apply(np.isfinite).all()

    def test_hierarchy_scm_nan_without_r_kpc(self, tmp_path):
        """hierarchy_scm must be NaN when the input has no r_kpc column."""
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=2, n_deep=20)
        df = pd.read_csv(csv).drop(columns=["r_kpc"])
        catalog = build_catalog(df)
        assert catalog["hierarchy_scm"].isna().all()

    def test_hierarchy_scm_value_matches_formula(self, tmp_path):
        """S_SCM must equal log_60(r_median_deep / r0_kpc) for a known dataset."""
        a0 = A0_DEFAULT
        r0 = 2.0  # custom reference
        # Deep-regime points spanning a range of g_bar at r=60*r0 → S_SCM should be 1
        rng = np.random.default_rng(42)
        g_bar_deep = rng.uniform(0.001 * a0, 0.29 * a0, 20)
        log_gbar = np.log10(g_bar_deep)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(a0)
        df = pd.DataFrame({
            "galaxy": ["TST"] * 20,
            "r_kpc": np.full(20, 60.0 * r0),
            "g_bar": g_bar_deep,
            "g_obs": 10.0 ** log_gobs,
            "log_g_bar": log_gbar,
            "log_g_obs": log_gobs,
        })
        catalog = build_catalog(df, r0_kpc=r0)
        assert catalog.iloc[0]["hierarchy_scm"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unit tests: analyze_catalog
# ---------------------------------------------------------------------------

class TestAnalyzeCatalog:
    def test_returns_required_keys(self, tmp_path):
        csv = _make_f3_catalog_csv(tmp_path)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        required = {
            "n_total_rows", "n_analyzed", "n_consistent", "n_inconsistent",
            "n_nan", "beta_mean", "beta_std", "t_stat", "p_value",
        }
        assert required.issubset(set(result.keys()))

    def test_counts_are_correct(self, tmp_path):
        csv = _make_f3_catalog_csv(
            tmp_path, n_consistent=8, n_inconsistent=2, n_nan=3
        )
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert result["n_total_rows"] == 13
        assert result["n_analyzed"] == 10  # 8 consistent + 2 inconsistent
        assert result["n_nan"] == 3

    def test_all_consistent_catalog(self, tmp_path):
        """All-consistent catalog should have n_inconsistent = 0."""
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=10, n_inconsistent=0, n_nan=0)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert result["n_inconsistent"] == 0
        assert result["n_nan"] == 0
        assert result["n_analyzed"] == 10

    def test_p_value_large_for_mond_data(self, tmp_path):
        """For data truly distributed around β=0.5, p-value should be > 0.05."""
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=20, n_inconsistent=0, n_nan=0)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert result["p_value"] > 0.05

    def test_p_value_small_for_deviant_data(self):
        """When all β values are far from 0.5, p-value should be very small."""
        rng = np.random.default_rng(100)
        slopes = rng.normal(0.75, 0.01, 30)  # far from 0.5
        df = pd.DataFrame({
            "friction_slope": slopes,
            "velo_inerte_flag": [0.0] * 30,
        })
        result = analyze_catalog(df)
        assert result["p_value"] < 0.001

    def test_nan_when_fewer_than_two_valid_rows(self):
        """With < 2 valid rows, stats should be NaN."""
        df = pd.DataFrame({
            "friction_slope": [float("nan"), float("nan")],
            "velo_inerte_flag": [float("nan"), float("nan")],
        })
        result = analyze_catalog(df)
        assert math.isnan(result["beta_mean"])
        assert math.isnan(result["t_stat"])

    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"galaxy": ["A"], "n_total": [10]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_catalog(bad_df)

    def test_n_consistent_plus_inconsistent_equals_analyzed(self, tmp_path):
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=7, n_inconsistent=3, n_nan=2)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert result["n_consistent"] + result["n_inconsistent"] == result["n_analyzed"]

    def test_beta_mean_near_half_for_mond_catalog(self, tmp_path):
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=20, n_inconsistent=0, n_nan=0)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert result["beta_mean"] == pytest.approx(0.5, abs=0.05)

    def test_pearson_keys_present(self, tmp_path):
        """analyze_catalog must always return pearson_r_scm and pearson_p_scm."""
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=10)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert "pearson_r_scm" in result
        assert "pearson_p_scm" in result

    def test_pearson_nan_without_hierarchy_scm(self):
        """Pearson fields must be NaN when hierarchy_scm column is absent."""
        df = pd.DataFrame({
            "friction_slope": [0.48, 0.51, 0.50, 0.49],
            "velo_inerte_flag": [1.0, 1.0, 1.0, 1.0],
        })
        result = analyze_catalog(df)
        assert math.isnan(result["pearson_r_scm"])
        assert math.isnan(result["pearson_p_scm"])

    def test_pearson_finite_with_hierarchy_scm(self, tmp_path):
        """Pearson fields must be finite when hierarchy_scm is populated."""
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=10, n_nan=0,
                                    with_hierarchy=True)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        # At least 3 rows have both valid slope and hierarchy_scm
        assert np.isfinite(result["pearson_r_scm"])
        assert np.isfinite(result["pearson_p_scm"])

    def test_pearson_r_range(self, tmp_path):
        """Pearson r must be in [-1, 1]."""
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=15, n_nan=0,
                                    with_hierarchy=True)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        assert -1.0 <= result["pearson_r_scm"] <= 1.0


# ---------------------------------------------------------------------------
# print_summary (smoke test — just checks it doesn't raise)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_runs_without_error(self, tmp_path, capsys):
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=5, n_inconsistent=2, n_nan=1)
        df = pd.read_csv(csv)
        result = analyze_catalog(df)
        print_summary(result)
        captured = capsys.readouterr()
        assert "F3 CATALOG ANALYSIS SUMMARY" in captured.out
        assert "Total rows in catalog" in captured.out
        assert "Galaxies analyzed" in captured.out
        assert "Consistent with" in captured.out
        assert "Inconsistent" in captured.out
        assert "Insufficient data" in captured.out
        assert "Pearson" in captured.out

    def test_output_contains_nan_summary_when_no_valid(self, capsys):
        df = pd.DataFrame({
            "friction_slope": [float("nan")] * 3,
            "velo_inerte_flag": [float("nan")] * 3,
        })
        result = analyze_catalog(df)
        print_summary(result)
        captured = capsys.readouterr()
        assert "nan" in captured.out.lower()


# ---------------------------------------------------------------------------
# CLI: generate_f3_catalog.main()
# ---------------------------------------------------------------------------

class TestGenerateMainCLI:
    def test_produces_catalog_file(self, tmp_path):
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=4)
        out = tmp_path / "catalog_out.csv"
        catalog = generate_main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()
        assert len(catalog) == 4

    def test_output_has_required_columns(self, tmp_path):
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=3)
        out = tmp_path / "cat.csv"
        generate_main(["--csv", str(csv), "--out", str(out)])
        df = pd.read_csv(out)
        for col in ["galaxy", "n_total", "n_deep", "friction_slope",
                    "friction_slope_stderr", "velo_inerte_flag"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_mond_data_yields_slope_near_half(self, tmp_path):
        csv = _make_mond_per_point_csv(
            tmp_path, n_galaxies=8, n_deep=40, planted_slope=0.5, noise_std=0.005
        )
        out = tmp_path / "cat.csv"
        catalog = generate_main(["--csv", str(csv), "--out", str(out)])
        valid = catalog.dropna(subset=["friction_slope"])
        assert len(valid) == 8
        assert valid["friction_slope"].mean() == pytest.approx(0.5, abs=0.05)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_main(["--csv", str(tmp_path / "nonexistent.csv")])

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "g_bar": [1e-11]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            generate_main(["--csv", str(bad_csv)])

    def test_creates_parent_directory(self, tmp_path):
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=2)
        out = tmp_path / "new_subdir" / "catalog.csv"
        generate_main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()

    def test_output_has_hierarchy_scm_column(self, tmp_path):
        """Catalog output must include the hierarchy_scm column."""
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=3)
        out = tmp_path / "cat_hier.csv"
        generate_main(["--csv", str(csv), "--out", str(out)])
        df = pd.read_csv(out)
        assert "hierarchy_scm" in df.columns

    def test_r0_kpc_arg_changes_hierarchy_scm(self, tmp_path):
        """Different --r0-kpc values must produce different hierarchy_scm values."""
        csv = _make_mond_per_point_csv(tmp_path, n_galaxies=3, n_deep=20)
        out1 = tmp_path / "cat1.csv"
        out2 = tmp_path / "cat2.csv"
        generate_main(["--csv", str(csv), "--out", str(out1), "--r0-kpc", "1.0"])
        generate_main(["--csv", str(csv), "--out", str(out2), "--r0-kpc", "5.0"])
        df1 = pd.read_csv(out1).dropna(subset=["hierarchy_scm"])
        df2 = pd.read_csv(out2).dropna(subset=["hierarchy_scm"])
        # S_SCM(r, 1.0) != S_SCM(r, 5.0) for the same r
        assert not np.allclose(
            df1["hierarchy_scm"].values, df2["hierarchy_scm"].values
        )


# ---------------------------------------------------------------------------
# CLI: f3_catalog_analysis.main()
# ---------------------------------------------------------------------------

class TestAnalysisMainCLI:
    def test_returns_result_dict(self, tmp_path):
        csv = _make_f3_catalog_csv(tmp_path, n_consistent=6, n_inconsistent=2, n_nan=1)
        result = analysis_main(["--catalog", str(csv)])
        assert isinstance(result, dict)
        assert result["n_total_rows"] == 9

    def test_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analysis_main(["--catalog", str(tmp_path / "no_such.csv")])

    def test_prints_summary_to_stdout(self, tmp_path, capsys):
        csv = _make_f3_catalog_csv(tmp_path)
        analysis_main(["--catalog", str(csv)])
        captured = capsys.readouterr()
        assert "F3 CATALOG ANALYSIS SUMMARY" in captured.out


# ---------------------------------------------------------------------------
# End-to-end: generate → analyze pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_full_pipeline_consistent_data(self, tmp_path):
        """Generate catalog from MOND data, then analyze → high p-value."""
        per_point_csv = _make_mond_per_point_csv(
            tmp_path, n_galaxies=10, n_deep=40, planted_slope=0.5, noise_std=0.005
        )
        catalog_csv = tmp_path / "catalog.csv"
        catalog = generate_main(["--csv", str(per_point_csv), "--out", str(catalog_csv)])

        # All 10 galaxies should have valid slopes
        assert catalog["friction_slope"].apply(np.isfinite).sum() == 10

        result = analysis_main(["--catalog", str(catalog_csv)])
        assert result["n_total_rows"] == 10
        assert result["n_analyzed"] == 10
        assert result["n_nan"] == 0

        # MOND data → p-value should not reject β=0.5
        assert result["p_value"] > 0.05

    def test_full_pipeline_no_deep_regime(self, tmp_path):
        """Generate catalog from shallow-only data → all NaN flags."""
        a0 = A0_DEFAULT
        rng = np.random.default_rng(55)
        n_gals, n_pts = 5, 20
        g_bar = rng.uniform(2.0 * a0, 10.0 * a0, n_gals * n_pts)
        log_gbar = np.log10(g_bar)
        log_gobs = log_gbar
        df = pd.DataFrame({
            "galaxy": np.repeat([f"SH{i:02d}" for i in range(n_gals)], n_pts),
            "r_kpc": np.tile(np.arange(n_pts, dtype=float), n_gals),
            "g_bar": g_bar,
            "g_obs": 10.0 ** log_gobs,
            "log_g_bar": log_gbar,
            "log_g_obs": log_gobs,
        })
        per_point_csv = tmp_path / "shallow.csv"
        df.to_csv(per_point_csv, index=False)

        catalog_csv = tmp_path / "catalog_shallow.csv"
        catalog = generate_main(["--csv", str(per_point_csv), "--out", str(catalog_csv)])

        assert len(catalog) == 5
        assert catalog["velo_inerte_flag"].isna().sum() == 5

        result = analysis_main(["--catalog", str(catalog_csv)])
        assert result["n_nan"] == 5
        assert result["n_analyzed"] == 0

    def test_real_catalog_artifact_is_loadable(self):
        """The committed results/f3_catalog_real.csv must be a valid catalog."""
        catalog_path = Path("results/f3_catalog_real.csv")
        assert catalog_path.exists(), (
            "results/f3_catalog_real.csv not found — "
            "run generate_f3_catalog.py to create it."
        )
        df = pd.read_csv(catalog_path)
        for col in ["galaxy", "n_total", "n_deep", "friction_slope",
                    "friction_slope_stderr", "velo_inerte_flag", "hierarchy_scm"]:
            assert col in df.columns, f"Missing column in artifact: {col}"
        assert len(df) > 0

    def test_real_catalog_analysis_runs(self):
        """analysis_main on the real artifact must complete without error."""
        catalog_path = Path("results/f3_catalog_real.csv")
        pytest.importorskip("scipy")
        result = analysis_main(["--catalog", str(catalog_path)])
        assert result["n_total_rows"] == result["n_analyzed"] + result["n_nan"]
        assert (
            result["n_consistent"] + result["n_inconsistent"]
            == result["n_analyzed"]
        )
