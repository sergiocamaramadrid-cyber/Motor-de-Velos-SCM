"""
tests/test_f3_catalog.py — Tests for scripts/generate_f3_catalog.py and
scripts/f3_catalog_analysis.py.

Uses deterministic synthetic SPARC-like data to ensure results are stable
across runs.  Note: flat-rotation-curve synthetic data produces β ≈ 1.0 in
the deep regime (g_obs and g_bar both scale as V²/r with constant V), whereas
real SPARC LSB galaxies are expected to yield β ≈ 0.5 (deep-MOND form).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_f3_catalog import (
    fit_galaxy_slope,
    build_catalog,
    main as gen_main,
    G0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
    MIN_DEEP_POINTS,
    EXPECTED_SLOPE,
)
from scripts.f3_catalog_analysis import (
    analyze_catalog,
    format_summary,
    main as ana_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparc_like_csv(
    tmp_path: Path,
    n_galaxies: int = 20,
    n_pts_per_galaxy: int = 20,
    planted_slope: float = 0.5,
    n_deep_per_galaxy: int = 12,
    g0: float = G0_DEFAULT,
    seed: int = 42,
) -> Path:
    """Create a synthetic per-radial-point CSV in SPARC column format.

    Each galaxy gets ``n_deep_per_galaxy`` points strictly in the deep regime
    (g_bar < 0.3·g0) with log_g_obs = planted_slope·log_g_bar + 0.5·log10(g0),
    and the remainder in the shallow regime.
    """
    rng = np.random.default_rng(seed)
    n_shallow = n_pts_per_galaxy - n_deep_per_galaxy
    rows: list[dict] = []

    for i in range(n_galaxies):
        gal = f"G{i:04d}"

        # Deep-regime points
        g_bar_deep = rng.uniform(0.001 * g0, 0.29 * g0, n_deep_per_galaxy)
        log_gb_d = np.log10(g_bar_deep)
        log_go_d = (
            planted_slope * log_gb_d
            + 0.5 * np.log10(g0)
            + rng.normal(0, 0.005, n_deep_per_galaxy)
        )

        # Shallow-regime points
        g_bar_shal = rng.uniform(g0, 10.0 * g0, n_shallow)
        log_gb_s = np.log10(g_bar_shal)
        log_go_s = log_gb_s + rng.normal(0, 0.01, n_shallow)

        log_gb = np.concatenate([log_gb_d, log_gb_s])
        log_go = np.concatenate([log_go_d, log_go_s])
        g_bar = 10.0 ** log_gb
        g_obs = 10.0 ** log_go

        for j in range(n_pts_per_galaxy):
            rows.append({
                "galaxy": gal,
                "r_kpc": float(j + 1),
                "g_bar": g_bar[j],
                "g_obs": g_obs[j],
                "log_g_bar": log_gb[j],
                "log_g_obs": log_go[j],
            })

    df = pd.DataFrame(rows)
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


def _make_catalog_csv(
    tmp_path: Path,
    n_galaxies: int = 20,
    mean_slope: float = 0.5,
    slope_std: float = 0.05,
    seed: int = 7,
) -> Path:
    """Create a synthetic per-galaxy F3 catalog CSV."""
    rng = np.random.default_rng(seed)
    slopes = rng.normal(mean_slope, slope_std, n_galaxies)
    errs = rng.uniform(0.01, 0.05, n_galaxies)
    flags = [1 if abs(s - 0.5) <= 2 * e else 0 for s, e in zip(slopes, errs)]

    df = pd.DataFrame({
        "galaxy": [f"G{i:04d}" for i in range(n_galaxies)],
        "n_total": 20,
        "n_deep": 12,
        "friction_slope": slopes,
        "friction_slope_err": errs,
        "velo_inerte_flag": flags,
    })
    p = tmp_path / "f3_catalog.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. fit_galaxy_slope unit tests
# ---------------------------------------------------------------------------


def rng_helper(loc: float, scale: float, n: int) -> np.ndarray:
    """Deterministic noise helper used in TestFitGalaxySlope."""
    return np.random.default_rng(999).normal(loc, scale, n)


class TestFitGalaxySlope:
    def test_returns_required_keys(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 20)
        log_gb = np.log10(g_bar)
        log_go = 0.5 * log_gb + 0.5 * np.log10(g0)
        result = fit_galaxy_slope(log_gb, log_go)
        assert {"n_total", "n_deep", "friction_slope",
                "friction_slope_err", "velo_inerte_flag"} <= set(result.keys())

    def test_exact_mond_slope_recovered(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 50)
        log_gb = np.log10(g_bar)
        log_go = 0.5 * log_gb + 0.5 * np.log10(g0)
        result = fit_galaxy_slope(log_gb, log_go)
        assert result["friction_slope"] == pytest.approx(0.5, abs=1e-4)

    def test_nan_when_insufficient_deep_points(self):
        g0 = G0_DEFAULT
        # All points shallow
        g_bar = np.linspace(2.0 * g0, 10.0 * g0, 30)
        log_gb = np.log10(g_bar)
        log_go = log_gb
        result = fit_galaxy_slope(log_gb, log_go)
        assert math.isnan(result["friction_slope"])
        assert math.isnan(result["friction_slope_err"])
        assert math.isnan(result["velo_inerte_flag"])

    def test_n_deep_counts_deep_mask(self):
        g0 = G0_DEFAULT
        threshold = DEEP_THRESHOLD_DEFAULT
        rng = np.random.default_rng(1)
        g_bar = rng.uniform(0.001 * g0, 5.0 * g0, 60)
        expected_deep = int((g_bar < threshold * g0).sum())
        log_gb = np.log10(g_bar)
        log_go = 0.5 * log_gb
        result = fit_galaxy_slope(log_gb, log_go)
        assert result["n_deep"] == expected_deep

    def test_velo_inerte_flag_one_when_consistent(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 50)
        log_gb = np.log10(g_bar)
        # Perfect MOND data — slope exactly 0.5, tiny error → flag must be 1
        log_go = 0.5 * log_gb + 0.5 * np.log10(g0)
        result = fit_galaxy_slope(log_gb, log_go)
        assert result["velo_inerte_flag"] == 1

    def test_velo_inerte_flag_zero_when_far_from_half(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 200)
        log_gb = np.log10(g_bar)
        # Slope = 1.0, well above 0.5 + 2σ for clean data
        log_go = 1.0 * log_gb + rng_helper(0, 0.001, 200)
        result = fit_galaxy_slope(log_gb, log_go)
        assert result["velo_inerte_flag"] == 0

    def test_n_total_equals_input_length(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(5.0 * g0), 37)
        log_gb = np.log10(g_bar)
        log_go = 0.5 * log_gb
        result = fit_galaxy_slope(log_gb, log_go)
        assert result["n_total"] == 37

    def test_planted_slope_recovered(self):
        rng = np.random.default_rng(3)
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 100)
        log_gb = np.log10(g_bar)
        log_go = 0.44 * log_gb + 0.5 * np.log10(g0) + rng.normal(0, 0.001, 100)
        result = fit_galaxy_slope(log_gb, log_go)
        assert result["friction_slope"] == pytest.approx(0.44, abs=0.02)


# ---------------------------------------------------------------------------
# 2. build_catalog unit tests
# ---------------------------------------------------------------------------


class TestBuildCatalog:
    def test_one_row_per_galaxy(self, tmp_path):
        csv = _make_sparc_like_csv(tmp_path, n_galaxies=10)
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        assert len(catalog) == 10

    def test_output_columns_present(self, tmp_path):
        csv = _make_sparc_like_csv(tmp_path, n_galaxies=5)
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        for col in ["galaxy", "n_total", "n_deep",
                    "friction_slope", "friction_slope_err", "velo_inerte_flag"]:
            assert col in catalog.columns

    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"galaxy": ["A"], "g_bar": [1e-11]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_catalog(bad_df)

    def test_slopes_near_planted_value(self, tmp_path):
        csv = _make_sparc_like_csv(
            tmp_path, n_galaxies=15, n_deep_per_galaxy=15,
            planted_slope=0.5, n_pts_per_galaxy=20
        )
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        valid = catalog.dropna(subset=["friction_slope"])
        assert len(valid) > 0
        assert valid["friction_slope"].mean() == pytest.approx(0.5, abs=0.1)

    def test_nan_galaxies_for_insufficient_deep(self, tmp_path):
        """Galaxies without enough deep points should have NaN slope."""
        csv = _make_sparc_like_csv(
            tmp_path, n_galaxies=5, n_deep_per_galaxy=0, n_pts_per_galaxy=20
        )
        df = pd.read_csv(csv)
        catalog = build_catalog(df)
        assert catalog["friction_slope"].isna().all()


# ---------------------------------------------------------------------------
# 3. generate_f3_catalog CLI tests
# ---------------------------------------------------------------------------


class TestGenerateCatalogCLI:
    def test_writes_output_csv(self, tmp_path):
        csv = _make_sparc_like_csv(tmp_path, n_galaxies=10)
        out = tmp_path / "catalog_out.csv"
        result = gen_main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()
        assert isinstance(result, pd.DataFrame)

    def test_output_has_expected_columns(self, tmp_path):
        csv = _make_sparc_like_csv(tmp_path, n_galaxies=5)
        out = tmp_path / "cat.csv"
        gen_main(["--csv", str(csv), "--out", str(out)])
        df = pd.read_csv(out)
        for col in ["galaxy", "n_total", "n_deep",
                    "friction_slope", "friction_slope_err", "velo_inerte_flag"]:
            assert col in df.columns

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gen_main(["--csv", str(tmp_path / "no_such.csv")])

    def test_returns_dataframe(self, tmp_path):
        csv = _make_sparc_like_csv(tmp_path, n_galaxies=5)
        out = tmp_path / "out.csv"
        result = gen_main(["--csv", str(csv), "--out", str(out)])
        assert isinstance(result, pd.DataFrame)

    def test_creates_parent_directories(self, tmp_path):
        csv = _make_sparc_like_csv(tmp_path, n_galaxies=5)
        out = tmp_path / "deep" / "nested" / "catalog.csv"
        gen_main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()


# ---------------------------------------------------------------------------
# 4. analyze_catalog unit tests
# ---------------------------------------------------------------------------


class TestAnalyzeCatalog:
    def test_returns_required_keys(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=20)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        for key in ["n_total", "n_analyzed", "mean", "std", "median",
                    "t_stat", "p_value", "n_consistent", "verdict"]:
            assert key in result

    def test_n_analyzed_excludes_nan_rows(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=10)
        catalog = pd.read_csv(cat)
        # Add rows with NaN slope
        extra = pd.DataFrame({
            "galaxy": ["NaN_A", "NaN_B"],
            "n_total": [20, 20],
            "n_deep": [0, 0],
            "friction_slope": [float("nan"), float("nan")],
            "friction_slope_err": [float("nan"), float("nan")],
            "velo_inerte_flag": [float("nan"), float("nan")],
        })
        catalog = pd.concat([catalog, extra], ignore_index=True)
        result = analyze_catalog(catalog)
        assert result["n_total"] == 12
        assert result["n_analyzed"] == 10

    def test_mean_close_to_planted(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=50, mean_slope=0.5,
                                 slope_std=0.02)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        assert result["mean"] == pytest.approx(0.5, abs=0.1)

    def test_pvalue_high_when_slopes_near_half(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=30, mean_slope=0.5,
                                 slope_std=0.02, seed=12)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        # Mean near 0.5 → should fail to reject H0
        assert result["p_value"] > 0.05

    def test_pvalue_low_when_slopes_far_from_half(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=30, mean_slope=1.0,
                                 slope_std=0.05)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        assert result["p_value"] < 0.05

    def test_insufficient_galaxies_verdict(self):
        catalog = pd.DataFrame({
            "friction_slope": [0.5],
            "velo_inerte_flag": [1],
        })
        result = analyze_catalog(catalog)
        assert math.isnan(result["mean"])
        assert "Insufficient" in result["verdict"]

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"galaxy": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_catalog(bad)

    def test_n_consistent_counts_flag_one(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=20, mean_slope=0.5)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        manual = int((catalog["velo_inerte_flag"] == 1).sum())
        assert result["n_consistent"] == manual


# ---------------------------------------------------------------------------
# 5. f3_catalog_analysis CLI tests
# ---------------------------------------------------------------------------


class TestAnalysisCLI:
    def test_returns_dict(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=20)
        result = ana_main(["--catalog", str(cat)])
        assert isinstance(result, dict)

    def test_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ana_main(["--catalog", str(tmp_path / "no_catalog.csv")])

    def test_verdict_in_result(self, tmp_path):
        cat = _make_catalog_csv(tmp_path, n_galaxies=20)
        result = ana_main(["--catalog", str(cat)])
        assert "verdict" in result
        assert isinstance(result["verdict"], str)


# ---------------------------------------------------------------------------
# 6. format_summary tests
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_contains_catalog_path(self, tmp_path):
        cat = _make_catalog_csv(tmp_path)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        lines = format_summary(result, "my/path.csv")
        combined = "\n".join(lines)
        assert "my/path.csv" in combined

    def test_contains_mean_beta_line(self, tmp_path):
        cat = _make_catalog_csv(tmp_path)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        lines = format_summary(result, "x.csv")
        combined = "\n".join(lines)
        assert "mean β" in combined

    def test_contains_ttest_section(self, tmp_path):
        cat = _make_catalog_csv(tmp_path)
        catalog = pd.read_csv(cat)
        result = analyze_catalog(catalog)
        lines = format_summary(result, "x.csv")
        combined = "\n".join(lines)
        assert "t-statistic" in combined
        assert "p-value" in combined


# ---------------------------------------------------------------------------
# 7. End-to-end integration test (synthetic SPARC-like data)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def e2e_outputs(self, tmp_path_factory):
        """Run generate_f3_catalog → f3_catalog_analysis on synthetic data."""
        root = tmp_path_factory.mktemp("e2e")
        csv = _make_sparc_like_csv(
            root, n_galaxies=20, n_pts_per_galaxy=20,
            planted_slope=0.5, n_deep_per_galaxy=12, seed=42
        )
        catalog_path = root / "f3_catalog.csv"
        catalog_df = gen_main(["--csv", str(csv), "--out", str(catalog_path)])
        analysis = ana_main(["--catalog", str(catalog_path)])
        return catalog_df, analysis, catalog_path

    def test_catalog_has_one_row_per_galaxy(self, e2e_outputs):
        catalog_df, _, _ = e2e_outputs
        assert len(catalog_df) == 20

    def test_some_galaxies_analyzed(self, e2e_outputs):
        catalog_df, _, _ = e2e_outputs
        assert catalog_df["friction_slope"].notna().sum() > 0

    def test_mean_slope_near_planted(self, e2e_outputs):
        _, analysis, _ = e2e_outputs
        assert analysis["mean"] == pytest.approx(0.5, abs=0.15)

    def test_catalog_csv_written(self, e2e_outputs):
        _, _, catalog_path = e2e_outputs
        assert catalog_path.exists()

    def test_analysis_n_analyzed_le_total(self, e2e_outputs):
        _, analysis, _ = e2e_outputs
        assert analysis["n_analyzed"] <= analysis["n_total"]

    def test_flat_curve_data_gives_slope_near_one(self, tmp_path_factory):
        """Flat-rotation-curve synthetic data (planted slope 1.0) must yield
        mean β ≈ 1.0, not 0.5.  This documents the expected behavior pattern
        that arises from synthetic pipeline output where g_obs and g_bar both
        scale as V²/r with constant V (unlike real SPARC LSB galaxies)."""
        root = tmp_path_factory.mktemp("flat_rc")
        csv = _make_sparc_like_csv(
            root, n_galaxies=20, n_pts_per_galaxy=20,
            planted_slope=1.0, n_deep_per_galaxy=12, seed=13
        )
        catalog_path = root / "flat_catalog.csv"
        catalog_df = gen_main(["--csv", str(csv), "--out", str(catalog_path)])
        analysis = ana_main(["--catalog", str(catalog_path)])
        valid = catalog_df.dropna(subset=["friction_slope"])
        assert len(valid) > 0
        assert analysis["mean"] == pytest.approx(1.0, abs=0.15)
