"""
tests/test_f3_catalog.py — Tests for generate_f3_catalog.py and
f3_catalog_analysis.py.

Uses synthetic SPARC-like datasets so CI runs without real SPARC data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.generate_f3_catalog import (
    compute_friction_slope,
    build_f3_catalog,
    main as gen_main,
    EXPECTED_SLOPE,
    A0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
)
from scripts.f3_catalog_analysis import (
    analyze_catalog,
    format_summary,
    main as ana_main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sparc_dir(root: Path, n_gal: int = 10, n_pts: int = 25,
                    rng: np.random.Generator | None = None) -> Path:
    """Create a minimal synthetic SPARC-like directory."""
    if rng is None:
        rng = np.random.default_rng(42)
    v_flats = np.linspace(80.0, 300.0, n_gal)
    names = [f"SYN{i:03d}" for i in range(n_gal)]

    pd.DataFrame({
        "Galaxy": names,
        "D": np.linspace(3, 50, n_gal),
        "Inc": np.linspace(30, 80, n_gal),
        "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(n_gal, 5.0),
    }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    for name, vf in zip(names, v_flats):
        r = np.linspace(0.5, 15.0, n_pts)
        rc = pd.DataFrame({
            "r": r,
            "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
            "v_obs_err": np.full(n_pts, 5.0),
            "v_gas": 0.3 * vf * np.ones(n_pts),
            "v_disk": 0.75 * vf * np.ones(n_pts),
            "v_bul": np.zeros(n_pts),
            "SBdisk": np.zeros(n_pts),
            "SBbul": np.zeros(n_pts),
        })
        rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)
    return root


def _make_mond_rc(n_pts: int = 30,
                  a0: float = A0_DEFAULT,
                  deep_frac: float = 0.5,
                  slope: float = 0.5,
                  rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Build a synthetic rotation-curve DataFrame with MOND-like deep regime.

    The baryonic and observed accelerations satisfy:
        log10(g_obs) = slope * log10(g_bar) + 0.5*log10(a0) + noise
    in the deep regime and g_obs ≈ g_bar elsewhere.

    The DataFrame is constructed directly in (g_bar, g_obs) space to
    give exact control over the deep-regime slope, independent of the
    V_bar / r decomposition used by _v_baryonic.
    """
    if rng is None:
        rng = np.random.default_rng(7)

    n_deep = int(n_pts * deep_frac)
    n_shallow = n_pts - n_deep

    KPC_TO_M = 3.085677581e16
    _CONV = 1e6 / KPC_TO_M

    # Deep regime: g_bar ∈ [0.001·a0, 0.29·a0]
    g_bar_deep = rng.uniform(0.001 * a0, 0.29 * a0, n_deep)
    # Shallow regime: g_bar ∈ [a0, 5·a0]
    g_bar_shallow = rng.uniform(a0, 5.0 * a0, n_shallow)
    g_bar_all = np.concatenate([g_bar_deep, g_bar_shallow])

    # Construct g_obs from planted slope
    log_gbar = np.log10(g_bar_all)
    log_gobs = (slope * log_gbar + 0.5 * np.log10(a0)
                + rng.normal(0, 0.005, n_pts))
    g_obs_all = 10.0 ** log_gobs

    # Convert g_bar/g_obs back to velocities via fixed r = 5 kpc
    r = np.full(n_pts, 5.0)
    v_bar = np.sqrt(g_bar_all * r / _CONV)   # km/s
    v_obs = np.sqrt(g_obs_all * r / _CONV)   # km/s

    return pd.DataFrame({
        "r": r,
        "v_obs": v_obs,
        "v_obs_err": np.full(n_pts, 2.0),
        "v_gas": 0.5 * v_bar,
        "v_disk": 0.5 * v_bar,
        "v_bul": np.zeros(n_pts),
    })


# ---------------------------------------------------------------------------
# Tests for compute_friction_slope
# ---------------------------------------------------------------------------

class TestComputeFrictionSlope:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(0)
        rc = _make_mond_rc(rng=rng)
        result = compute_friction_slope(rc)
        for key in ("n_deep", "friction_slope", "friction_slope_err",
                    "velo_inerte_flag"):
            assert key in result

    def test_consistent_flag_for_mond_slope(self):
        """Galaxy obeying β = 0.5 must get velo_inerte_flag = 1."""
        rng = np.random.default_rng(1)
        rc = _make_mond_rc(n_pts=200, slope=0.5, deep_frac=0.6, rng=rng)
        result = compute_friction_slope(rc, upsilon_disk=1.0)
        assert result["n_deep"] > 0
        assert abs(result["friction_slope"] - 0.5) < 0.1
        assert result["velo_inerte_flag"] == 1

    def test_nan_when_no_deep_points(self):
        """When no points fall in the deep regime, slope must be NaN."""
        rng = np.random.default_rng(2)
        # Use a very tiny threshold so nothing qualifies as deep
        rc = _make_mond_rc(n_pts=50, slope=0.5, deep_frac=0.5, rng=rng)
        result = compute_friction_slope(rc, a0=A0_DEFAULT,
                                        deep_threshold=1e-15)
        assert result["n_deep"] == 0
        assert np.isnan(result["friction_slope"])
        assert result["velo_inerte_flag"] == 0

    def test_velo_inerte_flag_type_is_int(self):
        rng = np.random.default_rng(3)
        rc = _make_mond_rc(rng=rng)
        result = compute_friction_slope(rc)
        assert isinstance(result["velo_inerte_flag"], int)

    def test_n_deep_nonnegative(self):
        rng = np.random.default_rng(4)
        rc = _make_mond_rc(n_pts=30, rng=rng)
        result = compute_friction_slope(rc)
        assert result["n_deep"] >= 0


# ---------------------------------------------------------------------------
# Tests for build_f3_catalog
# ---------------------------------------------------------------------------

class TestBuildF3Catalog:
    @pytest.fixture(scope="class")
    def sparc_dir(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("sparc")
        return _make_sparc_dir(root, n_gal=5, n_pts=20)

    def test_returns_dataframe(self, sparc_dir):
        df = build_f3_catalog(sparc_dir, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, sparc_dir):
        df = build_f3_catalog(sparc_dir, verbose=False)
        for col in ("galaxy", "n_deep", "friction_slope",
                    "friction_slope_err", "velo_inerte_flag"):
            assert col in df.columns

    def test_sorted_by_galaxy(self, sparc_dir):
        df = build_f3_catalog(sparc_dir, verbose=False)
        assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())

    def test_velo_inerte_flag_binary(self, sparc_dir):
        df = build_f3_catalog(sparc_dir, verbose=False)
        assert set(df["velo_inerte_flag"].unique()).issubset({0, 1})

    def test_n_deep_integer(self, sparc_dir):
        df = build_f3_catalog(sparc_dir, verbose=False)
        assert pd.api.types.is_integer_dtype(df["n_deep"])


# ---------------------------------------------------------------------------
# Tests for gen_main (CLI)
# ---------------------------------------------------------------------------

class TestGenMain:
    @pytest.fixture(scope="class")
    def sparc_dir(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("sparc_gen")
        return _make_sparc_dir(root, n_gal=4, n_pts=15)

    def test_writes_csv(self, sparc_dir, tmp_path):
        out = tmp_path / "catalog.csv"
        gen_main(["--data-dir", str(sparc_dir), "--out", str(out), "--quiet"])
        assert out.exists()

    def test_csv_has_expected_columns(self, sparc_dir, tmp_path):
        out = tmp_path / "cat2.csv"
        gen_main(["--data-dir", str(sparc_dir), "--out", str(out), "--quiet"])
        df = pd.read_csv(out)
        for col in ("galaxy", "n_deep", "friction_slope",
                    "friction_slope_err", "velo_inerte_flag"):
            assert col in df.columns

    def test_missing_data_dir_raises(self, tmp_path):
        out = tmp_path / "x.csv"
        with pytest.raises(FileNotFoundError):
            gen_main(["--data-dir", str(tmp_path / "nonexistent"),
                      "--out", str(out), "--quiet"])


# ---------------------------------------------------------------------------
# Tests for analyze_catalog
# ---------------------------------------------------------------------------

class TestAnalyzeCatalog:
    def _make_catalog(self, n: int = 20, mean: float = 0.5,
                      std: float = 0.05, flag_frac: float = 0.8,
                      seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        slopes = rng.normal(mean, std, n)
        flags = (rng.uniform(size=n) < flag_frac).astype(int)
        return pd.DataFrame({
            "galaxy": [f"G{i:03d}" for i in range(n)],
            "n_deep": rng.integers(2, 20, n),
            "friction_slope": slopes,
            "friction_slope_err": rng.uniform(0.01, 0.05, n),
            "velo_inerte_flag": flags,
        })

    def test_returns_required_keys(self):
        df = self._make_catalog()
        stats = analyze_catalog(df)
        for key in ("n_analyzed", "mean_slope", "std_slope",
                    "n_consistent", "n_inconsistent", "p_value"):
            assert key in stats

    def test_n_analyzed_counts_finite_slopes(self):
        df = self._make_catalog(n=10)
        df.loc[0, "friction_slope"] = float("nan")
        stats = analyze_catalog(df)
        assert stats["n_analyzed"] == 9

    def test_consistent_plus_inconsistent_equals_total(self):
        df = self._make_catalog(n=15)
        stats = analyze_catalog(df)
        assert stats["n_consistent"] + stats["n_inconsistent"] == len(df)

    def test_empty_catalog(self):
        df = pd.DataFrame(columns=["galaxy", "n_deep", "friction_slope",
                                   "friction_slope_err", "velo_inerte_flag"])
        stats = analyze_catalog(df)
        assert stats["n_analyzed"] == 0
        assert np.isnan(stats["mean_slope"])
        assert np.isnan(stats["p_value"])

    def test_all_nan_slopes(self):
        df = pd.DataFrame({
            "galaxy": ["G1", "G2"],
            "n_deep": [0, 0],
            "friction_slope": [float("nan"), float("nan")],
            "friction_slope_err": [float("nan"), float("nan")],
            "velo_inerte_flag": [0, 0],
        })
        stats = analyze_catalog(df)
        assert stats["n_analyzed"] == 0

    def test_p_value_high_when_mean_is_05(self):
        """With μ ≈ 0.5 the null hypothesis (β = 0.5) should not be rejected."""
        df = self._make_catalog(n=50, mean=0.5, std=0.03, seed=99)
        stats = analyze_catalog(df)
        # p should be well above any reasonable significance level
        assert stats["p_value"] > 0.05

    def test_p_value_low_when_mean_deviates(self):
        """With μ ≈ 0.8 the one-sample t-test should reject β = 0.5."""
        df = self._make_catalog(n=50, mean=0.8, std=0.03, seed=77)
        stats = analyze_catalog(df)
        assert stats["p_value"] < 0.01


# ---------------------------------------------------------------------------
# Tests for format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_contains_all_labels(self):
        stats = {
            "n_analyzed": 20,
            "mean_slope": 0.505,
            "std_slope": 0.04,
            "n_consistent": 15,
            "n_inconsistent": 5,
            "p_value": 0.42,
        }
        text = format_summary(stats)
        assert "N galaxies analyzed:" in text
        assert "Mean friction_slope:" in text
        assert "Std friction_slope:" in text
        assert "Consistent (velo_inerte_flag=1):" in text
        assert "Inconsistent (velo_inerte_flag=0):" in text
        assert "p-value:" in text

    def test_nan_stats_printed_as_nan(self):
        stats = {
            "n_analyzed": 0,
            "mean_slope": float("nan"),
            "std_slope": float("nan"),
            "n_consistent": 0,
            "n_inconsistent": 2,
            "p_value": float("nan"),
        }
        text = format_summary(stats)
        assert "nan" in text


# ---------------------------------------------------------------------------
# Tests for ana_main (CLI)
# ---------------------------------------------------------------------------

class TestAnaMain:
    @pytest.fixture
    def catalog_csv(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 15
        slopes = rng.normal(0.5, 0.04, n)
        flags = (np.abs(slopes - 0.5) < 0.08).astype(int)
        df = pd.DataFrame({
            "galaxy": [f"G{i:03d}" for i in range(n)],
            "n_deep": rng.integers(5, 30, n),
            "friction_slope": slopes,
            "friction_slope_err": rng.uniform(0.01, 0.05, n),
            "velo_inerte_flag": flags,
        })
        p = tmp_path / "catalog.csv"
        df.to_csv(p, index=False)
        return p

    def test_returns_dict(self, catalog_csv):
        stats = ana_main(["--catalog", str(catalog_csv)])
        assert isinstance(stats, dict)

    def test_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ana_main(["--catalog", str(tmp_path / "nofile.csv")])

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["G1"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            ana_main(["--catalog", str(bad)])

    def test_output_contains_summary_lines(self, catalog_csv, capsys):
        ana_main(["--catalog", str(catalog_csv)])
        captured = capsys.readouterr()
        assert "N galaxies analyzed:" in captured.out
        assert "p-value:" in captured.out


# ---------------------------------------------------------------------------
# Integration: gen_main → ana_main end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_pipeline_produces_valid_stats(self, tmp_path):
        sparc = tmp_path / "sparc"
        sparc.mkdir()
        _make_sparc_dir(sparc, n_gal=6, n_pts=20)
        out = tmp_path / "f3_catalog.csv"
        gen_main(["--data-dir", str(sparc), "--out", str(out), "--quiet"])
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 6
        stats = ana_main(["--catalog", str(out)])
        assert stats["n_consistent"] + stats["n_inconsistent"] == len(df)
