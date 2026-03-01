"""
tests/test_f3_catalog_analysis.py — Tests for scripts/f3_catalog_analysis.py.

Validates ensemble statistics, t-test computation, and CLI behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.f3_catalog_analysis import (
    analyze_catalog,
    format_report,
    main,
    REF_SLOPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(
    tmp_path: Path,
    n: int = 20,
    mean_slope: float = 0.5,
    std_slope: float = 0.04,
    seed: int = 0,
    include_flag: bool = True,
) -> Path:
    """Create a synthetic per-galaxy F3 catalog CSV."""
    rng = np.random.default_rng(seed)
    slopes = mean_slope + rng.normal(0, std_slope, n)
    errs = rng.uniform(0.01, 0.05, n)
    flags = (np.abs(slopes - REF_SLOPE) <= 2 * errs).astype(float) if include_flag else None

    data = {
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "n_total": rng.integers(30, 80, n),
        "n_deep": rng.integers(10, 30, n),
        "friction_slope": np.round(slopes, 6),
        "friction_slope_err": np.round(errs, 6),
        "r_value": rng.uniform(0.90, 0.999, n).round(6),
        "p_value": rng.uniform(1e-10, 1e-4, n).round(12),
    }
    if include_flag:
        data["velo_inerte_flag"] = flags

    df = pd.DataFrame(data)
    p = tmp_path / "f3_catalog.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Unit tests for analyze_catalog()
# ---------------------------------------------------------------------------


class TestAnalyzeCatalog:
    def test_returns_required_keys(self, tmp_path):
        p = _make_catalog(tmp_path, n=10)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        required = {
            "n_galaxies", "n_fitted", "mean_slope", "std_slope",
            "median_slope", "t_stat", "p_value",
            "n_consistent", "n_inconsistent", "velo_inerte_frac",
            "ref_slope",
        }
        assert required.issubset(set(stats.keys()))

    def test_n_fitted_matches_non_nan(self, tmp_path):
        p = _make_catalog(tmp_path, n=15)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        expected = int(catalog["friction_slope"].notna().sum())
        assert stats["n_fitted"] == expected

    def test_mean_slope_close_to_planted(self, tmp_path):
        """With 50 galaxies, mean should be close to the planted value."""
        p = _make_catalog(tmp_path, n=50, mean_slope=0.5, std_slope=0.02, seed=1)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        assert stats["mean_slope"] == pytest.approx(0.5, abs=0.05)

    def test_ttest_pvalue_high_for_mond_consistent(self, tmp_path):
        """When mean β ≈ 0.5 the t-test p-value should be > 0.05."""
        p = _make_catalog(tmp_path, n=50, mean_slope=0.5, std_slope=0.02, seed=2)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog, ref_slope=0.5)
        assert stats["p_value"] >= 0.05

    def test_ttest_pvalue_low_for_deviant(self, tmp_path):
        """When mean β ≠ 0.5, the t-test p-value should be < 0.05."""
        rng = np.random.default_rng(3)
        n = 50
        # Plant slope far from 0.5
        catalog = pd.DataFrame({
            "friction_slope": 0.2 + rng.normal(0, 0.01, n),
            "velo_inerte_flag": np.zeros(n),
        })
        stats = analyze_catalog(catalog, ref_slope=0.5)
        assert stats["p_value"] < 0.05

    def test_nan_when_no_fitted_galaxies(self):
        catalog = pd.DataFrame({"friction_slope": [float("nan")] * 5})
        stats = analyze_catalog(catalog)
        assert stats["n_fitted"] == 0
        assert np.isnan(stats["mean_slope"])
        assert np.isnan(stats["t_stat"])

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"galaxy": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_catalog(df)

    def test_ref_slope_stored_in_result(self, tmp_path):
        p = _make_catalog(tmp_path, n=10)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog, ref_slope=0.42)
        assert stats["ref_slope"] == 0.42

    def test_velo_inerte_frac_between_zero_and_one(self, tmp_path):
        p = _make_catalog(tmp_path, n=20, include_flag=True)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        assert 0.0 <= stats["velo_inerte_frac"] <= 1.0

    def test_n_consistent_plus_n_inconsistent_le_n_fitted(self, tmp_path):
        p = _make_catalog(tmp_path, n=20, include_flag=True)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        assert stats["n_consistent"] + stats["n_inconsistent"] <= stats["n_fitted"]
        assert stats["n_consistent"] >= 0
        assert stats["n_inconsistent"] >= 0


# ---------------------------------------------------------------------------
# format_report tests
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_report_contains_key_fields(self, tmp_path):
        p = _make_catalog(tmp_path, n=10)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        lines = format_report(stats)
        combined = "\n".join(lines)
        assert "Mean β" in combined
        assert "t-statistic" in combined
        assert "p-value" in combined
        assert "n_consistent" in combined
        assert "n_inconsistent" in combined

    def test_report_contains_flag_semantics_legend(self, tmp_path):
        """Report must include the velo_inerte_flag semantics legend."""
        p = _make_catalog(tmp_path, n=10)
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        lines = format_report(stats)
        combined = "\n".join(lines)
        assert "velo_inerte_flag semantics (ref slope" in combined
        assert "consistent with reference prediction" in combined
        assert "significant deviation from reference" in combined

    def test_report_handles_zero_fitted(self):
        stats = {
            "n_galaxies": 5, "n_fitted": 0,
            "mean_slope": float("nan"), "std_slope": float("nan"),
            "median_slope": float("nan"), "t_stat": float("nan"),
            "p_value": float("nan"),
            "n_consistent": 0, "n_inconsistent": 0,
            "velo_inerte_frac": float("nan"),
            "ref_slope": REF_SLOPE,
        }
        lines = format_report(stats)
        combined = "\n".join(lines)
        assert "No fitted galaxies" in combined


# ---------------------------------------------------------------------------
# CLI (main) integration tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_produces_stats_file(self, tmp_path):
        cat = _make_catalog(tmp_path, n=10)
        out = tmp_path / "stats.csv"
        stats = main(["--catalog", str(cat), "--out", str(out)])
        assert out.exists()
        assert "mean_slope" in stats

    def test_returns_dict_with_correct_keys(self, tmp_path):
        cat = _make_catalog(tmp_path, n=10)
        out = tmp_path / "stats.csv"
        stats = main(["--catalog", str(cat), "--out", str(out)])
        required = {"n_galaxies", "n_fitted", "mean_slope", "t_stat", "p_value"}
        assert required.issubset(set(stats.keys()))

    def test_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--catalog", str(tmp_path / "no.csv")])

    def test_ref_slope_override(self, tmp_path):
        cat = _make_catalog(tmp_path, n=15)
        out = tmp_path / "stats.csv"
        stats = main(["--catalog", str(cat), "--out", str(out),
                      "--ref-slope", "0.42"])
        assert stats["ref_slope"] == pytest.approx(0.42)

    def test_auto_derives_out_path_from_catalog(self, tmp_path):
        """When --out is omitted, stats go to <catalog_stem>_stats.csv."""
        cat = _make_catalog(tmp_path, n=10)
        # _make_catalog writes tmp_path/f3_catalog.csv → auto out: tmp_path/f3_catalog_stats.csv
        expected_out = tmp_path / "f3_catalog_stats.csv"
        main(["--catalog", str(cat)])
        assert expected_out.exists()

    def test_reference_catalog_passes_analysis(self):
        """The committed results/f3_catalog.csv must pass the analysis cleanly."""
        p = Path("results/f3_catalog.csv")
        assert p.exists(), "results/f3_catalog.csv not found"
        catalog = pd.read_csv(p)
        stats = analyze_catalog(catalog)
        assert stats["n_fitted"] > 0
        assert np.isfinite(stats["mean_slope"])
        assert np.isfinite(stats["t_stat"])
