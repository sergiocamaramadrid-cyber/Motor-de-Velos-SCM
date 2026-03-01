"""
tests/test_f3_catalog_analysis.py — Tests for scripts/f3_catalog_analysis.py.

Dedicated test module for the F3 catalog statistical analysis tool.

Covers:
  1. analyze_f3_catalog() — core statistics computation.
  2. format_analysis_report() — report formatting.
  3. main() CLI — end-to-end invocation.
  4. Integration with the committed synthetic CI fixture
     (results/f3_catalog_synthetic_flat.csv).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.f3_catalog_analysis import (
    EXPECTED_BETA_MOND,
    analyze_f3_catalog,
    format_analysis_report,
    main,
    _parse_args,
)

# ---------------------------------------------------------------------------
# Path to the committed synthetic CI fixture
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
FIXTURE_PATH = _REPO_ROOT / "results" / "f3_catalog_synthetic_flat.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    n: int = 10,
    beta: float = 1.0,
    reliable_all: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal synthetic F3 catalog for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "beta": beta + rng.normal(0, 0.02, n),
        "beta_err": np.full(n, 0.01),
        "intercept": np.zeros(n),
        "r_value": np.full(n, 0.99),
        "p_value": np.full(n, 1e-20),
        "n_deep": np.full(n, 15),
        "n_total": np.full(n, 30),
        "reliable": np.ones(n, dtype=bool) if reliable_all
                    else np.array([i % 2 == 0 for i in range(n)]),
    })


# ---------------------------------------------------------------------------
# 1. analyze_f3_catalog()
# ---------------------------------------------------------------------------

class TestAnalyzeF3Catalog:
    def test_returns_required_keys(self):
        stats = analyze_f3_catalog(_make_catalog())
        required = {
            "n_galaxies", "n_reliable", "beta_mean", "beta_median",
            "beta_std", "beta_mean_all", "delta_from_mond", "consistent_mond",
        }
        assert required.issubset(set(stats.keys()))

    def test_n_galaxies_matches_input_length(self):
        df = _make_catalog(n=15)
        stats = analyze_f3_catalog(df)
        assert stats["n_galaxies"] == 15

    def test_n_reliable_counts_only_reliable_rows(self):
        df = _make_catalog(n=10, reliable_all=False)  # 5 reliable, 5 not
        stats = analyze_f3_catalog(df)
        assert stats["n_reliable"] == 5

    def test_beta_mean_close_to_planted_value(self):
        beta_planted = 0.75
        df = _make_catalog(n=50, beta=beta_planted)
        stats = analyze_f3_catalog(df)
        assert stats["beta_mean"] == pytest.approx(beta_planted, abs=0.05)
        assert stats["beta_median"] == pytest.approx(beta_planted, abs=0.05)

    def test_delta_from_mond_is_median_minus_half(self):
        df = _make_catalog(n=20, beta=0.80, seed=7)
        stats = analyze_f3_catalog(df)
        expected_delta = stats["beta_median"] - EXPECTED_BETA_MOND
        assert stats["delta_from_mond"] == pytest.approx(expected_delta, abs=1e-9)

    def test_consistent_mond_true_when_beta_near_half(self):
        """β ≈ 0.5 (MOND/physical) must be flagged as MOND-consistent."""
        df = _make_catalog(n=30, beta=0.5)
        stats = analyze_f3_catalog(df)
        assert stats["consistent_mond"] is True

    def test_consistent_mond_false_when_beta_near_one(self):
        """β ≈ 1.0 (synthetic flat) must NOT be flagged as MOND-consistent."""
        df = _make_catalog(n=30, beta=1.0)
        stats = analyze_f3_catalog(df)
        assert stats["consistent_mond"] is False

    def test_nan_stats_when_no_reliable_galaxies(self):
        df = _make_catalog(n=5, reliable_all=True)
        df["reliable"] = False
        stats = analyze_f3_catalog(df)
        assert stats["n_reliable"] == 0
        assert math.isnan(stats["beta_mean"])
        assert math.isnan(stats["beta_median"])
        assert stats["consistent_mond"] is False

    def test_missing_beta_column_raises(self):
        bad = pd.DataFrame({"reliable": [True], "n_deep": [10]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_f3_catalog(bad)

    def test_missing_reliable_column_raises(self):
        bad = pd.DataFrame({"beta": [1.0], "n_deep": [10]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_f3_catalog(bad)

    def test_beta_mean_all_includes_unreliable(self):
        """beta_mean_all should average beta over all non-NaN rows."""
        rng = np.random.default_rng(11)
        betas = np.concatenate([
            0.5 + rng.normal(0, 0.01, 5),   # reliable
            1.0 + rng.normal(0, 0.01, 5),   # unreliable
        ])
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(10)],
            "beta": betas,
            "reliable": [True] * 5 + [False] * 5,
        })
        stats = analyze_f3_catalog(df)
        assert stats["beta_mean_all"] == pytest.approx(betas.mean(), abs=1e-9)
        # reliable mean should differ from all mean
        assert stats["beta_mean"] != pytest.approx(stats["beta_mean_all"], abs=0.1)

    def test_single_galaxy_catalog(self):
        """Single-galaxy catalog must not raise."""
        df = _make_catalog(n=1, beta=0.5)
        stats = analyze_f3_catalog(df)
        assert stats["n_galaxies"] == 1
        assert stats["n_reliable"] == 1
        assert stats["beta_median"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 2. format_analysis_report()
# ---------------------------------------------------------------------------

class TestFormatAnalysisReport:
    def test_report_contains_catalog_path(self):
        stats = analyze_f3_catalog(_make_catalog())
        lines = format_analysis_report(stats, "my/catalog.csv")
        combined = "\n".join(lines)
        assert "my/catalog.csv" in combined

    def test_report_contains_beta_statistics(self):
        stats = analyze_f3_catalog(_make_catalog(n=20, beta=1.0))
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        assert "β median" in combined
        assert "β mean" in combined
        assert "β std-dev" in combined

    def test_report_verdict_present(self):
        stats = analyze_f3_catalog(_make_catalog(n=10))
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        assert "Verdict" in combined

    def test_report_consistent_mond_message(self):
        """When β ≈ 0.5, report verdict must indicate MOND consistency."""
        df = _make_catalog(n=20, beta=0.5)
        stats = analyze_f3_catalog(df)
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        # Should mention MOND or deep-velos consistency
        assert "consistent" in combined.lower() or "0.5" in combined

    def test_report_deviates_from_mond_message(self):
        """When β ≈ 1.0, report verdict must note deviation from MOND."""
        df = _make_catalog(n=20, beta=1.0)
        stats = analyze_f3_catalog(df)
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        assert "deviates" in combined or "0.5" in combined

    def test_report_warning_when_no_reliable(self):
        df = _make_catalog(n=5)
        df["reliable"] = False
        stats = analyze_f3_catalog(df)
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        assert "reliable" in combined.lower() or "insufficient" in combined.lower()

    def test_report_n_galaxies_shown(self):
        df = _make_catalog(n=7)
        stats = analyze_f3_catalog(df)
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        assert "7" in combined  # n_galaxies


# ---------------------------------------------------------------------------
# 3. main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--catalog", str(tmp_path / "missing.csv")])

    def test_returns_stats_dict(self, tmp_path):
        catalog = tmp_path / "catalog.csv"
        _make_catalog(n=5).to_csv(catalog, index=False)
        result = main(["--catalog", str(catalog)])
        assert isinstance(result, dict)
        assert "beta_median" in result

    def test_default_catalog_path_is_f3_catalog_real(self):
        """Default --catalog must point to f3_catalog_real.csv."""
        args = _parse_args([])
        assert "f3_catalog_real.csv" in args.catalog

    def test_writes_output_csv_and_log(self, tmp_path):
        catalog = tmp_path / "catalog.csv"
        _make_catalog(n=5).to_csv(catalog, index=False)
        out_dir = tmp_path / "out"
        main(["--catalog", str(catalog), "--out", str(out_dir)])
        assert (out_dir / "f3_analysis.csv").exists()
        assert (out_dir / "f3_analysis.log").exists()

    def test_output_csv_contains_stats_row(self, tmp_path):
        catalog = tmp_path / "catalog.csv"
        df = _make_catalog(n=8, beta=0.5)
        df.to_csv(catalog, index=False)
        out_dir = tmp_path / "out"
        main(["--catalog", str(catalog), "--out", str(out_dir)])
        result_df = pd.read_csv(out_dir / "f3_analysis.csv")
        assert "beta_median" in result_df.columns
        assert len(result_df) == 1

    def test_output_log_contains_verdict(self, tmp_path):
        catalog = tmp_path / "catalog.csv"
        _make_catalog(n=8).to_csv(catalog, index=False)
        out_dir = tmp_path / "out"
        main(["--catalog", str(catalog), "--out", str(out_dir)])
        log_text = (out_dir / "f3_analysis.log").read_text(encoding="utf-8")
        assert "Verdict" in log_text

    def test_runs_without_output_dir(self, tmp_path):
        """Running without --out must succeed (no files written)."""
        catalog = tmp_path / "catalog.csv"
        _make_catalog(n=5).to_csv(catalog, index=False)
        result = main(["--catalog", str(catalog)])
        assert "n_galaxies" in result


# ---------------------------------------------------------------------------
# 4. Integration: synthetic CI fixture
# ---------------------------------------------------------------------------

class TestSyntheticFixtureIntegration:
    """Run the analysis on the committed synthetic CI fixture and verify
    that the tool reports β ≈ 1.0 (not MOND-consistent) as expected for
    flat-rotation-curve synthetic data.
    """

    def test_fixture_exists(self):
        assert FIXTURE_PATH.exists(), (
            f"CI fixture not found: {FIXTURE_PATH}\n"
            "Run 'python scripts/generate_f3_catalog.py' to regenerate it."
        )

    def test_analysis_runs_on_fixture(self):
        """main() must complete without error on the synthetic fixture."""
        result = main(["--catalog", str(FIXTURE_PATH)])
        assert isinstance(result, dict)

    def test_fixture_reports_beta_near_one(self):
        """Synthetic fixture → analysis must report β ≈ 1 (not 0.5)."""
        result = main(["--catalog", str(FIXTURE_PATH)])
        assert result["n_reliable"] > 0, "No reliable galaxies in fixture"
        assert result["beta_median"] == pytest.approx(1.0, abs=0.10)

    def test_fixture_not_mond_consistent(self):
        """Synthetic fixture must NOT be flagged MOND-consistent (β ≠ 0.5)."""
        result = main(["--catalog", str(FIXTURE_PATH)])
        assert result["consistent_mond"] is False, (
            "Synthetic CI fixture should NOT be MOND-consistent; "
            "only real SPARC data produces β ≈ 0.5."
        )

    def test_fixture_delta_from_mond_positive(self):
        """Fixture β > 0.5 so delta_from_mond must be clearly positive."""
        result = main(["--catalog", str(FIXTURE_PATH)])
        assert result["delta_from_mond"] > 0.3, (
            f"Expected delta > 0.3 for synthetic fixture, "
            f"got {result['delta_from_mond']:.3f}"
        )

    def test_analysis_output_files_from_fixture(self, tmp_path):
        """Writing analysis output from the fixture must produce valid files."""
        out_dir = tmp_path / "out"
        main(["--catalog", str(FIXTURE_PATH), "--out", str(out_dir)])
        assert (out_dir / "f3_analysis.csv").exists()
        assert (out_dir / "f3_analysis.log").exists()
        stats_df = pd.read_csv(out_dir / "f3_analysis.csv")
        assert stats_df["n_galaxies"].iloc[0] == 20
