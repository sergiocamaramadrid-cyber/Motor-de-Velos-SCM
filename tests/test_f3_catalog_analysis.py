"""
tests/test_f3_catalog_analysis.py — Tests for scripts/f3_catalog_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.f3_catalog_analysis import (
    analyze_catalog,
    print_summary,
    main,
    BETA_REF,
    REQUIRED_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(tmp_path: Path, slopes, flags) -> Path:
    """Write a minimal catalog CSV."""
    df = pd.DataFrame({
        "galaxy_id": [f"G{i:03d}" for i in range(len(slopes))],
        "friction_slope": slopes,
        "flag": flags,
    })
    p = tmp_path / "catalog.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# analyze_catalog
# ---------------------------------------------------------------------------

class TestAnalyzeCatalog:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_catalog(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"galaxy_id": ["G0"], "friction_slope": [0.5]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_catalog(p)

    def test_total_rows(self, tmp_path):
        p = _make_catalog(tmp_path, [0.5, 0.6, 0.4], [1, 0, 1])
        r = analyze_catalog(p)
        assert r["total_rows"] == 3

    def test_n_valid_excludes_nan(self, tmp_path):
        p = _make_catalog(tmp_path, [0.5, float("nan"), 0.4], [1, float("nan"), 1])
        r = analyze_catalog(p)
        assert r["n_valid"] == 2
        assert r["n_nan"] == 1

    def test_consistent_inconsistent_counts(self, tmp_path):
        p = _make_catalog(tmp_path, [0.5, 0.9, 0.48], [1, 0, 1])
        r = analyze_catalog(p)
        assert r["n_consistent"] == 2
        assert r["n_inconsistent"] == 1

    def test_mean_and_std(self, tmp_path):
        slopes = [0.4, 0.5, 0.6]
        p = _make_catalog(tmp_path, slopes, [1, 1, 1])
        r = analyze_catalog(p)
        assert r["mean_slope"] == pytest.approx(0.5, abs=1e-6)
        assert r["std_slope"] == pytest.approx(np.std(slopes, ddof=1), abs=1e-6)

    def test_t_stat_near_zero_when_mean_near_beta_ref(self, tmp_path):
        # Slopes symmetrically around BETA_REF → mean == BETA_REF → t-stat near 0
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, 40)
        slopes = (BETA_REF + noise).tolist()
        p = _make_catalog(tmp_path, slopes, [1] * 40)
        r = analyze_catalog(p)
        assert abs(r["t_stat"]) < 3.0  # mean is very close to BETA_REF

    def test_nan_stats_when_all_nan(self, tmp_path):
        p = _make_catalog(tmp_path, [float("nan"), float("nan")],
                          [float("nan"), float("nan")])
        r = analyze_catalog(p)
        assert math.isnan(r["mean_slope"])
        assert math.isnan(r["t_stat"])
        assert math.isnan(r["p_val"])

    def test_single_valid_slope(self, tmp_path):
        p = _make_catalog(tmp_path, [0.6], [0])
        r = analyze_catalog(p)
        assert r["n_valid"] == 1
        assert r["mean_slope"] == pytest.approx(0.6, abs=1e-6)
        assert math.isnan(r["std_slope"])
        assert math.isnan(r["t_stat"])

    def test_returns_required_keys(self, tmp_path):
        p = _make_catalog(tmp_path, [0.5, 0.6], [1, 0])
        r = analyze_catalog(p)
        required = {
            "total_rows", "n_valid", "n_consistent", "n_inconsistent",
            "n_nan", "mean_slope", "std_slope", "t_stat", "p_val",
        }
        assert required.issubset(set(r.keys()))


# ---------------------------------------------------------------------------
# print_summary (smoke test — just check it doesn't raise)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_runs_without_error(self, capsys):
        result = {
            "total_rows": 10,
            "n_valid": 9,
            "n_consistent": 3,
            "n_inconsistent": 6,
            "n_nan": 1,
            "mean_slope": 0.75,
            "std_slope": 0.1,
            "t_stat": 12.3,
            "p_val": 0.0001,
        }
        print_summary(result)
        captured = capsys.readouterr()
        assert "F3 CATALOG ANALYSIS SUMMARY" in captured.out
        assert "β=0.5" in captured.out
        assert "Mean friction_slope" in captured.out

    def test_nan_values_displayed_as_nan(self, capsys):
        result = {
            "total_rows": 1,
            "n_valid": 0,
            "n_consistent": 0,
            "n_inconsistent": 0,
            "n_nan": 1,
            "mean_slope": float("nan"),
            "std_slope": float("nan"),
            "t_stat": float("nan"),
            "p_val": float("nan"),
        }
        print_summary(result)
        captured = capsys.readouterr()
        assert "nan" in captured.out


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_result_dict(self, tmp_path):
        p = _make_catalog(tmp_path, [0.5, 0.6, 0.4], [1, 0, 1])
        result = main(["--catalog", str(p)])
        assert isinstance(result, dict)
        assert result["total_rows"] == 3

    def test_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--catalog", str(tmp_path / "missing.csv")])

    def test_output_contains_summary_header(self, tmp_path, capsys):
        p = _make_catalog(tmp_path, [0.5, 0.55, 0.45, 0.52], [1, 1, 1, 1])
        main(["--catalog", str(p)])
        captured = capsys.readouterr()
        assert "=== F3 CATALOG ANALYSIS SUMMARY ===" in captured.out


# ---------------------------------------------------------------------------
# Integration: real catalog
# ---------------------------------------------------------------------------

class TestRealCatalog:
    """Smoke-test against results/f3_catalog_real.csv if it exists."""

    def test_real_catalog_loads(self):
        catalog = Path("results/f3_catalog_real.csv")
        if not catalog.exists():
            pytest.skip("results/f3_catalog_real.csv not present")
        result = analyze_catalog(catalog)
        assert result["total_rows"] > 0
        assert result["n_valid"] + result["n_nan"] == result["total_rows"]
        assert result["n_consistent"] + result["n_inconsistent"] == result["n_valid"]
