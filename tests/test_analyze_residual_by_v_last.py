"""
tests/test_analyze_residual_by_v_last.py — Tests for the Mann-Whitney U
analysis script (analyze_residual_by_v_last.py).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_residual_by_v_last import (
    analyze_residual_by_v_last,
    load_residual_catalog,
    print_results,
    REQUIRED_COLS,
    main as analyze_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    n: int = 20,
    low_residual: float = -0.2,
    high_residual: float = 0.3,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a synthetic catalog where low-v_last group has lower residuals.

    The first ``n // 2`` rows get v_last < median and residuals drawn around
    *low_residual*; the remainder get v_last > median and residuals around
    *high_residual*.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "f3_residual": np.concatenate([
            rng.normal(low_residual, 0.05, half),
            rng.normal(high_residual, 0.05, n - half),
        ]),
        "v_last": np.concatenate([
            rng.uniform(50, 100, half),
            rng.uniform(150, 250, n - half),
        ]),
    })


# ---------------------------------------------------------------------------
# load_residual_catalog tests
# ---------------------------------------------------------------------------

class TestLoadResidualCatalog:
    def test_loads_valid_csv(self, tmp_path):
        df = _make_catalog()
        csv_path = tmp_path / "catalog.csv"
        df.to_csv(csv_path, index=False)
        loaded = load_residual_catalog(csv_path)
        assert len(loaded) == len(df)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_residual_catalog(tmp_path / "missing.csv")

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"galaxy": ["A"], "v_last": [100.0]})
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_residual_catalog(csv_path)

    def test_all_required_columns_present(self, tmp_path):
        df = _make_catalog()
        csv_path = tmp_path / "catalog.csv"
        df.to_csv(csv_path, index=False)
        loaded = load_residual_catalog(csv_path)
        for col in REQUIRED_COLS:
            assert col in loaded.columns


# ---------------------------------------------------------------------------
# analyze_residual_by_v_last unit tests
# ---------------------------------------------------------------------------

class TestAnalyzeResidualByVLast:
    def test_returns_required_keys(self):
        df = _make_catalog()
        result = analyze_residual_by_v_last(df)
        required = {
            "n_total", "n_low", "n_high",
            "v_last_median", "median_low", "median_high",
            "statistic", "p_value",
        }
        assert required.issubset(result.keys())

    def test_p_value_in_unit_interval(self):
        df = _make_catalog()
        result = analyze_residual_by_v_last(df)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_statistic_is_non_negative(self):
        df = _make_catalog()
        result = analyze_residual_by_v_last(df)
        assert result["statistic"] >= 0.0

    def test_n_low_plus_n_high_equals_n_total(self):
        df = _make_catalog(n=20)
        result = analyze_residual_by_v_last(df)
        assert result["n_low"] + result["n_high"] == result["n_total"]

    def test_n_total_excludes_nonfinite_rows(self):
        df = _make_catalog(n=20)
        df.loc[0, "f3_residual"] = float("nan")
        df.loc[1, "v_last"] = float("inf")
        result = analyze_residual_by_v_last(df)
        assert result["n_total"] == 18

    def test_significant_difference_detected(self):
        """Groups with clearly separated residuals should yield small p."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(40)],
            "f3_residual": np.concatenate([
                rng.normal(-0.5, 0.05, 20),
                rng.normal(0.5, 0.05, 20),
            ]),
            "v_last": np.concatenate([
                rng.uniform(50, 100, 20),
                rng.uniform(200, 300, 20),
            ]),
        })
        result = analyze_residual_by_v_last(df)
        assert result["p_value"] < 0.05

    def test_no_difference_yields_large_p(self):
        """Groups with identical residual distributions should yield large p."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(100)],
            "f3_residual": rng.normal(0.0, 0.1, 100),
            "v_last": rng.uniform(50, 300, 100),
        })
        result = analyze_residual_by_v_last(df)
        assert result["p_value"] > 0.05

    def test_v_last_median_is_split_threshold(self):
        """The split threshold must equal the median of v_last."""
        df = _make_catalog(n=30)
        result = analyze_residual_by_v_last(df)
        expected_median = float(df["v_last"].median())
        assert result["v_last_median"] == pytest.approx(expected_median, rel=1e-9)

    def test_median_low_below_median_high(self):
        """For the skewed catalog, low-v group has lower median residual."""
        df = _make_catalog(n=40, low_residual=-0.4, high_residual=0.4)
        result = analyze_residual_by_v_last(df)
        assert result["median_low"] < result["median_high"]

    def test_all_nonfinite_rows_drops_all(self):
        """Catalog with all NaN residuals → empty df → ValueError."""
        df = pd.DataFrame({
            "f3_residual": [float("nan"), float("nan")],
            "v_last": [100.0, 200.0],
        })
        with pytest.raises(ValueError, match="No finite rows"):
            analyze_residual_by_v_last(df)

    def test_handles_single_valid_column_name_f3_residual(self):
        """Column must be named exactly f3_residual (no aliases)."""
        df = pd.DataFrame({
            "residual": [0.1, -0.1, 0.2, -0.2],
            "v_last": [100.0, 150.0, 200.0, 250.0],
        })
        with pytest.raises(KeyError):
            analyze_residual_by_v_last(df)


# ---------------------------------------------------------------------------
# print_results smoke tests
# ---------------------------------------------------------------------------

class TestPrintResults:
    def test_runs_without_error(self, capsys):
        df = _make_catalog()
        result = analyze_residual_by_v_last(df)
        print_results(result)
        captured = capsys.readouterr()
        assert "p-value" in captured.out
        assert "Mann-Whitney" in captured.out

    def test_significant_output_contains_marker(self, capsys):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "f3_residual": np.concatenate([
                rng.normal(-1.0, 0.1, 30),
                rng.normal(1.0, 0.1, 30),
            ]),
            "v_last": np.concatenate([
                rng.uniform(50, 100, 30),
                rng.uniform(200, 300, 30),
            ]),
        })
        result = analyze_residual_by_v_last(df)
        print_results(result)
        captured = capsys.readouterr()
        assert "Significant" in captured.out or "significant" in captured.out


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestAnalyzeMain:
    def test_main_returns_dict(self, tmp_path):
        df = _make_catalog()
        csv_path = tmp_path / "catalog.csv"
        df.to_csv(csv_path, index=False)
        result = analyze_main(["--input", str(csv_path)])
        assert isinstance(result, dict)
        assert "p_value" in result

    def test_main_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_main(["--input", str(tmp_path / "nonexistent.csv")])

    def test_main_default_input_path_is_correct(self):
        from scripts.analyze_residual_by_v_last import _parse_args, DEFAULT_INPUT
        args = _parse_args([])
        assert args.input == DEFAULT_INPUT
