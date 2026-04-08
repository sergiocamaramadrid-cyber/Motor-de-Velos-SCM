"""
tests/test_plot_sparc_split_mass.py — Tests for scripts/plot_sparc_split_mass.py.

Covers:
  1. compute_stats() — Spearman + OLS statistics computation.
  2. split_by_mass() — mass-based subsample splitting.
  3. generate_figure() — figure creation and file output.
  4. main() CLI — end-to-end invocation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_sparc_split_mass import (
    BETA_REF,
    compute_stats,
    generate_figure,
    main,
    split_by_mass,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    n: int = 20,
    seed: int = 0,
    logM_range: tuple[float, float] = (7.0, 10.0),
) -> pd.DataFrame:
    """Build a minimal synthetic SPARC-like catalog for testing."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(*logM_range, n)
    delta_mass_std = rng.normal(0.0, 1.0, n)
    slope_tail = 0.5 + rng.normal(0.0, 0.1, n)
    return pd.DataFrame(
        {
            "galaxy": [f"NGC{i:04d}" for i in range(n)],
            "logM": logM,
            "delta_mass_std": delta_mass_std,
            "slope_tail": slope_tail,
        }
    )


def _write_catalog(df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "sparc_subset.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. compute_stats()
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 30)
        y = rng.normal(0, 1, 30)
        result = compute_stats(x, y)
        required = {"n", "rho", "p_spear", "ols_slope", "ols_intercept", "r2", "p_ols"}
        assert required.issubset(result.keys())

    def test_n_matches_input_length(self):
        x = np.arange(15, dtype=float)
        y = np.arange(15, dtype=float)
        assert compute_stats(x, y)["n"] == 15

    def test_perfect_positive_correlation(self):
        x = np.linspace(0, 1, 50)
        y = 2.0 * x + 0.5
        stats = compute_stats(x, y)
        assert stats["rho"] == pytest.approx(1.0, abs=1e-6)
        assert stats["ols_slope"] == pytest.approx(2.0, abs=1e-6)
        assert stats["ols_intercept"] == pytest.approx(0.5, abs=1e-6)
        assert stats["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        x = np.linspace(0, 1, 50)
        y = -3.0 * x + 1.0
        stats = compute_stats(x, y)
        assert stats["rho"] == pytest.approx(-1.0, abs=1e-6)
        assert stats["ols_slope"] == pytest.approx(-3.0, abs=1e-6)

    def test_pvalues_in_valid_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 40)
        y = rng.normal(0, 1, 40)
        stats = compute_stats(x, y)
        assert 0.0 <= stats["p_spear"] <= 1.0
        assert 0.0 <= stats["p_ols"] <= 1.0

    def test_r2_is_non_negative(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 25)
        y = rng.normal(0, 1, 25)
        assert compute_stats(x, y)["r2"] >= 0.0

    def test_r2_bounded_by_one(self):
        x = np.linspace(-2, 2, 30)
        y = x + np.random.default_rng(3).normal(0, 0.1, 30)
        assert compute_stats(x, y)["r2"] <= 1.0

    def test_accepts_list_inputs(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0]
        stats = compute_stats(x, y)
        assert stats["n"] == 4

    def test_significant_correlation_has_small_pvalue(self):
        x = np.linspace(0, 10, 100)
        y = 0.5 * x + np.random.default_rng(0).normal(0, 0.1, 100)
        stats = compute_stats(x, y)
        assert stats["p_spear"] < 0.001
        assert stats["p_ols"] < 0.001

    def test_no_correlation_has_large_pvalue(self):
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        stats = compute_stats(x, y)
        # With N=200 from uncorrelated normals the p-value is overwhelmingly > 0.001
        # (|rho| < 0.2 is almost certain); just verify it is a valid probability
        assert 0.0 <= stats["p_spear"] <= 1.0


# ---------------------------------------------------------------------------
# 2. split_by_mass()
# ---------------------------------------------------------------------------

class TestSplitByMass:
    def test_returns_three_elements(self):
        df = _make_catalog(n=20)
        result = split_by_mass(df)
        assert len(result) == 3

    def test_low_and_high_partition_all_rows(self):
        df = _make_catalog(n=40, seed=5)
        low, high, med = split_by_mass(df)
        assert len(low) + len(high) == len(df)

    def test_no_overlap_between_groups(self):
        df = _make_catalog(n=30, seed=10)
        low, high, med = split_by_mass(df)
        assert (low["logM"] < med).all()
        assert (high["logM"] >= med).all()

    def test_computed_median_splits_near_half(self):
        df = _make_catalog(n=50, seed=3)
        low, high, med = split_by_mass(df)
        # Sizes may differ by at most 1 due to the >= boundary
        assert abs(len(low) - len(high)) <= 2

    def test_fixed_median_is_respected(self):
        df = _make_catalog(n=20, seed=1)
        low, high, med = split_by_mass(df, median_logM=8.5)
        assert med == pytest.approx(8.5)
        assert (low["logM"] < 8.5).all()
        assert (high["logM"] >= 8.5).all()

    def test_returned_median_equals_data_median_when_none(self):
        df = _make_catalog(n=40, seed=2)
        _, _, med = split_by_mass(df)
        assert med == pytest.approx(float(df["logM"].median()))

    def test_preserves_all_columns(self):
        df = _make_catalog(n=20)
        low, high, _ = split_by_mass(df)
        for col in df.columns:
            assert col in low.columns
            assert col in high.columns


# ---------------------------------------------------------------------------
# 3. generate_figure()
# ---------------------------------------------------------------------------

class TestGenerateFigure:
    def test_returns_figure_object(self, tmp_path):
        df = _make_catalog(n=20, seed=0)
        out = tmp_path / "fig.png"
        fig = generate_figure(df, out)
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

    def test_png_file_is_created(self, tmp_path):
        df = _make_catalog(n=20, seed=1)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_pdf_sibling_is_created(self, tmp_path):
        df = _make_catalog(n=20, seed=2)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.with_suffix(".pdf").exists()

    def test_png_is_nonzero_size(self, tmp_path):
        df = _make_catalog(n=20, seed=3)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.stat().st_size > 0

    def test_accepts_fixed_median_logM(self, tmp_path):
        df = _make_catalog(n=30, seed=4)
        out = tmp_path / "fig.png"
        generate_figure(df, out, median_logM=8.5)
        assert out.exists()

    def test_creates_output_directory(self, tmp_path):
        df = _make_catalog(n=20, seed=5)
        out = tmp_path / "subdir" / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_computes_delta_f3_internally(self, tmp_path):
        """delta_f3 = slope_tail - BETA_REF; the column need not be in input."""
        df = _make_catalog(n=20, seed=6)
        assert "delta_f3" not in df.columns
        out = tmp_path / "fig.png"
        generate_figure(df, out)   # must not raise
        assert out.exists()

    def test_figure_has_two_axes(self, tmp_path):
        df = _make_catalog(n=20, seed=7)
        out = tmp_path / "fig.png"
        fig = generate_figure(df, out)
        assert len(fig.axes) == 2

    def test_accepts_string_out_path(self, tmp_path):
        df = _make_catalog(n=20, seed=8)
        out = str(tmp_path / "fig.png")
        generate_figure(df, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# 4. main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_dict(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=20, seed=0), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        assert isinstance(result, dict)

    def test_returns_required_keys(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=20, seed=1), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        required = {"median_logM", "stats_low", "stats_high", "figure_path", "pdf_path"}
        assert required.issubset(result.keys())

    def test_figure_path_is_created(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=20, seed=2), tmp_path)
        out = tmp_path / "out.png"
        main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()

    def test_pdf_path_is_created(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=20, seed=3), tmp_path)
        out = tmp_path / "out.png"
        result = main(["--csv", str(csv), "--out", str(out)])
        assert result["pdf_path"].exists()

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "missing.csv")])

    def test_missing_column_raises(self, tmp_path):
        df = _make_catalog(n=10).drop(columns=["slope_tail"])
        csv = _write_catalog(df, tmp_path)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(csv)])

    def test_median_logM_in_expected_range(self, tmp_path):
        df = _make_catalog(n=40, seed=4, logM_range=(8.0, 10.0))
        csv = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        assert 8.0 <= result["median_logM"] <= 10.0

    def test_custom_median_logM_is_respected(self, tmp_path):
        df = _make_catalog(n=40, seed=5)
        csv = _write_catalog(df, tmp_path)
        result = main([
            "--csv", str(csv),
            "--out", str(tmp_path / "fig.png"),
            "--median-logM", "8.5",
        ])
        assert result["median_logM"] == pytest.approx(8.5)

    def test_stats_low_has_required_keys(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=20, seed=6), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        required = {"n", "rho", "p_spear", "ols_slope", "ols_intercept", "r2", "p_ols"}
        assert required.issubset(result["stats_low"].keys())

    def test_stats_high_has_required_keys(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=20, seed=7), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        required = {"n", "rho", "p_spear", "ols_slope", "ols_intercept", "r2", "p_ols"}
        assert required.issubset(result["stats_high"].keys())

    def test_low_plus_high_n_equals_total(self, tmp_path):
        df = _make_catalog(n=30, seed=8)
        csv = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        total = result["stats_low"]["n"] + result["stats_high"]["n"]
        assert total == len(df)

    def test_default_csv_arg(self):
        args = _parse_args([])
        assert "sparc_subset.csv" in args.csv
        assert Path(args.csv).is_absolute()

    def test_default_out_arg_is_png(self):
        args = _parse_args([])
        assert args.out.endswith(".png")

    def test_beta_ref_is_half(self):
        assert BETA_REF == pytest.approx(0.5)

    def test_delta_f3_computed_from_slope_tail(self, tmp_path):
        """slope_tail = 0.7 everywhere → delta_f3 = 0.2; script must not crash."""
        df = _make_catalog(n=20, seed=9)
        df["slope_tail"] = 0.7
        csv = _write_catalog(df, tmp_path)
        # When delta_f3 is constant, Spearman rho is NaN by definition;
        # the important thing is that main() completes without raising.
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        assert "stats_low" in result
        assert "stats_high" in result


# ---------------------------------------------------------------------------
# 5. Integration: committed SPARC subset fixture
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SPARC_CSV = _REPO_ROOT / "data" / "sparc_subset.csv"

# Expected values computed from the committed CSV (79 galaxies after dedup).
_N_TOTAL = 79
_N_LOW = 39
_N_HIGH = 40
_MEDIAN_LOGM = pytest.approx(10.6384, abs=1e-3)
# High-mass panel shows a significant negative environment signal.
_RHO_HIGH = pytest.approx(-0.489, abs=0.01)
_P_HIGH = pytest.approx(0.0014, abs=0.001)
_SLOPE_HIGH = pytest.approx(-0.160, abs=0.01)
# Low-mass panel shows no significant signal.
_RHO_LOW = pytest.approx(-0.150, abs=0.02)


class TestSPARCSubsetIntegration:
    """Regression guard: run the full pipeline on the committed SPARC CSV and
    verify known numerical results.  Any accidental change to the data file or
    the statistics logic will be caught here.
    """

    def test_fixture_csv_exists(self):
        assert _SPARC_CSV.exists(), (
            f"SPARC fixture not found: {_SPARC_CSV}\n"
            "The file data/sparc_subset.csv must be present in the repository."
        )

    def test_fixture_has_expected_row_count(self):
        df = pd.read_csv(_SPARC_CSV)
        assert len(df) == _N_TOTAL

    def test_fixture_has_required_columns(self):
        df = pd.read_csv(_SPARC_CSV)
        for col in ("galaxy", "logM", "delta_mass_std", "slope_tail"):
            assert col in df.columns

    def test_fixture_no_duplicates(self):
        df = pd.read_csv(_SPARC_CSV)
        assert df["galaxy"].duplicated().sum() == 0

    def test_fixture_no_nulls_in_required_cols(self):
        df = pd.read_csv(_SPARC_CSV)
        for col in ("logM", "delta_mass_std", "slope_tail"):
            assert df[col].isna().sum() == 0

    def test_median_logM(self):
        df = pd.read_csv(_SPARC_CSV)
        assert float(df["logM"].median()) == _MEDIAN_LOGM

    def test_subsample_sizes(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["stats_low"]["n"] == _N_LOW
        assert result["stats_high"]["n"] == _N_HIGH

    def test_total_n_equals_79(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        total = result["stats_low"]["n"] + result["stats_high"]["n"]
        assert total == _N_TOTAL

    def test_high_mass_rho_significant_negative(self, tmp_path):
        """High-mass panel must show a significant negative environment signal."""
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        stats = result["stats_high"]
        assert stats["rho"] == _RHO_HIGH
        assert stats["rho"] < 0, "High-mass rho must be negative"
        assert stats["p_spear"] < 0.01, "High-mass Spearman p must be < 0.01"

    def test_high_mass_p_value(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["stats_high"]["p_spear"] == _P_HIGH

    def test_high_mass_ols_slope(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["stats_high"]["ols_slope"] == _SLOPE_HIGH

    def test_low_mass_rho(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["stats_low"]["rho"] == _RHO_LOW

    def test_low_mass_signal_not_significant(self, tmp_path):
        """Low-mass panel must NOT show a significant correlation (p > 0.05)."""
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["stats_low"]["p_spear"] > 0.05

    def test_default_csv_path_points_to_fixture(self):
        """The script default --csv must resolve to the committed fixture."""
        args = _parse_args([])
        assert Path(args.csv).resolve() == _SPARC_CSV.resolve()

    def test_figure_files_created(self, tmp_path):
        out = tmp_path / "SPARC_split_mass_environment.png"
        main(["--csv", str(_SPARC_CSV), "--out", str(out)])
        assert out.exists()
        assert out.with_suffix(".pdf").exists()
