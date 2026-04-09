"""
tests/test_plot_sparc_slope_tail_hist.py — Tests for
scripts/plot_sparc_slope_tail_hist.py.

Uses synthetic CSV data so the suite runs without any real SPARC download.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_sparc_slope_tail_hist import (
    AXVLINE_DEFAULT,
    BINS_DEFAULT,
    FIGURE_CAPTION,
    LOGM_CUT_DEFAULT,
    OUTPUT_PNG_DEFAULT,
    SLOPES_CSV_DEFAULT,
    SPARC_CSV_DEFAULT,
    compute_stats,
    generate_figure,
    load_and_merge,
    main,
)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _write_sparc_csv(path: Path, n: int = 20, seed: int = 0) -> Path:
    """Write a minimal sparc_basic.csv with galaxy + Mstar columns."""
    rng = np.random.default_rng(seed)
    galaxies = [f"NGC{1000 + i}" for i in range(n)]
    # Mstar spans ~10^9 to ~10^11 so we get a mix above/below logM=10
    mstar = 10 ** rng.uniform(9.0, 11.0, n)
    pd.DataFrame({"galaxy": galaxies, "Mstar": mstar}).to_csv(path, index=False)
    return path


def _write_slopes_csv(path: Path, galaxies: list[str], seed: int = 1) -> Path:
    """Write a slope_tail.csv matching the given galaxy names."""
    rng = np.random.default_rng(seed)
    slopes = rng.uniform(-0.4, 0.1, len(galaxies))
    pd.DataFrame({"galaxy": galaxies, "slope_tail": slopes}).to_csv(
        path, index=False
    )
    return path


def _make_data_pair(tmp_path: Path, n: int = 20, seed: int = 42):
    """Return (sparc_csv, slopes_csv) paths with matching galaxies."""
    sparc_csv = tmp_path / "sparc_basic.csv"
    slopes_csv = tmp_path / "slope_tail.csv"
    _write_sparc_csv(sparc_csv, n=n, seed=seed)
    galaxies = list(pd.read_csv(sparc_csv)["galaxy"])
    _write_slopes_csv(slopes_csv, galaxies, seed=seed + 1)
    return sparc_csv, slopes_csv


def _make_merged_df(n: int = 30, logm_range=(9.0, 11.5), seed: int = 0) -> pd.DataFrame:
    """Return a merged DataFrame suitable for generate_figure."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(*logm_range, n)
    return pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "Mstar": 10 ** logM,
        "slope_tail": rng.uniform(-0.5, 0.1, n),
        "logM": logM,
    })


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_returns_required_keys(self):
        result = compute_stats(np.array([-0.1, -0.2, -0.3]))
        assert {"n", "mean", "median", "std", "min", "max"}.issubset(result)

    def test_correct_n(self):
        assert compute_stats(np.array([1.0, 2.0, 3.0]))["n"] == 3

    def test_correct_mean(self):
        arr = np.array([-0.1, -0.2, -0.3])
        assert compute_stats(arr)["mean"] == pytest.approx(-0.2, abs=1e-10)

    def test_correct_median(self):
        arr = np.array([-0.1, -0.2, -0.5])
        assert compute_stats(arr)["median"] == pytest.approx(-0.2, abs=1e-10)

    def test_correct_min_max(self):
        arr = np.array([-0.5, -0.2, 0.1])
        r = compute_stats(arr)
        assert r["min"] == pytest.approx(-0.5, abs=1e-10)
        assert r["max"] == pytest.approx(0.1, abs=1e-10)

    def test_std_nan_for_single_element(self):
        r = compute_stats(np.array([0.5]))
        assert math.isnan(r["std"])

    def test_empty_array_all_nan(self):
        r = compute_stats(np.array([]))
        assert r["n"] == 0
        assert math.isnan(r["mean"])
        assert math.isnan(r["median"])
        assert math.isnan(r["std"])

    def test_all_values_same(self):
        arr = np.full(10, -0.15)
        r = compute_stats(arr)
        assert r["mean"] == pytest.approx(-0.15, abs=1e-10)
        assert r["std"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_float_types(self):
        r = compute_stats(np.array([-0.1, -0.2]))
        assert isinstance(r["mean"], float)
        assert isinstance(r["n"], int)


# ---------------------------------------------------------------------------
# load_and_merge
# ---------------------------------------------------------------------------

class TestLoadAndMerge:
    def test_returns_dataframe(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        df = load_and_merge(sparc, slopes)
        assert isinstance(df, pd.DataFrame)

    def test_has_logM_column(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        df = load_and_merge(sparc, slopes)
        assert "logM" in df.columns

    def test_logM_equals_log10_mstar(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path, n=10)
        df = load_and_merge(sparc, slopes)
        expected = np.log10(df["Mstar"])
        np.testing.assert_allclose(df["logM"].values, expected.values, rtol=1e-10)

    def test_all_required_columns_present(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        df = load_and_merge(sparc, slopes)
        for col in ["galaxy", "Mstar", "slope_tail", "logM"]:
            assert col in df.columns

    def test_inner_join_drops_unmatched(self, tmp_path):
        sparc_csv = tmp_path / "sparc.csv"
        slopes_csv = tmp_path / "slopes.csv"
        pd.DataFrame({"galaxy": ["A", "B", "C"], "Mstar": [1e10, 2e10, 3e10]}
                     ).to_csv(sparc_csv, index=False)
        # Only A and B in slopes
        pd.DataFrame({"galaxy": ["A", "B"], "slope_tail": [-0.1, -0.2]}
                     ).to_csv(slopes_csv, index=False)
        df = load_and_merge(sparc_csv, slopes_csv)
        assert len(df) == 2
        assert set(df["galaxy"]) == {"A", "B"}

    def test_missing_sparc_csv_raises(self, tmp_path):
        _, slopes = _make_data_pair(tmp_path)
        with pytest.raises(FileNotFoundError, match="SPARC summary CSV not found"):
            load_and_merge(tmp_path / "nonexistent.csv", slopes)

    def test_missing_slopes_csv_raises(self, tmp_path):
        sparc, _ = _make_data_pair(tmp_path)
        with pytest.raises(FileNotFoundError, match="Slope-tail CSV not found"):
            load_and_merge(sparc, tmp_path / "nonexistent.csv")

    def test_sparc_missing_mstar_column_raises(self, tmp_path):
        sparc_csv = tmp_path / "sparc.csv"
        slopes_csv = tmp_path / "slopes.csv"
        pd.DataFrame({"galaxy": ["A"], "luminosity": [1e9]}).to_csv(
            sparc_csv, index=False)
        pd.DataFrame({"galaxy": ["A"], "slope_tail": [-0.1]}).to_csv(
            slopes_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_and_merge(sparc_csv, slopes_csv)

    def test_slopes_missing_slope_tail_column_raises(self, tmp_path):
        sparc_csv = tmp_path / "sparc.csv"
        slopes_csv = tmp_path / "slopes.csv"
        pd.DataFrame({"galaxy": ["A"], "Mstar": [1e10]}).to_csv(
            sparc_csv, index=False)
        pd.DataFrame({"galaxy": ["A"], "beta": [-0.1]}).to_csv(
            slopes_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_and_merge(sparc_csv, slopes_csv)

    def test_n_rows_matches_inner_join(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path, n=15)
        df = load_and_merge(sparc, slopes)
        assert len(df) == 15


# ---------------------------------------------------------------------------
# generate_figure
# ---------------------------------------------------------------------------

class TestGenerateFigure:
    def test_returns_figure(self, tmp_path):
        df = _make_merged_df()
        out = tmp_path / "fig.png"
        fig = generate_figure(df, out)
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

    def test_png_written(self, tmp_path):
        df = _make_merged_df()
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_pdf_written_as_sibling(self, tmp_path):
        df = _make_merged_df()
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert (tmp_path / "fig.pdf").exists()

    def test_creates_output_directory(self, tmp_path):
        df = _make_merged_df()
        out = tmp_path / "sub" / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_custom_logm_cut(self, tmp_path):
        df = _make_merged_df(n=50)
        out = tmp_path / "fig.png"
        # Should not raise regardless of cut
        generate_figure(df, out, logm_cut=9.5)
        assert out.exists()

    def test_custom_axvline(self, tmp_path):
        df = _make_merged_df()
        out = tmp_path / "fig.png"
        generate_figure(df, out, axvline=-0.2)
        assert out.exists()

    def test_custom_bins(self, tmp_path):
        df = _make_merged_df(n=40)
        out = tmp_path / "fig.png"
        generate_figure(df, out, bins=10)
        assert out.exists()

    def test_empty_high_mass_subsample_does_not_raise(self, tmp_path):
        """All galaxies below the cut → empty histogram → should not crash."""
        df = _make_merged_df(n=20, logm_range=(8.0, 9.5))
        out = tmp_path / "fig.png"
        generate_figure(df, out, logm_cut=10.0)
        assert out.exists()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert isinstance(result, dict)

    def test_required_keys(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert {"stats", "logm_cut", "n_merged", "n_high_mass",
                "figure_path", "pdf_path"}.issubset(result)

    def test_figure_path_is_string(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert isinstance(result["figure_path"], str)

    def test_pdf_path_ends_with_pdf(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert result["pdf_path"].endswith(".pdf")

    def test_n_merged_equals_inner_join_count(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path, n=20)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert result["n_merged"] == 20

    def test_n_high_mass_consistent_with_logm_cut(self, tmp_path):
        sparc_csv = tmp_path / "sparc.csv"
        slopes_csv = tmp_path / "slopes.csv"
        # 5 high-mass, 5 low-mass
        galaxies = [f"G{i:02d}" for i in range(10)]
        mstar = [1e11] * 5 + [1e9] * 5  # logM=11 (high) and 9 (low)
        pd.DataFrame({"galaxy": galaxies, "Mstar": mstar}).to_csv(
            sparc_csv, index=False)
        slopes = [-0.1] * 10
        pd.DataFrame({"galaxy": galaxies, "slope_tail": slopes}).to_csv(
            slopes_csv, index=False)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc_csv),
            "--slopes-csv", str(slopes_csv),
            "--out", str(out),
            "--logm-cut", "10.0",
        ])
        assert result["n_high_mass"] == 5

    def test_custom_logm_cut_cli(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path, n=30, seed=7)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
            "--logm-cut", "9.5",
        ])
        assert result["logm_cut"] == pytest.approx(9.5)

    def test_custom_bins_cli(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
            "--bins", "10",
        ])
        assert out.exists()

    def test_custom_axvline_cli(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
            "--axvline", "-0.2",
        ])
        assert out.exists()

    def test_png_file_created(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "out" / "fig.png"
        main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert out.exists()

    def test_pdf_file_created(self, tmp_path):
        sparc, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--out", str(out),
        ])
        assert out.with_suffix(".pdf").exists()

    def test_missing_sparc_csv_raises(self, tmp_path):
        _, slopes = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        with pytest.raises(FileNotFoundError):
            main([
                "--sparc-csv", str(tmp_path / "missing.csv"),
                "--slopes-csv", str(slopes),
                "--out", str(out),
            ])

    def test_missing_slopes_csv_raises(self, tmp_path):
        sparc, _ = _make_data_pair(tmp_path)
        out = tmp_path / "fig.png"
        with pytest.raises(FileNotFoundError):
            main([
                "--sparc-csv", str(sparc),
                "--slopes-csv", str(tmp_path / "missing.csv"),
                "--out", str(out),
            ])


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_logm_cut_default(self):
        assert LOGM_CUT_DEFAULT == pytest.approx(10.0)

    def test_axvline_default(self):
        assert AXVLINE_DEFAULT == pytest.approx(-0.15)

    def test_bins_default(self):
        assert BINS_DEFAULT == 15

    def test_sparc_csv_default_is_string(self):
        assert isinstance(SPARC_CSV_DEFAULT, str)

    def test_slopes_csv_default_is_string(self):
        assert isinstance(SLOPES_CSV_DEFAULT, str)

    def test_output_png_default_is_string(self):
        assert isinstance(OUTPUT_PNG_DEFAULT, str)

    def test_figure_caption_is_string(self):
        assert isinstance(FIGURE_CAPTION, str)

    def test_figure_caption_mentions_slope(self):
        assert "slope" in FIGURE_CAPTION.lower()
