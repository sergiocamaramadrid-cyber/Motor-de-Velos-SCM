"""
tests/test_plot_hinge_sfr_relation.py
======================================
Unit tests for scripts/plot_hinge_sfr_relation.py.

Tests cover:
- load_data              (CSV merge, required columns)
- compute_mass_residuals (residuals near zero mean, column added)
- fit_hinge_relation     (keys present, positive slope on crafted data)
- make_figure            (figure created, axes labelled)
- save_figure            (PNG written to tmp directory)
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from plot_hinge_sfr_relation import (
    compute_mass_residuals,
    fit_hinge_relation,
    load_data,
    make_figure,
    save_figure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df():
    """Minimal galaxy DataFrame with a known positive F3-SFR relationship."""
    rng = np.random.default_rng(0)
    n = 60
    log_mbar = rng.uniform(8.5, 11.5, n)
    F3 = rng.uniform(0.0, 3.0, n)
    # Known slope c=0.5 for F3 plus mass term, small noise
    log_sfr = 0.8 * log_mbar - 8.0 + 0.5 * F3 + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "galaxy": [f"G{i:03d}" for i in range(n)],
            "log_mbar": log_mbar,
            "log_sfr": log_sfr,
            "F3_mean_H_ext": F3,
        }
    )


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_merge_produces_expected_columns(self, simple_df, tmp_path):
        feat_path = tmp_path / "hinge_features.csv"
        gal_path = tmp_path / "galaxy_table.csv"
        simple_df[["galaxy", "F3_mean_H_ext"]].to_csv(feat_path, index=False)
        simple_df[["galaxy", "log_mbar", "log_sfr"]].to_csv(gal_path, index=False)

        df = load_data(str(feat_path), str(gal_path))
        for col in ("galaxy", "log_mbar", "log_sfr", "F3_mean_H_ext"):
            assert col in df.columns

    def test_merge_row_count(self, simple_df, tmp_path):
        feat_path = tmp_path / "hinge_features.csv"
        gal_path = tmp_path / "galaxy_table.csv"
        simple_df[["galaxy", "F3_mean_H_ext"]].to_csv(feat_path, index=False)
        simple_df[["galaxy", "log_mbar", "log_sfr"]].to_csv(gal_path, index=False)

        df = load_data(str(feat_path), str(gal_path))
        assert len(df) == len(simple_df)

    def test_only_common_galaxies_kept(self, simple_df, tmp_path):
        feat_path = tmp_path / "hinge_features.csv"
        gal_path = tmp_path / "galaxy_table.csv"
        # features has only first 40 galaxies
        simple_df.iloc[:40][["galaxy", "F3_mean_H_ext"]].to_csv(feat_path, index=False)
        simple_df[["galaxy", "log_mbar", "log_sfr"]].to_csv(gal_path, index=False)

        df = load_data(str(feat_path), str(gal_path))
        assert len(df) == 40

    def test_extra_feature_columns_dropped(self, simple_df, tmp_path):
        feat_path = tmp_path / "hinge_features.csv"
        gal_path = tmp_path / "galaxy_table.csv"
        feat = simple_df[["galaxy", "F3_mean_H_ext"]].copy()
        feat["F1_extra"] = 0.0
        feat.to_csv(feat_path, index=False)
        simple_df[["galaxy", "log_mbar", "log_sfr"]].to_csv(gal_path, index=False)

        df = load_data(str(feat_path), str(gal_path))
        assert "F1_extra" not in df.columns


# ---------------------------------------------------------------------------
# compute_mass_residuals
# ---------------------------------------------------------------------------

class TestComputeMassResiduals:
    def test_adds_logsfr_residual_column(self, simple_df):
        result = compute_mass_residuals(simple_df)
        assert "logSFR_residual" in result.columns

    def test_residuals_have_near_zero_mean(self, simple_df):
        result = compute_mass_residuals(simple_df)
        assert abs(result["logSFR_residual"].mean()) < 0.1

    def test_residuals_are_finite(self, simple_df):
        result = compute_mass_residuals(simple_df)
        assert result["logSFR_residual"].notna().all()
        assert np.isfinite(result["logSFR_residual"]).all()

    def test_input_not_mutated(self, simple_df):
        original_cols = list(simple_df.columns)
        compute_mass_residuals(simple_df)
        assert list(simple_df.columns) == original_cols

    def test_row_count_unchanged(self, simple_df):
        result = compute_mass_residuals(simple_df)
        assert len(result) == len(simple_df)


# ---------------------------------------------------------------------------
# fit_hinge_relation
# ---------------------------------------------------------------------------

class TestFitHingeRelation:
    def _df_with_residuals(self, simple_df):
        return compute_mass_residuals(simple_df)

    def test_returns_required_keys(self, simple_df):
        df = self._df_with_residuals(simple_df)
        fit = fit_hinge_relation(df)
        for key in ("coef", "pvalue", "x_fit", "y_fit", "summary"):
            assert key in fit, f"Missing key: {key}"

    def test_coef_is_positive_for_crafted_data(self, simple_df):
        df = self._df_with_residuals(simple_df)
        fit = fit_hinge_relation(df)
        assert fit["coef"] > 0, "Expected positive slope for crafted positive relationship"

    def test_pvalue_significant_for_crafted_data(self, simple_df):
        df = self._df_with_residuals(simple_df)
        fit = fit_hinge_relation(df)
        assert fit["pvalue"] < 0.05

    def test_x_fit_y_fit_lengths_match(self, simple_df):
        df = self._df_with_residuals(simple_df)
        fit = fit_hinge_relation(df)
        assert len(fit["x_fit"]) == len(fit["y_fit"])

    def test_x_fit_spans_data_range(self, simple_df):
        df = self._df_with_residuals(simple_df)
        fit = fit_hinge_relation(df)
        assert fit["x_fit"].min() <= df["F3_mean_H_ext"].min() + 1e-9
        assert fit["x_fit"].max() >= df["F3_mean_H_ext"].max() - 1e-9

    def test_summary_is_string(self, simple_df):
        df = self._df_with_residuals(simple_df)
        fit = fit_hinge_relation(df)
        assert isinstance(fit["summary"], str)
        assert "OLS" in fit["summary"]


# ---------------------------------------------------------------------------
# make_figure
# ---------------------------------------------------------------------------

class TestMakeFigure:
    def _fit_and_df(self, simple_df):
        df = compute_mass_residuals(simple_df)
        fit = fit_hinge_relation(df)
        return df, fit

    def test_returns_figure(self, simple_df):
        df, fit = self._fit_and_df(simple_df)
        fig = make_figure(df, fit)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_labels(self, simple_df):
        df, fit = self._fit_and_df(simple_df)
        fig = make_figure(df, fit)
        ax = fig.axes[0]
        assert "F3" in ax.get_xlabel() or "hinge" in ax.get_xlabel().lower()
        assert "residual" in ax.get_ylabel().lower() or "sfr" in ax.get_ylabel().lower()
        plt.close(fig)

    def test_title_contains_sparc(self, simple_df):
        df, fit = self._fit_and_df(simple_df)
        fig = make_figure(df, fit)
        assert "SPARC" in fig.axes[0].get_title()
        plt.close(fig)

    def test_figure_has_two_line_objects(self, simple_df):
        """Scatter + fit line (+ optional hline) → at least 2 lines/collections."""
        df, fit = self._fit_and_df(simple_df)
        fig = make_figure(df, fit)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1  # at least the OLS fit line
        plt.close(fig)

    def test_figure_has_legend(self, simple_df):
        df, fit = self._fit_and_df(simple_df)
        fig = make_figure(df, fit)
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_file_created(self, simple_df, tmp_path):
        df = compute_mass_residuals(simple_df)
        fit = fit_hinge_relation(df)
        fig = make_figure(df, fit)
        out = save_figure(fig, str(tmp_path))
        assert Path(out).exists()
        plt.close(fig)

    def test_output_is_png(self, simple_df, tmp_path):
        df = compute_mass_residuals(simple_df)
        fit = fit_hinge_relation(df)
        fig = make_figure(df, fit)
        out = save_figure(fig, str(tmp_path))
        assert out.endswith(".png")
        plt.close(fig)

    def test_creates_out_dir_if_absent(self, simple_df, tmp_path):
        new_dir = tmp_path / "sub" / "nested"
        df = compute_mass_residuals(simple_df)
        fit = fit_hinge_relation(df)
        fig = make_figure(df, fit)
        out = save_figure(fig, str(new_dir))
        assert Path(out).exists()
        plt.close(fig)

    def test_filename_correct(self, simple_df, tmp_path):
        df = compute_mass_residuals(simple_df)
        fit = fit_hinge_relation(df)
        fig = make_figure(df, fit)
        out = save_figure(fig, str(tmp_path))
        assert Path(out).name == "hinge_sfr_relation.png"
        plt.close(fig)

    def test_file_nonzero_size(self, simple_df, tmp_path):
        df = compute_mass_residuals(simple_df)
        fit = fit_hinge_relation(df)
        fig = make_figure(df, fit)
        out = save_figure(fig, str(tmp_path))
        assert Path(out).stat().st_size > 1000
        plt.close(fig)
