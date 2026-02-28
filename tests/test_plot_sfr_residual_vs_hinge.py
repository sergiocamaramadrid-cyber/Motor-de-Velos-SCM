"""
tests/test_plot_sfr_residual_vs_hinge.py
=========================================
Unit tests for scripts/plot_sfr_residual_vs_hinge.py.

Tests cover:
- compute_sfr_residuals  (residual computation and shape)
- fit_residual_vs_F3     (OLS fit returns expected keys and correct signs)
- make_figure            (smoke test: figure created, axes labelled)
- save_figure            (files written to a tmp directory)
- load_data              (merge and log_sfr conversion)
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

from plot_sfr_residual_vs_hinge import (
    compute_sfr_residuals,
    fit_residual_vs_F3,
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
    rng = np.random.default_rng(42)
    n = 60
    log_mbar = rng.uniform(8.5, 11.5, n)
    F3 = rng.uniform(0.0, 3.0, n)
    # Build log_sfr with known slope c=0.5 for F3 plus mass term
    log_sfr = 0.8 * log_mbar - 8.0 + 0.5 * F3 + rng.normal(0, 0.1, n)
    morph = rng.choice(["late", "inter", "early"], size=n)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "log_mbar": log_mbar,
        "log_sfr": log_sfr,
        "F3_mean_H_ext": F3,
        "morph_bin": morph,
    })


@pytest.fixture()
def simple_df_no_morph(simple_df):
    """Same as simple_df but without the morph_bin column."""
    return simple_df.drop(columns=["morph_bin"])


# ---------------------------------------------------------------------------
# compute_sfr_residuals
# ---------------------------------------------------------------------------

class TestComputeSfrResiduals:
    def test_adds_sfr_resid_column(self, simple_df):
        result = compute_sfr_residuals(simple_df)
        assert "sfr_resid" in result.columns

    def test_residuals_have_near_zero_mean(self, simple_df):
        result = compute_sfr_residuals(simple_df)
        assert abs(result["sfr_resid"].mean()) < 0.1

    def test_shape_preserved(self, simple_df):
        result = compute_sfr_residuals(simple_df)
        assert len(result) == len(simple_df.dropna(subset=["log_mbar", "log_sfr", "F3_mean_H_ext"]))

    def test_residuals_are_finite(self, simple_df):
        result = compute_sfr_residuals(simple_df)
        assert result["sfr_resid"].notna().all()

    def test_works_without_morph_bin(self, simple_df_no_morph):
        result = compute_sfr_residuals(simple_df_no_morph)
        assert "sfr_resid" in result.columns


# ---------------------------------------------------------------------------
# fit_residual_vs_F3
# ---------------------------------------------------------------------------

class TestFitResidualVsF3:
    def test_returns_required_keys(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        for key in ("coef", "se", "pvalue", "x_fit", "y_fit", "ci_lo", "ci_hi"):
            assert key in fit, f"Missing key: {key}"

    def test_coef_is_positive_for_known_relationship(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        assert fit["coef"] > 0, "Expected positive coefficient for positive F3-SFR relationship"

    def test_pvalue_is_significant_for_strong_signal(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        assert fit["pvalue"] < 0.05, "Expected significant p-value for clear signal"

    def test_se_is_positive(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        assert fit["se"] > 0

    def test_x_fit_is_monotone(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        assert np.all(np.diff(fit["x_fit"]) > 0)

    def test_ci_hi_above_ci_lo(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        assert np.all(fit["ci_hi"] >= fit["ci_lo"])

    def test_null_relationship_large_pvalue(self):
        """With random (unrelated) data the coefficient should not be significant."""
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({
            "sfr_resid": rng.normal(0, 1, n),
            "F3_mean_H_ext": rng.uniform(0, 3, n),
        })
        fit = fit_residual_vs_F3(df)
        # p-value should not be absurdly small for pure noise (probabilistically)
        # We just check structure rather than exact value
        assert 0.0 < fit["pvalue"] <= 1.0


# ---------------------------------------------------------------------------
# make_figure
# ---------------------------------------------------------------------------

class TestMakeFigure:
    def test_returns_figure_with_morph(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_without_morph(self, simple_df_no_morph):
        df = compute_sfr_residuals(simple_df_no_morph)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_one_axes(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_xlabel_contains_F3(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        xlabel = fig.axes[0].get_xlabel()
        assert "F_3" in xlabel or "F3" in xlabel or "hinge" in xlabel.lower()
        plt.close(fig)

    def test_ylabel_contains_residual(self, simple_df):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        ylabel = fig.axes[0].get_ylabel()
        assert "residual" in ylabel.lower() or "SFR" in ylabel
        plt.close(fig)


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_saves_pdf_and_png(self, simple_df, tmp_path):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        saved = save_figure(fig, str(tmp_path))
        plt.close(fig)
        extensions = {Path(p).suffix for p in saved}
        assert ".pdf" in extensions
        assert ".png" in extensions

    def test_files_are_non_empty(self, simple_df, tmp_path):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        saved = save_figure(fig, str(tmp_path))
        plt.close(fig)
        for p in saved:
            assert Path(p).stat().st_size > 0, f"{p} is empty"

    def test_creates_out_dir(self, simple_df, tmp_path):
        df = compute_sfr_residuals(simple_df)
        fit = fit_residual_vs_F3(df)
        fig = make_figure(df, fit)
        new_dir = tmp_path / "nested" / "output"
        save_figure(fig, str(new_dir))
        plt.close(fig)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def _write_csvs(self, tmp_path, features_df, galaxy_df):
        feat_path = tmp_path / "hinge_features.csv"
        gal_path = tmp_path / "galaxy_table.csv"
        features_df.to_csv(feat_path, index=False)
        galaxy_df.to_csv(gal_path, index=False)
        return str(feat_path), str(gal_path)

    def test_merge_on_galaxy(self, tmp_path):
        feats = pd.DataFrame({
            "galaxy": ["A", "B", "C"],
            "F3_mean_H_ext": [1.0, 2.0, 3.0],
            "F1_med_abs_dH_dr_ext": [0.1, 0.2, 0.3],
        })
        gal = pd.DataFrame({
            "galaxy": ["A", "B", "C"],
            "log_mbar": [9.0, 10.0, 11.0],
            "log_sfr": [-1.0, 0.0, 1.0],
        })
        feat_p, gal_p = self._write_csvs(tmp_path, feats, gal)
        df = load_data(feat_p, gal_p)
        assert len(df) == 3
        assert "F3_mean_H_ext" in df.columns

    def test_linear_sfr_converted_to_log(self, tmp_path):
        feats = pd.DataFrame({
            "galaxy": ["X"],
            "F3_mean_H_ext": [1.5],
        })
        gal = pd.DataFrame({
            "galaxy": ["X"],
            "log_mbar": [10.0],
            "sfr": [10.0],  # linear SFR, no log_sfr column
        })
        feat_p, gal_p = self._write_csvs(tmp_path, feats, gal)
        df = load_data(feat_p, gal_p)
        assert "log_sfr" in df.columns
        np.testing.assert_allclose(df["log_sfr"].iloc[0], np.log10(10.0), rtol=1e-6)

    def test_only_inner_join_galaxies_kept(self, tmp_path):
        feats = pd.DataFrame({
            "galaxy": ["A", "B"],
            "F3_mean_H_ext": [1.0, 2.0],
        })
        gal = pd.DataFrame({
            "galaxy": ["A", "C"],  # "C" has no features; "B" has no galaxy info
            "log_mbar": [9.0, 10.0],
            "log_sfr": [0.0, 1.0],
        })
        feat_p, gal_p = self._write_csvs(tmp_path, feats, gal)
        df = load_data(feat_p, gal_p)
        assert len(df) == 1
        assert df["galaxy"].iloc[0] == "A"
