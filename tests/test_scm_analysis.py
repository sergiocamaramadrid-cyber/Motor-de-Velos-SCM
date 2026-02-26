"""
Unit tests for src/scm_analysis.py.

These tests exercise the full pipeline using synthetic (mock) galaxy data so
that no real SPARC download is required.
"""

import os
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_analysis import (
    load_galaxy_table,
    load_rotation_curve,
    fit_galaxy,
    run_pipeline,
    _write_executive_summary,
    _write_top10_latex,
    _build_audit_df,
    compute_vif_table,
)


# ---------------------------------------------------------------------------
# Fixtures — build a minimal synthetic SPARC dataset
# ---------------------------------------------------------------------------

@pytest.fixture()
def sparc_dir(tmp_path):
    """Create a tiny synthetic SPARC dataset in a temporary directory."""
    # Galaxy table
    galaxy_csv = tmp_path / "SPARC_Lelli2016c.csv"
    galaxy_table = pd.DataFrame({
        "Galaxy": ["NGC0000", "NGC0001", "NGC0002"],
        "D": [10.0, 20.0, 30.0],
        "Inc": [45.0, 60.0, 75.0],
        "L36": [1e9, 2e9, 3e9],
        "Vflat": [150.0, 200.0, 250.0],
        "e_Vflat": [5.0, 5.0, 5.0],
    })
    galaxy_table.to_csv(galaxy_csv, index=False)

    # Rotation curves
    rng = np.random.default_rng(42)
    for _, row in galaxy_table.iterrows():
        name = row["Galaxy"]
        v_flat = row["Vflat"]
        r = np.linspace(0.5, 15, 20)
        v_obs = np.full(20, v_flat) + rng.normal(0, 3, 20)
        v_obs_err = np.full(20, 5.0)
        v_gas = 0.3 * v_flat * np.ones(20)
        v_disk = 0.8 * v_flat * np.ones(20)
        v_bul = np.zeros(20)

        rc_df = pd.DataFrame({
            "r": r,
            "v_obs": v_obs,
            "v_obs_err": v_obs_err,
            "v_gas": v_gas,
            "v_disk": v_disk,
            "v_bul": v_bul,
            "SBdisk": np.zeros(20),
            "SBbul": np.zeros(20),
        })
        rc_path = tmp_path / f"{name}_rotmod.dat"
        rc_df.to_csv(rc_path, sep=" ", index=False, header=False)

    return tmp_path


# ---------------------------------------------------------------------------
# load_galaxy_table
# ---------------------------------------------------------------------------

class TestLoadGalaxyTable:
    def test_loads_csv(self, sparc_dir):
        df = load_galaxy_table(sparc_dir)
        assert "Galaxy" in df.columns
        assert len(df) == 3

    def test_missing_dir_raises(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_galaxy_table(missing)

    def test_no_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_galaxy_table(tmp_path)


# ---------------------------------------------------------------------------
# load_rotation_curve
# ---------------------------------------------------------------------------

class TestLoadRotationCurve:
    def test_returns_expected_columns(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        for col in ["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]:
            assert col in rc.columns

    def test_missing_galaxy_raises(self, sparc_dir):
        with pytest.raises(FileNotFoundError):
            load_rotation_curve(sparc_dir, "NOEXIST")

    def test_row_count(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0001")
        assert len(rc) == 20


# ---------------------------------------------------------------------------
# fit_galaxy
# ---------------------------------------------------------------------------

class TestFitGalaxy:
    def test_fit_returns_expected_keys(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        fit = fit_galaxy(rc)
        assert set(fit.keys()) == {"upsilon_disk", "chi2_reduced", "n_points"}

    def test_upsilon_disk_in_range(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        fit = fit_galaxy(rc)
        assert 0.1 <= fit["upsilon_disk"] <= 5.0

    def test_chi2_non_negative(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        fit = fit_galaxy(rc)
        assert fit["chi2_reduced"] >= 0.0

    def test_n_points_correct(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0001")
        fit = fit_galaxy(rc)
        assert fit["n_points"] == 20


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_dataframe(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        df = run_pipeline(sparc_dir, out_dir, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_processes_all_galaxies(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        df = run_pipeline(sparc_dir, out_dir, verbose=False)
        assert len(df) == 3

    def test_creates_csv(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        assert (out_dir / "universal_term_comparison_full.csv").exists()

    def test_creates_executive_summary(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        summary_path = out_dir / "executive_summary.txt"
        assert summary_path.exists()
        text = summary_path.read_text(encoding="utf-8")
        assert "SCM Pipeline" in text

    def test_creates_per_galaxy_csv(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        assert (out_dir / "per_galaxy_summary.csv").exists()

    def test_creates_latex_table(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        tex_path = out_dir / "top10_universal.tex"
        assert tex_path.exists()
        tex = tex_path.read_text(encoding="utf-8")
        assert r"\begin{table}" in tex

    def test_output_columns(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        df = run_pipeline(sparc_dir, out_dir, verbose=False)
        for col in ["galaxy", "chi2_reduced", "upsilon_disk", "n_points"]:
            assert col in df.columns

    def test_custom_a0(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results_a0"
        df = run_pipeline(sparc_dir, out_dir, a0=0.5e-10, verbose=False)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# _write_executive_summary
# ---------------------------------------------------------------------------

class TestWriteExecutiveSummary:
    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame()
        path = tmp_path / "summary.txt"
        _write_executive_summary(df, path)
        text = path.read_text(encoding="utf-8")
        assert "N_galaxies: 0" in text

    def test_non_empty_dataframe(self, tmp_path):
        df = pd.DataFrame({
            "chi2_reduced": [0.5, 1.0, 2.5, 3.0],
            "upsilon_disk": [1.0, 1.5, 2.0, 2.5],
        })
        path = tmp_path / "summary.txt"
        _write_executive_summary(df, path)
        text = path.read_text(encoding="utf-8")
        assert "N_galaxies: 4" in text
        assert "chi2_reduced median" in text
        assert "upsilon_disk median" in text


# ---------------------------------------------------------------------------
# _write_top10_latex
# ---------------------------------------------------------------------------

class TestWriteTop10Latex:
    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame()
        path = tmp_path / "top10.tex"
        _write_top10_latex(df, path)
        text = path.read_text(encoding="utf-8")
        assert "% No data" in text

    def test_contains_galaxy_names(self, tmp_path):
        df = pd.DataFrame({
            "galaxy": [f"NGC{i:04d}" for i in range(15)],
            "chi2_reduced": np.linspace(0.5, 3.0, 15),
            "upsilon_disk": np.ones(15),
            "Vflat_kms": np.full(15, 150.0),
        })
        path = tmp_path / "top10.tex"
        _write_top10_latex(df, path)
        text = path.read_text(encoding="utf-8")
        assert "NGC0000" in text
        # Only top 10 should be present
        assert "NGC0014" not in text

    def test_tabular_structure(self, tmp_path):
        df = pd.DataFrame({
            "galaxy": ["G1", "G2"],
            "chi2_reduced": [0.8, 1.2],
            "upsilon_disk": [1.0, 1.5],
            "Vflat_kms": [150.0, float("nan")],
        })
        path = tmp_path / "top10.tex"
        _write_top10_latex(df, path)
        text = path.read_text(encoding="utf-8")
        assert r"\begin{tabular}" in text
        assert "---" in text  # nan Vflat renders as ---


# ---------------------------------------------------------------------------
# _build_audit_df
# ---------------------------------------------------------------------------

@pytest.fixture()
def compare_df_sample():
    """Minimal per-radial-point compare_df with 10 clean rows."""
    rng = np.random.default_rng(7)
    n = 10
    r_kpc = np.linspace(1.0, 10.0, n)
    g_bar = 1e-11 * np.ones(n) + rng.uniform(-1e-12, 1e-12, n)
    g_obs = g_bar * 1.5
    return pd.DataFrame({
        "galaxy": [f"G{i}" for i in range(n)],
        "r_kpc": r_kpc,
        "g_bar": g_bar,
        "g_obs": g_obs,
        "log_g_bar": np.log10(g_bar),
        "log_g_obs": np.log10(g_obs),
    })


class TestBuildAuditDf:
    def test_returns_expected_columns(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        assert set(audit.columns) == {"logM", "log_gbar", "log_j"}

    def test_row_count_matches_input(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        assert len(audit) == len(compare_df_sample)

    def test_log_gbar_matches_log_g_bar(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        np.testing.assert_allclose(
            audit["log_gbar"].values,
            compare_df_sample["log_g_bar"].values,
        )

    def test_all_finite(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        for col in ["logM", "log_gbar", "log_j"]:
            bad = (~np.isfinite(audit[col].values)).sum()
            assert bad == 0, f"{col} contains {bad} non-finite value(s)"

    def test_does_not_mutate_input(self, compare_df_sample):
        original_cols = list(compare_df_sample.columns)
        _build_audit_df(compare_df_sample)
        assert list(compare_df_sample.columns) == original_cols


# ---------------------------------------------------------------------------
# compute_vif_table
# ---------------------------------------------------------------------------

class TestComputeVifTable:
    def test_returns_expected_columns(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        vif_df = compute_vif_table(audit)
        assert list(vif_df.columns) == ["feature", "VIF"]

    def test_returns_three_rows_by_default(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        vif_df = compute_vif_table(audit)
        assert len(vif_df) == 3

    def test_feature_names_match(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        vif_df = compute_vif_table(audit)
        assert list(vif_df["feature"]) == ["logM", "log_gbar", "log_j"]

    def test_vif_values_are_positive(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        vif_df = compute_vif_table(audit)
        assert (vif_df["VIF"] > 0).all()

    def test_nan_rows_are_dropped(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        # Inject a NaN row — should not crash
        audit.loc[0, "logM"] = float("nan")
        vif_df = compute_vif_table(audit)
        assert len(vif_df) == 3

    def test_inf_rows_are_dropped(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        # Inject ±inf — should not crash
        audit.loc[1, "log_j"] = float("inf")
        audit.loc[2, "log_gbar"] = float("-inf")
        vif_df = compute_vif_table(audit)
        assert len(vif_df) == 3

    def test_raises_on_insufficient_rows(self):
        audit = pd.DataFrame({"logM": [1.0], "log_gbar": [-10.0], "log_j": [20.0]})
        with pytest.raises(ValueError, match="Need at least 2 finite rows"):
            compute_vif_table(audit)

    def test_custom_features(self, compare_df_sample):
        audit = _build_audit_df(compare_df_sample)
        vif_df = compute_vif_table(audit, features=["logM", "log_j"])
        assert list(vif_df["feature"]) == ["logM", "log_j"]
        assert len(vif_df) == 2
