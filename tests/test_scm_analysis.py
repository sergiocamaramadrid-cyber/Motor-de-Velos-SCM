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
    export_sparc_global,
    compute_vif_table,
    _build_audit_df,
    _DEFAULT_LOGG0,
    _write_executive_summary,
    _write_top10_latex,
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

    def test_creates_audit_sparc_global(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        p = out_dir / "audit" / "sparc_global.csv"
        assert p.exists()
        df = pd.read_csv(p)
        assert set(df.columns) == {"galaxy_id", "logM", "log_gbar", "log_j", "v_obs"}
        assert len(df) == 3

    def test_creates_vif_table(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        p = out_dir / "audit" / "vif_table.csv"
        assert p.exists()
        vif = pd.read_csv(p)
        assert set(vif.columns) == {"variable", "VIF"}
        assert "hinge" in set(vif["variable"].astype(str))


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
def audit_inputs():
    """Synthetic results_df + compare_df compatible with new _build_audit_df."""
    rng = np.random.default_rng(7)
    n_gal = 5
    galaxy_names = [f"G{i}" for i in range(n_gal)]
    results_df = pd.DataFrame({
        "galaxy": galaxy_names,
        "M_bar_BTFR_Msun": [1e10, 2e10, 3e10, 4e10, 5e10],
        "Vflat_kms": [150.0, 180.0, 200.0, 220.0, 250.0],
    })
    # 4 radial points per galaxy
    rows = []
    for name in galaxy_names:
        for r in [1.0, 3.0, 6.0, 10.0]:
            g_bar = 1e-11 + rng.uniform(-1e-12, 1e-12)
            g_obs = g_bar * 1.5
            rows.append({"galaxy": name, "r_kpc": r, "g_bar": g_bar, "g_obs": g_obs})
    compare_df = pd.DataFrame(rows)
    return results_df, compare_df


class TestBuildAuditDf:
    def test_returns_expected_columns(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        assert set(audit.columns) == {"galaxy_id", "logM", "log_gbar", "log_j", "v_obs"}

    def test_row_count_matches_galaxies(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        assert len(audit) == len(results_df)

    def test_all_finite(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        for col in ["logM", "log_gbar", "log_j", "v_obs"]:
            bad = (~np.isfinite(audit[col].values)).sum()
            assert bad == 0, f"{col} contains {bad} non-finite value(s)"

    def test_missing_results_column_raises(self, audit_inputs):
        results_df, compare_df = audit_inputs
        with pytest.raises(ValueError, match="M_bar_BTFR_Msun"):
            _build_audit_df(results_df.drop(columns=["M_bar_BTFR_Msun"]), compare_df)

    def test_missing_compare_column_raises(self, audit_inputs):
        results_df, compare_df = audit_inputs
        with pytest.raises(ValueError, match="g_bar"):
            _build_audit_df(results_df, compare_df.drop(columns=["g_bar"]))


# ---------------------------------------------------------------------------
# compute_vif_table
# ---------------------------------------------------------------------------

class TestComputeVifTable:
    def test_returns_expected_columns(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        vif_df = compute_vif_table(audit)
        assert set(vif_df.columns) == {"variable", "VIF"}

    def test_returns_five_rows(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        vif_df = compute_vif_table(audit)
        assert len(vif_df) == 5

    def test_variable_names(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        vif_df = compute_vif_table(audit)
        assert set(vif_df["variable"]) == {"const", "logM", "log_gbar", "log_j", "hinge"}

    def test_vif_values_are_positive(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        vif_df = compute_vif_table(audit)
        assert (vif_df["VIF"] > 0).all()

    def test_nan_rows_are_dropped(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        audit.loc[0, "logM"] = float("nan")
        vif_df = compute_vif_table(audit)
        assert set(vif_df.columns) == {"variable", "VIF"}
        assert len(vif_df) == 5
        assert (vif_df["VIF"] > 0).all()

    def test_inf_rows_are_dropped(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        audit.loc[0, "log_j"] = float("inf")
        vif_df = compute_vif_table(audit)
        assert set(vif_df.columns) == {"variable", "VIF"}
        assert len(vif_df) == 5
        assert (vif_df["VIF"] > 0).all()

    def test_raises_on_insufficient_rows(self):
        audit = pd.DataFrame({"logM": [1.0], "log_gbar": [-10.0], "log_j": [20.0]})
        with pytest.raises(ValueError, match="Need at least 2 finite rows"):
            compute_vif_table(audit)

    def test_custom_logg0(self, audit_inputs):
        results_df, compare_df = audit_inputs
        audit = _build_audit_df(results_df, compare_df)
        vif_df = compute_vif_table(audit, logg0=-10.0)
        assert set(vif_df["variable"]) == {"const", "logM", "log_gbar", "log_j", "hinge"}


# ---------------------------------------------------------------------------
# TestVIFHelpers (standalone, no pipeline required)
# ---------------------------------------------------------------------------

class TestVIFHelpers:
    def test_compute_vif_table_returns_expected_rows(self):
        audit_df = pd.DataFrame({
            "galaxy_id": ["G1", "G2", "G3", "G4", "G5"],
            "logM": [9.0, 9.5, 10.0, 10.5, 11.0],
            "log_gbar": [-11.2, -10.8, -10.6, -10.4, -10.2],
            "log_j": [2.0, 2.1, 2.2, 2.3, 2.4],
            "v_obs": [2.1, 2.2, 2.25, 2.3, 2.35],
        })
        vif_df = compute_vif_table(audit_df, logg0=_DEFAULT_LOGG0)
        assert set(vif_df.columns) == {"variable", "VIF"}
        assert set(vif_df["variable"]) == {"const", "logM", "log_gbar", "log_j", "hinge"}
        assert np.all(np.isfinite(vif_df["VIF"].to_numpy()))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIArgs:
    def test_outdir_alias_accepted(self):
        from src.scm_analysis import _parse_args
        args = _parse_args(["--data-dir", "data/sparc", "--outdir", "results/"])
        assert args.out == "results/"

    def test_out_and_outdir_are_equivalent(self):
        from src.scm_analysis import _parse_args
        args_out = _parse_args(["--data-dir", "data/sparc", "--out", "my_out/"])
        args_outdir = _parse_args(["--data-dir", "data/sparc", "--outdir", "my_out/"])
        assert args_out.out == args_outdir.out
