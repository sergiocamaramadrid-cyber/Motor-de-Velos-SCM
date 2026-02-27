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
    load_pressure_calibration,
    estimate_xi_from_sfr,
    _write_executive_summary,
    _write_audit_summary,
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

    def test_creates_audit_summary(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        audit_path = out_dir / "audit_summary.json"
        assert audit_path.exists()
        import json
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        assert "xi_calibration" in data
        assert data["xi_calibration"]["version"] == "v0.6.1"
        assert "pipeline_stats" in data


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
# load_pressure_calibration
# ---------------------------------------------------------------------------

_CALIBRATION_PATH = Path(__file__).parent.parent / "data/calibration/local_group_xi_calibration.json"


class TestLoadPressureCalibration:
    def test_loads_from_default_path(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        assert data["calibration_id"] == "SCM_XI_LOCAL_GROUP_v0.6.1"

    def test_returns_dict(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        assert isinstance(data, dict)

    def test_required_top_level_keys(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        for key in ("calibration_id", "framework_version", "sample_size",
                    "xi_statistics", "sfr_model", "galaxies"):
            assert key in data, f"Missing key: {key}"

    def test_galaxies_list_length(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        assert len(data["galaxies"]) == 6

    def test_xi_statistics_values(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        stats = data["xi_statistics"]
        assert stats["mean"] == pytest.approx(1.36)
        assert stats["min"] == pytest.approx(1.28)
        assert stats["max"] == pytest.approx(1.42)

    def test_sfr_model_values(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        sfr = data["sfr_model"]
        assert sfr["intercept"] == pytest.approx(1.33)
        assert sfr["slope"] == pytest.approx(0.21)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pressure_calibration(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# estimate_xi_from_sfr
# ---------------------------------------------------------------------------

class TestEstimateXiFromSfr:
    def test_baseline_sfr_gives_intercept(self):
        # log10(SFR) = 0 → xi = 1.33
        assert estimate_xi_from_sfr(0.0) == pytest.approx(1.33)

    def test_positive_log_sfr_increases_xi(self):
        assert estimate_xi_from_sfr(1.0) > estimate_xi_from_sfr(0.0)

    def test_negative_log_sfr_decreases_xi(self):
        assert estimate_xi_from_sfr(-1.0) < estimate_xi_from_sfr(0.0)

    def test_clamped_at_maximum(self):
        # Very high SFR → xi clamped at 1.42
        assert estimate_xi_from_sfr(100.0) == pytest.approx(1.42)

    def test_clamped_at_minimum(self):
        # Very low SFR → xi clamped at 1.28
        assert estimate_xi_from_sfr(-100.0) == pytest.approx(1.28)

    def test_lmc_log_sfr(self):
        # LMC: log_sfr = 0.30 → xi ≈ 1.33 + 0.21*0.30 = 1.393 → clamped within [1.28, 1.42]
        xi = estimate_xi_from_sfr(0.30)
        assert 1.28 <= xi <= 1.42

    def test_returns_float(self):
        assert isinstance(estimate_xi_from_sfr(0.0), float)


# ---------------------------------------------------------------------------
# _write_audit_summary
# ---------------------------------------------------------------------------

import json as _json


class TestWriteAuditSummary:
    def _make_df(self):
        return pd.DataFrame({
            "galaxy": ["G1", "G2"],
            "chi2_reduced": [0.8, 1.2],
            "upsilon_disk": [1.0, 1.5],
        })

    def test_creates_json_file(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(self._make_df(), {}, path)
        assert path.exists()

    def test_xi_calibration_key_preserved(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        audit = {
            "xi_calibration": {
                "version": "v0.6.1",
                "model": "xi = 1.33 + 0.21 log10(SFR)",
                "range": [1.28, 1.42],
            }
        }
        _write_audit_summary(self._make_df(), audit, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["xi_calibration"]["version"] == "v0.6.1"
        assert data["xi_calibration"]["range"] == [1.28, 1.42]

    def test_pipeline_stats_present(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(self._make_df(), {}, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert "pipeline_stats" in data
        assert data["pipeline_stats"]["n_galaxies"] == 2

    def test_chi2_median_correct(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(self._make_df(), {}, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["pipeline_stats"]["chi2_reduced_median"] == pytest.approx(1.0)

    def test_empty_dataframe(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(pd.DataFrame(), {}, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["pipeline_stats"]["n_galaxies"] == 0
        assert data["pipeline_stats"]["chi2_reduced_median"] is None
