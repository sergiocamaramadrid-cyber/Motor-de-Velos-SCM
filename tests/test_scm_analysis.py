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
    _ensure_audit_dir,
    _binned_summary,
    _write_json,
    _generate_visual_audit,
    _DEFAULT_LOGG0,
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
# _ensure_audit_dir
# ---------------------------------------------------------------------------

class TestEnsureAuditDir:
    def test_creates_audit_subdir(self, tmp_path):
        audit_dir = _ensure_audit_dir(tmp_path / "out")
        assert audit_dir.exists()
        assert audit_dir.name == "audit"

    def test_idempotent(self, tmp_path):
        _ensure_audit_dir(tmp_path)
        _ensure_audit_dir(tmp_path)  # second call must not raise
        assert (tmp_path / "audit").exists()


# ---------------------------------------------------------------------------
# _binned_summary
# ---------------------------------------------------------------------------

class TestBinnedSummary:
    def test_returns_expected_columns(self):
        x = pd.Series(np.linspace(-3, 3, 60))
        y = pd.Series(np.sin(np.linspace(-3, 3, 60)))
        result = _binned_summary(x, y, nbins=6)
        for col in ["hinge_center", "resid_median", "resid_p16", "resid_p84", "n"]:
            assert col in result.columns

    def test_empty_input(self):
        result = _binned_summary(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert result.empty

    def test_non_numeric_coerced(self):
        x = pd.Series(["1.0", "2.0", "bad", "3.0"])
        y = pd.Series([0.1, 0.2, 0.3, 0.4])
        result = _binned_summary(x, y, nbins=3)
        assert not result.empty

    def test_n_bins_respected(self):
        x = pd.Series(np.linspace(0, 1, 100))
        y = pd.Series(np.zeros(100))
        result = _binned_summary(x, y, nbins=5)
        assert len(result) <= 5

    def test_percentiles_ordered(self):
        rng = np.random.default_rng(0)
        x = pd.Series(np.linspace(0, 1, 200))
        y = pd.Series(rng.normal(0, 1, 200))
        result = _binned_summary(x, y, nbins=10)
        assert (result["resid_p16"] <= result["resid_median"]).all()
        assert (result["resid_median"] <= result["resid_p84"]).all()


# ---------------------------------------------------------------------------
# _write_json
# ---------------------------------------------------------------------------

class TestWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "meta.json"
        _write_json(path, {"logg0": -10.45, "nbins": 12})
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["logg0"] == -10.45
        assert data["nbins"] == 12


# ---------------------------------------------------------------------------
# _generate_visual_audit
# ---------------------------------------------------------------------------

class TestGenerateVisualAudit:
    def _make_df(self, n=80, include_btfr=False, use_precomputed_residual=False):
        rng = np.random.default_rng(7)
        log_gbar = rng.uniform(-12, -9, n)
        v_obs = rng.uniform(100, 300, n)
        v_pred_full = v_obs + rng.normal(0, 5, n)
        df = pd.DataFrame({"log_gbar": log_gbar, "v_obs": v_obs})
        if use_precomputed_residual:
            df["residual"] = v_obs - v_pred_full
        else:
            df["v_pred_full"] = v_pred_full
        if include_btfr:
            df["v_pred_btfr"] = v_obs + rng.normal(0, 8, n)
        return df

    def test_creates_output_files(self, tmp_path):
        df = self._make_df()
        _generate_visual_audit(df, tmp_path)
        audit_dir = tmp_path / "audit"
        assert (audit_dir / "residual_vs_hinge.png").exists()
        assert (audit_dir / "binned_residuals.csv").exists()
        assert (audit_dir / "audit_meta.json").exists()

    def test_saves_in_audit_subdir(self, tmp_path):
        df = self._make_df()
        _generate_visual_audit(df, tmp_path)
        # Files must be inside out_dir/audit/, not directly in out_dir
        assert not (tmp_path / "residual_vs_hinge.png").exists()

    def test_computes_hinge_when_missing(self, tmp_path):
        df = self._make_df()
        assert "hinge" not in df.columns
        _generate_visual_audit(df, tmp_path, logg0=_DEFAULT_LOGG0)
        # Should succeed without error

    def test_accepts_precomputed_residual(self, tmp_path):
        df = self._make_df(use_precomputed_residual=True)
        assert "residual" in df.columns
        _generate_visual_audit(df, tmp_path)
        assert (tmp_path / "audit" / "residual_vs_hinge.png").exists()

    def test_raises_when_missing_log_gbar(self, tmp_path):
        df = pd.DataFrame({"v_obs": [100.0], "v_pred_full": [105.0]})
        with pytest.raises(ValueError, match="log_gbar"):
            _generate_visual_audit(df, tmp_path)

    def test_raises_when_missing_residual_cols(self, tmp_path):
        df = pd.DataFrame({"log_gbar": [-11.0], "v_obs": [100.0]})
        with pytest.raises(ValueError, match="residual"):
            _generate_visual_audit(df, tmp_path)

    def test_sidecar_json_content(self, tmp_path):
        import json
        df = self._make_df(n=50)
        _generate_visual_audit(df, tmp_path, logg0=_DEFAULT_LOGG0, nbins=8)
        meta = json.loads((tmp_path / "audit" / "audit_meta.json").read_text(encoding="utf-8"))
        assert meta["logg0"] == _DEFAULT_LOGG0
        assert meta["nbins"] == 8
        assert meta["n_rows"] == 50
        assert "log_gbar" in meta["source_columns"]
        assert len(meta["bins"]) == 9  # nbins + 1

    def test_with_btfr_column(self, tmp_path):
        df = self._make_df(include_btfr=True)
        _generate_visual_audit(df, tmp_path)
        assert (tmp_path / "audit" / "residual_vs_hinge.png").exists()

    def test_custom_logg0(self, tmp_path):
        import json
        df = self._make_df()
        _generate_visual_audit(df, tmp_path, logg0=-10.0)
        meta = json.loads((tmp_path / "audit" / "audit_meta.json").read_text(encoding="utf-8"))
        assert meta["logg0"] == -10.0

    def test_idempotent_second_call(self, tmp_path):
        df = self._make_df()
        _generate_visual_audit(df, tmp_path)
        _generate_visual_audit(df, tmp_path)  # second call must not raise
