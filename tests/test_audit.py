"""
tests/test_audit.py — Tests for PR #70 audit artefacts and OOS script.

Covers:
  * src.scm_analysis._write_audit_artifacts  (VIF, kappa, quality_status, audit_features)
  * scripts.audit_scm.run_oos_audit         (residual_vs_hinge.csv/.png, oos_summary.json)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_analysis import (
    run_pipeline,
    _write_audit_artifacts,
    _vif_from_matrix,
    _condition_number,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sparc_dir(tmp_path_factory):
    """Tiny 10-galaxy synthetic SPARC dataset."""
    root = tmp_path_factory.mktemp("sparc_audit")
    rng = np.random.default_rng(7)

    n_galaxies = 10
    names = [f"A{i:04d}" for i in range(n_galaxies)]
    v_flats = np.linspace(100.0, 280.0, n_galaxies)

    gal_table = pd.DataFrame({
        "Galaxy": names,
        "D": np.linspace(5.0, 50.0, n_galaxies),
        "Inc": np.full(n_galaxies, 60.0),
        "L36": 1e9 * np.arange(1, n_galaxies + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(n_galaxies, 5.0),
    })
    gal_table.to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 25
    for name, vf in zip(names, v_flats):
        r = np.linspace(0.5, 18.0, n_pts)
        rc = pd.DataFrame({
            "r": r,
            "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
            "v_obs_err": np.full(n_pts, 5.0),
            "v_gas": 0.25 * vf * np.ones(n_pts),
            "v_disk": 0.80 * vf * np.ones(n_pts),
            "v_bul": np.zeros(n_pts),
            "SBdisk": np.zeros(n_pts),
            "SBbul": np.zeros(n_pts),
        })
        rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

    return root


# ---------------------------------------------------------------------------
# _vif_from_matrix
# ---------------------------------------------------------------------------

class TestVifFromMatrix:
    def test_independent_cols_give_low_vif(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 3))
        vifs = _vif_from_matrix(X)
        assert len(vifs) == 3
        for v in vifs:
            assert v < 3.0, f"Independent columns should have VIF<3, got {v}"

    def test_collinear_gives_high_vif(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = x + rng.normal(scale=0.05, size=200)
        X = np.column_stack([x, y])
        vifs = _vif_from_matrix(X)
        assert vifs[0] > 10 and vifs[1] > 10


# ---------------------------------------------------------------------------
# _condition_number
# ---------------------------------------------------------------------------

class TestConditionNumber:
    def test_independent_data_gives_low_kappa(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 3))
        kappa = _condition_number(X)
        assert kappa < 10.0, f"Independent data should have low kappa, got {kappa}"

    def test_near_collinear_gives_high_kappa(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=100)
        y = x + rng.normal(scale=0.001, size=100)
        X = np.column_stack([x, y])
        kappa = _condition_number(X)
        assert kappa > 30, f"Expected high kappa for near-collinear data, got {kappa}"


# ---------------------------------------------------------------------------
# _write_audit_artifacts
# ---------------------------------------------------------------------------

class TestWriteAuditArtifacts:
    def test_creates_all_files(self, tmp_path):
        compare_df = pd.DataFrame({
            "galaxy": ["G0"] * 50,
            "r_kpc": np.linspace(1, 10, 50),
            "g_bar": np.logspace(-11.5, -9.5, 50),
            "g_obs": np.logspace(-11.0, -9.0, 50),
            "log_g_bar": np.linspace(-11.5, -9.5, 50),
            "log_g_obs": np.linspace(-11.0, -9.0, 50),
        })
        audit_dir = tmp_path / "audit"
        _write_audit_artifacts(compare_df, audit_dir)

        assert (audit_dir / "vif_table.csv").exists()
        assert (audit_dir / "stability_metrics.csv").exists()
        assert (audit_dir / "quality_status.txt").exists()
        assert (audit_dir / "audit_features.csv").exists()

    def test_vif_table_columns(self, tmp_path):
        compare_df = pd.DataFrame({
            "galaxy": ["G0"] * 50,
            "r_kpc": np.linspace(1, 10, 50),
            "g_bar": np.logspace(-11.5, -9.5, 50),
            "g_obs": np.logspace(-11.0, -9.0, 50),
            "log_g_bar": np.linspace(-11.5, -9.5, 50),
            "log_g_obs": np.linspace(-11.0, -9.0, 50),
        })
        audit_dir = tmp_path / "audit"
        _write_audit_artifacts(compare_df, audit_dir)

        vif_df = pd.read_csv(audit_dir / "vif_table.csv")
        assert "feature" in vif_df.columns
        assert "VIF" in vif_df.columns
        assert set(vif_df["feature"]) == {"log_g_bar", "hinge"}

    def test_stability_metrics_has_kappa(self, tmp_path):
        compare_df = pd.DataFrame({
            "galaxy": ["G0"] * 50,
            "r_kpc": np.linspace(1, 10, 50),
            "g_bar": np.logspace(-11.5, -9.5, 50),
            "g_obs": np.logspace(-11.0, -9.0, 50),
            "log_g_bar": np.linspace(-11.5, -9.5, 50),
            "log_g_obs": np.linspace(-11.0, -9.0, 50),
        })
        audit_dir = tmp_path / "audit"
        _write_audit_artifacts(compare_df, audit_dir)

        stab = pd.read_csv(audit_dir / "stability_metrics.csv")
        assert "kappa" in stab["metric"].values

    def test_quality_status_contains_pass_or_warning(self, tmp_path):
        compare_df = pd.DataFrame({
            "galaxy": ["G0"] * 50,
            "r_kpc": np.linspace(1, 10, 50),
            "g_bar": np.logspace(-11.5, -9.5, 50),
            "g_obs": np.logspace(-11.0, -9.0, 50),
            "log_g_bar": np.linspace(-11.5, -9.5, 50),
            "log_g_obs": np.linspace(-11.0, -9.0, 50),
        })
        audit_dir = tmp_path / "audit"
        _write_audit_artifacts(compare_df, audit_dir)

        text = (audit_dir / "quality_status.txt").read_text()
        assert "PASS" in text or "WARNING" in text

    def test_empty_dataframe_writes_stubs(self, tmp_path):
        audit_dir = tmp_path / "audit_empty"
        _write_audit_artifacts(pd.DataFrame(), audit_dir)
        assert (audit_dir / "vif_table.csv").exists()
        assert (audit_dir / "stability_metrics.csv").exists()
        assert (audit_dir / "quality_status.txt").exists()

    def test_audit_features_has_hinge_column(self, tmp_path):
        compare_df = pd.DataFrame({
            "galaxy": ["G0"] * 30,
            "r_kpc": np.linspace(1, 10, 30),
            "g_bar": np.logspace(-11.5, -9.5, 30),
            "g_obs": np.logspace(-11.0, -9.0, 30),
            "log_g_bar": np.linspace(-11.5, -9.5, 30),
            "log_g_obs": np.linspace(-11.0, -9.0, 30),
        })
        audit_dir = tmp_path / "audit_feat"
        _write_audit_artifacts(compare_df, audit_dir)

        feat = pd.read_csv(audit_dir / "audit_features.csv")
        assert "hinge" in feat.columns
        assert (feat["hinge"] >= 0).all()


# ---------------------------------------------------------------------------
# run_pipeline now creates audit/ subdirectory
# ---------------------------------------------------------------------------

class TestRunPipelineAuditDir:
    def test_creates_audit_subdir(self, sparc_dir, tmp_path):
        out = tmp_path / "results"
        run_pipeline(sparc_dir, out, verbose=False)
        audit = out / "audit"
        assert audit.is_dir()

    def test_audit_contains_required_files(self, sparc_dir, tmp_path):
        out = tmp_path / "results2"
        run_pipeline(sparc_dir, out, verbose=False)
        audit = out / "audit"
        for fname in ("vif_table.csv", "stability_metrics.csv",
                      "quality_status.txt", "audit_features.csv"):
            assert (audit / fname).exists(), f"Missing {fname}"

    def test_vif_hinge_is_finite(self, sparc_dir, tmp_path):
        out = tmp_path / "results3"
        run_pipeline(sparc_dir, out, verbose=False)
        vif_df = pd.read_csv(out / "audit" / "vif_table.csv")
        hinge_row = vif_df[vif_df["feature"] == "hinge"]
        assert len(hinge_row) == 1
        assert np.isfinite(float(hinge_row["VIF"].iloc[0]))


# ---------------------------------------------------------------------------
# scripts/audit_scm.py — run_oos_audit
# ---------------------------------------------------------------------------

class TestRunOosAudit:
    def test_creates_csv(self, sparc_dir, tmp_path):
        from scripts.audit_scm import run_oos_audit
        df = run_oos_audit(sparc_dir, tmp_path / "oos", verbose=False)
        csv_path = tmp_path / "oos" / "audit" / "residual_vs_hinge.csv"
        assert csv_path.exists(), "residual_vs_hinge.csv not created"

    def test_creates_png(self, sparc_dir, tmp_path):
        from scripts.audit_scm import run_oos_audit
        run_oos_audit(sparc_dir, tmp_path / "oos2", verbose=False)
        png_path = tmp_path / "oos2" / "audit" / "residual_vs_hinge.png"
        assert png_path.exists(), "residual_vs_hinge.png not created"

    def test_creates_json(self, sparc_dir, tmp_path):
        from scripts.audit_scm import run_oos_audit
        run_oos_audit(sparc_dir, tmp_path / "oos3", verbose=False)
        json_path = tmp_path / "oos3" / "audit" / "oos_summary.json"
        assert json_path.exists(), "oos_summary.json not created"
        with open(json_path) as fh:
            summary = json.load(fh)
        assert "residual_scm_median" in summary
        assert "improvement_median" in summary

    def test_csv_columns(self, sparc_dir, tmp_path):
        from scripts.audit_scm import run_oos_audit
        df = run_oos_audit(sparc_dir, tmp_path / "oos4", verbose=False)
        required = {"galaxy", "hinge", "residual_scm", "improvement"}
        assert required.issubset(set(df.columns)), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_hinge_non_negative(self, sparc_dir, tmp_path):
        from scripts.audit_scm import run_oos_audit
        df = run_oos_audit(sparc_dir, tmp_path / "oos5", verbose=False)
        assert (df["hinge"] >= 0).all(), "hinge values should be non-negative"

    def test_returns_dataframe(self, sparc_dir, tmp_path):
        from scripts.audit_scm import run_oos_audit
        result = run_oos_audit(sparc_dir, tmp_path / "oos6", verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
