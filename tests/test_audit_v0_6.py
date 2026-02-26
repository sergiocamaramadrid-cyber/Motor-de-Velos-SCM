"""
tests/test_audit_v0_6.py — Unit tests for src/audit_v0_6.py.

Validates the three invariants required by the structural audit:
  (i)  hinge VIF in [2.0, 5.0]  (PASS criterion)
  (ii) condition_number_kappa < 30  (PASS criterion)
  (iii) OVERALL STATUS == "PASS"
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit_v0_6 import (
    _generate_synthetic_data,
    build_design_matrix,
    compute_vif,
    compute_condition_number,
    run_audit,
    VIF_HINGE_MIN,
    VIF_HINGE_MAX,
    KAPPA_PASS,
)


class TestGenerateSyntheticData:
    def test_output_shapes(self):
        n_gal, n_pts = 10, 5
        r, vb, vv = _generate_synthetic_data(n_galaxies=n_gal, n_pts=n_pts)
        assert r.shape == (n_gal * n_pts,)
        assert vb.shape == (n_gal * n_pts,)
        assert vv.shape == (n_gal * n_pts,)

    def test_positive_values(self):
        r, vb, vv = _generate_synthetic_data(n_galaxies=5, n_pts=4)
        assert np.all(r > 0)
        assert np.all(vb > 0)
        assert np.all(vv > 0)


class TestBuildDesignMatrix:
    def test_output_shape(self):
        r, vb, vv = _generate_synthetic_data(n_galaxies=10, n_pts=5)
        X, names = build_design_matrix(r, vb, vv)
        assert X.shape[0] == len(r)
        assert X.shape[1] == len(names) == 5

    def test_feature_names_contains_hinge(self):
        r, vb, vv = _generate_synthetic_data(n_galaxies=5, n_pts=5)
        _, names = build_design_matrix(r, vb, vv)
        assert "hinge" in names

    def test_hinge_non_negative(self):
        r, vb, vv = _generate_synthetic_data(n_galaxies=20, n_pts=10)
        X, names = build_design_matrix(r, vb, vv)
        hinge_col = X[:, names.index("hinge")]
        assert np.all(hinge_col >= 0)


class TestComputeVif:
    def test_independent_predictors_have_vif_near_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        df = compute_vif(X, ["a", "b", "c"])
        assert df.shape == (3, 2)
        assert all(df["vif"] < 1.5), "Independent predictors should have VIF ≈ 1"

    def test_returns_dataframe_with_correct_columns(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 2))
        df = compute_vif(X, ["x1", "x2"])
        assert list(df.columns) == ["feature", "vif"]
        assert list(df["feature"]) == ["x1", "x2"]


class TestComputeConditionNumber:
    def test_independent_columns_give_small_kappa(self):
        rng = np.random.default_rng(99)
        X = rng.normal(size=(500, 4))
        kappa = compute_condition_number(X)
        assert kappa < 5.0, f"Independent random columns should have small κ, got {kappa:.2f}"

    def test_returns_positive_float(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 4))
        kappa = compute_condition_number(X)
        assert kappa > 0
        assert np.isfinite(kappa)


class TestRunAudit:
    def test_hinge_vif_in_pass_range(self, tmp_path):
        result = run_audit(out_dir=tmp_path)
        vif_df = result["vif_table"]
        hinge_vif = float(vif_df.loc[vif_df["feature"] == "hinge", "vif"].iloc[0])
        assert VIF_HINGE_MIN <= hinge_vif <= VIF_HINGE_MAX, (
            f"hinge VIF={hinge_vif:.3f} outside PASS range [{VIF_HINGE_MIN}, {VIF_HINGE_MAX}]"
        )

    def test_kappa_in_pass_range(self, tmp_path):
        result = run_audit(out_dir=tmp_path)
        kappa = result["condition_number_kappa"]
        assert kappa < KAPPA_PASS, (
            f"condition_number_kappa={kappa:.1f} exceeds PASS threshold {KAPPA_PASS}"
        )

    def test_overall_status_is_pass(self, tmp_path):
        result = run_audit(out_dir=tmp_path)
        assert result["overall_status"] == "PASS"

    def test_output_files_exist(self, tmp_path):
        run_audit(out_dir=tmp_path)
        assert (tmp_path / "vif_table.csv").exists()
        assert (tmp_path / "stability_metrics.csv").exists()
        assert (tmp_path / "quality_status.txt").exists()

    def test_vif_table_csv_content(self, tmp_path):
        run_audit(out_dir=tmp_path)
        df = pd.read_csv(tmp_path / "vif_table.csv")
        assert list(df.columns) == ["feature", "vif"]
        assert "hinge" in df["feature"].values
        assert (df["vif"] > 0).all()

    def test_stability_metrics_csv_content(self, tmp_path):
        run_audit(out_dir=tmp_path)
        df = pd.read_csv(tmp_path / "stability_metrics.csv")
        assert list(df.columns) == ["metric", "value", "status"]
        assert "condition_number_kappa" in df["metric"].values
        row = df[df["metric"] == "condition_number_kappa"].iloc[0]
        assert row["value"] < KAPPA_PASS
        assert row["status"] == "stable"

    def test_quality_status_txt_content(self, tmp_path):
        run_audit(out_dir=tmp_path)
        text = (tmp_path / "quality_status.txt").read_text(encoding="utf-8")
        assert "OVERALL STATUS: PASS" in text
