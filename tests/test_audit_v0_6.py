"""
tests/test_audit_v0_6.py — Tests for the Motor de Velos v0.6 model audit.

Validates that the audit script:
  1. Produces the three required output files.
  2. hinge VIF falls in the structural PASS range [2.0, 5.0].
  3. condition_number_kappa is below the PASS threshold (< 30).
  4. OVERALL STATUS is 'PASS'.
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_audit_v0_6 import run_audit, _build_design_matrix, _compute_vif

import numpy as np


# ---------------------------------------------------------------------------
# run_audit integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audit_results(tmp_path_factory):
    """Run the audit once and return (vif_df, stability_df, overall, out_dir)."""
    out_dir = tmp_path_factory.mktemp("audit_v0_6")
    vif_df, stability_df, overall = run_audit(out_dir=out_dir)
    return vif_df, stability_df, overall, out_dir


class TestAuditOutputFiles:
    def test_vif_table_exists(self, audit_results):
        _, _, _, out_dir = audit_results
        assert (out_dir / "vif_table.csv").exists()

    def test_stability_metrics_exists(self, audit_results):
        _, _, _, out_dir = audit_results
        assert (out_dir / "stability_metrics.csv").exists()

    def test_quality_status_exists(self, audit_results):
        _, _, _, out_dir = audit_results
        assert (out_dir / "quality_status.txt").exists()


class TestVifTable:
    def test_has_expected_columns(self, audit_results):
        vif_df, _, _, _ = audit_results
        assert list(vif_df.columns) == ["predictor", "vif", "status"]

    def test_has_hinge_row(self, audit_results):
        vif_df, _, _, _ = audit_results
        assert "hinge" in vif_df["predictor"].values

    def test_hinge_vif_pass_range(self, audit_results):
        vif_df, _, _, _ = audit_results
        hinge_vif = float(vif_df.loc[vif_df["predictor"] == "hinge", "vif"].iloc[0])
        assert 2.0 <= hinge_vif <= 5.0, (
            f"hinge VIF {hinge_vif:.4f} outside structural PASS range [2.0, 5.0]"
        )

    def test_hinge_status_is_pass(self, audit_results):
        vif_df, _, _, _ = audit_results
        status = vif_df.loc[vif_df["predictor"] == "hinge", "status"].iloc[0]
        assert status == "PASS"

    def test_csv_matches_dataframe(self, audit_results):
        vif_df, _, _, out_dir = audit_results
        csv_df = pd.read_csv(out_dir / "vif_table.csv")
        pd.testing.assert_frame_equal(
            csv_df.reset_index(drop=True),
            vif_df.reset_index(drop=True),
        )


class TestStabilityMetrics:
    def test_has_expected_columns(self, audit_results):
        _, stability_df, _, _ = audit_results
        assert list(stability_df.columns) == ["metric", "value", "status"]

    def test_kappa_row_present(self, audit_results):
        _, stability_df, _, _ = audit_results
        assert "condition_number_kappa" in stability_df["metric"].values

    def test_kappa_below_pass_threshold(self, audit_results):
        _, stability_df, _, _ = audit_results
        kappa = float(
            stability_df.loc[
                stability_df["metric"] == "condition_number_kappa", "value"
            ].iloc[0]
        )
        assert kappa < 30.0, f"kappa {kappa:.2f} ≥ 30 (PASS threshold)"

    def test_kappa_status_stable(self, audit_results):
        _, stability_df, _, _ = audit_results
        status = stability_df.loc[
            stability_df["metric"] == "condition_number_kappa", "status"
        ].iloc[0]
        assert status == "stable"


class TestQualityStatus:
    def test_overall_pass(self, audit_results):
        _, _, overall, _ = audit_results
        assert overall == "PASS"

    def test_quality_status_file_content(self, audit_results):
        _, _, _, out_dir = audit_results
        text = (out_dir / "quality_status.txt").read_text(encoding="utf-8")
        assert "OVERALL STATUS: PASS" in text


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestBuildDesignMatrix:
    def test_shape(self):
        X = _build_design_matrix()
        assert X.shape == (175 * 20, 3)

    def test_intercept_column(self):
        X = _build_design_matrix()
        np.testing.assert_array_equal(X[:, 0], np.ones(175 * 20))

    def test_no_nan_or_inf(self):
        X = _build_design_matrix()
        assert np.all(np.isfinite(X))

    def test_hinge_range(self):
        """Hinge column should be in [0, 1] (r/r_max)."""
        X = _build_design_matrix()
        assert X[:, 2].min() >= 0.0
        assert X[:, 2].max() <= 1.0 + 1e-9


class TestComputeVif:
    def test_orthogonal_columns_give_vif_one(self):
        """Perfectly orthogonal predictors should yield VIF ≈ 1."""
        n = 100
        X = np.column_stack([
            np.ones(n),
            np.sin(np.linspace(0, 2 * np.pi, n)),
            np.cos(np.linspace(0, 2 * np.pi, n)),
        ])
        col_std = X.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_norm = X / col_std
        vifs = _compute_vif(X_norm)
        # sin and cos are orthogonal → VIF ≈ 1
        assert abs(vifs[1] - 1.0) < 0.05
        assert abs(vifs[2] - 1.0) < 0.05

    def test_perfectly_correlated_gives_large_vif(self):
        """Near-perfectly collinear predictors should give large VIF."""
        n = 100
        x = np.linspace(0, 1, n)
        X = np.column_stack([np.ones(n), x, x + 1e-6 * np.arange(n)])
        col_std = X.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_norm = X / col_std
        vifs = _compute_vif(X_norm)
        assert vifs[1] > 100
