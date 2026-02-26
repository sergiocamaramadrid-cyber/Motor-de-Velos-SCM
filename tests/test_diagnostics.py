"""
tests/test_diagnostics.py — Unit and integration tests for src/diagnostics.py.

Tests cover:
* compute_vif        — known collinear and orthogonal matrices
* compute_condition_number — known matrices
* compute_partial_correlations — known relationships
* build_feature_matrix — synthetic pipeline data
* build_audit_table   — galaxy-level aggregation and index integrity
* run_diagnostics     — all four output files are produced with valid content
* G0_HINGE constant   — frozen value is correct
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import (
    G0_HINGE,
    VIF_THRESHOLD_LOW,
    VIF_THRESHOLD_HIGH,
    build_feature_matrix,
    build_audit_table,
    compute_condition_number,
    compute_partial_correlations,
    compute_vif,
    run_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_compare_df(n_galaxies=5, n_points=10, seed=0):
    """Synthetic per-radial-point data mimicking universal_term_comparison_full.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_galaxies):
        name = f"G{i:04d}"
        r = np.linspace(0.5, 15.0, n_points)
        g_bar = 1e-11 * rng.uniform(0.5, 5.0, n_points)
        g_obs = g_bar * rng.uniform(1.0, 3.0, n_points)
        for k in range(n_points):
            rows.append(
                {
                    "galaxy": name,
                    "r_kpc": r[k],
                    "g_bar": g_bar[k],
                    "g_obs": g_obs[k],
                    "log_g_bar": np.log10(g_bar[k]),
                    "log_g_obs": np.log10(g_obs[k]),
                }
            )
    return pd.DataFrame(rows)


def _make_per_galaxy_df(n_galaxies=5, seed=0):
    """Synthetic per-galaxy summary mimicking per_galaxy_summary.csv."""
    rng = np.random.default_rng(seed)
    v_flats = np.linspace(100.0, 300.0, n_galaxies)
    a0 = 1.2e-10
    from src.scm_models import KPC_TO_M, baryonic_tully_fisher

    return pd.DataFrame(
        {
            "galaxy": [f"G{i:04d}" for i in range(n_galaxies)],
            "upsilon_disk": rng.uniform(0.5, 2.0, n_galaxies),
            "chi2_reduced": rng.uniform(0.5, 3.0, n_galaxies),
            "n_points": np.full(n_galaxies, 10, dtype=int),
            "Vflat_kms": v_flats,
            "M_bar_BTFR_Msun": baryonic_tully_fisher(v_flats, a0=a0),
        }
    )


@pytest.fixture()
def synthetic_data():
    c = _make_compare_df(n_galaxies=10, n_points=20)
    g = _make_per_galaxy_df(n_galaxies=10)
    return c, g


# ---------------------------------------------------------------------------
# G0_HINGE constant
# ---------------------------------------------------------------------------

class TestG0Hinge:
    def test_frozen_value(self):
        """G0_HINGE must equal exactly 3.5e-11 m/s² (frozen for v0.6.0)."""
        assert G0_HINGE == pytest.approx(3.5e-11, rel=1e-9)

    def test_is_float(self):
        assert isinstance(G0_HINGE, float)


# ---------------------------------------------------------------------------
# compute_vif
# ---------------------------------------------------------------------------

class TestComputeVIF:
    def test_orthogonal_predictors_give_vif_one(self):
        """Perfectly orthogonal predictors → VIF = 1."""
        rng = np.random.default_rng(7)
        n = 200
        # Build three orthogonal columns via QR decomposition
        A = rng.standard_normal((n, 3))
        Q, _ = np.linalg.qr(A)
        vifs = compute_vif(Q)
        np.testing.assert_allclose(vifs, np.ones(3), atol=0.05)

    def test_collinear_pair_raises_vif(self):
        """If X[:, 1] ≈ X[:, 0], VIF for both should be large."""
        rng = np.random.default_rng(13)
        n = 100
        x0 = rng.standard_normal(n)
        X = np.column_stack([x0, x0 + 1e-6 * rng.standard_normal(n),
                             rng.standard_normal(n)])
        vifs = compute_vif(X)
        assert vifs[0] > VIF_THRESHOLD_HIGH
        assert vifs[1] > VIF_THRESHOLD_HIGH

    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        vifs = compute_vif(X)
        assert vifs.shape == (3,)

    def test_vif_positive(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 3))
        vifs = compute_vif(X)
        assert np.all(vifs >= 1.0 - 1e-6)

    def test_constant_column_gives_inf(self):
        rng = np.random.default_rng(2)
        X = np.column_stack([rng.standard_normal(50),
                             np.ones(50),
                             rng.standard_normal(50)])
        vifs = compute_vif(X)
        assert np.isinf(vifs[1])


# ---------------------------------------------------------------------------
# compute_condition_number
# ---------------------------------------------------------------------------

class TestComputeConditionNumber:
    def test_identity_gives_one(self):
        """Standardised identity → condition number = 1."""
        n = 50
        # Build a matrix whose columns are already orthonormal after standardising
        X = np.eye(n, 3) * 10.0 + np.random.default_rng(3).standard_normal((n, 3)) * 0.01
        kappa = compute_condition_number(X)
        assert kappa >= 1.0

    def test_increases_with_collinearity(self):
        rng = np.random.default_rng(5)
        n = 100
        x = rng.standard_normal(n)
        X_ok = np.column_stack([x, rng.standard_normal(n), rng.standard_normal(n)])
        X_bad = np.column_stack([x, x + 1e-4 * rng.standard_normal(n),
                                  rng.standard_normal(n)])
        kappa_ok = compute_condition_number(X_ok)
        kappa_bad = compute_condition_number(X_bad)
        assert kappa_bad > kappa_ok

    def test_returns_float(self):
        rng = np.random.default_rng(6)
        kappa = compute_condition_number(rng.standard_normal((30, 3)))
        assert isinstance(kappa, float)

    def test_positive(self):
        rng = np.random.default_rng(9)
        kappa = compute_condition_number(rng.standard_normal((40, 3)))
        assert kappa > 0.0


# ---------------------------------------------------------------------------
# compute_partial_correlations
# ---------------------------------------------------------------------------

class TestComputePartialCorrelations:
    def test_returns_all_predictor_keys(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((60, 3))
        y = rng.standard_normal(60)
        names = ["log_M_bar", "log_g_bar", "log_j_star"]
        result = compute_partial_correlations(X, y, names)
        assert set(result.keys()) == set(names)

    def test_values_in_minus_one_to_one(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((80, 3))
        y = rng.standard_normal(80)
        result = compute_partial_correlations(X, y, ["a", "b", "c"])
        for v in result.values():
            assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9

    def test_strong_relationship_detected(self):
        """Partial corr of the true predictor should be large in magnitude."""
        rng = np.random.default_rng(12)
        n = 200
        x0 = rng.standard_normal(n)
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 3.0 * x0 + 0.01 * rng.standard_normal(n)  # only x0 matters
        X = np.column_stack([x0, x1, x2])
        result = compute_partial_correlations(X, y, ["x0", "x1", "x2"])
        assert abs(result["x0"]) > 0.9
        assert abs(result["x1"]) < 0.2
        assert abs(result["x2"]) < 0.2

    def test_constant_predictor_returns_zero(self):
        rng = np.random.default_rng(14)
        n = 60
        X = np.column_stack([rng.standard_normal(n),
                             np.ones(n),
                             rng.standard_normal(n)])
        y = rng.standard_normal(n)
        result = compute_partial_correlations(X, y, ["a", "const", "c"])
        assert result["const"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_expected_columns(self, synthetic_data):
        c, g = synthetic_data
        feat = build_feature_matrix(c, g)
        for col in ["galaxy", "r_kpc", "log_M_bar", "log_g_bar",
                    "log_j_star", "log_v_obs", "velo_dominated"]:
            assert col in feat.columns, f"Missing column: {col}"

    def test_no_invalid_rows(self, synthetic_data):
        c, g = synthetic_data
        feat = build_feature_matrix(c, g)
        for col in ["log_M_bar", "log_g_bar", "log_j_star", "log_v_obs"]:
            assert feat[col].notna().all(), f"NaN found in {col}"
            assert np.isfinite(feat[col].values).all(), f"Inf found in {col}"

    def test_velo_dominated_uses_g0_hinge(self, synthetic_data):
        """velo_dominated flag is consistent with G0_HINGE."""
        c, g = synthetic_data
        feat = build_feature_matrix(c, g)
        expected = feat.apply(
            lambda row: 10.0 ** row["log_g_bar"] < G0_HINGE, axis=1
        )
        pd.testing.assert_series_equal(
            feat["velo_dominated"], expected, check_names=False
        )

    def test_log_base_ten(self, synthetic_data):
        """log_M_bar == log10(M_bar); log_g_bar == log10(g_bar)."""
        c, g = synthetic_data
        feat = build_feature_matrix(c, g)
        merged = c.merge(g[["galaxy", "M_bar_BTFR_Msun"]], on="galaxy")
        merged = merged[(merged["g_bar"] > 0) & (merged["r_kpc"] > 0)
                        & (merged["g_obs"] > 0) & (merged["M_bar_BTFR_Msun"] > 0)]
        expected_log_m = np.log10(merged["M_bar_BTFR_Msun"].values)
        actual_log_m = feat["log_M_bar"].values
        np.testing.assert_allclose(
            np.sort(actual_log_m), np.sort(expected_log_m), rtol=1e-6
        )

    def test_row_count_matches_valid_points(self, synthetic_data):
        c, g = synthetic_data
        feat = build_feature_matrix(c, g)
        merged = c.merge(g[["galaxy", "M_bar_BTFR_Msun"]], on="galaxy")
        valid = (
            (merged["M_bar_BTFR_Msun"] > 0)
            & (merged["g_bar"] > 0)
            & (merged["g_obs"] > 0)
            & (merged["r_kpc"] > 0)
        )
        assert len(feat) == valid.sum()


# ---------------------------------------------------------------------------
# build_audit_table
# ---------------------------------------------------------------------------

class TestBuildAuditTable:
    def test_one_row_per_galaxy(self, synthetic_data):
        c, g = synthetic_data
        audit = build_audit_table(c, g)
        assert len(audit) == len(g)

    def test_no_index_gaps(self, synthetic_data):
        """Index must be 0 … N-1 with no gaps."""
        c, g = synthetic_data
        audit = build_audit_table(c, g)
        assert list(audit.index) == list(range(len(audit)))

    def test_expected_columns(self, synthetic_data):
        c, g = synthetic_data
        audit = build_audit_table(c, g)
        for col in ["galaxy", "log_M_bar", "log_g_bar", "log_j_star",
                    "log_v_obs", "velo_dominated", "n_points"]:
            assert col in audit.columns

    def test_sorted_by_galaxy(self, synthetic_data):
        c, g = synthetic_data
        audit = build_audit_table(c, g)
        assert list(audit["galaxy"]) == sorted(audit["galaxy"].tolist())

    def test_n_points_non_negative(self, synthetic_data):
        c, g = synthetic_data
        audit = build_audit_table(c, g)
        assert (audit["n_points"] >= 0).all()

    def test_large_sample_175_galaxies(self):
        """Audit table must accommodate a full 175-galaxy sample with no gaps."""
        n = 175
        c = _make_compare_df(n_galaxies=n, n_points=20, seed=99)
        g = _make_per_galaxy_df(n_galaxies=n, seed=99)
        audit = build_audit_table(c, g)
        assert len(audit) == n
        assert list(audit.index) == list(range(n))


# ---------------------------------------------------------------------------
# run_diagnostics
# ---------------------------------------------------------------------------

class TestRunDiagnostics:
    @pytest.fixture()
    def diag_result(self, synthetic_data, tmp_path):
        c, g = synthetic_data
        feat = build_feature_matrix(c, g)
        summary = run_diagnostics(feat, tmp_path / "diag", verbose=False)
        return summary, tmp_path / "diag"

    def test_vif_results_csv_exists(self, diag_result):
        _, d = diag_result
        assert (d / "vif_results.csv").exists()

    def test_condition_number_txt_exists(self, diag_result):
        _, d = diag_result
        assert (d / "condition_number.txt").exists()

    def test_partial_correlation_json_exists(self, diag_result):
        _, d = diag_result
        assert (d / "partial_correlation.json").exists()

    def test_diagnostics_summary_json_exists(self, diag_result):
        _, d = diag_result
        assert (d / "diagnostics_summary.json").exists()

    def test_vif_csv_has_three_rows(self, diag_result):
        _, d = diag_result
        df = pd.read_csv(d / "vif_results.csv")
        assert len(df) == 3
        assert set(df["predictor"]) == {"log_M_bar", "log_g_bar", "log_j_star"}

    def test_condition_number_txt_contains_key(self, diag_result):
        _, d = diag_result
        text = (d / "condition_number.txt").read_text(encoding="utf-8")
        assert "condition_number:" in text
        assert "g0_hinge_frozen:" in text

    def test_partial_correlation_json_keys(self, diag_result):
        _, d = diag_result
        with open(d / "partial_correlation.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert set(data.keys()) == {"log_M_bar", "log_g_bar", "log_j_star"}

    def test_summary_json_contains_g0_hinge(self, diag_result):
        summary, _ = diag_result
        assert summary["g0_hinge_frozen_m_s2"] == pytest.approx(3.5e-11, rel=1e-9)

    def test_summary_json_scm_version(self, diag_result):
        summary, _ = diag_result
        assert summary["scm_version"] == "v0.6.0"

    def test_summary_json_n_galaxies_positive(self, diag_result):
        summary, _ = diag_result
        assert summary["n_galaxies"] > 0

    def test_vif_values_finite(self, diag_result):
        summary, _ = diag_result
        for name, v in summary["vif"].items():
            assert np.isfinite(v), f"VIF for {name} is not finite"

    def test_condition_number_positive(self, diag_result):
        summary, _ = diag_result
        assert summary["condition_number"] > 0.0


# ---------------------------------------------------------------------------
# Integration: run_pipeline produces audit and diagnostics artefacts
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    @pytest.fixture(scope="class")
    def pipeline_result(self, tmp_path_factory):
        from src.scm_analysis import run_pipeline

        n = 20
        root = tmp_path_factory.mktemp("sparc_diag")
        rng = np.random.default_rng(42)
        names = [f"H{i:03d}" for i in range(n)]
        v_flats = np.linspace(80.0, 300.0, n)

        gal_df = pd.DataFrame(
            {
                "Galaxy": names,
                "D": np.linspace(5.0, 50.0, n),
                "Inc": np.linspace(30.0, 80.0, n),
                "L36": 1e9 * np.arange(1, n + 1, dtype=float),
                "Vflat": v_flats,
                "e_Vflat": np.full(n, 5.0),
            }
        )
        gal_df.to_csv(root / "SPARC_Lelli2016c.csv", index=False)

        for name, vf in zip(names, v_flats):
            n_pts = 15
            r = np.linspace(0.5, 12.0, n_pts)
            rc = pd.DataFrame(
                {
                    "r": r,
                    "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
                    "v_obs_err": np.full(n_pts, 5.0),
                    "v_gas": 0.3 * vf * np.ones(n_pts),
                    "v_disk": 0.75 * vf * np.ones(n_pts),
                    "v_bul": np.zeros(n_pts),
                    "SBdisk": np.zeros(n_pts),
                    "SBbul": np.zeros(n_pts),
                }
            )
            rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

        out = tmp_path_factory.mktemp("out_diag")
        run_pipeline(root, out, verbose=False)
        return out

    def test_audit_sparc_global_csv_exists(self, pipeline_result):
        assert (pipeline_result / "audit" / "sparc_global.csv").exists()

    def test_audit_no_index_gaps(self, pipeline_result):
        df = pd.read_csv(pipeline_result / "audit" / "sparc_global.csv")
        assert list(df.index) == list(range(len(df)))

    def test_audit_has_galaxy_column(self, pipeline_result):
        df = pd.read_csv(pipeline_result / "audit" / "sparc_global.csv")
        assert "galaxy" in df.columns

    def test_diagnostics_summary_json_written(self, pipeline_result):
        assert (pipeline_result / "diagnostics" / "diagnostics_summary.json").exists()

    def test_vif_results_csv_written(self, pipeline_result):
        assert (pipeline_result / "diagnostics" / "vif_results.csv").exists()

    def test_condition_number_txt_written(self, pipeline_result):
        assert (pipeline_result / "diagnostics" / "condition_number.txt").exists()

    def test_partial_correlation_json_written(self, pipeline_result):
        assert (pipeline_result / "diagnostics" / "partial_correlation.json").exists()

    def test_g0_hinge_frozen_in_summary(self, pipeline_result):
        with open(
            pipeline_result / "diagnostics" / "diagnostics_summary.json",
            encoding="utf-8",
        ) as fh:
            data = json.load(fh)
        assert data["g0_hinge_frozen_m_s2"] == pytest.approx(3.5e-11, rel=1e-9)
