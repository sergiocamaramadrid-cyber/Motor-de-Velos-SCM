"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Validates:
  - load_and_prepare: column derivation, row filtering
  - run_groupkfold: output shape, column contracts
  - run_permutation_test: output structure and types
  - run_model_comparison: AICc ordering, delta_AICc
  - freeze_master_coeffs: coefficient dict structure
  - main (CLI): all 7 output files are written
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow importing from scripts/ without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from audit_scm import (
    KPC_TO_M,
    _CONV,
    freeze_master_coeffs,
    load_and_prepare,
    main,
    run_groupkfold,
    run_model_comparison,
    run_permutation_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_flat_csv(tmp_path_factory):
    """20-galaxy, 15-point flat CSV written to a temp file."""
    root = tmp_path_factory.mktemp("audit_data")
    rng = np.random.default_rng(99)
    n_gal, n_pts = 20, 15
    rows = []
    for i in range(n_gal):
        gid = f"G{i:03d}"
        vf = 80.0 + i * 12.0
        r = np.linspace(0.5, 10.0, n_pts)
        v_obs = vf + rng.normal(0, 3, n_pts)
        v_bar = 0.65 * vf * np.ones(n_pts)
        g_bar = v_bar ** 2 / r * _CONV
        for j in range(n_pts):
            rows.append({
                "galaxy_id": gid,
                "r": float(r[j]),
                "v_obs": float(max(v_obs[j], 1.0)),
                "e_v_obs": 5.0,
                "v_bar": float(v_bar[j]),
                "g_bar": float(g_bar[j]),
            })
    csv_path = root / "sparc_flat.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def prepared_df(small_flat_csv):
    return load_and_prepare(small_flat_csv)


# ---------------------------------------------------------------------------
# 0) load_and_prepare
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_required_derived_columns_present(self, prepared_df):
        assert "log_gbar" in prepared_df.columns
        assert "log_vobs" in prepared_df.columns

    def test_no_nonpositive_v_obs(self, prepared_df):
        assert (prepared_df["v_obs"] > 0).all()

    def test_no_nonpositive_g_bar(self, prepared_df):
        assert (prepared_df["g_bar"] > 0).all()

    def test_galaxy_id_present(self, prepared_df):
        assert "galaxy_id" in prepared_df.columns
        assert prepared_df["galaxy_id"].notna().all()

    def test_strict_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_and_prepare(bad_csv, strict=True)

    def test_derives_log_gbar_from_v_bar(self, tmp_path):
        """If g_bar is absent but v_bar is present, log_gbar should be derived."""
        rows = [{"galaxy_id": "A", "r": 1.0, "v_obs": 100.0,
                 "e_v_obs": 5.0, "v_bar": 70.0} for _ in range(5)]
        csv = tmp_path / "vbar.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        df = load_and_prepare(csv)
        assert "log_gbar" in df.columns
        assert (df["log_gbar"].notna()).all()


# ---------------------------------------------------------------------------
# 1) GroupKFold
# ---------------------------------------------------------------------------

class TestGroupKFold:
    def test_metrics_shape(self, prepared_df):
        metrics, _, _ = run_groupkfold(prepared_df, n_splits=5)
        assert len(metrics) == 5

    def test_metrics_columns(self, prepared_df):
        metrics, _, _ = run_groupkfold(prepared_df, n_splits=5)
        for col in ("fold", "rmse_log_vobs", "bias_log_vobs",
                    "n_train_points", "n_test_points",
                    "n_train_galaxies", "n_test_galaxies"):
            assert col in metrics.columns, f"Missing column: {col}"

    def test_per_galaxy_n_galaxies(self, prepared_df):
        _, per_gal, _ = run_groupkfold(prepared_df, n_splits=5)
        assert len(per_gal) == prepared_df["galaxy_id"].nunique()

    def test_per_galaxy_columns(self, prepared_df):
        _, per_gal, _ = run_groupkfold(prepared_df, n_splits=5)
        for col in ("galaxy_id", "n_points", "rmse_log_vobs"):
            assert col in per_gal.columns

    def test_coeffs_shape(self, prepared_df):
        _, _, coeffs = run_groupkfold(prepared_df, n_splits=5)
        assert len(coeffs) == 5
        assert "fold" in coeffs.columns
        assert "intercept" in coeffs.columns

    def test_rmse_positive(self, prepared_df):
        metrics, _, _ = run_groupkfold(prepared_df, n_splits=5)
        assert (metrics["rmse_log_vobs"] > 0).all()

    def test_no_galaxy_in_train_and_test(self, prepared_df):
        """GroupKFold must never expose the same galaxy to both sets."""
        from sklearn.model_selection import GroupKFold
        from audit_scm import _select_features

        y = prepared_df["log_vobs"].values
        X, _ = _select_features(prepared_df, include_hinge=True, base_only=False)
        groups = prepared_df["galaxy_id"].values
        gkf = GroupKFold(n_splits=5)
        for train_idx, test_idx in gkf.split(X, y, groups):
            train_gal = set(groups[train_idx])
            test_gal = set(groups[test_idx])
            assert train_gal.isdisjoint(test_gal), (
                "Galaxy appears in both train and test sets!"
            )


# ---------------------------------------------------------------------------
# 2) Permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_summary_keys(self, prepared_df):
        summary, _ = run_permutation_test(
            prepared_df, n_splits=5, n_permutations=5, seed=7
        )
        for key in ("real_rmse", "perm_rmse_mean", "perm_rmse_std",
                    "delta_rmse_real_vs_perm", "p_value_le",
                    "n_permutations", "significant"):
            assert key in summary, f"Missing key: {key}"

    def test_runs_df_length(self, prepared_df):
        _, runs = run_permutation_test(
            prepared_df, n_splits=5, n_permutations=5, seed=7
        )
        assert len(runs) == 5

    def test_runs_df_columns(self, prepared_df):
        _, runs = run_permutation_test(
            prepared_df, n_splits=5, n_permutations=5, seed=7
        )
        assert "permutation" in runs.columns
        assert "rmse_log_vobs" in runs.columns

    def test_p_value_in_range(self, prepared_df):
        summary, _ = run_permutation_test(
            prepared_df, n_splits=5, n_permutations=5, seed=7
        )
        assert 0.0 <= summary["p_value_le"] <= 1.0

    def test_significant_is_bool(self, prepared_df):
        summary, _ = run_permutation_test(
            prepared_df, n_splits=5, n_permutations=5, seed=7
        )
        assert isinstance(summary["significant"], bool)


# ---------------------------------------------------------------------------
# 3) Model duel (AICc)
# ---------------------------------------------------------------------------

class TestModelComparison:
    def test_three_models(self, prepared_df):
        comp = run_model_comparison(prepared_df)
        assert set(comp["model"]) == {"btfr", "scm_base", "scm_full"}

    def test_columns(self, prepared_df):
        comp = run_model_comparison(prepared_df)
        for col in ("model", "n_features", "k_total", "n_points",
                    "LL", "AICc", "delta_AICc", "RMSE_log_vobs"):
            assert col in comp.columns, f"Missing column: {col}"

    def test_delta_aicc_minimum_is_zero(self, prepared_df):
        comp = run_model_comparison(prepared_df)
        assert float(comp["delta_AICc"].min()) == pytest.approx(0.0)

    def test_sorted_by_aicc(self, prepared_df):
        comp = run_model_comparison(prepared_df)
        assert list(comp["AICc"]) == sorted(comp["AICc"].tolist())

    def test_scm_full_has_more_features(self, prepared_df):
        comp = run_model_comparison(prepared_df)
        n_btfr = int(comp.loc[comp["model"] == "btfr", "n_features"].iloc[0])
        n_full = int(comp.loc[comp["model"] == "scm_full", "n_features"].iloc[0])
        assert n_full >= n_btfr


# ---------------------------------------------------------------------------
# 4) Master coefficient freeze
# ---------------------------------------------------------------------------

class TestMasterCoeffs:
    def test_keys_present(self, prepared_df):
        coeffs = freeze_master_coeffs(prepared_df)
        for key in ("model", "n_points", "n_galaxies", "intercept",
                    "features", "coefficients", "scaler_mean",
                    "scaler_scale", "RMSE_log_vobs"):
            assert key in coeffs, f"Missing key: {key}"

    def test_feature_coeff_length_match(self, prepared_df):
        coeffs = freeze_master_coeffs(prepared_df)
        assert len(coeffs["features"]) == len(coeffs["coefficients"])
        assert len(coeffs["features"]) == len(coeffs["scaler_mean"])

    def test_rmse_positive(self, prepared_df):
        coeffs = freeze_master_coeffs(prepared_df)
        assert coeffs["RMSE_log_vobs"] > 0

    def test_json_serialisable(self, prepared_df):
        coeffs = freeze_master_coeffs(prepared_df)
        dumped = json.dumps(coeffs)
        loaded = json.loads(dumped)
        assert loaded["model"] == "scm_full"


# ---------------------------------------------------------------------------
# CLI / integration test
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_all_outputs_written(self, small_flat_csv, tmp_path):
        out_dir = tmp_path / "audit_out"
        main([
            "--input", str(small_flat_csv),
            "--outdir", str(out_dir),
            "--ref", "test123",
            "--seed", "0",
            "--kfold", "5",
            "--permutations", "5",
        ])
        suffix = "-vtest123"
        expected = [
            f"groupkfold_metrics{suffix}.csv",
            f"groupkfold_per_galaxy{suffix}.csv",
            f"coeffs_by_fold{suffix}.csv",
            f"permutation_summary{suffix}.json",
            f"permutation_runs{suffix}.csv",
            f"model_comparison_aicc{suffix}.csv",
            f"master_coeffs{suffix}.json",
        ]
        for fname in expected:
            assert (out_dir / fname).exists(), f"Missing output: {fname}"

    def test_skip_permutation_flag(self, small_flat_csv, tmp_path):
        out_dir = tmp_path / "audit_skip"
        main([
            "--input", str(small_flat_csv),
            "--outdir", str(out_dir),
            "--ref", "skip",
            "--kfold", "5",
            "--skip-permutation",
        ])
        # Permutation files should NOT exist
        assert not (out_dir / "permutation_summary-vskip.json").exists()
        # But other outputs should
        assert (out_dir / "groupkfold_metrics-vskip.csv").exists()
        assert (out_dir / "master_coeffs-vskip.json").exists()

    def test_master_coeffs_has_git_ref(self, small_flat_csv, tmp_path):
        out_dir = tmp_path / "audit_ref"
        main([
            "--input", str(small_flat_csv),
            "--outdir", str(out_dir),
            "--ref", "abc1234",
            "--kfold", "5",
            "--skip-permutation",
        ])
        with open(out_dir / "master_coeffs-vabc1234.json", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["git_ref"] == "abc1234"
