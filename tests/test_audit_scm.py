"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Uses synthetic per-radial-point data so no real SPARC download is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.audit_scm import (
    scm_btfr_pred,
    scm_nohinge_pred,
    scm_full_pred,
    fit_model,
    safe_aicc,
    _group_kfold_splits,
    groupkfold_audit,
    model_comparison,
    permutation_test,
    run_audit,
    MODEL_K,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_galaxies: int = 20, n_pts: int = 15, seed: int = 0) -> pd.DataFrame:
    """Synthetic per-radial-point DataFrame resembling universal_term_comparison_full.csv."""
    rng = np.random.default_rng(seed)
    G0 = 1.2e-10  # m/s²
    rows = []
    for i in range(n_galaxies):
        name = f"G{i:03d}"
        r_kpc = np.linspace(0.5, 15.0, n_pts)
        # Simulate MOND-like: log_g_obs ≈ 0.5·log_g_bar + 0.5·log(g0) + noise
        g_bar = rng.uniform(0.01 * G0, 2.0 * G0, n_pts)
        log_g_bar = np.log10(g_bar)
        log_g_obs = 0.5 * log_g_bar + 0.5 * np.log10(G0) + rng.normal(0, 0.02, n_pts)
        for k in range(n_pts):
            rows.append({
                "galaxy": name,
                "r_kpc": float(r_kpc[k]),
                "g_bar": float(g_bar[k]),
                "g_obs": float(10.0 ** log_g_obs[k]),
                "log_g_bar": float(log_g_bar[k]),
                "log_g_obs": float(log_g_obs[k]),
            })
    return pd.DataFrame(rows)


def _make_csv(tmp_path: Path, **kwargs) -> Path:
    df = _make_df(**kwargs)
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Model predictor tests
# ---------------------------------------------------------------------------

class TestModelPredictors:
    def _arrays(self, n=50, seed=1):
        rng = np.random.default_rng(seed)
        log_g_bar = rng.uniform(-12, -9, n)
        r_norm = rng.uniform(0.0, 2.0, n)
        return log_g_bar, r_norm

    def test_btfr_shape(self):
        lg, rn = self._arrays()
        out = scm_btfr_pred(lg, rn, np.array([0.5, 0.0]))
        assert out.shape == lg.shape

    def test_nohinge_shape(self):
        lg, rn = self._arrays()
        out = scm_nohinge_pred(lg, rn, np.array([0.5, 0.0, 0.0, 0.0]))
        assert out.shape == lg.shape

    def test_full_shape(self):
        lg, rn = self._arrays()
        out = scm_full_pred(lg, rn, np.array([0.5, 0.0, 0.0, 0.0, 0.1, -10.0]))
        assert out.shape == lg.shape

    def test_btfr_linear(self):
        """BTFR must equal β·x + C exactly."""
        lg = np.array([-11.0, -10.5, -10.0])
        rn = np.zeros(3)
        beta, C = 0.4, -0.2
        np.testing.assert_allclose(
            scm_btfr_pred(lg, rn, np.array([beta, C])),
            beta * lg + C,
        )

    def test_nohinge_reduces_to_btfr_when_extra_zero(self):
        """SCM no-hinge reduces to BTFR when a=b=0."""
        lg = np.linspace(-12, -9, 30)
        rn = np.zeros(30)
        params_btfr = np.array([0.5, 0.1])
        params_nohinge = np.array([0.5, 0.1, 0.0, 0.0])
        np.testing.assert_allclose(
            scm_nohinge_pred(lg, rn, params_nohinge),
            scm_btfr_pred(lg, rn, params_btfr),
            rtol=1e-10,
        )

    def test_full_hinge_activates_below_logg0(self):
        """Hinge term d·max(0, logg0−lg) must be positive for lg < logg0."""
        logg0 = -10.5
        lg_below = np.array([-11.0, -11.5])
        lg_above = np.array([-10.0, -9.5])
        rn = np.zeros(2)
        params = np.array([0.5, 0.0, 0.0, 0.0, 1.0, logg0])
        out_below = scm_full_pred(lg_below, rn, params)
        out_above = scm_full_pred(lg_above, rn, params)
        # Below the hinge: extra positive contribution
        assert np.all(out_below > scm_nohinge_pred(lg_below, rn, params[:4]))
        # Above the hinge: no extra contribution
        np.testing.assert_allclose(
            out_above, scm_nohinge_pred(lg_above, rn, params[:4]), rtol=1e-10
        )

    def test_all_finite(self):
        rng = np.random.default_rng(99)
        lg = rng.uniform(-13, -8, 200)
        rn = rng.uniform(0, 3, 200)
        for pred_fn, _, p0 in [
            (scm_btfr_pred, None, np.array([0.5, 0.0])),
            (scm_nohinge_pred, None, np.array([0.5, 0.0, 0.01, 0.01])),
            (scm_full_pred, None, np.array([0.5, 0.0, 0.01, 0.01, 0.1, -10.0])),
        ]:
            out = pred_fn(lg, rn, p0)
            assert np.all(np.isfinite(out)), f"{pred_fn.__name__} produced non-finite output"


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------

class TestFitModel:
    def test_recovers_btfr_params(self):
        """Fit should recover planted β=0.5, C=−5.2 on noise-free data."""
        rng = np.random.default_rng(10)
        lg = rng.uniform(-12, -9, 100)
        rn = np.zeros(100)
        beta_true, C_true = 0.5, -5.2
        y = scm_btfr_pred(lg, rn, np.array([beta_true, C_true]))
        params, rmse = fit_model(lg, rn, y, scm_btfr_pred, np.array([0.4, -4.0]))
        assert params[0] == pytest.approx(beta_true, abs=0.01)
        assert params[1] == pytest.approx(C_true, abs=0.05)
        assert rmse < 1e-4

    def test_rmse_non_negative(self):
        df = _make_df(n_galaxies=5, n_pts=10)
        lg = df["log_g_bar"].to_numpy()
        rn = df["r_kpc"].to_numpy() / 10.0
        lo = df["log_g_obs"].to_numpy()
        _, rmse = fit_model(lg, rn, lo, scm_full_pred,
                            np.array([0.5, 0.0, 0.0, 0.0, 0.1, -10.0]))
        assert rmse >= 0.0

    def test_returns_correct_param_length(self):
        df = _make_df(n_galaxies=5, n_pts=10)
        lg = df["log_g_bar"].to_numpy()
        rn = df["r_kpc"].to_numpy() / 10.0
        lo = df["log_g_obs"].to_numpy()
        for pred_fn, k, p0 in [(scm_btfr_pred, 2, np.array([0.5, 0.0])),
                                (scm_nohinge_pred, 4, np.array([0.5, 0.0, 0.0, 0.0])),
                                (scm_full_pred, 6, np.array([0.5, 0.0, 0.0, 0.0, 0.1, -10.0]))]:
            params, _ = fit_model(lg, rn, lo, pred_fn, p0.copy())
            assert len(params) == k


# ---------------------------------------------------------------------------
# safe_aicc
# ---------------------------------------------------------------------------

class TestSafeAicc:
    def test_increases_with_k(self):
        """AICc must penalise more parameters (same LL)."""
        rng = np.random.default_rng(5)
        y = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        aicc2 = safe_aicc(y, y_pred, k=2)
        aicc6 = safe_aicc(y, y_pred, k=6)
        assert aicc6 > aicc2

    def test_k_values_match_model_registry(self):
        """k values hard-coded in MODEL_K must be 2, 4, 6."""
        assert MODEL_K["btfr"] == 2
        assert MODEL_K["scm_nohinge"] == 4
        assert MODEL_K["scm_full"] == 6

    def test_perfect_prediction_finite(self):
        y = np.linspace(-12, -9, 50)
        assert np.isfinite(safe_aicc(y, y, k=2))

    def test_large_n_approaches_aic(self):
        """For large n the small-sample correction vanishes: AICc → AIC."""
        y = np.zeros(10000)
        y_pred = y.copy()
        sigma2 = max(np.mean((y - y_pred) ** 2), 1e-30)
        n = len(y)
        k = 3
        ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
        aic_expected = -2.0 * ll + 2.0 * k
        aic_c = safe_aicc(y, y_pred, k=k)
        assert aic_c == pytest.approx(aic_expected, rel=0.01)


# ---------------------------------------------------------------------------
# _group_kfold_splits
# ---------------------------------------------------------------------------

class TestGroupKfoldSplits:
    def _groups(self, n_galaxies=20, n_pts=10):
        return np.array([f"G{i:03d}" for i in range(n_galaxies)
                         for _ in range(n_pts)])

    def test_no_galaxy_leaks(self):
        """No galaxy must appear in both train and test in the same fold."""
        groups = self._groups()
        for train_idx, test_idx in _group_kfold_splits(groups, n_splits=5):
            train_gals = set(groups[train_idx])
            test_gals = set(groups[test_idx])
            assert train_gals.isdisjoint(test_gals)

    def test_every_point_covered(self):
        """Union of all test indices must cover every row exactly once."""
        groups = self._groups()
        all_test = []
        for _, test_idx in _group_kfold_splits(groups, n_splits=5):
            all_test.extend(test_idx.tolist())
        assert sorted(all_test) == list(range(len(groups)))

    def test_n_splits_folds(self):
        groups = self._groups(n_galaxies=20)
        folds = list(_group_kfold_splits(groups, n_splits=5))
        assert len(folds) == 5

    def test_empty_fold_skipped(self):
        """If n_splits > n_galaxies, some folds may be empty and are skipped."""
        groups = np.array(["A", "A", "B", "B"])
        folds = list(_group_kfold_splits(groups, n_splits=5))
        # Only 2 unique groups → at most 2 valid folds
        assert len(folds) <= 2

    def test_reproducible_with_same_seed(self):
        groups = self._groups()
        splits1 = [(t.tolist(), v.tolist())
                   for t, v in _group_kfold_splits(groups, seed=7)]
        splits2 = [(t.tolist(), v.tolist())
                   for t, v in _group_kfold_splits(groups, seed=7)]
        assert splits1 == splits2

    def test_different_seeds_differ(self):
        groups = self._groups(n_galaxies=30)
        splits1 = [v.tolist() for _, v in _group_kfold_splits(groups, seed=0)]
        splits2 = [v.tolist() for _, v in _group_kfold_splits(groups, seed=99)]
        assert splits1 != splits2


# ---------------------------------------------------------------------------
# groupkfold_audit
# ---------------------------------------------------------------------------

class TestGroupKfoldAudit:
    @pytest.fixture(scope="class")
    def kf_result(self):
        df = _make_df(n_galaxies=20, n_pts=10, seed=42)
        return groupkfold_audit(df, n_splits=5, seed=0)

    def test_returns_expected_keys(self, kf_result):
        for key in ("rmse_train", "rmse_test", "params", "n_folds", "param_names"):
            assert key in kf_result

    def test_n_folds(self, kf_result):
        assert kf_result["n_folds"] == 5

    def test_rmse_lists_length(self, kf_result):
        assert len(kf_result["rmse_train"]) == kf_result["n_folds"]
        assert len(kf_result["rmse_test"]) == kf_result["n_folds"]

    def test_params_length(self, kf_result):
        for p in kf_result["params"]:
            assert len(p) == 6

    def test_rmse_non_negative(self, kf_result):
        assert all(r >= 0 for r in kf_result["rmse_train"])
        assert all(r >= 0 for r in kf_result["rmse_test"])

    def test_param_names(self, kf_result):
        assert kf_result["param_names"] == ["beta", "C", "a", "b", "d", "logg0"]

    def test_params_vary_across_folds(self, kf_result):
        """Parameters must be re-fitted per fold, not identical."""
        params_arr = np.array(kf_result["params"])
        # beta should not be exactly the same in all folds
        assert params_arr[:, 0].std() > 0.0, "beta identical in every fold — no per-fold fitting"

    def test_rmse_finite(self, kf_result):
        assert all(np.isfinite(r) for r in kf_result["rmse_train"])
        assert all(np.isfinite(r) for r in kf_result["rmse_test"])


# ---------------------------------------------------------------------------
# model_comparison
# ---------------------------------------------------------------------------

class TestModelComparison:
    @pytest.fixture(scope="class")
    def comp_result(self):
        df = _make_df(n_galaxies=20, n_pts=10, seed=1)
        return model_comparison(df)

    def test_all_models_present(self, comp_result):
        assert set(comp_result.keys()) == {"btfr", "scm_nohinge", "scm_full"}

    def test_k_values_correct(self, comp_result):
        assert comp_result["btfr"]["k"] == 2
        assert comp_result["scm_nohinge"]["k"] == 4
        assert comp_result["scm_full"]["k"] == 6

    def test_delta_aicc_winner_is_zero(self, comp_result):
        min_delta = min(v["delta_aicc"] for v in comp_result.values())
        assert min_delta == pytest.approx(0.0, abs=1e-8)

    def test_aicc_finite(self, comp_result):
        for v in comp_result.values():
            assert np.isfinite(v["aicc"])

    def test_rmse_non_negative(self, comp_result):
        for v in comp_result.values():
            assert v["rmse"] >= 0.0

    def test_params_length_matches_k(self, comp_result):
        for name, v in comp_result.items():
            assert len(v["params"]) == v["k"], (
                f"{name}: expected {v['k']} params, got {len(v['params'])}"
            )

    def test_full_model_rmse_le_btfr(self, comp_result):
        """More-flexible SCM-full must fit at least as well as BTFR."""
        assert comp_result["scm_full"]["rmse"] <= comp_result["btfr"]["rmse"] + 1e-6


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    @pytest.fixture(scope="class")
    def perm_result(self):
        df = _make_df(n_galaxies=20, n_pts=10, seed=77)
        return permutation_test(df, n_perm=20, seed=0)

    def test_returns_expected_keys(self, perm_result):
        for key in ("rmse_real", "rmse_perm", "p_value"):
            assert key in perm_result

    def test_p_value_in_range(self, perm_result):
        assert 0.0 <= perm_result["p_value"] <= 1.0

    def test_rmse_real_non_negative(self, perm_result):
        assert perm_result["rmse_real"] >= 0.0

    def test_rmse_perm_array_length(self, perm_result):
        assert len(perm_result["rmse_perm"]) == 20

    def test_perm_rmse_non_negative(self, perm_result):
        assert np.all(perm_result["rmse_perm"] >= 0.0)

    def test_perm_rmse_larger_than_real_on_average(self):
        """Permuting predictors should (on average) increase RMSE."""
        df = _make_df(n_galaxies=30, n_pts=15, seed=5)
        result = permutation_test(df, n_perm=50, seed=0)
        # Median permuted RMSE should exceed real RMSE for genuine data
        assert np.median(result["rmse_perm"]) > result["rmse_real"]


# ---------------------------------------------------------------------------
# run_audit  (integration)
# ---------------------------------------------------------------------------

class TestRunAudit:
    def test_creates_output_files(self, tmp_path):
        csv = _make_csv(tmp_path, n_galaxies=15, n_pts=8)
        out = tmp_path / "audit_out"
        run_audit(csv, out, n_splits=3, n_perm=5)
        assert (out / "kfold_results.csv").exists()
        assert (out / "model_comparison.csv").exists()
        assert (out / "permutation_test.csv").exists()
        assert (out / "audit.log").exists()

    def test_returns_expected_keys(self, tmp_path):
        csv = _make_csv(tmp_path, n_galaxies=15, n_pts=8)
        out = tmp_path / "audit_out2"
        result = run_audit(csv, out, n_splits=3, n_perm=5)
        assert set(result.keys()) == {"kfold", "comparison", "permutation"}

    def test_missing_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "log_g_bar": [-11.0]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            run_audit(bad, tmp_path / "out")

    def test_kfold_csv_columns(self, tmp_path):
        csv = _make_csv(tmp_path, n_galaxies=15, n_pts=8)
        out = tmp_path / "audit_out3"
        run_audit(csv, out, n_splits=3, n_perm=5)
        kf_df = pd.read_csv(out / "kfold_results.csv")
        for col in ("fold", "rmse_train", "rmse_test", "beta", "C", "a", "b", "d", "logg0"):
            assert col in kf_df.columns, f"Missing column: {col}"

    def test_model_comparison_csv_k_values(self, tmp_path):
        csv = _make_csv(tmp_path, n_galaxies=15, n_pts=8)
        out = tmp_path / "audit_out4"
        run_audit(csv, out, n_splits=3, n_perm=5)
        comp_df = pd.read_csv(out / "model_comparison.csv")
        k_by_model = dict(zip(comp_df["model"], comp_df["k"]))
        assert k_by_model["btfr"] == 2
        assert k_by_model["scm_nohinge"] == 4
        assert k_by_model["scm_full"] == 6

    def test_audit_log_contains_rmse(self, tmp_path):
        csv = _make_csv(tmp_path, n_galaxies=15, n_pts=8)
        out = tmp_path / "audit_out5"
        run_audit(csv, out, n_splits=3, n_perm=5)
        log_text = (out / "audit.log").read_text(encoding="utf-8")
        assert "RMSE Train" in log_text
        assert "RMSE Test" in log_text
