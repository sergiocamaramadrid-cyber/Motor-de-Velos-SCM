"""
tests/test_analyze_env_real.py — Tests for scripts/analyze_env_real.py.

Covers: load_crossmatched_table, fit_mass_only, fit_full,
        permutation_pvalue, summarize, save_outputs, main (CLI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_env_real import (
    fit_full,
    fit_mass_only,
    load_crossmatched_table,
    permutation_pvalue,
    save_outputs,
    summarize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(
    n: int = 30,
    seed: int = 0,
    *,
    env_effect: float = 0.0,
) -> pd.DataFrame:
    """Synthetic crossmatched table.

    Parameters
    ----------
    n : int
        Number of rows.
    seed : int
        RNG seed.
    env_effect : float
        Coefficient on e_env added to delta_f3 to simulate a real effect.
    """
    rng = np.random.default_rng(seed)
    logM = rng.uniform(8.0, 11.5, n)
    e_env = rng.uniform(-1.0, 1.0, n)
    noise = rng.normal(0.0, 0.1, n)
    delta_f3 = 0.3 * logM + env_effect * e_env + noise
    return pd.DataFrame(
        {
            "galaxy_name": [f"G{i:03d}" for i in range(n)],
            "logM": logM,
            "delta_f3": delta_f3,
            "e_env": e_env,
        }
    )


# ---------------------------------------------------------------------------
# load_crossmatched_table
# ---------------------------------------------------------------------------

class TestLoadCrossmatched:
    def test_loads_valid_csv(self, tmp_path):
        df = _make_table()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = load_crossmatched_table(p)
        assert len(loaded) == len(df)

    def test_missing_column_raises(self, tmp_path):
        df = _make_table().drop(columns=["e_env"])
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_crossmatched_table(p)

    def test_missing_multiple_columns_listed(self, tmp_path):
        df = _make_table().drop(columns=["e_env", "logM"])
        p = tmp_path / "bad2.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_crossmatched_table(p)

    def test_nonfinite_rows_dropped(self, tmp_path):
        df = _make_table(n=10)
        df.loc[0, "logM"] = float("nan")
        df.loc[1, "e_env"] = float("nan")
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = load_crossmatched_table(p)
        assert len(loaded) == 8

    def test_galaxy_name_stripped(self, tmp_path):
        df = _make_table(n=5)
        df["galaxy_name"] = ["  G1  ", "G2 ", " G3", "G4", "G5"]
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = load_crossmatched_table(p)
        assert all(loaded["galaxy_name"] == loaded["galaxy_name"].str.strip())

    def test_numeric_coercion(self, tmp_path):
        df = _make_table(n=5)
        df["logM"] = df["logM"].astype(str)
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = load_crossmatched_table(p)
        assert pd.api.types.is_float_dtype(loaded["logM"])

    def test_index_reset(self, tmp_path):
        df = _make_table(n=10)
        df.loc[3, "logM"] = float("nan")
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = load_crossmatched_table(p)
        assert list(loaded.index) == list(range(len(loaded)))

    def test_required_columns_present(self, tmp_path):
        df = _make_table()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = load_crossmatched_table(p)
        for col in ["galaxy_name", "logM", "delta_f3", "e_env"]:
            assert col in loaded.columns


# ---------------------------------------------------------------------------
# fit_mass_only
# ---------------------------------------------------------------------------

class TestFitMassOnly:
    def test_returns_three_elements(self):
        df = _make_table()
        result = fit_mass_only(df)
        assert len(result) == 3

    def test_predictions_length(self):
        df = _make_table(n=25)
        _, pred, _ = fit_mass_only(df)
        assert len(pred) == 25

    def test_residuals_length(self):
        df = _make_table(n=25)
        _, pred, resid = fit_mass_only(df)
        assert len(resid) == 25

    def test_residuals_sum_near_zero(self):
        df = _make_table(n=50)
        _, _, resid = fit_mass_only(df)
        assert abs(resid.sum()) < 1e-8

    def test_model_has_logM_coef(self):
        df = _make_table()
        model, _, _ = fit_mass_only(df)
        assert "logM" in model.params.index

    def test_model_rsquared_in_unit_interval(self):
        df = _make_table()
        model, _, _ = fit_mass_only(df)
        assert 0.0 <= model.rsquared <= 1.0

    def test_residuals_equal_y_minus_pred(self):
        df = _make_table(n=20)
        _, pred, resid = fit_mass_only(df)
        expected = df["delta_f3"].values - pred.values
        np.testing.assert_allclose(resid.values, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# fit_full
# ---------------------------------------------------------------------------

class TestFitFull:
    def test_returns_model(self):
        df = _make_table()
        model = fit_full(df)
        assert model is not None

    def test_model_has_env_coef(self):
        df = _make_table()
        model = fit_full(df)
        assert "e_env" in model.params.index

    def test_model_has_logM_coef(self):
        df = _make_table()
        model = fit_full(df)
        assert "logM" in model.params.index

    def test_full_r2_ge_base_r2(self):
        df = _make_table(n=40, env_effect=0.5)
        model_base, _, _ = fit_mass_only(df)
        model_full = fit_full(df)
        assert model_full.rsquared >= model_base.rsquared - 1e-9

    def test_env_effect_detected(self):
        """With a large env effect, e_env coef should be significant."""
        rng = np.random.default_rng(7)
        n = 100
        logM = rng.uniform(8, 11, n)
        e_env = rng.uniform(-1, 1, n)
        delta_f3 = 0.3 * logM + 1.5 * e_env + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"galaxy_name": [f"G{i}" for i in range(n)],
                           "logM": logM, "delta_f3": delta_f3, "e_env": e_env})
        model = fit_full(df)
        assert model.pvalues["e_env"] < 0.05

    def test_model_aic_is_finite(self):
        df = _make_table()
        model = fit_full(df)
        assert np.isfinite(model.aic)


# ---------------------------------------------------------------------------
# permutation_pvalue
# ---------------------------------------------------------------------------

class TestPermutationPvalue:
    def test_returns_four_elements(self):
        df = _make_table(n=30, env_effect=0.5)
        _, _, resid = fit_mass_only(df)
        result = permutation_pvalue(resid, df["e_env"], n_perms=100, seed=0)
        assert len(result) == 4

    def test_rho_in_valid_range(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        rho, _, _, _ = permutation_pvalue(resid, df["e_env"], n_perms=100, seed=0)
        assert -1.0 <= rho <= 1.0

    def test_p_spearman_in_unit_interval(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        _, p, _, _ = permutation_pvalue(resid, df["e_env"], n_perms=100, seed=0)
        assert 0.0 <= p <= 1.0

    def test_p_perm_in_unit_interval(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        _, _, _, p_perm = permutation_pvalue(resid, df["e_env"], n_perms=100, seed=0)
        assert 0.0 <= p_perm <= 1.0

    def test_perm_list_length(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        _, _, perm, _ = permutation_pvalue(resid, df["e_env"], n_perms=200, seed=0)
        assert len(perm) == 200

    def test_seed_reproducibility(self):
        df = _make_table(n=40, env_effect=0.3)
        _, _, resid = fit_mass_only(df)
        r1 = permutation_pvalue(resid, df["e_env"], n_perms=50, seed=99)
        r2 = permutation_pvalue(resid, df["e_env"], n_perms=50, seed=99)
        assert r1[0] == r2[0]
        assert r1[3] == r2[3]

    def test_strong_effect_low_p_perm(self):
        rng = np.random.default_rng(3)
        n = 80
        env = rng.uniform(-1, 1, n)
        resid = pd.Series(env + rng.normal(0, 0.01, n))
        env_s = pd.Series(env)
        _, _, _, p_perm = permutation_pvalue(resid, env_s, n_perms=500, seed=42)
        assert p_perm < 0.05

    def test_no_effect_large_p_perm(self):
        rng = np.random.default_rng(7)
        n = 200
        resid = pd.Series(rng.normal(0, 1, n))
        env_s = pd.Series(rng.normal(0, 1, n))
        _, _, _, p_perm = permutation_pvalue(resid, env_s, n_perms=1000, seed=42)
        assert p_perm > 0.05


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def _run(self, n=30, env_effect=0.3):
        df = _make_table(n=n, env_effect=env_effect)
        model_base, pred, resid = fit_mass_only(df)
        df["pred_mass"] = pred
        df["residual_mass"] = resid
        rho, p_spear, _, p_perm = permutation_pvalue(
            df["residual_mass"], df["e_env"], n_perms=50, seed=1
        )
        model_full = fit_full(df)
        return summarize(df, model_base, model_full, rho, p_spear, p_perm)

    def test_returns_dict(self):
        assert isinstance(self._run(), dict)

    def test_required_keys_present(self):
        s = self._run()
        for key in [
            "N", "rho_residual_env", "p_spearman", "p_perm",
            "aic_base", "aic_full", "delta_aic",
            "coef_logM_full", "p_logM_full",
            "coef_env_full", "p_env_full",
            "r2_base", "r2_full",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_N_matches_input(self):
        s = self._run(n=25)
        assert s["N"] == 25

    def test_delta_aic_is_base_minus_full(self):
        df = _make_table(n=30)
        model_base, pred, resid = fit_mass_only(df)
        df["pred_mass"] = pred
        df["residual_mass"] = resid
        rho, p_spear, _, p_perm = permutation_pvalue(
            df["residual_mass"], df["e_env"], n_perms=50, seed=0
        )
        model_full = fit_full(df)
        s = summarize(df, model_base, model_full, rho, p_spear, p_perm)
        assert s["delta_aic"] == pytest.approx(model_base.aic - model_full.aic, rel=1e-9)

    def test_r2_values_in_unit_interval(self):
        s = self._run()
        assert 0.0 <= s["r2_base"] <= 1.0
        assert 0.0 <= s["r2_full"] <= 1.0

    def test_all_values_finite(self):
        s = self._run()
        for k, v in s.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"Non-finite value for key: {k}"


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------

class TestSaveOutputs:
    def _summary(self):
        return {
            "N": 20,
            "rho_residual_env": 0.35,
            "p_spearman": 0.12,
            "p_perm": 0.08,
            "aic_base": -50.0,
            "aic_full": -55.0,
            "delta_aic": 5.0,
            "coef_logM_full": 0.3,
            "p_logM_full": 0.001,
            "coef_env_full": 0.1,
            "p_env_full": 0.04,
            "r2_base": 0.4,
            "r2_full": 0.5,
        }

    def test_creates_analysis_table_csv(self, tmp_path):
        df = _make_table(n=10)
        save_outputs(df, self._summary(), tmp_path)
        assert (tmp_path / "env_real_analysis_table.csv").exists()

    def test_creates_summary_csv(self, tmp_path):
        df = _make_table(n=10)
        save_outputs(df, self._summary(), tmp_path)
        assert (tmp_path / "env_real_summary.csv").exists()

    def test_creates_summary_txt(self, tmp_path):
        df = _make_table(n=10)
        save_outputs(df, self._summary(), tmp_path)
        assert (tmp_path / "env_real_summary.txt").exists()

    def test_creates_out_dir_if_missing(self, tmp_path):
        subdir = tmp_path / "new" / "deep"
        df = _make_table(n=5)
        save_outputs(df, self._summary(), subdir)
        assert subdir.exists()

    def test_analysis_table_has_correct_rows(self, tmp_path):
        df = _make_table(n=15)
        save_outputs(df, self._summary(), tmp_path)
        loaded = pd.read_csv(tmp_path / "env_real_analysis_table.csv")
        assert len(loaded) == 15

    def test_summary_csv_has_correct_keys(self, tmp_path):
        df = _make_table(n=10)
        s = self._summary()
        save_outputs(df, s, tmp_path)
        loaded = pd.read_csv(tmp_path / "env_real_summary.csv")
        for k in s:
            assert k in loaded.columns

    def test_summary_txt_contains_key_value_pairs(self, tmp_path):
        df = _make_table(n=5)
        s = self._summary()
        save_outputs(df, s, tmp_path)
        text = (tmp_path / "env_real_summary.txt").read_text(encoding="utf-8")
        assert "N: 20" in text
        assert "delta_aic: 5.0" in text


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_runs(self, tmp_path):
        df = _make_table(n=40, env_effect=0.4)
        in_csv = tmp_path / "input.csv"
        df.to_csv(in_csv, index=False)

        df_in = load_crossmatched_table(in_csv)
        model_base, pred, resid = fit_mass_only(df_in)
        df_in["pred_mass"] = pred
        df_in["residual_mass"] = resid
        rho, p_spear, _, p_perm = permutation_pvalue(
            df_in["residual_mass"], df_in["e_env"], n_perms=100, seed=42
        )
        model_full = fit_full(df_in)
        summary = summarize(df_in, model_base, model_full, rho, p_spear, p_perm)
        save_outputs(df_in, summary, tmp_path / "out")

        assert (tmp_path / "out" / "env_real_summary.csv").exists()
        assert (tmp_path / "out" / "env_real_analysis_table.csv").exists()
        assert summary["N"] == 40

    def test_analysis_table_has_pred_and_residual_cols(self, tmp_path):
        df = _make_table(n=20)
        in_csv = tmp_path / "input.csv"
        df.to_csv(in_csv, index=False)
        df_in = load_crossmatched_table(in_csv)
        _, pred, resid = fit_mass_only(df_in)
        df_in["pred_mass"] = pred
        df_in["residual_mass"] = resid
        save_outputs(df_in, {"N": 20}, tmp_path / "out")
        loaded = pd.read_csv(tmp_path / "out" / "env_real_analysis_table.csv")
        assert "pred_mass" in loaded.columns
        assert "residual_mass" in loaded.columns
