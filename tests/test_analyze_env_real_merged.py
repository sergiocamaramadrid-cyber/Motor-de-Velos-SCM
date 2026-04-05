"""
tests/test_analyze_env_real_merged.py — Tests for
scripts/analyze_env_real_merged.py.

Covers: load_merged_table (galaxy/galaxy_name normalisation, e_env_err
        passthrough, validation errors), fit_mass_only, fit_full,
        permutation_pvalue, summarize (ΔBIC/ΔR² keys), save_outputs,
        and the main() CLI entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_env_real_merged import (
    fit_full,
    fit_mass_only,
    load_merged_table,
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
    col_galaxy: str = "galaxy_name",
    include_err: bool = False,
) -> pd.DataFrame:
    """Return a synthetic merged table."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(8.0, 11.5, n)
    e_env = rng.uniform(-1.0, 1.0, n)
    noise = rng.normal(0.0, 0.1, n)
    delta_f3 = 0.3 * logM + env_effect * e_env + noise
    data = {
        col_galaxy: [f"G{i:03d}" for i in range(n)],
        "logM": logM,
        "delta_f3": delta_f3,
        "e_env": e_env,
    }
    if include_err:
        data["e_env_err"] = rng.uniform(0.01, 0.5, n)
    return pd.DataFrame(data)


def _write_csv(df: pd.DataFrame, tmp_path: Path, name: str = "merged.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_merged_table
# ---------------------------------------------------------------------------

class TestLoadMergedTable:
    def test_galaxy_name_col_accepted(self, tmp_path):
        df = _make_table(col_galaxy="galaxy_name")
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert "galaxy_name" in out.columns

    def test_galaxy_col_normalised(self, tmp_path):
        df = _make_table(col_galaxy="galaxy")
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert "galaxy_name" in out.columns
        assert "galaxy" not in out.columns

    def test_galaxy_name_takes_precedence_over_galaxy(self, tmp_path):
        df = _make_table(col_galaxy="galaxy_name")
        df["galaxy"] = "extra"
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert "galaxy_name" in out.columns

    def test_missing_id_column_raises(self, tmp_path):
        df = _make_table().drop(columns=["galaxy_name"])
        p = _write_csv(df, tmp_path)
        with pytest.raises(ValueError, match="galaxy"):
            load_merged_table(p)

    def test_missing_logM_raises(self, tmp_path):
        df = _make_table().drop(columns=["logM"])
        p = _write_csv(df, tmp_path)
        with pytest.raises(ValueError, match="logM"):
            load_merged_table(p)

    def test_missing_delta_f3_raises(self, tmp_path):
        df = _make_table().drop(columns=["delta_f3"])
        p = _write_csv(df, tmp_path)
        with pytest.raises(ValueError, match="delta_f3"):
            load_merged_table(p)

    def test_missing_e_env_raises(self, tmp_path):
        df = _make_table().drop(columns=["e_env"])
        p = _write_csv(df, tmp_path)
        with pytest.raises(ValueError, match="e_env"):
            load_merged_table(p)

    def test_e_env_err_preserved_when_present(self, tmp_path):
        df = _make_table(include_err=True)
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert "e_env_err" in out.columns

    def test_e_env_err_absent_when_not_present(self, tmp_path):
        df = _make_table(include_err=False)
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert "e_env_err" not in out.columns

    def test_non_finite_rows_dropped(self, tmp_path):
        df = _make_table(n=20)
        df.loc[0, "logM"] = float("nan")
        df.loc[1, "delta_f3"] = float("inf")
        df.loc[2, "e_env"] = float("nan")
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert len(out) == 17

    def test_numeric_coercion(self, tmp_path):
        df = _make_table(n=5)
        df["logM"] = df["logM"].astype(str)
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert pd.api.types.is_float_dtype(out["logM"])

    def test_galaxy_name_stripped(self, tmp_path):
        df = _make_table(n=3)
        df["galaxy_name"] = ["  G000  ", " G001", "G002 "]
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert list(out["galaxy_name"]) == ["G000", "G001", "G002"]

    def test_returns_dataframe(self, tmp_path):
        p = _write_csv(_make_table(), tmp_path)
        out = load_merged_table(p)
        assert isinstance(out, pd.DataFrame)

    def test_row_count(self, tmp_path):
        p = _write_csv(_make_table(n=25), tmp_path)
        out = load_merged_table(p)
        assert len(out) == 25

    def test_e_env_err_nan_not_dropped(self, tmp_path):
        df = _make_table(n=10, include_err=True)
        df.loc[0, "e_env_err"] = float("nan")
        p = _write_csv(df, tmp_path)
        out = load_merged_table(p)
        assert len(out) == 10  # NaN in e_env_err does NOT drop the row


# ---------------------------------------------------------------------------
# fit_mass_only
# ---------------------------------------------------------------------------

class TestFitMassOnly:
    def test_returns_three_items(self):
        df = _make_table(n=30)
        result = fit_mass_only(df)
        assert len(result) == 3

    def test_residuals_length(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        assert len(resid) == 30

    def test_pred_length(self):
        df = _make_table(n=30)
        _, pred, _ = fit_mass_only(df)
        assert len(pred) == 30

    def test_residuals_mean_near_zero(self):
        df = _make_table(n=100)
        _, _, resid = fit_mass_only(df)
        assert abs(resid.mean()) < 0.1

    def test_model_has_logM_param(self):
        df = _make_table(n=30)
        model, _, _ = fit_mass_only(df)
        assert "logM" in model.params.index

    def test_model_has_const_param(self):
        df = _make_table(n=30)
        model, _, _ = fit_mass_only(df)
        assert "const" in model.params.index


# ---------------------------------------------------------------------------
# fit_full
# ---------------------------------------------------------------------------

class TestFitFull:
    def test_has_e_env_param(self):
        df = _make_table(n=30)
        model = fit_full(df)
        assert "e_env" in model.params.index

    def test_has_logM_param(self):
        df = _make_table(n=30)
        model = fit_full(df)
        assert "logM" in model.params.index

    def test_r2_positive(self):
        df = _make_table(n=50)
        model = fit_full(df)
        assert model.rsquared >= 0.0

    def test_coef_sign_with_positive_env_effect(self):
        df = _make_table(n=200, env_effect=0.5, seed=7)
        model = fit_full(df)
        assert model.params["e_env"] > 0

    def test_nobs_matches(self):
        df = _make_table(n=40)
        model = fit_full(df)
        assert int(model.nobs) == 40


# ---------------------------------------------------------------------------
# permutation_pvalue
# ---------------------------------------------------------------------------

class TestPermutationPvalue:
    def test_returns_four_items(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        result = permutation_pvalue(resid, df["e_env"], n_perms=50, seed=0)
        assert len(result) == 4

    def test_rho_in_minus_one_one(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        rho, _, _, _ = permutation_pvalue(resid, df["e_env"], n_perms=50, seed=0)
        assert -1.0 <= rho <= 1.0

    def test_p_perm_in_zero_one(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        _, _, _, p_perm = permutation_pvalue(resid, df["e_env"], n_perms=50, seed=0)
        assert 0.0 <= p_perm <= 1.0

    def test_perm_list_length(self):
        df = _make_table(n=30)
        _, _, resid = fit_mass_only(df)
        _, _, perm, _ = permutation_pvalue(resid, df["e_env"], n_perms=77, seed=0)
        assert len(perm) == 77

    def test_strong_signal_low_p_perm(self):
        rng = np.random.default_rng(42)
        env = pd.Series(rng.uniform(0, 1, 50))
        resid = env + rng.normal(0, 0.01, 50)
        _, _, _, p_perm = permutation_pvalue(resid, env, n_perms=500, seed=0)
        assert p_perm < 0.05

    def test_seed_reproducibility(self):
        df = _make_table(n=40)
        _, _, resid = fit_mass_only(df)
        _, _, _, p1 = permutation_pvalue(resid, df["e_env"], n_perms=100, seed=7)
        _, _, _, p2 = permutation_pvalue(resid, df["e_env"], n_perms=100, seed=7)
        assert p1 == p2

    def test_different_seeds_may_differ(self):
        df = _make_table(n=40, seed=99)
        _, _, resid = fit_mass_only(df)
        _, _, p1, _ = permutation_pvalue(resid, df["e_env"], n_perms=200, seed=1)
        _, _, p2, _ = permutation_pvalue(resid, df["e_env"], n_perms=200, seed=2)
        # Not a hard equality requirement; just confirm the call succeeds
        assert p1 is not None and p2 is not None


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def _build(self):
        df = _make_table(n=40, env_effect=0.2)
        model_base, pred, resid = fit_mass_only(df)
        df["pred_mass"] = pred
        df["residual_mass"] = resid
        rho, p_sp, _, p_perm = permutation_pvalue(
            df["residual_mass"], df["e_env"], n_perms=50, seed=0
        )
        model_full = fit_full(df)
        return df, model_base, model_full, rho, p_sp, p_perm

    def test_returns_dict(self):
        args = self._build()
        result = summarize(*args)
        assert isinstance(result, dict)

    def test_key_N(self):
        args = self._build()
        result = summarize(*args)
        assert result["N"] == 40

    def test_key_delta_aic(self):
        args = self._build()
        result = summarize(*args)
        assert "delta_aic" in result

    def test_key_delta_bic(self):
        args = self._build()
        result = summarize(*args)
        assert "delta_bic" in result

    def test_key_delta_r2(self):
        args = self._build()
        result = summarize(*args)
        assert "delta_r2" in result

    def test_delta_r2_non_negative(self):
        args = self._build()
        result = summarize(*args)
        assert result["delta_r2"] >= 0.0

    def test_key_rho_residual_env(self):
        args = self._build()
        result = summarize(*args)
        assert "rho_residual_env" in result

    def test_key_p_spearman(self):
        args = self._build()
        result = summarize(*args)
        assert "p_spearman" in result

    def test_key_p_perm(self):
        args = self._build()
        result = summarize(*args)
        assert "p_perm" in result

    def test_key_coef_env_full(self):
        args = self._build()
        result = summarize(*args)
        assert "coef_env_full" in result

    def test_key_p_env_full(self):
        args = self._build()
        result = summarize(*args)
        assert "p_env_full" in result

    def test_key_r2_base(self):
        args = self._build()
        result = summarize(*args)
        assert "r2_base" in result

    def test_key_r2_full(self):
        args = self._build()
        result = summarize(*args)
        assert "r2_full" in result

    def test_bic_full_less_than_base_with_effect(self):
        df = _make_table(n=100, env_effect=1.5, seed=3)
        model_base, pred, resid = fit_mass_only(df)
        df["pred_mass"] = pred
        df["residual_mass"] = resid
        rho, p_sp, _, p_perm = permutation_pvalue(
            df["residual_mass"], df["e_env"], n_perms=50, seed=0
        )
        model_full = fit_full(df)
        result = summarize(df, model_base, model_full, rho, p_sp, p_perm)
        assert result["delta_bic"] > 0


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------

class TestSaveOutputs:
    def _run(self, tmp_path):
        df = _make_table(n=20)
        model_base, pred, resid = fit_mass_only(df)
        df["pred_mass"] = pred
        df["residual_mass"] = resid
        rho, p_sp, _, p_perm = permutation_pvalue(
            df["residual_mass"], df["e_env"], n_perms=50, seed=0
        )
        model_full = fit_full(df)
        summary = summarize(df, model_base, model_full, rho, p_sp, p_perm)
        save_outputs(df, summary, tmp_path)
        return df, summary

    def test_table_csv_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "env_real_merged_table.csv").exists()

    def test_summary_csv_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "env_real_merged_summary.csv").exists()

    def test_summary_txt_created(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / "env_real_merged_summary.txt").exists()

    def test_table_csv_row_count(self, tmp_path):
        df, _ = self._run(tmp_path)
        loaded = pd.read_csv(tmp_path / "env_real_merged_table.csv")
        assert len(loaded) == len(df)

    def test_summary_csv_has_one_row(self, tmp_path):
        self._run(tmp_path)
        loaded = pd.read_csv(tmp_path / "env_real_merged_summary.csv")
        assert len(loaded) == 1

    def test_summary_txt_has_N_key(self, tmp_path):
        self._run(tmp_path)
        text = (tmp_path / "env_real_merged_summary.txt").read_text()
        assert "N:" in text

    def test_output_dir_created_if_absent(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        df = _make_table(n=10)
        model_base, pred, resid = fit_mass_only(df)
        df["pred_mass"] = pred
        df["residual_mass"] = resid
        rho, p_sp, _, p_perm = permutation_pvalue(
            df["residual_mass"], df["e_env"], n_perms=20, seed=0
        )
        model_full = fit_full(df)
        summary = summarize(df, model_base, model_full, rho, p_sp, p_perm)
        save_outputs(df, summary, subdir)
        assert subdir.exists()


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def _invoke(self, tmp_path, extra_args=None):
        from scripts.analyze_env_real_merged import main

        df = _make_table(n=30)
        input_path = _write_csv(df, tmp_path, "input.csv")
        out_dir = tmp_path / "out"

        argv = [
            "analyze_env_real_merged.py",
            "--input", str(input_path),
            "--out", str(out_dir),
            "--n-perms", "50",
            "--seed", "0",
        ]
        if extra_args:
            argv.extend(extra_args)

        sys.argv = argv
        main()
        return out_dir

    def test_creates_table_csv(self, tmp_path):
        out = self._invoke(tmp_path)
        assert (out / "env_real_merged_table.csv").exists()

    def test_creates_summary_csv(self, tmp_path):
        out = self._invoke(tmp_path)
        assert (out / "env_real_merged_summary.csv").exists()

    def test_creates_summary_txt(self, tmp_path):
        out = self._invoke(tmp_path)
        assert (out / "env_real_merged_summary.txt").exists()

    def test_summary_contains_delta_bic(self, tmp_path):
        out = self._invoke(tmp_path)
        loaded = pd.read_csv(out / "env_real_merged_summary.csv")
        assert "delta_bic" in loaded.columns

    def test_summary_contains_delta_r2(self, tmp_path):
        out = self._invoke(tmp_path)
        loaded = pd.read_csv(out / "env_real_merged_summary.csv")
        assert "delta_r2" in loaded.columns

    def test_works_with_galaxy_col(self, tmp_path):
        from scripts.analyze_env_real_merged import main

        df = _make_table(n=30, col_galaxy="galaxy")
        input_path = _write_csv(df, tmp_path, "galaxy_col.csv")
        out_dir = tmp_path / "out_galaxy"

        sys.argv = [
            "analyze_env_real_merged.py",
            "--input", str(input_path),
            "--out", str(out_dir),
            "--n-perms", "50",
            "--seed", "0",
        ]
        main()
        assert (out_dir / "env_real_merged_table.csv").exists()

    def test_works_with_e_env_err(self, tmp_path):
        from scripts.analyze_env_real_merged import main

        df = _make_table(n=30, include_err=True)
        input_path = _write_csv(df, tmp_path, "with_err.csv")
        out_dir = tmp_path / "out_err"

        sys.argv = [
            "analyze_env_real_merged.py",
            "--input", str(input_path),
            "--out", str(out_dir),
            "--n-perms", "50",
            "--seed", "0",
        ]
        main()
        loaded = pd.read_csv(out_dir / "env_real_merged_table.csv")
        assert "e_env_err" in loaded.columns
