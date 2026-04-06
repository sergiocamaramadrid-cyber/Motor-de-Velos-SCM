"""
tests/test_analyze_env_real_merged.py — Tests for scripts/analyze_env_real_merged.py.

Covers:
  1. load_merged_csv()               — loading and column validation
  2. fit_ols_base()                  — OLS delta_f3 ~ logM (HC3)
  3. fit_ols_full()                  — OLS delta_f3 ~ logM + e_env (HC3)
  4. compute_spearman_permutation()  — Spearman + permutation test
  5. compute_model_comparison()      — ΔAIC / ΔBIC / ΔR²
  6. analyze_env_real_merged()       — full pipeline
  7. save_outputs()                  — file writing
  8. main()                          — CLI and keyword-argument API
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_env_real_merged import (
    DEFAULT_N_PERMS,
    _TABLE_STEM,
    _SUMMARY_CSV_STEM,
    _SUMMARY_TXT_STEM,
    load_merged_csv,
    fit_ols_base,
    fit_ols_full,
    compute_spearman_permutation,
    compute_model_comparison,
    analyze_env_real_merged,
    save_outputs,
    main,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Build a reproducible synthetic merged DataFrame."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(8.5, 11.5, n)
    e_env = rng.uniform(-1.0, 1.0, n)
    noise = rng.normal(0, 0.05, n)
    delta_f3 = 0.1 * (logM - 10.0) + 0.05 * e_env + noise
    return pd.DataFrame({
        "galaxy_name": [f"NGC{1000 + i}" for i in range(n)],
        "logM": logM,
        "delta_f3": delta_f3,
        "e_env": e_env,
    })


def _write_merged_csv(tmp_path: Path, df: pd.DataFrame | None = None) -> Path:
    """Write a merged CSV to tmp_path and return its path."""
    if df is None:
        df = _make_df()
    p = tmp_path / "merged.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. load_merged_csv()
# ---------------------------------------------------------------------------

class TestLoadMergedCsv:
    def test_loads_standard_columns(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        df = load_merged_csv(p)
        assert {"galaxy_name", "logM", "delta_f3", "e_env"}.issubset(df.columns)

    def test_accepts_galaxy_column_alias(self, tmp_path):
        df = _make_df()
        df = df.rename(columns={"galaxy_name": "galaxy"})
        p = tmp_path / "galaxy_col.csv"
        df.to_csv(p, index=False)
        result = load_merged_csv(p)
        assert "galaxy_name" in result.columns

    def test_galaxy_name_takes_precedence_over_galaxy(self, tmp_path):
        df = _make_df()
        df["galaxy"] = "other"
        p = tmp_path / "both.csv"
        df.to_csv(p, index=False)
        result = load_merged_csv(p)
        assert "galaxy_name" in result.columns

    def test_retains_e_env_err_when_present(self, tmp_path):
        df = _make_df()
        df["e_env_err"] = 0.05
        p = _write_merged_csv(tmp_path, df)
        result = load_merged_csv(p)
        assert "e_env_err" in result.columns

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_merged_csv(tmp_path / "missing.csv")

    def test_raises_missing_galaxy_identifier(self, tmp_path):
        df = _make_df().drop(columns=["galaxy_name"])
        p = tmp_path / "no_id.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="galaxy"):
            load_merged_csv(p)

    def test_raises_missing_logM(self, tmp_path):
        df = _make_df().drop(columns=["logM"])
        p = tmp_path / "no_logM.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="logM"):
            load_merged_csv(p)

    def test_raises_missing_delta_f3(self, tmp_path):
        df = _make_df().drop(columns=["delta_f3"])
        p = tmp_path / "no_delta.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="delta_f3"):
            load_merged_csv(p)

    def test_raises_missing_e_env(self, tmp_path):
        df = _make_df().drop(columns=["e_env"])
        p = tmp_path / "no_eenv.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="e_env"):
            load_merged_csv(p)

    def test_row_count_preserved(self, tmp_path):
        df = _make_df(n=30)
        p = _write_merged_csv(tmp_path, df)
        result = load_merged_csv(p)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# 2. fit_ols_base()
# ---------------------------------------------------------------------------

class TestFitOlsBase:
    def test_returns_result_object(self):
        df = _make_df()
        result = fit_ols_base(df)
        assert hasattr(result, "params")

    def test_has_logM_coefficient(self):
        df = _make_df()
        result = fit_ols_base(df)
        assert "logM" in result.params

    def test_has_intercept(self):
        df = _make_df()
        result = fit_ols_base(df)
        assert "Intercept" in result.params

    def test_residuals_length_matches_data(self):
        df = _make_df(n=25)
        result = fit_ols_base(df)
        assert len(result.resid) == 25

    def test_r_squared_in_unit_interval(self):
        df = _make_df()
        result = fit_ols_base(df)
        assert 0.0 <= result.rsquared <= 1.0

    def test_no_e_env_in_base_model(self):
        df = _make_df()
        result = fit_ols_base(df)
        assert "e_env" not in result.params


# ---------------------------------------------------------------------------
# 3. fit_ols_full()
# ---------------------------------------------------------------------------

class TestFitOlsFull:
    def test_returns_result_object(self):
        df = _make_df()
        result = fit_ols_full(df)
        assert hasattr(result, "params")

    def test_has_e_env_coefficient(self):
        df = _make_df()
        result = fit_ols_full(df)
        assert "e_env" in result.params

    def test_has_logM_coefficient(self):
        df = _make_df()
        result = fit_ols_full(df)
        assert "logM" in result.params

    def test_residuals_length_matches_data(self):
        df = _make_df(n=25)
        result = fit_ols_full(df)
        assert len(result.resid) == 25

    def test_r_squared_not_less_than_base(self):
        df = _make_df(seed=7)
        r_base = fit_ols_base(df).rsquared
        r_full = fit_ols_full(df).rsquared
        assert r_full >= r_base - 1e-10


# ---------------------------------------------------------------------------
# 4. compute_spearman_permutation()
# ---------------------------------------------------------------------------

class TestComputeSpearmanPermutation:
    def test_returns_dict_with_required_keys(self):
        rng = np.random.default_rng(0)
        res = rng.normal(size=20)
        env = rng.normal(size=20)
        result = compute_spearman_permutation(res, env, n_perms=100, seed=0)
        assert {"rho", "p", "p_perm"}.issubset(result.keys())

    def test_rho_in_minus_one_to_one(self):
        rng = np.random.default_rng(1)
        res = rng.normal(size=30)
        env = rng.normal(size=30)
        r = compute_spearman_permutation(res, env, n_perms=100, seed=1)
        assert -1.0 <= r["rho"] <= 1.0

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(2)
        res = rng.normal(size=30)
        env = rng.normal(size=30)
        r = compute_spearman_permutation(res, env, n_perms=100, seed=2)
        assert 0.0 <= r["p"] <= 1.0

    def test_p_perm_in_unit_interval(self):
        rng = np.random.default_rng(3)
        res = rng.normal(size=30)
        env = rng.normal(size=30)
        r = compute_spearman_permutation(res, env, n_perms=200, seed=3)
        assert 0.0 <= r["p_perm"] <= 1.0

    def test_perfect_correlation_gives_small_p_perm(self):
        x = np.arange(50, dtype=float)
        r = compute_spearman_permutation(x, x, n_perms=500, seed=42)
        assert r["rho"] == pytest.approx(1.0, abs=1e-9)
        assert r["p_perm"] <= 0.05

    def test_reproducible_with_same_seed(self):
        rng = np.random.default_rng(5)
        res = rng.normal(size=25)
        env = rng.normal(size=25)
        r1 = compute_spearman_permutation(res, env, n_perms=200, seed=99)
        r2 = compute_spearman_permutation(res, env, n_perms=200, seed=99)
        assert r1["p_perm"] == r2["p_perm"]

    def test_different_seeds_may_differ(self):
        rng = np.random.default_rng(6)
        res = rng.normal(size=25)
        env = rng.normal(size=25)
        r1 = compute_spearman_permutation(res, env, n_perms=200, seed=1)
        r2 = compute_spearman_permutation(res, env, n_perms=200, seed=2)
        # rho is deterministic; p_perm may differ
        assert r1["rho"] == pytest.approx(r2["rho"])


# ---------------------------------------------------------------------------
# 5. compute_model_comparison()
# ---------------------------------------------------------------------------

class TestComputeModelComparison:
    def _fitted_pair(self, seed: int = 0):
        df = _make_df(n=30, seed=seed)
        return fit_ols_base(df), fit_ols_full(df)

    def test_returns_required_keys(self):
        base, full = self._fitted_pair()
        result = compute_model_comparison(base, full)
        assert {"delta_aic", "delta_bic", "delta_r2", "coef_env", "p_env"}.issubset(
            result.keys()
        )

    def test_delta_r2_is_float(self):
        base, full = self._fitted_pair()
        result = compute_model_comparison(base, full)
        assert isinstance(result["delta_r2"], float)

    def test_delta_r2_nonnegative(self):
        base, full = self._fitted_pair(seed=1)
        result = compute_model_comparison(base, full)
        assert result["delta_r2"] >= -1e-10

    def test_coef_env_is_finite(self):
        base, full = self._fitted_pair()
        result = compute_model_comparison(base, full)
        assert math.isfinite(result["coef_env"])

    def test_p_env_in_unit_interval(self):
        base, full = self._fitted_pair()
        result = compute_model_comparison(base, full)
        assert 0.0 <= result["p_env"] <= 1.0


# ---------------------------------------------------------------------------
# 6. analyze_env_real_merged()
# ---------------------------------------------------------------------------

class TestAnalyzeEnvRealMerged:
    def test_returns_dict(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        required = {
            "N", "rho", "p", "p_perm",
            "delta_aic", "delta_bic", "delta_r2",
            "coef_env", "p_env", "df_table",
        }
        assert required.issubset(result.keys())

    def test_N_matches_dataframe_length(self):
        df = _make_df(n=15)
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert result["N"] == 15

    def test_df_table_has_residual_base_column(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert "residual_base" in result["df_table"].columns

    def test_df_table_row_count_matches_N(self):
        df = _make_df(n=18)
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert len(result["df_table"]) == result["N"]

    def test_df_table_has_e_env_err_when_present(self):
        df = _make_df()
        df["e_env_err"] = 0.05
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert "e_env_err" in result["df_table"].columns

    def test_df_table_no_e_env_err_when_absent(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert "e_env_err" not in result["df_table"].columns

    def test_rho_in_range(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert -1.0 <= result["rho"] <= 1.0

    def test_p_in_unit_interval(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert 0.0 <= result["p"] <= 1.0

    def test_p_perm_in_unit_interval(self):
        df = _make_df()
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert 0.0 <= result["p_perm"] <= 1.0

    def test_nan_rows_dropped(self):
        df = _make_df(n=20)
        df.loc[0, "e_env"] = float("nan")
        result = analyze_env_real_merged(df, n_perms=50, seed=0)
        assert result["N"] == 19

    def test_reproducible_with_seed(self):
        df = _make_df()
        r1 = analyze_env_real_merged(df, n_perms=100, seed=7)
        r2 = analyze_env_real_merged(df, n_perms=100, seed=7)
        assert r1["p_perm"] == r2["p_perm"]


# ---------------------------------------------------------------------------
# 7. save_outputs()
# ---------------------------------------------------------------------------

class TestSaveOutputs:
    def _run(self, tmp_path: Path) -> dict:
        df = _make_df()
        stats = analyze_env_real_merged(df, n_perms=50, seed=0)
        save_outputs(stats, tmp_path, "test_input.csv")
        return stats

    def test_table_csv_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / _TABLE_STEM).exists()

    def test_summary_csv_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / _SUMMARY_CSV_STEM).exists()

    def test_summary_txt_written(self, tmp_path):
        self._run(tmp_path)
        assert (tmp_path / _SUMMARY_TXT_STEM).exists()

    def test_table_csv_readable(self, tmp_path):
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / _TABLE_STEM)
        assert "residual_base" in df.columns

    def test_summary_csv_has_one_row(self, tmp_path):
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / _SUMMARY_CSV_STEM)
        assert len(df) == 1

    def test_summary_csv_has_N_column(self, tmp_path):
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / _SUMMARY_CSV_STEM)
        assert "N" in df.columns

    def test_summary_txt_contains_spearman(self, tmp_path):
        self._run(tmp_path)
        text = (tmp_path / _SUMMARY_TXT_STEM).read_text()
        assert "Spearman" in text or "rho" in text.lower() or "ρ" in text

    def test_creates_out_dir(self, tmp_path):
        df = _make_df()
        stats = analyze_env_real_merged(df, n_perms=50, seed=0)
        nested = tmp_path / "deep" / "nested"
        save_outputs(stats, nested, "test.csv")
        assert nested.is_dir()


# ---------------------------------------------------------------------------
# 8. main() — CLI and keyword-argument API
# ---------------------------------------------------------------------------

class TestMain:
    def test_keyword_args_run_without_argv(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        result = main(
            input_path=str(p),
            out_dir=str(tmp_path / "out"),
            n_perms=50,
            seed=0,
        )
        assert isinstance(result, dict)
        assert "N" in result

    def test_returns_dict_without_df_table(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        result = main(input_path=str(p), n_perms=50, seed=0)
        assert "df_table" not in result

    def test_argv_list_works(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        result = main([
            "--input", str(p),
            "--n-perms", "50",
            "--seed", "0",
        ])
        assert isinstance(result, dict)

    def test_keyword_overrides_argv_input(self, tmp_path):
        p1 = _write_merged_csv(tmp_path)
        p2 = tmp_path / "p2.csv"
        _make_df(n=5).to_csv(p2, index=False)
        result = main(
            ["--input", str(p1)],
            input_path=str(p2),
            n_perms=50,
            seed=0,
        )
        assert result["N"] == 5

    def test_missing_input_raises(self):
        with pytest.raises(ValueError, match="input"):
            main(n_perms=50)

    def test_no_out_dir_skips_file_writing(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        result = main(input_path=str(p), n_perms=50, seed=0)
        assert not (tmp_path / _TABLE_STEM).exists()
        assert isinstance(result, dict)

    def test_out_dir_created_and_files_written(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        out = tmp_path / "results"
        main(input_path=str(p), out_dir=str(out), n_perms=50, seed=0)
        assert (out / _TABLE_STEM).exists()
        assert (out / _SUMMARY_CSV_STEM).exists()
        assert (out / _SUMMARY_TXT_STEM).exists()

    def test_n_perms_kwarg_overrides_argv(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        # Just check it runs without error and honours the kwarg
        result = main(
            ["--input", str(p), "--n-perms", "10"],
            n_perms=20,
            seed=1,
        )
        assert isinstance(result, dict)

    def test_seed_kwarg_overrides_argv(self, tmp_path):
        p = _write_merged_csv(tmp_path)
        r1 = main(["--input", str(p), "--n-perms", "100"], seed=42)
        r2 = main(["--input", str(p), "--n-perms", "100"], seed=42)
        assert r1["p_perm"] == r2["p_perm"]
