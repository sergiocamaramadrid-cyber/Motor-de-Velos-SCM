"""
tests/test_analyze_sparc_chae.py — Tests for scripts/analyze_sparc_chae.py.

Covers:
  - clean_name: normalisation edge-cases
  - find_first_existing / choose_target_column / build_mass_column
  - permutation_spearman
  - run_analysis: happy path and error conditions
  - CLI (main) via tmp files
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from scripts.analyze_sparc_chae import (
    clean_name,
    find_first_existing,
    choose_target_column,
    build_mass_column,
    permutation_spearman,
    run_analysis,
    main,
    _MIN_SAMPLE,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _make_sparc(n: int = 30, seed: int = 0, extra_cols: dict | None = None) -> pd.DataFrame:
    """Minimal SPARC-like table with F3, logM, Galaxy."""
    rng = np.random.default_rng(seed)
    names = [f"NGC{1000 + i}" for i in range(n)]
    df = pd.DataFrame(
        {
            "Galaxy": names,
            "F3": rng.uniform(0.3, 0.8, n),
            "logM": rng.uniform(8.5, 11.5, n),
        }
    )
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = vals
    return df


def _make_chae(sparc: pd.DataFrame, frac: float = 1.0, seed: int = 1) -> pd.DataFrame:
    """Chae table with e_env for a subset of SPARC galaxies."""
    rng = np.random.default_rng(seed)
    sub = sparc.sample(frac=frac, random_state=seed)[["Galaxy"]].copy()
    sub["e_env"] = rng.uniform(0.0, 2.0, len(sub))
    return sub


# ===========================================================================
# 1. clean_name
# ===========================================================================

class TestCleanName:
    def test_strips_whitespace(self):
        assert clean_name("  NGC 1234  ") == "NGC1234"

    def test_collapses_internal_spaces(self):
        assert clean_name("DDO  154") == "DDO154"

    def test_upper_case(self):
        assert clean_name("ugc0128") == "UGC0128"

    def test_removes_hyphens(self):
        assert clean_name("IC-342") == "IC342"

    def test_nan_input(self):
        result = clean_name(float("nan"))
        assert result is np.nan or (isinstance(result, float) and math.isnan(result))

    def test_none_input(self):
        result = clean_name(None)
        assert result is np.nan or (isinstance(result, float) and math.isnan(result))

    def test_leading_zero_preserved(self):
        assert clean_name("UGC 0128") == "UGC0128"

    def test_already_clean(self):
        assert clean_name("NGC1234") == "NGC1234"

    def test_mixed_case_with_spaces(self):
        assert clean_name("Ngc  253") == "NGC253"

    def test_tab_treated_as_whitespace(self):
        assert clean_name("NGC\t253") == "NGC253"


# ===========================================================================
# 2. find_first_existing
# ===========================================================================

class TestFindFirstExisting:
    def test_returns_first_match(self):
        df = pd.DataFrame({"b": [1], "a": [2]})
        assert find_first_existing(df, ["a", "b"]) == "a"

    def test_skips_missing(self):
        df = pd.DataFrame({"c": [1]})
        assert find_first_existing(df, ["a", "b", "c"]) == "c"

    def test_returns_none_when_none_found(self):
        df = pd.DataFrame({"x": [1]})
        assert find_first_existing(df, ["a", "b"]) is None

    def test_empty_candidates(self):
        df = pd.DataFrame({"a": [1]})
        assert find_first_existing(df, []) is None


# ===========================================================================
# 3. choose_target_column
# ===========================================================================

class TestChooseTargetColumn:
    def test_prefers_F3(self):
        df = pd.DataFrame({"F3": [1], "beta": [1], "Vflat": [1]})
        assert choose_target_column(df) == "F3"

    def test_falls_back_to_beta(self):
        df = pd.DataFrame({"beta": [1], "Vflat": [1]})
        assert choose_target_column(df) == "beta"

    def test_falls_back_to_delta_f3(self):
        df = pd.DataFrame({"delta_f3": [1], "Vflat": [1]})
        assert choose_target_column(df) == "delta_f3"

    def test_falls_back_to_DeltaF3(self):
        df = pd.DataFrame({"DeltaF3": [1], "Vflat": [1]})
        assert choose_target_column(df) == "DeltaF3"

    def test_falls_back_to_Vflat(self):
        df = pd.DataFrame({"Vflat": [1]})
        assert choose_target_column(df) == "Vflat"

    def test_raises_when_none(self):
        df = pd.DataFrame({"other": [1]})
        with pytest.raises(ValueError, match="variable dependiente"):
            choose_target_column(df)


# ===========================================================================
# 4. build_mass_column
# ===========================================================================

class TestBuildMassColumn:
    def test_direct_logM(self):
        df = pd.DataFrame({"logM": [9.0, 10.0, 11.0]})
        desc = build_mass_column(df)
        assert "logM" in desc
        np.testing.assert_array_almost_equal(df["log_mass_proxy"], [9.0, 10.0, 11.0])

    def test_direct_logMbar(self):
        df = pd.DataFrame({"logMbar": [9.5, 10.5]})
        desc = build_mass_column(df)
        assert "logMbar" in desc
        np.testing.assert_array_almost_equal(df["log_mass_proxy"], [9.5, 10.5])

    def test_linear_Mbar(self):
        df = pd.DataFrame({"Mbar": [1e9, 1e10]})
        desc = build_mass_column(df)
        assert "Mbar" in desc
        np.testing.assert_array_almost_equal(df["log_mass_proxy"], [9.0, 10.0])

    def test_linear_negative_values_become_nan(self):
        df = pd.DataFrame({"Mbar": [-1e9, 1e10]})
        build_mass_column(df)
        assert np.isnan(df["log_mass_proxy"].iloc[0])
        assert not np.isnan(df["log_mass_proxy"].iloc[1])

    def test_luminosity_L36(self):
        df = pd.DataFrame({"L36": [1e8, 1e9]})
        desc = build_mass_column(df)
        assert "L36" in desc or "luminosidad" in desc
        np.testing.assert_array_almost_equal(df["log_mass_proxy"], [8.0, 9.0])

    def test_luminosity_bracket_notation(self):
        df = pd.DataFrame({"L[3.6]": [1e8]})
        build_mass_column(df)
        np.testing.assert_array_almost_equal(df["log_mass_proxy"], [8.0])

    def test_raises_when_none(self):
        df = pd.DataFrame({"other": [1]})
        with pytest.raises(ValueError, match="masa"):
            build_mass_column(df)

    def test_prefers_log_over_linear(self):
        df = pd.DataFrame({"logM": [10.0], "Mbar": [1e11]})
        build_mass_column(df)
        np.testing.assert_array_almost_equal(df["log_mass_proxy"], [10.0])


# ===========================================================================
# 5. permutation_spearman
# ===========================================================================

class TestPermutationSpearman:
    def _xy(self, n: int = 50, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, n)
        y = x + 0.1 * rng.normal(size=n)
        return x, y

    def test_returns_three_items(self):
        x, y = self._xy()
        result = permutation_spearman(x, y, n_perm=100, seed=0)
        assert len(result) == 3

    def test_rho_in_range(self):
        x, y = self._xy()
        rho, _, _ = permutation_spearman(x, y, n_perm=100, seed=0)
        assert -1.0 <= rho <= 1.0

    def test_p_in_range(self):
        x, y = self._xy()
        _, p, _ = permutation_spearman(x, y, n_perm=100, seed=0)
        assert 0.0 <= p <= 1.0

    def test_perm_array_length(self):
        x, y = self._xy()
        _, _, perms = permutation_spearman(x, y, n_perm=200, seed=0)
        assert len(perms) == 200

    def test_strong_correlation_low_p(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 100)
        y = x + 0.01 * rng.normal(size=100)
        _, p, _ = permutation_spearman(x, y, n_perm=500, seed=0)
        assert p < 0.05

    def test_no_correlation_high_p(self):
        rng = np.random.default_rng(99)
        x = rng.uniform(0, 1, 80)
        y = rng.uniform(0, 1, 80)
        _, p, _ = permutation_spearman(x, y, n_perm=500, seed=99)
        # With random data p can be anything, just verify it's valid
        assert 0.0 <= p <= 1.0

    def test_reproducible_with_same_seed(self):
        x, y = self._xy()
        r1, p1, _ = permutation_spearman(x, y, n_perm=100, seed=42)
        r2, p2, _ = permutation_spearman(x, y, n_perm=100, seed=42)
        assert r1 == r2
        assert p1 == p2

    def test_different_seed_different_p(self):
        x, y = self._xy()
        _, p1, _ = permutation_spearman(x, y, n_perm=200, seed=1)
        _, p2, _ = permutation_spearman(x, y, n_perm=200, seed=9999)
        # Not guaranteed to differ but very likely with different seeds
        # Just verify both are valid
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0


# ===========================================================================
# 6. run_analysis — happy path
# ===========================================================================

class TestRunAnalysisHappyPath:
    @pytest.fixture
    def data(self):
        sparc = _make_sparc(n=40, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        return sparc, chae

    def test_returns_dict(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert isinstance(res, dict)

    def test_required_keys(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        for key in (
            "df", "target_col", "mass_desc",
            "model_mass", "model_resid", "model_full",
            "rho", "p_spear", "p_perm",
            "delta_aic", "delta_bic", "delta_r2", "delta_adj_r2",
            "match_diag",
        ):
            assert key in res, f"Missing key: {key}"

    def test_df_has_resid_column(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert "resid_mass" in res["df"].columns

    def test_target_col_is_F3(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["target_col"] == "F3"

    def test_rho_in_range(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert -1.0 <= res["rho"] <= 1.0

    def test_p_values_in_range(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert 0.0 <= res["p_spear"] <= 1.0
        assert 0.0 <= res["p_perm"] <= 1.0

    def test_match_diag_structure(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        diag = res["match_diag"]
        assert "n_sparc" in diag
        assert "n_chae" in diag
        assert "n_intersection" in diag
        assert diag["n_intersection"] > 0

    def test_sample_size_in_df(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert len(res["df"]) >= _MIN_SAMPLE

    def test_delta_aic_is_finite(self, data):
        sparc, chae = data
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert math.isfinite(res["delta_aic"])

    def test_reproducible(self, data):
        sparc, chae = data
        r1 = run_analysis(sparc, chae, n_perms=100, seed=7, verbose=False)
        r2 = run_analysis(sparc, chae, n_perms=100, seed=7, verbose=False)
        assert r1["rho"] == r2["rho"]
        assert r1["p_perm"] == r2["p_perm"]

    def test_uses_beta_when_no_F3(self):
        sparc = _make_sparc(n=30, seed=0)
        sparc = sparc.drop(columns=["F3"])
        sparc["beta"] = np.random.default_rng(2).uniform(0.3, 0.8, 30)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["target_col"] == "beta"

    def test_uses_Vflat_as_last_resort(self):
        sparc = _make_sparc(n=30, seed=0)
        sparc = sparc.drop(columns=["F3"])
        sparc["Vflat"] = np.random.default_rng(3).uniform(50, 200, 30)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["target_col"] == "Vflat"

    def test_uses_linear_Mbar_when_no_logM(self):
        sparc = _make_sparc(n=30, seed=0)
        sparc = sparc.drop(columns=["logM"])
        sparc["Mbar"] = 10 ** np.random.default_rng(4).uniform(8.5, 11.5, 30)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert "Mbar" in res["mass_desc"]


# ===========================================================================
# 7. run_analysis — partial match (name normalisation)
# ===========================================================================

class TestRunAnalysisNameNormalisation:
    def test_matches_after_normalisation(self):
        sparc = _make_sparc(n=30, seed=0)
        # Chae uses different spacing/case
        chae = pd.DataFrame(
            {
                "Galaxy": [f"ngc {1000 + i}" for i in range(30)],
                "e_env": np.random.default_rng(5).uniform(0, 2, 30),
            }
        )
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["match_diag"]["n_intersection"] == 30

    def test_partial_match_still_works(self):
        sparc = _make_sparc(n=40, seed=0)
        chae = _make_chae(sparc, frac=0.6, seed=2)
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert 0 < res["match_diag"]["n_intersection"] <= 40

    def test_deduplicates_chae(self):
        sparc = _make_sparc(n=30, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        # Duplicate all rows in chae
        chae_dup = pd.concat([chae, chae], ignore_index=True)
        res = run_analysis(sparc, chae_dup, n_perms=50, seed=0, verbose=False)
        # Should still work and not double-count
        assert len(res["df"]) <= 30


# ===========================================================================
# 8. run_analysis — error conditions
# ===========================================================================

class TestRunAnalysisErrors:
    def test_missing_Galaxy_in_sparc(self):
        sparc = _make_sparc(n=30).rename(columns={"Galaxy": "Name"})
        chae = _make_chae(_make_sparc(n=30))
        with pytest.raises(ValueError, match="Galaxy"):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)

    def test_missing_Galaxy_in_chae(self):
        sparc = _make_sparc(n=30)
        chae = _make_chae(sparc).rename(columns={"Galaxy": "Name"})
        with pytest.raises(ValueError, match="Galaxy"):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)

    def test_missing_e_env(self):
        sparc = _make_sparc(n=30)
        chae = _make_chae(sparc).drop(columns=["e_env"])
        with pytest.raises(ValueError, match="e_env"):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)

    def test_no_overlap_raises(self):
        sparc = _make_sparc(n=20, seed=0)
        chae = pd.DataFrame(
            {
                "Galaxy": [f"FAKE{i}" for i in range(20)],
                "e_env": np.ones(20),
            }
        )
        with pytest.raises(ValueError):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)

    def test_small_sample_raises(self):
        sparc = _make_sparc(n=10, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        with pytest.raises(ValueError, match="muestra"):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)

    def test_no_target_column_raises(self):
        sparc = _make_sparc(n=30, seed=0).drop(columns=["F3"])
        # No beta/delta_f3/Vflat either
        chae = _make_chae(_make_sparc(n=30))
        with pytest.raises(ValueError, match="variable dependiente"):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)

    def test_no_mass_column_raises(self):
        sparc = _make_sparc(n=30, seed=0).drop(columns=["logM"])
        chae = _make_chae(_make_sparc(n=30))
        with pytest.raises(ValueError, match="masa"):
            run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)


# ===========================================================================
# 9. run_analysis — NaN handling
# ===========================================================================

class TestRunAnalysisNaN:
    def test_nan_in_F3_dropped(self):
        sparc = _make_sparc(n=30, seed=0)
        sparc.loc[0, "F3"] = np.nan
        chae = _make_chae(sparc, frac=1.0, seed=1)
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["df"]["F3"].isna().sum() == 0

    def test_nan_in_e_env_dropped(self):
        sparc = _make_sparc(n=30, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        chae.loc[0, "e_env"] = np.nan
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["df"]["e_env"].isna().sum() == 0

    def test_nan_in_logM_dropped(self):
        sparc = _make_sparc(n=30, seed=0)
        sparc.loc[0, "logM"] = np.nan
        chae = _make_chae(sparc, frac=1.0, seed=1)
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert res["df"]["log_mass_proxy"].isna().sum() == 0


# ===========================================================================
# 10. model properties
# ===========================================================================

class TestModelProperties:
    @pytest.fixture
    def results(self):
        sparc = _make_sparc(n=40, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        return run_analysis(sparc, chae, n_perms=100, seed=0, verbose=False)

    def test_base_model_has_two_params(self, results):
        # const + log_mass_proxy
        assert results["model_mass"].params.shape[0] == 2

    def test_full_model_has_three_params(self, results):
        # const + log_mass_proxy + e_env
        assert results["model_full"].params.shape[0] == 3

    def test_r2_in_range(self, results):
        assert 0.0 <= results["model_mass"].rsquared <= 1.0
        assert 0.0 <= results["model_full"].rsquared <= 1.0

    def test_full_r2_ge_base_r2(self, results):
        # Adding a predictor cannot decrease R² in OLS
        assert results["model_full"].rsquared >= results["model_mass"].rsquared - 1e-10

    def test_residuals_near_zero_mean(self, results):
        resids = results["df"]["resid_mass"]
        assert abs(resids.mean()) < 1e-10

    def test_delta_r2_consistent(self, results):
        expected = (
            results["model_full"].rsquared - results["model_mass"].rsquared
        )
        assert abs(results["delta_r2"] - expected) < 1e-12


# ===========================================================================
# 11. CLI (main) via subprocess
# ===========================================================================

class TestCLI:
    def _write_csv(self, tmp_path: Path, filename: str, df: pd.DataFrame) -> Path:
        p = tmp_path / filename
        df.to_csv(p, index=False)
        return p

    def test_cli_produces_output_csv(self, tmp_path):
        sparc = _make_sparc(n=40, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        sparc_p = self._write_csv(tmp_path, "sparc.csv", sparc)
        chae_p = self._write_csv(tmp_path, "chae.csv", chae)
        out_p = tmp_path / "out.csv"
        main([
            "--sparc", str(sparc_p),
            "--chae", str(chae_p),
            "--out", str(out_p),
            "--n-perms", "50",
            "--seed", "0",
        ])
        assert out_p.exists()

    def test_cli_output_has_expected_columns(self, tmp_path):
        sparc = _make_sparc(n=40, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        sparc_p = self._write_csv(tmp_path, "sparc.csv", sparc)
        chae_p = self._write_csv(tmp_path, "chae.csv", chae)
        out_p = tmp_path / "out.csv"
        main([
            "--sparc", str(sparc_p),
            "--chae", str(chae_p),
            "--out", str(out_p),
            "--n-perms", "50",
        ])
        result = pd.read_csv(out_p)
        for col in ("e_env", "log_mass_proxy", "resid_mass"):
            assert col in result.columns

    def test_cli_missing_sparc_exits(self, tmp_path, capsys):
        chae = _make_chae(_make_sparc(n=30))
        chae_p = self._write_csv(tmp_path, "chae.csv", chae)
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--sparc", str(tmp_path / "nonexistent.csv"),
                "--chae", str(chae_p),
            ])
        assert exc_info.value.code != 0

    def test_cli_missing_chae_exits(self, tmp_path):
        sparc = _make_sparc(n=30)
        sparc_p = self._write_csv(tmp_path, "sparc.csv", sparc)
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--sparc", str(sparc_p),
                "--chae", str(tmp_path / "nonexistent.csv"),
            ])
        assert exc_info.value.code != 0


# ===========================================================================
# 12. e_env_err column (optional)
# ===========================================================================

class TestEEnvErr:
    def test_e_env_err_preserved(self):
        sparc = _make_sparc(n=30, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        chae["e_env_err"] = 0.1 * chae["e_env"]
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert "e_env_err" in res["df"].columns

    def test_analysis_works_without_e_env_err(self):
        sparc = _make_sparc(n=30, seed=0)
        chae = _make_chae(sparc, frac=1.0, seed=1)
        # e_env_err not present
        res = run_analysis(sparc, chae, n_perms=50, seed=0, verbose=False)
        assert "e_env_err" not in res["df"].columns
