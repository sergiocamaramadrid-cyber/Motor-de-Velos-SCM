"""tests/test_lt_environmental_test.py — Tests for run_lt_environmental_test.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_lt_environmental_test import (
    _compute_beta,
    _compute_f3_residual,
    load_extreme_cases,
    load_lt_dataset,
    run_extreme_cases_analysis,
    run_standard_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LT_COLS = ["galaxy_id", "logM", "logVobs", "log_gbar", "log_j"]


def _make_lt_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    galaxy_ids = [f"GAL{i:03d}" for i in range(n)]
    return pd.DataFrame(
        {
            "galaxy_id": galaxy_ids,
            "logM": rng.uniform(6.0, 9.0, n),
            "logVobs": rng.uniform(1.0, 2.2, n),
            "log_gbar": rng.uniform(-13.0, -11.0, n),
            "log_j": rng.uniform(0.5, 2.5, n),
        }
    )


def _make_yang_df(galaxy_ids: list[str], yang_ids: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(galaxy_ids)
    return pd.DataFrame(
        {
            "yang_id": yang_ids[:n],
            "galaxy": galaxy_ids,
            "delta_mass_yang": rng.uniform(-1.0, 2.0, n),
        }
    )


# ---------------------------------------------------------------------------
# load_extreme_cases
# ---------------------------------------------------------------------------


class TestLoadExtremeCases:
    def test_returns_dataframe(self):
        df = load_extreme_cases()
        assert isinstance(df, pd.DataFrame)

    def test_exactly_25_rows(self):
        df = load_extreme_cases()
        assert len(df) == 25

    def test_required_columns(self):
        df = load_extreme_cases()
        for col in ("yang_id", "tipo", "logMh", "N_members"):
            assert col in df.columns, f"Missing column: {col}"

    def test_five_categories(self):
        df = load_extreme_cases()
        cats = set(df["tipo"])
        assert cats == {"rico", "aislada", "masivo", "ligero", "fusion"}

    def test_five_per_category(self):
        df = load_extreme_cases()
        for cat in ("rico", "aislada", "masivo", "ligero", "fusion"):
            assert (df["tipo"] == cat).sum() == 5, f"Expected 5 rows for '{cat}'"

    def test_logMh_plausible(self):
        df = load_extreme_cases()
        assert df["logMh"].between(10.0, 16.0).all()

    def test_N_members_positive(self):
        df = load_extreme_cases()
        assert (df["N_members"] >= 1).all()

    def test_isolated_and_light_have_n1(self):
        df = load_extreme_cases()
        for cat in ("aislada", "ligero"):
            assert (df[df["tipo"] == cat]["N_members"] == 1).all()

    def test_rich_groups_high_nmembers(self):
        df = load_extreme_cases()
        assert (df[df["tipo"] == "rico"]["N_members"] > 50).all()


# ---------------------------------------------------------------------------
# _compute_beta
# ---------------------------------------------------------------------------


class TestComputeBeta:
    def test_returns_array(self):
        df = _make_lt_df()
        b = _compute_beta(df)
        assert isinstance(b, np.ndarray)
        assert len(b) == len(df)

    def test_finite_values(self):
        df = _make_lt_df()
        b = _compute_beta(df)
        assert np.all(np.isfinite(b))

    def test_zero_residual_self_consistent(self):
        # If logVobs == predicted, beta == 0
        KPC_TO_M = 3.085677581e19
        KMS_TO_MS = 1.0e3
        A0 = 1.2e-10
        C = np.log10(A0) + 2.0 * np.log10(KPC_TO_M * KMS_TO_MS) - 18.0
        log_gbar = np.array([-12.0])
        log_j = np.array([1.5])
        logVobs_pred = (log_gbar + 2.0 * log_j + C) / 6.0
        df = pd.DataFrame(
            {"logVobs": logVobs_pred, "log_gbar": log_gbar, "log_j": log_j}
        )
        b = _compute_beta(df)
        np.testing.assert_allclose(b, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _compute_f3_residual
# ---------------------------------------------------------------------------


class TestComputeF3Residual:
    def test_returns_array(self):
        df = _make_lt_df()
        r = _compute_f3_residual(df)
        assert isinstance(r, np.ndarray)
        assert len(r) == len(df)

    def test_finite_values(self):
        df = _make_lt_df()
        r = _compute_f3_residual(df)
        assert np.all(np.isfinite(r))

    def test_formula(self):
        df = pd.DataFrame(
            {
                "logVobs": [1.8],
                "log_gbar": [-11.5],
                "log_j": [2.0],
            }
        )
        expected = 3.0 * 1.8 - 2.0 - (-11.5)
        r = _compute_f3_residual(df)
        np.testing.assert_allclose(r[0], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# load_lt_dataset
# ---------------------------------------------------------------------------


class TestLoadLtDataset:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lt_dataset(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_lt_dataset(csv)

    def test_valid_file_returns_dataframe(self, tmp_path):
        csv = tmp_path / "lt.csv"
        df = _make_lt_df(5)
        df.to_csv(csv, index=False)
        loaded = load_lt_dataset(csv)
        assert len(loaded) == 5
        for col in LT_COLS:
            assert col in loaded.columns


# ---------------------------------------------------------------------------
# run_standard_analysis
# ---------------------------------------------------------------------------


class TestRunStandardAnalysis:
    def test_creates_output_csv(self, tmp_path):
        lt_df = _make_lt_df(8)
        run_standard_analysis(lt_df, tmp_path)
        assert (tmp_path / "lt_environmental_results.csv").exists()

    def test_output_columns(self, tmp_path):
        lt_df = _make_lt_df(8)
        result = run_standard_analysis(lt_df, tmp_path)
        for col in ("galaxy", "beta", "F3_residual"):
            assert col in result.columns

    def test_output_row_count(self, tmp_path):
        n = 12
        lt_df = _make_lt_df(n)
        result = run_standard_analysis(lt_df, tmp_path)
        assert len(result) == n

    def test_galaxy_id_alias(self, tmp_path):
        # Input with 'galaxy_id' column → output uses 'galaxy'
        lt_df = _make_lt_df(5)
        assert "galaxy_id" in lt_df.columns
        result = run_standard_analysis(lt_df, tmp_path)
        assert "galaxy" in result.columns

    def test_creates_output_directory(self, tmp_path):
        new_dir = tmp_path / "subdir" / "output"
        lt_df = _make_lt_df(4)
        run_standard_analysis(lt_df, new_dir)
        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# run_extreme_cases_analysis
# ---------------------------------------------------------------------------


def _make_matched_dfs():
    """Return (lt_df, yang_df) that share 5 galaxy IDs from the extreme catalogue."""
    extreme = load_extreme_cases()
    # Take 5 yang_ids from the extreme catalogue
    yang_ids_sample = extreme["yang_id"].iloc[:5].tolist()
    galaxy_names = [f"ExGal{i}" for i in range(5)]

    lt_df = pd.DataFrame(
        {
            "galaxy_id": galaxy_names,
            "logM": [7.5] * 5,
            "logVobs": [1.5, 1.6, 1.7, 1.4, 1.8],
            "log_gbar": [-12.0, -11.8, -11.5, -12.2, -11.3],
            "log_j": [1.5, 1.6, 1.7, 1.4, 1.8],
        }
    )
    yang_df = pd.DataFrame(
        {
            "yang_id": yang_ids_sample,
            "galaxy": galaxy_names,
            "delta_mass_yang": [0.5, -0.3, 1.2, -0.8, 0.9],
        }
    )
    return lt_df, yang_df


class TestRunExtremeCasesAnalysis:
    def test_creates_csv(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        assert (tmp_path / "extreme_cases_analysis.csv").exists()

    def test_creates_figure(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        assert (tmp_path / "extreme_cases_scatter.png").exists()

    def test_creates_summary_text(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        assert (tmp_path / "extreme_cases_summary.txt").exists()

    def test_result_columns(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        result = run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        for col in ("galaxy", "tipo", "beta", "F3_residual", "delta_mass_yang"):
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_matches_overlap(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        result = run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        # 5 galaxies in both lt_df and yang_df, all matching extreme catalogue
        assert len(result) == 5

    def test_no_overlap_raises(self, tmp_path):
        lt_df = _make_lt_df(5)
        extreme = load_extreme_cases()
        yang_df = pd.DataFrame(
            {
                "yang_id": extreme["yang_id"].iloc[:5].tolist(),
                "galaxy": ["NOMATCH_A", "NOMATCH_B", "NOMATCH_C", "NOMATCH_D", "NOMATCH_E"],
                "delta_mass_yang": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        with pytest.raises(ValueError, match="No galaxy overlap"):
            run_extreme_cases_analysis(lt_df, yang_df, tmp_path)

    def test_summary_text_contains_spearman(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        text = (tmp_path / "extreme_cases_summary.txt").read_text(encoding="utf-8")
        assert "Spearman" in text
        assert "ρ" in text

    def test_csv_round_trip(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        result = run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        loaded = pd.read_csv(tmp_path / "extreme_cases_analysis.csv")
        assert list(loaded.columns) == list(result.columns)
        assert len(loaded) == len(result)

    def test_creates_output_directory(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested"
        lt_df, yang_df = _make_matched_dfs()
        run_extreme_cases_analysis(lt_df, yang_df, new_dir)
        assert new_dir.is_dir()

    def test_yang_df_with_galaxy_id_column(self, tmp_path):
        # yang_df may use 'galaxy_id' instead of 'galaxy'
        lt_df, yang_df = _make_matched_dfs()
        yang_df = yang_df.rename(columns={"galaxy": "galaxy_id"})
        result = run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        assert len(result) == 5

    def test_beta_finite(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        result = run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        assert result["beta"].notna().all()

    def test_f3_residual_finite(self, tmp_path):
        lt_df, yang_df = _make_matched_dfs()
        result = run_extreme_cases_analysis(lt_df, yang_df, tmp_path)
        assert result["F3_residual"].notna().all()
