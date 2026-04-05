"""tests/test_scm_extreme_25_test.py — Tests for scm_extreme_25_test.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scm_extreme_25_test import (
    compute_stats,
    load_data,
    save_outputs,
    select_25_extremes,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_f3_csv(tmp_path: Path, n: int = 40, col: str = "friction_slope") -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "galaxy": [f"G{i:03d}" for i in range(n)],
            col: rng.uniform(0.2, 1.0, n),
            "log_M_bar": rng.uniform(8.0, 11.0, n),
            "log_Rmax": rng.uniform(0.5, 2.0, n),
        }
    )
    p = tmp_path / "f3.csv"
    df.to_csv(p, index=False)
    return p


def _make_env_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "galaxy": [f"G{i:03d}" for i in range(n)],
            "delta_mass_std": rng.uniform(-2.0, 3.0, n),
        }
    )
    p = tmp_path / "env.csv"
    df.to_csv(p, index=False)
    return p


def _make_list_csv(tmp_path: Path, galaxies: list[str]) -> Path:
    p = tmp_path / "list.csv"
    pd.DataFrame({"galaxy": galaxies}).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_basic_merge(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, 30)
        env = _make_env_csv(tmp_path, 30)
        df = load_data(f3, env)
        assert "galaxy" in df.columns
        assert "beta" in df.columns
        assert "delta_mass_std" in df.columns
        assert len(df) == 30

    def test_friction_slope_renamed_to_beta(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, 10, col="friction_slope")
        env = _make_env_csv(tmp_path, 10)
        df = load_data(f3, env)
        assert "beta" in df.columns
        assert "friction_slope" not in df.columns

    def test_beta_column_kept_as_is(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, 10, col="beta")
        env = _make_env_csv(tmp_path, 10)
        df = load_data(f3, env)
        assert "beta" in df.columns

    def test_inner_join_drops_unmatched(self, tmp_path):
        # f3 has 20 galaxies, env has 15 overlapping
        rng = np.random.default_rng(5)
        f3_df = pd.DataFrame(
            {
                "galaxy": [f"G{i:03d}" for i in range(20)],
                "beta": rng.uniform(0.3, 0.9, 20),
                "log_M_bar": rng.uniform(8, 11, 20),
                "log_Rmax": rng.uniform(0.5, 2, 20),
            }
        )
        env_df = pd.DataFrame(
            {
                "galaxy": [f"G{i:03d}" for i in range(5, 20)],
                "delta_mass_std": rng.uniform(-2, 3, 15),
            }
        )
        f3_p = tmp_path / "f3.csv"
        env_p = tmp_path / "env.csv"
        f3_df.to_csv(f3_p, index=False)
        env_df.to_csv(env_p, index=False)
        df = load_data(f3_p, env_p)
        assert len(df) == 15

    def test_missing_f3_file_raises(self, tmp_path):
        env = _make_env_csv(tmp_path, 5)
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "no_such_file.csv", env)

    def test_missing_env_file_raises(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, 5)
        with pytest.raises(FileNotFoundError):
            load_data(f3, tmp_path / "no_such_file.csv")

    def test_missing_beta_column_raises(self, tmp_path):
        bad_df = pd.DataFrame({"galaxy": ["A"], "log_M_bar": [9.0]})
        p = tmp_path / "bad_f3.csv"
        bad_df.to_csv(p, index=False)
        env = _make_env_csv(tmp_path, 1)
        with pytest.raises(ValueError, match="missing required columns"):
            load_data(p, env)

    def test_missing_delta_mass_raises(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, 5)
        bad_env = pd.DataFrame({"galaxy": [f"G{i:03d}" for i in range(5)]})
        p = tmp_path / "bad_env.csv"
        bad_env.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_data(f3, p)


# ---------------------------------------------------------------------------
# select_25_extremes
# ---------------------------------------------------------------------------


class TestSelect25Extremes:
    def _make_df(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "galaxy": [f"G{i:03d}" for i in range(n)],
                "beta": rng.uniform(0.2, 1.0, n),
                "delta_mass_std": np.linspace(-3.0, 3.0, n),
            }
        )

    def test_auto_returns_25_rows(self):
        df = self._make_df(50)
        sub = select_25_extremes(df)
        assert len(sub) == 25

    def test_auto_low_head(self):
        df = self._make_df(50)
        sub = select_25_extremes(df)
        # 12 lowest delta_mass_std
        expected_low = df.sort_values("delta_mass_std").head(12)["galaxy"].tolist()
        for g in expected_low:
            assert g in sub["galaxy"].values

    def test_auto_high_tail(self):
        df = self._make_df(50)
        sub = select_25_extremes(df)
        expected_high = df.sort_values("delta_mass_std").tail(13)["galaxy"].tolist()
        for g in expected_high:
            assert g in sub["galaxy"].values

    def test_list_path_filters(self, tmp_path):
        df = self._make_df(50)
        galaxy_subset = df["galaxy"].iloc[5:15].tolist()
        list_p = _make_list_csv(tmp_path, galaxy_subset)
        sub = select_25_extremes(df, list_path=list_p)
        assert len(sub) == 10
        assert set(sub["galaxy"]) == set(galaxy_subset)

    def test_list_path_missing_file_raises(self, tmp_path):
        df = self._make_df(10)
        with pytest.raises(FileNotFoundError):
            select_25_extremes(df, list_path=tmp_path / "no_file.csv")

    def test_list_path_missing_galaxy_column_raises(self, tmp_path):
        df = self._make_df(10)
        bad_list = tmp_path / "bad_list.csv"
        pd.DataFrame({"name": ["G001"]}).to_csv(bad_list, index=False)
        with pytest.raises(ValueError, match="galaxy"):
            select_25_extremes(df, list_path=bad_list)

    def test_auto_small_df(self):
        # Fewer than 25 rows — should return all rows
        df = self._make_df(10)
        sub = select_25_extremes(df)
        # head(12) + tail(13) on a 10-row df may have overlaps; concat is used
        assert len(sub) <= 25

    def test_auto_no_duplicates_large(self):
        # With 26+ rows the 12+13 selections should not overlap
        df = self._make_df(26)
        sub = select_25_extremes(df)
        assert sub["galaxy"].nunique() == 25


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def _make_df(self, n: int = 25, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "galaxy": [f"G{i}" for i in range(n)],
                "beta": rng.uniform(0.2, 1.0, n),
                "delta_mass_std": rng.uniform(-2.0, 3.0, n),
            }
        )

    def test_returns_dict_with_required_keys(self):
        df = self._make_df()
        stats = compute_stats(df)
        for k in ("N", "beta_mean", "beta_std", "rho_spearman", "p_spearman"):
            assert k in stats, f"Missing key: {k}"

    def test_N_correct(self):
        df = self._make_df(25)
        assert compute_stats(df)["N"] == 25

    def test_beta_mean_correct(self):
        df = self._make_df(10)
        stats = compute_stats(df)
        np.testing.assert_allclose(stats["beta_mean"], df["beta"].mean(), rtol=1e-10)

    def test_beta_std_correct(self):
        df = self._make_df(10)
        stats = compute_stats(df)
        np.testing.assert_allclose(stats["beta_std"], df["beta"].std(ddof=0), rtol=1e-10)

    def test_p_value_in_0_1(self):
        df = self._make_df(20)
        p = compute_stats(df)["p_spearman"]
        assert 0.0 <= p <= 1.0

    def test_perfect_negative_correlation(self):
        n = 25
        delta = np.linspace(-1, 1, n)
        beta = np.linspace(1, -1, n)  # perfectly anti-correlated
        df = pd.DataFrame(
            {
                "galaxy": [f"G{i}" for i in range(n)],
                "beta": beta,
                "delta_mass_std": delta,
            }
        )
        stats = compute_stats(df)
        np.testing.assert_allclose(stats["rho_spearman"], -1.0, atol=1e-10)

    def test_perfect_positive_correlation(self):
        n = 25
        delta = np.linspace(-1, 1, n)
        df = pd.DataFrame(
            {
                "galaxy": [f"G{i}" for i in range(n)],
                "beta": delta.copy(),
                "delta_mass_std": delta,
            }
        )
        stats = compute_stats(df)
        np.testing.assert_allclose(stats["rho_spearman"], 1.0, atol=1e-10)

    def test_all_values_are_python_scalars(self):
        stats = compute_stats(self._make_df())
        assert isinstance(stats["N"], int)
        for k in ("beta_mean", "beta_std", "rho_spearman", "p_spearman"):
            assert isinstance(stats[k], float)


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def _sample(self) -> tuple[pd.DataFrame, dict]:
        rng = np.random.default_rng(3)
        n = 25
        df = pd.DataFrame(
            {
                "galaxy": [f"G{i}" for i in range(n)],
                "beta": rng.uniform(0.2, 1.0, n),
                "delta_mass_std": rng.uniform(-2.0, 3.0, n),
            }
        )
        stats = {
            "N": n,
            "beta_mean": float(df["beta"].mean()),
            "beta_std": float(df["beta"].std()),
            "rho_spearman": -0.42,
            "p_spearman": 0.034,
        }
        return df, stats

    def test_creates_results_csv(self, tmp_path):
        df, stats = self._sample()
        save_outputs(df, stats, tmp_path)
        assert (tmp_path / "extreme_25_results.csv").exists()

    def test_creates_summary_txt(self, tmp_path):
        df, stats = self._sample()
        save_outputs(df, stats, tmp_path)
        assert (tmp_path / "extreme_25_summary.txt").exists()

    def test_csv_round_trip(self, tmp_path):
        df, stats = self._sample()
        save_outputs(df, stats, tmp_path)
        loaded = pd.read_csv(tmp_path / "extreme_25_results.csv")
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)

    def test_summary_contains_key_fields(self, tmp_path):
        df, stats = self._sample()
        save_outputs(df, stats, tmp_path)
        text = (tmp_path / "extreme_25_summary.txt").read_text(encoding="utf-8")
        assert "N = 25" in text
        assert "Spearman rho" in text
        assert "p-value" in text
        assert "Interpretation" in text

    def test_creates_output_directory(self, tmp_path):
        df, stats = self._sample()
        new_dir = tmp_path / "deep" / "nested"
        save_outputs(df, stats, new_dir)
        assert new_dir.is_dir()

    def test_summary_rho_value(self, tmp_path):
        df, stats = self._sample()
        save_outputs(df, stats, tmp_path)
        text = (tmp_path / "extreme_25_summary.txt").read_text(encoding="utf-8")
        assert "-0.420" in text

    def test_summary_p_value(self, tmp_path):
        df, stats = self._sample()
        save_outputs(df, stats, tmp_path)
        text = (tmp_path / "extreme_25_summary.txt").read_text(encoding="utf-8")
        assert "3.400e-02" in text
