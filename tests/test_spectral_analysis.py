"""
Unit tests for scripts/spectral_analysis.py.

All tests use synthetic in-memory data; no real dataset is required.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.spectral_analysis import (
    DEFAULT_LOGM_MIN,
    compute_env_residual_correlation,
    compute_mass_controlled_residuals,
    filter_high_mass,
    load_dataset,
    main,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic spectral dataset DataFrame."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(9.0, 12.0, n)
    # power correlated with logM plus noise
    power = 2.5 * logM + rng.normal(0, 0.5, n)
    # delta_mass_std mildly anti-correlated with power
    delta_mass_std = -0.2 * power + rng.normal(0, 0.3, n)
    return pd.DataFrame({"logM": logM, "power": power, "delta_mass_std": delta_mass_std})


def _make_csv(df: pd.DataFrame) -> str:
    """Write *df* to a temp CSV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_returns_dataframe(self):
        df = _make_df()
        path = _make_csv(df)
        result = load_dataset(path)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        df = _make_df()
        path = _make_csv(df)
        result = load_dataset(path)
        assert {"logM", "power", "delta_mass_std"}.issubset(result.columns)

    def test_row_count_preserved(self):
        df = _make_df(n=80)
        path = _make_csv(df)
        result = load_dataset(path)
        assert len(result) == 80

    def test_raises_on_missing_column(self):
        df = _make_df().drop(columns=["power"])
        path = _make_csv(df)
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(path)

    def test_raises_on_multiple_missing_columns(self):
        df = _make_df().drop(columns=["power", "delta_mass_std"])
        path = _make_csv(df)
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(path)

    def test_extra_columns_kept(self):
        df = _make_df()
        df["extra"] = 99
        path = _make_csv(df)
        result = load_dataset(path)
        assert "extra" in result.columns

    def test_path_as_string(self):
        df = _make_df()
        path = _make_csv(df)
        result = load_dataset(str(path))
        assert len(result) > 0

    def test_path_as_pathlib(self):
        df = _make_df()
        path = _make_csv(df)
        result = load_dataset(Path(path))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# filter_high_mass
# ---------------------------------------------------------------------------

class TestFilterHighMass:
    def test_default_threshold(self):
        df = _make_df(n=200)
        out = filter_high_mass(df)
        assert (out["logM"] >= DEFAULT_LOGM_MIN).all()

    def test_custom_threshold(self):
        df = _make_df(n=200)
        out = filter_high_mass(df, logm_min=11.0)
        assert (out["logM"] >= 11.0).all()

    def test_returns_copy(self):
        df = _make_df()
        out = filter_high_mass(df)
        out["logM"] = 0.0
        assert not (df["logM"] == 0.0).all()

    def test_empty_result_for_very_high_threshold(self):
        df = _make_df(n=50)
        out = filter_high_mass(df, logm_min=99.0)
        assert len(out) == 0

    def test_all_rows_for_very_low_threshold(self):
        df = _make_df(n=50)
        out = filter_high_mass(df, logm_min=-99.0)
        assert len(out) == len(df)

    def test_preserves_index(self):
        df = _make_df(n=100)
        out = filter_high_mass(df, logm_min=10.0)
        assert set(out.index).issubset(set(df.index))


# ---------------------------------------------------------------------------
# compute_mass_controlled_residuals
# ---------------------------------------------------------------------------

class TestComputeMassControlledResiduals:
    def test_adds_residual_column(self):
        df = _make_df(n=60)
        out = compute_mass_controlled_residuals(df)
        assert "residual" in out.columns

    def test_residuals_near_zero_mean(self):
        df = _make_df(n=200)
        out = compute_mass_controlled_residuals(df)
        assert abs(out["residual"].mean()) < 1e-8

    def test_original_df_unchanged(self):
        df = _make_df(n=60)
        _ = compute_mass_controlled_residuals(df)
        assert "residual" not in df.columns

    def test_residual_length_matches(self):
        df = _make_df(n=75)
        out = compute_mass_controlled_residuals(df)
        assert len(out) == 75

    def test_returns_dataframe(self):
        df = _make_df(n=40)
        out = compute_mass_controlled_residuals(df)
        assert isinstance(out, pd.DataFrame)

    def test_residuals_not_all_zero(self):
        df = _make_df(n=50)
        out = compute_mass_controlled_residuals(df)
        assert out["residual"].std() > 0


# ---------------------------------------------------------------------------
# compute_env_residual_correlation
# ---------------------------------------------------------------------------

class TestComputeEnvResidualCorrelation:
    def _prepared_df(self, n: int = 100) -> pd.DataFrame:
        df = _make_df(n=n)
        df = filter_high_mass(df, logm_min=9.0)
        return compute_mass_controlled_residuals(df)

    def test_returns_dict_with_required_keys(self):
        df = self._prepared_df()
        result = compute_env_residual_correlation(df)
        assert {"rho", "p", "n"} == set(result.keys())

    def test_rho_in_valid_range(self):
        df = self._prepared_df()
        result = compute_env_residual_correlation(df)
        assert -1.0 <= result["rho"] <= 1.0

    def test_p_in_valid_range(self):
        df = self._prepared_df()
        result = compute_env_residual_correlation(df)
        assert 0.0 <= result["p"] <= 1.0

    def test_n_matches_dataframe_length(self):
        df = self._prepared_df(n=80)
        result = compute_env_residual_correlation(df)
        assert result["n"] == len(df)

    def test_rho_is_float(self):
        df = self._prepared_df()
        result = compute_env_residual_correlation(df)
        assert isinstance(result["rho"], float)

    def test_p_is_float(self):
        df = self._prepared_df()
        result = compute_env_residual_correlation(df)
        assert isinstance(result["p"], float)

    def test_n_is_int(self):
        df = self._prepared_df()
        result = compute_env_residual_correlation(df)
        assert isinstance(result["n"], int)

    def test_negative_correlation_detected(self):
        # Build dataset with strong negative correlation
        rng = np.random.default_rng(0)
        n = 200
        residual = rng.uniform(-1, 1, n)
        delta = -0.9 * residual + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"delta_mass_std": delta, "residual": residual})
        result = compute_env_residual_correlation(df)
        assert result["rho"] < -0.5

    def test_positive_correlation_detected(self):
        rng = np.random.default_rng(1)
        n = 200
        residual = rng.uniform(-1, 1, n)
        delta = 0.9 * residual + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"delta_mass_std": delta, "residual": residual})
        result = compute_env_residual_correlation(df)
        assert result["rho"] > 0.5


# ---------------------------------------------------------------------------
# run_analysis
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_returns_dict(self):
        df = _make_df(n=150)
        path = _make_csv(df)
        result = run_analysis(csv_path=path, logm_min=9.0)
        assert isinstance(result, dict)

    def test_keys_present(self):
        df = _make_df(n=150)
        path = _make_csv(df)
        result = run_analysis(csv_path=path, logm_min=9.0)
        assert {"rho", "p", "n"} == set(result.keys())

    def test_n_matches_filter(self):
        df = _make_df(n=200)
        path = _make_csv(df)
        expected_n = int((df["logM"] >= 10.5).sum())
        result = run_analysis(csv_path=path, logm_min=10.5)
        assert result["n"] == expected_n

    def test_string_path(self):
        df = _make_df(n=100)
        path = _make_csv(df)
        result = run_analysis(csv_path=str(path), logm_min=9.0)
        assert result["n"] > 0

    def test_pathlib_path(self):
        df = _make_df(n=100)
        path = _make_csv(df)
        result = run_analysis(csv_path=Path(path), logm_min=9.0)
        assert result["n"] > 0

    def test_rho_range(self):
        df = _make_df(n=150)
        path = _make_csv(df)
        result = run_analysis(csv_path=path, logm_min=9.0)
        assert -1.0 <= result["rho"] <= 1.0

    def test_p_range(self):
        df = _make_df(n=150)
        path = _make_csv(df)
        result = run_analysis(csv_path=path, logm_min=9.0)
        assert 0.0 <= result["p"] <= 1.0


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, capsys):
        df = _make_df(n=150)
        path = _make_csv(df)
        result = main(["--csv", path, "--logM-min", "9.0"])
        assert isinstance(result, dict)

    def test_prints_rho(self, capsys):
        df = _make_df(n=100)
        path = _make_csv(df)
        main(["--csv", path, "--logM-min", "9.0"])
        out = capsys.readouterr().out
        assert "rho" in out

    def test_prints_p(self, capsys):
        df = _make_df(n=100)
        path = _make_csv(df)
        main(["--csv", path, "--logM-min", "9.0"])
        out = capsys.readouterr().out
        assert "p =" in out

    def test_prints_n(self, capsys):
        df = _make_df(n=100)
        path = _make_csv(df)
        main(["--csv", path, "--logM-min", "9.0"])
        out = capsys.readouterr().out
        assert "N =" in out

    def test_default_logm_min(self, capsys):
        df = _make_df(n=200)
        path = _make_csv(df)
        result = main(["--csv", path])
        expected_n = int((df["logM"] >= DEFAULT_LOGM_MIN).sum())
        assert result["n"] == expected_n

    def test_custom_logm_min(self, capsys):
        df = _make_df(n=200)
        path = _make_csv(df)
        result = main(["--csv", path, "--logM-min", "11.0"])
        expected_n = int((df["logM"] >= 11.0).sum())
        assert result["n"] == expected_n

    def test_output_header(self, capsys):
        df = _make_df(n=80)
        path = _make_csv(df)
        main(["--csv", path, "--logM-min", "9.0"])
        out = capsys.readouterr().out
        assert "ENV vs residual" in out
