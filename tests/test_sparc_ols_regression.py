"""
tests/test_sparc_ols_regression.py -- Tests for scripts/sparc_ols_regression.py.

Covers:
  1. fit_models() -- subsample filtering, model fitting, output schema.
  2. format_summary() -- output is a non-empty string with key phrases.
  3. main() CLI -- end-to-end invocation with temp CSV.
  4. Integration: committed SPARC subset fixture (regression guard).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.sparc_ols_regression import (
    BETA_REF,
    M_CRIT_DEFAULT,
    _REQUIRED_COLS,
    fit_models,
    format_summary,
    main,
    _parse_args,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SPARC_CSV = _REPO_ROOT / "data" / "sparc_subset.csv"


def _make_catalog(
    n: int = 40,
    seed: int = 0,
    logM_range: tuple = (10.0, 11.5),
) -> pd.DataFrame:
    """Minimal synthetic catalog with all required columns + delta_f3."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(*logM_range, n)
    delta_mass_std = rng.normal(0.0, 1.0, n)
    slope_tail = 0.5 + rng.normal(0.0, 0.1, n) - 0.05 * delta_mass_std
    df = pd.DataFrame(
        {
            "galaxy": [f"NGC{i:04d}" for i in range(n)],
            "logM": logM,
            "delta_mass_std": delta_mass_std,
            "slope_tail": slope_tail,
        }
    )
    df["delta_f3"] = df["slope_tail"] - BETA_REF
    return df


def _write_catalog(df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write catalog to CSV without delta_f3 (main() must compute it)."""
    p = tmp_path / "catalog.csv"
    cols = [c for c in df.columns if c != "delta_f3"]
    df[cols].to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. fit_models()
# ---------------------------------------------------------------------------


class TestFitModels:
    def test_returns_dict(self):
        df = _make_catalog()
        result = fit_models(df, m_crit=10.0)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _make_catalog()
        result = fit_models(df, m_crit=10.0)
        for key in ("subsample", "model1", "model2", "n", "m_crit"):
            assert key in result

    def test_subsample_filters_by_logM(self):
        df = _make_catalog(n=60)
        m_crit = 10.5
        result = fit_models(df, m_crit=m_crit)
        assert (result["subsample"]["logM"] >= m_crit).all()

    def test_n_matches_subsample_length(self):
        df = _make_catalog(n=60)
        result = fit_models(df, m_crit=10.0)
        assert result["n"] == len(result["subsample"])

    def test_m_crit_stored(self):
        df = _make_catalog(n=60)
        m_crit = 10.3
        result = fit_models(df, m_crit=m_crit)
        assert result["m_crit"] == pytest.approx(m_crit)

    def test_model1_has_two_params(self):
        df = _make_catalog(n=40)
        result = fit_models(df, m_crit=10.0)
        # intercept + delta_mass_std
        assert len(result["model1"].params) == 2

    def test_model2_has_three_params(self):
        df = _make_catalog(n=40)
        result = fit_models(df, m_crit=10.0)
        # intercept + delta_mass_std + logM
        assert len(result["model2"].params) == 3

    def test_model1_params_names(self):
        df = _make_catalog(n=40)
        result = fit_models(df, m_crit=10.0)
        assert "const" in result["model1"].params.index
        assert "delta_mass_std" in result["model1"].params.index

    def test_model2_params_names(self):
        df = _make_catalog(n=40)
        result = fit_models(df, m_crit=10.0)
        for name in ("const", "delta_mass_std", "logM"):
            assert name in result["model2"].params.index

    def test_r_squared_in_0_1(self):
        df = _make_catalog(n=50)
        result = fit_models(df, m_crit=10.0)
        assert 0.0 <= result["model1"].rsquared <= 1.0
        assert 0.0 <= result["model2"].rsquared <= 1.0

    def test_pvalues_in_0_1(self):
        df = _make_catalog(n=50)
        result = fit_models(df, m_crit=10.0)
        for model in (result["model1"], result["model2"]):
            assert (model.pvalues >= 0).all()
            assert (model.pvalues <= 1).all()

    def test_hc3_covariance_used(self):
        df = _make_catalog(n=40)
        result = fit_models(df, m_crit=10.0)
        # statsmodels stores cov_type in model results
        assert result["model1"].cov_type == "HC3"
        assert result["model2"].cov_type == "HC3"

    def test_raises_on_missing_delta_f3(self):
        df = _make_catalog(n=40)
        df_no_f3 = df.drop(columns=["delta_f3"])
        # Should raise because delta_f3 is missing
        with pytest.raises(ValueError, match="delta_f3"):
            fit_models(df_no_f3, m_crit=10.0)

    def test_raises_on_missing_required_columns(self):
        df = _make_catalog(n=40)
        df_bad = df.drop(columns=["delta_mass_std"])
        with pytest.raises(ValueError, match="missing required columns"):
            fit_models(df_bad, m_crit=10.0)

    def test_raises_when_subsample_too_small(self):
        df = _make_catalog(n=10, logM_range=(9.0, 9.5))
        with pytest.raises(ValueError, match="at least 4"):
            fit_models(df, m_crit=10.0)

    def test_default_m_crit(self):
        df = _make_catalog(n=60)
        result = fit_models(df)
        assert result["m_crit"] == pytest.approx(M_CRIT_DEFAULT)

    def test_model2_r2_geq_model1(self):
        """Adding logM should not reduce R-squared."""
        df = _make_catalog(n=60, seed=7)
        result = fit_models(df, m_crit=10.0)
        # R2 increases or stays same with extra predictor
        assert result["model2"].rsquared >= result["model1"].rsquared - 1e-10

    def test_signal_detected_synthetic(self):
        """With a planted signal, delta_mass_std coefficient should be negative."""
        rng = np.random.default_rng(42)
        n = 60
        logM = rng.uniform(10.0, 11.5, n)
        delta_mass_std = rng.normal(0, 1, n)
        slope_tail = 0.5 - 0.1 * delta_mass_std + rng.normal(0, 0.02, n)
        df = pd.DataFrame(
            {
                "logM": logM,
                "delta_mass_std": delta_mass_std,
                "slope_tail": slope_tail,
            }
        )
        df["delta_f3"] = df["slope_tail"] - BETA_REF
        result = fit_models(df, m_crit=10.0)
        assert result["model1"].params["delta_mass_std"] < 0

    def test_all_galaxies_included_below_all_data(self):
        """Setting m_crit very low should include all galaxies."""
        df = _make_catalog(n=40, logM_range=(10.5, 11.5))
        result = fit_models(df, m_crit=9.0)
        assert result["n"] == len(df)


# ---------------------------------------------------------------------------
# 2. format_summary()
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def _make_result(self):
        df = _make_catalog(n=40)
        return fit_models(df, m_crit=10.0)

    def test_returns_string(self):
        result = self._make_result()
        summary = format_summary(result)
        assert isinstance(summary, str)

    def test_non_empty(self):
        result = self._make_result()
        summary = format_summary(result)
        assert len(summary) > 0

    def test_contains_model1_header(self):
        result = self._make_result()
        summary = format_summary(result)
        assert "Model 1" in summary

    def test_contains_model2_header(self):
        result = self._make_result()
        summary = format_summary(result)
        assert "Model 2" in summary

    def test_contains_n_value(self):
        result = self._make_result()
        summary = format_summary(result)
        assert str(result["n"]) in summary

    def test_contains_m_crit(self):
        result = self._make_result()
        summary = format_summary(result)
        assert "10.00" in summary

    def test_contains_delta_mass_std(self):
        result = self._make_result()
        summary = format_summary(result)
        assert "delta_mass_std" in summary

    def test_contains_logM(self):
        result = self._make_result()
        summary = format_summary(result)
        assert "logM" in summary

    def test_contains_hc3_indicator(self):
        result = self._make_result()
        summary = format_summary(result)
        # statsmodels summary includes the covariance type
        assert "HC3" in summary


# ---------------------------------------------------------------------------
# 3. _parse_args()
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.m_crit == pytest.approx(M_CRIT_DEFAULT)
        assert args.out is None

    def test_m_crit_override(self):
        args = _parse_args(["--m-crit", "10.8"])
        assert args.m_crit == pytest.approx(10.8)

    def test_csv_override(self, tmp_path):
        p = tmp_path / "test.csv"
        args = _parse_args(["--csv", str(p)])
        assert args.csv == str(p)

    def test_out_override(self, tmp_path):
        p = tmp_path / "summary.txt"
        args = _parse_args(["--out", str(p)])
        assert args.out == str(p)


# ---------------------------------------------------------------------------
# 4. main() -- CLI end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_dict(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv_path)])
        assert isinstance(result, dict)

    def test_required_keys_present(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv_path)])
        for key in ("subsample", "model1", "model2", "n", "m_crit", "summary"):
            assert key in result

    def test_summary_is_string(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv_path)])
        assert isinstance(result["summary"], str)

    def test_out_file_written(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        out_path = tmp_path / "summary.txt"
        main(["--csv", str(csv_path), "--out", str(out_path)])
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_out_file_content_matches_summary(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        out_path = tmp_path / "summary.txt"
        result = main(["--csv", str(csv_path), "--out", str(out_path)])
        written = out_path.read_text(encoding="utf-8")
        assert written == result["summary"]

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv")])

    def test_csv_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"logM": [10.5], "delta_mass_std": [0.1]}).to_csv(
            bad_csv, index=False
        )
        with pytest.raises(ValueError):
            main(["--csv", str(bad_csv)])

    def test_m_crit_passed_through(self, tmp_path):
        df = _make_catalog(n=60)
        csv_path = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv_path), "--m-crit", "10.2"])
        assert result["m_crit"] == pytest.approx(10.2)

    def test_n_is_positive(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv_path)])
        assert result["n"] > 0

    def test_out_parent_created(self, tmp_path):
        df = _make_catalog(n=50)
        csv_path = _write_catalog(df, tmp_path)
        out_path = tmp_path / "subdir" / "summary.txt"
        main(["--csv", str(csv_path), "--out", str(out_path)])
        assert out_path.exists()


# ---------------------------------------------------------------------------
# 5. Integration: committed SPARC subset (regression guard)
# ---------------------------------------------------------------------------


class TestIntegrationSPARC:
    @pytest.fixture
    def sparc_result(self):
        pytest.importorskip("statsmodels")
        if not _SPARC_CSV.exists():
            pytest.skip(f"SPARC CSV not found: {_SPARC_CSV}")
        return main(["--csv", str(_SPARC_CSV), "--m-crit", str(M_CRIT_DEFAULT)])

    def test_runs_without_error(self, sparc_result):
        assert sparc_result is not None

    def test_n_is_56(self, sparc_result):
        assert sparc_result["n"] == 56

    def test_model1_coeff_negative(self, sparc_result):
        coeff = sparc_result["model1"].params["delta_mass_std"]
        assert coeff < 0, f"Expected negative coefficient, got {coeff}"

    def test_model1_pvalue_significant(self, sparc_result):
        """delta_mass_std should be significant at 5 % after HC3 correction."""
        pval = sparc_result["model1"].pvalues["delta_mass_std"]
        assert pval < 0.05, f"Expected p < 0.05, got {pval:.4e}"

    def test_model1_r2_positive(self, sparc_result):
        assert sparc_result["model1"].rsquared > 0

    def test_model2_delta_mass_std_coefficient_negative(self, sparc_result):
        coeff = sparc_result["model2"].params["delta_mass_std"]
        assert coeff < 0

    def test_model2_r2_geq_model1(self, sparc_result):
        r2_1 = sparc_result["model1"].rsquared
        r2_2 = sparc_result["model2"].rsquared
        assert r2_2 >= r2_1 - 1e-10

    def test_summary_contains_n_56(self, sparc_result):
        assert "56" in sparc_result["summary"]

    def test_summary_contains_m_crit(self, sparc_result):
        assert "10.05" in sparc_result["summary"]

    def test_subsample_all_high_mass(self, sparc_result):
        assert (sparc_result["subsample"]["logM"] >= M_CRIT_DEFAULT).all()
