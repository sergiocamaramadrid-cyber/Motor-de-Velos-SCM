"""
test_population_stats.py — Unit tests for src.analysis.population_stats
and src.analysis.bias_diagnostics.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.population_stats import (
    beta_summary,
    beta_vs_mass,
    beta_by_quality,
    beta_by_survey,
)
from src.analysis.bias_diagnostics import (
    selection_bias_check,
    survey_comparison,
    n_deep_distribution,
)


# ---------------------------------------------------------------------------
# Synthetic catalog fixtures
# ---------------------------------------------------------------------------

def _make_catalog(n=50, seed=0):
    rng = np.random.default_rng(seed)
    betas = rng.normal(0.5, 0.05, n)
    return pd.DataFrame({
        "galaxy_id":        [f"G{i:04d}" for i in range(n)],
        "beta":             betas,
        "beta_err":         np.abs(rng.normal(0.02, 0.005, n)),
        "r_value":          rng.uniform(0.8, 1.0, n),
        "n_deep":           rng.integers(2, 15, n),
        "velo_inerte_flag": (betas >= 0.35) & (betas <= 0.65),
    })


def _make_catalog_with_extras(n=30, seed=1):
    df = _make_catalog(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    df["log_mstar"] = rng.uniform(7.5, 11.5, n)
    df["quality"]   = rng.choice([1, 2, 3], n)
    df["survey"]    = rng.choice(["SPARC", "BIG-SPARC"], n)
    return df


# ---------------------------------------------------------------------------
# Tests: beta_summary
# ---------------------------------------------------------------------------

class TestBetaSummary:
    def test_fields_present(self):
        cat = _make_catalog()
        s = beta_summary(cat)
        for key in ("n_total", "n_valid", "mean", "median", "std", "q16", "q84"):
            assert key in s.index

    def test_mean_close_to_05(self):
        cat = _make_catalog(n=200, seed=7)
        s = beta_summary(cat)
        assert s["mean"] == pytest.approx(0.5, abs=0.05)

    def test_n_total_correct(self):
        cat = _make_catalog(n=20)
        s = beta_summary(cat)
        assert s["n_total"] == 20

    def test_n_velo_inerte_counted(self):
        cat = _make_catalog(n=50)
        s = beta_summary(cat)
        assert s["n_velo_inerte"] == int(cat["velo_inerte_flag"].sum())

    def test_missing_beta_col_raises(self):
        with pytest.raises(KeyError):
            beta_summary(pd.DataFrame({"galaxy_id": ["G1"]}))


# ---------------------------------------------------------------------------
# Tests: beta_vs_mass
# ---------------------------------------------------------------------------

class TestBetaVsMass:
    def test_returns_dataframe(self):
        cat = _make_catalog_with_extras()
        result = beta_vs_mass(cat)
        assert isinstance(result, pd.DataFrame)
        assert "beta_median" in result.columns

    def test_n_bins(self):
        cat = _make_catalog_with_extras(n=60)
        result = beta_vs_mass(cat, n_bins=4)
        assert len(result) <= 4   # some bins might be empty

    def test_missing_mass_col_raises(self):
        cat = _make_catalog()
        with pytest.raises(KeyError):
            beta_vs_mass(cat)


# ---------------------------------------------------------------------------
# Tests: beta_by_quality
# ---------------------------------------------------------------------------

class TestBetaByQuality:
    def test_groups(self):
        cat = _make_catalog_with_extras(n=90)
        result = beta_by_quality(cat)
        assert set(result["quality"]).issubset({1, 2, 3})

    def test_missing_quality_col_raises(self):
        cat = _make_catalog()
        with pytest.raises(KeyError):
            beta_by_quality(cat)


# ---------------------------------------------------------------------------
# Tests: beta_by_survey
# ---------------------------------------------------------------------------

class TestBetaBySurvey:
    def test_survey_labels(self):
        cat = _make_catalog_with_extras(n=60)
        result = beta_by_survey(cat)
        assert set(result["survey"]).issubset({"SPARC", "BIG-SPARC"})


# ---------------------------------------------------------------------------
# Tests: selection_bias_check
# ---------------------------------------------------------------------------

class TestSelectionBiasCheck:
    def test_unbiased_sample(self):
        rng = np.random.default_rng(99)
        betas = rng.normal(0.5, 0.05, 100)   # centred at 0.5 → expect p large
        cat = pd.DataFrame({"galaxy_id": [f"G{i}" for i in range(100)], "beta": betas})
        result = selection_bias_check(cat)
        assert result["n"] == 100
        # Should NOT be flagged as biased (most of the time)
        assert result["p_value"] > 0.001

    def test_biased_sample(self):
        betas = np.full(80, 0.8)   # clearly above 0.5
        cat = pd.DataFrame({"galaxy_id": [f"G{i}" for i in range(80)], "beta": betas})
        result = selection_bias_check(cat)
        assert result["biased"] is True

    def test_too_few_points(self):
        cat = pd.DataFrame({"galaxy_id": ["G1", "G2"], "beta": [0.5, 0.51]})
        result = selection_bias_check(cat)
        assert np.isnan(result["p_value"])


# ---------------------------------------------------------------------------
# Tests: survey_comparison
# ---------------------------------------------------------------------------

class TestSurveyComparison:
    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        betas = rng.normal(0.5, 0.05, 50)
        cat_a = pd.DataFrame({"beta": betas})
        cat_b = pd.DataFrame({"beta": betas})
        result = survey_comparison(cat_a, cat_b)
        # Same data → p should be > 0.05
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_different_distributions(self):
        rng = np.random.default_rng(7)
        cat_a = pd.DataFrame({"beta": rng.normal(0.5, 0.05, 60)})
        cat_b = pd.DataFrame({"beta": rng.normal(0.8, 0.05, 60)})
        result = survey_comparison(cat_a, cat_b)
        assert result["significant"] is True

    def test_too_few_returns_nan(self):
        cat_a = pd.DataFrame({"beta": [0.5, 0.6]})
        cat_b = pd.DataFrame({"beta": [0.7, 0.8]})
        result = survey_comparison(cat_a, cat_b)
        assert np.isnan(result["p_value"])


# ---------------------------------------------------------------------------
# Tests: n_deep_distribution
# ---------------------------------------------------------------------------

class TestNDeepDistribution:
    def test_counts_sum_to_n(self):
        cat = _make_catalog(n=40)
        dist = n_deep_distribution(cat)
        assert dist["count"].sum() == len(cat)

    def test_fraction_sums_to_1(self):
        cat = _make_catalog(n=50)
        dist = n_deep_distribution(cat)
        assert dist["fraction"].sum() == pytest.approx(1.0, abs=1e-9)
