"""Unit tests for src/g0_universality.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scm_models import G0_DEFAULT, rar_g_obs
from src.g0_universality import (
    fit_g0_per_galaxy,
    fit_g0_by_group,
    universality_stats,
    ks_test_mass_quartiles,
    run_universality_analysis,
    MIN_POINTS_PER_GALAXY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparc_like_df(
    n_galaxies: int = 20,
    pts_per_galaxy: int = 15,
    g0_true: float = G0_DEFAULT,
    scatter: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic SPARC-like DataFrame."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_galaxies):
        g_bar = 10 ** rng.uniform(-13, -8, pts_per_galaxy)
        g_obs = rar_g_obs(g_bar, g0_true) * 10 ** rng.normal(0, scatter, pts_per_galaxy)
        r_kpc = rng.uniform(0.5, 30, pts_per_galaxy)
        for gb, go, r in zip(g_bar, g_obs, r_kpc):
            records.append({
                "galaxy": f"NGC{1000 + i}",
                "r_kpc": r,
                "g_bar": gb,
                "g_obs": go,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# fit_g0_per_galaxy
# ---------------------------------------------------------------------------

class TestFitG0PerGalaxy:
    def test_returns_one_row_per_galaxy(self):
        df = _make_sparc_like_df(n_galaxies=10, pts_per_galaxy=10)
        result = fit_g0_per_galaxy(df)
        assert len(result) == 10

    def test_required_columns(self):
        df = _make_sparc_like_df()
        result = fit_g0_per_galaxy(df)
        for col in ("galaxy", "n_points", "g0", "g0_err", "rms", "log10_g0", "median_g_bar"):
            assert col in result.columns

    def test_g0_near_true_value(self):
        df = _make_sparc_like_df(n_galaxies=30, pts_per_galaxy=30, scatter=0.03)
        result = fit_g0_per_galaxy(df)
        # Median g0 across galaxies should be close to true value
        assert abs(np.median(result["g0"]) - G0_DEFAULT) / G0_DEFAULT < 0.10

    def test_skips_galaxies_with_few_points(self):
        df = _make_sparc_like_df(n_galaxies=5, pts_per_galaxy=3)
        result = fit_g0_per_galaxy(df, min_points=MIN_POINTS_PER_GALAXY)
        # All galaxies have only 3 points < MIN_POINTS_PER_GALAXY=5, so none fitted
        assert len(result) == 0

    def test_includes_max_r_kpc_when_present(self):
        df = _make_sparc_like_df()
        result = fit_g0_per_galaxy(df)
        assert "max_r_kpc" in result.columns


# ---------------------------------------------------------------------------
# fit_g0_by_group
# ---------------------------------------------------------------------------

class TestFitG0ByGroup:
    def test_returns_one_row_per_group(self):
        df = _make_sparc_like_df(n_galaxies=40, pts_per_galaxy=15)
        df["test_group"] = (df["g_bar"] > df["g_bar"].median()).astype(int)
        result = fit_g0_by_group(df, group_col="test_group")
        assert len(result) == 2

    def test_required_columns(self):
        df = _make_sparc_like_df()
        df["grp"] = 0
        result = fit_g0_by_group(df, group_col="grp")
        for col in ("group", "n_galaxies", "n_points", "g0", "g0_err", "rms", "log10_g0"):
            assert col in result.columns

    def test_g0_positive(self):
        df = _make_sparc_like_df(n_galaxies=20, pts_per_galaxy=20)
        df["grp"] = (df["g_bar"] > np.median(df["g_bar"])).astype(int)
        result = fit_g0_by_group(df, group_col="grp")
        assert (result["g0"] > 0).all()


# ---------------------------------------------------------------------------
# universality_stats
# ---------------------------------------------------------------------------

class TestUniversalityStats:
    def test_returns_required_keys(self):
        df = _make_sparc_like_df(n_galaxies=20, pts_per_galaxy=15)
        pg = fit_g0_per_galaxy(df)
        stats = universality_stats(pg)
        for key in ("n_galaxies", "g0_median", "g0_std", "log10_g0_std",
                    "g0_p16", "g0_p84", "reference_g0", "ref_within_1sigma"):
            assert key in stats

    def test_n_galaxies_matches(self):
        df = _make_sparc_like_df(n_galaxies=15, pts_per_galaxy=10)
        pg = fit_g0_per_galaxy(df)
        stats = universality_stats(pg)
        assert stats["n_galaxies"] == len(pg)

    def test_ref_within_1sigma_for_perfect_data(self):
        df = _make_sparc_like_df(n_galaxies=50, pts_per_galaxy=30, scatter=0.02)
        pg = fit_g0_per_galaxy(df)
        stats = universality_stats(pg, reference_g0=G0_DEFAULT)
        assert stats["ref_within_1sigma"] is True


# ---------------------------------------------------------------------------
# ks_test_mass_quartiles
# ---------------------------------------------------------------------------

class TestKsTestMassQuartiles:
    def test_universal_for_identical_g0(self):
        """KS should find no difference when g₀ is identical everywhere."""
        df = _make_sparc_like_df(n_galaxies=60, pts_per_galaxy=20, scatter=0.01)
        pg = fit_g0_per_galaxy(df)
        ks = ks_test_mass_quartiles(df, pg, n_quartiles=4)
        # With small scatter and uniform g0, KS p-value should be > 0.05
        assert ks["is_universal"] is True

    def test_returns_required_keys(self):
        df = _make_sparc_like_df(n_galaxies=40, pts_per_galaxy=15)
        pg = fit_g0_per_galaxy(df)
        ks = ks_test_mass_quartiles(df, pg)
        for key in ("ks_stat", "ks_pvalue", "low_median_g0",
                    "high_median_g0", "delta_log10_g0", "is_universal"):
            assert key in ks

    def test_insufficient_data_returns_nan(self):
        df = _make_sparc_like_df(n_galaxies=2, pts_per_galaxy=10)
        pg = fit_g0_per_galaxy(df)
        ks = ks_test_mass_quartiles(df, pg, n_quartiles=4)
        assert np.isnan(ks["ks_stat"])
        assert ks["is_universal"] is False


# ---------------------------------------------------------------------------
# run_universality_analysis (integration)
# ---------------------------------------------------------------------------

class TestRunUniversalityAnalysis:
    def test_runs_on_sparc_csv(self):
        results = run_universality_analysis("data/sparc_rar_sample.csv")
        assert "per_galaxy" in results
        assert "stats" in results
        assert "group_fits" in results
        assert "ks" in results

    def test_per_galaxy_not_empty(self):
        results = run_universality_analysis("data/sparc_rar_sample.csv")
        assert len(results["per_galaxy"]) > 10

    def test_g0_median_near_canonical(self):
        """Median fitted g₀ across SPARC galaxies should be near 1.2e-10 m/s²."""
        results = run_universality_analysis("data/sparc_rar_sample.csv")
        g0_med = results["stats"]["g0_median"]
        # Should be within a factor of 3 of the canonical value
        assert 0.3e-10 < g0_med < 5e-10

    def test_group_fits_have_positive_g0(self):
        results = run_universality_analysis("data/sparc_rar_sample.csv")
        assert (results["group_fits"]["g0"] > 0).all()

    def test_saves_csv_when_out_dir_given(self, tmp_path):
        run_universality_analysis(
            "data/sparc_rar_sample.csv",
            out_dir=str(tmp_path),
        )
        assert (tmp_path / "g0_per_galaxy.csv").exists()
        assert (tmp_path / "g0_by_mass_quartile.csv").exists()

    def test_raises_without_galaxy_column(self, tmp_path):
        import pandas as pd
        # Create a CSV without a galaxy column
        df = pd.DataFrame({"g_bar": [1e-11, 1e-10], "g_obs": [1e-11, 1e-10]})
        tmp_csv = tmp_path / "no_galaxy.csv"
        df.to_csv(tmp_csv, index=False)
        with pytest.raises(ValueError, match="galaxy"):
            run_universality_analysis(str(tmp_csv))
