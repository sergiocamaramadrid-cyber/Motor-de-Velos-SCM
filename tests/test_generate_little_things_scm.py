"""
tests/test_generate_little_things_scm.py — Tests for the LITTLE THINGS SCM pipeline.

Covers:
  - Physics helpers: compute_log_gobs, compute_f3, compute_reliable
  - build_catalog: correct columns and deep-MOND limiting case
  - compute_summary: statistics dict structure
  - run_little_things_scm: end-to-end output file generation
  - CLI: main() round-trip
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_little_things_scm import (
    A0_DEFAULT,
    ANALYSIS_TITLE,
    DEEP_THRESHOLD_DEFAULT,
    EXPECTED_F3_MOND,
    KPC_TO_M,
    REQUIRED_COLS,
    _LOG10_KPC_TO_M,
    build_catalog,
    compute_f3,
    compute_log_gobs,
    compute_mass_correlation_stats,
    compute_mass_detrend,
    compute_reliable,
    compute_summary,
    main,
    run_little_things_scm,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DATASET = _REPO_ROOT / "data" / "little_things_global.csv"

N_GALAXIES = 26


@pytest.fixture
def real_df() -> pd.DataFrame:
    return pd.read_csv(_DATASET)


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame with 3 toy galaxies."""
    return pd.DataFrame(
        {
            "galaxy_id": ["G1", "G2", "G3"],
            "logM": [7.0, 8.0, 6.5],
            # logVobs, log_gbar, log_j chosen so R_eff = 1 kpc for simplicity
            "logVobs": [1.5, 1.7, 1.3],
            "log_gbar": [-12.0, -11.5, -12.5],
            "log_j": [1.5, 1.7, 1.3],  # j = Vlast × 1 kpc → R_eff = 1 kpc
        }
    )


# ---------------------------------------------------------------------------
# compute_log_gobs
# ---------------------------------------------------------------------------


class TestComputeLogGobs:
    def test_scalar_finite(self):
        val = compute_log_gobs(1.5, 1.5)
        assert math.isfinite(val)

    def test_array_shape(self):
        logV = np.array([1.3, 1.5, 1.7])
        log_j = np.array([1.3, 1.5, 1.7])
        result = compute_log_gobs(logV, log_j)
        assert result.shape == (3,)

    def test_unit_consistency(self):
        """When R_eff = 1 kpc and Vlast = 10 km/s, check log_gobs manually."""
        logVobs = 1.0          # Vlast = 10 km/s
        log_j = 1.0            # j = 10 kpc·km/s → R_eff = j/V = 1 kpc
        log_gobs = compute_log_gobs(logVobs, log_j)

        V_ms = 10.0 * 1e3
        R_m = 1.0 * KPC_TO_M
        expected = math.log10(V_ms**2 / R_m)
        assert abs(log_gobs - expected) < 1e-10

    def test_higher_velocity_larger_gobs(self):
        """Larger Vlast at same R_eff should give larger g_obs."""
        g1 = compute_log_gobs(1.5, 1.5)
        g2 = compute_log_gobs(1.8, 1.8)
        assert g2 > g1

    def test_larger_radius_smaller_gobs(self):
        """Same Vlast but larger j (larger R_eff) → smaller g_obs."""
        g1 = compute_log_gobs(1.5, 1.5)
        g2 = compute_log_gobs(1.5, 2.0)
        assert g2 < g1


# ---------------------------------------------------------------------------
# compute_f3
# ---------------------------------------------------------------------------


class TestComputeF3:
    def test_mond_limit(self):
        """When g_obs = sqrt(g_bar × a0), F3 must equal 0.5 exactly."""
        log_gbar = -12.0
        log_gobs = 0.5 * (log_gbar + math.log10(A0_DEFAULT))
        f3 = compute_f3(log_gobs, log_gbar, a0=A0_DEFAULT)
        assert abs(f3 - 0.5) < 1e-12

    def test_scalar_finite(self):
        assert math.isfinite(compute_f3(-10.5, -12.0))

    def test_array_shape(self):
        log_gobs = np.array([-10.5, -11.0, -10.8])
        log_gbar = np.array([-12.0, -12.5, -11.8])
        result = compute_f3(log_gobs, log_gbar)
        assert result.shape == (3,)

    def test_above_mond_gives_lower_f3(self):
        """g_obs above the MOND prediction → F3 < 0.5.

        F3 = (log_gobs − 0.5·log_a0) / log_gbar.  Because log_gbar < 0,
        increasing log_gobs makes the numerator less negative while the
        denominator stays constant negative, so the ratio (F3) decreases.
        """
        log_gbar = -12.0
        log_gobs_mond = 0.5 * (log_gbar + math.log10(A0_DEFAULT))
        f3_mond = compute_f3(log_gobs_mond, log_gbar)
        f3_above = compute_f3(log_gobs_mond + 0.1, log_gbar)
        assert abs(f3_mond - 0.5) < 1e-10, "baseline should be 0.5"
        assert f3_above < f3_mond


# ---------------------------------------------------------------------------
# compute_reliable
# ---------------------------------------------------------------------------


class TestComputeReliable:
    def test_deep_regime_flagged(self):
        """Very small g_bar (deep regime) should return True."""
        assert compute_reliable(-12.0, a0=A0_DEFAULT, deep_threshold=0.3)

    def test_newtonian_not_flagged(self):
        """g_bar > threshold × a0 should return False."""
        log_threshold = math.log10(0.3 * A0_DEFAULT)
        # Set log_gbar above threshold
        log_gbar_high = log_threshold + 0.5  # more positive → larger g_bar
        assert not compute_reliable(log_gbar_high, a0=A0_DEFAULT, deep_threshold=0.3)

    def test_array(self):
        log_gbar = np.array([-12.0, -11.0, -9.5])  # -9.5 is above a0
        result = compute_reliable(log_gbar, a0=A0_DEFAULT, deep_threshold=0.3)
        assert result.shape == (3,)
        assert result[0]   # deep
        assert result[1]   # deep
        assert not result[2]  # near Newtonian


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------


class TestBuildCatalog:
    def test_required_columns_present(self, minimal_df):
        cat = build_catalog(minimal_df)
        for col in [
            "galaxy_id", "logM", "logVobs", "log_gbar", "log_j",
            "log_gobs", "friction_slope", "friction_slope_err",
            "delta_F3", "reliable", "velo_inerte_flag",
        ]:
            assert col in cat.columns, f"Missing column: {col}"

    def test_friction_slope_alias(self, minimal_df):
        cat = build_catalog(minimal_df)
        pd.testing.assert_series_equal(
            cat["reliable"], cat["velo_inerte_flag"],
            check_names=False,
        )

    def test_delta_f3_definition(self, minimal_df):
        cat = build_catalog(minimal_df)
        np.testing.assert_allclose(
            cat["delta_F3"].values,
            cat["friction_slope"].values - EXPECTED_F3_MOND,
        )

    def test_row_count_preserved(self, minimal_df):
        cat = build_catalog(minimal_df)
        assert len(cat) == len(minimal_df)

    def test_friction_slope_err_nan(self, minimal_df):
        cat = build_catalog(minimal_df)
        assert cat["friction_slope_err"].isna().all()

    def test_mond_limit_single_row(self):
        """Galaxy exactly on deep-MOND RAR should have F3 = 0.5."""
        log_gbar = -12.0
        a0 = A0_DEFAULT
        # Construct logVobs and log_j such that log_gobs = MOND prediction
        log_gobs_mond = 0.5 * (log_gbar + math.log10(a0))
        # log_gobs = 3*logVobs - log_j + 6 - log10(KPC_TO_M)
        # Set logVobs = log_j (R_eff = 1 kpc) → log_gobs = 2*logVobs + 6 - log10(KPC_TO_M)
        # Solve for logVobs: logVobs = (log_gobs - 6 + log10(KPC_TO_M)) / 2
        logVobs = (log_gobs_mond - 6.0 + _LOG10_KPC_TO_M) / 2.0
        log_j = logVobs  # ensures R_eff = 1 kpc

        df_mond = pd.DataFrame(
            {
                "galaxy_id": ["MOND_galaxy"],
                "logM": [7.0],
                "logVobs": [logVobs],
                "log_gbar": [log_gbar],
                "log_j": [log_j],
            }
        )
        cat = build_catalog(df_mond)
        assert abs(float(cat["friction_slope"].iloc[0]) - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_keys_present(self, minimal_df):
        cat = build_catalog(minimal_df)
        s = compute_summary(cat)
        for key in [
            "n_galaxies", "n_reliable", "f3_mean", "f3_median", "f3_std",
            "delta_f3_mean", "delta_f3_median", "delta_f3_std",
            "t_stat", "p_value_ttest", "consistent_mond",
            "spearman_f3_vlast_rho", "spearman_f3_vlast_p",
            "ols_slope", "ols_intercept",
            "spearman_resid_vlast_rho", "spearman_resid_vlast_p",
        ]:
            assert key in s, f"Missing summary key: {key}"

    def test_n_galaxies(self, minimal_df):
        cat = build_catalog(minimal_df)
        s = compute_summary(cat)
        assert s["n_galaxies"] == len(minimal_df)

    def test_consistent_mond_is_bool(self, minimal_df):
        cat = build_catalog(minimal_df)
        s = compute_summary(cat)
        assert isinstance(s["consistent_mond"], bool)


# ---------------------------------------------------------------------------
# run_little_things_scm (end-to-end)
# ---------------------------------------------------------------------------


class TestRunLittleThingsScm:
    def test_output_files_created(self, tmp_path):
        summary = run_little_things_scm(
            csv_path=_DATASET,
            out_dir=tmp_path,
            no_figures=True,
        )
        assert (tmp_path / "little_things_scm_catalog.csv").exists()
        assert (tmp_path / "scm_clean_sample.csv").exists()
        assert (tmp_path / "scm_clean_with_residual.csv").exists()
        assert (tmp_path / "summary.json").exists()

    def test_figures_created(self, tmp_path):
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=False)
        for fname in [
            "faseA_f3_vs_vlast.png",
            "scatter_f3_vlast.png",
            "hist_f3.png",
            "hist_delta_f3.png",
        ]:
            assert (tmp_path / fname).exists(), f"Missing figure: {fname}"

    def test_catalog_row_count(self, tmp_path):
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        df = pd.read_csv(tmp_path / "little_things_scm_catalog.csv")
        assert len(df) == N_GALAXIES

    def test_catalog_columns(self, tmp_path):
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        df = pd.read_csv(tmp_path / "little_things_scm_catalog.csv")
        for col in [
            "galaxy_id", "friction_slope", "delta_F3", "reliable",
            "log_gobs", "f3_mass_residual",
        ]:
            assert col in df.columns

    def test_clean_sample_subset_of_catalog(self, tmp_path):
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        cat = pd.read_csv(tmp_path / "little_things_scm_catalog.csv")
        clean = pd.read_csv(tmp_path / "scm_clean_sample.csv")
        assert len(clean) <= len(cat)
        assert set(clean["galaxy_id"]).issubset(set(cat["galaxy_id"]))

    def test_clean_with_residual_has_residual_column(self, tmp_path):
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        df = pd.read_csv(tmp_path / "scm_clean_with_residual.csv")
        assert "residual" in df.columns

    def test_summary_json_valid(self, tmp_path):
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        data = json.loads((tmp_path / "summary.json").read_text())
        assert data["n_galaxies"] == N_GALAXIES
        assert isinstance(data["consistent_mond"], bool)
        assert "spearman_f3_vlast_rho" in data
        assert "spearman_resid_vlast_rho" in data
        assert data.get("analysis_title") == "SCM \u2013 LITTLE THINGS F3 Analysis (Phase A)"

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_little_things_scm(
                csv_path=tmp_path / "nonexistent.csv",
                out_dir=tmp_path,
                no_figures=True,
            )

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy_id": ["X"], "logM": [7.0]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            run_little_things_scm(csv_path=bad_csv, out_dir=tmp_path, no_figures=True)

    def test_friction_slope_near_mond(self, tmp_path):
        """For LITTLE THINGS deep-regime dwarfs, median F3 should be near 0.5."""
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        df = pd.read_csv(tmp_path / "scm_clean_sample.csv")
        median_f3 = df["friction_slope"].median()
        assert 0.2 < median_f3 < 0.8, f"Unexpected median F3: {median_f3}"

    def test_delta_f3_zero_centered(self, tmp_path):
        """δF3 distribution should be centered near 0 for MOND-consistent dwarfs."""
        run_little_things_scm(csv_path=_DATASET, out_dir=tmp_path, no_figures=True)
        df = pd.read_csv(tmp_path / "scm_clean_sample.csv")
        mean_delta = df["delta_F3"].mean()
        assert abs(mean_delta) < 0.5, f"Unexpected mean δF3: {mean_delta}"


class TestAnalysisTitle:
    def test_title_value(self):
        assert ANALYSIS_TITLE == "SCM \u2013 LITTLE THINGS F3 Analysis (Phase A)"

    def test_title_in_summary(self, real_df):
        cat = build_catalog(real_df)
        s = compute_summary(cat)
        assert s["analysis_title"] == ANALYSIS_TITLE


# ---------------------------------------------------------------------------
# compute_mass_detrend
# ---------------------------------------------------------------------------


class TestComputeMassDetrend:
    def test_column_added(self, minimal_df):
        cat = build_catalog(minimal_df)
        cat = compute_mass_detrend(cat)
        assert "f3_mass_residual" in cat.columns

    def test_reliable_galaxies_have_residuals(self, real_df):
        cat = build_catalog(real_df)
        cat = compute_mass_detrend(cat)
        reliable = cat[cat["reliable"]]
        assert reliable["f3_mass_residual"].notna().all()

    def test_residuals_near_zero_mean(self, real_df):
        """OLS residuals must sum to zero by construction."""
        cat = build_catalog(real_df)
        cat = compute_mass_detrend(cat)
        resid = cat.loc[cat["reliable"], "f3_mass_residual"].dropna()
        assert abs(resid.mean()) < 1e-10

    def test_unreliable_galaxies_nan(self, tmp_path):
        """Galaxies outside deep regime get NaN residual."""
        df = pd.DataFrame({
            "galaxy_id": ["deep", "shallow"],
            "logM": [7.0, 8.0],
            "logVobs": [1.3, 1.6],
            "log_gbar": [-12.0, -9.0],   # -9.0 is above a0 → not reliable
            "log_j": [1.3, 1.6],
        })
        cat = build_catalog(df)
        cat = compute_mass_detrend(cat)
        # Only deep galaxy should have a finite residual
        shallow_resid = cat.loc[cat["galaxy_id"] == "shallow", "f3_mass_residual"].iloc[0]
        assert np.isnan(shallow_resid)


# ---------------------------------------------------------------------------
# compute_mass_correlation_stats
# ---------------------------------------------------------------------------


class TestComputeMassCorrelationStats:
    def test_keys_present(self, real_df):
        cat = build_catalog(real_df)
        stats = compute_mass_correlation_stats(cat)
        for key in [
            "spearman_f3_vlast_rho",
            "spearman_f3_vlast_p",
            "ols_slope",
            "ols_intercept",
            "spearman_resid_vlast_rho",
            "spearman_resid_vlast_p",
        ]:
            assert key in stats, f"Missing key: {key}"

    def test_raw_correlation_negative(self, real_df):
        """F3 should negatively correlate with logVobs for deep-MOND dwarfs."""
        cat = build_catalog(real_df)
        stats = compute_mass_correlation_stats(cat)
        assert stats["spearman_f3_vlast_rho"] < 0

    def test_raw_correlation_significant(self, real_df):
        """Raw p-value should be < 0.05 (significant mass dependence)."""
        cat = build_catalog(real_df)
        stats = compute_mass_correlation_stats(cat)
        assert stats["spearman_f3_vlast_p"] < 0.05

    def test_residual_correlation_nonsignificant(self, real_df):
        """After rank-detrending, residual correlation should not be significant."""
        cat = build_catalog(real_df)
        stats = compute_mass_correlation_stats(cat)
        assert stats["spearman_resid_vlast_p"] > 0.05

    def test_too_few_galaxies_returns_nan(self):
        """Fewer than 5 reliable galaxies → all correlation keys are NaN.
        Uses 4 total galaxies (all reliable) to test the n<5 boundary."""
        df = pd.DataFrame({
            "galaxy_id": ["G1", "G2", "G3", "G4"],
            "logM": [7.0, 7.5, 8.0, 8.5],
            "logVobs": [1.3, 1.4, 1.5, 1.6],
            "log_gbar": [-12.0, -11.8, -11.5, -11.2],
            "log_j": [1.3, 1.4, 1.5, 1.6],
        })
        cat = build_catalog(df)
        stats = compute_mass_correlation_stats(cat)
        assert np.isnan(stats["spearman_f3_vlast_rho"])

    def test_too_few_reliable_among_more_total(self):
        """7 total galaxies but only 4 reliable → still returns NaN."""
        df = pd.DataFrame({
            "galaxy_id": [f"G{i}" for i in range(7)],
            "logM": [7.0 + 0.3 * i for i in range(7)],
            "logVobs": [1.3 + 0.1 * i for i in range(7)],
            # First 4 deep (reliable), last 3 near Newtonian (not reliable)
            "log_gbar": [-12.0, -11.8, -11.5, -11.2, -9.5, -9.3, -9.1],
            "log_j": [1.3 + 0.1 * i for i in range(7)],
        })
        cat = build_catalog(df)
        assert cat["reliable"].sum() == 4
        stats = compute_mass_correlation_stats(cat)
        assert np.isnan(stats["spearman_f3_vlast_rho"])

    def test_exactly_five_galaxies_returns_finite(self):
        """Exactly 5 reliable galaxies should produce finite correlation values."""
        df = pd.DataFrame({
            "galaxy_id": [f"G{i}" for i in range(5)],
            "logM": [7.0 + 0.3 * i for i in range(5)],
            "logVobs": [1.3 + 0.1 * i for i in range(5)],
            "log_gbar": [-12.0 + 0.2 * i for i in range(5)],
            "log_j": [1.3 + 0.1 * i for i in range(5)],
        })
        cat = build_catalog(df)
        stats = compute_mass_correlation_stats(cat)
        assert not np.isnan(stats["spearman_f3_vlast_rho"])


# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_main_runs_and_returns_dict(self, tmp_path):
        result = main([
            "--csv", str(_DATASET),
            "--out", str(tmp_path),
            "--no-figures",
        ])
        assert isinstance(result, dict)
        assert "n_galaxies" in result

    def test_main_produces_files(self, tmp_path):
        main([
            "--csv", str(_DATASET),
            "--out", str(tmp_path),
            "--no-figures",
        ])
        assert (tmp_path / "little_things_scm_catalog.csv").exists()
        assert (tmp_path / "summary.json").exists()
