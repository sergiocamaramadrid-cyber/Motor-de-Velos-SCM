"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Validates the five audit checks (A–E) using a synthetic SPARC-like dataset
so no real download is required.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_scm import (
    A0_DEFAULT,
    _fit_v_flat,
    _fit_upsilon,
    _rmse,
    _v_baryonic,
    _v_pred_full,
    _v_pred_no_hinge,
    _evaluate_galaxy,
    _permute_baryonic_within_galaxies,
    build_coeffs_by_fold,
    coefficient_stability_stats,
    group_kfold_split,
    load_all_data,
    run_groupkfold_cv,
    run_permutation_test,
    run_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sparc_dir(tmp_path_factory):
    """Synthetic 12-galaxy SPARC-like dataset (fast, deterministic)."""
    root = tmp_path_factory.mktemp("audit_sparc")
    rng = np.random.default_rng(0)
    n_gal = 12
    names = [f"A{i:03d}" for i in range(n_gal)]
    v_flats = np.linspace(80.0, 280.0, n_gal)

    pd.DataFrame({
        "Galaxy": names,
        "D": np.linspace(5, 50, n_gal),
        "Inc": np.linspace(30, 80, n_gal),
        "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(n_gal, 5.0),
    }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 15
    for name, vf in zip(names, v_flats):
        r = np.linspace(0.5, 15.0, n_pts)
        rc = pd.DataFrame({
            "r": r,
            "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
            "v_obs_err": np.full(n_pts, 5.0),
            "v_gas": 0.3 * vf * np.ones(n_pts),
            "v_disk": 0.75 * vf * np.ones(n_pts),
            "v_bul": np.zeros(n_pts),
            "SBdisk": np.zeros(n_pts),
            "SBbul": np.zeros(n_pts),
        })
        rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

    return root


@pytest.fixture(scope="module")
def full_df(sparc_dir):
    """Loaded DataFrame for the synthetic dataset."""
    return load_all_data(sparc_dir)


@pytest.fixture(scope="module")
def cv_results(full_df):
    """GroupKFold CV results (per-point, per-galaxy)."""
    return run_groupkfold_cv(full_df, n_splits=3, a0=A0_DEFAULT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestRmse:
    def test_perfect_prediction_is_zero(self):
        v = np.array([100.0, 150.0, 200.0])
        assert _rmse(v, v) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self):
        v_obs = np.array([0.0, 0.0, 0.0])
        v_pred = np.array([3.0, 4.0, 0.0])
        # RMSE = sqrt((9+16+0)/3) = sqrt(25/3)
        expected = np.sqrt(25.0 / 3.0)
        assert _rmse(v_obs, v_pred) == pytest.approx(expected, rel=1e-6)


class TestFitVFlat:
    def test_returns_mean(self):
        v_obs = np.array([100.0, 120.0, 140.0])
        assert _fit_v_flat(v_obs) == pytest.approx(np.mean(v_obs))

    def test_scalar_input(self):
        v_obs = np.array([200.0])
        assert _fit_v_flat(v_obs) == pytest.approx(200.0)


class TestVBaryonic:
    def test_zero_velocities(self):
        v = _v_baryonic(
            np.zeros(5), np.zeros(5), np.zeros(5), upsilon_disk=1.0
        )
        np.testing.assert_allclose(v, np.zeros(5))

    def test_positive_upsilon_increases_v(self):
        v_gas = np.zeros(5)
        v_disk = np.full(5, 100.0)
        v_bul = np.zeros(5)
        v1 = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk=1.0)
        v2 = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk=2.0)
        assert np.all(v2 > v1)


class TestFitUpsilon:
    def test_result_in_bounds(self):
        rng = np.random.default_rng(1)
        n = 20
        r = np.linspace(0.5, 15.0, n)
        v_gas = 30.0 * np.ones(n)
        v_disk = 80.0 * np.ones(n)
        v_bul = np.zeros(n)
        v_obs = np.full(n, 150.0) + rng.normal(0, 3, n)
        pred_fn = lambda r_, vg_, vd_, vb_, ud_: _v_pred_no_hinge(
            r_, vg_, vd_, vb_, ud_
        )
        ud = _fit_upsilon(r, v_obs, v_gas, v_disk, v_bul, pred_fn)
        assert 0.1 <= ud <= 5.0


# ---------------------------------------------------------------------------
# A) GroupKFold splitter
# ---------------------------------------------------------------------------

class TestGroupKFoldSplit:
    def test_every_sample_in_exactly_one_test_fold(self):
        """Each sample index appears in test exactly once."""
        groups = ["G0"] * 5 + ["G1"] * 5 + ["G2"] * 5 + ["G3"] * 5
        n = len(groups)
        test_counts = np.zeros(n, dtype=int)
        for _, _train, test in group_kfold_split(groups, n_splits=4):
            test_counts[test] += 1
        np.testing.assert_array_equal(test_counts, np.ones(n, dtype=int))

    def test_train_test_disjoint(self):
        """No overlap between train and test index sets."""
        groups = [f"G{i}" for i in range(10) for _ in range(3)]
        for _, train, test in group_kfold_split(groups, n_splits=5):
            assert set(train).isdisjoint(set(test))

    def test_no_galaxy_in_both_train_and_test(self):
        """No galaxy name appears in both train and test groups."""
        groups = [f"G{i}" for i in range(8) for _ in range(4)]
        grp_arr = np.array(groups)
        for _, train, test in group_kfold_split(groups, n_splits=4):
            train_gals = set(grp_arr[train])
            test_gals = set(grp_arr[test])
            assert train_gals.isdisjoint(test_gals), (
                "GroupKFold violated: galaxy appears in both train and test"
            )

    def test_correct_number_of_folds(self):
        groups = [f"G{i}" for i in range(6) for _ in range(2)]
        folds = list(group_kfold_split(groups, n_splits=3))
        assert len(folds) == 3

    def test_fold_ids_sequential(self):
        groups = [f"G{i}" for i in range(6) for _ in range(2)]
        fold_ids = [f for f, _, _ in group_kfold_split(groups, n_splits=3)]
        assert fold_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# B) Per-galaxy metrics
# ---------------------------------------------------------------------------

class TestEvaluateGalaxy:
    @pytest.fixture()
    def single_galaxy_df(self):
        rng = np.random.default_rng(7)
        n = 20
        vf = 150.0
        r = np.linspace(0.5, 15.0, n)
        return pd.DataFrame({
            "galaxy": ["G0"] * n,
            "r": r,
            "v_obs": np.full(n, vf) + rng.normal(0, 3, n),
            "v_obs_err": np.full(n, 5.0),
            "v_gas": 0.3 * vf * np.ones(n),
            "v_disk": 0.75 * vf * np.ones(n),
            "v_bul": np.zeros(n),
        })

    def test_returns_all_keys(self, single_galaxy_df):
        metrics = _evaluate_galaxy(single_galaxy_df, a0=A0_DEFAULT)
        required = {
            "v_flat", "ud_no_hinge", "ud_full",
            "rmse_btfr", "rmse_no_hinge", "rmse_full",
            "delta_rmse_full_vs_btfr",
            "v_pred_btfr", "v_pred_no_hinge", "v_pred_full",
        }
        assert required.issubset(set(metrics.keys()))

    def test_rmse_non_negative(self, single_galaxy_df):
        metrics = _evaluate_galaxy(single_galaxy_df, a0=A0_DEFAULT)
        assert metrics["rmse_btfr"] >= 0.0
        assert metrics["rmse_no_hinge"] >= 0.0
        assert metrics["rmse_full"] >= 0.0

    def test_delta_equals_full_minus_btfr(self, single_galaxy_df):
        metrics = _evaluate_galaxy(single_galaxy_df, a0=A0_DEFAULT)
        expected = metrics["rmse_full"] - metrics["rmse_btfr"]
        assert metrics["delta_rmse_full_vs_btfr"] == pytest.approx(expected, abs=1e-9)

    def test_upsilon_in_bounds(self, single_galaxy_df):
        metrics = _evaluate_galaxy(single_galaxy_df, a0=A0_DEFAULT)
        assert 0.1 <= metrics["ud_no_hinge"] <= 5.0
        assert 0.1 <= metrics["ud_full"] <= 5.0

    def test_pred_arrays_correct_length(self, single_galaxy_df):
        metrics = _evaluate_galaxy(single_galaxy_df, a0=A0_DEFAULT)
        n = len(single_galaxy_df)
        assert len(metrics["v_pred_btfr"]) == n
        assert len(metrics["v_pred_no_hinge"]) == n
        assert len(metrics["v_pred_full"]) == n

    def test_btfr_pred_is_constant(self, single_galaxy_df):
        """BTFR baseline must be a constant (flat rotation curve)."""
        metrics = _evaluate_galaxy(single_galaxy_df, a0=A0_DEFAULT)
        assert np.all(metrics["v_pred_btfr"] == metrics["v_pred_btfr"][0])


class TestRunGroupKFoldCV:
    def test_returns_two_dataframes(self, cv_results):
        per_point, per_galaxy = cv_results
        assert isinstance(per_point, pd.DataFrame)
        assert isinstance(per_galaxy, pd.DataFrame)

    def test_per_point_has_required_columns(self, cv_results):
        per_point, _ = cv_results
        required = {
            "galaxy", "fold", "r_kpc", "v_obs",
            "v_pred_btfr", "v_pred_no_hinge", "v_pred_full",
            "residual_btfr", "residual_no_hinge", "residual_full",
        }
        assert required.issubset(per_point.columns)

    def test_per_galaxy_has_required_columns(self, cv_results):
        _, per_galaxy = cv_results
        required = {
            "galaxy", "fold", "n_points",
            "rmse_btfr", "rmse_no_hinge", "rmse_full",
            "delta_rmse_full_vs_btfr",
        }
        assert required.issubset(per_galaxy.columns)

    def test_per_galaxy_one_row_per_galaxy(self, cv_results, full_df):
        _, per_galaxy = cv_results
        n_galaxies = full_df["galaxy"].nunique()
        assert len(per_galaxy) == n_galaxies

    def test_per_point_row_count_matches_full_dataset(self, cv_results, full_df):
        per_point, _ = cv_results
        assert len(per_point) == len(full_df)

    def test_residuals_consistent(self, cv_results):
        """residual_btfr must equal v_obs − v_pred_btfr."""
        per_point, _ = cv_results
        diff = per_point["v_obs"] - per_point["v_pred_btfr"]
        np.testing.assert_allclose(
            per_point["residual_btfr"].values, diff.values, atol=1e-9
        )

    def test_no_galaxy_in_train_and_test(self, full_df):
        """Verify no galaxy leaks between train and test in any fold."""
        groups = full_df["galaxy"].tolist()
        grp_arr = np.array(groups)
        for _, train_idx, test_idx in group_kfold_split(groups, n_splits=3):
            train_gals = set(grp_arr[train_idx])
            test_gals = set(grp_arr[test_idx])
            assert train_gals.isdisjoint(test_gals)

    def test_rmse_values_non_negative(self, cv_results):
        _, per_galaxy = cv_results
        assert (per_galaxy["rmse_btfr"] >= 0.0).all()
        assert (per_galaxy["rmse_no_hinge"] >= 0.0).all()
        assert (per_galaxy["rmse_full"] >= 0.0).all()

    def test_delta_equals_full_minus_btfr(self, cv_results):
        _, per_galaxy = cv_results
        expected = per_galaxy["rmse_full"] - per_galaxy["rmse_btfr"]
        np.testing.assert_allclose(
            per_galaxy["delta_rmse_full_vs_btfr"].values, expected.values, atol=1e-9
        )


# ---------------------------------------------------------------------------
# C) Permutation test
# ---------------------------------------------------------------------------

class TestPermuteBaryonic:
    def test_r_and_v_obs_unchanged(self, full_df):
        rng = np.random.default_rng(0)
        permuted = _permute_baryonic_within_galaxies(full_df, rng)
        pd.testing.assert_series_equal(
            permuted["r"].reset_index(drop=True),
            full_df["r"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            permuted["v_obs"].reset_index(drop=True),
            full_df["v_obs"].reset_index(drop=True),
        )

    def test_baryonic_columns_permuted(self):
        """At least one baryonic column must differ after permutation when
        within-galaxy v_disk values are not all identical."""
        rng = np.random.default_rng(99)
        # Build a galaxy with non-constant v_disk so permutation is visible
        n = 10
        df_varying = pd.DataFrame({
            "galaxy": ["G0"] * n,
            "r": np.linspace(0.5, 10.0, n),
            "v_obs": np.full(n, 150.0),
            "v_obs_err": np.full(n, 5.0),
            "v_gas": np.linspace(10.0, 50.0, n),   # deliberately varied
            "v_disk": np.linspace(80.0, 120.0, n),  # deliberately varied
            "v_bul": np.zeros(n),
        })
        permuted = _permute_baryonic_within_galaxies(df_varying, rng)
        changed = not np.allclose(
            df_varying["v_disk"].values, permuted["v_disk"].values
        )
        assert changed, "v_disk unchanged after permutation (expected shuffle of varied values)"

    def test_baryonic_set_preserved_per_galaxy(self, full_df):
        """Each galaxy's baryonic values are the same set, just reordered."""
        rng = np.random.default_rng(5)
        permuted = _permute_baryonic_within_galaxies(full_df, rng)
        for galaxy in full_df["galaxy"].unique():
            orig = full_df[full_df["galaxy"] == galaxy]["v_disk"].values
            perm = permuted[permuted["galaxy"] == galaxy]["v_disk"].values
            np.testing.assert_allclose(
                sorted(orig), sorted(perm),
                err_msg=f"v_disk values changed for galaxy {galaxy}"
            )


class TestRunPermutationTest:
    def test_returns_correct_types(self, full_df):
        rmse_real, perm_rmse, p_value, perm_df = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=5, rng_seed=0
        )
        assert isinstance(rmse_real, float)
        assert isinstance(perm_rmse, np.ndarray)
        assert isinstance(p_value, float)
        assert isinstance(perm_df, pd.DataFrame)

    def test_p_value_in_unit_interval(self, full_df):
        _, _, p_value, _ = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=5, rng_seed=1
        )
        assert 0.0 < p_value <= 1.0

    def test_perm_distribution_length(self, full_df):
        n_perm = 8
        _, perm_rmse, _, perm_df = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=n_perm, rng_seed=2
        )
        assert len(perm_rmse) == n_perm
        assert len(perm_df) == n_perm

    def test_perm_df_has_required_columns(self, full_df):
        _, _, _, perm_df = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=4, rng_seed=3
        )
        assert "perm_id" in perm_df.columns
        assert "rmse_full" in perm_df.columns

    def test_p_value_formula(self, full_df):
        """p = (1 + Σ[perm≤real]) / (n_perm+1)."""
        n_perm = 10
        rmse_real, perm_rmse, p_value, _ = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=n_perm, rng_seed=7
        )
        expected_p = (1.0 + np.sum(perm_rmse <= rmse_real)) / (n_perm + 1)
        assert p_value == pytest.approx(expected_p, rel=1e-9)

    def test_rmse_real_is_positive(self, full_df):
        rmse_real, _, _, _ = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=3, rng_seed=4
        )
        assert rmse_real > 0.0

    def test_perm_rmse_all_finite(self, full_df):
        _, perm_rmse, _, _ = run_permutation_test(
            full_df, n_splits=3, a0=A0_DEFAULT, n_perm=5, rng_seed=5
        )
        assert np.all(np.isfinite(perm_rmse))


# ---------------------------------------------------------------------------
# D) Coefficient stability
# ---------------------------------------------------------------------------

class TestCoefficientStability:
    def test_build_coeffs_has_required_columns(self, cv_results):
        _, per_galaxy = cv_results
        coeffs = build_coeffs_by_fold(per_galaxy)
        assert "fold" in coeffs.columns
        assert "galaxy" in coeffs.columns
        assert "upsilon_disk_no_hinge" in coeffs.columns
        assert "upsilon_disk_full" in coeffs.columns

    def test_build_coeffs_row_count(self, cv_results, full_df):
        _, per_galaxy = cv_results
        coeffs = build_coeffs_by_fold(per_galaxy)
        n_galaxies = full_df["galaxy"].nunique()
        assert len(coeffs) == n_galaxies

    def test_stability_stats_keys(self, cv_results):
        _, per_galaxy = cv_results
        coeffs = build_coeffs_by_fold(per_galaxy)
        stats = coefficient_stability_stats(coeffs)
        required = {
            "ud_no_hinge_mean", "ud_no_hinge_std", "ud_no_hinge_range",
            "ud_full_mean", "ud_full_std", "ud_full_range",
        }
        assert required.issubset(stats.keys())

    def test_stability_stats_finite(self, cv_results):
        _, per_galaxy = cv_results
        coeffs = build_coeffs_by_fold(per_galaxy)
        stats = coefficient_stability_stats(coeffs)
        for key, val in stats.items():
            assert np.isfinite(val), f"{key} is not finite"

    def test_stability_stats_non_negative_std(self, cv_results):
        _, per_galaxy = cv_results
        coeffs = build_coeffs_by_fold(per_galaxy)
        stats = coefficient_stability_stats(coeffs)
        assert stats["ud_no_hinge_std"] >= 0.0
        assert stats["ud_full_std"] >= 0.0

    def test_stability_stats_non_negative_range(self, cv_results):
        _, per_galaxy = cv_results
        coeffs = build_coeffs_by_fold(per_galaxy)
        stats = coefficient_stability_stats(coeffs)
        assert stats["ud_no_hinge_range"] >= 0.0
        assert stats["ud_full_range"] >= 0.0


# ---------------------------------------------------------------------------
# E) Output artefacts — run_audit
# ---------------------------------------------------------------------------

class TestRunAudit:
    @pytest.fixture(scope="class")
    def audit_result(self, sparc_dir, tmp_path_factory):
        out = tmp_path_factory.mktemp("audit_out")
        return run_audit(
            data_dir=sparc_dir,
            out_dir=out,
            n_splits=3,
            a0=A0_DEFAULT,
            n_perm=5,
            rng_seed=0,
            verbose=False,
        ), out

    def test_returns_dict(self, audit_result):
        result, _ = audit_result
        assert isinstance(result, dict)

    def test_per_point_csv_written(self, audit_result):
        result, out = audit_result
        assert (out / "groupkfold_per_point.csv").exists()

    def test_per_galaxy_csv_written(self, audit_result):
        result, out = audit_result
        assert (out / "groupkfold_per_galaxy.csv").exists()

    def test_permutation_distribution_csv_written(self, audit_result):
        result, out = audit_result
        assert (out / "permutation_distribution.csv").exists()

    def test_coeffs_by_fold_csv_written(self, audit_result):
        result, out = audit_result
        assert (out / "coeffs_by_fold.csv").exists()

    def test_audit_report_written(self, audit_result):
        result, out = audit_result
        assert (out / "audit_report.txt").exists()

    def test_per_galaxy_csv_has_required_columns(self, audit_result):
        _, out = audit_result
        df = pd.read_csv(out / "groupkfold_per_galaxy.csv")
        for col in ("rmse_btfr", "rmse_no_hinge", "rmse_full",
                    "delta_rmse_full_vs_btfr"):
            assert col in df.columns, f"Missing column: {col}"

    def test_per_point_csv_has_required_columns(self, audit_result):
        _, out = audit_result
        df = pd.read_csv(out / "groupkfold_per_point.csv")
        for col in ("galaxy", "fold", "r_kpc", "v_obs",
                    "v_pred_btfr", "v_pred_no_hinge", "v_pred_full"):
            assert col in df.columns, f"Missing column: {col}"

    def test_permutation_distribution_has_correct_length(self, audit_result):
        _, out = audit_result
        perm_df = pd.read_csv(out / "permutation_distribution.csv")
        assert len(perm_df) == 5  # n_perm=5

    def test_result_dict_has_rmse_real(self, audit_result):
        result, _ = audit_result
        assert "rmse_real" in result
        assert result["rmse_real"] > 0.0

    def test_result_dict_has_p_value(self, audit_result):
        result, _ = audit_result
        assert "p_value" in result
        assert 0.0 < result["p_value"] <= 1.0

    def test_result_dict_has_coeff_stats(self, audit_result):
        result, _ = audit_result
        assert "coeff_stats" in result
        stats = result["coeff_stats"]
        assert "ud_full_mean" in stats
        assert "ud_full_std" in stats
        assert "ud_full_range" in stats

    def test_audit_report_mentions_a0_fixed(self, audit_result):
        _, out = audit_result
        text = (out / "audit_report.txt").read_text(encoding="utf-8")
        assert "FIXED" in text

    def test_audit_report_mentions_p_value(self, audit_result):
        _, out = audit_result
        text = (out / "audit_report.txt").read_text(encoding="utf-8")
        assert "p-value" in text

    def test_out_dir_in_results_audit_by_default(self, sparc_dir, tmp_path):
        """Default out_dir sub-path is inside user-specified path."""
        out = tmp_path / "myresults" / "audit"
        run_audit(
            data_dir=sparc_dir,
            out_dir=out,
            n_splits=3,
            a0=A0_DEFAULT,
            n_perm=3,
            rng_seed=0,
            verbose=False,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestLoadAllData:
    def test_returns_dataframe(self, sparc_dir):
        df = load_all_data(sparc_dir)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sparc_dir):
        df = load_all_data(sparc_dir)
        for col in ("galaxy", "r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"):
            assert col in df.columns

    def test_galaxy_column_populated(self, sparc_dir):
        df = load_all_data(sparc_dir)
        assert df["galaxy"].notna().all()
        assert df["galaxy"].nunique() == 12

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_data(tmp_path / "nonexistent")

    def test_fallback_no_table(self, tmp_path):
        """No galaxy table: should fall back to scanning *_rotmod.dat files."""
        rng = np.random.default_rng(0)
        n = 10
        r = np.linspace(0.5, 10.0, n)
        rc = pd.DataFrame({
            "r": r,
            "v_obs": np.full(n, 100.0),
            "v_obs_err": np.full(n, 5.0),
            "v_gas": 30.0 * np.ones(n),
            "v_disk": 80.0 * np.ones(n),
            "v_bul": np.zeros(n),
            "SBdisk": np.zeros(n),
            "SBbul": np.zeros(n),
        })
        rc.to_csv(tmp_path / "NGC9999_rotmod.dat", sep=" ", index=False, header=False)
        df = load_all_data(tmp_path)
        assert "galaxy" in df.columns
        assert "NGC9999" in df["galaxy"].values
