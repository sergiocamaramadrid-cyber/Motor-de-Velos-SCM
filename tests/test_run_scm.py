"""
tests/test_run_scm.py — Test suite for scripts/run_scm.py.

Tests use synthetic DataFrames so that no real data download is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_scm import (
    load_catalog,
    split_by_mass,
    compute_ols,
    compute_spearman,
    bootstrap_ols,
    permutation_test,
    ridge_regression,
    control_regression,
    mass_threshold_scan,
    generate_figure,
    format_report,
    main,
    MASS_THRESHOLD_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_catalog(
    n_high: int = 20,
    n_low: int = 10,
    seed: int = 0,
    coeff: float = -0.15,
) -> pd.DataFrame:
    """Build a small synthetic catalog for testing."""
    rng = np.random.default_rng(seed)
    total = n_high + n_low

    logM_hi = rng.uniform(10.5, 11.5, n_high)
    logM_lo = rng.uniform(9.0, 10.4, n_low)
    logM = np.concatenate([logM_hi, logM_lo])

    env = rng.uniform(0.5, 3.5, total)
    F3 = coeff * env + 0.45 + rng.normal(0, 0.25, total)

    return pd.DataFrame(
        {
            "galaxy": [f"G{i:04d}" for i in range(total)],
            "logM": logM,
            "logMbar": logM + rng.normal(0.1, 0.05, total),
            "F3_SCM": F3,
            "env_proxy": env,
            "F_gas": rng.beta(2, 5, total),
        }
    )


@pytest.fixture()
def catalog_df():
    return _make_catalog(n_high=30, n_low=15, seed=1)


@pytest.fixture()
def high_df(catalog_df):
    hi, _ = split_by_mass(catalog_df)
    return hi


@pytest.fixture()
def catalog_csv(tmp_path, catalog_df):
    path = tmp_path / "test_catalog.csv"
    catalog_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_catalog
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_loads_from_csv(self, catalog_csv, catalog_df):
        df = load_catalog(catalog_csv)
        assert len(df) == len(catalog_df)
        for col in ("logM", "logMbar", "F3_SCM", "env_proxy"):
            assert col in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "no_such_file.csv")

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"logM": [10.5], "F3_SCM": [0.3]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(p)

    def test_nan_rows_dropped(self, tmp_path, catalog_df):
        df = catalog_df.copy()
        df.loc[0, "F3_SCM"] = float("nan")
        p = tmp_path / "nan_catalog.csv"
        df.to_csv(p, index=False)
        loaded = load_catalog(p)
        assert len(loaded) == len(catalog_df) - 1

    def test_returns_dataframe(self, catalog_csv):
        df = load_catalog(catalog_csv)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# split_by_mass
# ---------------------------------------------------------------------------


class TestSplitByMass:
    def test_default_threshold(self, catalog_df):
        hi, lo = split_by_mass(catalog_df)
        assert (hi["logM"] >= MASS_THRESHOLD_DEFAULT).all()
        assert (lo["logM"] < MASS_THRESHOLD_DEFAULT).all()

    def test_custom_threshold(self, catalog_df):
        hi, lo = split_by_mass(catalog_df, threshold=10.0)
        assert (hi["logM"] >= 10.0).all()
        assert (lo["logM"] < 10.0).all()

    def test_counts_sum_to_total(self, catalog_df):
        hi, lo = split_by_mass(catalog_df)
        assert len(hi) + len(lo) == len(catalog_df)

    def test_high_mass_count(self):
        df = _make_catalog(n_high=20, n_low=10, seed=99)
        hi, lo = split_by_mass(df)
        assert len(hi) == 20
        assert len(lo) == 10

    def test_empty_high(self):
        df = _make_catalog(n_high=0, n_low=10, seed=2)
        hi, lo = split_by_mass(df)
        assert len(hi) == 0
        assert len(lo) == 10

    def test_resets_index(self, catalog_df):
        hi, lo = split_by_mass(catalog_df)
        assert hi.index.tolist() == list(range(len(hi)))
        assert lo.index.tolist() == list(range(len(lo)))


# ---------------------------------------------------------------------------
# compute_ols
# ---------------------------------------------------------------------------


class TestComputeOLS:
    def test_returns_required_keys(self, high_df):
        result = compute_ols(high_df)
        for key in ("coeff", "intercept", "se", "t_stat", "p_value",
                    "r_squared", "n"):
            assert key in result

    def test_coeff_negative_for_negative_data(self):
        x = np.linspace(0.5, 3.5, 40)
        y = -0.15 * x + 0.45
        df = pd.DataFrame({"env_proxy": x, "F3_SCM": y})
        res = compute_ols(df)
        assert res["coeff"] < 0

    def test_n_matches_input(self, high_df):
        res = compute_ols(high_df)
        assert res["n"] == len(high_df)

    def test_r_squared_range(self, high_df):
        res = compute_ols(high_df)
        assert 0.0 <= res["r_squared"] <= 1.0

    def test_p_value_range(self, high_df):
        res = compute_ols(high_df)
        assert 0.0 <= res["p_value"] <= 1.0

    def test_custom_columns(self, catalog_df):
        catalog_df = catalog_df.rename(
            columns={"env_proxy": "x_var", "F3_SCM": "y_var"}
        )
        res = compute_ols(catalog_df, x_col="x_var", y_col="y_var")
        assert "coeff" in res


# ---------------------------------------------------------------------------
# compute_spearman
# ---------------------------------------------------------------------------


class TestComputeSpearman:
    def test_returns_rho_and_p(self, high_df):
        res = compute_spearman(high_df)
        assert "rho" in res and "p_value" in res

    def test_rho_in_range(self, high_df):
        res = compute_spearman(high_df)
        assert -1.0 <= res["rho"] <= 1.0

    def test_rho_sign_correct(self):
        x = np.linspace(1, 5, 30)
        y = -x + np.random.default_rng(0).normal(0, 0.1, 30)
        df = pd.DataFrame({"env_proxy": x, "F3_SCM": y})
        res = compute_spearman(df)
        assert res["rho"] < 0


# ---------------------------------------------------------------------------
# bootstrap_ols
# ---------------------------------------------------------------------------


class TestBootstrapOLS:
    def test_returns_required_keys(self, high_df):
        res = bootstrap_ols(high_df, n_boot=100, seed=0)
        for k in ("coeff_mean", "coeff_std", "ci_low", "ci_high",
                  "n_boot", "seed"):
            assert k in res

    def test_ci_ordered(self, high_df):
        res = bootstrap_ols(high_df, n_boot=200, seed=1)
        assert res["ci_low"] < res["ci_high"]

    def test_n_boot_stored(self, high_df):
        res = bootstrap_ols(high_df, n_boot=77, seed=2)
        assert res["n_boot"] == 77

    def test_seed_stored(self, high_df):
        res = bootstrap_ols(high_df, n_boot=50, seed=999)
        assert res["seed"] == 999

    def test_reproducible_with_same_seed(self, high_df):
        r1 = bootstrap_ols(high_df, n_boot=200, seed=7)
        r2 = bootstrap_ols(high_df, n_boot=200, seed=7)
        assert r1["ci_low"] == r2["ci_low"]
        assert r1["ci_high"] == r2["ci_high"]

    def test_different_seeds_differ(self, high_df):
        r1 = bootstrap_ols(high_df, n_boot=200, seed=0)
        r2 = bootstrap_ols(high_df, n_boot=200, seed=1)
        # Very unlikely to match exactly
        assert r1["ci_low"] != r2["ci_low"] or r1["ci_high"] != r2["ci_high"]

    def test_ci_near_true_coeff(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.5, 3.5, 60)
        y = -0.15 * x + 0.5 + rng.normal(0, 0.1, 60)
        df = pd.DataFrame({"env_proxy": x, "F3_SCM": y})
        res = bootstrap_ols(df, n_boot=2000, seed=0)
        assert res["ci_low"] < -0.15 < res["ci_high"]


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_returns_required_keys(self, high_df):
        res = permutation_test(high_df, n_perm=200, seed=0)
        for k in ("observed_coeff", "p_value", "n_perm", "seed"):
            assert k in res

    def test_p_value_range(self, high_df):
        res = permutation_test(high_df, n_perm=200, seed=0)
        assert 0.0 < res["p_value"] <= 1.0

    def test_null_data_high_p(self):
        rng = np.random.default_rng(10)
        x = rng.uniform(0, 3, 40)
        y = rng.normal(0, 1, 40)
        df = pd.DataFrame({"env_proxy": x, "F3_SCM": y})
        res = permutation_test(df, n_perm=500, seed=0)
        assert res["p_value"] > 0.05

    def test_strong_signal_low_p(self):
        x = np.linspace(0.5, 3.5, 50)
        y = -0.5 * x + rng_val.normal(0, 0.02, 50) if False else (
            -0.5 * x + np.random.default_rng(0).normal(0, 0.02, 50)
        )
        df = pd.DataFrame({"env_proxy": x, "F3_SCM": y})
        res = permutation_test(df, n_perm=500, seed=0)
        assert res["p_value"] < 0.05

    def test_n_perm_stored(self, high_df):
        res = permutation_test(high_df, n_perm=123, seed=0)
        assert res["n_perm"] == 123

    def test_observed_coeff_matches_ols(self, high_df):
        ols = compute_ols(high_df)
        perm = permutation_test(high_df, n_perm=50, seed=0)
        assert abs(perm["observed_coeff"] - ols["coeff"]) < 1e-12


# ---------------------------------------------------------------------------
# ridge_regression
# ---------------------------------------------------------------------------


class TestRidgeRegression:
    def test_returns_scan_and_alpha1(self, high_df):
        res = ridge_regression(high_df)
        assert "scan" in res
        assert "coeff_at_alpha_1" in res

    def test_scan_length(self, high_df):
        from run_scm import RIDGE_ALPHAS_DEFAULT
        res = ridge_regression(high_df)
        assert len(res["scan"]) == len(RIDGE_ALPHAS_DEFAULT)

    def test_coeffs_shrink_with_alpha(self, high_df):
        res = ridge_regression(high_df)
        coeffs = [abs(entry["coeff"]) for entry in res["scan"]]
        # Higher alpha → smaller |coeff|
        assert coeffs[0] >= coeffs[-1]

    def test_all_negative_for_negative_data(self, high_df):
        res = ridge_regression(high_df)
        for entry in res["scan"]:
            assert entry["coeff"] < 0 or abs(entry["coeff"]) < 0.01

    def test_custom_alphas(self, high_df):
        res = ridge_regression(high_df, alphas=(0.5, 5.0))
        assert len(res["scan"]) == 2
        assert res["scan"][0]["alpha"] == 0.5


# ---------------------------------------------------------------------------
# control_regression
# ---------------------------------------------------------------------------


class TestControlRegression:
    def test_returns_required_keys(self, high_df):
        res = control_regression(high_df, control_col="logMbar")
        for k in ("coeff_env", "coeff_control", "intercept",
                  "p_env", "p_control", "r_squared", "n", "control_col"):
            assert k in res

    def test_control_col_stored(self, high_df):
        res = control_regression(high_df, control_col="logMbar")
        assert res["control_col"] == "logMbar"

    def test_n_matches(self, high_df):
        res = control_regression(high_df, control_col="logMbar")
        assert res["n"] == len(high_df)

    def test_r_squared_range(self, high_df):
        res = control_regression(high_df, control_col="logMbar")
        assert 0.0 <= res["r_squared"] <= 1.0

    def test_p_values_range(self, high_df):
        res = control_regression(high_df, control_col="logMbar")
        assert 0.0 <= res["p_env"] <= 1.0
        assert 0.0 <= res["p_control"] <= 1.0

    def test_gas_control(self, high_df):
        res = control_regression(high_df, control_col="F_gas")
        assert res["control_col"] == "F_gas"
        assert "coeff_env" in res

    def test_too_few_rows(self):
        df = pd.DataFrame({
            "env_proxy": [1.0],
            "logMbar": [10.5],
            "F3_SCM": [0.3],
        })
        res = control_regression(df, control_col="logMbar")
        assert np.isnan(res["coeff_env"])


# ---------------------------------------------------------------------------
# mass_threshold_scan
# ---------------------------------------------------------------------------


class TestMassThresholdScan:
    def test_returns_dataframe(self, catalog_df):
        result = mass_threshold_scan(catalog_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, catalog_df):
        result = mass_threshold_scan(catalog_df)
        for col in ("threshold", "n", "coeff", "p_value"):
            assert col in result.columns

    def test_thresholds_monotone(self, catalog_df):
        result = mass_threshold_scan(catalog_df)
        thresholds = result["threshold"].tolist()
        assert thresholds == sorted(thresholds)

    def test_n_decreases_with_threshold(self, catalog_df):
        result = mass_threshold_scan(catalog_df)
        ns = result["n"].tolist()
        # n should be non-increasing
        for i in range(1, len(ns)):
            assert ns[i] <= ns[i - 1]

    def test_custom_scan_range(self, catalog_df):
        result = mass_threshold_scan(
            catalog_df, scan_min=10.5, scan_max=11.0, scan_step=0.25
        )
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# generate_figure
# ---------------------------------------------------------------------------


class TestGenerateFigure:
    def test_creates_png_and_pdf(self, tmp_path, high_df):
        ols = compute_ols(high_df)
        boot = bootstrap_ols(high_df, n_boot=50, seed=0)
        sp = compute_spearman(high_df)
        out = tmp_path / "test_fig.png"
        generate_figure(
            df=high_df,
            x_col="env_proxy",
            y_col="F3_SCM",
            coeff=ols["coeff"],
            intercept=ols["intercept"],
            ci_low=boot["ci_low"],
            ci_high=boot["ci_high"],
            spearman_rho=sp["rho"],
            p_value=ols["p_value"],
            out_path=out,
        )
        assert out.exists()
        assert out.with_suffix(".pdf").exists()

    def test_figure_files_non_empty(self, tmp_path, high_df):
        ols = compute_ols(high_df)
        boot = bootstrap_ols(high_df, n_boot=50, seed=0)
        sp = compute_spearman(high_df)
        out = tmp_path / "fig.png"
        generate_figure(
            df=high_df,
            x_col="env_proxy",
            y_col="F3_SCM",
            coeff=ols["coeff"],
            intercept=ols["intercept"],
            ci_low=boot["ci_low"],
            ci_high=boot["ci_high"],
            spearman_rho=sp["rho"],
            p_value=ols["p_value"],
            out_path=out,
        )
        assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def _make_results(self):
        df = _make_catalog(n_high=30, n_low=15, seed=5)
        hi, lo = split_by_mass(df)
        ols = compute_ols(hi)
        boot = bootstrap_ols(hi, n_boot=100, seed=0)
        perm = permutation_test(hi, n_perm=100, seed=0)
        sp = compute_spearman(hi)
        ctrl_mass = control_regression(hi, control_col="logMbar")
        ctrl_gas = control_regression(hi, control_col="F_gas")
        return {
            "catalog_path": "test_catalog.csv",
            "mass_threshold": MASS_THRESHOLD_DEFAULT,
            "n_total": len(df),
            "n_high_mass": len(hi),
            "n_low_mass": len(lo),
            "ols": ols,
            "bootstrap": boot,
            "permutation": perm,
            "ridge": ridge_regression(hi),
            "spearman": sp,
            "control_mass": ctrl_mass,
            "control_gas": ctrl_gas,
        }

    def test_returns_string(self):
        res = self._make_results()
        report = format_report(res)
        assert isinstance(report, str)

    def test_contains_catalog_path(self):
        res = self._make_results()
        report = format_report(res)
        assert "test_catalog.csv" in report

    def test_contains_ols_coeff(self):
        res = self._make_results()
        report = format_report(res)
        assert "β_env" in report

    def test_contains_bootstrap_ci(self):
        res = self._make_results()
        report = format_report(res)
        assert "IC95" in report

    def test_contains_spearman(self):
        res = self._make_results()
        report = format_report(res)
        assert "Spearman" in report

    def test_gas_control_included_when_present(self):
        res = self._make_results()
        report = format_report(res)
        assert "F_gas" in report


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_dict(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        res = main(["--catalog", str(catalog_csv), "--out", str(out),
                    "--n-boot", "50", "--n-perm", "100"])
        assert isinstance(res, dict)

    def test_creates_json(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        assert (out / "scm_summary.json").exists()

    def test_creates_txt(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        assert (out / "scm_summary.txt").exists()

    def test_creates_mass_scan_csv(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        assert (out / "mass_scan.csv").exists()

    def test_creates_figure_png(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        assert (out / "env_slope_scatter.png").exists()

    def test_creates_figure_pdf(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        assert (out / "env_slope_scatter.pdf").exists()

    def test_json_has_ols_keys(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        data = json.loads((out / "scm_summary.json").read_text())
        assert "ols" in data
        assert "coeff" in data["ols"]

    def test_json_has_bootstrap(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        data = json.loads((out / "scm_summary.json").read_text())
        assert "bootstrap" in data
        assert "ci_low" in data["bootstrap"]
        assert "ci_high" in data["bootstrap"]

    def test_json_has_permutation(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        data = json.loads((out / "scm_summary.json").read_text())
        assert "permutation" in data

    def test_json_has_ridge(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        data = json.loads((out / "scm_summary.json").read_text())
        assert "ridge" in data

    def test_json_has_controls(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out"
        main(["--catalog", str(catalog_csv), "--out", str(out),
              "--n-boot", "50", "--n-perm", "100"])
        data = json.loads((out / "scm_summary.json").read_text())
        assert "control_mass" in data
        assert "control_gas" in data

    def test_n_high_mass_correct(self, catalog_csv, tmp_path, catalog_df):
        out = tmp_path / "scm_out"
        res = main(["--catalog", str(catalog_csv), "--out", str(out),
                    "--n-boot", "50", "--n-perm", "100"])
        expected_hi = int((catalog_df["logM"] >= MASS_THRESHOLD_DEFAULT).sum())
        assert res["n_high_mass"] == expected_hi

    def test_custom_threshold(self, catalog_csv, tmp_path):
        out = tmp_path / "scm_out2"
        res = main(["--catalog", str(catalog_csv), "--out", str(out),
                    "--threshold", "10.0",
                    "--n-boot", "50", "--n-perm", "100"])
        assert res["mass_threshold"] == 10.0

    def test_on_real_catalog(self, tmp_path):
        """Test on the actual repository data file (data/galaxy_catalog_with_env.csv)."""
        repo_root = Path(__file__).parent.parent
        catalog = repo_root / "data" / "galaxy_catalog_with_env.csv"
        if not catalog.exists():
            pytest.skip("galaxy_catalog_with_env.csv not present")
        out = tmp_path / "scm_real"
        res = main(["--catalog", str(catalog), "--out", str(out),
                    "--n-boot", "200", "--n-perm", "500"])
        assert res["n_high_mass"] == 47
        assert res["n_low_mass"] == 32
        assert res["ols"]["coeff"] < 0
        assert res["ols"]["p_value"] < 0.05

    def test_missing_catalog_raises(self, tmp_path):
        out = tmp_path / "scm_err"
        with pytest.raises(FileNotFoundError):
            main(["--catalog", str(tmp_path / "nope.csv"),
                  "--out", str(out)])
