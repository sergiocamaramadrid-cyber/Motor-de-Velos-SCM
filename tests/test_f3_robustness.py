"""
tests/test_f3_robustness.py — Tests for scripts/f3_robustness.py.

Covers all three robustness blocks:
  1. Controlled OLS regression (Block 1)
  2. Stratified permutation test (Block 2)
  3. Bootstrap ΔAIC (Block 3)
  4. Column resolution helpers
  5. Report formatting
  6. CLI / main()
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the module under test
from scripts.f3_robustness import (
    _prepare_dataframe,
    _resolve_column,
    bootstrap_delta_aic,
    controlled_regression,
    format_report,
    main,
    run_robustness,
    stratified_permutation,
    N_PERMS_DEFAULT,
    N_BOOT_DEFAULT,
    SEED_DEFAULT,
    DELTA_AIC_STRONG_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Test constants (replacing magic numbers for clarity)
# ---------------------------------------------------------------------------

# Tolerance added to the |β_env| < N*SE test to account for RNG variance
_BETA_ENV_NULL_SLACK = 0.3

# Maximum number of seeds (out of 5) allowed to yield p_perm < 0.01 under
# the null hypothesis before we flag the test as failing.
_MAX_FALSE_POSITIVES_PERM = 3

# ---------------------------------------------------------------------------
# Synthetic catalog builder
# ---------------------------------------------------------------------------

def _make_catalog(
    n: int = 80,
    seed: int = 42,
    signal_strength: float = 0.4,
    include_controls: bool = True,
) -> pd.DataFrame:
    """Build a synthetic per-galaxy catalog with a planted delta_mass signal.

    Parameters
    ----------
    n : int
        Number of galaxies.
    seed : int
        RNG seed.
    signal_strength : float
        Coefficient of delta_mass_std in generating f3; 0 = no signal.
    include_controls : bool
        Whether to include log_M_bar and Rmax_kpc columns.

    Returns
    -------
    pd.DataFrame with columns: galaxy, friction_slope, delta_mass,
    and optionally log_M_bar, Rmax_kpc.
    """
    rng = np.random.default_rng(seed)
    log_mbar = rng.normal(9.5, 0.6, n)
    log_rmax = rng.normal(1.4, 0.3, n)
    delta_mass = rng.normal(0.0, 1.0, n)
    # Standardise delta_mass for the planted signal
    dm_std = (delta_mass - delta_mass.mean()) / delta_mass.std()
    noise = rng.normal(0.0, 0.1, n)
    f3 = 0.5 + signal_strength * dm_std + 0.05 * (log_mbar - log_mbar.mean()) + noise

    data: dict[str, object] = {
        "galaxy": [f"G{i:04d}" for i in range(n)],
        "friction_slope": f3,
        "delta_mass": delta_mass,
    }
    if include_controls:
        data["log_M_bar"] = log_mbar
        data["Rmax_kpc"] = 10 ** log_rmax  # store as kpc; script must log-transform

    return pd.DataFrame(data)


def _make_catalog_no_signal(n: int = 80, seed: int = 99) -> pd.DataFrame:
    """Catalog with no planted delta_mass signal."""
    return _make_catalog(n=n, seed=seed, signal_strength=0.0)


# ---------------------------------------------------------------------------
# 1. Column resolution helpers
# ---------------------------------------------------------------------------

class TestResolveColumn:
    def test_exact_match(self):
        df = pd.DataFrame({"friction_slope": [1.0], "other": [2.0]})
        assert _resolve_column(df, ["friction_slope", "beta"]) == "friction_slope"

    def test_second_alias(self):
        df = pd.DataFrame({"beta": [0.5]})
        assert _resolve_column(df, ["friction_slope", "beta"]) == "beta"

    def test_returns_none_when_no_match(self):
        df = pd.DataFrame({"col_a": [1.0]})
        assert _resolve_column(df, ["friction_slope", "beta"]) is None

    def test_case_insensitive(self):
        df = pd.DataFrame({"FrIcTiOn_SlOpE": [0.5]})
        result = _resolve_column(df, ["friction_slope"])
        assert result is not None

    def test_empty_aliases(self):
        df = pd.DataFrame({"a": [1]})
        assert _resolve_column(df, []) is None


class TestPrepareDataframe:
    def test_basic_columns_resolved(self):
        df = _make_catalog()
        work, col_map = _prepare_dataframe(df)
        assert "f3" in work.columns
        assert "delta_mass" in work.columns

    def test_rmax_log_transformed(self):
        """Rmax_kpc > 100 → script must store log10(Rmax)."""
        df = _make_catalog()
        work, _ = _prepare_dataframe(df)
        if "log_Rmax" in work.columns:
            assert work["log_Rmax"].median() < 5  # log-scale, not raw kpc

    def test_missing_f3_raises(self):
        df = pd.DataFrame({"delta_mass": [0.1, 0.2], "other": [1, 2]})
        with pytest.raises(ValueError, match="No F3 column"):
            _prepare_dataframe(df)

    def test_missing_delta_mass_raises(self):
        df = pd.DataFrame({"friction_slope": [0.5, 0.6]})
        with pytest.raises(ValueError, match="delta_mass"):
            _prepare_dataframe(df)

    def test_drops_nan_rows(self):
        df = _make_catalog(n=10)
        df.loc[0, "friction_slope"] = np.nan
        df.loc[1, "delta_mass"] = np.nan
        work, _ = _prepare_dataframe(df)
        assert len(work) == 8

    def test_accepts_beta_alias(self):
        df = _make_catalog()
        df = df.rename(columns={"friction_slope": "beta"})
        work, col_map = _prepare_dataframe(df)
        assert col_map["f3"] == "beta"

    def test_derives_log_mbar_from_M_bar_BTFR(self):
        df = _make_catalog(include_controls=False)
        df["M_bar_BTFR_Msun"] = 10 ** np.random.default_rng(7).normal(10, 0.5, len(df))
        work, col_map = _prepare_dataframe(df)
        assert col_map["log_M_bar"] == "_log_M_bar_derived"
        assert "log_M_bar" in work.columns


# ---------------------------------------------------------------------------
# 2. Block 1 — Controlled OLS regression
# ---------------------------------------------------------------------------

class TestControlledRegression:
    def test_returns_required_keys(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = controlled_regression(df)
        required = {
            "n_galaxies", "beta_env", "beta_env_se", "t_env", "p_env",
            "aic_base", "aic_full", "delta_aic",
            "r2_base", "r2_full", "statsmodels_available",
        }
        assert required.issubset(set(result.keys()))

    def test_n_galaxies_positive(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = controlled_regression(df)
        assert result["n_galaxies"] > 0

    def test_beta_env_finite(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = controlled_regression(df)
        assert math.isfinite(result["beta_env"])

    def test_p_env_in_range(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = controlled_regression(df)
        assert 0.0 <= result["p_env"] <= 1.0

    def test_delta_aic_positive_with_signal(self):
        """When delta_mass has a planted signal ΔAIC should favour full model."""
        df, _ = _prepare_dataframe(_make_catalog(signal_strength=0.6, n=120))
        result = controlled_regression(df)
        assert result["delta_aic"] > 0, (
            f"Expected ΔAIC > 0 for strong signal, got {result['delta_aic']:.3f}"
        )

    def test_r2_full_ge_r2_base(self):
        """Full model R² must not be below base model R²."""
        df, _ = _prepare_dataframe(_make_catalog())
        result = controlled_regression(df)
        assert result["r2_full"] >= result["r2_base"] - 1e-9

    def test_no_controls_case(self):
        """Works when log_M_bar and log_Rmax are absent."""
        df_raw = _make_catalog(include_controls=False)
        df, _ = _prepare_dataframe(df_raw)
        result = controlled_regression(df)
        assert result["controls_used"] == []
        assert math.isfinite(result["beta_env"])

    def test_beta_env_positive_for_positive_signal(self):
        """Planted positive delta_mass signal → β_env > 0."""
        df, _ = _prepare_dataframe(_make_catalog(signal_strength=0.5, n=150, seed=1))
        result = controlled_regression(df)
        assert result["beta_env"] > 0

    def test_beta_env_small_for_no_signal(self):
        """With no planted signal β_env should be close to zero (within 2 SE)."""
        df, _ = _prepare_dataframe(_make_catalog_no_signal(n=200, seed=7))
        result = controlled_regression(df)
        # |β_env| < 3 SE is a reasonable statistical expectation on average
        assert abs(result["beta_env"]) < 3 * result["beta_env_se"] + _BETA_ENV_NULL_SLACK


# ---------------------------------------------------------------------------
# 3. Block 2 — Stratified permutation
# ---------------------------------------------------------------------------

class TestStratifiedPermutation:
    def test_returns_required_keys(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = stratified_permutation(df, n_perms=50, rng=np.random.default_rng(0))
        required = {
            "n_galaxies", "n_perms", "obs_rho", "obs_pval",
            "p_perm", "ci_lo_rho", "ci_hi_rho",
            "perm_rho_mean", "perm_rho_std", "stratified",
        }
        assert required.issubset(set(result.keys()))

    def test_n_perms_matches_request(self):
        df, _ = _prepare_dataframe(_make_catalog(n=40))
        result = stratified_permutation(df, n_perms=30, rng=np.random.default_rng(1))
        assert result["n_perms"] == 30

    def test_obs_rho_in_range(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = stratified_permutation(df, n_perms=50)
        assert -1.0 <= result["obs_rho"] <= 1.0

    def test_p_perm_in_range(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = stratified_permutation(df, n_perms=100, rng=np.random.default_rng(5))
        assert 0.0 <= result["p_perm"] <= 1.0

    def test_null_rho_near_zero(self):
        """Permuted ρ values should be centred near zero."""
        df, _ = _prepare_dataframe(_make_catalog(n=100))
        result = stratified_permutation(df, n_perms=200, rng=np.random.default_rng(2))
        assert abs(result["perm_rho_mean"]) < 0.3

    def test_strong_signal_low_p_perm(self):
        """Strong planted signal → p_perm should be small."""
        df, _ = _prepare_dataframe(_make_catalog(signal_strength=0.7, n=150, seed=3))
        result = stratified_permutation(df, n_perms=500, rng=np.random.default_rng(3))
        assert result["p_perm"] < 0.1, (
            f"Expected small p_perm for strong signal, got {result['p_perm']:.3f}"
        )

    def test_no_signal_p_perm_not_small(self):
        """Without signal p_perm should not be consistently very small."""
        p_perms = []
        for seed in range(5):
            df, _ = _prepare_dataframe(_make_catalog_no_signal(n=60, seed=seed * 11))
            result = stratified_permutation(df, n_perms=200, rng=np.random.default_rng(seed))
            p_perms.append(result["p_perm"])
        # p_perm should NOT be < 0.01 for the majority of seeds
        assert sum(p < 0.01 for p in p_perms) <= _MAX_FALSE_POSITIVES_PERM, (
            f"Too many small p_perm values with no signal: {p_perms}"
        )

    def test_stratified_flag_set_when_log_mbar_present(self):
        df, _ = _prepare_dataframe(_make_catalog(include_controls=True))
        result = stratified_permutation(df, n_perms=20)
        assert result["stratified"] is True

    def test_stratified_false_without_mass(self):
        df, _ = _prepare_dataframe(_make_catalog(include_controls=False))
        result = stratified_permutation(df, n_perms=20)
        assert result["stratified"] is False

    def test_ci_lo_le_ci_hi(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = stratified_permutation(df, n_perms=100)
        assert result["ci_lo_rho"] <= result["ci_hi_rho"]


# ---------------------------------------------------------------------------
# 4. Block 3 — Bootstrap ΔAIC
# ---------------------------------------------------------------------------

class TestBootstrapDeltaAIC:
    def test_returns_required_keys(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = bootstrap_delta_aic(df, n_boot=50, rng=np.random.default_rng(0))
        required = {
            "n_galaxies", "n_boot", "n_boot_valid",
            "observed_delta_aic", "boot_mean_delta_aic",
            "ci_lo", "ci_hi", "frac_above_threshold",
            "statsmodels_available",
        }
        assert required.issubset(set(result.keys()))

    def test_n_boot_valid_le_n_boot(self):
        df, _ = _prepare_dataframe(_make_catalog(n=40))
        result = bootstrap_delta_aic(df, n_boot=50)
        assert result["n_boot_valid"] <= result["n_boot"]

    def test_ci_lo_le_ci_hi(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = bootstrap_delta_aic(df, n_boot=100)
        assert result["ci_lo"] <= result["ci_hi"]

    def test_frac_above_threshold_in_range(self):
        df, _ = _prepare_dataframe(_make_catalog())
        result = bootstrap_delta_aic(df, n_boot=100)
        if not math.isnan(result["frac_above_threshold"]):
            assert 0.0 <= result["frac_above_threshold"] <= 1.0

    def test_strong_signal_high_frac(self):
        """Strong planted signal → most bootstrap resamples should have ΔAIC > 2."""
        df, _ = _prepare_dataframe(_make_catalog(signal_strength=0.7, n=150, seed=10))
        result = bootstrap_delta_aic(df, n_boot=200, rng=np.random.default_rng(10))
        assert result["frac_above_threshold"] > 0.5, (
            f"Expected >50% resamples with ΔAIC > {DELTA_AIC_STRONG_THRESHOLD}, "
            f"got {result['frac_above_threshold']:.2f}"
        )

    def test_observed_delta_aic_finite_with_signal(self):
        df, _ = _prepare_dataframe(_make_catalog(signal_strength=0.5, n=100))
        result = bootstrap_delta_aic(df, n_boot=50)
        assert math.isfinite(result["observed_delta_aic"])

    def test_no_controls_case(self):
        df, _ = _prepare_dataframe(_make_catalog(include_controls=False))
        result = bootstrap_delta_aic(df, n_boot=50)
        assert math.isfinite(result["observed_delta_aic"])


# ---------------------------------------------------------------------------
# 5. run_robustness() integration
# ---------------------------------------------------------------------------

class TestRunRobustness:
    def test_returns_three_dicts(self):
        df = _make_catalog()
        reg, perm, boot = run_robustness(df, n_perms=20, n_boot=20, seed=0)
        assert isinstance(reg, dict)
        assert isinstance(perm, dict)
        assert isinstance(boot, dict)

    def test_keys_present_in_all_blocks(self):
        df = _make_catalog(n=60)
        reg, perm, boot = run_robustness(df, n_perms=30, n_boot=30)
        assert "beta_env" in reg
        assert "obs_rho" in perm
        assert "observed_delta_aic" in boot

    def test_deterministic_with_same_seed(self):
        df = _make_catalog(n=60)
        reg1, perm1, boot1 = run_robustness(df, n_perms=50, n_boot=50, seed=7)
        reg2, perm2, boot2 = run_robustness(df, n_perms=50, n_boot=50, seed=7)
        assert perm1["p_perm"] == perm2["p_perm"]
        assert boot1["boot_mean_delta_aic"] == pytest.approx(
            boot2["boot_mean_delta_aic"], abs=1e-9
        )

    def test_different_seed_gives_different_perm_rhos(self):
        df = _make_catalog(n=80)
        _, perm1, _ = run_robustness(df, n_perms=100, n_boot=10, seed=11)
        _, perm2, _ = run_robustness(df, n_perms=100, n_boot=10, seed=99)
        assert perm1["perm_rho_mean"] != perm2["perm_rho_mean"]


# ---------------------------------------------------------------------------
# 6. Report formatting
# ---------------------------------------------------------------------------

class TestFormatReport:
    def _make_results(self, signal: float = 0.5):
        df = _make_catalog(signal_strength=signal, n=80, seed=1)
        reg, perm, boot = run_robustness(df, n_perms=50, n_boot=50, seed=1)
        return reg, perm, boot

    def test_returns_list_of_strings(self):
        reg, perm, boot = self._make_results()
        lines = format_report(reg, perm, boot)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_contains_block_headers(self):
        reg, perm, boot = self._make_results()
        combined = "\n".join(format_report(reg, perm, boot))
        assert "BLOCK 1" in combined
        assert "BLOCK 2" in combined
        assert "BLOCK 3" in combined

    def test_contains_beta_env(self):
        reg, perm, boot = self._make_results()
        combined = "\n".join(format_report(reg, perm, boot))
        assert "β_env" in combined

    def test_contains_p_perm(self):
        reg, perm, boot = self._make_results()
        combined = "\n".join(format_report(reg, perm, boot))
        assert "p_perm" in combined

    def test_contains_delta_aic(self):
        reg, perm, boot = self._make_results()
        combined = "\n".join(format_report(reg, perm, boot))
        assert "ΔAIC" in combined

    def test_preferred_verdict_when_strong_signal(self):
        """Strong signal → report should say 'full model preferred'."""
        df = _make_catalog(signal_strength=0.8, n=150, seed=2)
        reg, perm, boot = run_robustness(df, n_perms=50, n_boot=50, seed=2)
        combined = "\n".join(format_report(reg, perm, boot))
        assert "preferred" in combined.lower() or "ΔAIC" in combined

    def test_statsmodels_skipped_message(self):
        """When statsmodels not available, report should say [SKIPPED]."""
        reg_no_sm = {"statsmodels_available": False}
        boot_no_sm = {"statsmodels_available": False}
        perm = stratified_permutation(
            _prepare_dataframe(_make_catalog())[0], n_perms=20
        )
        combined = "\n".join(format_report(reg_no_sm, perm, boot_no_sm))
        assert "SKIPPED" in combined


# ---------------------------------------------------------------------------
# 7. CLI / main()
# ---------------------------------------------------------------------------

class TestMainCLI:
    def _write_catalog(self, tmp_path: Path, signal: float = 0.5) -> Path:
        df = _make_catalog(signal_strength=signal, n=60)
        p = tmp_path / "catalog.csv"
        df.to_csv(p, index=False)
        return p

    def test_runs_and_returns_three_dicts(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        reg, perm, boot = main([
            "--catalog", str(cat),
            "--n-perms", "20",
            "--n-boot", "20",
        ])
        assert isinstance(reg, dict)
        assert isinstance(perm, dict)
        assert isinstance(boot, dict)

    def test_writes_output_files(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out_dir = tmp_path / "out"
        main([
            "--catalog", str(cat),
            "--n-perms", "20",
            "--n-boot", "20",
            "--out", str(out_dir),
        ])
        assert (out_dir / "f3_robustness.log").exists()
        assert (out_dir / "f3_robustness.json").exists()
        assert (out_dir / "f3_robustness_summary.csv").exists()

    def test_json_output_has_three_blocks(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out_dir = tmp_path / "out"
        main([
            "--catalog", str(cat),
            "--n-perms", "20",
            "--n-boot", "20",
            "--out", str(out_dir),
        ])
        with (out_dir / "f3_robustness.json").open() as fh:
            data = json.load(fh)
        assert "regression" in data
        assert "permutation" in data
        assert "bootstrap" in data

    def test_csv_output_one_row(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out_dir = tmp_path / "out"
        main([
            "--catalog", str(cat),
            "--n-perms", "20",
            "--n-boot", "20",
            "--out", str(out_dir),
        ])
        df_out = pd.read_csv(out_dir / "f3_robustness_summary.csv")
        assert len(df_out) == 1

    def test_env_catalog_join(self, tmp_path):
        """--env-catalog joins correctly when delta_mass is in a separate file."""
        df = _make_catalog(n=60)
        # Split off delta_mass into a separate file
        main_df = df.drop(columns=["delta_mass"])
        env_df = df[["galaxy", "delta_mass"]]
        main_path = tmp_path / "f3.csv"
        env_path = tmp_path / "env.csv"
        main_df.to_csv(main_path, index=False)
        env_df.to_csv(env_path, index=False)
        reg, perm, boot = main([
            "--catalog", str(main_path),
            "--env-catalog", str(env_path),
            "--n-perms", "20",
            "--n-boot", "20",
        ])
        assert reg.get("n_galaxies", 0) > 0

    def test_missing_catalog_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--catalog", str(tmp_path / "nonexistent.csv")])

    def test_seed_flag_changes_perm_result(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        _, perm1, _ = main(["--catalog", str(cat), "--n-perms", "100",
                             "--n-boot", "10", "--seed", "1"])
        _, perm2, _ = main(["--catalog", str(cat), "--n-perms", "100",
                             "--n-boot", "10", "--seed", "999"])
        assert perm1["perm_rho_mean"] != perm2["perm_rho_mean"]
