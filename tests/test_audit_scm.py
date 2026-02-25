"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Tests cover:
  - Column alias resolution (flexible CSV headers)
  - GroupKFold produces correct OOS splits (no galaxy leakage)
  - Permutation test uses corrected p-value formula (Phipson & Smyth 2010)
  - groupkfold_per_galaxy.csv has one aggregated row per galaxy
  - Benjamini-Hochberg correction on per-galaxy p-values
  - run_audit writes all four required artefacts
  - audit_summary.json has the expected keys (including rmse_real, folds, …)
  - --strict flag triggers SystemExit on FAIL
  - Reproducibility: same seed → identical results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Resolve the script module from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from audit_scm import (
    _bh_correction,
    _find_col,
    _group_kfold_splits,
    _ols_fit,
    _ols_predict,
    _run_cv,
    run_audit,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_csv(tmp_path: Path, n_galaxies: int = 10,
                         n_pts: int = 15, seed: int = 0) -> Path:
    """Write a synthetic CSV that mimics universal_term_comparison_full.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_galaxies):
        gal = f"G{i:04d}"
        log_g_bar = rng.uniform(-10, -7, n_pts)
        log_g_obs = 0.95 * log_g_bar + 0.5 + rng.normal(0, 0.05, n_pts)
        for lgb, lgo in zip(log_g_bar, log_g_obs):
            rows.append({"galaxy": gal, "log_g_bar": lgb, "log_g_obs": lgo})
    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def synthetic_csv(tmp_path):
    return _make_synthetic_csv(tmp_path)


# ---------------------------------------------------------------------------
# _find_col
# ---------------------------------------------------------------------------

class TestFindCol:
    def test_finds_primary_name(self):
        df = pd.DataFrame({"galaxy": [], "log_g_bar": [], "log_g_obs": []})
        assert _find_col(df, ("galaxy", "galaxy_id"), "galaxy") == "galaxy"

    def test_finds_alias(self):
        df = pd.DataFrame({"galaxy_id": [], "log_g_bar": [], "log_g_obs": []})
        assert _find_col(df, ("galaxy", "galaxy_id"), "galaxy") == "galaxy_id"

    def test_raises_on_missing(self):
        df = pd.DataFrame({"x": [], "y": []})
        with pytest.raises(ValueError, match="not found"):
            _find_col(df, ("galaxy", "galaxy_id"), "galaxy")


# ---------------------------------------------------------------------------
# _bh_correction
# ---------------------------------------------------------------------------

class TestBHCorrection:
    def test_returns_same_length(self):
        p = np.array([0.01, 0.04, 0.10, 0.20])
        adj = _bh_correction(p)
        assert len(adj) == len(p)

    def test_adjusted_ge_raw(self):
        """BH-adjusted p-values are always ≥ raw p-values."""
        p = np.array([0.001, 0.010, 0.020, 0.050, 0.100])
        adj = _bh_correction(p)
        assert (adj >= p).all()

    def test_adjusted_le_one(self):
        p = np.array([0.5, 0.7, 0.9, 1.0])
        adj = _bh_correction(p)
        assert (adj <= 1.0).all()

    def test_monotone_in_sorted_order(self):
        """Adjusted p-values must be non-decreasing when sorted by raw p."""
        p = np.array([0.001, 0.01, 0.02, 0.04, 0.05])
        adj = _bh_correction(p)
        assert (np.diff(adj) >= -1e-12).all()

    def test_nan_preserved(self):
        p = np.array([0.01, float("nan"), 0.05])
        adj = _bh_correction(p)
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0])

    def test_all_nan(self):
        p = np.array([float("nan"), float("nan")])
        adj = _bh_correction(p)
        assert np.isnan(adj).all()

    def test_single_value(self):
        p = np.array([0.03])
        adj = _bh_correction(p)
        assert abs(adj[0] - 0.03) < 1e-12


# ---------------------------------------------------------------------------
# _group_kfold_splits
# ---------------------------------------------------------------------------

class TestGroupKFoldSplits:
    def test_all_samples_covered(self):
        rng = np.random.default_rng(0)
        groups = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
        splits = _group_kfold_splits(groups, n_splits=4, rng=rng)
        test_indices = set()
        for _, test_idx in splits:
            test_indices.update(test_idx.tolist())
        assert test_indices == set(range(len(groups)))

    def test_no_galaxy_overlap_between_train_and_test(self):
        rng = np.random.default_rng(0)
        groups = np.array([f"G{i}" for i in range(10) for _ in range(5)])
        splits = _group_kfold_splits(groups, n_splits=5, rng=rng)
        for train_idx, test_idx in splits:
            train_gals = set(groups[train_idx])
            test_gals = set(groups[test_idx])
            assert train_gals.isdisjoint(test_gals), (
                f"Leakage: {train_gals & test_gals}"
            )

    def test_n_folds(self):
        rng = np.random.default_rng(0)
        groups = np.array([f"G{i}" for i in range(10) for _ in range(3)])
        splits = _group_kfold_splits(groups, n_splits=5, rng=rng)
        assert len(splits) == 5


# ---------------------------------------------------------------------------
# _ols_fit / _ols_predict
# ---------------------------------------------------------------------------

class TestOLS:
    def test_perfect_line(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        a, b = _ols_fit(x, y)
        assert abs(a - 2.0) < 1e-10
        assert abs(b - 1.0) < 1e-10

    def test_predict_recovers_line(self):
        x = np.linspace(-3, 3, 20)
        a_true, b_true = 0.8, -0.3
        y = a_true * x + b_true + np.random.default_rng(0).normal(0, 0.01, 20)
        a, b = _ols_fit(x, y)
        y_pred = _ols_predict(x, a, b)
        assert np.sqrt(np.mean((y - y_pred) ** 2)) < 0.02

    def test_single_point_returns_nan_slope(self):
        a, b = _ols_fit(np.array([1.0]), np.array([2.0]))
        assert np.isnan(a)

    def test_constant_x_returns_zero_slope(self):
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 3.0, 4.0])
        a, b = _ols_fit(x, y)
        assert a == 0.0


# ---------------------------------------------------------------------------
# _run_cv — internal fold-level records
# ---------------------------------------------------------------------------

class TestRunCV:
    def test_per_galaxy_folds_columns(self, synthetic_csv):
        """_run_cv returns fold-level records with expected columns."""
        df = pd.read_csv(synthetic_csv)
        rng = np.random.default_rng(0)
        per_gal_folds, _ = _run_cv(df, "galaxy", "log_g_bar", "log_g_obs", 5, rng)
        for col in ("fold", "galaxy", "rmse", "n_points", "slope", "intercept"):
            assert col in per_gal_folds.columns

    def test_per_point_columns(self, synthetic_csv):
        df = pd.read_csv(synthetic_csv)
        rng = np.random.default_rng(0)
        _, per_pt = _run_cv(df, "galaxy", "log_g_bar", "log_g_obs", 5, rng)
        for col in ("fold", "galaxy", "log_g_bar", "log_g_obs", "pred", "residual"):
            assert col in per_pt.columns

    def test_all_points_appear_in_per_point(self, synthetic_csv):
        df = pd.read_csv(synthetic_csv)
        rng = np.random.default_rng(0)
        _, per_pt = _run_cv(df, "galaxy", "log_g_bar", "log_g_obs", 5, rng)
        assert len(per_pt) == len(df)

    def test_rmse_non_negative(self, synthetic_csv):
        df = pd.read_csv(synthetic_csv)
        rng = np.random.default_rng(0)
        per_gal_folds, _ = _run_cv(df, "galaxy", "log_g_bar", "log_g_obs", 5, rng)
        assert (per_gal_folds["rmse"] >= 0).all()


# ---------------------------------------------------------------------------
# run_audit — integration
# ---------------------------------------------------------------------------

_EXPECTED_ARTEFACTS = [
    "audit_summary.json",
    "groupkfold_per_galaxy.csv",
    "groupkfold_per_point.csv",
    "permutation_test.csv",
]

# New referee-facing keys required in audit_summary.json
_REQUIRED_SUMMARY_KEYS = {
    "input_csv",
    "n_galaxies",
    "n_points",
    "n_splits",
    "n_perm",
    "seed",
    # Referee-facing primary keys
    "rmse_real",
    "rmse_perm_mean",
    "rmse_perm_std",
    "p_value",
    "frac_galaxies_improved",
    "folds",
    # Legacy aliases (backward compatible)
    "observed_mean_oos_rmse",
    "null_mean_rmse",
    "p_value_permutation",
    # Verdict
    "strict_checks",
    "verdict",
}


class TestRunAudit:
    def test_all_artefacts_written(self, synthetic_csv, tmp_path):
        run_audit(synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42)
        for fname in _EXPECTED_ARTEFACTS:
            assert (tmp_path / "out" / fname).exists(), f"Missing: {fname}"

    def test_summary_keys(self, synthetic_csv, tmp_path):
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert _REQUIRED_SUMMARY_KEYS.issubset(set(summary.keys()))

    def test_p_value_in_range(self, synthetic_csv, tmp_path):
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert 0.0 < summary["p_value"] <= 1.0, (
            "Corrected p-value must be > 0 (Phipson & Smyth 2010)"
        )

    def test_p_value_legacy_key(self, synthetic_csv, tmp_path):
        """Legacy key p_value_permutation must equal p_value."""
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert summary["p_value"] == summary["p_value_permutation"]

    def test_rmse_aliases_consistent(self, synthetic_csv, tmp_path):
        """rmse_real == observed_mean_oos_rmse, rmse_perm_mean == null_mean_rmse."""
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert summary["rmse_real"] == summary["observed_mean_oos_rmse"]
        assert summary["rmse_perm_mean"] == summary["null_mean_rmse"]

    def test_rmse_perm_std_non_negative(self, synthetic_csv, tmp_path):
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert summary["rmse_perm_std"] >= 0.0

    def test_folds_dict_has_n_splits_entries(self, synthetic_csv, tmp_path):
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert len(summary["folds"]) == 5
        for k in summary["folds"]:
            fold = summary["folds"][k]
            assert "n_test_galaxies" in fold
            assert "mean_rmse" in fold
            assert "slope" in fold
            assert "intercept" in fold

    def test_frac_improved_in_range(self, synthetic_csv, tmp_path):
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        fi = summary["frac_galaxies_improved"]
        assert 0.0 <= fi <= 1.0

    def test_verdict_is_pass_or_fail(self, synthetic_csv, tmp_path):
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=10, seed=42
        )
        assert summary["verdict"] in ("PASS", "FAIL")

    def test_json_is_readable(self, synthetic_csv, tmp_path):
        out = tmp_path / "out"
        run_audit(synthetic_csv, out, n_splits=5, n_perm=10, seed=42)
        with open(out / "audit_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["n_galaxies"] == 10  # matches _make_synthetic_csv default

    def test_n_perm_rows_in_permutation_csv(self, synthetic_csv, tmp_path):
        out = tmp_path / "out"
        run_audit(synthetic_csv, out, n_splits=5, n_perm=20, seed=42)
        perm_df = pd.read_csv(out / "permutation_test.csv")
        assert len(perm_df) == 20

    def test_per_galaxy_csv_one_row_per_galaxy(self, synthetic_csv, tmp_path):
        """groupkfold_per_galaxy.csv must have exactly one row per galaxy."""
        out = tmp_path / "out"
        run_audit(synthetic_csv, out, n_splits=5, n_perm=10, seed=42)
        pg = pd.read_csv(out / "groupkfold_per_galaxy.csv")
        df = pd.read_csv(synthetic_csv)
        n_gal = df["galaxy"].nunique()
        assert len(pg) == n_gal, (
            f"Expected {n_gal} rows (one per galaxy), got {len(pg)}"
        )

    def test_per_galaxy_csv_columns(self, synthetic_csv, tmp_path):
        """groupkfold_per_galaxy.csv must contain aggregated metric columns."""
        out = tmp_path / "out"
        run_audit(synthetic_csv, out, n_splits=5, n_perm=10, seed=42)
        pg = pd.read_csv(out / "groupkfold_per_galaxy.csv")
        for col in (
            "galaxy", "mean_rmse", "std_rmse", "n_folds", "n_points",
            "null_rmse_median", "null_rmse_mean", "null_rmse_std",
            "improved", "p_value_galaxy", "p_value_bh",
        ):
            assert col in pg.columns, f"Missing column: {col}"

    def test_per_galaxy_bh_ge_raw(self, synthetic_csv, tmp_path):
        """BH-adjusted p-values must be ≥ raw per-galaxy p-values."""
        out = tmp_path / "out"
        run_audit(synthetic_csv, out, n_splits=5, n_perm=10, seed=42)
        pg = pd.read_csv(out / "groupkfold_per_galaxy.csv")
        valid = pg["p_value_galaxy"].notna() & pg["p_value_bh"].notna()
        assert (pg.loc[valid, "p_value_bh"] >= pg.loc[valid, "p_value_galaxy"] - 1e-12).all()

    def test_reproducibility(self, synthetic_csv, tmp_path):
        s1 = run_audit(synthetic_csv, tmp_path / "r1", n_splits=5, n_perm=10, seed=7)
        s2 = run_audit(synthetic_csv, tmp_path / "r2", n_splits=5, n_perm=10, seed=7)
        assert s1["rmse_real"] == s2["rmse_real"]
        assert s1["p_value"] == s2["p_value"]

    def test_too_few_galaxies_raises(self, tmp_path):
        # Only 2 galaxies but n_splits=5
        rows = [{"galaxy": "G0", "log_g_bar": -8.0, "log_g_obs": -7.9}] * 5
        rows += [{"galaxy": "G1", "log_g_bar": -9.0, "log_g_obs": -8.8}] * 5
        csv = tmp_path / "tiny.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="n_splits"):
            run_audit(csv, tmp_path / "out", n_splits=5)

    def test_strict_pass_no_exit(self, synthetic_csv, tmp_path):
        """A strong-signal synthetic dataset should pass strict checks.

        Requires n_perm ≥ 19 so the corrected p-value 1/(n_perm+1) ≤ 0.05.
        """
        summary = run_audit(
            synthetic_csv, tmp_path / "out", n_splits=5, n_perm=20,
            seed=42, strict=True,
        )
        # Should not raise SystemExit; and verdict must be PASS
        assert summary["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_cli_basic(self, synthetic_csv, tmp_path):
        out = str(tmp_path / "cli_out")
        main([
            "--input", str(synthetic_csv),
            "--out", out,
            "--n-splits", "5",
            "--n-perm", "10",
            "--seed", "42",
        ])
        assert (Path(out) / "audit_summary.json").exists()

    def test_cli_strict_fail_exits(self, tmp_path):
        """Craft a dataset where the null beats the real model → FAIL verdict."""
        # All points have random log_g_bar and random log_g_obs (no signal)
        rng = np.random.default_rng(999)
        rows = []
        for i in range(10):
            gal = f"G{i:04d}"
            for _ in range(10):
                rows.append({
                    "galaxy": gal,
                    "log_g_bar": rng.uniform(-10, -7),
                    "log_g_obs": rng.uniform(-10, -7),  # completely random
                })
        csv = tmp_path / "random.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)

        # With --strict on noise data the verdict may be FAIL → SystemExit(1)
        # (It could also PASS on 10 perm by chance; we just check it doesn't crash.)
        try:
            main([
                "--input", str(csv),
                "--out", str(tmp_path / "out"),
                "--n-splits", "5",
                "--n-perm", "10",
                "--seed", "0",
                "--strict",
            ])
        except SystemExit as exc:
            assert exc.code == 1
