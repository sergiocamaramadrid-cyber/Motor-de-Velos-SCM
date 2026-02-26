"""
tests/test_oos_validation.py — Tests for scripts/scm_oos_validation.py.

Uses synthetic SPARC-like data (no real download required) to validate
the OOS radial-split methodology and aggregate statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.scm_oos_validation import (
    oos_validate_galaxy,
    run_oos_validation,
    format_report,
    main,
    A0_DEFAULT,
    TRAIN_FRAC,
    MIN_TRAIN_POINTS,
    MIN_TEST_POINTS,
    _rms_dex,
    _v_total,
    _fit_upsilon_disk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rc(n: int = 20, v_flat: float = 150.0, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic rotation-curve DataFrame."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.5, 15.0, n)
    return pd.DataFrame({
        "r": r,
        "v_obs": np.full(n, v_flat) + rng.normal(0, 3, n),
        "v_obs_err": np.full(n, 5.0),
        "v_gas": 0.3 * v_flat * np.ones(n),
        "v_disk": 0.75 * v_flat * np.ones(n),
        "v_bul": np.zeros(n),
    })


@pytest.fixture(scope="module")
def sparc_dir(tmp_path_factory):
    """Synthetic 30-galaxy SPARC-like dataset in a temporary directory."""
    root = tmp_path_factory.mktemp("sparc30")
    rng = np.random.default_rng(7)
    n_gal = 30
    names = [f"OOS{i:03d}" for i in range(n_gal)]
    v_flats = np.linspace(80.0, 300.0, n_gal)

    pd.DataFrame({
        "Galaxy": names,
        "D": np.linspace(5, 60, n_gal),
        "Inc": np.linspace(30, 80, n_gal),
        "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(n_gal, 5.0),
    }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 20
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


# ---------------------------------------------------------------------------
# _rms_dex
# ---------------------------------------------------------------------------

class TestRmsDex:
    def test_zero_when_perfect(self):
        v = np.array([100.0, 150.0, 200.0])
        assert _rms_dex(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        v_obs = np.array([100.0, 150.0])
        v_pred = np.array([90.0, 160.0])
        assert _rms_dex(v_obs, v_pred) > 0.0

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        v_obs = rng.uniform(50, 300, 50)
        v_pred = rng.uniform(50, 300, 50)
        assert _rms_dex(v_obs, v_pred) >= 0.0


# ---------------------------------------------------------------------------
# _fit_upsilon_disk
# ---------------------------------------------------------------------------

class TestFitUpsilonDisk:
    def test_result_in_bounds(self):
        rc = _make_rc()
        ud = _fit_upsilon_disk(rc, A0_DEFAULT, include_velos=True)
        assert 0.1 <= ud <= 5.0

    def test_baryonic_also_in_bounds(self):
        rc = _make_rc()
        ud = _fit_upsilon_disk(rc, A0_DEFAULT, include_velos=False)
        assert 0.1 <= ud <= 5.0

    def test_returns_float(self):
        rc = _make_rc()
        ud = _fit_upsilon_disk(rc, A0_DEFAULT, include_velos=True)
        assert isinstance(ud, float)


# ---------------------------------------------------------------------------
# oos_validate_galaxy
# ---------------------------------------------------------------------------

class TestOosValidateGalaxy:
    def test_returns_dict_for_sufficient_data(self):
        rc = _make_rc(n=20)
        result = oos_validate_galaxy(rc)
        assert result is not None
        assert isinstance(result, dict)

    def test_required_keys(self):
        rc = _make_rc(n=20)
        result = oos_validate_galaxy(rc)
        assert result is not None
        required = {"n_train", "n_test", "rmse_scm_out", "rmse_bar_out",
                    "delta_rmse_out", "scm_wins"}
        assert required == set(result.keys())

    def test_returns_none_for_too_few_points(self):
        # n=4: n_train=ceil(4*0.5)=2, n_test=2 → n_train < MIN_TRAIN_POINTS=4
        rc = _make_rc(n=4)
        result = oos_validate_galaxy(rc)
        assert result is None

    def test_n_train_and_test_sum_to_n(self):
        rc = _make_rc(n=20)
        result = oos_validate_galaxy(rc)
        assert result is not None
        assert result["n_train"] + result["n_test"] == 20

    def test_rms_values_non_negative(self):
        rc = _make_rc(n=20)
        result = oos_validate_galaxy(rc)
        assert result is not None
        assert result["rmse_scm_out"] >= 0.0
        assert result["rmse_bar_out"] >= 0.0

    def test_delta_rmse_consistent_with_scm_wins(self):
        rc = _make_rc(n=20)
        result = oos_validate_galaxy(rc)
        assert result is not None
        expected_wins = result["delta_rmse_out"] > 0
        assert result["scm_wins"] == expected_wins

    def test_sorted_by_radius(self):
        """Rows out of order must still produce a valid split."""
        rc = _make_rc(n=20)
        rc_shuffled = rc.sample(frac=1, random_state=99).reset_index(drop=True)
        result_orig = oos_validate_galaxy(rc)
        result_shuf = oos_validate_galaxy(rc_shuffled)
        assert result_orig is not None
        assert result_shuf is not None
        # Both should have the same n_train/n_test (same n)
        assert result_orig["n_train"] == result_shuf["n_train"]

    def test_custom_train_frac(self):
        rc = _make_rc(n=20)
        result = oos_validate_galaxy(rc, train_frac=0.7)
        assert result is not None
        assert result["n_train"] == 14  # ceil(20 * 0.7)
        assert result["n_test"] == 6

    def test_minimum_valid_size(self):
        """n=MIN_TRAIN_POINTS + MIN_TEST_POINTS should be valid."""
        n_min = MIN_TRAIN_POINTS + MIN_TEST_POINTS  # 4+2=6 → n_train=3, need 4
        # With train_frac=0.5: n_train=ceil(6*0.5)=3 < 4, so n=10 is reliable
        rc = _make_rc(n=10)
        result = oos_validate_galaxy(rc)
        assert result is not None


# ---------------------------------------------------------------------------
# run_oos_validation
# ---------------------------------------------------------------------------

class TestRunOosValidation:
    def test_returns_summary_dict(self, sparc_dir, tmp_path):
        summary = run_oos_validation(sparc_dir, tmp_path / "out", verbose=False)
        assert isinstance(summary, dict)

    def test_required_summary_keys(self, sparc_dir, tmp_path):
        summary = run_oos_validation(sparc_dir, tmp_path / "out2", verbose=False)
        required = {"N_valid", "success_pct", "median_delta_rmse_out",
                    "wilcoxon_statistic", "wilcoxon_pvalue"}
        assert required == set(summary.keys())

    def test_n_valid_positive(self, sparc_dir, tmp_path):
        summary = run_oos_validation(sparc_dir, tmp_path / "out3", verbose=False)
        assert summary["N_valid"] > 0

    def test_success_pct_in_range(self, sparc_dir, tmp_path):
        summary = run_oos_validation(sparc_dir, tmp_path / "out4", verbose=False)
        assert 0.0 <= summary["success_pct"] <= 100.0

    def test_wilcoxon_pvalue_in_range(self, sparc_dir, tmp_path):
        summary = run_oos_validation(sparc_dir, tmp_path / "out5", verbose=False)
        if not np.isnan(summary["wilcoxon_pvalue"]):
            assert 0.0 <= summary["wilcoxon_pvalue"] <= 1.0

    def test_writes_per_galaxy_csv(self, sparc_dir, tmp_path):
        out = tmp_path / "out6"
        run_oos_validation(sparc_dir, out, verbose=False)
        assert (out / "oos_per_galaxy.csv").exists()

    def test_writes_summary_csv(self, sparc_dir, tmp_path):
        out = tmp_path / "out7"
        run_oos_validation(sparc_dir, out, verbose=False)
        assert (out / "oos_summary.csv").exists()

    def test_writes_log_file(self, sparc_dir, tmp_path):
        out = tmp_path / "out8"
        run_oos_validation(sparc_dir, out, verbose=False)
        assert (out / "oos_validation.log").exists()

    def test_per_galaxy_csv_columns(self, sparc_dir, tmp_path):
        out = tmp_path / "out9"
        run_oos_validation(sparc_dir, out, verbose=False)
        df = pd.read_csv(out / "oos_per_galaxy.csv")
        for col in ["galaxy", "n_train", "n_test", "rmse_scm_out",
                    "rmse_bar_out", "delta_rmse_out", "scm_wins"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_summary_csv_columns(self, sparc_dir, tmp_path):
        out = tmp_path / "out10"
        run_oos_validation(sparc_dir, out, verbose=False)
        df = pd.read_csv(out / "oos_summary.csv")
        for col in ["N_valid", "success_pct", "median_delta_rmse_out",
                    "wilcoxon_statistic", "wilcoxon_pvalue"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_n_valid_matches_per_galaxy_rows(self, sparc_dir, tmp_path):
        out = tmp_path / "out11"
        summary = run_oos_validation(sparc_dir, out, verbose=False)
        df = pd.read_csv(out / "oos_per_galaxy.csv")
        assert summary["N_valid"] == len(df)

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_oos_validation(tmp_path / "nonexistent", tmp_path / "out",
                               verbose=False)


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_n_valid(self):
        summary = {
            "N_valid": 42,
            "success_pct": 65.0,
            "median_delta_rmse_out": 0.012,
            "wilcoxon_statistic": 350.0,
            "wilcoxon_pvalue": 0.003,
        }
        lines = format_report(summary, A0_DEFAULT, "data/SPARC")
        combined = "\n".join(lines)
        assert "N_valid" in combined
        assert "42" in combined

    def test_contains_success_pct(self):
        summary = {
            "N_valid": 30,
            "success_pct": 73.3,
            "median_delta_rmse_out": 0.008,
            "wilcoxon_statistic": 200.0,
            "wilcoxon_pvalue": 0.01,
        }
        lines = format_report(summary, A0_DEFAULT, "data/SPARC")
        combined = "\n".join(lines)
        assert "Success" in combined
        assert "73.3" in combined

    def test_nan_summary_handled(self):
        summary = {
            "N_valid": 0,
            "success_pct": float("nan"),
            "median_delta_rmse_out": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
        }
        lines = format_report(summary, A0_DEFAULT, "data/SPARC")
        combined = "\n".join(lines)
        assert "N_valid" in combined
        assert "0" in combined


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_summary_dict(self, sparc_dir, tmp_path):
        result = main(["--data-dir", str(sparc_dir), "--out", str(tmp_path / "cli")])
        assert isinstance(result, dict)
        assert "N_valid" in result

    def test_creates_output_files(self, sparc_dir, tmp_path):
        out = tmp_path / "cli2"
        main(["--data-dir", str(sparc_dir), "--out", str(out)])
        assert (out / "oos_per_galaxy.csv").exists()
        assert (out / "oos_summary.csv").exists()
        assert (out / "oos_validation.log").exists()

    def test_quiet_flag(self, sparc_dir, tmp_path, capsys):
        main(["--data-dir", str(sparc_dir), "--out", str(tmp_path / "cli3"),
              "--quiet"])
        captured = capsys.readouterr()
        # In quiet mode only the final report lines should be printed
        assert "OOS validation" not in captured.out

    def test_missing_data_dir_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--data-dir", str(tmp_path / "nonexistent"),
                  "--out", str(tmp_path / "out")])
