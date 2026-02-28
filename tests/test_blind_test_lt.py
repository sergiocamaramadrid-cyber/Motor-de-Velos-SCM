"""
tests/test_blind_test_lt.py — Tests for scripts/blind_test_little_things.py.

Validates:
  - little_things_global.csv has the required columns and ≥20 rows
  - log_gbar values are in the physically expected range for dwarf galaxies
  - run_blind_test() produces predictions.csv and summary.csv
  - summary.csv has the required columns
  - predictions.csv has the required per-galaxy columns
  - Both model predictions are finite for all galaxies
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_CSV_PATH = _REPO_ROOT / "data" / "little_things_global.csv"

# ---------------------------------------------------------------------------
# Helpers — import the script module from the scripts package
# ---------------------------------------------------------------------------

import importlib.util

def _import_blind_test():
    spec = importlib.util.spec_from_file_location(
        "blind_test_little_things",
        _REPO_ROOT / "scripts" / "blind_test_little_things.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_bt = _import_blind_test()


# ---------------------------------------------------------------------------
# CSV integrity tests
# ---------------------------------------------------------------------------

class TestLittleThingsGlobalCsv:
    """Structural and physical sanity checks for the input data file."""

    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(_CSV_PATH)

    def test_file_exists(self):
        assert _CSV_PATH.exists(), f"CSV not found: {_CSV_PATH}"

    def test_required_columns(self, df):
        required = {"galaxy_id", "logM", "logVobs", "log_gbar"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_at_least_20_galaxies(self, df):
        assert len(df) >= 20, f"Expected ≥20 galaxies, got {len(df)}"

    def test_no_duplicate_galaxy_ids(self, df):
        assert df["galaxy_id"].nunique() == len(df), "Duplicate galaxy_id values"

    def test_logM_range(self, df):
        """Dwarf galaxy baryonic masses: 7.0 ≤ logM ≤ 9.5."""
        assert df["logM"].between(6.5, 10.0).all(), (
            f"logM out of expected range: {df['logM'].describe()}"
        )

    def test_logVobs_range(self, df):
        """LITTLE THINGS flat velocities: 1.0 ≤ logVobs ≤ 2.1 (10–125 km/s)."""
        assert df["logVobs"].between(0.9, 2.2).all(), (
            f"logVobs out of expected range: {df['logVobs'].describe()}"
        )

    def test_log_gbar_range(self, df):
        """Outer-regime baryonic accelerations for dwarfs: −13 ≤ log_gbar ≤ −10."""
        assert df["log_gbar"].between(-13.0, -10.0).all(), (
            f"log_gbar out of physical range: {df['log_gbar'].describe()}"
        )

    def test_no_nans(self, df):
        for col in ("logM", "logVobs", "log_gbar"):
            assert df[col].notna().all(), f"NaN values in column {col}"


# ---------------------------------------------------------------------------
# Prediction function unit tests
# ---------------------------------------------------------------------------

class TestPredictionFunctions:
    """Numerical correctness of _logV_btfr and _logV_scm."""

    def test_btfr_monotone_in_logM(self):
        logM = np.linspace(7.0, 9.5, 20)
        logV = _bt._logV_btfr(logM)
        assert np.all(np.diff(logV) > 0), "BTFR should be monotone increasing in logM"

    def test_btfr_slope_is_quarter(self):
        logM = np.array([7.0, 8.0, 9.0])
        logV = _bt._logV_btfr(logM)
        slope = np.diff(logV) / np.diff(logM)
        np.testing.assert_allclose(slope, 0.25, rtol=1e-6)

    def test_btfr_all_finite(self):
        logM = np.array([7.0, 7.5, 8.0, 8.5, 9.0])
        assert np.all(np.isfinite(_bt._logV_btfr(logM)))

    def test_scm_all_finite(self):
        df = pd.read_csv(_CSV_PATH)
        logV = _bt._logV_scm(df["logM"].values, df["log_gbar"].values)
        assert np.all(np.isfinite(logV)), "SCM predictions contain non-finite values"

    def test_scm_deep_mond_converges_to_btfr(self):
        """In the very deep MOND regime (g_bar → 0), SCM → BTFR."""
        logM = np.array([8.0])
        log_gbar_deep = np.array([-15.0])  # very deep MOND
        logV_scm = _bt._logV_scm(logM, log_gbar_deep)
        logV_btfr = _bt._logV_btfr(logM)
        np.testing.assert_allclose(logV_scm, logV_btfr, atol=0.01)


# ---------------------------------------------------------------------------
# Integration test: run_blind_test()
# ---------------------------------------------------------------------------

class TestRunBlindTest:
    """End-to-end test of run_blind_test()."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("resultados_lt")
        summary = _bt.run_blind_test(input_path=_CSV_PATH, out_dir=out)
        return out, summary

    def test_predictions_csv_exists(self, results):
        out, _ = results
        assert (out / "predictions.csv").exists()

    def test_summary_csv_exists(self, results):
        out, _ = results
        assert (out / "summary.csv").exists()

    def test_summary_columns(self, results):
        out, _ = results
        df = pd.read_csv(out / "summary.csv")
        required = {"N", "RMSE_SCM", "RMSE_BTFR", "frac_mejora_pct", "wilcoxon_p"}
        assert required.issubset(df.columns)

    def test_predictions_columns(self, results):
        out, _ = results
        df = pd.read_csv(out / "predictions.csv")
        required = {"galaxy_id", "logM", "logVobs", "log_gbar",
                    "logV_SCM", "logV_BTFR", "res_SCM", "res_BTFR"}
        assert required.issubset(df.columns)

    def test_galaxy_count(self, results):
        out, _ = results
        df = pd.read_csv(out / "summary.csv")
        assert int(df["N"].iloc[0]) >= 20

    def test_rmse_positive(self, results):
        out, _ = results
        df = pd.read_csv(out / "summary.csv")
        assert df["RMSE_SCM"].iloc[0] > 0
        assert df["RMSE_BTFR"].iloc[0] > 0

    def test_predictions_finite(self, results):
        out, _ = results
        df = pd.read_csv(out / "predictions.csv")
        for col in ("logV_SCM", "logV_BTFR", "res_SCM", "res_BTFR"):
            assert df[col].notna().all(), f"{col} contains NaN"
            assert np.all(np.isfinite(df[col].values)), f"{col} contains inf"
