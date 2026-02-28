"""
tests/test_blind_test_lt.py — Unit tests for the LITTLE THINGS blind-test pipeline.

Tests cover:
  - CSV integrity: required columns, row count, no NaN values
  - Physical ranges: logVobs, logM, log_gbar, log_j within expected bounds
  - Model functions: BTFR and interpolation models return finite values
  - End-to-end pipeline: CLI produces predictions.csv and summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.blind_test_little_things import (
    REQUIRED_COLS,
    A0_DEFAULT,
    GAS_FRACTION_DEFAULT,
    predict_logv_btfr,
    predict_logv_interp,
    _interp_constant,
    load_dataset,
    run_blind_test,
    main,
)

# ---------------------------------------------------------------------------
# Path to the shipped dataset
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DATASET = _REPO_ROOT / "data" / "little_things_global.csv"

N_GALAXIES = 26


# ---------------------------------------------------------------------------
# CSV integrity
# ---------------------------------------------------------------------------

class TestDatasetIntegrity:
    def test_file_exists(self):
        assert _DATASET.exists(), f"Dataset not found: {_DATASET}"

    def test_row_count(self):
        df = pd.read_csv(_DATASET)
        assert len(df) == N_GALAXIES, (
            f"Expected {N_GALAXIES} rows, got {len(df)}"
        )

    def test_required_columns_present(self):
        df = pd.read_csv(_DATASET)
        for col in REQUIRED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nan_values(self):
        df = pd.read_csv(_DATASET)
        for col in REQUIRED_COLS:
            assert df[col].notna().all(), f"Column '{col}' contains NaN values"

    def test_galaxy_ids_unique(self):
        df = pd.read_csv(_DATASET)
        assert df["galaxy_id"].nunique() == len(df), (
            "galaxy_id values are not unique"
        )


# ---------------------------------------------------------------------------
# Physical ranges
# ---------------------------------------------------------------------------

class TestPhysicalRanges:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(_DATASET)

    def test_logVobs_range(self, df):
        """log10(Vflat/km/s) expected in [1.0, 2.0] for dwarf galaxies."""
        assert df["logVobs"].between(1.0, 2.0).all(), (
            f"logVobs out of [1.0, 2.0]: {df['logVobs'].describe()}"
        )

    def test_logM_range(self, df):
        """log10(M_star/Msun) expected in [5.0, 9.0] for LITTLE THINGS."""
        assert df["logM"].between(5.0, 9.0).all(), (
            f"logM out of [5.0, 9.0]: {df['logM'].describe()}"
        )

    def test_log_gbar_range(self, df):
        """log10(g_bar/m·s⁻²) expected in [-13.5, -10.0] for dwarf galaxies."""
        assert df["log_gbar"].between(-13.5, -10.0).all(), (
            f"log_gbar out of [-13.5, -10.0]: {df['log_gbar'].describe()}"
        )

    def test_log_j_range(self, df):
        """log10(j/kpc·km/s) expected in [0.5, 2.5] for dwarf galaxies."""
        assert df["log_j"].between(0.5, 2.5).all(), (
            f"log_j out of [0.5, 2.5]: {df['log_j'].describe()}"
        )

    def test_logVobs_monotone_with_logM(self, df):
        """Correlation between logM and logVobs should be positive (BTFR)."""
        corr = df["logM"].corr(df["logVobs"])
        assert corr > 0.5, f"Expected positive correlation logM vs logVobs, got {corr:.3f}"


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

class TestPredictLogvBtfr:
    def test_returns_finite_array(self):
        logM = np.array([6.0, 7.0, 8.0])
        result = predict_logv_btfr(logM)
        assert np.all(np.isfinite(result))

    def test_monotone_in_logM(self):
        logM = np.linspace(5.0, 9.0, 20)
        result = predict_logv_btfr(logM)
        assert np.all(np.diff(result) > 0), "BTFR prediction not monotone in logM"

    def test_slope_quarter(self):
        """The BTFR slope must be exactly 0.25."""
        logM = np.array([6.0, 8.0])
        result = predict_logv_btfr(logM)
        slope = (result[1] - result[0]) / (logM[1] - logM[0])
        assert abs(slope - 0.25) < 1e-10, f"BTFR slope {slope} ≠ 0.25"

    def test_physical_range(self):
        """Predictions should fall in a physically plausible range."""
        logM = np.array([5.8, 6.8, 7.5, 8.4])
        result = predict_logv_btfr(logM)
        assert np.all(result > 0.5), "logV predictions below 0.5"
        assert np.all(result < 2.5), "logV predictions above 2.5"

    def test_gas_fraction_effect(self):
        """Higher gas fraction → higher Mbar → higher Vflat."""
        logM = np.array([7.0])
        v_low = predict_logv_btfr(logM, gas_fraction=1.0)
        v_high = predict_logv_btfr(logM, gas_fraction=10.0)
        assert v_high > v_low

    def test_scalar_input(self):
        result = predict_logv_btfr(7.5)
        assert np.isfinite(result)


class TestPredictLogvInterp:
    def test_returns_finite_array(self):
        log_gbar = np.array([-11.5, -11.0, -10.5])
        log_j = np.array([1.5, 1.7, 1.9])
        result = predict_logv_interp(log_gbar, log_j)
        assert np.all(np.isfinite(result))

    def test_monotone_in_gbar(self):
        """Higher g_bar (less deep MOND) → higher Vflat."""
        log_gbar = np.linspace(-13.0, -10.0, 20)
        log_j = np.full(20, 1.5)
        result = predict_logv_interp(log_gbar, log_j)
        assert np.all(np.diff(result) > 0), "interp prediction not monotone in log_gbar"

    def test_monotone_in_log_j(self):
        """Higher specific angular momentum → higher Vflat."""
        log_gbar = np.full(20, -11.5)
        log_j = np.linspace(0.5, 2.5, 20)
        result = predict_logv_interp(log_gbar, log_j)
        assert np.all(np.diff(result) > 0), "interp prediction not monotone in log_j"

    def test_slope_one_sixth_in_gbar(self):
        """The slope d(logV)/d(log_gbar) must be exactly 1/6."""
        log_gbar = np.array([-12.0, -11.0])
        log_j = np.array([1.5, 1.5])
        result = predict_logv_interp(log_gbar, log_j)
        slope = (result[1] - result[0]) / (log_gbar[1] - log_gbar[0])
        assert abs(slope - 1.0 / 6.0) < 1e-10, f"Slope {slope} ≠ 1/6"

    def test_slope_one_third_in_log_j(self):
        """The slope d(logV)/d(log_j) must be exactly 2/6 = 1/3."""
        log_gbar = np.array([-11.5, -11.5])
        log_j = np.array([1.0, 2.0])
        result = predict_logv_interp(log_gbar, log_j)
        slope = (result[1] - result[0]) / (log_j[1] - log_j[0])
        assert abs(slope - 1.0 / 3.0) < 1e-10, f"Slope {slope} ≠ 1/3"

    def test_scalar_input(self):
        result = predict_logv_interp(-11.5, 1.7)
        assert np.isfinite(result)


class TestInterpConstant:
    def test_finite(self):
        C = _interp_constant()
        assert np.isfinite(C)

    def test_approximately_17(self):
        """The constant should be ≈ 17.06 (from KPC_TO_M, KMS_TO_MS, a0)."""
        C = _interp_constant()
        assert abs(C - 17.06) < 0.1, f"C_interp = {C:.4f}, expected ≈17.06"


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestRunBlindTest:
    @pytest.fixture()
    def out_dir(self, tmp_path):
        return tmp_path / "blind_test_out"

    def test_produces_predictions_csv(self, out_dir):
        predictions, _ = run_blind_test(_DATASET, out_dir)
        assert (out_dir / "predictions.csv").exists()
        assert len(predictions) == N_GALAXIES

    def test_produces_summary_csv(self, out_dir):
        _, summary = run_blind_test(_DATASET, out_dir)
        assert (out_dir / "summary.csv").exists()
        assert set(summary["model"]) == {"btfr", "interp"}

    def test_predictions_columns(self, out_dir):
        from scripts.blind_test_little_things import PRED_COLS
        predictions, _ = run_blind_test(_DATASET, out_dir)
        assert list(predictions.columns) == PRED_COLS

    def test_predictions_no_nan(self, out_dir):
        predictions, _ = run_blind_test(_DATASET, out_dir)
        assert predictions.notna().all().all(), "predictions.csv contains NaN"

    def test_summary_columns(self, out_dir):
        _, summary = run_blind_test(_DATASET, out_dir)
        for col in ["model", "N", "RMSE_dex", "MAE_dex", "bias_dex"]:
            assert col in summary.columns, f"Missing column '{col}' in summary"

    def test_summary_N_equals_dataset(self, out_dir):
        _, summary = run_blind_test(_DATASET, out_dir)
        assert (summary["N"] == N_GALAXIES).all()

    def test_summary_rmse_positive(self, out_dir):
        _, summary = run_blind_test(_DATASET, out_dir)
        assert (summary["RMSE_dex"] > 0).all()

    def test_predictions_csv_readable(self, out_dir):
        run_blind_test(_DATASET, out_dir)
        df = pd.read_csv(out_dir / "predictions.csv")
        assert len(df) == N_GALAXIES

    def test_summary_csv_readable(self, out_dir):
        run_blind_test(_DATASET, out_dir)
        df = pd.read_csv(out_dir / "summary.csv")
        assert len(df) == 2

    def test_custom_gas_fraction(self, out_dir):
        """Running with different gas_fraction must change BTFR residuals."""
        _, s1 = run_blind_test(_DATASET, out_dir / "gf1", gas_fraction=1.0)
        _, s2 = run_blind_test(_DATASET, out_dir / "gf5", gas_fraction=5.0)
        bias_1 = s1.loc[s1["model"] == "btfr", "bias_dex"].values[0]
        bias_5 = s2.loc[s2["model"] == "btfr", "bias_dex"].values[0]
        assert abs(bias_1 - bias_5) > 0.05, "gas_fraction has no effect on BTFR bias"

    def test_missing_csv_raises(self, out_dir):
        with pytest.raises(FileNotFoundError):
            run_blind_test(out_dir / "nonexistent.csv", out_dir)

    def test_reproducible_outputs(self, tmp_path):
        """Running twice with the same input must produce identical output."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        p1, s1 = run_blind_test(_DATASET, out1)
        p2, s2 = run_blind_test(_DATASET, out2)
        pd.testing.assert_frame_equal(p1, p2)
        pd.testing.assert_frame_equal(s1, s2)


class TestCLIMain:
    def test_main_runs_without_error(self, tmp_path):
        out = str(tmp_path / "out")
        main(["--csv", str(_DATASET), "--out", out])
        assert (tmp_path / "out" / "predictions.csv").exists()
        assert (tmp_path / "out" / "summary.csv").exists()

    def test_main_custom_a0(self, tmp_path):
        out = str(tmp_path / "out_a0")
        main(["--csv", str(_DATASET), "--out", out, "--a0", "1.0e-10"])
        assert (tmp_path / "out_a0" / "predictions.csv").exists()

    def test_main_custom_gas_fraction(self, tmp_path):
        out = str(tmp_path / "out_gf")
        main(["--csv", str(_DATASET), "--out", out, "--gas-fraction", "3.0"])
        assert (tmp_path / "out_gf" / "predictions.csv").exists()
