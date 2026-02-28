"""
tests/test_pilot_f3.py — Unit tests for the F3 pilot test (Phase 1).

Covers:
  - compute_f3: mathematical correctness, column contract, value range
  - run_pilot_f3: CSV output, statsmodels result, error handling
  - CLI (main): produces F3_values.csv, handles custom a0
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.pilot_f3_test import (
    PILOT_GALAXIES,
    F3_COLS,
    compute_f3,
    run_pilot_f3,
    main,
)

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DATASET = _REPO_ROOT / "data" / "little_things_global.csv"


# ---------------------------------------------------------------------------
# compute_f3 — unit tests
# ---------------------------------------------------------------------------

class TestComputeF3:
    @pytest.fixture(scope="class")
    def pilot_df(self):
        full = pd.read_csv(_DATASET)
        return full[full["galaxy_id"].isin(PILOT_GALAXIES)].reset_index(drop=True)

    def test_returns_correct_columns(self, pilot_df):
        result = compute_f3(pilot_df)
        assert list(result.columns) == F3_COLS

    def test_row_count_matches_input(self, pilot_df):
        result = compute_f3(pilot_df)
        assert len(result) == len(pilot_df)

    def test_f3_finite(self, pilot_df):
        result = compute_f3(pilot_df)
        assert result["F3"].notna().all()
        assert np.isfinite(result["F3"].values).all()

    def test_logV_model_finite(self, pilot_df):
        result = compute_f3(pilot_df)
        assert result["logV_model"].notna().all()

    def test_f3_equals_logVobs_minus_logV_model(self, pilot_df):
        """F3 must equal logVobs − logV_model to machine precision."""
        result = compute_f3(pilot_df)
        diff = result["logVobs"].values - result["logV_model"].values
        np.testing.assert_allclose(result["F3"].values, diff, atol=1e-5)

    def test_f3_range_dwarf_galaxies(self, pilot_df):
        """F3 for LITTLE THINGS dwarfs must lie in [-0.2, 0.3]."""
        result = compute_f3(pilot_df)
        assert result["F3"].between(-0.2, 0.3).all(), (
            f"F3 out of expected range: {result[['galaxy_id', 'F3']].to_dict('records')}"
        )

    def test_galaxy_ids_preserved(self, pilot_df):
        result = compute_f3(pilot_df)
        assert set(result["galaxy_id"]) == set(pilot_df["galaxy_id"])

    def test_a0_sensitivity(self, pilot_df):
        """Different a0 values must produce different logV_model (and F3)."""
        r1 = compute_f3(pilot_df, a0=1.2e-10)
        r2 = compute_f3(pilot_df, a0=1.0e-10)
        assert not np.allclose(r1["F3"].values, r2["F3"].values), (
            "a0 variation has no effect on F3"
        )


# ---------------------------------------------------------------------------
# run_pilot_f3 — integration tests
# ---------------------------------------------------------------------------

class TestRunPilotF3:
    def test_writes_f3_values_csv(self, tmp_path):
        f3_df, _ = run_pilot_f3(_DATASET, tmp_path)
        csv = tmp_path / "F3_values.csv"
        assert csv.exists(), "F3_values.csv not written"

    def test_csv_columns(self, tmp_path):
        run_pilot_f3(_DATASET, tmp_path)
        df = pd.read_csv(tmp_path / "F3_values.csv")
        assert list(df.columns) == F3_COLS

    def test_csv_row_count(self, tmp_path):
        run_pilot_f3(_DATASET, tmp_path)
        df = pd.read_csv(tmp_path / "F3_values.csv")
        assert len(df) == len(PILOT_GALAXIES)

    def test_pilot_galaxies_present_in_csv(self, tmp_path):
        run_pilot_f3(_DATASET, tmp_path)
        df = pd.read_csv(tmp_path / "F3_values.csv")
        assert set(df["galaxy_id"]) == set(PILOT_GALAXIES)

    def test_ols_result_has_params(self, tmp_path):
        _, ols = run_pilot_f3(_DATASET, tmp_path)
        # OLS fit with constant + 1 regressor → 2 params
        assert len(ols.params) == 2

    def test_ols_rsquared_finite(self, tmp_path):
        _, ols = run_pilot_f3(_DATASET, tmp_path)
        assert np.isfinite(ols.rsquared)
        assert 0.0 <= ols.rsquared <= 1.0

    def test_ols_slope_positive(self, tmp_path):
        """Slope of F3 ~ log_gbar expected positive for these 4 dwarfs."""
        _, ols = run_pilot_f3(_DATASET, tmp_path)
        slope = ols.params[1]
        assert slope > 0, f"Expected positive slope, got {slope:.4f}"

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_pilot_f3(tmp_path / "nonexistent.csv", tmp_path)

    def test_unknown_galaxies_raises(self, tmp_path):
        with pytest.raises(ValueError, match="None of"):
            run_pilot_f3(_DATASET, tmp_path, galaxies=["NGC9999", "NGC8888"])

    def test_custom_a0(self, tmp_path):
        """Custom a0 must still produce a valid CSV."""
        f3_df, _ = run_pilot_f3(_DATASET, tmp_path / "a0", a0=1.0e-10)
        assert len(f3_df) == len(PILOT_GALAXIES)

    def test_reproducible(self, tmp_path):
        """Two runs with identical inputs must produce identical F3 values."""
        df1, _ = run_pilot_f3(_DATASET, tmp_path / "r1")
        df2, _ = run_pilot_f3(_DATASET, tmp_path / "r2")
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# CLI main() — smoke tests
# ---------------------------------------------------------------------------

class TestCLIMain:
    def test_main_runs_without_error(self, tmp_path):
        main(["--csv", str(_DATASET), "--out", str(tmp_path)])
        assert (tmp_path / "F3_values.csv").exists()

    def test_main_custom_a0(self, tmp_path):
        main(["--csv", str(_DATASET), "--out", str(tmp_path), "--a0", "1.0e-10"])
        assert (tmp_path / "F3_values.csv").exists()

    def test_main_default_galaxies_are_pilot(self, tmp_path):
        main(["--csv", str(_DATASET), "--out", str(tmp_path)])
        df = pd.read_csv(tmp_path / "F3_values.csv")
        assert set(df["galaxy_id"]) == set(PILOT_GALAXIES)
