"""
tests/test_scm_analysis.py — unit tests for SCM v0.2 analysis pipeline.

Validates:
- Telemetry output format ([SCM v0.2], [INFO], [WARN])
- Residuals CSV columns and content
- Boundary-warning logic
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pytest

from src.scm_models import fit_g0, scm_g_obs
from src.scm_analysis import (
    _N_BINS,
    _print_telemetry,
    _synthetic_dataset,
    _write_residuals_csv,
    reprint_residuals,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    return _synthetic_dataset(rng=np.random.default_rng(0))


@pytest.fixture()
def fit_result(synthetic_data):
    g_bar, g_obs = synthetic_data
    return g_bar, g_obs, fit_g0(g_bar, g_obs)


# ---------------------------------------------------------------------------
# scm_models
# ---------------------------------------------------------------------------

class TestScmGObs:
    def test_scalar_positive(self):
        g = scm_g_obs(1e-10, 1.2e-10)
        assert g > 1e-10

    def test_array(self):
        g_bar = np.array([1e-12, 1e-11, 1e-10])
        result = scm_g_obs(g_bar, 1.2e-10)
        assert result.shape == (3,)
        assert np.all(result >= g_bar)

    def test_high_acceleration_limit(self):
        # When g_bar >> g0 the interpolation returns ~ g_bar
        g_bar = np.array([1e-6])
        g0 = 1.2e-10
        result = scm_g_obs(g_bar, g0)
        np.testing.assert_allclose(result, g_bar, rtol=1e-4)


class TestFitG0:
    def test_recovers_true_g0(self, synthetic_data):
        g_bar, g_obs = synthetic_data
        result = fit_g0(g_bar, g_obs)
        # True g0 = 1.2e-10; allow 10 % tolerance
        assert abs(result["g0_hat"] / 1.2e-10 - 1.0) < 0.10

    def test_residuals_shape(self, synthetic_data):
        g_bar, g_obs = synthetic_data
        result = fit_g0(g_bar, g_obs)
        assert result["residuals"].shape == g_bar.shape

    def test_at_bound_false_for_good_data(self, synthetic_data):
        g_bar, g_obs = synthetic_data
        result = fit_g0(g_bar, g_obs)
        assert not result["at_bound"]

    def test_at_bound_true_when_forced(self):
        # Force data that push g0 to the lower bound
        rng = np.random.default_rng(1)
        g_bar = 10.0 ** rng.uniform(-13, -12, size=200)
        # Use an extremely small true g0 (below default lower bound)
        g0_tiny = 1e-12
        x = np.sqrt(g_bar / g0_tiny)
        g_obs = g_bar / (1.0 - np.exp(-x)) * 10.0 ** rng.normal(0, 0.05, 200)
        result = fit_g0(g_bar, g_obs)
        assert result["at_bound"]


# ---------------------------------------------------------------------------
# Telemetry output
# ---------------------------------------------------------------------------

class TestPrintTelemetry:
    def test_scm_tag_present(self, fit_result, caplog):
        g_bar, g_obs, fit = fit_result
        with caplog.at_level(logging.INFO):
            _print_telemetry(fit["g0_hat"], fit["g0_err"], g_bar, fit["at_bound"])
        scm_lines = [r for r in caplog.records if "[SCM v0.2]" in r.message]
        assert len(scm_lines) == 1

    def test_info_quantiles_tag_present(self, fit_result, caplog):
        g_bar, g_obs, fit = fit_result
        with caplog.at_level(logging.INFO):
            _print_telemetry(fit["g0_hat"], fit["g0_err"], g_bar, fit["at_bound"])
        info_lines = [r for r in caplog.records if "[INFO] g_bar quantiles:" in r.message]
        assert len(info_lines) == 1

    def test_no_warn_for_good_fit(self, fit_result, caplog):
        g_bar, g_obs, fit = fit_result
        with caplog.at_level(logging.WARNING):
            _print_telemetry(fit["g0_hat"], fit["g0_err"], g_bar, False)
        warn_lines = [r for r in caplog.records if "[WARN]" in r.message]
        assert len(warn_lines) == 0

    def test_warn_when_at_bound(self, fit_result, caplog):
        g_bar, g_obs, fit = fit_result
        with caplog.at_level(logging.WARNING):
            _print_telemetry(fit["g0_hat"], fit["g0_err"], g_bar, at_bound=True)
        warn_lines = [r for r in caplog.records if "[WARN]" in r.message]
        assert len(warn_lines) == 1


# ---------------------------------------------------------------------------
# Residuals CSV
# ---------------------------------------------------------------------------

class TestWriteResidualsCsv:
    def test_csv_created(self, fit_result, tmp_path):
        g_bar, g_obs, fit = fit_result
        csv_path = _write_residuals_csv(g_bar, fit["residuals"], tmp_path)
        assert csv_path.is_file()

    def test_csv_has_correct_columns(self, fit_result, tmp_path):
        g_bar, g_obs, fit = fit_result
        csv_path = _write_residuals_csv(g_bar, fit["residuals"], tmp_path)
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames) == {
                "bin_center_log10",
                "mean_residual",
                "std_residual",
                "n_points",
            }

    def test_csv_row_count(self, fit_result, tmp_path):
        g_bar, g_obs, fit = fit_result
        csv_path = _write_residuals_csv(g_bar, fit["residuals"], tmp_path, n_bins=_N_BINS)
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == _N_BINS

    def test_n_points_sum_equals_dataset_size(self, fit_result, tmp_path):
        g_bar, g_obs, fit = fit_result
        csv_path = _write_residuals_csv(g_bar, fit["residuals"], tmp_path)
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        total = sum(int(r["n_points"]) for r in rows)
        assert total == len(g_bar)


# ---------------------------------------------------------------------------
# Reprint helper
# ---------------------------------------------------------------------------

class TestReprintResiduals:
    def test_prints_table(self, fit_result, tmp_path, capsys):
        g_bar, g_obs, fit = fit_result
        csv_path = _write_residuals_csv(g_bar, fit["residuals"], tmp_path)
        reprint_residuals(csv_path)
        captured = capsys.readouterr()
        assert "Notarial de Residuos" in captured.out
        assert "bin_center_log10" in captured.out

    def test_missing_file(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR):
            reprint_residuals(tmp_path / "nonexistent.csv")
        assert any("[ERROR]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_creates_csv(self, tmp_path):
        csv_path = run_analysis(data_dir=tmp_path / "data", out_dir=tmp_path / "results")
        assert csv_path.name == "residuals_binned_v02.csv"
        assert csv_path.is_file()

    def test_csv_path_in_diagnostics_subdir(self, tmp_path):
        csv_path = run_analysis(data_dir=tmp_path / "data", out_dir=tmp_path / "results")
        assert csv_path.parent.name == "diagnostics"
