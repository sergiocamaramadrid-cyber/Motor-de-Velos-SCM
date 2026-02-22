"""Tests for scripts/compute_residuals_binned.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts and src packages importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.compute_residuals_binned import (
    A0_SI,
    G0_BOUNDS,
    G0_BORDER_TOL,
    bin_residuals,
    compute_log_residuals,
    fit_g0,
    g_pred_rar,
    load_dataset,
    nu_rar,
    print_telemetry,
    reprint_notarial,
    write_csv,
)
from src.g0_utils import g0_touches_bounds


# ---------------------------------------------------------------------------
# nu_rar
# ---------------------------------------------------------------------------

class TestNuRar:
    def test_deep_mond_limit(self):
        """In deep-MOND (x → 0), ν ≈ 1/√x."""
        x = np.array([1e-6])
        nu = nu_rar(x)
        expected = 1.0 / np.sqrt(x)
        np.testing.assert_allclose(nu, expected, rtol=1e-3)

    def test_newtonian_limit(self):
        """In Newtonian regime (x → ∞), ν → 1."""
        x = np.array([1e6])
        nu = nu_rar(x)
        np.testing.assert_allclose(nu, 1.0, rtol=1e-4)

    def test_always_geq_one(self):
        x = np.logspace(-4, 4, 50)
        assert np.all(nu_rar(x) >= 1.0)


# ---------------------------------------------------------------------------
# g_pred_rar
# ---------------------------------------------------------------------------

class TestGPredRar:
    def test_g_pred_geq_g_bar(self):
        """Predicted acceleration must be ≥ g_bar (ν ≥ 1)."""
        g_bar = np.logspace(-14, -9, 100)
        g_pred = g_pred_rar(g_bar, a0=A0_SI)
        assert np.all(g_pred >= g_bar)

    def test_newtonian_limit(self):
        """When g_bar >> a0, g_pred ≈ g_bar."""
        g_bar = np.array([1e-5])  # >> a0 = 1.2e-10
        g_pred = g_pred_rar(g_bar, a0=A0_SI)
        np.testing.assert_allclose(g_pred, g_bar, rtol=1e-4)

    def test_scales_with_a0(self):
        """Doubling a0 changes g_pred (ν depends on g_bar/a0)."""
        g_bar = np.array([A0_SI])  # g_bar == a0 is the 'crossover' regime
        pred1 = g_pred_rar(g_bar, a0=A0_SI)
        pred2 = g_pred_rar(g_bar, a0=2 * A0_SI)
        # Values must differ by more than 1 % (avoid allclose atol trap)
        assert abs(pred1[0] - pred2[0]) / pred1[0] > 0.01


# ---------------------------------------------------------------------------
# compute_log_residuals
# ---------------------------------------------------------------------------

class TestComputeLogResiduals:
    def test_zero_residual_when_obs_equals_pred(self):
        g_bar = np.logspace(-13, -10, 20)
        g_obs = g_pred_rar(g_bar)
        delta = compute_log_residuals(g_bar, g_obs)
        np.testing.assert_allclose(delta, 0.0, atol=1e-10)

    def test_shape_preserved(self):
        g_bar = np.logspace(-13, -10, 50)
        g_obs = g_bar * 1.5
        delta = compute_log_residuals(g_bar, g_obs)
        assert delta.shape == g_bar.shape

    def test_positive_when_obs_gt_pred(self):
        g_bar = np.array([A0_SI])
        g_obs = g_pred_rar(g_bar) * 2.0
        delta = compute_log_residuals(g_bar, g_obs)
        assert delta[0] > 0.0


# ---------------------------------------------------------------------------
# fit_g0
# ---------------------------------------------------------------------------

class TestFitG0:
    def test_returns_required_keys(self):
        g_bar = np.logspace(-13, -10, 50)
        g_obs = g_pred_rar(g_bar, a0=A0_SI) * 1.0
        result = fit_g0(g_bar, g_obs)
        for key in ("g0_hat", "g0_lo", "g0_hi", "touches_lower", "touches_upper", "success"):
            assert key in result, f"Missing key: {key}"

    def test_g0_within_bounds(self):
        g_bar = np.logspace(-13, -10, 50)
        g_obs = g_pred_rar(g_bar, a0=A0_SI)
        result = fit_g0(g_bar, g_obs)
        assert G0_BOUNDS[0] <= result["g0_hat"] <= G0_BOUNDS[1]

    def test_recovers_a0_approximately(self):
        """When g_obs = g_pred(a0=A0_SI), the fitted g0 should be near A0_SI."""
        g_bar = np.logspace(-13, -9, 200)
        g_obs = g_pred_rar(g_bar, a0=A0_SI)
        result = fit_g0(g_bar, g_obs)
        # Should be within 50 % of the true value (MLE from exact predictions)
        assert result["g0_hat"] == pytest.approx(A0_SI, rel=0.5)

    def test_touches_lower_flag(self):
        """With data that pushes g0 to the lower bound, flag is set."""
        # Tiny g_bar values make g_pred >> g_obs, pushing a0 to minimum
        g_bar = np.logspace(-14, -13, 20)
        g_obs = g_bar * 1.001  # barely above g_bar, far below g_pred
        custom_bounds = (1e-12, 1e-8)
        result = fit_g0(g_bar, g_obs, bounds=custom_bounds)
        assert result["touches_lower"] is True

    def test_confidence_interval_ordering(self):
        g_bar = np.logspace(-13, -10, 50)
        g_obs = g_pred_rar(g_bar, a0=A0_SI)
        result = fit_g0(g_bar, g_obs)
        assert result["g0_lo"] <= result["g0_hat"] <= result["g0_hi"]

    def test_lower_bound_is_1e_minus_16(self):
        """v0.2b fix: lower bound must be 1e-16 (log10 = -16.0)."""
        assert G0_BOUNDS[0] == pytest.approx(1e-16)

    def test_upper_bound_is_1e_minus_8(self):
        """Upper bound must be 1e-8 (log10 = -8.0)."""
        assert G0_BOUNDS[1] == pytest.approx(1e-8)

    def test_no_longer_clamps_at_1e_minus_12(self):
        """After the v0.2b fix, g0_hat should not be clamped at 1e-12."""
        # Data at very deep MOND regime pushes optimal g0 well below 1e-12
        g_bar = np.logspace(-14, -13, 40)
        g_obs = g_bar * 1.001  # obs barely above g_bar (deep-field under-prediction)
        result = fit_g0(g_bar, g_obs)
        # With the expanded bounds, g0_hat is free to fall below 1e-12
        assert result["g0_hat"] < 1e-12

    def test_optimizer_uses_log10_bounds(self):
        """v0.2b: optimizer bounds are in log10 space (-16.0, -8.0), not linear."""
        assert np.log10(G0_BOUNDS[0]) == pytest.approx(-16.0)
        assert np.log10(G0_BOUNDS[1]) == pytest.approx(-8.0)

    def test_touches_lower_uses_log10_proximity(self):
        """v0.2b: touches_lower flag uses log10 proximity, not linear fraction."""
        g_bar = np.logspace(-14, -13, 30)
        g_obs = g_bar * 1.001
        custom_bounds = (1e-12, 1e-8)
        result = fit_g0(g_bar, g_obs, bounds=custom_bounds)
        log10_g0 = np.log10(result["g0_hat"])
        log10_lo = np.log10(custom_bounds[0])
        assert abs(log10_g0 - log10_lo) < G0_BORDER_TOL
        assert result["touches_lower"] is True

    def test_touches_upper_uses_log10_proximity(self):
        """v0.2b: touches_upper flag uses log10 proximity, not linear fraction."""
        g_bar = np.logspace(-9, -8.1, 20)
        g_obs = g_pred_rar(g_bar, a0=1e-8)  # data generated at a0 near upper bound
        result = fit_g0(g_bar, g_obs)
        if result["touches_upper"]:
            log10_g0 = np.log10(result["g0_hat"])
            log10_hi = np.log10(G0_BOUNDS[1])
            assert abs(log10_g0 - log10_hi) < G0_BORDER_TOL

    def test_border_tol_is_log10_scale(self):
        """v0.2b: G0_BORDER_TOL must be 0.05 (log10 decades, not linear fraction)."""
        assert G0_BORDER_TOL == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# g0_touches_bounds (imported from src.g0_utils)
# ---------------------------------------------------------------------------

class TestG0TouchesBounds:
    def test_touches_lower_when_at_lower_bound(self):
        lo, hi = 1e-16, 1e-8
        lo_tl, lo_tu = g0_touches_bounds(lo, (lo, hi))
        assert lo_tl is True
        assert lo_tu is False

    def test_touches_upper_when_at_upper_bound(self):
        lo, hi = 1e-16, 1e-8
        tl, tu = g0_touches_bounds(hi, (lo, hi))
        assert tl is False
        assert tu is True

    def test_no_touch_in_interior(self):
        lo, hi = 1e-16, 1e-8
        mid = 1e-12
        tl, tu = g0_touches_bounds(mid, (lo, hi))
        assert tl is False
        assert tu is False

    def test_nonfinite_g0_hat_returns_false_false(self):
        bounds = (1e-16, 1e-8)
        assert g0_touches_bounds(float("nan"), bounds) == (False, False)
        assert g0_touches_bounds(float("inf"), bounds) == (False, False)
        assert g0_touches_bounds(float("-inf"), bounds) == (False, False)

    def test_zero_g0_hat_returns_false_false(self):
        bounds = (1e-16, 1e-8)
        assert g0_touches_bounds(0.0, bounds) == (False, False)
        assert g0_touches_bounds(-1e-10, bounds) == (False, False)

    def test_invalid_bounds_lo_zero_raises(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            g0_touches_bounds(1e-12, (0.0, 1e-8))

    def test_invalid_bounds_lo_negative_raises(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            g0_touches_bounds(1e-12, (-1e-16, 1e-8))

    def test_invalid_bounds_lo_ge_hi_raises(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            g0_touches_bounds(1e-12, (1e-8, 1e-16))
        with pytest.raises(ValueError, match="Invalid bounds"):
            g0_touches_bounds(1e-12, (1e-8, 1e-8))

    def test_invalid_bounds_nonfinite_raises(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            g0_touches_bounds(1e-12, (float("nan"), 1e-8))
        with pytest.raises(ValueError, match="Invalid bounds"):
            g0_touches_bounds(1e-12, (1e-16, float("inf")))

    def test_custom_tol_wider(self):
        lo, hi = 1e-16, 1e-8
        g0 = 10 ** (-16 + 0.1)
        tl_narrow, _ = g0_touches_bounds(g0, (lo, hi), tol_log10=0.05)
        tl_wide, _ = g0_touches_bounds(g0, (lo, hi), tol_log10=0.15)
        assert tl_narrow is False
        assert tl_wide is True

    def test_returns_bools(self):
        tl, tu = g0_touches_bounds(1e-12, (1e-16, 1e-8))
        assert type(tl) is bool
        assert type(tu) is bool


class TestPrintTelemetry:
    def _make_fit(self, touches_lower=False, touches_upper=False):
        return {
            "g0_hat": 1.2e-10,
            "g0_lo": 1.0e-10,
            "g0_hi": 1.4e-10,
            "touches_lower": touches_lower,
            "touches_upper": touches_upper,
        }

    def test_scm_v02_line_present(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(), g_bar)
        captured = capsys.readouterr()
        assert "[SCM v0.2]" in captured.out
        assert "g0_hat=" in captured.out

    def test_info_quantiles_line_present(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(), g_bar)
        captured = capsys.readouterr()
        assert "[INFO] g_bar quantiles:" in captured.out
        assert "q10=" in captured.out
        assert "q50=" in captured.out
        assert "q90=" in captured.out

    def test_warn_lower_printed(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(touches_lower=True), g_bar)
        captured = capsys.readouterr()
        assert "[WARN]" in captured.out
        assert "lower" in captured.out.lower()

    def test_warn_upper_printed(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(touches_upper=True), g_bar)
        captured = capsys.readouterr()
        assert "[WARN]" in captured.out
        assert "upper" in captured.out.lower()

    def test_no_warn_when_not_touching(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(), g_bar)
        captured = capsys.readouterr()
        assert "[WARN]" not in captured.out

    def test_warn_lower_message_exact(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(touches_lower=True), g_bar)
        captured = capsys.readouterr()
        assert "[WARN] g0_hat hit LOWER bound again" in captured.out

    def test_warn_upper_message_exact(self, capsys):
        g_bar = np.logspace(-13, -10, 30)
        print_telemetry(self._make_fit(touches_upper=True), g_bar)
        captured = capsys.readouterr()
        assert "[WARN] g0_hat hit UPPER bound" in captured.out


# ---------------------------------------------------------------------------
# bin_residuals
# ---------------------------------------------------------------------------

class TestBinResiduals:
    def _make_data(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        g_bar = np.logspace(-14, -9, n)
        delta = rng.normal(0.0, 0.1, size=n)
        return g_bar, delta

    def test_returns_dataframe(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta)
        for col in ("g_bar_center", "median_residual", "mad_residual", "count"):
            assert col in result.columns

    def test_count_geq_min_count(self):
        g_bar, delta = self._make_data()
        min_c = 5
        result = bin_residuals(g_bar, delta, min_count=min_c)
        assert (result["count"] >= min_c).all()

    def test_n_bins_at_most_requested(self):
        g_bar, delta = self._make_data()
        n_bins = 10
        result = bin_residuals(g_bar, delta, n_bins=n_bins)
        assert len(result) <= n_bins


# ---------------------------------------------------------------------------
# reprint_notarial
# ---------------------------------------------------------------------------

class TestReprintNotarial:
    def test_output_contains_header(self, capsys):
        df = pd.DataFrame({
            "g_bar_center": [1e-13, 1e-12],
            "median_residual": [-0.5, 0.1],
            "mad_residual": [0.05, 0.02],
            "count": [10, 12],
        })
        reprint_notarial(df, Path("results/diagnostics/residuals_binned_v02.csv"))
        out = capsys.readouterr().out
        assert "REPRINT NOTARIAL" in out
        assert "residuals_binned_v02.csv" in out

    def test_output_contains_data(self, capsys):
        df = pd.DataFrame({
            "g_bar_center": [1e-13],
            "median_residual": [-0.42],
            "mad_residual": [0.03],
            "count": [7],
        })
        reprint_notarial(df, Path("results/diagnostics/residuals_binned_v02.csv"))
        out = capsys.readouterr().out
        assert "-0.42" in out


# ---------------------------------------------------------------------------
# load_dataset and write_csv
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_loads_valid_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("g_bar,g_obs\n1e-13,2e-13\n2e-13,4e-13\n")
        df = load_dataset(csv)
        assert list(df.columns) == ["g_bar", "g_obs"]
        assert len(df) == 2

    def test_skips_comment_lines(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("# comment\ng_bar,g_obs\n1e-13,2e-13\n")
        df = load_dataset(csv)
        assert len(df) == 1

    def test_raises_on_missing_column(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("g_bar,something_else\n1e-13,2e-13\n")
        with pytest.raises(ValueError, match="g_obs"):
            load_dataset(csv)


class TestWriteCsv:
    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "dir" / "out.csv"
        df = pd.DataFrame({
            "g_bar_center": [1e-13],
            "median_residual": [-0.5],
            "mad_residual": [0.05],
            "count": [5],
        })
        write_csv(df, out, n_bins_effective=1)
        assert out.exists()

    def test_csv_has_data_row(self, tmp_path):
        out = tmp_path / "out.csv"
        df = pd.DataFrame({
            "g_bar_center": [1e-13],
            "median_residual": [-0.5],
            "mad_residual": [0.05],
            "count": [5],
        })
        write_csv(df, out, n_bins_effective=1)
        content = out.read_text()
        assert "g_bar_center" in content
        assert "1.000000e-13" in content


# ---------------------------------------------------------------------------
# Integration: main() end-to-end
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_runs_and_writes_csv(self, tmp_path, capsys):
        """main() should write the CSV and print telemetry + Reprint Notarial."""
        from scripts.compute_residuals_binned import main

        data_path = Path(__file__).parent.parent / "data" / "sparc_rar_sample.csv"
        out_path = tmp_path / "diagnostics" / "residuals_binned_v02.csv"

        main([
            "--data", str(data_path),
            "--out", str(out_path),
            "--bins", "10",
        ])

        assert out_path.exists()
        captured = capsys.readouterr()
        assert "[SCM v0.2]" in captured.out
        assert "[INFO] g_bar quantiles:" in captured.out
        assert "REPRINT NOTARIAL" in captured.out

    def test_main_fixed_a0_skips_fit(self, tmp_path, capsys):
        """When --a0 is given, no fitting telemetry is printed."""
        from scripts.compute_residuals_binned import main

        data_path = Path(__file__).parent.parent / "data" / "sparc_rar_sample.csv"
        out_path = tmp_path / "out.csv"

        main([
            "--data", str(data_path),
            "--out", str(out_path),
            "--a0", str(A0_SI),
        ])

        captured = capsys.readouterr()
        # No fit → no [SCM v0.2] line
        assert "[SCM v0.2]" not in captured.out
        # Reprint Notarial still happens
        assert "REPRINT NOTARIAL" in captured.out
