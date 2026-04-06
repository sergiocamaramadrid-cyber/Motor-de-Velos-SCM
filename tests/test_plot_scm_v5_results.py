"""
tests/test_plot_scm_v5_results.py
Tests for scripts/plot_scm_v5_results.py.

Covers:
  1.  build_data            — analytic approximation data
  2.  plot_results          — figure generation (panels, save)
  3.  main()  CLI           — analytic and ODE-backed paths
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_scm_v5_results import (
    _HELIOPAUSE_LABEL_AU,
    _N_POINTS_DEFAULT,
    _T_YEARS_DEFAULT,
    build_data,
    main,
    plot_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(**kw):
    return build_data(**kw)


# ---------------------------------------------------------------------------
# 1. build_data
# ---------------------------------------------------------------------------

class TestBuildData:
    def test_required_keys(self):
        d = _data()
        assert {"t_years", "r_AU", "v_kms", "v_sw_kms", "P_net_W"}.issubset(d)

    def test_default_length(self):
        d = _data()
        for key in ("t_years", "r_AU", "v_kms", "v_sw_kms", "P_net_W"):
            assert len(d[key]) == _N_POINTS_DEFAULT

    def test_custom_n_points(self):
        d = _data(n_points=200)
        assert len(d["t_years"]) == 200

    def test_t_years_range(self):
        d = _data()
        assert d["t_years"][0] == pytest.approx(0.0, abs=1e-10)
        assert d["t_years"][-1] == pytest.approx(_T_YEARS_DEFAULT, rel=1e-6)

    def test_custom_t_years_total(self):
        d = _data(t_years_total=10.0)
        assert d["t_years"][-1] == pytest.approx(10.0, rel=1e-6)

    def test_r_AU_starts_near_1(self):
        d = _data()
        assert d["r_AU"][0] == pytest.approx(1.0, abs=0.1)

    def test_r_AU_terminal_near_680(self):
        d = _data()
        assert 600.0 < d["r_AU"][-1] < 800.0

    def test_r_AU_monotone_increasing(self):
        d = _data()
        assert np.all(np.diff(d["r_AU"]) > 0)

    def test_v_kms_starts_near_30(self):
        d = _data()
        assert d["v_kms"][0] == pytest.approx(30.0, abs=5.0)

    def test_v_kms_terminal_near_318(self):
        d = _data()
        assert 290.0 < d["v_kms"][-1] < 340.0

    def test_v_kms_monotone_increasing(self):
        d = _data()
        assert np.all(np.diff(d["v_kms"]) >= 0)

    def test_v_sw_kms_has_solar_wind_regime(self):
        """Solar-wind speed should be ~400 km/s near t=0."""
        d = _data()
        assert d["v_sw_kms"][0] == pytest.approx(400.0, abs=100.0)

    def test_v_sw_kms_drops_to_ism_regime(self):
        """Beyond heliopause (~7-8 yr into the run) speed should be ~25 km/s."""
        d = _data(t_years_total=20.0, n_points=1000)
        # Take last 10 % of the run — well past the heliopause crossing
        tail = d["v_sw_kms"][int(0.9 * len(d["v_sw_kms"])):]
        assert np.all(tail < 50.0)

    def test_P_net_always_positive(self):
        d = _data()
        assert np.all(d["P_net_W"] > 0)

    def test_all_finite(self):
        d = _data()
        for key, arr in d.items():
            assert np.all(np.isfinite(arr)), f"Non-finite in '{key}'"


# ---------------------------------------------------------------------------
# 2. plot_results
# ---------------------------------------------------------------------------

class TestPlotResults:
    def _d(self):
        return _data(n_points=200)

    def test_returns_figure(self):
        fig = plot_results(self._d())
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_three_axes(self):
        fig = plot_results(self._d())
        assert len(fig.axes) == 3
        plt.close("all")

    def test_save_to_file(self, tmp_path):
        out = tmp_path / "v5.png"
        fig = plot_results(self._d(), out_path=out)
        plt.close("all")
        assert out.exists() and out.stat().st_size > 0

    def test_save_respects_format(self, tmp_path):
        out = tmp_path / "v5.pdf"
        fig = plot_results(self._d(), out_path=out)
        plt.close("all")
        assert out.exists()

    def test_no_out_does_not_write_file(self, tmp_path):
        out = tmp_path / "shouldnotexist.png"
        plot_results(self._d(), out_path=None)
        plt.close("all")
        assert not out.exists()

    def test_y_axis_labels_present(self):
        fig = plot_results(self._d())
        labels = [ax.get_ylabel() for ax in fig.axes]
        plt.close("all")
        assert any("UA" in lbl or "Distancia" in lbl for lbl in labels)
        assert any("km/s" in lbl or "Velocidad" in lbl for lbl in labels)
        assert any("W" in lbl or "Potencia" in lbl for lbl in labels)

    def test_p_net_positive_uses_log_scale(self):
        """When P_net > 0 everywhere, yscale should be 'log'."""
        d = self._d()
        assert np.all(d["P_net_W"] > 0)
        fig = plot_results(d)
        ax3 = fig.axes[2]
        plt.close("all")
        assert ax3.get_yscale() == "log"

    def test_p_net_with_negatives_uses_symlog(self):
        d = self._d()
        d["P_net_W"] = d["P_net_W"].copy()
        d["P_net_W"][10] = -500.0      # inject a negative value
        fig = plot_results(d)
        ax3 = fig.axes[2]
        plt.close("all")
        assert ax3.get_yscale() == "symlog"

    def test_heliopause_vline_present_when_reachable(self):
        """A vertical line should be drawn when the probe crosses the heliopause."""
        d = _data(t_years_total=20.0, n_points=500)
        fig = plot_results(d)
        ax1 = fig.axes[0]
        # Collect x-coordinates of all vertical lines in panel 1
        vlines_x = [line.get_xdata()[0] for line in ax1.lines
                    if len(line.get_xdata()) == 2 and
                    line.get_xdata()[0] == line.get_xdata()[1]]
        plt.close("all")
        assert len(vlines_x) >= 1


# ---------------------------------------------------------------------------
# 3. main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_figure(self):
        fig = main(["--n-points", "100"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_three_axes(self):
        fig = main(["--n-points", "100"])
        assert len(fig.axes) == 3
        plt.close("all")

    def test_writes_output_file(self, tmp_path):
        out = str(tmp_path / "v5.png")
        fig = main(["--n-points", "100", "--out", out])
        plt.close("all")
        assert Path(out).exists() and Path(out).stat().st_size > 0

    def test_out_not_given_no_file_created(self, tmp_path):
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            main(["--n-points", "100"])
        finally:
            os.chdir(old_cwd)
        plt.close("all")
        # No PNG file should land in tmp_path
        assert not any(tmp_path.glob("*.png"))

    def test_custom_n_points(self):
        fig = main(["--n-points", "50"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_simulate_flag(self):
        """--simulate should call the real V5 ODE and still return a Figure."""
        fig = main(["--simulate", "--t-days", "5", "--n-steps", "100"])
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close("all")

    def test_simulate_r_au_grows(self):
        """The simulated trajectory must move outward."""
        from scripts.plot_scm_v5_results import build_data_from_simulation
        d = build_data_from_simulation(t_days=5, n_steps=100)
        assert d["r_AU"][-1] > 1.0

    def test_simulate_returns_required_keys(self):
        from scripts.plot_scm_v5_results import build_data_from_simulation
        d = build_data_from_simulation(t_days=5, n_steps=100)
        assert {"t_years", "r_AU", "v_kms", "v_sw_kms", "P_net_W"}.issubset(d)

    def test_simulate_all_finite(self):
        from scripts.plot_scm_v5_results import build_data_from_simulation
        d = build_data_from_simulation(t_days=5, n_steps=100)
        for key, arr in d.items():
            assert np.all(np.isfinite(arr)), f"Non-finite in '{key}'"
