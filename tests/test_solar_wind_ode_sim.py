"""
tests/test_solar_wind_ode_sim.py — Tests for scripts/solar_wind_ode_sim.py.

Covers:
  1. accel_net()    — physics model, gravitational & drag components.
  2. make_derivs()  — ODE right-hand-side structure.
  3. run_simulation() — integrator output structure and physics.
  4. plot_results() — figure generation.
  5. main() CLI    — end-to-end invocation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.solar_wind_ode_sim import (
    AU,
    ETA_DEFAULT,
    G,
    M_SOL,
    MASS_PROBE_DEFAULT,
    R0_DEFAULT,
    R_BUBBLE_DEFAULT,
    RHO0_DEFAULT,
    V0_DEFAULT,
    V_SW_DEFAULT,
    accel_net,
    main,
    make_derivs,
    plot_results,
    run_simulation,
)


# ---------------------------------------------------------------------------
# 1. accel_net()
# ---------------------------------------------------------------------------

class TestAccelNet:
    def test_returns_float(self):
        a = accel_net(AU, 0.0)
        assert isinstance(a, float)

    def test_gravity_only_when_v_equals_v_sw(self):
        """At v = v_sw drag is zero; only gravity acts."""
        a = accel_net(AU, V_SW_DEFAULT)
        expected = -G * M_SOL / AU ** 2
        assert a == pytest.approx(expected, rel=1e-9)

    def test_gravity_only_when_v_above_v_sw(self):
        """Probe faster than solar wind → no drag, only gravity."""
        a = accel_net(AU, V_SW_DEFAULT + 1.0)
        expected = -G * M_SOL / AU ** 2
        assert a == pytest.approx(expected, rel=1e-9)

    def test_drag_adds_positive_thrust(self):
        """Slow probe at 1 AU: drag must exceed gravity (net positive accel)."""
        a_zero_v = accel_net(AU, 0.0)
        a_grav_only = -G * M_SOL / AU ** 2
        # Drag contribution is positive
        assert a_zero_v > a_grav_only

    def test_drag_decreases_with_distance(self):
        """Drag ∝ 1/r² — more distant → less drag."""
        a1 = accel_net(AU, 0.0)
        a5 = accel_net(5 * AU, 0.0)
        # gravity also changes, but drag drop dominates at these params
        rho1 = RHO0_DEFAULT
        rho5 = RHO0_DEFAULT * (1.0 / 5.0) ** 2
        A = np.pi * R_BUBBLE_DEFAULT ** 2
        F1 = 0.5 * ETA_DEFAULT * rho1 * A * V_SW_DEFAULT ** 2
        F5 = 0.5 * ETA_DEFAULT * rho5 * A * V_SW_DEFAULT ** 2
        assert F5 < F1

    def test_no_nan_or_inf(self):
        a = accel_net(AU, V0_DEFAULT)
        assert math.isfinite(a)

    def test_manual_drag_calculation(self):
        """Verify drag component against hand computation."""
        r = AU
        v = 0.0
        rho = RHO0_DEFAULT * (AU / r) ** 2
        A = np.pi * R_BUBBLE_DEFAULT ** 2
        F_drag = 0.5 * ETA_DEFAULT * rho * A * (V_SW_DEFAULT - v) ** 2
        a_drag = F_drag / MASS_PROBE_DEFAULT
        a_grav = -G * M_SOL / r ** 2
        expected = a_drag + a_grav
        assert accel_net(r, v) == pytest.approx(expected, rel=1e-10)

    def test_zero_eta_gives_gravity_only(self):
        a = accel_net(AU, 0.0, eta=0.0)
        expected = -G * M_SOL / AU ** 2
        assert a == pytest.approx(expected, rel=1e-9)

    def test_larger_bubble_increases_drag(self):
        a_small = accel_net(AU, 0.0, R_bubble=10_000.0)
        a_large = accel_net(AU, 0.0, R_bubble=200_000.0)
        assert a_large > a_small

    def test_accel_negative_far_from_sun_slow_wind(self):
        """Very far from Sun with tiny bubble → gravity dominates → net negative."""
        a = accel_net(100 * AU, 0.0, R_bubble=1.0, rho0=1e-30)
        assert a < 0.0


# ---------------------------------------------------------------------------
# 2. make_derivs()
# ---------------------------------------------------------------------------

class TestMakeDerivs:
    def test_returns_callable(self):
        derivs = make_derivs()
        assert callable(derivs)

    def test_output_length(self):
        derivs = make_derivs()
        out = derivs([AU, V0_DEFAULT], 0.0)
        assert len(out) == 2

    def test_first_element_is_velocity(self):
        """dr/dt = v"""
        derivs = make_derivs()
        v = V0_DEFAULT
        out = derivs([AU, v], 0.0)
        assert out[0] == pytest.approx(v, rel=1e-10)

    def test_second_element_matches_accel_net(self):
        """dv/dt = accel_net(r, v)"""
        derivs = make_derivs()
        r, v = AU, V0_DEFAULT
        out = derivs([r, v], 0.0)
        expected_a = accel_net(r, v)
        assert out[1] == pytest.approx(expected_a, rel=1e-10)

    def test_custom_params_propagate(self):
        """Parameters passed to make_derivs affect dv/dt."""
        d1 = make_derivs(R_bubble=10_000.0)
        d2 = make_derivs(R_bubble=100_000.0)
        state = [AU, V0_DEFAULT]
        a1 = d1(state, 0.0)[1]
        a2 = d2(state, 0.0)[1]
        assert a2 > a1


# ---------------------------------------------------------------------------
# 3. run_simulation()
# ---------------------------------------------------------------------------

class TestRunSimulation:
    _SHORT = 10 * 86400   # 10 days in seconds

    def test_returns_required_keys(self):
        res = run_simulation(self._SHORT, n_steps=200)
        required = {"t", "r", "v", "r_au", "r_final_au", "v_final_ms", "delta_v_kms"}
        assert required.issubset(res.keys())

    def test_array_lengths_match_n_steps(self):
        n = 300
        res = run_simulation(self._SHORT, n_steps=n)
        assert len(res["t"]) == n
        assert len(res["r"]) == n
        assert len(res["v"]) == n
        assert len(res["r_au"]) == n

    def test_initial_conditions_preserved(self):
        res = run_simulation(self._SHORT, n_steps=200)
        assert res["r"][0] == pytest.approx(R0_DEFAULT, rel=1e-6)
        assert res["v"][0] == pytest.approx(V0_DEFAULT, rel=1e-6)

    def test_r_au_consistent_with_r(self):
        res = run_simulation(self._SHORT, n_steps=200)
        np.testing.assert_allclose(res["r"] / AU, res["r_au"], rtol=1e-10)

    def test_r_final_au_matches_array(self):
        res = run_simulation(self._SHORT, n_steps=200)
        assert res["r_final_au"] == pytest.approx(res["r_au"][-1], rel=1e-10)

    def test_v_final_ms_matches_array(self):
        res = run_simulation(self._SHORT, n_steps=200)
        assert res["v_final_ms"] == pytest.approx(res["v"][-1], rel=1e-10)

    def test_delta_v_consistent(self):
        res = run_simulation(self._SHORT, n_steps=200)
        expected_dv = (res["v"][-1] - V0_DEFAULT) / 1000.0
        assert res["delta_v_kms"] == pytest.approx(expected_dv, rel=1e-10)

    def test_positive_delta_v_with_default_params(self):
        """Default Plasma Magnet parameters must produce net delta-v > 0."""
        res = run_simulation(self._SHORT, n_steps=200)
        assert res["delta_v_kms"] > 0.0

    def test_sonde_moves_outward(self):
        res = run_simulation(self._SHORT, n_steps=200)
        assert res["r_final_au"] > 1.0

    def test_all_values_finite(self):
        res = run_simulation(self._SHORT, n_steps=200)
        for key in ("r", "v", "r_au"):
            assert np.all(np.isfinite(res[key])), f"Non-finite values in '{key}'"

    def test_heavier_sonde_less_delta_v(self):
        r1 = run_simulation(self._SHORT, n_steps=200, mass=10.0)
        r2 = run_simulation(self._SHORT, n_steps=200, mass=500.0)
        assert r1["delta_v_kms"] > r2["delta_v_kms"]

    def test_larger_bubble_more_delta_v(self):
        r1 = run_simulation(self._SHORT, n_steps=200, R_bubble=10_000.0)
        r2 = run_simulation(self._SHORT, n_steps=200, R_bubble=200_000.0)
        assert r2["delta_v_kms"] > r1["delta_v_kms"]

    def test_one_year_simulation(self):
        t_year = 365 * 86400
        res = run_simulation(t_year, n_steps=10_000)
        assert math.isfinite(res["r_final_au"])
        assert math.isfinite(res["delta_v_kms"])
        assert res["r_final_au"] > 1.0

    def test_n_steps_respected(self):
        for n in (100, 500, 1000):
            res = run_simulation(self._SHORT, n_steps=n)
            assert len(res["t"]) == n


# ---------------------------------------------------------------------------
# 4. plot_results()
# ---------------------------------------------------------------------------

class TestPlotResults:
    def _res(self):
        return run_simulation(10 * 86400, n_steps=200)

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        fig = plot_results(self._res())
        assert fig is not None
        plt.close("all")

    def test_figure_has_three_axes(self):
        import matplotlib.pyplot as plt
        fig = plot_results(self._res())
        assert len(fig.axes) == 3
        plt.close("all")

    def test_save_figure_to_file(self, tmp_path):
        import matplotlib.pyplot as plt
        out_file = tmp_path / "ode_sim.png"
        plot_results(self._res(), out_path=out_file)
        plt.close("all")
        assert out_file.exists()
        assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# 5. main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_returns_dict(self):
        res = main(["--no-plot", "--t-years", "0.05"])
        assert isinstance(res, dict)

    def test_main_required_keys(self):
        res = main(["--no-plot", "--t-years", "0.05"])
        assert "delta_v_kms" in res
        assert "r_final_au" in res
        assert "v_final_ms" in res

    def test_main_positive_delta_v(self):
        res = main(["--no-plot", "--t-years", "0.1"])
        assert res["delta_v_kms"] > 0.0

    def test_main_writes_output_files(self, tmp_path):
        main(["--t-years", "0.05", "--out", str(tmp_path)])
        summary = tmp_path / "solar_wind_ode_summary.txt"
        figure = tmp_path / "solar_wind_ode_sim.png"
        assert summary.exists()
        assert figure.exists()
        content = summary.read_text(encoding="utf-8")
        assert "Delta-V" in content

    def test_main_no_plot_skips_figure(self, tmp_path):
        main(["--t-years", "0.05", "--no-plot", "--out", str(tmp_path)])
        figure = tmp_path / "solar_wind_ode_sim.png"
        assert not figure.exists()

    def test_main_custom_mass_param(self):
        r_light = main(["--no-plot", "--t-years", "0.1", "--mass", "10"])
        r_heavy = main(["--no-plot", "--t-years", "0.1", "--mass", "500"])
        assert r_light["delta_v_kms"] > r_heavy["delta_v_kms"]

    def test_main_custom_r_bubble_param(self):
        r_small = main(["--no-plot", "--t-years", "0.1", "--r-bubble", "10000"])
        r_large = main(["--no-plot", "--t-years", "0.1", "--r-bubble", "200000"])
        assert r_large["delta_v_kms"] > r_small["delta_v_kms"]

    def test_main_one_year_default(self, tmp_path):
        res = main(["--no-plot", "--out", str(tmp_path)])
        assert res["r_final_au"] > 1.0
