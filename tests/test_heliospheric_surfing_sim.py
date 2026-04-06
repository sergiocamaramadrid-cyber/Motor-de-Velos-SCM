"""
tests/test_heliospheric_surfing_sim.py — Tests for scripts/heliospheric_surfing_sim.py.

Covers:
  1. calcular_fuerza_surfing() — physical model and edge cases.
  2. run_simulation() — numerical integration structure and physics.
  3. plot_results() — figure generation without errors.
  4. main() CLI — end-to-end invocation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.heliospheric_surfing_sim import (
    AU,
    DISTANCIA_INICIAL_AU,
    EFICIENCIA_DEFAULT,
    MASS_PROBE_DEFAULT,
    R_BUBBLE_DEFAULT,
    RHO_EARTH_DEFAULT,
    V_SW_DEFAULT,
    VELOCIDAD_INICIAL_MS,
    calcular_fuerza_surfing,
    main,
    plot_results,
    run_simulation,
)


# ---------------------------------------------------------------------------
# 1. calcular_fuerza_surfing()
# ---------------------------------------------------------------------------

class TestCalcularFuerzaSurfing:
    def test_returns_float(self):
        f = calcular_fuerza_surfing(1.0, 0.0)
        assert isinstance(f, float)

    def test_positive_force_at_rest(self):
        """A stationary sonde at 1 AU must receive positive thrust."""
        f = calcular_fuerza_surfing(1.0, 0.0)
        assert f > 0.0

    def test_zero_force_when_faster_than_wind(self):
        """Sonde faster than v_sw → no net thrust."""
        f = calcular_fuerza_surfing(1.0, V_SW_DEFAULT + 1.0)
        assert f == 0.0

    def test_zero_force_at_exact_wind_speed(self):
        """v_rel = 0 → zero thrust."""
        f = calcular_fuerza_surfing(1.0, V_SW_DEFAULT)
        assert f == pytest.approx(0.0, abs=1e-20)

    def test_force_decreases_with_distance(self):
        """Thrust decreases with heliocentric distance (∝ 1/r²)."""
        f1 = calcular_fuerza_surfing(1.0, 0.0)
        f5 = calcular_fuerza_surfing(5.0, 0.0)
        assert f5 < f1

    def test_force_inverse_square_scaling(self):
        """Thrust at 2 AU should be ¼ of thrust at 1 AU (1/r² law)."""
        f1 = calcular_fuerza_surfing(1.0, 0.0)
        f2 = calcular_fuerza_surfing(2.0, 0.0)
        assert f2 == pytest.approx(f1 / 4.0, rel=1e-10)

    def test_force_increases_with_larger_bubble(self):
        """Larger bubble → more drag area → more thrust."""
        f_small = calcular_fuerza_surfing(1.0, 0.0, r_bubble=10_000.0)
        f_large = calcular_fuerza_surfing(1.0, 0.0, r_bubble=100_000.0)
        assert f_large > f_small

    def test_bubble_area_quadratic_scaling(self):
        """Thrust scales as R² (area of bubble)."""
        f1 = calcular_fuerza_surfing(1.0, 0.0, r_bubble=10_000.0)
        f2 = calcular_fuerza_surfing(1.0, 0.0, r_bubble=20_000.0)
        assert f2 == pytest.approx(4.0 * f1, rel=1e-10)

    def test_manual_calculation(self):
        """Verify against a hand-computed reference value."""
        dist_au = 1.0
        v_sonda = 0.0
        v_sw = 400_000.0
        rho_earth = 8e-20
        r_bubble = 50_000.0
        eficiencia = 0.5

        rho_local = rho_earth * (1.0 / dist_au) ** 2
        area = np.pi * r_bubble ** 2
        v_rel = v_sw - v_sonda
        expected = eficiencia * 0.5 * rho_local * area * v_rel ** 2

        f = calcular_fuerza_surfing(dist_au, v_sonda)
        assert f == pytest.approx(expected, rel=1e-10)

    def test_zero_efficiency_gives_zero_force(self):
        f = calcular_fuerza_surfing(1.0, 0.0, eficiencia=0.0)
        assert f == 0.0

    def test_custom_wind_speed(self):
        """Higher solar wind speed → more thrust (at same sonde speed)."""
        f_slow = calcular_fuerza_surfing(1.0, 0.0, v_sw=300_000.0)
        f_fast = calcular_fuerza_surfing(1.0, 0.0, v_sw=600_000.0)
        assert f_fast > f_slow

    def test_no_nan_or_inf(self):
        """Result must be finite for typical inputs."""
        f = calcular_fuerza_surfing(1.0, VELOCIDAD_INICIAL_MS)
        assert math.isfinite(f)


# ---------------------------------------------------------------------------
# 2. run_simulation()
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def _short_run(self, **kwargs):
        """10-day simulation with 1-day steps for fast tests."""
        defaults = dict(t_total_s=10 * 86400, dt_s=86400)
        defaults.update(kwargs)
        return run_simulation(**defaults)

    def test_returns_required_keys(self):
        res = self._short_run()
        required = {
            "tiempos", "distancia_au", "velocidad", "empuje",
            "delta_v_kms", "distancia_final_au", "velocidad_final_kms",
        }
        assert required.issubset(set(res.keys()))

    def test_array_lengths_match(self):
        res = self._short_run()
        n = len(res["tiempos"])
        assert len(res["distancia_au"]) == n
        assert len(res["velocidad"]) == n
        assert len(res["empuje"]) == n

    def test_initial_conditions_preserved(self):
        res = self._short_run()
        assert res["distancia_au"][0] == pytest.approx(DISTANCIA_INICIAL_AU)
        assert res["velocidad"][0] == pytest.approx(VELOCIDAD_INICIAL_MS)

    def test_velocity_increases_over_time(self):
        """Thrust from solar wind must accelerate the sonde."""
        res = self._short_run()
        assert res["velocidad"][-1] > res["velocidad"][0]

    def test_distance_increases_over_time(self):
        """Sonde must move outward during the simulation."""
        res = self._short_run()
        assert res["distancia_au"][-1] > res["distancia_au"][0]

    def test_delta_v_positive(self):
        res = self._short_run()
        assert res["delta_v_kms"] > 0.0

    def test_delta_v_consistent_with_arrays(self):
        res = self._short_run()
        expected_dv = (res["velocidad"][-1] - res["velocidad"][0]) / 1000.0
        assert res["delta_v_kms"] == pytest.approx(expected_dv, rel=1e-10)

    def test_thrust_nonnegative(self):
        """Thrust must never be negative."""
        res = self._short_run()
        assert np.all(res["empuje"] >= 0.0)

    def test_all_arrays_finite(self):
        res = self._short_run()
        for key in ("distancia_au", "velocidad", "empuje"):
            assert np.all(np.isfinite(res[key])), f"Non-finite values in '{key}'"

    def test_custom_mass_affects_acceleration(self):
        """Heavier sonde → less delta-v for same thrust."""
        res_light = self._short_run(mass_probe=10.0)
        res_heavy = self._short_run(mass_probe=500.0)
        assert res_light["delta_v_kms"] > res_heavy["delta_v_kms"]

    def test_larger_bubble_gives_more_delta_v(self):
        res_small = self._short_run(r_bubble=10_000.0)
        res_large = self._short_run(r_bubble=100_000.0)
        assert res_large["delta_v_kms"] > res_small["delta_v_kms"]

    def test_one_year_simulation(self):
        """Full 1-year simulation must complete and yield finite results."""
        t_year = 365 * 24 * 3600
        res = run_simulation(t_total_s=t_year, dt_s=86400)
        assert math.isfinite(res["distancia_final_au"])
        assert math.isfinite(res["velocidad_final_kms"])
        assert res["distancia_final_au"] > 1.0

    def test_timestep_independence_coarse_vs_fine(self):
        """Final distance should be roughly consistent for different dt."""
        t = 30 * 86400
        res_coarse = run_simulation(t_total_s=t, dt_s=86400)
        res_fine = run_simulation(t_total_s=t, dt_s=3600)
        # Euler integration: allow 5% difference between dt=1d and dt=1h
        assert abs(res_coarse["distancia_final_au"] - res_fine["distancia_final_au"]) < 0.05


# ---------------------------------------------------------------------------
# 3. plot_results()
# ---------------------------------------------------------------------------

class TestPlotResults:
    def _short_results(self):
        return run_simulation(t_total_s=10 * 86400, dt_s=86400)

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        res = self._short_results()
        fig = plot_results(res)
        assert fig is not None
        plt.close("all")

    def test_figure_has_three_axes(self):
        import matplotlib.pyplot as plt
        res = self._short_results()
        fig = plot_results(res)
        assert len(fig.axes) == 3
        plt.close("all")

    def test_save_figure_to_file(self, tmp_path):
        import matplotlib.pyplot as plt
        res = self._short_results()
        out_file = tmp_path / "test_fig.png"
        plot_results(res, out_path=out_file)
        plt.close("all")
        assert out_file.exists()
        assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# 4. main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_returns_dict(self):
        res = main(["--no-plot", "--t-years", "0.05"])
        assert isinstance(res, dict)

    def test_main_required_keys(self):
        res = main(["--no-plot", "--t-years", "0.05"])
        assert "delta_v_kms" in res
        assert "distancia_final_au" in res
        assert "velocidad_final_kms" in res

    def test_main_positive_delta_v(self):
        res = main(["--no-plot", "--t-years", "0.1"])
        assert res["delta_v_kms"] > 0.0

    def test_main_writes_output_files(self, tmp_path):
        res = main(["--t-years", "0.05", "--out", str(tmp_path)])
        summary = tmp_path / "heliospheric_surfing_summary.txt"
        figure = tmp_path / "heliospheric_surfing.png"
        assert summary.exists()
        assert figure.exists()
        content = summary.read_text(encoding="utf-8")
        assert "Delta-V" in content

    def test_main_no_plot_skips_figure(self, tmp_path):
        main(["--t-years", "0.05", "--no-plot", "--out", str(tmp_path)])
        figure = tmp_path / "heliospheric_surfing.png"
        assert not figure.exists()

    def test_main_custom_mass_param(self):
        res_default = main(["--no-plot", "--t-years", "0.1"])
        res_heavy = main(["--no-plot", "--t-years", "0.1", "--mass", "500"])
        assert res_default["delta_v_kms"] > res_heavy["delta_v_kms"]

    def test_main_custom_r_bubble_param(self):
        res_small = main(["--no-plot", "--t-years", "0.1", "--r-bubble", "10000"])
        res_large = main(["--no-plot", "--t-years", "0.1", "--r-bubble", "100000"])
        assert res_large["delta_v_kms"] > res_small["delta_v_kms"]

    def test_main_default_args_runs(self, tmp_path):
        """Default parameters (1-year sim) must complete without error."""
        res = main(["--no-plot", "--out", str(tmp_path)])
        assert res["distancia_final_au"] > 1.0
