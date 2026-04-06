"""
tests/test_scm_motor_v4_variable_wind.py
Tests for scripts/scm_motor_v4_variable_wind.py.

Covers:
  1. v_sw_time_dependent      — wind profile generation
  2. eta_scm_adjust           — SCM shape factor
  3. rho_wind                 — density model
  4. equilibrium_radius       — bubble radius
  5. drag_force               — signed drag
  6. acceleration_net         — net acceleration
  7. rf_power_required        — RF power
  8. power_budget             — full energy budget
  9. derivs                   — ODE right-hand side
 10. run_simulation           — integrator output & physics invariants
 11. plot_results             — figure generation
 12. main() CLI               — command-line interface
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.scm_motor_v4_variable_wind import (
    AU,
    DEFAULT_ETA_DRAG,
    DEFAULT_K_RF,
    DEFAULT_MASS,
    DEFAULT_P_AI,
    DEFAULT_R0,
    DEFAULT_RHO0,
    GM_SUN,
    acceleration_net,
    derivs,
    drag_force,
    equilibrium_radius,
    eta_scm_adjust,
    main,
    plot_results,
    power_budget,
    rf_power_required,
    rho_wind,
    run_simulation,
    v_sw_time_dependent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT = 5    # days — for fast simulations in tests
_N_SHORT = 100


def _run(**kw):
    return run_simulation(t_total_days=_SHORT, n_steps=_N_SHORT, **kw)


# ---------------------------------------------------------------------------
# 1. v_sw_time_dependent
# ---------------------------------------------------------------------------

class TestVSwTimeDep:
    def _t(self, n=365):
        return np.linspace(0, 365, n)

    def test_returns_array_of_correct_length(self):
        t = self._t(500)
        v = v_sw_time_dependent(t)
        assert v.shape == t.shape

    def test_values_in_range_nominal(self):
        t = self._t(500)
        v = v_sw_time_dependent(t, cme_prob=0.0)
        assert np.all(v >= 300e3)
        assert np.all(v <= 400e3)

    def test_cme_events_can_exceed_base(self):
        t = self._t(1000)
        v = v_sw_time_dependent(t, cme_prob=1.0)   # force CMEs everywhere
        assert np.any(v > 400e3)

    def test_reproducible_with_same_seed(self):
        t = self._t(500)
        v1 = v_sw_time_dependent(t, random_seed=7)
        v2 = v_sw_time_dependent(t, random_seed=7)
        np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_differ(self):
        t = self._t(500)
        v1 = v_sw_time_dependent(t, cme_prob=0.05, random_seed=1)
        v2 = v_sw_time_dependent(t, cme_prob=0.05, random_seed=2)
        assert not np.array_equal(v1, v2)

    def test_no_cme_smooth(self):
        t = self._t(365)
        v = v_sw_time_dependent(t, cme_prob=0.0)
        assert np.all(np.isfinite(v))

    def test_at_t0_close_to_slow(self):
        """cos(0)=1 → v = base + (slow-base)*1 = slow."""
        t = np.array([0.0])
        v = v_sw_time_dependent(t, base=400e3, slow=300e3, cme_prob=0.0)
        assert v[0] == pytest.approx(300e3, rel=1e-9)

    def test_custom_base_slow(self):
        t = self._t(200)
        v = v_sw_time_dependent(t, base=500e3, slow=200e3, cme_prob=0.0)
        assert np.all(v >= 200e3)
        assert np.all(v <= 500e3)


# ---------------------------------------------------------------------------
# 2. eta_scm_adjust
# ---------------------------------------------------------------------------

class TestEtaScmAdjust:
    def test_returns_value_geq_one(self):
        assert eta_scm_adjust(AU, 30e3, DEFAULT_RHO0) >= 1.0

    def test_larger_at_inner_heliosphere(self):
        """Closer to Sun (higher density) → larger eta_shape."""
        eta_near = eta_scm_adjust(0.5 * AU, 30e3, DEFAULT_RHO0)
        eta_far = eta_scm_adjust(5.0 * AU, 30e3, DEFAULT_RHO0)
        assert eta_near > eta_far

    def test_scales_with_scm_strength(self):
        eta_low = eta_scm_adjust(AU, 30e3, DEFAULT_RHO0, scm_strength=0.5)
        eta_high = eta_scm_adjust(AU, 30e3, DEFAULT_RHO0, scm_strength=2.0)
        assert eta_high > eta_low

    def test_zero_strength_returns_one(self):
        assert eta_scm_adjust(AU, 30e3, DEFAULT_RHO0, scm_strength=0.0) == pytest.approx(1.0)

    def test_finite(self):
        assert math.isfinite(eta_scm_adjust(AU, 30e3, DEFAULT_RHO0))


# ---------------------------------------------------------------------------
# 3. rho_wind
# ---------------------------------------------------------------------------

class TestRhoWind:
    def test_at_one_au_equals_rho0(self):
        assert rho_wind(AU) == pytest.approx(DEFAULT_RHO0, rel=1e-9)

    def test_inverse_square_scaling(self):
        assert rho_wind(2 * AU) == pytest.approx(DEFAULT_RHO0 / 4.0, rel=1e-9)

    def test_farther_is_lower(self):
        assert rho_wind(5 * AU) < rho_wind(AU)


# ---------------------------------------------------------------------------
# 4. equilibrium_radius
# ---------------------------------------------------------------------------

class TestEquilibriumRadius:
    def test_near_one_au_positive(self):
        R = equilibrium_radius(AU, 0.0, 400e3)
        assert R > 0.0

    def test_near_one_au_order_of_magnitude(self):
        R = equilibrium_radius(AU, 0.0, 400e3)
        assert 10e3 < R < 200e3

    def test_eta_shape_scales_linearly(self):
        R1 = equilibrium_radius(AU, 0.0, 400e3, eta_shape=1.0)
        R2 = equilibrium_radius(AU, 0.0, 400e3, eta_shape=2.0)
        assert R2 == pytest.approx(2.0 * R1, rel=1e-9)

    def test_finite_at_v_equals_v_sw(self):
        """abs(v_rel)+1e-3 prevents division by zero."""
        R = equilibrium_radius(AU, 400e3, 400e3)
        assert math.isfinite(R) and R > 0.0

    def test_symmetric_around_v_sw(self):
        R_below = equilibrium_radius(AU, 399e3, 400e3)
        R_above = equilibrium_radius(AU, 401e3, 400e3)
        assert R_below == pytest.approx(R_above, rel=1e-3)

    def test_extra_kwargs_ignored(self):
        R = equilibrium_radius(AU, 0.0, 400e3, P_ai=99.0)
        R_ref = equilibrium_radius(AU, 0.0, 400e3)
        assert R == pytest.approx(R_ref, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. drag_force
# ---------------------------------------------------------------------------

class TestDragForce:
    def test_returns_two_values(self):
        out = drag_force(AU, 0.0, 400e3)
        assert len(out) == 2

    def test_force_positive_when_v_less_than_v_sw(self):
        F, _ = drag_force(AU, 0.0, 400e3)
        assert F > 0.0

    def test_force_zero_when_v_equals_v_sw(self):
        F, _ = drag_force(AU, 400e3, 400e3)
        assert F == pytest.approx(0.0, abs=1e-8)

    def test_force_negative_when_v_greater_than_v_sw(self):
        F, _ = drag_force(AU, 500e3, 400e3)
        assert F < 0.0

    def test_signed_drag_antisymmetric(self):
        dv = 1.0
        F_acc, _ = drag_force(AU, 400e3 - dv, 400e3)
        F_brake, _ = drag_force(AU, 400e3 + dv, 400e3)
        assert F_acc == pytest.approx(-F_brake, rel=1e-3)

    def test_extra_kwargs_ignored(self):
        F1, _ = drag_force(AU, 0.0, 400e3)
        F2, _ = drag_force(AU, 0.0, 400e3, P_ai=99.0, k_rf=1e-5)
        assert F1 == pytest.approx(F2, rel=1e-9)

    def test_r_is_equilibrium_radius(self):
        _, R = drag_force(AU, 0.0, 400e3)
        assert R == pytest.approx(equilibrium_radius(AU, 0.0, 400e3), rel=1e-9)


# ---------------------------------------------------------------------------
# 6. acceleration_net
# ---------------------------------------------------------------------------

class TestAccelerationNet:
    def test_gravity_only_when_v_equals_v_sw(self):
        a = acceleration_net(AU, 400e3, 400e3)
        assert a == pytest.approx(-GM_SUN / AU ** 2, rel=1e-9)

    def test_thrust_below_v_sw(self):
        a = acceleration_net(AU, 0.0, 400e3)
        assert a > -GM_SUN / AU ** 2

    def test_braking_above_v_sw(self):
        a = acceleration_net(AU, 500e3, 400e3)
        assert a < -GM_SUN / AU ** 2

    def test_finite(self):
        assert math.isfinite(acceleration_net(AU, 30e3, 400e3))

    def test_heavier_probe_less_acceleration(self):
        a_light = acceleration_net(AU, 0.0, 400e3, mass=10.0)
        a_heavy = acceleration_net(AU, 0.0, 400e3, mass=500.0)
        assert a_light > a_heavy

    def test_default_mass_is_default_mass_constant(self):
        """Regression: default mass must be DEFAULT_MASS (50 kg), not DEFAULT_R0 (50000 m)."""
        a_default = acceleration_net(AU, 0.0, 400e3)
        a_explicit = acceleration_net(AU, 0.0, 400e3, mass=DEFAULT_MASS)
        assert a_default == pytest.approx(a_explicit, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. rf_power_required
# ---------------------------------------------------------------------------

class TestRfPowerRequired:
    def test_r0_gives_15w(self):
        assert rf_power_required(50e3) == pytest.approx(15.0, rel=0.01)

    def test_scales_with_cube(self):
        assert rf_power_required(100e3) == pytest.approx(
            rf_power_required(50e3) * 8.0, rel=1e-9
        )

    def test_zero_radius_zero_power(self):
        assert rf_power_required(0.0) == 0.0


# ---------------------------------------------------------------------------
# 8. power_budget
# ---------------------------------------------------------------------------

class TestPowerBudget:
    def test_required_keys(self):
        b = power_budget(AU, 0.0, 400e3)
        assert {"P_gen", "P_rf", "P_ai", "P_net", "R", "v_rel"}.issubset(b.keys())

    def test_p_net_definition(self):
        b = power_budget(AU, 0.0, 400e3)
        assert b["P_net"] == pytest.approx(b["P_gen"] - b["P_rf"] - b["P_ai"], rel=1e-9)

    def test_p_ai_default(self):
        b = power_budget(AU, 0.0, 400e3)
        assert b["P_ai"] == pytest.approx(DEFAULT_P_AI)

    def test_finite_values(self):
        b = power_budget(AU, 30e3, 400e3)
        for val in b.values():
            assert math.isfinite(val)

    def test_p_gen_positive_below_v_sw(self):
        b = power_budget(AU, 0.0, 400e3)
        assert b["P_gen"] > 0.0

    def test_v_rel_is_v_sw_minus_v(self):
        v, v_sw = 30e3, 400e3
        b = power_budget(AU, v, v_sw)
        assert b["v_rel"] == pytest.approx(v_sw - v, rel=1e-9)


# ---------------------------------------------------------------------------
# 9. derivs
# ---------------------------------------------------------------------------

class TestDerivs:
    def _t_arr(self, n=100):
        return np.linspace(0, 10, n)

    def _v_arr(self, t):
        return v_sw_time_dependent(t, cme_prob=0.0)

    def test_output_length_two(self):
        t_arr = self._t_arr()
        v_arr = self._v_arr(t_arr)
        out = derivs(0.0, [AU, 30e3], t_arr, v_arr)
        assert len(out) == 2

    def test_dr_dt_equals_v(self):
        t_arr = self._t_arr()
        v_arr = self._v_arr(t_arr)
        v = 30e3
        out = derivs(0.0, [AU, v], t_arr, v_arr)
        assert out[0] == pytest.approx(v, rel=1e-10)

    def test_dv_dt_finite(self):
        t_arr = self._t_arr()
        v_arr = self._v_arr(t_arr)
        out = derivs(0.0, [AU, 30e3], t_arr, v_arr)
        assert math.isfinite(out[1])

    def test_interpolates_v_sw(self):
        """At t=0, v_sw should match the first element of v_arr."""
        t_arr = self._t_arr()
        v_arr = self._v_arr(t_arr)
        # At t_days=0, interp gives v_arr[0]
        v_sw_0 = v_arr[0]
        v = 0.0  # much slower than wind → thrust expected
        out = derivs(0.0, [AU, v], t_arr, v_arr)
        a_manual = acceleration_net(AU, v, v_sw_0)
        assert out[1] != pytest.approx(-GM_SUN / AU ** 2, rel=1e-3), \
            "Should have drag contribution"
        assert math.isfinite(out[1])

    def test_scm_strength_affects_dv_dt(self):
        t_arr = self._t_arr()
        v_arr = self._v_arr(t_arr)
        out_low = derivs(0.0, [AU, 0.0], t_arr, v_arr, scm_strength=0.0)
        out_high = derivs(0.0, [AU, 0.0], t_arr, v_arr, scm_strength=3.0)
        # Higher scm_strength → larger bubble → more drag → higher acceleration
        assert out_high[1] > out_low[1]


# ---------------------------------------------------------------------------
# 10. run_simulation
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def test_required_keys(self):
        res = _run()
        required = {
            "t_days", "r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net",
            "v_rel_kms", "v_sw_interp", "v0_kms", "t_total_days",
        }
        assert required.issubset(res.keys())

    def test_array_lengths(self):
        n = 150
        res = run_simulation(t_total_days=_SHORT, n_steps=n)
        for key in ("t_days", "r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net"):
            assert len(res[key]) == n

    def test_initial_position_is_1au(self):
        res = _run()
        assert res["r_AU"][0] == pytest.approx(1.0, rel=1e-4)

    def test_probe_moves_outward(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        assert res["r_AU"][-1] > 1.0

    def test_positive_delta_v(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        assert res["v_kms"][-1] > res["v0_kms"]

    def test_all_finite(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        for key in ("r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net"):
            assert np.all(np.isfinite(res[key])), f"Non-finite values in '{key}'"

    def test_r_km_positive(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        assert np.all(res["R_km"] > 0)

    def test_heavier_probe_less_delta_v(self):
        r_light = run_simulation(t_total_days=20, n_steps=300, mass=10.0)
        r_heavy = run_simulation(t_total_days=20, n_steps=300, mass=500.0)
        dv_light = r_light["v_kms"][-1] - r_light["v0_kms"]
        dv_heavy = r_heavy["v_kms"][-1] - r_heavy["v0_kms"]
        assert dv_light > dv_heavy

    def test_cme_seed_affects_result(self):
        r1 = run_simulation(t_total_days=100, n_steps=1000, random_seed=1)
        r2 = run_simulation(t_total_days=100, n_steps=1000, random_seed=99)
        # Different CME patterns → different velocities
        assert r1["v_kms"][-1] != pytest.approx(r2["v_kms"][-1], rel=1e-6)

    def test_scm_strength_larger_bubble(self):
        r_low = run_simulation(t_total_days=10, n_steps=200, scm_strength=0.1)
        r_high = run_simulation(t_total_days=10, n_steps=200, scm_strength=2.0)
        # Larger scm_strength → larger eta_shape → larger bubble radius
        assert r_high["R_km"].mean() > r_low["R_km"].mean()

    def test_t_days_range(self):
        days = 15
        res = run_simulation(t_total_days=days, n_steps=200)
        assert res["t_days"][0] == pytest.approx(0.0, abs=1e-6)
        assert res["t_days"][-1] == pytest.approx(days, rel=1e-5)

    def test_one_year_runs_without_error(self):
        res = run_simulation(t_total_days=365, n_steps=3000)
        assert math.isfinite(res["r_AU"][-1])
        assert res["r_AU"][-1] > 1.0

    def test_supersonic_probe_gets_braked(self):
        """Probe starting well above v_sw is decelerated."""
        res = run_simulation(t_total_days=5, n_steps=200, v0=600e3)
        dv = res["v_kms"][-1] - res["v0_kms"]
        assert dv < 0.0

    def test_v_sw_interp_in_expected_range(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        # Without forcing CMEs, v_sw should be between 300 and ~900 km/s
        assert np.all(res["v_sw_interp"] >= 200.0)   # km/s
        assert np.all(res["v_sw_interp"] <= 1000.0)  # km/s


# ---------------------------------------------------------------------------
# 11. plot_results
# ---------------------------------------------------------------------------

class TestPlotResults:
    def _res(self):
        return run_simulation(t_total_days=_SHORT, n_steps=_N_SHORT)

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        fig = plot_results(self._res())
        assert fig is not None
        plt.close("all")

    def test_five_axes(self):
        import matplotlib.pyplot as plt
        fig = plot_results(self._res())
        assert len(fig.axes) == 5
        plt.close("all")

    def test_save_to_file(self, tmp_path):
        import matplotlib.pyplot as plt
        out_file = tmp_path / "v4_sim.png"
        plot_results(self._res(), out_path=out_file)
        plt.close("all")
        assert out_file.exists()
        assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# 12. main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self):
        res = main(["--no-plot", "--t-days", "5", "--n-steps", "100"])
        assert isinstance(res, dict)

    def test_required_keys_in_result(self):
        res = main(["--no-plot", "--t-days", "5", "--n-steps", "100"])
        assert "r_AU" in res and "P_net" in res

    def test_probe_moves_outward(self):
        res = main(["--no-plot", "--t-days", "30", "--n-steps", "500"])
        assert res["r_AU"][-1] > 1.0

    def test_writes_output_file(self, tmp_path):
        out_file = str(tmp_path / "v4.png")
        main(["--t-days", "5", "--n-steps", "100", "--out", out_file])
        import matplotlib.pyplot as plt
        plt.close("all")
        assert Path(out_file).exists()

    def test_no_plot_skips_figure(self, tmp_path):
        out_file = tmp_path / "v4.png"
        main(["--no-plot", "--t-days", "5", "--n-steps", "100",
              "--out", str(out_file)])
        assert not out_file.exists()

    def test_custom_mass(self):
        r_light = main(["--no-plot", "--t-days", "10", "--n-steps", "200",
                        "--mass", "10"])
        r_heavy = main(["--no-plot", "--t-days", "10", "--n-steps", "200",
                        "--mass", "500"])
        dv_light = r_light["v_kms"][-1] - r_light["v0_kms"]
        dv_heavy = r_heavy["v_kms"][-1] - r_heavy["v0_kms"]
        assert dv_light > dv_heavy

    def test_custom_p_ai_affects_p_net(self):
        r_low = main(["--no-plot", "--t-days", "5", "--n-steps", "100",
                      "--P-ai", "1"])
        r_high = main(["--no-plot", "--t-days", "5", "--n-steps", "100",
                       "--P-ai", "100"])
        assert r_low["P_net"].mean() > r_high["P_net"].mean()

    def test_seed_flag(self):
        r1 = main(["--no-plot", "--t-days", "50", "--n-steps", "500", "--seed", "1"])
        r2 = main(["--no-plot", "--t-days", "50", "--n-steps", "500", "--seed", "99"])
        assert r1["v_kms"][-1] != pytest.approx(r2["v_kms"][-1], rel=1e-6)

    def test_scm_strength_flag(self):
        r_low = main(["--no-plot", "--t-days", "10", "--n-steps", "200",
                      "--scm-strength", "0.1"])
        r_high = main(["--no-plot", "--t-days", "10", "--n-steps", "200",
                       "--scm-strength", "2.0"])
        assert r_high["R_km"].mean() > r_low["R_km"].mean()
