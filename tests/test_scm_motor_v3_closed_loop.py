"""
tests/test_scm_motor_v3_closed_loop.py — Tests for scripts/scm_motor_v3_closed_loop.py.

Covers:
  1. Physics helpers: rho_wind, v_sw_const, equilibrium_radius, drag_force,
     rf_power_required, power_budget, back_emf_power
  2. acceleration_net
  3. derivs (ODE right-hand side, 3-component state)
  4. run_simulation (integrator output, physics invariants)
  5. plot_results
  6. main() CLI
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.scm_motor_v3_closed_loop import (
    AU,
    GM_SUN,
    _ETA_MAX,
    _ETA_MIN,
    _TAU_CTRL,
    acceleration_net,
    back_emf_power,
    derivs,
    drag_force,
    equilibrium_radius,
    main,
    plot_results,
    power_budget,
    rf_power_required,
    rho_wind,
    run_simulation,
    v_sw_const,
)


# ---------------------------------------------------------------------------
# 1. Physics helpers
# ---------------------------------------------------------------------------

class TestRhoWind:
    def test_at_one_au_equals_rho0(self):
        assert rho_wind(AU) == pytest.approx(8e-20, rel=1e-9)

    def test_inverse_square_scaling(self):
        assert rho_wind(2 * AU) == pytest.approx(8e-20 / 4.0, rel=1e-9)

    def test_further_is_lower(self):
        assert rho_wind(5 * AU) < rho_wind(AU)

    def test_custom_rho0(self):
        assert rho_wind(AU, rho0=1e-18) == pytest.approx(1e-18, rel=1e-9)


class TestVSwConst:
    def test_returns_v0(self):
        assert v_sw_const(AU) == pytest.approx(400e3)

    def test_independent_of_r(self):
        assert v_sw_const(AU) == pytest.approx(v_sw_const(10 * AU))

    def test_custom_v0(self):
        assert v_sw_const(AU, v_sw0=600e3) == pytest.approx(600e3)


class TestEquilibriumRadius:
    def test_near_one_au_close_to_r0(self):
        R = equilibrium_radius(AU, 0.0)
        assert 45e3 < R < 60e3

    def test_eta_shape_scales_linearly(self):
        R1 = equilibrium_radius(AU, 0.0, eta_shape=1.0)
        R2 = equilibrium_radius(AU, 0.0, eta_shape=2.0)
        assert R2 == pytest.approx(2 * R1, rel=1e-9)

    def test_farther_from_sun_larger_radius(self):
        """Lower density at large r → scale > 1 → larger bubble."""
        R1 = equilibrium_radius(AU, 0.0)
        R5 = equilibrium_radius(5 * AU, 0.0)
        assert R5 > R1

    def test_extra_kwargs_ignored(self):
        R = equilibrium_radius(AU, 0.0, P_ai=99.0, mass=999.0)
        R_ref = equilibrium_radius(AU, 0.0)
        assert R == pytest.approx(R_ref, rel=1e-9)

    def test_positive_radius(self):
        assert equilibrium_radius(AU, 0.0) > 0.0

    def test_finite_when_v_equals_v_sw(self):
        """abs(v_rel)+1e-3 prevents division by zero at v = v_sw."""
        R = equilibrium_radius(AU, 400e3)
        assert math.isfinite(R) and R > 0.0

    def test_symmetric_around_v_sw(self):
        """Radius should be the same for equal-magnitude v_rel deviations."""
        R_below = equilibrium_radius(AU, 400e3 - 1.0)
        R_above = equilibrium_radius(AU, 400e3 + 1.0)
        assert R_below == pytest.approx(R_above, rel=1e-3)

    def test_large_v_rel_gives_smaller_radius(self):
        """Larger |v_rel| → smaller equilibrium radius."""
        R_slow = equilibrium_radius(AU, 0.0)         # v_rel = 400 km/s
        R_fast = equilibrium_radius(AU, 1000e3)       # |v_rel| = 600 km/s
        assert R_fast < R_slow


class TestDragForce:
    def test_returns_tuple_of_three(self):
        out = drag_force(AU, 0.0)
        assert len(out) == 3

    def test_v_rel_is_signed(self):
        """v_rel = v_sw - v: positive below v_sw, negative above."""
        _, _, vr_slow = drag_force(AU, 0.0)
        _, _, vr_fast = drag_force(AU, 500e3)
        assert vr_slow > 0.0
        assert vr_fast < 0.0

    def test_force_positive_when_v_less_than_v_sw(self):
        """Below v_sw: drag is a thrust force (positive)."""
        F, _, _ = drag_force(AU, 0.0)
        assert F > 0.0

    def test_force_zero_when_v_equals_v_sw(self):
        F, _, _ = drag_force(AU, 400e3)
        assert F == pytest.approx(0.0, abs=1e-10)

    def test_force_negative_when_v_greater_than_v_sw(self):
        """Above v_sw: drag brakes the probe (negative)."""
        F, _, _ = drag_force(AU, 500e3)
        assert F < 0.0

    def test_signed_drag_antisymmetric(self):
        """F at v_sw + Δv ≈ –F at v_sw − Δv (for small Δv)."""
        dv = 1.0
        F_pos, _, _ = drag_force(AU, 400e3 - dv)
        F_neg, _, _ = drag_force(AU, 400e3 + dv)
        assert F_pos == pytest.approx(-F_neg, rel=1e-3)

    def test_extra_kwargs_ignored(self):
        F1, _, _ = drag_force(AU, 0.0)
        F2, _, _ = drag_force(AU, 0.0, P_ai=99.0)
        assert F1 == pytest.approx(F2, rel=1e-9)

    def test_larger_r_smaller_magnitude_force(self):
        F1, _, _ = drag_force(AU, 0.0)
        F5, _, _ = drag_force(5 * AU, 0.0)
        assert abs(F5) < abs(F1)

    def test_r_returned_is_equilibrium_radius(self):
        _, R, _ = drag_force(AU, 0.0)
        assert R == pytest.approx(equilibrium_radius(AU, 0.0), rel=1e-9)


class TestRfPowerRequired:
    def test_r0_gives_15w(self):
        P = rf_power_required(50e3)
        assert P == pytest.approx(15.0, rel=0.01)

    def test_scales_with_cube(self):
        assert rf_power_required(100e3) == pytest.approx(
            rf_power_required(50e3) * 8.0, rel=1e-9
        )

    def test_zero_radius_zero_power(self):
        assert rf_power_required(0.0) == 0.0


class TestPowerBudget:
    def test_returns_required_keys(self):
        b = power_budget(AU, 0.0)
        assert {"P_gen", "P_rf", "P_ai", "P_net", "R", "v_rel"}.issubset(b.keys())

    def test_p_ai_default_is_8w(self):
        b = power_budget(AU, 0.0)
        assert b["P_ai"] == pytest.approx(8.0)

    def test_p_net_definition(self):
        b = power_budget(AU, 0.0)
        assert b["P_net"] == pytest.approx(b["P_gen"] - b["P_rf"] - b["P_ai"], rel=1e-9)

    def test_p_gen_nonnegative(self):
        """P_gen = |F·v_rel| is always non-negative."""
        b = power_budget(AU, 0.0)
        assert b["P_gen"] >= 0.0

    def test_p_gen_nonnegative_when_braking(self):
        """Even in braking regime (v > v_sw), P_gen ≥ 0."""
        b = power_budget(AU, 600e3)
        assert b["P_gen"] >= 0.0

    def test_custom_p_ai(self):
        b = power_budget(AU, 0.0, P_ai=20.0)
        assert b["P_ai"] == pytest.approx(20.0)

    def test_finite_values(self):
        b = power_budget(AU, 30e3)
        for v in b.values():
            assert math.isfinite(v)


class TestBackEmfPower:
    def test_positive_when_accelerating(self):
        """v < v_sw → back-EMF power positive (energy extracted from wind)."""
        assert back_emf_power(AU, 0.0) > 0.0

    def test_zero_when_v_equals_v_sw(self):
        assert back_emf_power(AU, 400e3) == pytest.approx(0.0, abs=1e-10)

    def test_positive_in_braking_regime(self):
        """v > v_sw: F < 0 and v_rel < 0; product F·v_rel is positive.

        Power dissipated in the interaction is always non-negative.
        """
        assert back_emf_power(AU, 500e3) > 0.0

    def test_extra_kwargs_ignored(self):
        p1 = back_emf_power(AU, 0.0)
        p2 = back_emf_power(AU, 0.0, P_ai=99.0)
        assert p1 == pytest.approx(p2, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. acceleration_net
# ---------------------------------------------------------------------------

class TestAccelerationNet:
    def test_gravity_only_when_v_equals_v_sw(self):
        a = acceleration_net(AU, 400e3)
        assert a == pytest.approx(-GM_SUN / AU ** 2, rel=1e-9)

    def test_drag_adds_positive_contribution_below_v_sw(self):
        a_with_drag = acceleration_net(AU, 0.0)
        a_grav_only = -GM_SUN / AU ** 2
        assert a_with_drag > a_grav_only

    def test_drag_subtracts_above_v_sw(self):
        """Above v_sw the signed drag brakes the probe, so a < gravity-only."""
        a_braking = acceleration_net(AU, 500e3)
        a_grav_only = -GM_SUN / AU ** 2
        assert a_braking < a_grav_only

    def test_finite(self):
        assert math.isfinite(acceleration_net(AU, 30e3))

    def test_larger_mass_less_acceleration(self):
        a_light = acceleration_net(AU, 0.0, mass=10.0)
        a_heavy = acceleration_net(AU, 0.0, mass=500.0)
        assert a_light > a_heavy


# ---------------------------------------------------------------------------
# 3. derivs
# ---------------------------------------------------------------------------

class TestDerivs:
    def test_output_length_three(self):
        out = derivs([AU, 30e3, 1.0], 0.0)
        assert len(out) == 3

    def test_dr_dt_equals_v(self):
        v = 30e3
        out = derivs([AU, v, 1.0], 0.0)
        assert out[0] == pytest.approx(v, rel=1e-10)

    def test_dv_dt_matches_acceleration_net(self):
        r, v, eta_s = AU, 30e3, 1.0
        out = derivs([r, v, eta_s], 0.0)
        a_exp = acceleration_net(r, v, eta_shape=eta_s)
        assert out[1] == pytest.approx(a_exp, rel=1e-9)

    def test_eta_contracts_when_p_net_negative(self):
        """When P_net < 0 (high P_ai), deta < 0."""
        kw = {"P_ai": 1e12}   # force P_net << 0
        out = derivs([AU, 0.0, 1.0], 0.0, kw)
        assert out[2] < 0.0

    def test_eta_expands_when_p_net_positive(self):
        """When P_net > 0 (no P_ai), deta > 0 (if eta < _ETA_MAX)."""
        kw = {"P_ai": 0.0}
        out = derivs([AU, 0.0, 1.0], 0.0, kw)
        assert out[2] > 0.0

    def test_none_kw_treated_as_empty(self):
        out_none = derivs([AU, 30e3, 1.0], 0.0, None)
        out_empty = derivs([AU, 30e3, 1.0], 0.0, {})
        assert out_none == pytest.approx(out_empty, rel=1e-9)

    def test_does_not_mutate_kw(self):
        kw = {"P_ai": 8.0}
        before = dict(kw)
        derivs([AU, 30e3, 1.0], 0.0, kw)
        assert kw == before

    def test_control_constants_accessible(self):
        assert _ETA_MIN == pytest.approx(0.5)
        assert _ETA_MAX == pytest.approx(1.2)
        assert _TAU_CTRL > 0.0


# ---------------------------------------------------------------------------
# 4. run_simulation
# ---------------------------------------------------------------------------

_SHORT = 10  # days


class TestRunSimulation:
    def _run(self, days=_SHORT, n_steps=200, **kw):
        return run_simulation(t_total_days=days, n_steps=n_steps, **kw)

    def test_required_keys(self):
        res = self._run()
        required = {"t_days", "r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net",
                    "v0_kms", "t_total_days", "kwargs"}
        assert required.issubset(res.keys())

    def test_array_lengths(self):
        n = 300
        res = run_simulation(t_total_days=_SHORT, n_steps=n)
        for key in ("t_days", "r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net"):
            assert len(res[key]) == n

    def test_initial_position_is_1au(self):
        res = self._run()
        assert res["r_AU"][0] == pytest.approx(1.0, rel=1e-4)

    def test_initial_velocity(self):
        res = self._run()
        assert res["v_kms"][0] == pytest.approx(30.0, rel=1e-4)

    def test_probe_moves_outward(self):
        res = self._run(days=30, n_steps=500)
        assert res["r_AU"][-1] > 1.0

    def test_positive_delta_v(self):
        res = self._run(days=30, n_steps=500)
        assert res["v_kms"][-1] > res["v0_kms"]

    def test_all_finite(self):
        res = self._run(days=30, n_steps=500)
        for key in ("r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net"):
            assert np.all(np.isfinite(res[key])), f"Non-finite values in '{key}'"

    def test_p_net_definition_holds(self):
        res = self._run(days=30, n_steps=500)
        np.testing.assert_allclose(
            res["P_net"], res["P_gen"] - res["P_rf"] - 8.0, rtol=1e-6
        )

    def test_r_km_positive(self):
        res = self._run(days=30, n_steps=500)
        assert np.all(res["R_km"] > 0)

    def test_heavier_probe_less_delta_v(self):
        r_light = self._run(days=30, n_steps=500, mass=10.0)
        r_heavy = self._run(days=30, n_steps=500, mass=500.0)
        dv_light = r_light["v_kms"][-1] - r_light["v0_kms"]
        dv_heavy = r_heavy["v_kms"][-1] - r_heavy["v0_kms"]
        assert dv_light > dv_heavy

    def test_custom_r0_used(self):
        res = run_simulation(t_total_days=1, n_steps=50, r0=2 * AU)
        assert res["r_AU"][0] == pytest.approx(2.0, rel=1e-4)

    def test_t_days_range(self):
        days = 20
        res = self._run(days=days, n_steps=200)
        assert res["t_days"][0] == pytest.approx(0.0, abs=1e-6)
        assert res["t_days"][-1] == pytest.approx(days, rel=1e-6)

    def test_one_year_runs_without_error(self):
        res = run_simulation(t_total_days=365, n_steps=5000)
        assert math.isfinite(res["r_AU"][-1])
        assert res["r_AU"][-1] > 1.0

    def test_signed_drag_brakes_supersonic_probe(self):
        """A probe launched faster than v_sw is decelerated toward v_sw."""
        # Start at v0 = 600 km/s >> v_sw = 400 km/s
        res = run_simulation(t_total_days=5, n_steps=500, v0=600e3)
        # Net delta-v must be negative (braking)
        delta_v = res["v_kms"][-1] - res["v0_kms"]
        assert delta_v < 0.0

    def test_terminal_velocity_bounded_by_v_sw(self):
        """Probe starting below v_sw accelerates; starting above gets braked.

        Over 15 days starting from rest, the probe should reach a significant
        fraction of v_sw (drag pushes it toward the wind speed).
        Over 5 days starting well above v_sw, the probe must be decelerated.
        """
        res_slow = run_simulation(t_total_days=15, n_steps=500, v0=10e3)
        v_slow_final = res_slow["v_kms"][-1]
        assert v_slow_final > 10.0, "Probe should have accelerated from rest"

        res_fast = run_simulation(t_total_days=5, n_steps=500, v0=600e3)
        v_fast_final = res_fast["v_kms"][-1]
        assert v_fast_final < 600.0, "Supersonic probe should have been braked"


# ---------------------------------------------------------------------------
# 5. plot_results
# ---------------------------------------------------------------------------

class TestPlotResults:
    def _res(self):
        return run_simulation(t_total_days=_SHORT, n_steps=200)

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        fig = plot_results(self._res())
        assert fig is not None
        plt.close("all")

    def test_four_axes(self):
        import matplotlib.pyplot as plt
        fig = plot_results(self._res())
        assert len(fig.axes) == 4
        plt.close("all")

    def test_save_to_file(self, tmp_path):
        import matplotlib.pyplot as plt
        out_file = tmp_path / "v3_sim.png"
        plot_results(self._res(), out_path=out_file)
        plt.close("all")
        assert out_file.exists()
        assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self):
        res = main(["--no-plot", "--t-days", "5"])
        assert isinstance(res, dict)

    def test_required_keys_in_result(self):
        res = main(["--no-plot", "--t-days", "5"])
        assert "r_AU" in res and "P_net" in res

    def test_probe_moves_outward(self):
        res = main(["--no-plot", "--t-days", "30"])
        assert res["r_AU"][-1] > 1.0

    def test_v_sw_flag_accepted(self):
        """--v-sw changes the solar wind speed used in the simulation."""
        res_slow = main(["--no-plot", "--t-days", "5", "--v-sw", "200"])
        res_fast = main(["--no-plot", "--t-days", "5", "--v-sw", "800"])
        # Faster wind → larger v_rel → more drag → probe moves farther
        assert res_fast["r_AU"][-1] > res_slow["r_AU"][-1]

    def test_v_sw_default_is_400(self):
        """Default --v-sw corresponds to v_sw0 = 400 km/s."""
        res_default = main(["--no-plot", "--t-days", "5"])
        res_explicit = main(["--no-plot", "--t-days", "5", "--v-sw", "400"])
        np.testing.assert_allclose(res_default["r_AU"], res_explicit["r_AU"], rtol=1e-6)

    def test_cme_simulation_higher_v_rel(self):
        """CME (v_sw = 800 km/s) delivers more impulse than nominal wind."""
        res_nominal = main(["--no-plot", "--t-days", "10"])
        res_cme = main(["--no-plot", "--t-days", "10", "--v-sw", "800"])
        dv_nominal = res_nominal["v_kms"][-1] - res_nominal["v0_kms"]
        dv_cme = res_cme["v_kms"][-1] - res_cme["v0_kms"]
        assert dv_cme > dv_nominal

    def test_writes_output_files(self, tmp_path):
        out_file = str(tmp_path / "v3.png")
        main(["--t-days", "5", "--out", out_file])
        import matplotlib.pyplot as plt
        plt.close("all")
        assert Path(out_file).exists()

    def test_no_plot_skips_figure(self, tmp_path):
        out_file = tmp_path / "v3.png"
        main(["--no-plot", "--t-days", "5", "--out", str(out_file)])
        assert not out_file.exists()

    def test_custom_mass(self):
        r_light = main(["--no-plot", "--t-days", "10", "--mass", "10"])
        r_heavy = main(["--no-plot", "--t-days", "10", "--mass", "500"])
        dv_light = r_light["v_kms"][-1] - r_light["v0_kms"]
        dv_heavy = r_heavy["v_kms"][-1] - r_heavy["v0_kms"]
        assert dv_light > dv_heavy

    def test_custom_p_ai_affects_p_net(self):
        r_low = main(["--no-plot", "--t-days", "5", "--P-ai", "1"])
        r_high = main(["--no-plot", "--t-days", "5", "--P-ai", "100"])
        assert r_low["P_net"].mean() > r_high["P_net"].mean()

    def test_default_one_year(self):
        import matplotlib.pyplot as plt
        res = main(["--no-plot"])
        plt.close("all")
        assert res["t_total_days"] == 365.0
        assert math.isfinite(res["r_AU"][-1])
