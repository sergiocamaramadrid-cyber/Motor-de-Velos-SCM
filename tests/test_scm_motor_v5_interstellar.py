"""
tests/test_scm_motor_v5_interstellar.py
Tests for scripts/scm_motor_v5_interstellar.py.

Covers:
  1.  _blend                   — smooth transition weight
  2.  rho_wind                 — density model (solar + ISM + transition)
  3.  v_wind_at                — wind velocity model (solar + ISM + transition)
  4.  eta_scm_adjust           — SCM shape factor
  5.  equilibrium_radius       — bubble radius
  6.  drag_force               — signed drag + density return
  7.  acceleration_net         — net acceleration
  8.  rf_power_required        — RF power
  9.  power_budget             — full energy budget
 10.  derivs                   — ODE right-hand side
 11.  run_simulation           — integrator output & physics invariants
 12.  plot_results             — figure generation
 13.  main() CLI               — command-line interface
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.scm_motor_v5_interstellar import (
    AU,
    DEFAULT_ETA_DRAG,
    DEFAULT_HELIOPAUSE_AU,
    DEFAULT_K_RF,
    DEFAULT_MASS,
    DEFAULT_P_AI,
    DEFAULT_R0,
    DEFAULT_RHO0,
    DEFAULT_RHO_ISM,
    DEFAULT_TRANSITION_AU,
    DEFAULT_V_ISM,
    DEFAULT_V_SW,
    GM_SUN,
    _blend,
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
    v_wind_at,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT = 5      # days — for fast simulations in tests
_N_SHORT = 100


def _run(**kw):
    return run_simulation(t_total_days=_SHORT, n_steps=_N_SHORT, **kw)


# ---------------------------------------------------------------------------
# 1. _blend
# ---------------------------------------------------------------------------

class TestBlend:
    def test_zero_inside_heliosphere(self):
        assert _blend(100 * AU, DEFAULT_HELIOPAUSE_AU * AU, 20 * AU) == 0.0

    def test_one_beyond_transition(self):
        r_hp = DEFAULT_HELIOPAUSE_AU * AU
        delta = DEFAULT_TRANSITION_AU * AU
        assert _blend(r_hp + delta + 1, r_hp, delta) == 1.0

    def test_half_at_mid_transition(self):
        r_hp = 120 * AU
        delta = 20 * AU
        assert _blend(r_hp + delta / 2, r_hp, delta) == pytest.approx(0.5, rel=1e-9)

    def test_monotone_increasing(self):
        r_hp = 120 * AU
        delta = 20 * AU
        alphas = [_blend(r_hp + k * delta / 4, r_hp, delta) for k in range(5)]
        assert alphas == sorted(alphas)

    def test_clip_below_zero_at_heliosphere(self):
        assert _blend(0.5 * AU, 120 * AU, 20 * AU) == 0.0

    def test_clip_above_one_far_ism(self):
        assert _blend(1000 * AU, 120 * AU, 20 * AU) == 1.0


# ---------------------------------------------------------------------------
# 2. rho_wind
# ---------------------------------------------------------------------------

class TestRhoWind:
    def test_at_1au_near_rho0(self):
        """Well inside heliosphere → density ≈ rho0."""
        rho = rho_wind(AU, rho0=DEFAULT_RHO0, heliopause_au=120.0)
        assert rho == pytest.approx(DEFAULT_RHO0, rel=1e-9)

    def test_inverse_square_solar(self):
        rho1 = rho_wind(AU, heliopause_au=120.0)
        rho2 = rho_wind(2 * AU, heliopause_au=120.0)
        assert rho1 / rho2 == pytest.approx(4.0, rel=1e-6)

    def test_ism_density_beyond_heliopause(self):
        r_far = 200 * AU  # well past heliopause (120 AU) + transition (20 AU)
        rho = rho_wind(r_far, heliopause_au=120.0, transition_au=20.0)
        assert rho == pytest.approx(DEFAULT_RHO_ISM, rel=1e-3)

    def test_density_decreases_across_transition(self):
        r_hp = 120 * AU
        rho_inner = rho_wind(r_hp - 5 * AU, heliopause_au=120.0, transition_au=20.0)
        rho_outer = rho_wind(r_hp + 25 * AU, heliopause_au=120.0, transition_au=20.0)
        # Solar rho at ~115 AU is very low; ISM is 1e-21
        # Both should be finite and positive
        assert rho_inner > 0 and rho_outer > 0

    def test_all_finite(self):
        for r_au in [1, 10, 50, 100, 120, 130, 140, 200]:
            assert math.isfinite(rho_wind(r_au * AU))


# ---------------------------------------------------------------------------
# 3. v_wind_at
# ---------------------------------------------------------------------------

class TestVWindAt:
    def test_solar_wind_inside_heliosphere(self):
        v = v_wind_at(AU, heliopause_au=120.0)
        assert v == pytest.approx(DEFAULT_V_SW, rel=1e-9)

    def test_ism_velocity_beyond_heliopause(self):
        r_far = 200 * AU
        v = v_wind_at(r_far, heliopause_au=120.0, transition_au=20.0)
        assert v == pytest.approx(DEFAULT_V_ISM, rel=1e-3)

    def test_intermediate_in_transition(self):
        r_hp = 120 * AU
        delta = 20 * AU
        v_mid = v_wind_at(r_hp + delta / 2, heliopause_au=120.0, transition_au=20.0)
        expected = 0.5 * DEFAULT_V_SW + 0.5 * DEFAULT_V_ISM
        assert v_mid == pytest.approx(expected, rel=1e-9)

    def test_decreases_with_distance(self):
        v1 = v_wind_at(50 * AU, heliopause_au=120.0, transition_au=20.0)
        v2 = v_wind_at(150 * AU, heliopause_au=120.0, transition_au=20.0)
        assert v1 > v2

    def test_custom_v_sw_v_ism(self):
        v = v_wind_at(AU, v_sw=500e3, v_ism=10e3, heliopause_au=120.0)
        assert v == pytest.approx(500e3, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. eta_scm_adjust
# ---------------------------------------------------------------------------

class TestEtaScmAdjust:
    def test_returns_geq_one(self):
        assert eta_scm_adjust(AU, 30e3, DEFAULT_RHO0) >= 1.0

    def test_zero_strength_returns_one(self):
        assert eta_scm_adjust(AU, 30e3, DEFAULT_RHO0, scm_strength=0.0) == pytest.approx(1.0)

    def test_scales_with_scm_strength(self):
        e_low = eta_scm_adjust(AU, 0.0, DEFAULT_RHO0, scm_strength=0.5)
        e_high = eta_scm_adjust(AU, 0.0, DEFAULT_RHO0, scm_strength=2.0)
        assert e_high > e_low

    def test_dense_environment_higher(self):
        """At same distance, higher rho → larger eta_shape."""
        e_low = eta_scm_adjust(AU, 0.0, DEFAULT_RHO0 * 0.01)
        e_high = eta_scm_adjust(AU, 0.0, DEFAULT_RHO0 * 10.0)
        assert e_high > e_low

    def test_finite(self):
        assert math.isfinite(eta_scm_adjust(AU, 30e3, DEFAULT_RHO0))

    def test_positive_at_ism_density(self):
        eta = eta_scm_adjust(200 * AU, 25e3, DEFAULT_RHO_ISM)
        assert eta >= 1.0


# ---------------------------------------------------------------------------
# 5. equilibrium_radius
# ---------------------------------------------------------------------------

class TestEquilibriumRadius:
    def test_near_one_au_positive(self):
        R = equilibrium_radius(AU, 0.0, DEFAULT_V_SW)
        assert R > 0.0

    def test_order_of_magnitude(self):
        R = equilibrium_radius(AU, 0.0, DEFAULT_V_SW)
        assert 10e3 < R < 200e3

    def test_eta_shape_scales_linearly(self):
        R1 = equilibrium_radius(AU, 0.0, DEFAULT_V_SW, eta_shape=1.0)
        R2 = equilibrium_radius(AU, 0.0, DEFAULT_V_SW, eta_shape=2.0)
        assert R2 == pytest.approx(2.0 * R1, rel=1e-9)

    def test_finite_at_v_equals_v_sw(self):
        R = equilibrium_radius(AU, DEFAULT_V_SW, DEFAULT_V_SW)
        assert math.isfinite(R) and R > 0.0

    def test_symmetric_around_v_sw(self):
        R_below = equilibrium_radius(AU, DEFAULT_V_SW - 1, DEFAULT_V_SW)
        R_above = equilibrium_radius(AU, DEFAULT_V_SW + 1, DEFAULT_V_SW)
        assert R_below == pytest.approx(R_above, rel=1e-3)


# ---------------------------------------------------------------------------
# 6. drag_force
# ---------------------------------------------------------------------------

class TestDragForce:
    def test_returns_three_values(self):
        out = drag_force(AU, 0.0, DEFAULT_V_SW)
        assert len(out) == 3

    def test_force_positive_below_v_sw(self):
        F, _, _ = drag_force(AU, 0.0, DEFAULT_V_SW)
        assert F > 0.0

    def test_force_zero_at_v_equals_v_sw(self):
        F, _, _ = drag_force(AU, DEFAULT_V_SW, DEFAULT_V_SW)
        assert F == pytest.approx(0.0, abs=1e-8)

    def test_force_negative_above_v_sw(self):
        F, _, _ = drag_force(AU, DEFAULT_V_SW + 1e3, DEFAULT_V_SW)
        assert F < 0.0

    def test_rho_returned_is_positive(self):
        _, _, rho = drag_force(AU, 0.0, DEFAULT_V_SW)
        assert rho > 0.0

    def test_extra_kwargs_ignored(self):
        F1, _, _ = drag_force(AU, 0.0, DEFAULT_V_SW)
        F2, _, _ = drag_force(AU, 0.0, DEFAULT_V_SW, P_ai=99.0, k_rf=1e-5)
        assert F1 == pytest.approx(F2, rel=1e-9)

    def test_far_from_heliopause_ism_density(self):
        """Beyond heliopause, density returned should be near ISM value."""
        r_far = 200 * AU
        v_sw_ism = DEFAULT_V_ISM
        _, _, rho = drag_force(r_far, 0.0, v_sw_ism,
                               heliopause_au=120.0, transition_au=20.0)
        assert rho == pytest.approx(DEFAULT_RHO_ISM, rel=1e-2)


# ---------------------------------------------------------------------------
# 7. acceleration_net
# ---------------------------------------------------------------------------

class TestAccelerationNet:
    def test_gravity_only_when_v_equals_v_sw(self):
        a = acceleration_net(AU, DEFAULT_V_SW, DEFAULT_V_SW)
        assert a == pytest.approx(-GM_SUN / AU ** 2, rel=1e-6)

    def test_thrust_below_v_sw(self):
        a = acceleration_net(AU, 0.0, DEFAULT_V_SW)
        assert a > -GM_SUN / AU ** 2

    def test_finite(self):
        assert math.isfinite(acceleration_net(AU, 30e3, DEFAULT_V_SW))

    def test_heavier_probe_lower_acceleration(self):
        a_light = acceleration_net(AU, 0.0, DEFAULT_V_SW, mass=10.0)
        a_heavy = acceleration_net(AU, 0.0, DEFAULT_V_SW, mass=500.0)
        assert a_light > a_heavy

    def test_braking_above_v_sw(self):
        a = acceleration_net(AU, DEFAULT_V_SW + 50e3, DEFAULT_V_SW)
        assert a < -GM_SUN / AU ** 2


# ---------------------------------------------------------------------------
# 8. rf_power_required
# ---------------------------------------------------------------------------

class TestRfPowerRequired:
    def test_r0_gives_15w(self):
        assert rf_power_required(50e3) == pytest.approx(15.0, rel=0.01)

    def test_cube_scaling(self):
        assert rf_power_required(100e3) == pytest.approx(
            rf_power_required(50e3) * 8.0, rel=1e-9
        )

    def test_zero_radius_zero_power(self):
        assert rf_power_required(0.0) == 0.0


# ---------------------------------------------------------------------------
# 9. power_budget
# ---------------------------------------------------------------------------

class TestPowerBudget:
    def test_required_keys(self):
        b = power_budget(AU, 0.0, DEFAULT_V_SW)
        assert {"P_gen", "P_rf", "P_ai", "P_net", "R", "v_rel", "rho"}.issubset(b.keys())

    def test_p_net_definition(self):
        b = power_budget(AU, 0.0, DEFAULT_V_SW)
        assert b["P_net"] == pytest.approx(b["P_gen"] - b["P_rf"] - b["P_ai"], rel=1e-9)

    def test_p_ai_default(self):
        b = power_budget(AU, 0.0, DEFAULT_V_SW)
        assert b["P_ai"] == pytest.approx(DEFAULT_P_AI)

    def test_finite(self):
        b = power_budget(AU, 30e3, DEFAULT_V_SW)
        for val in b.values():
            assert math.isfinite(val)

    def test_rho_is_positive(self):
        b = power_budget(AU, 0.0, DEFAULT_V_SW)
        assert b["rho"] > 0.0

    def test_v_rel_equals_v_sw_minus_v(self):
        v, v_sw = 30e3, DEFAULT_V_SW
        b = power_budget(AU, v, v_sw)
        assert b["v_rel"] == pytest.approx(v_sw - v, rel=1e-9)


# ---------------------------------------------------------------------------
# 10. derivs
# ---------------------------------------------------------------------------

class TestDerivs:
    def test_output_length_two(self):
        out = derivs(0.0, [AU, 30e3])
        assert len(out) == 2

    def test_dr_dt_equals_v(self):
        v = 30e3
        out = derivs(0.0, [AU, v])
        assert out[0] == pytest.approx(v, rel=1e-10)

    def test_dv_dt_finite(self):
        out = derivs(0.0, [AU, 30e3])
        assert math.isfinite(out[1])

    def test_thrust_below_v_sw(self):
        """At v=0, drag is fully accelerating → dv/dt > gravity alone."""
        a_drag_only = derivs(0.0, [AU, 0.0])[1]
        a_grav = -GM_SUN / AU ** 2
        assert a_drag_only > a_grav

    def test_scm_strength_affects_acceleration(self):
        a_low = derivs(0.0, [AU, 0.0], scm_strength=0.0)[1]
        a_high = derivs(0.0, [AU, 0.0], scm_strength=3.0)[1]
        assert a_high > a_low

    def test_ism_regime_finite(self):
        """Derivs should not blow up beyond heliopause."""
        r_ism = 150 * AU
        v_ism = DEFAULT_V_ISM
        out = derivs(0.0, [r_ism, v_ism], heliopause_au=120.0, transition_au=20.0)
        assert all(math.isfinite(x) for x in out)


# ---------------------------------------------------------------------------
# 11. run_simulation
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def test_required_keys(self):
        res = _run()
        required = {
            "t_days", "r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net",
            "v_rel_kms", "v_wind_kms", "rho", "v0_kms", "t_total_days",
        }
        assert required.issubset(res.keys())

    def test_array_lengths(self):
        n = 150
        res = run_simulation(t_total_days=_SHORT, n_steps=n)
        for key in ("t_days", "r_AU", "v_kms", "R_km", "P_net", "rho"):
            assert len(res[key]) == n

    def test_initial_position_1au(self):
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
        for key in ("r_AU", "v_kms", "R_km", "P_gen", "P_rf", "P_net", "rho"):
            assert np.all(np.isfinite(res[key])), f"Non-finite in '{key}'"

    def test_r_km_positive(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        assert np.all(res["R_km"] > 0)

    def test_rho_positive(self):
        res = run_simulation(t_total_days=30, n_steps=500)
        assert np.all(res["rho"] > 0)

    def test_heavier_probe_less_delta_v(self):
        r_light = run_simulation(t_total_days=20, n_steps=300, mass=10.0)
        r_heavy = run_simulation(t_total_days=20, n_steps=300, mass=500.0)
        assert (r_light["v_kms"][-1] - r_light["v0_kms"]) > (r_heavy["v_kms"][-1] - r_heavy["v0_kms"])

    def test_supersonic_probe_gets_braked(self):
        res = run_simulation(t_total_days=5, n_steps=200, v0=600e3)
        assert res["v_kms"][-1] < res["v0_kms"]

    def test_density_at_t0_near_rho0(self):
        res = run_simulation(t_total_days=5, n_steps=200)
        assert res["rho"][0] == pytest.approx(DEFAULT_RHO0, rel=1e-4)

    def test_v_wind_solar_at_t0(self):
        res = run_simulation(t_total_days=5, n_steps=200)
        assert res["v_wind_kms"][0] == pytest.approx(DEFAULT_V_SW / 1000.0, rel=1e-6)

    def test_t_days_range(self):
        days = 15
        res = run_simulation(t_total_days=days, n_steps=200)
        assert res["t_days"][0] == pytest.approx(0.0, abs=1e-6)
        assert res["t_days"][-1] == pytest.approx(days, rel=1e-5)

    def test_scm_strength_larger_bubble(self):
        r_low = run_simulation(t_total_days=10, n_steps=200, scm_strength=0.0)
        r_high = run_simulation(t_total_days=10, n_steps=200, scm_strength=2.0)
        assert r_high["R_km"].mean() >= r_low["R_km"].mean()

    def test_custom_heliopause(self):
        """Heliopause at 50 AU: density at 60 AU should be near ISM."""
        res = run_simulation(
            t_total_days=5, n_steps=200,
            r0=60 * AU, v0=400e3,
            heliopause_au=50.0, transition_au=5.0,
        )
        # Well past heliopause: rho should be close to DEFAULT_RHO_ISM
        assert res["rho"][-1] == pytest.approx(DEFAULT_RHO_ISM, rel=0.05)

    def test_one_year_no_error(self):
        res = run_simulation(t_total_days=365, n_steps=2000)
        assert math.isfinite(res["r_AU"][-1])
        assert res["r_AU"][-1] > 1.0


# ---------------------------------------------------------------------------
# 12. plot_results
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
        out = tmp_path / "v5_sim.png"
        plot_results(self._res(), out_path=out)
        plt.close("all")
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# 13. main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self):
        res = main(["--no-plot", "--t-days", "5", "--n-steps", "100"])
        assert isinstance(res, dict)

    def test_required_keys_in_result(self):
        res = main(["--no-plot", "--t-days", "5", "--n-steps", "100"])
        assert "r_AU" in res and "P_net" in res and "rho" in res

    def test_probe_moves_outward(self):
        res = main(["--no-plot", "--t-days", "30", "--n-steps", "500"])
        assert res["r_AU"][-1] > 1.0

    def test_writes_output_file(self, tmp_path):
        import matplotlib.pyplot as plt
        out = str(tmp_path / "v5.png")
        main(["--t-days", "5", "--n-steps", "100", "--out", out])
        plt.close("all")
        assert Path(out).exists()

    def test_no_plot_skips_file(self, tmp_path):
        out = tmp_path / "v5.png"
        main(["--no-plot", "--t-days", "5", "--n-steps", "100", "--out", str(out)])
        assert not out.exists()

    def test_custom_mass(self):
        r_light = main(["--no-plot", "--t-days", "10", "--n-steps", "200", "--mass", "10"])
        r_heavy = main(["--no-plot", "--t-days", "10", "--n-steps", "200", "--mass", "500"])
        dv_l = r_light["v_kms"][-1] - r_light["v0_kms"]
        dv_h = r_heavy["v_kms"][-1] - r_heavy["v0_kms"]
        assert dv_l > dv_h

    def test_custom_p_ai_affects_p_net(self):
        r_low = main(["--no-plot", "--t-days", "5", "--n-steps", "100", "--P-ai", "1"])
        r_high = main(["--no-plot", "--t-days", "5", "--n-steps", "100", "--P-ai", "100"])
        assert r_low["P_net"].mean() > r_high["P_net"].mean()

    def test_scm_strength_flag(self):
        r_low = main(["--no-plot", "--t-days", "10", "--n-steps", "200", "--scm-strength", "0.0"])
        r_high = main(["--no-plot", "--t-days", "10", "--n-steps", "200", "--scm-strength", "2.0"])
        assert r_high["R_km"].mean() >= r_low["R_km"].mean()

    def test_heliopause_au_flag(self):
        """Probe starting well inside a close heliopause crosses into ISM quickly."""
        # r0=1.2 AU, heliopause=1.3 AU, transition=0.1 AU, v0=400 km/s
        # In 5 days the probe travels ~1.15 AU → reaches ~2.35 AU, past transition.
        res = run_simulation(
            t_total_days=5, n_steps=200,
            r0=1.2 * AU, v0=400e3,
            heliopause_au=1.3, transition_au=0.1,
        )
        # Beyond transition zone (1.4 AU) v_wind should drop toward v_ism
        assert np.any(res["v_wind_kms"] < DEFAULT_V_SW / 1000.0)

    def test_transition_au_flag_accepted(self):
        res = main(["--no-plot", "--t-days", "5", "--n-steps", "100",
                    "--transition-au", "10.0"])
        assert isinstance(res, dict)
