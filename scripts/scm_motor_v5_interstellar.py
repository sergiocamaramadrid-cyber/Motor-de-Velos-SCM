#!/usr/bin/env python3
"""
scripts/scm_motor_v5_interstellar.py
V5: transición al medio interestelar + SCM predictivo

Extiende V4 (scm_motor_v4_variable_wind.py) con:
  · Dos regímenes de densidad: viento solar (∝ r⁻²) y medio interestelar
    (constante ≈ 10⁻²¹ kg/m³).
  · Transición suave en la heliopausa (por defecto 120 UA): densidad y
    velocidad del viento se interpolan linealmente durante 20 UA.
  · eta_scm_adjust extendida: el factor SCM crece con la densidad relativa
    al valor de referencia en 1 UA, de modo que el ajuste es físicamente
    significativo tanto en el medio solar como en el interestelar.
  · Panel extra en la figura: densidad ρ vs tiempo.
  · CLI: --heliopause-au, --transition-au, --t-days 7300 (20 años).

Bugs corregidos respecto al script del problem statement:
  1. Usaba ``odeint`` (interfaz antigua, mala para problemas rígidos).
     Corregido a ``solve_ivp`` DOP853, igual que V3 y V4.
  2. ``derivs(state, t, **kwargs)`` con ``odeint(args=(kwargs,))`` pasa el
     dict como argumento posicional → TypeError. Corregido con closure.
  3. ``v_wind(t_days, r, ...)`` ignoraba silenciosamente t_days. Simplificado
     a ``v_wind_at(r, ...)`` que devuelve la velocidad escalar según la zona.
  4. Transición heliopausa era un if/else duro; reemplazado por una función
     ``blend()`` que interpola suavemente densidad y velocidad en 20 UA.
  5. ``equilibrium_radius`` dividía por ``v_rel**2`` sin protección frente a
     v_rel = 0. Corregido con ``abs(v_rel) + 1e-3``.
  6. ``drag_force`` no aceptaba kwargs extra → TypeError al propagarlos.
     Corregido añadiendo ``**_``.
  7. ``main()`` no aceptaba ``argv`` → no testeable. Corregido a
     ``main(argv=None)`` con ``parse_args(argv)``.
  8. ``plot_results`` llamaba ``plt.show()`` → bloquea en entornos no
     interactivos. Corregido a devolver la figura sin mostrarla.
  9. El ajuste SCM (eta_shape) no se propagaba al ODE; sólo al bucle de
     métricas post-integración. Corregido: ``derivs`` llama a
     ``eta_scm_adjust`` en cada paso.
 10. ``power_budget`` usaba ``k_rf`` y ``P_ai`` hardcodeados. Parametrizado.
 11. kwargs de ``run_simulation`` mezclaban ``P_ai/k_rf`` con los de arrastre.
     Corregido separando kwargs de mecánica y de energía.

Usage
-----
20 años por defecto::

    python scripts/scm_motor_v5_interstellar.py

5 años para pruebas rápidas::

    python scripts/scm_motor_v5_interstellar.py --t-days 1825 --n-steps 5000

Con salida PNG::

    python scripts/scm_motor_v5_interstellar.py --out results/v5/sim.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------

AU = 1.495978707e11       # Unidad Astronómica (m)
G = 6.67430e-11           # Constante gravitacional (m³ kg⁻¹ s⁻²)
M_SUN = 1.9885e30         # Masa del Sol (kg)
GM_SUN = G * M_SUN        # Parámetro gravitacional estándar (m³/s²)

# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------

DEFAULT_MASS = 50.0            # kg
DEFAULT_R0 = 50e3              # m (radio de referencia de la burbuja)
DEFAULT_ETA_DRAG = 0.5
DEFAULT_RHO0 = 8e-20           # kg/m³  a 1 UA (viento solar)
DEFAULT_RHO_ISM = 1e-21        # kg/m³  medio interestelar
DEFAULT_V_SW = 400e3           # m/s   viento solar de referencia
DEFAULT_V_ISM = 25e3           # m/s   velocidad relativa interestelar
DEFAULT_P_AI = 8.0             # W
DEFAULT_K_RF = 1.2e-13         # W/m³  (15 W @ R = 50 km)
DEFAULT_HELIOPAUSE_AU = 120.0  # UA
DEFAULT_TRANSITION_AU = 20.0   # UA  anchura de la zona de transición

# ---------------------------------------------------------------------------
# Medio ambiente: densidad y velocidad del viento
# ---------------------------------------------------------------------------

def _blend(r: float, r_hp: float, delta: float) -> float:
    """Factor de mezcla ∈ [0, 1] para la transición heliopausa.

    - 0 → completamente en el viento solar (r ≤ r_hp).
    - 1 → completamente en el medio interestelar (r ≥ r_hp + delta).
    """
    if r <= r_hp:
        return 0.0
    if r >= r_hp + delta:
        return 1.0
    return (r - r_hp) / delta


def rho_wind(
    r: float,
    rho0: float = DEFAULT_RHO0,
    rho_ism: float = DEFAULT_RHO_ISM,
    heliopause_au: float = DEFAULT_HELIOPAUSE_AU,
    transition_au: float = DEFAULT_TRANSITION_AU,
    **_,
) -> float:
    """Densidad del viento (kg/m³) con transición suave en la heliopausa."""
    r_hp = heliopause_au * AU
    delta = transition_au * AU
    alpha = _blend(r, r_hp, delta)
    rho_solar = rho0 * (AU / r) ** 2
    return (1.0 - alpha) * rho_solar + alpha * rho_ism


def v_wind_at(
    r: float,
    v_sw: float = DEFAULT_V_SW,
    v_ism: float = DEFAULT_V_ISM,
    heliopause_au: float = DEFAULT_HELIOPAUSE_AU,
    transition_au: float = DEFAULT_TRANSITION_AU,
    **_,
) -> float:
    """Velocidad del viento ambiente (m/s) con transición suave en heliopausa."""
    r_hp = heliopause_au * AU
    delta = transition_au * AU
    alpha = _blend(r, r_hp, delta)
    return (1.0 - alpha) * v_sw + alpha * v_ism


# ---------------------------------------------------------------------------
# Ajuste SCM
# ---------------------------------------------------------------------------

def eta_scm_adjust(
    r: float,
    v: float,   # noqa: ARG001  (reservado para extensiones dependientes de v)
    rho: float,
    scm_strength: float = 1.2,
    rho0: float = DEFAULT_RHO0,
    **_,
) -> float:
    """Factor de forma de la burbuja según la densidad local (SCM).

    Escala con la densidad relativa al valor de referencia en 1 UA, de modo
    que el ajuste es físicamente significativo en todo el recorrido (viento
    solar y medio interestelar).

    Returns
    -------
    float ≥ 1.0
    """
    rho_ref = rho0  # densidad a 1 UA
    rho_ratio = max(rho / rho_ref, 1e-30)
    return 1.0 + 0.3 * scm_strength * rho_ratio ** 0.25


# ---------------------------------------------------------------------------
# Física de la burbuja
# ---------------------------------------------------------------------------

def equilibrium_radius(
    r: float,
    v: float,
    v_sw: float,
    R0: float = DEFAULT_R0,
    eta_shape: float = 1.0,
    **_,
) -> float:
    """Radio dinámico de la burbuja (m).

    Escala la presión dinámica relativa a la referencia en 1 UA con
    v_sw0 = 400 km/s.  El parámetro ``**_`` absorbe kwargs extra (rho0,
    rho_ism, …) propagados desde el resto del pipeline sin causar TypeError.

    ``abs(v_rel) + 1e-3`` evita la singularidad en v_rel = 0 y asegura
    continuidad simétrica alrededor de v_sw.
    """
    # Presión de referencia fija a 1 UA (constante de calibración)
    P_ref = DEFAULT_RHO0 * DEFAULT_V_SW ** 2
    v_rel_abs = abs(v_sw - v) + 1e-3
    P_dyn = DEFAULT_RHO0 * v_rel_abs ** 2
    scale = P_ref / (P_dyn + 1e-30)
    return R0 * scale ** (1.0 / 6.0) * eta_shape


def drag_force(
    r: float,
    v: float,
    v_sw: float,
    mass: float = DEFAULT_MASS,
    R0: float = DEFAULT_R0,
    eta_drag: float = DEFAULT_ETA_DRAG,
    rho0: float = DEFAULT_RHO0,
    rho_ism: float = DEFAULT_RHO_ISM,
    eta_shape: float = 1.0,
    heliopause_au: float = DEFAULT_HELIOPAUSE_AU,
    transition_au: float = DEFAULT_TRANSITION_AU,
    **_,
) -> tuple[float, float, float]:
    """Fuerza de arrastre firmada (N), radio de burbuja (m), densidad local (kg/m³).

    Ley cuadrática firmada:  F = 0.5 η ρ A v_rel |v_rel|
    """
    rho = rho_wind(
        r, rho0=rho0, rho_ism=rho_ism,
        heliopause_au=heliopause_au, transition_au=transition_au,
    )
    v_rel = v_sw - v
    R = equilibrium_radius(r, v, v_sw, R0=R0, rho0=rho0, eta_shape=eta_shape)
    area = np.pi * R ** 2
    F = 0.5 * eta_drag * rho * area * v_rel * abs(v_rel)
    return F, R, rho


def acceleration_net(
    r: float,
    v: float,
    v_sw: float,
    mass: float = DEFAULT_MASS,
    **kwargs,
) -> float:
    """Aceleración neta radial (m/s²): arrastre firmado − gravedad."""
    F_drag, _, _ = drag_force(r, v, v_sw, mass=mass, **kwargs)
    a_drag = F_drag / mass
    a_grav = -GM_SUN / r ** 2
    return a_drag + a_grav


def rf_power_required(R: float, k_rf: float = DEFAULT_K_RF) -> float:
    """Potencia RF para mantener el campo de la burbuja (W)."""
    return k_rf * R ** 3


def power_budget(
    r: float,
    v: float,
    v_sw: float,
    mass: float = DEFAULT_MASS,
    P_ai: float = DEFAULT_P_AI,
    k_rf: float = DEFAULT_K_RF,
    **kwargs,
) -> dict:
    """Balance de potencia completo.

    Returns
    -------
    dict with keys: P_gen, P_rf, P_ai, P_net, R, v_rel, rho
    """
    F_drag, R, rho = drag_force(r, v, v_sw, mass=mass, **kwargs)
    v_rel = v_sw - v
    P_gen = F_drag * v_rel
    P_rf = rf_power_required(R, k_rf=k_rf)
    P_net = P_gen - (P_rf + P_ai)
    return {
        "P_gen": P_gen,
        "P_rf": P_rf,
        "P_ai": P_ai,
        "P_net": P_net,
        "R": R,
        "v_rel": v_rel,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# Sistema ODE
# ---------------------------------------------------------------------------

def derivs(
    t: float,
    state: list[float],
    mass: float = DEFAULT_MASS,
    R0: float = DEFAULT_R0,
    eta_drag: float = DEFAULT_ETA_DRAG,
    rho0: float = DEFAULT_RHO0,
    rho_ism: float = DEFAULT_RHO_ISM,
    v_sw: float = DEFAULT_V_SW,
    v_ism: float = DEFAULT_V_ISM,
    heliopause_au: float = DEFAULT_HELIOPAUSE_AU,
    transition_au: float = DEFAULT_TRANSITION_AU,
    scm_strength: float = 1.2,
    **_,
) -> list[float]:
    """Derivadas del sistema ``[r, v]`` para ``solve_ivp``.

    El viento ambiente se calcula en cada paso según la posición,
    y el ajuste SCM se aplica antes de calcular el arrastre.
    """
    r, v = state
    v_sw_local = v_wind_at(
        r, v_sw=v_sw, v_ism=v_ism,
        heliopause_au=heliopause_au, transition_au=transition_au,
    )
    rho = rho_wind(
        r, rho0=rho0, rho_ism=rho_ism,
        heliopause_au=heliopause_au, transition_au=transition_au,
    )
    eta_shape = eta_scm_adjust(r, v, rho, scm_strength=scm_strength, rho0=rho0)
    a = acceleration_net(
        r, v, v_sw_local,
        mass=mass, R0=R0, eta_drag=eta_drag,
        rho0=rho0, rho_ism=rho_ism, eta_shape=eta_shape,
        heliopause_au=heliopause_au, transition_au=transition_au,
    )
    return [v, a]


# ---------------------------------------------------------------------------
# Simulación principal
# ---------------------------------------------------------------------------

def run_simulation(
    t_total_days: float = 7300.0,
    n_steps: int = 20_000,
    r0: float = AU,
    v0: float = 30_000.0,
    mass: float = DEFAULT_MASS,
    R0: float = DEFAULT_R0,
    eta_drag: float = DEFAULT_ETA_DRAG,
    rho0: float = DEFAULT_RHO0,
    rho_ism: float = DEFAULT_RHO_ISM,
    v_sw: float = DEFAULT_V_SW,
    v_ism: float = DEFAULT_V_ISM,
    P_ai: float = DEFAULT_P_AI,
    k_rf: float = DEFAULT_K_RF,
    scm_strength: float = 1.2,
    heliopause_au: float = DEFAULT_HELIOPAUSE_AU,
    transition_au: float = DEFAULT_TRANSITION_AU,
) -> dict:
    """Integra la trayectoria V5 con transición heliósfera–ISM.

    Returns
    -------
    dict with keys:
        t_days, r_AU, v_kms, R_km, P_gen, P_rf, P_net,
        v_rel_kms, v_wind_kms, rho, v0_kms, t_total_days
    """
    t_span = (0.0, t_total_days * 86400.0)
    t_eval = np.linspace(0.0, t_total_days * 86400.0, n_steps)

    mech_kw = dict(
        mass=mass, R0=R0, eta_drag=eta_drag,
        rho0=rho0, rho_ism=rho_ism,
        v_sw=v_sw, v_ism=v_ism,
        heliopause_au=heliopause_au, transition_au=transition_au,
        scm_strength=scm_strength,
    )

    sol = solve_ivp(
        lambda t, y: derivs(t, y, **mech_kw),
        t_span,
        [r0, v0],
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )

    r_arr = sol.y[0]
    v_arr = sol.y[1]
    t_sec = sol.t
    n = len(t_sec)

    R_arr = np.zeros(n)
    P_gen_arr = np.zeros(n)
    P_rf_arr = np.zeros(n)
    P_net_arr = np.zeros(n)
    v_rel_arr = np.zeros(n)
    v_wind_arr = np.zeros(n)
    rho_arr = np.zeros(n)

    for i in range(n):
        ri, vi = r_arr[i], v_arr[i]
        v_sw_local = v_wind_at(ri, v_sw=v_sw, v_ism=v_ism,
                               heliopause_au=heliopause_au, transition_au=transition_au)
        rho_loc = rho_wind(ri, rho0=rho0, rho_ism=rho_ism,
                           heliopause_au=heliopause_au, transition_au=transition_au)
        eta_shape = eta_scm_adjust(ri, vi, rho_loc, scm_strength=scm_strength, rho0=rho0)
        b = power_budget(
            ri, vi, v_sw_local,
            mass=mass, P_ai=P_ai, k_rf=k_rf,
            R0=R0, eta_drag=eta_drag,
            rho0=rho0, rho_ism=rho_ism, eta_shape=eta_shape,
            heliopause_au=heliopause_au, transition_au=transition_au,
        )
        R_arr[i] = b["R"]
        P_gen_arr[i] = b["P_gen"]
        P_rf_arr[i] = b["P_rf"]
        P_net_arr[i] = b["P_net"]
        v_rel_arr[i] = b["v_rel"]
        v_wind_arr[i] = v_sw_local
        rho_arr[i] = b["rho"]

    return {
        "t_days": t_sec / 86400.0,
        "r_AU": r_arr / AU,
        "v_kms": v_arr / 1000.0,
        "R_km": R_arr / 1000.0,
        "P_gen": P_gen_arr,
        "P_rf": P_rf_arr,
        "P_net": P_net_arr,
        "v_rel_kms": v_rel_arr / 1000.0,
        "v_wind_kms": v_wind_arr / 1000.0,
        "rho": rho_arr,
        "v0_kms": v0 / 1000.0,
        "t_total_days": t_total_days,
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_results(res: dict, out_path: str | Path | None = None) -> plt.Figure:
    """Figura de 5 paneles: velocidad, distancia, radio, potencia, densidad."""
    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    t = res["t_days"]

    axs[0].plot(t, res["v_kms"], "b-")
    axs[0].set_ylabel("Velocidad (km/s)")
    axs[0].grid(True)

    axs[1].plot(t, res["r_AU"], color="orange")
    axs[1].set_ylabel("Distancia (AU)")
    axs[1].grid(True)

    axs[2].plot(t, res["R_km"], "g-")
    axs[2].set_ylabel("Radio burbuja (km)")
    axs[2].grid(True)

    axs[3].plot(t, res["P_gen"], "r-", label="P_gen")
    axs[3].plot(t, res["P_rf"], color="orange", label="P_rf")
    axs[3].plot(t, res["P_net"], "purple", label="P_net")
    axs[3].axhline(0, color="gray", linestyle="--")
    axs[3].legend()
    axs[3].set_ylabel("Potencia (W)")
    axs[3].grid(True)

    axs[4].plot(t, res["rho"], color="brown", label="ρ (kg/m³)")
    axs[4].set_ylabel("Densidad (kg/m³)")
    axs[4].set_xlabel("Tiempo (días)")
    axs[4].legend()
    axs[4].grid(True)

    plt.suptitle("SCM Motor de Velos V5 — Interestelar")
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    """Punto de entrada CLI. Devuelve el diccionario de resultados."""
    parser = argparse.ArgumentParser(
        description="SCM Motor de Velos V5 — transición heliósfera/ISM."
    )
    parser.add_argument("--t-days", type=float, default=7300.0, metavar="DÍAS",
                        help="Días de simulación (por defecto: 7300 = 20 años).")
    parser.add_argument("--n-steps", type=int, default=20_000, metavar="N",
                        help="Pasos de evaluación (por defecto: 20000).")
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS, metavar="KG")
    parser.add_argument("--r0-km", type=float, default=50.0, metavar="KM",
                        help="Radio inicial burbuja (km, por defecto: 50).")
    parser.add_argument("--eta-drag", type=float, default=DEFAULT_ETA_DRAG)
    parser.add_argument("--P-ai", type=float, default=DEFAULT_P_AI, metavar="W")
    parser.add_argument("--scm-strength", type=float, default=1.2)
    parser.add_argument("--heliopause-au", type=float, default=DEFAULT_HELIOPAUSE_AU,
                        help="Posición de la heliopausa (UA, por defecto: 120).")
    parser.add_argument("--transition-au", type=float, default=DEFAULT_TRANSITION_AU,
                        help="Anchura de la zona de transición (UA, por defecto: 20).")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    res = run_simulation(
        t_total_days=args.t_days,
        n_steps=args.n_steps,
        mass=args.mass,
        R0=args.r0_km * 1000.0,
        eta_drag=args.eta_drag,
        P_ai=args.P_ai,
        scm_strength=args.scm_strength,
        heliopause_au=args.heliopause_au,
        transition_au=args.transition_au,
    )

    print("--- SCM V5: Viento variable + ISM + Heliopausa ---")
    print(f"Días simulados:           {res['t_days'][-1]:.1f}  "
          f"({res['t_days'][-1]/365.25:.2f} años)")
    print(f"Distancia final:          {res['r_AU'][-1]:.1f} UA")
    print(f"Velocidad final:          {res['v_kms'][-1]:.2f} km/s")
    print(f"Delta-V total:            {res['v_kms'][-1] - res['v0_kms']:.2f} km/s")
    print(f"Radio burbuja final:      {res['R_km'][-1]:.2f} km")
    print(f"Potencia neta final:      {res['P_net'][-1]:.2f} W")
    print(f"Potencia neta media:      {np.mean(res['P_net']):.2f} W")
    print(f"Potencia neta mínima:     {np.min(res['P_net']):.2f} W")
    print(f"Siempre P_net > 0?        {bool(np.all(res['P_net'] > 0))}")

    if not args.no_plot:
        fig = plot_results(res, out_path=args.out)
        if args.out:
            print(f"Figura guardada en {args.out}")
        plt.close(fig)

    return res


if __name__ == "__main__":
    main()
