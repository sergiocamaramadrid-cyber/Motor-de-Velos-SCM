#!/usr/bin/env python3
"""
scripts/scm_motor_v3_closed_loop.py
V3: burbuja dinámica + back-EMF + balance energético + control RF adaptativo

Bugs corregidos respecto al script original del problem statement:
  1. ``def derivs(state, t, **kwargs)`` + ``args=(kwargs,)`` → TypeError.
     Corregido a ``def derivs(state, t, kw=None)`` con ``args=(kw,)``.
  2. ``drag_force`` / ``equilibrium_radius`` / ``acceleration_net`` no
     aceptaban kwargs extra (P_ai, eta_shape, …). Corregido añadiendo ``**_``.
  3. ``back_emf_power`` propagaba kwargs extra a ``drag_force``. Corregido.
  4. ``main()`` no aceptaba ``argv`` → no testeable. Corregido a
     ``main(argv=None)`` con ``parse_args(argv)``.
  5. ``derivs`` mutaba el dict ``kw`` compartido entre llamadas de odeint,
     corrompiendo la estimación del Jacobiano y causando divergencia numérica.
     Corregido extendiendo el vector de estado a ``[r, v, eta_shape]`` para
     que el control adaptativo sea dinámica ODE correcta.

Usage
-----
Simulación por defecto (1 año)::

    python scripts/scm_motor_v3_closed_loop.py

5 años con más pasos::

    python scripts/scm_motor_v3_closed_loop.py --t-days 1825 --n-steps 50000

Con directorio de salida::

    python scripts/scm_motor_v3_closed_loop.py --out results/v3/sim.png
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

AU = 1.495978707e11      # Unidad Astronómica (m)
G = 6.67430e-11          # Constante gravitacional (m³ kg⁻¹ s⁻²)
M_SUN = 1.9885e30        # Masa del Sol (kg)
GM_SUN = G * M_SUN       # Parámetro gravitacional estándar (m³/s²)

# ---------------------------------------------------------------------------
# Medio heliosférico
# ---------------------------------------------------------------------------

def rho_wind(r: float, rho0: float = 8e-20, r0: float = AU) -> float:
    """Densidad del viento solar a distancia *r* (kg/m³)."""
    return rho0 * (r0 / r) ** 2


def v_sw_const(r: float, v_sw0: float = 400e3) -> float:  # noqa: ARG001
    """Velocidad del viento solar (modelo constante, m/s)."""
    return v_sw0


# ---------------------------------------------------------------------------
# Burbuja magnética dinámica
# ---------------------------------------------------------------------------

def equilibrium_radius(
    r: float,
    v: float,
    R0: float = 50e3,
    rho0: float = 8e-20,
    v_sw0: float = 400e3,
    eta_shape: float = 1.0,
    **_,
) -> float:
    """Radio de equilibrio dinámico de la burbuja (m)."""
    rho = rho_wind(r, rho0)
    v_rel = max(v_sw_const(r, v_sw0) - v, 1.0)
    scale = (rho0 * v_sw0 ** 2) / (rho * v_rel ** 2 + 1e-12)
    return R0 * scale ** (1.0 / 6.0) * eta_shape


# ---------------------------------------------------------------------------
# Fuerzas
# ---------------------------------------------------------------------------

def drag_force(
    r: float,
    v: float,
    mass: float = 50.0,
    R0: float = 50e3,
    eta_drag: float = 0.5,
    rho0: float = 8e-20,
    v_sw0: float = 400e3,
    eta_shape: float = 1.0,
    **_,
) -> tuple[float, float, float]:
    """Devuelve ``(F_drag, R, v_rel)``."""
    R = equilibrium_radius(r, v, R0=R0, rho0=rho0, v_sw0=v_sw0, eta_shape=eta_shape)
    area = np.pi * R ** 2
    rho = rho_wind(r, rho0)
    v_rel = max(v_sw_const(r, v_sw0) - v, 0.0)
    F = 0.5 * eta_drag * rho * area * v_rel ** 2
    return F, R, v_rel


def acceleration_net(
    r: float,
    v: float,
    mass: float = 50.0,
    **kwargs,
) -> float:
    """Aceleración neta radial (m/s²): arrastre − gravedad."""
    F_drag, _, _ = drag_force(r, v, mass=mass, **kwargs)
    a_drag = F_drag / mass
    a_grav = -GM_SUN / r ** 2
    return a_drag + a_grav


def back_emf_power(r: float, v: float, **kwargs) -> float:
    """Potencia back-EMF extraída de la interacción (W)."""
    F_drag, _, v_rel = drag_force(r, v, **kwargs)
    return F_drag * v_rel


# ---------------------------------------------------------------------------
# Balance energético
# ---------------------------------------------------------------------------

def rf_power_required(R: float, k_rf: float = 1.2e-13) -> float:
    """Potencia RF para mantener el campo (W).

    Calibrado de forma que R = 50 km (= 50 000 m) → P_rf ≈ 15 W.
    El coeficiente k_rf = 1.2e-13 W/m³ corresponde a la relación
    P_rf = k · R³ cuando R se expresa en metros.
    (El script original usaba k_rf=1.2e-8 calibrado para R en km,
    lo que producía ~1.5 MW en lugar de 15 W.)
    """
    return k_rf * R ** 3


def power_budget(
    r: float,
    v: float,
    mass: float = 50.0,
    P_ai: float = 8.0,
    **kwargs,
) -> dict:
    """Calcula el balance de potencia completo.

    Returns
    -------
    dict with keys: P_gen, P_rf, P_ai, P_net, R, v_rel
    """
    F_drag, R, v_rel = drag_force(r, v, mass=mass, **kwargs)
    P_gen = F_drag * v_rel
    P_rf = rf_power_required(R)
    P_net = P_gen - (P_rf + P_ai)
    return {
        "P_gen": P_gen,
        "P_rf": P_rf,
        "P_ai": P_ai,
        "P_net": P_net,
        "R": R,
        "v_rel": v_rel,
    }


# ---------------------------------------------------------------------------
# Sistema ODE con control adaptativo
# ---------------------------------------------------------------------------

# Constante de tiempo del control adaptativo.
# Corresponde aproximadamente a una variación de ±0.1 % por paso si el paso
# típico es 1/10 000 del año de simulación.
_TAU_CTRL = 36.5 * 86400.0   # 36.5 días (s)
_ETA_MIN = 0.5
_ETA_MAX = 1.2


def derivs(state: list[float], t: float, kw: dict | None = None) -> list[float]:
    """Derivadas del sistema.  Usada por ``solve_ivp`` vía clausura ``rhs``.

    Estado extendido: ``[r, v, eta_shape]``.

    El control adaptativo está formulado como dinámica ODE continua sobre
    ``eta_shape`` para evitar la corrupción del estimador del Jacobiano que
    produce la mutación del dict compartido ``kw``.

    Parameters
    ----------
    state : [r, v, eta_shape]
    t : float — tiempo (s), no usado directamente
    kw : dict — parámetros del sistema (R0, rho0, v_sw0, mass, P_ai, …)
    """
    if kw is None:
        kw = {}
    r, v, eta_s = state
    kw_local = dict(kw, eta_shape=float(eta_s))

    budget = power_budget(r, v, **kw_local)
    a = acceleration_net(r, v, **kw_local)

    if budget["P_net"] < 0:
        deta = -(eta_s - _ETA_MIN) / _TAU_CTRL  # contrae hacia _ETA_MIN
    else:
        deta = (_ETA_MAX - eta_s) / _TAU_CTRL   # expande hacia _ETA_MAX

    return [v, a, deta]


# ---------------------------------------------------------------------------
# Simulación principal
# ---------------------------------------------------------------------------

def run_simulation(
    t_total_days: float = 365,
    n_steps: int = 10_000,
    r0: float = AU,
    v0: float = 30_000.0,
    **kwargs,
) -> dict:
    """Integra la trayectoria V3 con balance energético cerrado.

    Estado ODE: ``[r, v, eta_shape]``.

    Parameters
    ----------
    t_total_days : float
        Duración de la simulación (días).
    n_steps : int
        Número de pasos de integración.
    r0 : float
        Posición inicial (m).
    v0 : float
        Velocidad inicial (m/s).
    **kwargs
        Parámetros del Plasma Magnet / sonda.

    Returns
    -------
    dict with keys:
        t_days, r_AU, v_kms, R_km, P_gen, P_rf, P_net,
        v0_kms, t_total_days, kwargs
    """
    t_sec = t_total_days * 86400.0
    eta_shape0 = kwargs.pop("eta_shape", 1.0)
    kw = dict(kwargs)
    state0 = [r0, v0, eta_shape0]

    t_eval = np.linspace(0, t_sec, n_steps)

    def rhs(t, state):
        return derivs(state, t, kw)

    sol_ivp = solve_ivp(
        rhs,
        [0.0, t_sec],
        state0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-6,
        atol=[1e6, 1e-1, 1e-10],
        dense_output=False,
    )

    r_arr = sol_ivp.y[0]
    v_arr = sol_ivp.y[1]
    eta_arr = sol_ivp.y[2]

    R_arr = np.zeros(n_steps)
    P_gen_arr = np.zeros(n_steps)
    P_rf_arr = np.zeros(n_steps)
    P_net_arr = np.zeros(n_steps)
    for i, (ri, vi, ei) in enumerate(zip(r_arr, v_arr, eta_arr)):
        b = power_budget(ri, vi, eta_shape=float(ei), **kw)
        R_arr[i] = b["R"]
        P_gen_arr[i] = b["P_gen"]
        P_rf_arr[i] = b["P_rf"]
        P_net_arr[i] = b["P_net"]

    return {
        "t_days": sol_ivp.t / 86400.0,
        "r_AU": r_arr / AU,
        "v_kms": v_arr / 1000.0,
        "R_km": R_arr / 1000.0,
        "P_gen": P_gen_arr,
        "P_rf": P_rf_arr,
        "P_net": P_net_arr,
        "v0_kms": v0 / 1000.0,
        "t_total_days": t_total_days,
        "kwargs": kw,
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_results(res: dict, out_path: str | Path | None = None) -> plt.Figure:
    """Genera la figura de cuatro paneles."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(res["t_days"], res["v_kms"] - res["v0_kms"], "b-")
    axs[0].set_ylabel("Delta-V (km/s)")
    axs[0].grid(True)

    axs[1].plot(res["t_days"], res["r_AU"], color="orange")
    axs[1].set_ylabel("Distancia (AU)")
    axs[1].grid(True)

    axs[2].plot(res["t_days"], res["R_km"], "g-")
    axs[2].set_ylabel("Radio burbuja (km)")
    axs[2].grid(True)

    axs[3].plot(res["t_days"], res["P_gen"], "r-", label="P_gen")
    axs[3].plot(res["t_days"], res["P_rf"], color="orange", label="P_rf")
    axs[3].plot(res["t_days"], res["P_net"], "purple", label="P_net")
    axs[3].axhline(0, color="gray", linestyle="--")
    axs[3].legend()
    axs[3].set_ylabel("Potencia (W)")
    axs[3].set_xlabel("Tiempo (días)")
    axs[3].grid(True)

    plt.suptitle("SCM-Motor de Velos V3: Bucle energético cerrado")
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    """Punto de entrada CLI.  Devuelve el diccionario de resultados."""
    parser = argparse.ArgumentParser(
        description="SCM Motor de Velos V3 — simulación ODE con bucle energético cerrado."
    )
    parser.add_argument("--t-days", type=float, default=365.0, metavar="DÍAS",
                        help="Días de simulación (por defecto: 365).")
    parser.add_argument("--n-steps", type=int, default=10_000, metavar="N",
                        help="Pasos de integración (por defecto: 10000).")
    parser.add_argument("--mass", type=float, default=50.0, metavar="KG",
                        help="Masa de la sonda (kg, por defecto: 50).")
    parser.add_argument("--r0-km", type=float, default=50.0, metavar="KM",
                        help="Radio inicial burbuja (km, por defecto: 50).")
    parser.add_argument("--eta-drag", type=float, default=0.5,
                        help="Eficiencia de arrastre (por defecto: 0.5).")
    parser.add_argument("--P-ai", type=float, default=8.0, metavar="W",
                        help="Potencia IA fija (W, por defecto: 8).")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de la figura PNG de salida.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Omite la generación de la figura.")
    args = parser.parse_args(argv)

    kwargs = {
        "mass": args.mass,
        "R0": args.r0_km * 1000.0,
        "eta_drag": args.eta_drag,
        "rho0": 8e-20,
        "v_sw0": 400e3,
        "P_ai": args.P_ai,
    }
    res = run_simulation(t_total_days=args.t_days, n_steps=args.n_steps, **kwargs)

    print("--- SCM MOTOR DE VELOS V3 (Bucle cerrado) ---")
    print(f"Distancia final:          {res['r_AU'][-1]:.3f} UA")
    print(f"Delta-V total:            {res['v_kms'][-1] - res['v0_kms']:.2f} km/s")
    print(f"Radio burbuja final:      {res['R_km'][-1]:.2f} km")
    print(f"Potencia generada final:  {res['P_gen'][-1]:.2f} W")
    print(f"Potencia RF final:        {res['P_rf'][-1]:.2f} W")
    print(f"Potencia neta final:      {res['P_net'][-1]:.2f} W")
    print(f"Potencia neta media:      {np.mean(res['P_net']):.2f} W")

    if not args.no_plot:
        fig = plot_results(res, out_path=args.out)
        if args.out:
            print(f"Figura guardada en {args.out}")
        plt.close(fig)

    return res


if __name__ == "__main__":
    main()
