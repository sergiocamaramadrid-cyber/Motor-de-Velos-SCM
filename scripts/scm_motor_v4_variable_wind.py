#!/usr/bin/env python3
"""
scripts/scm_motor_v4_variable_wind.py
V4: viento solar variable + ajuste SCM + balance energético cerrado

Extiende V3 (scm_motor_v3_closed_loop.py) con:
  · Viento solar variable: oscilación semestral lenta + CMEs aleatorias.
  · eta_scm_adjust(): factor de forma de la burbuja que depende de la
    densidad local, simulando la salida del Framework SCM.
  · Figura de 5 paneles (ΔV, distancia, radio, potencia, v_rel/v_sw).

Bugs corregidos respecto al script del problem statement:
  1. ``acceleration_net(…, mass=DEFAULT_R0, …)`` usaba DEFAULT_R0=50e3 como
     masa por defecto en lugar de DEFAULT_MASS=50.0. Corregido.
  2. ``drag_force`` no aceptaba kwargs extra (P_ai, k_rf) → TypeError cuando
     se propagan desde ``derivs``. Corregido añadiendo ``**_``.
  3. ``eta_scm_adjust`` no se llamaba dentro de ``derivs`` (solo en el bucle
     de métricas), de modo que la integración usaba eta_shape=1 siempre.
     Corregido: derivs calcula eta_shape en cada paso.
  4. El bucle de métricas post-integración recalculaba R/P con eta_shape SCM
     pero la trayectoria ODE no la usó → inconsistencia. Corregido al unificar.

Usage
-----
10 años por defecto::

    python scripts/scm_motor_v4_variable_wind.py

1 año para pruebas rápidas::

    python scripts/scm_motor_v4_variable_wind.py --t-days 365 --n-steps 5000

Con salida PNG::

    python scripts/scm_motor_v4_variable_wind.py --out results/v4/sim.png
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

DEFAULT_MASS = 50.0         # kg
DEFAULT_R0 = 50e3           # m (radio de referencia de la burbuja)
DEFAULT_ETA_DRAG = 0.5
DEFAULT_RHO0 = 8e-20        # kg/m³ a 1 UA
DEFAULT_P_AI = 8.0          # W (consumo IA)
DEFAULT_K_RF = 1.2e-13      # W/m³ (calibrado: 15 W @ R = 50 km)

# ---------------------------------------------------------------------------
# Modelo de viento solar variable
# ---------------------------------------------------------------------------

def v_sw_time_dependent(
    t_days: np.ndarray,
    base: float = 400e3,
    slow: float = 300e3,
    cme_prob: float = 0.003,
    random_seed: int = 42,
) -> np.ndarray:
    """Velocidad del viento solar (m/s) en función del tiempo (días).

    - Oscilación semestral entre *base* y *slow* (2 ciclos por año).
    - CMEs aleatorias: probabilidad diaria *cme_prob* (~1 cada 3 meses con
      el valor por defecto 0.003), pico entre 600 y 900 km/s, duración 2–5 d.

    Parameters
    ----------
    t_days : np.ndarray
        Vector de tiempos en días.
    base : float
        Velocidad base del viento rápido (m/s).
    slow : float
        Velocidad del viento lento (m/s).
    cme_prob : float
        Probabilidad diaria de CME por elemento de t_days.
    random_seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    np.ndarray of shape (len(t_days),)
    """
    t_years = t_days / 365.25
    # Viento lento periódico (2 ciclos por año)
    v = base + (slow - base) * (0.5 + 0.5 * np.cos(4 * np.pi * t_years))
    v = np.clip(v, slow, base)

    # CMEs aleatorias
    rng = np.random.default_rng(random_seed)
    cme_events = rng.random(len(t_days)) < cme_prob
    for i in np.where(cme_events)[0]:
        duration = rng.integers(2, 6)        # días
        peak = rng.uniform(600e3, 900e3)     # m/s
        start = max(0, i - duration // 2)
        end = min(len(t_days), i + duration // 2 + 1)
        for j in range(start, end):
            dt = t_days[j] - t_days[i]
            factor = np.exp(-(dt / max(duration / 2.0, 1e-9)) ** 2)
            v[j] = max(v[j], peak * factor)
    return v


# ---------------------------------------------------------------------------
# Ajuste SCM (simula salida del Framework de Velos)
# ---------------------------------------------------------------------------

def eta_scm_adjust(
    r: float,
    v: float,   # noqa: ARG001  (kept for API symmetry with future velocity-dependent models)
    rho: float, # noqa: ARG001  (kept for API symmetry; use rho_wind(r) internally if needed)
    scm_strength: float = 1.2,
) -> float:
    """Factor de forma de la burbuja según la densidad local.

    Parameters
    ----------
    r : float
        Posición heliocéntrica (m). Determina la tendencia de densidad.
    v : float
        Velocidad de la sonda (m/s). Reservado para modelos futuros
        dependientes de la velocidad relativa.
    rho : float
        Densidad local del viento (kg/m³). Reservado para extensiones
        que escalen directamente con la densidad medida.
    scm_strength : float
        Intensidad del ajuste (adimensional). En una implementación real
        vendría de la predicción de densidad del velo.

    Returns
    -------
    float ≥ 1.0
    """
    return 1.0 + 0.3 * scm_strength * (AU / r) ** 0.5


# ---------------------------------------------------------------------------
# Funciones físicas
# ---------------------------------------------------------------------------

def rho_wind(r: float, rho0: float = DEFAULT_RHO0, r0: float = AU) -> float:
    """Densidad del viento solar a distancia *r* (kg/m³)."""
    return rho0 * (r0 / r) ** 2


def equilibrium_radius(
    r: float,
    v: float,
    v_sw: float,
    R0: float = DEFAULT_R0,
    rho0: float = DEFAULT_RHO0,
    eta_shape: float = 1.0,
    **_,
) -> float:
    """Radio dinámico de la burbuja (m).

    Usa ``abs(v_rel) + 1e-3`` para evitar la singularidad en v_rel = 0 y
    para que el radio sea continuo y simétrico alrededor de v_sw.
    """
    rho = rho_wind(r, rho0)
    v_rel_abs = abs(v_sw - v) + 1e-3
    P_dyn = rho * v_rel_abs ** 2
    P_ref = rho0 * (400e3) ** 2   # presión dinámica de referencia a 1 UA
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
    eta_shape: float = 1.0,
    **_,
) -> tuple[float, float]:
    """Fuerza de arrastre firmada (N) y radio de burbuja (m).

    Ley cuadrática firmada: ``F = 0.5 η ρ A v_rel |v_rel|``
    - v_rel > 0 (v < v_sw): F > 0 → empuje.
    - v_rel < 0 (v > v_sw): F < 0 → frenado aerodinámico.
    """
    rho = rho_wind(r, rho0)
    v_rel = v_sw - v
    R = equilibrium_radius(r, v, v_sw, R0=R0, rho0=rho0, eta_shape=eta_shape)
    area = np.pi * R ** 2
    F = 0.5 * eta_drag * rho * area * v_rel * abs(v_rel)
    return F, R


def acceleration_net(
    r: float,
    v: float,
    v_sw: float,
    mass: float = DEFAULT_MASS,
    **kwargs,
) -> float:
    """Aceleración neta radial (m/s²): arrastre firmado − gravedad."""
    F_drag, _ = drag_force(r, v, v_sw, mass=mass, **kwargs)
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
    dict with keys: P_gen, P_rf, P_ai, P_net, R, v_rel
    """
    F_drag, R = drag_force(r, v, v_sw, mass=mass, **kwargs)
    v_rel = v_sw - v
    P_gen = F_drag * v_rel   # positivo al acelerar, negativo al frenar
    P_rf = rf_power_required(R, k_rf=k_rf)
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
# Sistema ODE
# ---------------------------------------------------------------------------

def derivs(
    t: float,
    state: list[float],
    t_days_arr: np.ndarray,
    v_sw_arr: np.ndarray,
    mass: float = DEFAULT_MASS,
    R0: float = DEFAULT_R0,
    eta_drag: float = DEFAULT_ETA_DRAG,
    rho0: float = DEFAULT_RHO0,
    scm_strength: float = 1.2,
    **_,
) -> list[float]:
    """Derivadas del sistema ``[r, v]`` para ``solve_ivp``.

    El viento solar se interpola del array pre-generado *v_sw_arr*.
    El ajuste SCM (eta_shape) se calcula en cada llamada para que la
    integración use la burbuja correcta, no un valor fijo de eta_shape=1.
    """
    r, v = state
    t_days = t / 86400.0
    v_sw = float(np.interp(t_days, t_days_arr, v_sw_arr))
    rho = rho_wind(r, rho0)
    eta_shape = eta_scm_adjust(r, v, rho, scm_strength=scm_strength)
    a = acceleration_net(
        r, v, v_sw,
        mass=mass, R0=R0, eta_drag=eta_drag, rho0=rho0, eta_shape=eta_shape,
    )
    return [v, a]


# ---------------------------------------------------------------------------
# Simulación principal
# ---------------------------------------------------------------------------

def run_simulation(
    t_total_days: float = 3650.0,
    n_steps: int = 50_000,
    r0: float = AU,
    v0: float = 30_000.0,
    mass: float = DEFAULT_MASS,
    R0: float = DEFAULT_R0,
    eta_drag: float = DEFAULT_ETA_DRAG,
    rho0: float = DEFAULT_RHO0,
    P_ai: float = DEFAULT_P_AI,
    k_rf: float = DEFAULT_K_RF,
    scm_strength: float = 1.2,
    random_seed: int = 42,
) -> dict:
    """Integra la trayectoria V4 con viento variable y ajuste SCM.

    Parameters
    ----------
    t_total_days : float
        Duración de la simulación (días).
    n_steps : int
        Número de pasos de evaluación del integrador.
    r0 : float
        Posición inicial (m).
    v0 : float
        Velocidad inicial (m/s).

    Returns
    -------
    dict with keys:
        t_days, r_AU, v_kms, R_km, P_gen, P_rf, P_net,
        v_rel_kms, v_sw_interp, v0_kms, t_total_days
    """
    # Pregenerar el perfil del viento solar
    t_days_arr = np.linspace(0.0, t_total_days, n_steps)
    v_sw_arr = v_sw_time_dependent(t_days_arr, random_seed=random_seed)

    # Integrar con solve_ivp DOP853
    t_span = (0.0, t_total_days * 86400.0)
    t_eval = np.linspace(0.0, t_total_days * 86400.0, n_steps)

    sol = solve_ivp(
        lambda t, y: derivs(
            t, y, t_days_arr, v_sw_arr,
            mass=mass, R0=R0, eta_drag=eta_drag, rho0=rho0,
            scm_strength=scm_strength,
        ),
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

    # Métricas en cada paso evaluado
    R_arr = np.zeros(n)
    P_gen_arr = np.zeros(n)
    P_rf_arr = np.zeros(n)
    P_net_arr = np.zeros(n)
    v_rel_arr = np.zeros(n)

    for i in range(n):
        t_d = t_sec[i] / 86400.0
        v_sw_local = float(np.interp(t_d, t_days_arr, v_sw_arr))
        rho_loc = rho_wind(r_arr[i], rho0)
        eta_shape = eta_scm_adjust(r_arr[i], v_arr[i], rho_loc, scm_strength)
        b = power_budget(
            r_arr[i], v_arr[i], v_sw_local,
            mass=mass, P_ai=P_ai, k_rf=k_rf,
            R0=R0, eta_drag=eta_drag, rho0=rho0, eta_shape=eta_shape,
        )
        R_arr[i] = b["R"]
        P_gen_arr[i] = b["P_gen"]
        P_rf_arr[i] = b["P_rf"]
        P_net_arr[i] = b["P_net"]
        v_rel_arr[i] = b["v_rel"]

    return {
        "t_days": t_sec / 86400.0,
        "r_AU": r_arr / AU,
        "v_kms": v_arr / 1000.0,
        "R_km": R_arr / 1000.0,
        "P_gen": P_gen_arr,
        "P_rf": P_rf_arr,
        "P_net": P_net_arr,
        "v_rel_kms": v_rel_arr / 1000.0,
        "v_sw_interp": np.interp(t_sec / 86400.0, t_days_arr, v_sw_arr) / 1000.0,
        "v0_kms": v0 / 1000.0,
        "t_total_days": t_total_days,
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_results(res: dict, out_path: str | Path | None = None) -> plt.Figure:
    """Figura de 5 paneles: ΔV, distancia, radio, potencia, velocidades."""
    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    t = res["t_days"]

    axs[0].plot(t, res["v_kms"] - res["v0_kms"], "b-")
    axs[0].set_ylabel("Delta-V (km/s)")
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

    axs[4].plot(t, res["v_rel_kms"], "c-", label="v_rel")
    axs[4].plot(t, res["v_sw_interp"], "k--", alpha=0.5, label="v_sw")
    axs[4].set_ylabel("Velocidad (km/s)")
    axs[4].set_xlabel("Tiempo (días)")
    axs[4].legend()
    axs[4].grid(True)

    plt.suptitle("SCM V4: Viento variable + ajuste SCM")
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
        description="SCM Motor de Velos V4 — viento variable + ajuste SCM."
    )
    parser.add_argument("--t-days", type=float, default=3650.0, metavar="DÍAS",
                        help="Días de simulación (por defecto: 3650 = 10 años).")
    parser.add_argument("--n-steps", type=int, default=50_000, metavar="N",
                        help="Pasos de evaluación (por defecto: 50000).")
    parser.add_argument("--mass", type=float, default=DEFAULT_MASS, metavar="KG",
                        help=f"Masa de la sonda (kg, por defecto: {DEFAULT_MASS}).")
    parser.add_argument("--r0-km", type=float, default=50.0, metavar="KM",
                        help="Radio inicial burbuja (km, por defecto: 50).")
    parser.add_argument("--eta-drag", type=float, default=DEFAULT_ETA_DRAG,
                        help=f"Eficiencia de arrastre (por defecto: {DEFAULT_ETA_DRAG}).")
    parser.add_argument("--P-ai", type=float, default=DEFAULT_P_AI, metavar="W",
                        help=f"Potencia IA fija (W, por defecto: {DEFAULT_P_AI}).")
    parser.add_argument("--scm-strength", type=float, default=1.2,
                        help="Intensidad del ajuste SCM (por defecto: 1.2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para CMEs (por defecto: 42).")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de la figura PNG de salida.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Omite la generación de la figura.")
    args = parser.parse_args(argv)

    res = run_simulation(
        t_total_days=args.t_days,
        n_steps=args.n_steps,
        mass=args.mass,
        R0=args.r0_km * 1000.0,
        eta_drag=args.eta_drag,
        P_ai=args.P_ai,
        scm_strength=args.scm_strength,
        random_seed=args.seed,
    )

    print("--- SCM V4: Viento variable + Ajuste SCM ---")
    print(f"Días simulados:           {res['t_days'][-1]:.1f}  "
          f"({res['t_days'][-1]/365.25:.2f} años)")
    print(f"Distancia final:          {res['r_AU'][-1]:.2f} UA")
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
