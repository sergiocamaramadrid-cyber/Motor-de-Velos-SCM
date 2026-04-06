"""
scripts/solar_wind_ode_sim.py — Simulación ODE del "Surfing" Heliosférico con Plasma Magnet.

Integra la trayectoria radial de una sonda equipada con un Plasma Magnet usando
``scipy.integrate.odeint``.  La aceleración neta combina el empuje de arrastre
del viento solar sobre la burbuja magnética y la gravedad heliocéntrica del Sol.

Bugs corregidos respecto al código original del problem statement:
  1. ``v &lt; v_sw``  →  ``v < v_sw``  (entidad HTML → operador Python)
  2. ``return`` de ``derivs`` separado de la asignación de ``t`` (línea fusionada)
  3. ``state0`` definido explícitamente antes de llamar a ``odeint``
  4. Indexación de ``sol`` corregida: ``sol[-1, 0]`` y ``sol[-1, 1]``
  5. ``delta_v`` definido como ``sol[-1, 1] - state0[1]``

Usage
-----
Simulación con parámetros por defecto (1 año)::

    python scripts/solar_wind_ode_sim.py

Con directorio de salida::

    python scripts/solar_wind_ode_sim.py --out results/ode_sim

Sin figura::

    python scripts/solar_wind_ode_sim.py --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------

G = 6.67430e-11          # Constante gravitacional (m³ kg⁻¹ s⁻²)
M_SOL = 1.989e30         # Masa del Sol (kg)
AU = 1.496e11            # Unidad Astronómica (m)

# ---------------------------------------------------------------------------
# Parámetros de la sonda / Plasma Magnet (por defecto)
# ---------------------------------------------------------------------------

MASS_PROBE_DEFAULT = 50.0          # Masa de la sonda (kg)
R_BUBBLE_DEFAULT = 50e3            # Radio burbuja magnética (m)
ETA_DEFAULT = 0.5                  # Eficiencia de interacción (0–1)
RHO0_DEFAULT = 8e-20               # Densidad viento solar a 1 UA (kg/m³)
V_SW_DEFAULT = 400e3               # Velocidad viento solar (m/s)

# ---------------------------------------------------------------------------
# Condiciones iniciales (por defecto)
# ---------------------------------------------------------------------------

R0_DEFAULT = 1.0 * AU              # Posición inicial: 1 UA (m)
V0_DEFAULT = 30e3                  # Velocidad heliocéntrica inicial (m/s)


# ---------------------------------------------------------------------------
# Física del motor de velas (aceleración neta)
# ---------------------------------------------------------------------------

def accel_net(
    r: float,
    v: float,
    *,
    R_bubble: float = R_BUBBLE_DEFAULT,
    eta: float = ETA_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    v_sw: float = V_SW_DEFAULT,
    mass: float = MASS_PROBE_DEFAULT,
) -> float:
    """Aceleración neta radial de la sonda (m/s²).

    Combina el empuje de arrastre del viento solar (Plasma Magnet) y la
    gravedad heliocéntrica.

    Parameters
    ----------
    r : float
        Posición heliocéntrica radial (m).
    v : float
        Velocidad heliocéntrica radial (m/s).
    R_bubble : float, optional
        Radio de la burbuja magnética (m).
    eta : float, optional
        Factor de eficiencia de la interacción (0–1).
    rho0 : float, optional
        Densidad del viento solar a 1 UA (kg/m³).
    v_sw : float, optional
        Velocidad del viento solar (m/s).
    mass : float, optional
        Masa de la sonda (kg).

    Returns
    -------
    float
        Aceleración neta (m/s²).  Positiva = hacia afuera.
    """
    rho = rho0 * (AU / r) ** 2
    A = np.pi * R_bubble ** 2
    if v < v_sw:
        F_drag = 0.5 * eta * rho * A * (v_sw - v) ** 2
    else:
        F_drag = 0.0
    a_drag = F_drag / mass
    a_grav = -G * M_SOL / r ** 2
    return a_drag + a_grav


# ---------------------------------------------------------------------------
# Sistema ODE
# ---------------------------------------------------------------------------

def make_derivs(
    R_bubble: float = R_BUBBLE_DEFAULT,
    eta: float = ETA_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    v_sw: float = V_SW_DEFAULT,
    mass: float = MASS_PROBE_DEFAULT,
):
    """Devuelve la función ``derivs(state, t)`` compatible con ``odeint``.

    Parameters
    ----------
    R_bubble, eta, rho0, v_sw, mass : float, optional
        Parámetros del Plasma Magnet / sonda (ver ``accel_net``).

    Returns
    -------
    callable
        ``derivs(state, t) → [dr/dt, dv/dt]``
    """
    def derivs(state: list[float], t: float) -> list[float]:
        r, v = state
        a = accel_net(r, v, R_bubble=R_bubble, eta=eta, rho0=rho0,
                      v_sw=v_sw, mass=mass)
        return [v, a]
    return derivs


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------

def run_simulation(
    t_total_s: float,
    n_steps: int = 10_000,
    *,
    r0: float = R0_DEFAULT,
    v0: float = V0_DEFAULT,
    R_bubble: float = R_BUBBLE_DEFAULT,
    eta: float = ETA_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    v_sw: float = V_SW_DEFAULT,
    mass: float = MASS_PROBE_DEFAULT,
) -> dict:
    """Integra la trayectoria radial con ``scipy.integrate.odeint``.

    Parameters
    ----------
    t_total_s : float
        Duración de la simulación (segundos).
    n_steps : int, optional
        Número de puntos temporales (por defecto 10 000).
    r0 : float, optional
        Posición inicial (m).
    v0 : float, optional
        Velocidad inicial (m/s).
    R_bubble, eta, rho0, v_sw, mass : float, optional
        Parámetros del Plasma Magnet / sonda.

    Returns
    -------
    dict with keys:
        t              — array de tiempos (s)
        r              — posición heliocéntrica (m)
        v              — velocidad heliocéntrica (m/s)
        r_au           — posición en UA
        r_final_au     — posición final (UA)
        v_final_ms     — velocidad final (m/s)
        delta_v_kms    — ganancia total de velocidad (km/s)
    """
    t = np.linspace(0, t_total_s, n_steps)
    state0 = [r0, v0]
    derivs = make_derivs(R_bubble=R_bubble, eta=eta, rho0=rho0,
                         v_sw=v_sw, mass=mass)
    sol = odeint(derivs, state0, t)

    r_arr = sol[:, 0]
    v_arr = sol[:, 1]
    delta_v = v_arr[-1] - v0

    return {
        "t": t,
        "r": r_arr,
        "v": v_arr,
        "r_au": r_arr / AU,
        "r_final_au": float(r_arr[-1] / AU),
        "v_final_ms": float(v_arr[-1]),
        "delta_v_kms": float(delta_v / 1000.0),
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_path: str | Path | None = None) -> plt.Figure:
    """Genera la figura de tres paneles de la simulación ODE.

    Parameters
    ----------
    results : dict
        Diccionario devuelto por ``run_simulation``.
    out_path : str or Path, optional
        Ruta donde guardar la figura (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    dias = results["t"] / 86400

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(dias, (results["v"] - results["v"][0]) / 1000.0,
                 color='blue', linewidth=2)
    axes[0].set_title('Simulación ODE "Surfing" Heliosférico: Ganancia de Velocidad')
    axes[0].set_ylabel('Delta-V (km/s)')
    axes[0].grid(True)

    axes[1].plot(dias, results["r_au"], color='orange', linewidth=2)
    axes[1].set_title('Trayectoria: Distancia al Sol')
    axes[1].set_ylabel('Distancia (UA)')
    axes[1].grid(True)

    # Aceleración neta en cada instante
    a_arr = np.diff(results["v"]) / np.diff(results["t"])
    axes[2].plot(dias[:-1], a_arr * 1e6, color='red', linewidth=1.5)
    axes[2].set_title('Aceleración Neta (Plasma Magnet − Gravedad)')
    axes[2].set_xlabel('Tiempo (Días)')
    axes[2].set_ylabel('Aceleración (μm/s²)')
    axes[2].grid(True)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulación ODE de surfing heliosférico (Plasma Magnet / Motor de Velos)."
    )
    parser.add_argument(
        "--t-years", type=float, default=1.0, metavar="AÑOS",
        help="Duración de la simulación en años (por defecto: 1.0).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=10_000, metavar="N",
        help="Número de pasos de integración (por defecto: 10000).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Directorio de salida para figura (PNG) y resumen (TXT).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Omite la generación de la figura.",
    )
    parser.add_argument(
        "--mass", type=float, default=MASS_PROBE_DEFAULT, metavar="KG",
        help=f"Masa de la sonda en kg (por defecto: {MASS_PROBE_DEFAULT}).",
    )
    parser.add_argument(
        "--r-bubble", type=float, default=R_BUBBLE_DEFAULT, metavar="M",
        help=f"Radio de la burbuja magnética en m (por defecto: {R_BUBBLE_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Ejecuta la simulación ODE y devuelve el diccionario de resultados.

    Returns
    -------
    dict
        Resultados de la simulación (ver ``run_simulation``).
    """
    args = _parse_args(argv)

    t_total_s = args.t_years * 365 * 86400

    results = run_simulation(
        t_total_s=t_total_s,
        n_steps=args.n_steps,
        mass=args.mass,
        R_bubble=args.r_bubble,
    )

    r_final = results["r_final_au"]
    delta_v = results["delta_v_kms"]
    print(f"Distancia final: {r_final:.2f} UA, Delta-V: {delta_v:.1f} km/s")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_plot:
            fig_path = out_dir / "solar_wind_ode_sim.png"
            plot_results(results, out_path=fig_path)
            plt.close("all")
            print(f"Figura guardada en: {fig_path}")

        summary_lines = [
            "--- Resultados de la Simulación ODE ---",
            f"Distancia final al Sol: {r_final:.2f} UA",
            f"Velocidad final: {results['v_final_ms'] / 1000.0:.2f} km/s",
            f"Delta-V total ganado por 'surfing': {delta_v:.2f} km/s",
        ]
        summary_path = out_dir / "solar_wind_ode_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Resumen escrito en: {summary_path}")
    elif not args.no_plot:
        plot_results(results)
        plt.close("all")

    return results


if __name__ == "__main__":
    main()
