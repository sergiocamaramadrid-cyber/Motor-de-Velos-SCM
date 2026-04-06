"""
scripts/heliospheric_surfing_sim.py — Simulación de "Surfing" Heliosférico con Plasma Magnet.

Implementa el framework SCM-Motor de Velos para modelar la propulsión de una
sonda espacial mediante la interacción con el viento solar (Plasma Magnet /
Motor de Velos electromagnético).

La fuerza de empuje se calcula como la fuerza de arrastre que ejerce el flujo
de momentum del viento solar sobre la burbuja magnética inflada por la sonda.

Usage
-----
Simulación con parámetros por defecto (1 año, salida en resultados/)::

    python scripts/heliospheric_surfing_sim.py

Con directorio de salida y duración personalizada::

    python scripts/heliospheric_surfing_sim.py \\
        --out results/heliospheric \\
        --t-years 2.0

Sólo resultados numéricos (sin figura)::

    python scripts/heliospheric_surfing_sim.py --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------

MU_0 = 4 * np.pi * 1e-7   # Permeabilidad del vacío (H/m)
AU = 1.496e11              # Unidad Astronómica (m)

# ---------------------------------------------------------------------------
# Parámetros del medio heliosférico (por defecto)
# ---------------------------------------------------------------------------

V_SW_DEFAULT = 400_000.0        # Velocidad media del viento solar (m/s)
RHO_EARTH_DEFAULT = 8e-20       # Densidad del viento solar en 1 UA (kg/m³)

# ---------------------------------------------------------------------------
# Parámetros de la sonda (por defecto)
# ---------------------------------------------------------------------------

MASS_PROBE_DEFAULT = 50.0       # Masa de la sonda (kg)
R_BUBBLE_DEFAULT = 50_000.0     # Radio efectivo de la burbuja magnética (m)
EFICIENCIA_DEFAULT = 0.5        # Eficiencia de transferencia de momentum

# ---------------------------------------------------------------------------
# Condiciones iniciales (por defecto)
# ---------------------------------------------------------------------------

DISTANCIA_INICIAL_AU = 1.0      # Posición inicial (UA)
VELOCIDAD_INICIAL_MS = 30_000.0 # Velocidad heliocéntrica inicial (m/s)


# ---------------------------------------------------------------------------
# Física del motor de velas
# ---------------------------------------------------------------------------

def calcular_fuerza_surfing(
    distancia_sol_au: float,
    velocidad_sonda: float,
    *,
    v_sw: float = V_SW_DEFAULT,
    rho_earth: float = RHO_EARTH_DEFAULT,
    r_bubble: float = R_BUBBLE_DEFAULT,
    eficiencia: float = EFICIENCIA_DEFAULT,
) -> float:
    """Calcula el empuje generado por el Plasma Magnet (Motor de Velos).

    El empuje F es proporcional al flujo de momentum del viento solar
    (ρ · v_rel²) y al área de la burbuja magnética (π · R²).

    Parameters
    ----------
    distancia_sol_au : float
        Distancia heliocéntrica de la sonda en Unidades Astronómicas (UA).
    velocidad_sonda : float
        Velocidad radial heliocéntrica de la sonda (m/s).
    v_sw : float, optional
        Velocidad del viento solar (m/s).  Por defecto ``V_SW_DEFAULT``.
    rho_earth : float, optional
        Densidad de masa del viento solar a 1 UA (kg/m³).
        Por defecto ``RHO_EARTH_DEFAULT``.
    r_bubble : float, optional
        Radio efectivo de la burbuja magnética (m).
        Por defecto ``R_BUBBLE_DEFAULT``.
    eficiencia : float, optional
        Factor de eficiencia de la interacción (0–1).
        Por defecto ``EFICIENCIA_DEFAULT``.

    Returns
    -------
    float
        Fuerza de empuje en Newtons.  Devuelve 0.0 si la sonda supera la
        velocidad del viento solar (sin empuje neto).
    """
    # 1. Adaptación al medio: densidad ∝ 1/r²
    rho_local = rho_earth * (1.0 / distancia_sol_au) ** 2

    # 2. Velocidad relativa entre viento solar y sonda
    v_rel = v_sw - velocidad_sonda
    if v_rel < 0:
        # Sonda más rápida que el viento solar: sin empuje neto
        return 0.0

    # 3. Área de interacción (vela electromagnética)
    area_burbuja = np.pi * r_bubble ** 2

    # 4. Fuerza de arrastre: F = eficiencia · ½ · ρ · A · v_rel²
    fuerza = eficiencia * 0.5 * rho_local * area_burbuja * v_rel ** 2
    return float(fuerza)


# ---------------------------------------------------------------------------
# Simulación numérica
# ---------------------------------------------------------------------------

def run_simulation(
    t_total_s: float,
    dt_s: float,
    *,
    distancia_inicial_au: float = DISTANCIA_INICIAL_AU,
    velocidad_inicial_ms: float = VELOCIDAD_INICIAL_MS,
    mass_probe: float = MASS_PROBE_DEFAULT,
    v_sw: float = V_SW_DEFAULT,
    rho_earth: float = RHO_EARTH_DEFAULT,
    r_bubble: float = R_BUBBLE_DEFAULT,
    eficiencia: float = EFICIENCIA_DEFAULT,
) -> dict:
    """Integra la trayectoria de la sonda mediante el método de Euler.

    Parameters
    ----------
    t_total_s : float
        Duración total de la simulación (segundos).
    dt_s : float
        Paso de tiempo (segundos).
    distancia_inicial_au : float, optional
        Posición heliocéntrica inicial (UA).
    velocidad_inicial_ms : float, optional
        Velocidad heliocéntrica inicial (m/s).
    mass_probe : float, optional
        Masa de la sonda (kg).
    v_sw, rho_earth, r_bubble, eficiencia : float, optional
        Parámetros del motor de velas (ver ``calcular_fuerza_surfing``).

    Returns
    -------
    dict with keys:
        tiempos        — array de instantes (s)
        distancia_au   — distancia heliocéntrica (UA)
        velocidad      — velocidad heliocéntrica (m/s)
        empuje         — fuerza de empuje (N)
        delta_v_kms    — ganancia de velocidad total (km/s)
        distancia_final_au — distancia final (UA)
        velocidad_final_kms — velocidad final (km/s)
    """
    tiempos = np.arange(0.0, t_total_s, dt_s)
    n = len(tiempos)

    distancia_au = np.zeros(n)
    velocidad = np.zeros(n)
    empuje = np.zeros(n)

    distancia_au[0] = distancia_inicial_au
    velocidad[0] = velocidad_inicial_ms

    for i in range(1, n):
        f_surf = calcular_fuerza_surfing(
            distancia_au[i - 1],
            velocidad[i - 1],
            v_sw=v_sw,
            rho_earth=rho_earth,
            r_bubble=r_bubble,
            eficiencia=eficiencia,
        )
        empuje[i - 1] = f_surf
        a_surf = f_surf / mass_probe

        velocidad[i] = velocidad[i - 1] + a_surf * dt_s
        distancia_au[i] = distancia_au[i - 1] + (velocidad[i - 1] * dt_s) / AU

    return {
        "tiempos": tiempos,
        "distancia_au": distancia_au,
        "velocidad": velocidad,
        "empuje": empuje,
        "delta_v_kms": (velocidad[-1] - velocidad[0]) / 1000.0,
        "distancia_final_au": float(distancia_au[-1]),
        "velocidad_final_kms": float(velocidad[-1]) / 1000.0,
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_path: str | Path | None = None) -> plt.Figure:
    """Genera la figura de tres paneles de la simulación.

    Parameters
    ----------
    results : dict
        Diccionario devuelto por ``run_simulation``.
    out_path : str or Path, optional
        Si se proporciona, guarda la figura en esta ruta (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    tiempos = results["tiempos"]
    distancia_au = results["distancia_au"]
    velocidad = results["velocidad"]
    empuje = results["empuje"]
    dias = tiempos / (3600 * 24)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(dias, (velocidad - velocidad[0]) / 1000.0, color='blue', linewidth=2)
    axes[0].set_title('Simulación "Surfing" Heliosférico: Ganancia de Velocidad (Delta-V)')
    axes[0].set_ylabel('Delta-V (km/s)')
    axes[0].grid(True)

    axes[1].plot(dias, distancia_au, color='orange', linewidth=2)
    axes[1].set_title('Trayectoria: Distancia al Sol')
    axes[1].set_ylabel('Distancia (UA)')
    axes[1].grid(True)

    axes[2].plot(dias, empuje, color='red', linewidth=2)
    axes[2].set_title('Fuerza de Empuje Generada por el Framework SCM')
    axes[2].set_xlabel('Tiempo (Días)')
    axes[2].set_ylabel('Empuje (Newtons)')
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
        description="Simulación de surfing heliosférico con Plasma Magnet (Motor de Velos SCM)."
    )
    parser.add_argument(
        "--t-years", type=float, default=1.0, metavar="AÑOS",
        help="Duración de la simulación en años (por defecto: 1.0).",
    )
    parser.add_argument(
        "--dt-days", type=float, default=1.0, metavar="DÍAS",
        help="Paso de tiempo en días (por defecto: 1.0).",
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
    """Ejecuta la simulación y devuelve el diccionario de resultados.

    Returns
    -------
    dict
        Resultados de la simulación (ver ``run_simulation``).
    """
    args = _parse_args(argv)

    t_total_s = args.t_years * 365 * 24 * 3600
    dt_s = args.dt_days * 24 * 3600

    results = run_simulation(
        t_total_s=t_total_s,
        dt_s=dt_s,
        mass_probe=args.mass,
        r_bubble=args.r_bubble,
    )

    # Resumen de resultados
    print("--- Resultados de la Simulación ---")
    print(f"Distancia final al Sol: {results['distancia_final_au']:.2f} UA")
    print(f"Velocidad heliocéntrica final: {results['velocidad_final_kms']:.2f} km/s")
    print(f"Delta-V total ganado por 'surfing': {results['delta_v_kms']:.2f} km/s")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_plot:
            fig_path = out_dir / "heliospheric_surfing.png"
            plot_results(results, out_path=fig_path)
            plt.close("all")
            print(f"Figura guardada en: {fig_path}")

        summary_lines = [
            "--- Resultados de la Simulación ---",
            f"Distancia final al Sol: {results['distancia_final_au']:.2f} UA",
            f"Velocidad heliocéntrica final: {results['velocidad_final_kms']:.2f} km/s",
            f"Delta-V total ganado por 'surfing': {results['delta_v_kms']:.2f} km/s",
        ]
        (out_dir / "heliospheric_surfing_summary.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )
        print(f"Resumen escrito en: {out_dir / 'heliospheric_surfing_summary.txt'}")
    elif not args.no_plot:
        plot_results(results)
        plt.close("all")

    return results


if __name__ == "__main__":
    main()
