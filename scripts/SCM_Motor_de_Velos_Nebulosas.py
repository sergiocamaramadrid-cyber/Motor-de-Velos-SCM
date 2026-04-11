#!/usr/bin/env python3
"""
SIMULACIÓN SCM - MOTOR DE VELOS
Nebulosas: NGC 7635 (Burbuja) y M16 (Pilares de la Creación)

Autor: Framework SCM-Motor de Velos (repositorio público)
Basado en parámetros proporcionados por Sergio Cámara Madrid.
Cumple con el Protocolo de Veracidad y Rigor (PVR-AI):
- Resultados simulados, no observacionales.
- Reproducible y verificable.

Uso::

    python scripts/SCM_Motor_de_Velos_Nebulosas.py

Archivos generados:
    results/nebulosas/resultados_burbuja_NGC7635.csv
    results/nebulosas/resultados_pilares_M16.csv
    results/nebulosas/simulacion_nebulosas_SCM.png
    results/nebulosas/animacion_SCM_burbuja.gif  (requiere pillow)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no-display safe
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import spearmanr, linregress

# ---------------------------------------------------------------------------
# Parámetros de simulación
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
LOG_MBAR_BURBUJA = 10.1        # masa estelar crítica (log M☉)
SPEARMAN_RHO_ESPERADO = -0.45  # correlación esperada densidad–F3
N_PUNTOS_M16 = 54              # puntos de muestreo para M16
UMBRAL_PUNTAS = 0.9            # percentil 90 → "puntas de pilar"
N_PERMUTACIONES = 1000         # permutaciones para p-valor

OUTPUT_DIR_DEFAULT = Path("results/nebulosas")

# ---------------------------------------------------------------------------
# Simulación: NGC 7635 – Nebulosa de la Burbuja
# ---------------------------------------------------------------------------


def simular_burbuja(rng: np.random.Generator | None = None) -> dict:
    """Simula la fricción asimétrica del Velo en NGC 7635.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Generador de números aleatorios (reproducibilidad).

    Returns
    -------
    dict con claves: theta, densidad, F3, rho, p_val
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    theta = np.linspace(0, 2 * np.pi, 200)
    densidad = 1 + 0.8 * np.cos(theta - 0.7)
    F3 = -0.5 * densidad + rng.normal(0, 0.05, size=len(theta))
    rho, p_val = spearmanr(densidad, F3)

    return {"theta": theta, "densidad": densidad, "F3": F3,
            "rho": float(rho), "p_val": float(p_val)}


# ---------------------------------------------------------------------------
# Simulación: M16 – Pilares de la Creación
# ---------------------------------------------------------------------------


def simular_pilares(rng: np.random.Generator | None = None) -> dict:
    """Simula el flujo del Velo entre los pilares de M16 con acumulación de energía.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Generador de números aleatorios (reproducibilidad).

    Returns
    -------
    dict con claves: x, densidad, acumulacion, F3, delta_AIC, p_perm, slope
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    x = np.linspace(0, 10, N_PUNTOS_M16)

    # Tres pilares gaussianos con ruido
    densidad = (
        1.0
        + 2.5 * np.exp(-((x - 2) / 0.5) ** 2)
        + 1.8 * np.exp(-((x - 5) / 0.6) ** 2)
        + 2.0 * np.exp(-((x - 8) / 0.7) ** 2)
    )
    densidad += rng.normal(0, 0.2, size=len(x))

    grad = np.gradient(densidad, x)
    acumulacion = densidad * np.abs(grad)

    F3 = -0.8 * acumulacion + rng.normal(0, 0.05, size=len(x))

    # Modelo simple (expansión uniforme, 1 parámetro)
    modelo_simple = np.full(len(x), np.mean(F3))
    rss_simple = float(np.sum((F3 - modelo_simple) ** 2))

    # Modelo con acumulación (regresión lineal, 2 parámetros)
    slope, intercept, _r, _p, _se = linregress(acumulacion, F3)
    rss_acum = float(np.sum((F3 - (intercept + slope * acumulacion)) ** 2))

    n = len(x)
    aic_simple = n * np.log(rss_simple / n) + 2 * 1
    aic_acum = n * np.log(rss_acum / n) + 2 * 2
    delta_AIC = float(aic_simple - aic_acum)

    # p-valor por permutación en las puntas (percentil 90)
    umbral_val = float(np.percentile(acumulacion, UMBRAL_PUNTAS * 100))
    puntas = acumulacion > umbral_val
    rho_puntas, _ = spearmanr(acumulacion[puntas], F3[puntas])
    perm_rhos = [
        spearmanr(acumulacion[puntas], rng.permutation(F3[puntas]))[0]
        for _ in range(N_PERMUTACIONES)
    ]
    p_perm = float(np.mean(np.abs(perm_rhos) >= np.abs(rho_puntas)))

    return {
        "x": x,
        "densidad": densidad,
        "acumulacion": acumulacion,
        "F3": F3,
        "delta_AIC": delta_AIC,
        "p_perm": p_perm,
        "slope": float(slope),
    }


# ---------------------------------------------------------------------------
# Guardar CSV
# ---------------------------------------------------------------------------


def guardar_csv_burbuja(resultado: dict, out_dir: Path) -> Path:
    """Guarda los datos de NGC 7635 en CSV.

    Parameters
    ----------
    resultado : dict
        Salida de :func:`simular_burbuja`.
    out_dir : Path
        Directorio de salida.

    Returns
    -------
    Path al archivo CSV generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resultados_burbuja_NGC7635.csv"

    df = pd.DataFrame({
        "theta_rad": resultado["theta"],
        "densidad_entorno": resultado["densidad"],
        "F3_pendiente_externa": resultado["F3"],
    })
    # Metadatos en las dos primeras filas de comentario
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# rho_Spearman={resultado['rho']:.6f}, p_valor={resultado['p_val']:.6e}\n")
        df.to_csv(fh, index=False)

    print(f"✅ CSV guardado: {out_path}")
    return out_path


def guardar_csv_pilares(resultado: dict, out_dir: Path) -> Path:
    """Guarda los datos de M16 en CSV.

    Parameters
    ----------
    resultado : dict
        Salida de :func:`simular_pilares`.
    out_dir : Path
        Directorio de salida.

    Returns
    -------
    Path al archivo CSV generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resultados_pilares_M16.csv"

    df = pd.DataFrame({
        "posicion": resultado["x"],
        "densidad_barionica": resultado["densidad"],
        "acumulacion_energia": resultado["acumulacion"],
        "F3_pendiente_externa": resultado["F3"],
    })
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# delta_AIC={resultado['delta_AIC']:.4f},"
            f" p_perm_puntas={resultado['p_perm']:.6f},"
            f" slope_acum_F3={resultado['slope']:.6f}\n"
        )
        df.to_csv(fh, index=False)

    print(f"✅ CSV guardado: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Animación GIF: barrido angular en NGC 7635
# ---------------------------------------------------------------------------


def crear_animacion_burbuja(resultado: dict, out_dir: Path) -> Path:
    """Crea un GIF animado del flujo del Velo a lo largo del ángulo en NGC 7635.

    Requiere ``pillow`` (``pip install pillow``).

    Parameters
    ----------
    resultado : dict
        Salida de :func:`simular_burbuja`.
    out_dir : Path
        Directorio de salida.

    Returns
    -------
    Path al archivo GIF generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "animacion_SCM_burbuja.gif"

    theta = resultado["theta"]
    densidad = resultado["densidad"]
    F3 = resultado["F3"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("Ángulo (rad)")
    ax.set_ylabel("Magnitud")
    ax.set_title("Simulación SCM: Flujo del Velo en NGC 7635 (Burbuja)")

    (line_dens,) = ax.plot([], [], "b-", lw=2, label="Densidad entorno")
    (line_F3,) = ax.plot([], [], "r-", lw=2, label="F3 (pendiente externa)")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.legend()

    angle_text = ax.text(
        0.05, 0.95, "", transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    def _init():
        line_dens.set_data([], [])
        line_F3.set_data([], [])
        angle_text.set_text("")
        return line_dens, line_F3, angle_text

    def _update(frame):
        line_dens.set_data(theta[: frame + 1], densidad[: frame + 1])
        line_F3.set_data(theta[: frame + 1], F3[: frame + 1])
        angle_text.set_text(f"Ángulo = {theta[frame]:.2f} rad")
        return line_dens, line_F3, angle_text

    ani = animation.FuncAnimation(
        fig, _update, frames=len(theta),
        init_func=_init, blit=True, interval=50, repeat=True,
    )
    ani.save(str(out_path), writer="pillow", fps=20)
    plt.close(fig)
    print(f"✅ Animación guardada: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figura estática con 4 paneles
# ---------------------------------------------------------------------------


def generar_figura_estatica(
    res_burbuja: dict,
    res_pilares: dict,
    out_dir: Path,
) -> Path:
    """Genera la figura estática de 4 paneles (PNG).

    Parameters
    ----------
    res_burbuja : dict
        Salida de :func:`simular_burbuja`.
    res_pilares : dict
        Salida de :func:`simular_pilares`.
    out_dir : Path
        Directorio de salida.

    Returns
    -------
    Path al archivo PNG generado.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulacion_nebulosas_SCM.png"

    theta = res_burbuja["theta"]
    dens_b = res_burbuja["densidad"]
    F3_b = res_burbuja["F3"]
    rho = res_burbuja["rho"]
    p_val = res_burbuja["p_val"]

    x = res_pilares["x"]
    dens_m16 = res_pilares["densidad"]
    acum = res_pilares["acumulacion"]
    F3_m16 = res_pilares["F3"]
    dAIC = res_pilares["delta_AIC"]
    p_perm = res_pilares["p_perm"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Simulación SCM-Motor de Velos en Nebulosas", fontsize=14)

    # Panel 1: NGC 7635 perfil angular
    ax1 = axes[0, 0]
    ax1.plot(theta, dens_b, "b-", label="Densidad entorno")
    ax1.set_xlabel("Ángulo (rad)")
    ax1.set_ylabel("Densidad relativa", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(theta, F3_b, "r-", label="F3")
    ax2.set_ylabel("F3", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title(f"NGC 7635: ρ = {rho:.3f}, p = {p_val:.4f}")

    # Panel 2: correlación dispersión NGC 7635
    ax3 = axes[0, 1]
    ax3.scatter(dens_b, F3_b, alpha=0.6)
    ax3.set_xlabel("Densidad entorno")
    ax3.set_ylabel("F3")
    ax3.set_title("Correlación local NGC 7635")
    z = np.polyfit(dens_b, F3_b, 1)
    p_tend = np.poly1d(z)
    x_sorted = np.sort(dens_b)
    ax3.plot(x_sorted, p_tend(x_sorted), "k--", label=f"pendiente = {z[0]:.2f}")
    ax3.legend()

    # Panel 3: M16 perfiles
    ax4 = axes[1, 0]
    ax4.plot(x, dens_m16, "g-", label="Densidad bariónica")
    ax4.plot(x, acum, color="orange", label="Acumulación energía")
    ax4.set_xlabel("Posición")
    ax4.set_ylabel("Magnitud")
    ax4.set_title(f"M16: ΔAIC = {dAIC:.2f}")
    ax4.legend()

    # Panel 4: F3 vs acumulación con puntas destacadas
    ax5 = axes[1, 1]
    umbral_val = float(np.percentile(acum, UMBRAL_PUNTAS * 100))
    puntas = acum > umbral_val
    ax5.scatter(
        acum[~puntas], F3_m16[~puntas],
        c="gray", alpha=0.5, label="Regiones normales",
    )
    ax5.scatter(
        acum[puntas], F3_m16[puntas],
        c="red", edgecolors="k", label="Puntas de pilares",
    )
    ax5.set_xlabel("Acumulación de energía")
    ax5.set_ylabel("F3")
    ax5.set_title(f"Puntas: p-perm = {p_perm:.4f} (esperado <0.001)")
    ax5.legend()

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"✅ Figura estática guardada: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Punto de entrada principal.

    Parameters
    ----------
    argv : list[str], optional
        Argumentos CLI (None → ``sys.argv``).

    Returns
    -------
    dict con claves: burbuja, pilares, csv_burbuja, csv_pilares,
    figura, animacion (o None si falla).
    """
    parser = argparse.ArgumentParser(
        description="Simulación SCM-Motor de Velos en nebulosas NGC 7635 y M16."
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUTPUT_DIR_DEFAULT),
        help=f"Directorio de salida (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Semilla aleatoria (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Omitir la generación del GIF animado.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    print("=== SIMULACIÓN SCM - MOTOR DE VELOS ===")

    res_burbuja = simular_burbuja(rng=rng)
    print(
        f"🌌 NGC 7635: ρ = {res_burbuja['rho']:.3f}"
        f" (p={res_burbuja['p_val']:.4f}) | Esperado ≈ {SPEARMAN_RHO_ESPERADO}"
    )

    res_pilares = simular_pilares(rng=rng)
    print(
        f"🌌 M16: ΔAIC = {res_pilares['delta_AIC']:.2f}"
        f" | p-perm puntas = {res_pilares['p_perm']:.4f}"
    )

    csv_b = guardar_csv_burbuja(res_burbuja, out_dir)
    csv_p = guardar_csv_pilares(res_pilares, out_dir)
    fig_path = generar_figura_estatica(res_burbuja, res_pilares, out_dir)

    anim_path = None
    if not args.no_animation:
        try:
            anim_path = crear_animacion_burbuja(res_burbuja, out_dir)
        except Exception as exc:
            print(
                f"⚠️  No se pudo generar la animación: {exc}. "
                "Asegúrate de tener pillow instalado (pip install pillow)."
            )

    print("\n✅ Simulación completada. Archivos generados:")
    for path in [csv_b, csv_p, fig_path, anim_path]:
        if path is not None:
            print(f"   - {path}")

    return {
        "burbuja": res_burbuja,
        "pilares": res_pilares,
        "csv_burbuja": csv_b,
        "csv_pilares": csv_p,
        "figura": fig_path,
        "animacion": anim_path,
    }


if __name__ == "__main__":
    main()
