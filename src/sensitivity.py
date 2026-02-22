"""
sensitivity.py — Análisis de sensibilidad del ajuste SCM respecto a g0.

Permite explorar cómo varía el RMS de residuos log10 al barrer log10(g0)
dentro del rango de bounds, validando que el óptimo no está en el borde.

Uso como módulo:
    from src.sensitivity import sensitivity_sweep

Uso desde CLI:
    python -m src.sensitivity --data-dir data/SPARC --out results/sensitivity/
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from src.scm_models import BOUNDS_LOG10_G0, _cost_rms, fit_g0


# ---------------------------------------------------------------------------
# Barrido de sensibilidad
# ---------------------------------------------------------------------------

def sensitivity_sweep(
    g_obs: np.ndarray,
    g_bar: np.ndarray,
    bounds_log10_g0: tuple[float, float] = BOUNDS_LOG10_G0,
    n_points: int = 60,
) -> list[dict]:
    """Barre log10(g0) en el rango de bounds y evalúa el RMS en cada punto.

    Parameters
    ----------
    g_obs : np.ndarray
        Aceleración observada [m/s^2].
    g_bar : np.ndarray
        Aceleración bariónica [m/s^2].
    bounds_log10_g0 : tuple[float, float]
        Rango de búsqueda para log10(g0).
    n_points : int
        Número de puntos del barrido.

    Returns
    -------
    list[dict]
        Lista de dicts con ``log10_g0``, ``g0``, ``rms``.
    """
    lo, hi = bounds_log10_g0
    grid = np.linspace(lo, hi, n_points)
    rows = []
    for lg in grid:
        rms = _cost_rms(lg, g_obs, g_bar)
        rows.append({"log10_g0": float(lg), "g0": float(10.0 ** lg), "rms": float(rms)})
    return rows


def run_sensitivity(
    g_obs: np.ndarray,
    g_bar: np.ndarray,
    bounds_log10_g0: tuple[float, float] = BOUNDS_LOG10_G0,
    n_points: int = 60,
    verbose: bool = True,
) -> dict:
    """Ejecuta el barrido de sensibilidad y el ajuste óptimo.

    Parameters
    ----------
    g_obs : np.ndarray
    g_bar : np.ndarray
    bounds_log10_g0 : tuple[float, float]
    n_points : int
    verbose : bool

    Returns
    -------
    dict
        ``sweep`` (lista de dicts) y ``fit`` (resultado de :func:`fit_g0`).
    """
    sweep = sensitivity_sweep(g_obs, g_bar, bounds_log10_g0, n_points)
    fit_result = fit_g0(g_obs, g_bar, bounds_log10_g0)

    if verbose:
        print(
            f"[Sensitivity] optimal log10_g0={fit_result['log10_g0_hat']:.3f}  "
            f"g0_hat={fit_result['g0_hat']:.4e}  rms={fit_result['rms']:.4f}"
        )
        if fit_result["at_lower_bound"]:
            lo = bounds_log10_g0[0]
            print(
                f"  ⚠️  g0_hat está en el lower bound ({10.0**lo:.2e}). "
                "Considera ampliar bounds_log10_g0."
            )
        else:
            print("  ✔️  g0_hat está dentro de los bounds.")

    return {"sweep": sweep, "fit": fit_result}


def save_sensitivity(results: dict, out_dir: str | Path) -> None:
    """Guarda el barrido de sensibilidad en CSV.

    Parameters
    ----------
    results : dict
        Salida de :func:`run_sensitivity`.
    out_dir : str | Path
        Directorio de salida.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = out_dir / "sensitivity_sweep.csv"
    with open(sweep_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["log10_g0", "g0", "rms"])
        writer.writeheader()
        writer.writerows(results["sweep"])
    print(f"Barrido guardado en {sweep_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.scm_analysis import load_sparc_csv  # local import to avoid circular

    parser = argparse.ArgumentParser(
        description="Análisis de sensibilidad SCM sobre g0."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SPARC_DATA_DIR", "data/SPARC"),
    )
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--out", default="results/sensitivity/")
    parser.add_argument(
        "--bounds-log10-g0",
        nargs=2,
        type=float,
        default=list(BOUNDS_LOG10_G0),
        metavar=("LO", "HI"),
    )
    parser.add_argument("--n-points", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from src.scm_analysis import load_sparc_csv

    args = _parse_args(argv)
    data_file = args.data_file or str(Path(args.data_dir) / "sparc_rar.csv")
    g_bar, g_obs = load_sparc_csv(data_file)

    if g_bar.size == 0:
        raise RuntimeError(f"No se encontraron datos válidos en {data_file}.")

    bounds = tuple(args.bounds_log10_g0)
    results = run_sensitivity(g_obs, g_bar, bounds_log10_g0=bounds, n_points=args.n_points)
    save_sensitivity(results, args.out)


if __name__ == "__main__":
    main()
