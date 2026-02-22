"""
scm_analysis.py — Pipeline principal de análisis SCM sobre el dataset SPARC.

Ejecuta el ajuste de g0 con la Relación de Aceleración Radial (RAR) y genera
diagnósticos de residuos por bins.

Uso como módulo:
    from src.scm_analysis import run_analysis

Uso desde CLI:
    python -m src.scm_analysis --data-dir data/SPARC --out results/
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from src.scm_models import (
    BOUNDS_LOG10_G0,
    bin_residuals,
    fit_g0,
    print_fit_summary,
    quantiles_g_bar,
)


# ---------------------------------------------------------------------------
# Carga de datos SPARC
# ---------------------------------------------------------------------------

def load_sparc_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga un CSV de SPARC con columnas g_bar y g_obs (en m/s^2).

    Parameters
    ----------
    path : str | Path
        Ruta al archivo CSV.  Debe tener cabecera con columnas
        ``g_bar`` y ``g_obs``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (g_bar, g_obs) filtrados para valores positivos.
    """
    path = Path(path)
    g_bar_list, g_obs_list = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                gb = float(row["g_bar"])
                go = float(row["g_obs"])
            except (KeyError, ValueError):
                continue
            if gb > 0 and go > 0:
                g_bar_list.append(gb)
                g_obs_list.append(go)
    return np.array(g_bar_list), np.array(g_obs_list)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_analysis(
    g_obs: np.ndarray,
    g_bar: np.ndarray,
    bounds_log10_g0: tuple[float, float] = BOUNDS_LOG10_G0,
    n_bins: int = 10,
    verbose: bool = True,
) -> dict:
    """Ejecuta el ajuste de g0 y devuelve resultados y diagnósticos.

    Parameters
    ----------
    g_obs : np.ndarray
        Aceleración observada [m/s^2].
    g_bar : np.ndarray
        Aceleración bariónica [m/s^2].
    bounds_log10_g0 : tuple[float, float]
        Límites para log10(g0).  Por defecto BOUNDS_LOG10_G0 = (-16.0, -8.0).
    n_bins : int
        Número de bins para diagnóstico de residuos.
    verbose : bool
        Si True, imprime resumen en stdout.

    Returns
    -------
    dict
        Contiene ``fit`` (resultado de :func:`fit_g0`) y
        ``bins`` (lista de dicts de :func:`bin_residuals`).
    """
    fit_result = fit_g0(g_obs, g_bar, bounds_log10_g0=bounds_log10_g0)
    bins = bin_residuals(g_obs, g_bar, fit_result["g0_hat"], n_bins=n_bins)

    if verbose:
        print_fit_summary(fit_result, bins, g_bar)

    return {"fit": fit_result, "bins": bins}


def save_results(results: dict, out_dir: str | Path) -> None:
    """Guarda resultados de análisis en ``out_dir``.

    Parameters
    ----------
    results : dict
        Salida de :func:`run_analysis`.
    out_dir : str | Path
        Directorio de salida.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = results["fit"]
    summary_path = out_dir / "executive_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"g0_hat={fit['g0_hat']:.6e}\n")
        fh.write(f"log10_g0_hat={fit['log10_g0_hat']:.4f}\n")
        fh.write(f"rms={fit['rms']:.6f}\n")
        fh.write(f"lower_bound={fit['lower_bound']:.6e}\n")
        fh.write(f"upper_bound={fit['upper_bound']:.6e}\n")
        fh.write(f"at_lower_bound={fit['at_lower_bound']}\n")

    bins_path = out_dir / "bin_residuals.csv"
    with open(bins_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["g_bar_center", "median_residual", "n_points"])
        writer.writeheader()
        writer.writerows(results["bins"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis SCM RAR sobre dataset SPARC."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SPARC_DATA_DIR", "data/SPARC"),
        help="Directorio raíz de SPARC (por defecto: data/SPARC o $SPARC_DATA_DIR).",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Archivo CSV de datos (por defecto: <data-dir>/sparc_rar.csv).",
    )
    parser.add_argument(
        "--out",
        default="results/",
        help="Directorio de salida (por defecto: results/).",
    )
    parser.add_argument(
        "--bounds-log10-g0",
        nargs=2,
        type=float,
        default=list(BOUNDS_LOG10_G0),
        metavar=("LO", "HI"),
        help=f"Límites log10(g0) (por defecto: {BOUNDS_LOG10_G0}).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Número de bins para diagnóstico (por defecto: 10).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    data_file = args.data_file or str(Path(args.data_dir) / "sparc_rar.csv")
    g_bar, g_obs = load_sparc_csv(data_file)

    if g_bar.size == 0:
        raise RuntimeError(f"No se encontraron datos válidos en {data_file}.")

    bounds = tuple(args.bounds_log10_g0)
    results = run_analysis(g_obs, g_bar, bounds_log10_g0=bounds, n_bins=args.n_bins)
    save_results(results, args.out)
    print(f"Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
