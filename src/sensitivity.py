"""
sensitivity.py
--------------
Análisis de sensibilidad del Framework SCM para las curvas de rotación SPARC/Iorio.

Evalúa cómo varía el término universal SCM al perturbar los parámetros de entrada
(errores de velocidad, relación masa-luminosidad, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.read_iorio import read_batch, _delimiter_arg
from src.scm_models import scm_universal_term, v_baryon

logger = logging.getLogger(__name__)


def sensitivity_upsilon(
    df: pd.DataFrame,
    upsilon_range: tuple[float, float] = (0.1, 2.0),
    n_steps: int = 20,
) -> pd.DataFrame:
    """Variación del término universal SCM con la relación masa-luminosidad del disco.

    Parameters
    ----------
    df:            DataFrame de una galaxia (de read_iorio).
    upsilon_range: Rango (mín, máx) de Υ_disk a explorar.
    n_steps:       Número de puntos en el barrido.

    Returns
    -------
    pd.DataFrame  con columnas upsilon_disk, scm_u_term.
    """
    upsilons = np.linspace(upsilon_range[0], upsilon_range[1], n_steps)
    rows = []
    for ups in upsilons:
        vb = v_baryon(
            df["Vgas"].to_numpy(),
            df["Vdisk"].to_numpy(),
            df["Vbul"].to_numpy(),
            upsilon_disk=ups,
        )
        residuals = (df["Vobs"].to_numpy() - vb) / df["errV"].to_numpy()
        u = float(np.sqrt(np.mean(residuals**2)))
        rows.append({"upsilon_disk": ups, "scm_u_term": u})
    return pd.DataFrame(rows)


def sensitivity_errV(
    df: pd.DataFrame,
    scale_range: tuple[float, float] = (0.5, 2.0),
    n_steps: int = 20,
) -> pd.DataFrame:
    """Variación del término universal SCM al escalar los errores de velocidad.

    Parameters
    ----------
    df:           DataFrame de una galaxia.
    scale_range:  Rango de factores de escala sobre errV.
    n_steps:      Número de puntos en el barrido.

    Returns
    -------
    pd.DataFrame  con columnas errV_scale, scm_u_term.
    """
    scales = np.linspace(scale_range[0], scale_range[1], n_steps)
    base_u = scm_universal_term(df)
    rows = []
    for s in scales:
        df_s = df.copy()
        df_s["errV"] = df["errV"] * s
        u = scm_universal_term(df_s)
        rows.append({"errV_scale": s, "scm_u_term": u})
    return pd.DataFrame(rows)


def run_sensitivity(
    filepaths: list[Path],
    delimiter: str = "auto",
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Ejecuta el análisis de sensibilidad para una lista de galaxias.

    Returns
    -------
    dict[galaxy_name, DataFrame_sensitivity_upsilon]
    """
    galaxies = read_batch(filepaths, delimiter=delimiter)
    results: dict[str, pd.DataFrame] = {}

    for name, df in galaxies.items():
        sens = sensitivity_upsilon(df)
        results[name] = sens
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}_sensitivity.csv"
            sens.to_csv(out_path, index=False)
            logger.info("Sensibilidad guardada en %s", out_path)

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.sensitivity",
        description="Análisis de sensibilidad SCM para curvas de rotación SPARC/Iorio.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--batch",    metavar="LIST_FILE")
    input_group.add_argument("--data-dir", metavar="DIR")
    parser.add_argument("--out",       metavar="DIR", default="results/sensitivity/")
    parser.add_argument("--delimiter", default="auto")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    delimiter = _delimiter_arg(args.delimiter)
    if args.batch:
        bp = Path(args.batch)
        if not bp.exists():
            sys.exit(f"Fichero de lista no encontrado: {bp}")
        filepaths = [
            Path(line.strip())
            for line in bp.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        data_dir = Path(args.data_dir)
        filepaths = sorted(data_dir.glob("*.txt"))

    run_sensitivity(filepaths, delimiter=delimiter, out_dir=Path(args.out))
    logger.info("Análisis de sensibilidad completado.")


if __name__ == "__main__":
    main()
