"""
scm_analysis.py
---------------
Pipeline de análisis SCM para el catálogo SPARC/Iorio.

Uso típico (CLI)
----------------
Procesar 17 galaxias listadas en un fichero::

    python -m src.scm_analysis --batch galaxies.txt --out results/

Procesar mediante patrón glob::

    python -m src.scm_analysis --glob "data/SPARC/Rotmod_LTG/*.txt" --out results/

El comando final genera:
  results/universal_term_comparison_full.csv
  results/executive_summary.txt
  results/top10_universal.tex  (si hay ≥ 10 galaxias)
"""

from __future__ import annotations

import argparse
import glob as _glob
import logging
import sys
from pathlib import Path

import pandas as pd

from src.read_iorio import read_batch, _delimiter_arg
from src.scm_models import fit_piso, scm_universal_term

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def analyse_galaxy(name: str, df: pd.DataFrame) -> dict:
    """Run the full SCM analysis for one galaxy.

    Returns a summary dict suitable for a DataFrame row.
    """
    piso = fit_piso(df)
    u_term = scm_universal_term(df)
    has_sigma = "sigma_V" in df.columns
    return {
        "galaxy":      name,
        "n_points":    len(df),
        "R_max_kpc":   float(df["R"].max()),
        "Vobs_max":    float(df["Vobs"].max()),
        "has_sigma_V": has_sigma,
        "v_inf_kms":   piso["v_inf"],
        "r_c_kpc":     piso["r_c"],
        "chi2_red":    piso["chi2_red"],
        "fit_ok":      piso["success"],
        "scm_u_term":  u_term,
        "veredicto":   "OK" if piso["success"] else "FALLO",
    }


def run_pipeline(
    filepaths: list[Path],
    delimiter: str = "auto",
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Read all galaxy files and run the SCM analysis pipeline.

    Parameters
    ----------
    filepaths:  List of data-file paths.
    delimiter:  Passed to read_batch.
    out_dir:    If given, per-galaxy CSVs and the master results are written here.

    Returns
    -------
    pd.DataFrame  One row per galaxy with analysis results.
    """
    galaxies = read_batch(filepaths, delimiter=delimiter)
    if not galaxies:
        logger.error("No galaxy data loaded. Aborting.")
        sys.exit(1)

    rows = [analyse_galaxy(name, df) for name, df in galaxies.items()]
    results = pd.DataFrame(rows)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "universal_term_comparison_full.csv"
        results.to_csv(csv_path, index=False)
        logger.info("Results saved to %s", csv_path)

        _write_executive_summary(results, out_dir)

        if len(results) >= 10:
            _write_top10_latex(results, out_dir)

    return results


def _write_executive_summary(results: pd.DataFrame, out_dir: Path) -> None:
    n_total = len(results)
    n_ok    = (results["veredicto"] == "OK").sum()
    n_fail  = n_total - n_ok

    summary_lines = [
        "=== Executive Summary – SCM Framework ===",
        f"Galaxias procesadas : {n_total}",
        f"Ajustes exitosos    : {n_ok}",
        f"Ajustes fallidos    : {n_fail}",
        "",
        "Estadísticas del término universal SCM:",
        f"  Media   : {results['scm_u_term'].mean():.4f}",
        f"  Mediana : {results['scm_u_term'].median():.4f}",
        f"  Máximo  : {results['scm_u_term'].max():.4f}  ({results.loc[results['scm_u_term'].idxmax(), 'galaxy']})",
        f"  Mínimo  : {results['scm_u_term'].min():.4f}  ({results.loc[results['scm_u_term'].idxmin(), 'galaxy']})",
        "",
        "Galaxias con dispersión de velocidades (sigma_V): "
        + str(results["has_sigma_V"].sum()),
        "",
        f"Veredicto global: {'ÉXITO' if n_fail == 0 else 'PARCIAL – revisar galaxias fallidas'}",
    ]

    path = out_dir / "executive_summary.txt"
    path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info("Executive summary written to %s", path)


def _write_top10_latex(results: pd.DataFrame, out_dir: Path) -> None:
    top10 = results.nlargest(10, "scm_u_term")[["galaxy", "scm_u_term", "chi2_red", "veredicto"]]
    path = out_dir / "top10_universal.tex"
    try:
        top10.to_latex(path, index=False, float_format="%.4f")
    except ImportError:
        # Fallback: write a minimal LaTeX table without jinja2
        lines = [
            r"\begin{tabular}{llll}",
            r"\hline",
            "galaxy & scm\\_u\\_term & chi2\\_red & veredicto \\\\",
            r"\hline",
        ]
        for _, row in top10.iterrows():
            chi2 = f"{row['chi2_red']:.4f}" if pd.notna(row['chi2_red']) else "nan"
            lines.append(
                f"{row['galaxy']} & {row['scm_u_term']:.4f} & {chi2} & {row['veredicto']} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Top-10 LaTeX table written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.scm_analysis",
        description="Pipeline SCM para análisis de curvas de rotación SPARC/Iorio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Procesar 17 galaxias desde un fichero de lista:\n"
            "  python -m src.scm_analysis --batch galaxies.txt --out results/\n\n"
            "  # Procesar mediante glob:\n"
            "  python -m src.scm_analysis --glob 'data/SPARC/Rotmod_LTG/*.txt' --out results/\n"
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--batch",        metavar="LIST_FILE",
                             help="Fichero con una ruta de galaxia por línea.")
    input_group.add_argument("--glob",         metavar="PATTERN", dest="glob_pattern",
                             help='Patrón glob, p. ej. "data/SPARC/Rotmod_LTG/*.txt".')
    input_group.add_argument("--data-dir",     metavar="DIR",
                             help="Directorio con ficheros *.txt de galaxias.")

    parser.add_argument("--out",       metavar="DIR", default="results/",
                        help="Directorio de salida (por defecto: results/).")
    parser.add_argument("--delimiter", default="auto",
                        help="Separador de columnas: 'auto' (defecto), 'tab', ','.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Activar logging DEBUG.")
    return parser


def _resolve_filepaths(args: argparse.Namespace) -> list[Path]:
    if args.batch:
        bp = Path(args.batch)
        if not bp.exists():
            sys.exit(f"Fichero de lista no encontrado: {bp}")
        return [
            Path(line.strip())
            for line in bp.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if args.glob_pattern:
        matched = sorted(_glob.glob(args.glob_pattern))
        if not matched:
            sys.exit(f"Ningún fichero coincide con el patrón: {args.glob_pattern}")
        return [Path(p) for p in matched]
    # --data-dir
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Directorio no encontrado: {data_dir}")
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        sys.exit(f"No se encontraron ficheros *.txt en {data_dir}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    delimiter = _delimiter_arg(args.delimiter)
    filepaths = _resolve_filepaths(args)
    logger.info("Procesando %d fichero(s) de galaxias.", len(filepaths))

    out_dir = Path(args.out)
    results = run_pipeline(filepaths, delimiter=delimiter, out_dir=out_dir)

    print(results[["galaxy", "n_points", "chi2_red", "scm_u_term", "veredicto"]].to_string(index=False))

    n_ok = (results["veredicto"] == "OK").sum()
    print(f"\nVeredicto Framework SCM: {n_ok}/{len(results)} galaxias OK.")
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
