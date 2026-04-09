"""
scripts/run_full_pipeline.py — End-to-end SPARC SCM analysis pipeline.

Executes the four canonical analysis steps in order:

1. **compute_slope_tail** — fits the outer-disk velocity slope for each SPARC
   rotation curve and writes ``results/slope_tail.csv``.
2. **build_master_catalog** — merges the SPARC summary, slope-tail catalog, and
   environmental proxy table into ``data/galaxy_catalog_env.csv``.
3. **mass_split_analysis** — generates the high-mass slope-tail distribution
   histogram figure.
4. **env_mass_scan** — computes the Spearman ρ vs mass-threshold curve and
   writes the corresponding figure.

Each step is executed as a subprocess so it is fully isolated from the caller.
A non-zero exit code from any step raises :exc:`RuntimeError` by default
(override with ``--no-check``).

Usage
-----
::

    python scripts/run_full_pipeline.py

Skip individual steps with ``--skip``::

    python scripts/run_full_pipeline.py --skip mass_split_analysis

Dry-run (print commands without executing)::

    python scripts/run_full_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

#: Each entry is a ``(key, label, script_path)`` tuple.
#: *key*         — short identifier used with ``--skip``
#: *label*       — human-readable description printed during the run
#: *script_path* — path relative to the repository root
PIPELINE_STEPS: list[tuple[str, str, str]] = [
    (
        "compute_slope_tail",
        "Computar slope-tail exterior para galaxias SPARC",
        "scripts/compute_slope_tail.py",
    ),
    (
        "build_master_catalog",
        "Ensamblar catálogo maestro SPARC con entorno",
        "scripts/build_master_catalog.py",
    ),
    (
        "mass_split_analysis",
        "Análisis de distribución slope-tail por masa estelar",
        "scripts/mass_split_analysis.py",
    ),
    (
        "env_mass_scan",
        "Escaneo de correlación ambiental vs umbral de masa",
        "scripts/env_mass_scan.py",
    ),
]

#: Ordered list of valid step keys (used for ``--skip`` validation).
STEP_KEYS: list[str] = [key for key, _, _ in PIPELINE_STEPS]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_step(
    cmd: str | list[str],
    *,
    check: bool = True,
    dry_run: bool = False,
) -> dict:
    """Run a single pipeline step as a subprocess.

    Parameters
    ----------
    cmd : str or list of str
        The command to execute.  A plain string is split on whitespace.
    check : bool
        If ``True`` (default) a :exc:`RuntimeError` is raised when the
        subprocess exits with a non-zero return code.
    dry_run : bool
        When ``True`` the command is printed but not executed.

    Returns
    -------
    dict with keys:
        cmd         — str, the command that was (or would be) run
        returncode  — int (0 on dry-run)
        success     — bool
    """
    cmd_list: list[str] = cmd if isinstance(cmd, list) else cmd.split()
    cmd_str: str = " ".join(cmd_list)

    print(f"> {cmd_str}")

    if dry_run:
        return {"cmd": cmd_str, "returncode": 0, "success": True}

    result = subprocess.run(cmd_list, check=False)
    success = result.returncode == 0

    if not success and check:
        raise RuntimeError(
            f"Fallo en el paso del pipeline (salida {result.returncode}): {cmd_str}"
        )

    return {
        "cmd": cmd_str,
        "returncode": result.returncode,
        "success": success,
    }


def run_pipeline(
    steps: list[tuple[str, str, str]] | None = None,
    *,
    skip: set[str] | None = None,
    check: bool = True,
    dry_run: bool = False,
    python: str | None = None,
) -> list[dict]:
    """Execute the full pipeline (or a subset).

    Parameters
    ----------
    steps : list of (key, label, script_path) or None
        Steps to consider; defaults to :data:`PIPELINE_STEPS`.
    skip : set of str or None
        Step keys to omit.
    check : bool
        Propagated to :func:`run_step`.
    dry_run : bool
        Propagated to :func:`run_step`.
    python : str or None
        Python interpreter to use; defaults to ``sys.executable``.

    Returns
    -------
    list of dict
        One result dict per executed step (skipped steps excluded).
    """
    if steps is None:
        steps = PIPELINE_STEPS
    if skip is None:
        skip = set()
    if python is None:
        python = sys.executable

    results: list[dict] = []
    for key, label, script in steps:
        if key in skip:
            print(f"[OMITIR] {label}")
            continue

        print(f"\n{'='*60}")
        print(f"[PASO:{key}] {label}")
        print(f"{'='*60}")

        result = run_step([python, script], check=check, dry_run=dry_run)
        result["key"] = key
        result["label"] = label
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecutar el pipeline completo de análisis SPARC SCM "
            "(slope-tail → catálogo → histograma → escaneo ambiental)."
        )
    )
    parser.add_argument(
        "--skip",
        metavar="KEY",
        action="append",
        default=[],
        choices=STEP_KEYS,
        help=(
            f"Clave del paso a omitir (se puede repetir). "
            f"Claves válidas: {', '.join(STEP_KEYS)}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Mostrar comandos sin ejecutarlos.",
    )
    parser.add_argument(
        "--no-check",
        action="store_false",
        dest="check",
        help="Continuar aunque un paso salga con código de error.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Intérprete Python a usar (por defecto: intérprete actual).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Punto de entrada: analizar argumentos CLI y ejecutar el pipeline.

    Returns
    -------
    dict with keys:
        steps    — list of step result dicts
        n_run    — int, número de pasos ejecutados
        n_ok     — int, número de pasos exitosos
        success  — bool, True cuando todos los pasos ejecutados tuvieron éxito
    """
    args = _parse_args(argv)

    results = run_pipeline(
        skip=set(args.skip),
        check=args.check,
        dry_run=args.dry_run,
        python=args.python,
    )

    n_ok = sum(1 for r in results if r["success"])
    n_run = len(results)
    success = n_ok == n_run

    print(f"\n{'='*60}")
    if success:
        print(f"✅ Pipeline completo ejecutado  ({n_ok}/{n_run} pasos OK)")
    else:
        n_fail = n_run - n_ok
        print(f"⚠️  Pipeline terminado con {n_fail} fallo(s)  ({n_ok}/{n_run} OK)")
    print(f"{'='*60}")

    return {
        "steps": results,
        "n_run": n_run,
        "n_ok": n_ok,
        "success": success,
    }


if __name__ == "__main__":
    main()
