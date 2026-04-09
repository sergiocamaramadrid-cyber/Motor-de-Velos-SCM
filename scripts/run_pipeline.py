"""
scripts/run_pipeline.py — End-to-end SPARC SCM analysis pipeline.

Executes the four analysis steps in order:

1. **Slope-tail computation** (``sparc_slope_tail.py``) — fits the outer-disk
   velocity slope for each SPARC rotation curve and writes
   ``results/slope_tail.csv``.
2. **Master catalog assembly** (``build_galaxy_catalog_env.py``) — merges the
   SPARC summary table, the slope catalog, and the environmental proxy table
   into ``data/galaxy_catalog_env.csv``.
3. **Slope-tail histogram by mass** (``plot_sparc_slope_tail_hist.py``) —
   generates the high-mass slope-tail distribution figure.
4. **Environmental mass scan** (``plot_env_mass_scan.py``) — computes the
   Spearman ρ vs mass-threshold curve and writes the corresponding figure.

Each step is run as a subprocess so it is fully isolated from the caller.
A non-zero exit code from any step raises :exc:`RuntimeError` by default
(controlled by the ``check`` parameter).

Usage
-----
::

    python scripts/run_pipeline.py

Skip individual steps with ``--skip``::

    python scripts/run_pipeline.py --skip slope_tail --skip slope_hist

Dry-run (print commands without executing)::

    python scripts/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

#: Each entry is a ``(key, label, script_path)`` tuple.
#: *key*         — short identifier used with ``--skip``
#: *label*       — human-readable description printed during the run
#: *script_path* — path relative to the repository root
PIPELINE_STEPS: list[tuple[str, str, str]] = [
    (
        "slope_tail",
        "Compute outer-disk slope-tail for SPARC galaxies",
        "scripts/sparc_slope_tail.py",
    ),
    (
        "master_catalog",
        "Assemble master SPARC environment catalog",
        "scripts/build_galaxy_catalog_env.py",
    ),
    (
        "slope_hist",
        "Generate slope-tail histogram by stellar mass",
        "scripts/plot_sparc_slope_tail_hist.py",
    ),
    (
        "env_scan",
        "Run environmental correlation vs mass-threshold scan",
        "scripts/plot_env_mass_scan.py",
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
        The command to execute.  When a plain string is given it is split on
        whitespace; pass a list to avoid shell quoting issues.
    check : bool
        If ``True`` (default) a :exc:`RuntimeError` is raised when the
        subprocess exits with a non-zero return code.
    dry_run : bool
        When ``True`` the command is printed but not executed.  The returned
        dict contains ``returncode=0`` and ``success=True``.

    Returns
    -------
    dict with keys:
        cmd         — str, the command that was (or would be) run
        returncode  — int, subprocess exit code (0 on dry-run)
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
            f"Pipeline step failed (exit {result.returncode}): {cmd_str}"
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
        The steps to consider.  Defaults to :data:`PIPELINE_STEPS`.
    skip : set of str or None
        Step keys to omit.
    check : bool
        Propagated to :func:`run_step`.  ``True`` aborts on first failure.
    dry_run : bool
        Propagated to :func:`run_step`.
    python : str or None
        Python interpreter to use.  Defaults to ``sys.executable``.

    Returns
    -------
    list of dict
        One result dict per executed step (skipped steps are excluded).
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
            print(f"[SKIP] {label}")
            continue

        print(f"\n{'='*60}")
        print(f"[STEP:{key}] {label}")
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
            "Run the full SPARC SCM analysis pipeline "
            "(slope-tail → master catalog → histogram → env scan)."
        )
    )
    parser.add_argument(
        "--skip",
        metavar="KEY",
        action="append",
        default=[],
        choices=STEP_KEYS,
        help=(
            f"Step key to skip (may be repeated). "
            f"Valid keys: {', '.join(STEP_KEYS)}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--no-check",
        action="store_false",
        dest="check",
        help="Continue even if a step exits with a non-zero code.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter to use (default: current interpreter).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: parse CLI args and run the pipeline.

    Returns
    -------
    dict with keys:
        steps    — list of step result dicts
        n_run    — int, number of steps executed
        n_ok     — int, number of successful steps
        success  — bool, True when all executed steps succeeded
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
