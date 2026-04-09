"""
scripts/compute_slope_tail.py — Entry point: compute outer-disk slope-tail for SPARC.

Reads SPARC rotation-curve files from ``data/SPARC/rotmod/`` and writes the
per-galaxy outer-disk log-log slope to ``results/slope_tail.csv``.

This script is a named entry point that delegates all logic to
:mod:`scripts.sparc_slope_tail`.  All CLI flags accepted by that module
(``--data-dir``, ``--out``, ``--tail-frac``, ``--min-points``) are
forwarded transparently.

Usage
-----
::

    python scripts/compute_slope_tail.py

Custom paths::

    python scripts/compute_slope_tail.py \\
        --data-dir data/SPARC/rotmod \\
        --out      results/slope_tail.csv
"""

from __future__ import annotations

import sys

from scripts.sparc_slope_tail import main as _main


def main(argv: list[str] | None = None) -> dict:
    """Compute outer-disk slope-tail for all SPARC rotation curves.

    Delegates entirely to :func:`scripts.sparc_slope_tail.main`.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments forwarded to the underlying pipeline.
        When ``None`` the process ``sys.argv[1:]`` are used.

    Returns
    -------
    dict
        Whatever :func:`scripts.sparc_slope_tail.main` returns
        (keys: ``slopes``, ``n``, ``out_path``).
    """
    return _main(argv)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
