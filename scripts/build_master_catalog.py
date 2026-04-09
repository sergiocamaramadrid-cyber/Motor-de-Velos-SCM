"""
scripts/build_master_catalog.py — Entry point: assemble the master SPARC catalog.

Merges the SPARC summary table (``data/sparc_basic.csv``), the slope-tail
catalog produced by :mod:`scripts.compute_slope_tail`, and the environmental
proxy table (``data/env_proxy.csv``) into the master catalog
``data/galaxy_catalog_env.csv``.

This script is a named entry point that delegates all logic to
:mod:`scripts.build_galaxy_catalog_env`.  All CLI flags accepted by that
module are forwarded transparently.

Usage
-----
::

    python scripts/build_master_catalog.py

Custom paths::

    python scripts/build_master_catalog.py \\
        --sparc   data/sparc_basic.csv \\
        --slopes  results/slope_tail.csv \\
        --env     data/env_proxy.csv \\
        --out     data/galaxy_catalog_env.csv
"""

from __future__ import annotations

import sys

from scripts.build_galaxy_catalog_env import main as _main


def main(argv: list[str] | None = None) -> dict:
    """Assemble the master SPARC environment catalog.

    Delegates entirely to :func:`scripts.build_galaxy_catalog_env.main`.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments forwarded to the underlying pipeline.
        When ``None`` the process ``sys.argv[1:]`` are used.

    Returns
    -------
    dict
        Whatever :func:`scripts.build_galaxy_catalog_env.main` returns
        (keys: ``catalog``, ``n``, ``out_path``).
    """
    return _main(argv)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
