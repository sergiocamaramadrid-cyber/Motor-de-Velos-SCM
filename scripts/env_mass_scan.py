"""
scripts/env_mass_scan.py — Entry point: environmental correlation vs mass-threshold scan.

For each mass threshold in a configurable list this script selects the
high-mass subsample (``logM > threshold``), computes the Spearman rank
correlation between ``env_proxy`` and ``slope_tail``, and plots the resulting
ρ vs threshold curve.

This script is a named entry point that delegates all logic to
:mod:`scripts.plot_env_mass_scan`.  All CLI flags accepted by that module
(``--catalog``, ``--out``, ``--thresholds``, ``--n-min``) are forwarded
transparently.

Usage
-----
::

    python scripts/env_mass_scan.py

Custom paths and thresholds::

    python scripts/env_mass_scan.py \\
        --catalog    data/galaxy_catalog_env.csv \\
        --out        results/fig_env_mass_scan.png \\
        --thresholds 9.8 10.0 10.05 10.2 10.3 \\
        --n-min      10
"""

from __future__ import annotations

import sys

from scripts.plot_env_mass_scan import main as _main


def main(argv: list[str] | None = None) -> dict:
    """Run environmental Spearman correlation vs mass-threshold scan.

    Delegates entirely to :func:`scripts.plot_env_mass_scan.main`.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments forwarded to the underlying pipeline.
        When ``None`` the process ``sys.argv[1:]`` are used.

    Returns
    -------
    dict
        Whatever :func:`scripts.plot_env_mass_scan.main` returns
        (keys: ``scan_df``, ``figure_path``, ``pdf_path``).
    """
    return _main(argv)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
