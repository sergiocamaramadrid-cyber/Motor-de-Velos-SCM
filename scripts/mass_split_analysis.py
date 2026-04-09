"""
scripts/mass_split_analysis.py — Entry point: slope-tail distribution by stellar mass.

Generates the high-mass slope-tail histogram figure (and its PDF companion)
using the SPARC summary and slope-tail catalogs.  Galaxies with
``logM > LOGM_CUT_DEFAULT`` are selected for the analysis.

This script is a named entry point that delegates all logic to
:mod:`scripts.plot_sparc_slope_tail_hist`.  All CLI flags accepted by that
module (``--sparc``, ``--slopes``, ``--out``, ``--logm-cut``) are forwarded
transparently.

Usage
-----
::

    python scripts/mass_split_analysis.py

Custom paths and threshold::

    python scripts/mass_split_analysis.py \\
        --sparc   data/sparc_basic.csv \\
        --slopes  results/slope_tail.csv \\
        --out     results/fig_slope_tail_high_mass.png \\
        --logm-cut 10.0
"""

from __future__ import annotations

import sys

from scripts.plot_sparc_slope_tail_hist import main as _main


def main(argv: list[str] | None = None) -> dict:
    """Run slope-tail distribution analysis split by stellar mass.

    Delegates entirely to :func:`scripts.plot_sparc_slope_tail_hist.main`.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments forwarded to the underlying pipeline.
        When ``None`` the process ``sys.argv[1:]`` are used.

    Returns
    -------
    dict
        Whatever :func:`scripts.plot_sparc_slope_tail_hist.main` returns
        (keys: ``figure_path``, ``pdf_path``, ``stats``, ``n``).
    """
    return _main(argv)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
