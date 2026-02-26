"""
scm_sparc_full_analysis.py — Root-level entry point for the Motor de Velos SCM
full analysis pipeline.

This script is a thin wrapper around ``src.scm_analysis.main`` that exposes the
same CLI as the module entry-point but can be invoked directly without the
``python -m`` prefix::

    python scm_sparc_full_analysis.py --data-dir data/sparc \\
        --outdir scm_results_full --seed 20260211

All arguments are forwarded to :func:`src.scm_analysis.main`.  Run with
``--help`` for a full description of available options.
"""

from src.scm_analysis import main

if __name__ == "__main__":
    main()
