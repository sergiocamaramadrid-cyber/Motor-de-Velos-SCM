"""
big_sparc.py — BIGSPARCAdapter stub.

This adapter is a placeholder for BIG-SPARC (~4000 galaxies).  It defines the
correct interface and documents what the implementation will expect, but raises
:class:`NotImplementedError` until the official derived-table products are
released.

When BIG-SPARC derived tables become available, implement :meth:`BIGSPARCAdapter.ingest`
here without touching the core pipeline or the catalog modules.

Expected input formats (any of):
    - ``galaxies.(csv|parquet)`` + ``rc_points.(csv|parquet)``
    - A directory of ``*_rotmod.dat`` files (same as SPARC)
    - A single mega-catalog (CSV / MRT / parquet)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .base import IngestConfig


class BIGSPARCAdapter:
    """
    Adapter for BIG-SPARC derived-table products (stub).

    Implement :meth:`ingest` once the official BIG-SPARC tables are released.
    The rest of the pipeline (β catalog, population stats, CLI) requires no
    changes — just a working implementation of this method.
    """

    name: str = "big-sparc"

    def ingest(
        self,
        input_path: Path,
        config: IngestConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Not yet implemented — awaiting official BIG-SPARC data release.

        Parameters
        ----------
        input_path : Path
            Path to the BIG-SPARC data (directory or catalog file).
        config : IngestConfig
            Ingestion configuration.

        Raises
        ------
        NotImplementedError
            Always.  Implement this method when derived tables are available.
        """
        raise NotImplementedError(
            "BIGSPARCAdapter.ingest() is a stub until the official BIG-SPARC "
            "derived tables are released.  "
            "Expected inputs: galaxies.(csv|parquet) and rc_points.(csv|parquet) "
            "or equivalent per-galaxy rotation-curve tables."
        )
