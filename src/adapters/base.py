"""
base.py — Common base types for all SCM survey adapters.

Defines:
    IngestConfig   — immutable config shared by every adapter
    Adapter        — Protocol that every adapter must satisfy
    add_metadata   — helper to stamp survey/instrument onto a galaxies DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple

import pandas as pd


@dataclass(frozen=True)
class IngestConfig:
    """Config common to all survey adapters."""
    deep_frac: float = 0.3
    a0_m_s2: float = 1.2e-10
    min_deep: int = 4
    survey: str = "unknown"
    instrument: Optional[str] = None   # e.g. ASKAP / MeerKAT


class Adapter(Protocol):
    """
    Protocol satisfied by every survey adapter.

    An adapter converts raw survey data into the SCM internal contract:
        galaxies_df   — one row per galaxy (at least: galaxy_id, survey, instrument)
        rc_points_df  — one row per radial point (at least: galaxy_id, r_kpc, vrot_kms)
    """

    name: str

    def ingest(
        self,
        input_path: Path,
        config: IngestConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (galaxies_df, rc_points_df) in the SCM contract format."""
        ...


def add_metadata(
    df_galaxies: pd.DataFrame,
    survey: str,
    instrument: Optional[str],
) -> pd.DataFrame:
    """Stamp survey/instrument columns onto a galaxies DataFrame.

    Parameters
    ----------
    df_galaxies : DataFrame
        Galaxy-level DataFrame (must already contain ``galaxy_id``).
    survey : str
        Survey name (e.g. 'SPARC', 'BIG-SPARC').
    instrument : str or None
        Instrument tag (e.g. 'ASKAP').  Stored as empty string if None.

    Returns
    -------
    DataFrame
        Copy of *df_galaxies* with ``survey`` and ``instrument`` columns added.
    """
    df = df_galaxies.copy()
    df["survey"] = survey
    df["instrument"] = instrument if instrument is not None else ""
    return df
