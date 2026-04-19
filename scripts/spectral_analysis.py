"""
SCM – Spectral Environmental Modulation
========================================
Tests whether environmental modulation of spectral power persists after
removing baryonic mass dependence (OLS mass-controlled residuals).

Usage
-----
    python scripts/spectral_analysis.py [--csv PATH] [--logM-min FLOAT]

Outputs
-------
    Prints Spearman ρ, p-value, and N to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_CSV = Path(__file__).parent.parent / "spectral_dataset_final.csv"
DEFAULT_LOGM_MIN: float = 10.0


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the spectral dataset CSV.

    Parameters
    ----------
    csv_path:
        Path to ``spectral_dataset_final.csv`` (or equivalent).

    Returns
    -------
    pandas.DataFrame with at least the columns:
        ``logM``, ``power``, ``delta_mass_std``.
    """
    df = pd.read_csv(csv_path)
    required = {"logM", "power", "delta_mass_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def filter_high_mass(df: pd.DataFrame, logm_min: float = DEFAULT_LOGM_MIN) -> pd.DataFrame:
    """Return rows where logM >= *logm_min*."""
    return df[df["logM"] >= logm_min].copy()


def compute_mass_controlled_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Add an OLS residual column after regressing ``power`` on ``logM``.

    Parameters
    ----------
    df:
        DataFrame containing ``logM`` and ``power`` columns.

    Returns
    -------
    A copy of *df* with a new ``residual`` column.
    """
    X = sm.add_constant(df["logM"])
    model = sm.OLS(df["power"], X).fit()
    out = df.copy()
    out["residual"] = model.resid
    return out


def compute_env_residual_correlation(df: pd.DataFrame) -> dict:
    """Compute Spearman ρ between ``delta_mass_std`` and ``residual``.

    Parameters
    ----------
    df:
        DataFrame containing ``delta_mass_std`` and ``residual`` columns.

    Returns
    -------
    dict with keys ``rho``, ``p``, ``n``.
    """
    rho, p = spearmanr(df["delta_mass_std"], df["residual"])
    return {"rho": float(rho), "p": float(p), "n": int(len(df))}


def run_analysis(
    csv_path: str | Path = DEFAULT_CSV,
    logm_min: float = DEFAULT_LOGM_MIN,
) -> dict:
    """End-to-end analysis pipeline.

    Parameters
    ----------
    csv_path:
        Path to the spectral dataset CSV.
    logm_min:
        Minimum logM threshold for the high-mass sub-sample.

    Returns
    -------
    dict with keys ``rho``, ``p``, ``n``.
    """
    df = load_dataset(csv_path)
    df_high = filter_high_mass(df, logm_min)
    df_high = compute_mass_controlled_residuals(df_high)
    return compute_env_residual_correlation(df_high)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SCM spectral environmental modulation test."
    )
    p.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to spectral_dataset_final.csv (default: %(default)s)",
    )
    p.add_argument(
        "--logM-min",
        type=float,
        default=DEFAULT_LOGM_MIN,
        dest="logm_min",
        help="Minimum logM for high-mass sub-sample (default: %(default)s)",
    )
    return p


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    Result dict with keys ``rho``, ``p``, ``n``.
    """
    args = _build_parser().parse_args(argv)
    result = run_analysis(csv_path=args.csv, logm_min=args.logm_min)
    print("ENV vs residual:")
    print("rho =", result["rho"])
    print("p =", result["p"])
    print("N =", result["n"])
    return result


if __name__ == "__main__":
    sys.exit(main())
