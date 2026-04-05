#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/scm_extreme_25_test.py — SCM test on the 25 environmental-extreme galaxies.

Loads an F3 catalog and an environmental catalog, selects the 25 most extreme
galaxies (either from an explicit list or automatically as the 12 lowest and
13 highest δ_mass_std values), computes Spearman statistics, and writes two
output files.

Inputs
------
f3_catalog.csv
    Required columns: ``galaxy``, ``friction_slope`` (or ``beta``),
    ``log_M_bar``, ``log_Rmax``.
env_catalog.csv
    Required columns: ``galaxy``, ``delta_mass_std``.
extreme_list.csv (optional)
    Explicit list of 25 galaxy names.  Must have column ``galaxy``.

Outputs (written to --out directory)
--------------------------------------
extreme_25_results.csv
    Merged per-galaxy data for the selected subsample.
extreme_25_summary.txt
    Plain-text summary: N, β mean ± std, Spearman ρ, and p-value.

Usage
-----
::

    python scripts/scm_extreme_25_test.py \\
        --f3-catalog results/f3_catalog_real.csv \\
        --env-catalog results/delta_mass_yang_sparc.csv

    python scripts/scm_extreme_25_test.py \\
        --f3-catalog results/f3_catalog_real.csv \\
        --env-catalog results/delta_mass_yang_sparc.csv \\
        --extreme-list data/my_25_galaxies.csv \\
        --out results/extreme_25
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

F3_REQUIRED = {"galaxy"}
F3_BETA_ALIASES = {"friction_slope", "beta"}
ENV_REQUIRED = {"galaxy", "delta_mass_std"}


def load_data(f3_path: str | Path, env_path: str | Path) -> pd.DataFrame:
    """Load and merge the F3 and environmental catalogs.

    Parameters
    ----------
    f3_path : path-like
        F3 catalog CSV.  Must contain ``galaxy`` and either ``friction_slope``
        or ``beta``.
    env_path : path-like
        Environmental catalog CSV.  Must contain ``galaxy`` and
        ``delta_mass_std``.

    Returns
    -------
    pd.DataFrame
        Inner-merged table with columns ``galaxy``, ``beta``,
        ``delta_mass_std``, and any additional columns from either input.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    ValueError
        If required columns are absent.
    """
    f3_path = Path(f3_path)
    env_path = Path(env_path)

    if not f3_path.exists():
        raise FileNotFoundError(f"F3 catalog not found: {f3_path}")
    if not env_path.exists():
        raise FileNotFoundError(f"Environmental catalog not found: {env_path}")

    df_f3 = pd.read_csv(f3_path)
    df_env = pd.read_csv(env_path)

    # Accept either 'friction_slope' or 'beta' as the slope column in df_f3
    if "friction_slope" in df_f3.columns and "beta" not in df_f3.columns:
        df_f3 = df_f3.rename(columns={"friction_slope": "beta"})

    missing_f3 = ({"galaxy", "beta"}) - set(df_f3.columns)
    if missing_f3:
        raise ValueError(f"F3 catalog missing required columns: {missing_f3}")

    missing_env = ENV_REQUIRED - set(df_env.columns)
    if missing_env:
        raise ValueError(f"Environmental catalog missing required columns: {missing_env}")

    df = pd.merge(df_f3, df_env, on="galaxy", how="inner")
    return df


# ---------------------------------------------------------------------------
# Galaxy selection
# ---------------------------------------------------------------------------


def select_25_extremes(
    df: pd.DataFrame,
    list_path: str | Path | None = None,
) -> pd.DataFrame:
    """Select the 25 most environmentally extreme galaxies.

    If *list_path* is provided, the subsample is the intersection of *df* with
    the galaxy names listed in the file (column ``galaxy``).  Otherwise the
    function takes the 12 galaxies with the lowest ``delta_mass_std`` and the
    13 with the highest.

    Parameters
    ----------
    df : pd.DataFrame
        Merged catalog with at least ``galaxy`` and ``delta_mass_std``.
    list_path : path-like or None
        Optional explicit list CSV.

    Returns
    -------
    pd.DataFrame
        Subsample of *df* (may be smaller than 25 if the catalog is small or
        the list has fewer matches).

    Raises
    ------
    FileNotFoundError
        If *list_path* is provided but the file does not exist.
    ValueError
        If *list_path* does not contain a ``galaxy`` column.
    """
    if list_path is not None:
        list_path = Path(list_path)
        if not list_path.exists():
            raise FileNotFoundError(f"Extreme-galaxy list not found: {list_path}")
        sel = pd.read_csv(list_path)
        if "galaxy" not in sel.columns:
            raise ValueError(
                f"Extreme-galaxy list file '{list_path}' must contain a 'galaxy' column."
            )
        return df[df["galaxy"].isin(sel["galaxy"])].copy()

    # Automatic selection: bottom 12 + top 13 by delta_mass_std
    df_sorted = df.sort_values("delta_mass_std")
    low = df_sorted.head(12)
    high = df_sorted.tail(13)
    return pd.concat([low, high], ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the extreme subsample.

    Parameters
    ----------
    df : pd.DataFrame
        Subsample with at least ``beta`` and ``delta_mass_std``.

    Returns
    -------
    dict
        Keys: ``N``, ``beta_mean``, ``beta_std``, ``rho_spearman``,
        ``p_spearman``.
    """
    beta = df["beta"].values
    delta = df["delta_mass_std"].values

    rho, p = spearmanr(delta, beta)

    return {
        "N": int(len(df)),
        "beta_mean": float(np.mean(beta)),
        "beta_std": float(np.std(beta)),
        "rho_spearman": float(rho),
        "p_spearman": float(p),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(df: pd.DataFrame, stats: dict, out_dir: str | Path) -> None:
    """Write results CSV and summary text file.

    Parameters
    ----------
    df : pd.DataFrame
        Per-galaxy results for the extreme subsample.
    stats : dict
        Output of :func:`compute_stats`.
    out_dir : path-like
        Output directory (created if it does not exist).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "extreme_25_results.csv", index=False)

    with open(out_dir / "extreme_25_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("=== EXTREME 25 GALAXIES TEST ===\n\n")
        fh.write(f"N = {stats['N']}\n")
        fh.write(
            f"beta mean \u00b1 std = {stats['beta_mean']:.4f} \u00b1 {stats['beta_std']:.4f}\n"
        )
        fh.write(f"Spearman rho = {stats['rho_spearman']:.3f}\n")
        fh.write(f"p-value = {stats['p_spearman']:.3e}\n\n")
        fh.write("Interpretation:\n")
        fh.write(
            "Negative correlation indicates environmental modulation of outer dynamics.\n"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCM test on the 25 environmentally most extreme galaxies."
    )
    parser.add_argument(
        "--f3-catalog",
        required=True,
        dest="f3_catalog",
        metavar="FILE",
        help="F3 catalog CSV (columns: galaxy, friction_slope/beta, log_M_bar, log_Rmax).",
    )
    parser.add_argument(
        "--env-catalog",
        required=True,
        dest="env_catalog",
        metavar="FILE",
        help="Environmental catalog CSV (columns: galaxy, delta_mass_std).",
    )
    parser.add_argument(
        "--extreme-list",
        default=None,
        dest="extreme_list",
        metavar="FILE",
        help="Optional explicit list of 25 galaxy names (column: galaxy).",
    )
    parser.add_argument(
        "--out",
        default="results/extreme_25",
        metavar="DIR",
        help="Output directory (default: results/extreme_25).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    df = load_data(args.f3_catalog, args.env_catalog)
    df_extreme = select_25_extremes(df, args.extreme_list)
    stats = compute_stats(df_extreme)
    save_outputs(df_extreme, stats, args.out)

    print("\n=== DONE ===")
    print(f"  N = {stats['N']}")
    print(f"  beta mean ± std = {stats['beta_mean']:.4f} ± {stats['beta_std']:.4f}")
    print(f"  Spearman rho = {stats['rho_spearman']:.3f},  p = {stats['p_spearman']:.3e}")
    print(f"\n  Results written to: {args.out}")


if __name__ == "__main__":
    main()
