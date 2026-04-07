"""
scripts/build_delta_f3_from_sparc.py — Build Paper 1 galaxy catalog.

Merges the per-galaxy F3/β catalog (from ``generate_f3_catalog.py``) with
the SPARC photometric table to produce the base catalogs used in the
environmental modulation analysis (Paper 1).

Outputs
-------
Three CSV files written to *out_dir*:

``galaxy_catalog.csv``
    Per-galaxy catalog with columns: galaxy, beta, delta_f3, logMbar, logRd.

``galaxy_catalog_with_env.csv``
    Same as above plus an environmental proxy column (env_proxy / delta_mass).

``delta_mass_proxy.csv``
    Minimal table: galaxy, delta_mass (the environmental proxy).

Environmental proxy
-------------------
delta_mass = log10(MHI / L36)

This gas-fraction ratio captures the asymmetry between HI mass and stellar
luminosity.  HI-rich galaxies in low-density environments show higher
delta_mass, making it a useful proxy for the local density field.

Baryonic mass
-------------
logMbar = log10(M_star + M_gas)
        = log10(0.5 × L36 × 10⁹  +  1.33 × MHI × 10⁹)

using the standard SPARC mass-to-light ratio (Υ★ = 0.5 at 3.6 μm) and the
factor 1.33 to account for helium in the gas mass.

Usage
-----
::

    python scripts/build_delta_f3_from_sparc.py \\
        --f3-catalog results/f3_catalog_real.csv \\
        --sparc-table data/SPARC/SPARC_Lelli2016c.csv \\
        --out-dir results/paper1_environment/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF_DEFAULT: float = 0.5
"""Reference deep-regime slope under MOND / SCM (β = 0.5)."""

DEFAULT_OUT_DIR: str = "results/paper1_environment/data"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_logMbar(L36: np.ndarray, MHI: np.ndarray) -> np.ndarray:
    """Compute log10 baryonic mass from SPARC photometry.

    Parameters
    ----------
    L36 : array_like
        3.6 μm luminosity in units of 10⁹ L_sun.
    MHI : array_like
        HI mass in units of 10⁹ M_sun.

    Returns
    -------
    ndarray
        log10(M_bar / M_sun).  Rows where M_bar ≤ 0 are returned as NaN.
    """
    L36 = np.asarray(L36, dtype=float)
    MHI = np.asarray(MHI, dtype=float)
    mbar = 0.5 * L36 * 1e9 + 1.33 * MHI * 1e9
    return np.where(mbar > 0, np.log10(mbar), np.nan)


def compute_logRd(Re: np.ndarray) -> np.ndarray:
    """Compute log10 disk scale length from the SPARC effective radius.

    The effective (half-light) radius ``Re`` is related to the exponential
    disk scale length ``Rd`` by ``Re ≈ 1.678 × Rd``, so
    ``Rd = Re / 1.678``.

    Parameters
    ----------
    Re : array_like
        Effective radius in kpc.

    Returns
    -------
    ndarray
        log10(Rd / kpc).  Non-positive values are returned as NaN.
    """
    Re = np.asarray(Re, dtype=float)
    Rd = Re / 1.678
    return np.where(Rd > 0, np.log10(Rd), np.nan)


def compute_env_proxy(L36: np.ndarray, MHI: np.ndarray) -> np.ndarray:
    """Compute the environmental proxy delta_mass = log10(MHI / L36).

    Parameters
    ----------
    L36 : array_like
        3.6 μm luminosity (10⁹ L_sun).
    MHI : array_like
        HI mass (10⁹ M_sun).

    Returns
    -------
    ndarray
        log10(MHI / L36).  Rows where either quantity is ≤ 0 are NaN.
    """
    L36 = np.asarray(L36, dtype=float)
    MHI = np.asarray(MHI, dtype=float)
    ratio = np.where((L36 > 0) & (MHI > 0), MHI / L36, np.nan)
    return np.log10(ratio)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_catalog(
    f3_catalog: str | Path,
    sparc_table: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    beta_ref: float = BETA_REF_DEFAULT,
) -> dict[str, pd.DataFrame]:
    """Merge the F3 catalog with SPARC photometry and write Paper 1 catalogs.

    Parameters
    ----------
    f3_catalog : str or Path
        Path to the per-galaxy F3 catalog CSV produced by
        ``generate_f3_catalog.py``.  Must contain columns ``galaxy`` and
        ``beta`` (or the SCM aliases ``friction_slope``).
    sparc_table : str or Path
        Path to the SPARC galaxy table CSV with at least the columns
        ``Galaxy``, ``L36``, ``MHI``, and ``Re``.
    out_dir : str or Path
        Directory where the three output CSVs will be written.
    beta_ref : float
        Reference β value for computing delta_f3 = β − beta_ref.
        Default is 0.5 (MOND deep-regime prediction).

    Returns
    -------
    dict
        Keys: ``'catalog'``, ``'catalog_with_env'``, ``'delta_mass_proxy'``.
        Values: corresponding :class:`pandas.DataFrame` objects.

    Raises
    ------
    FileNotFoundError
        If *f3_catalog* or *sparc_table* does not exist.
    ValueError
        If required columns are absent from either input file.
    """
    f3_path = Path(f3_catalog)
    sparc_path = Path(sparc_table)

    if not f3_path.exists():
        raise FileNotFoundError(f"F3 catalog not found: {f3_path}")
    if not sparc_path.exists():
        raise FileNotFoundError(f"SPARC table not found: {sparc_path}")

    # --- Load F3 catalog ---
    df_f3 = pd.read_csv(f3_path)
    # Accept SCM canonical aliases
    if "beta" not in df_f3.columns and "friction_slope" in df_f3.columns:
        df_f3 = df_f3.rename(columns={"friction_slope": "beta"})
    _require_columns(df_f3, {"galaxy", "beta"}, f3_path)

    # --- Load SPARC photometric table ---
    df_sparc = pd.read_csv(sparc_path)
    _require_columns(df_sparc, {"Galaxy", "L36", "MHI", "Re"}, sparc_path)

    # Normalise column types
    df_sparc["L36"] = pd.to_numeric(df_sparc["L36"], errors="coerce")
    df_sparc["MHI"] = pd.to_numeric(df_sparc["MHI"], errors="coerce")
    df_sparc["Re"] = pd.to_numeric(df_sparc["Re"], errors="coerce")

    # Rename Galaxy → galaxy for merge
    df_sparc_m = df_sparc.rename(columns={"Galaxy": "galaxy"})

    # --- Merge ---
    df = df_f3[["galaxy", "beta"]].merge(
        df_sparc_m[["galaxy", "L36", "MHI", "Re"]],
        on="galaxy",
        how="inner",
    )

    # --- Derived columns ---
    df["delta_f3"] = df["beta"] - beta_ref
    df["logMbar"] = compute_logMbar(df["L36"].values, df["MHI"].values)
    df["logRd"] = compute_logRd(df["Re"].values)
    df["env_proxy"] = compute_env_proxy(df["L36"].values, df["MHI"].values)

    # --- Assemble output tables ---
    catalog_cols = ["galaxy", "beta", "delta_f3", "logMbar", "logRd"]
    df_catalog = df[catalog_cols].dropna(subset=["logMbar", "logRd"]).copy()

    env_cols = catalog_cols + ["env_proxy"]
    df_env = df[env_cols].dropna(subset=["logMbar", "logRd", "env_proxy"]).copy()
    df_env = df_env.rename(columns={"env_proxy": "delta_mass"})

    df_delta = df_env[["galaxy", "delta_mass"]].copy()

    # --- Write outputs ---
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_catalog.to_csv(out_dir / "galaxy_catalog.csv", index=False)
    df_env.to_csv(out_dir / "galaxy_catalog_with_env.csv", index=False)
    df_delta.to_csv(out_dir / "delta_mass_proxy.csv", index=False)

    return {
        "catalog": df_catalog,
        "catalog_with_env": df_env,
        "delta_mass_proxy": df_delta,
    }


def _require_columns(
    df: pd.DataFrame, required: set[str], path: Path
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing in {path}: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Paper 1 galaxy catalog by merging the F3/β catalog "
            "with SPARC photometry."
        )
    )
    parser.add_argument(
        "--f3-catalog", dest="f3_catalog", required=True,
        help="Path to F3 catalog CSV (from generate_f3_catalog.py).",
    )
    parser.add_argument(
        "--sparc-table", dest="sparc_table", required=True,
        help="Path to SPARC galaxy table CSV (Galaxy, L36, MHI, Re).",
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--beta-ref", dest="beta_ref", type=float, default=BETA_REF_DEFAULT,
        help=f"Reference β for delta_f3 = β − beta_ref (default: {BETA_REF_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, **kwargs) -> dict[str, pd.DataFrame]:
    """Entry point: parse CLI arguments and run catalog build.

    Accepts either a list of CLI tokens via *argv* or keyword arguments
    ``f3_catalog``, ``sparc_table``, ``out_dir``, ``beta_ref`` directly
    (keyword args take precedence).

    Returns
    -------
    dict
        Keys: ``'catalog'``, ``'catalog_with_env'``, ``'delta_mass_proxy'``.
    """
    if kwargs:
        f3_catalog = kwargs.get("f3_catalog")
        sparc_table = kwargs.get("sparc_table")
        out_dir = kwargs.get("out_dir", DEFAULT_OUT_DIR)
        beta_ref = kwargs.get("beta_ref", BETA_REF_DEFAULT)
        if f3_catalog is None or sparc_table is None:
            raise ValueError("f3_catalog and sparc_table keyword arguments are required")
    else:
        args = _parse_args(argv)
        f3_catalog = args.f3_catalog
        sparc_table = args.sparc_table
        out_dir = args.out_dir
        beta_ref = args.beta_ref
    return build_catalog(
        f3_catalog=f3_catalog,
        sparc_table=sparc_table,
        out_dir=out_dir,
        beta_ref=beta_ref,
    )


if __name__ == "__main__":
    main()
