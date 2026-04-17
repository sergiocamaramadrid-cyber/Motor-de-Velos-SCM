"""
scripts/build_galaxy_signal_table.py — Galaxy physical signal table.

For each SPARC galaxy, assembles the core kinematic and structural variables
into a single per-galaxy CSV (the "galaxy signal table").

Output columns
--------------
galaxy        — galaxy name
logMbar       — log10 total baryonic mass (Mstar + Mgas) in Msun
Mgas          — gas mass (1.33 × M_HI) in 1e9 Msun; NaN if unavailable
Rmax          — maximum observed radius (kpc)
Vmax          — maximum observed circular velocity (km/s)
slope_tail    — log-log slope β of g_obs vs g_bar in the outer half of radial points
delta_f3      — slope_tail minus the deep-regime reference value (default BETA_REF = 0.5)
env_proxy     — environmental density proxy; NaN if not provided
width_kpc     — approximate disk diameter = 2.5 × R_d (kpc); NaN if R_d unavailable
thickness_kpc — approximate disk thickness = 0.1 × R_d (kpc); NaN if R_d unavailable

Usage
-----
::

    python scripts/build_galaxy_signal_table.py \\
        --sparc-dir data/SPARC \\
        --out data/galaxy_signal_table.csv

With optional environmental proxy CSV (columns: galaxy, env_proxy)::

    python scripts/build_galaxy_signal_table.py \\
        --sparc-dir data/SPARC \\
        --env-csv data/env_proxy.csv \\
        --out data/galaxy_signal_table.csv

Notes
-----
* ``width_kpc`` and ``thickness_kpc`` use the exponential disk scale radius
  R_d from the SPARC catalog (``Rdisk`` column).  The approximation
  ``thickness ≈ 0.1 × R_d`` is declared as an order-of-magnitude estimate;
  see Kregel, van der Kruit & de Grijs (2002) for empirical thickness scaling.
* ``logMbar`` uses a fixed stellar mass-to-light ratio *upsilon* (default 0.5
  in solar units at 3.6 μm) applied to the SPARC L36 column (1e9 Lsun).
  Gas mass is 1.33 × M_HI (HI + He), both in units of 1e9 Msun.
* ``slope_tail`` is computed with ``upsilon_disk = 1.0`` (fixed) to avoid
  requiring a full SCM minimisation.  For publication-quality β values,
  use ``generate_f3_catalog.py`` which fits upsilon_disk per galaxy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF: float = 0.5           # deep-regime reference slope (MOND / SCM)
UPSILON_DEFAULT: float = 0.5    # 3.6 μm mass-to-light ratio (solar units)
HE_CORRECTION: float = 1.33     # HI → total gas (helium included)
A0_DEFAULT: float = 1.2e-10     # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT: float = 0.3   # outer-regime: g_bar < threshold × a0
_KPC_TO_M: float = 3.085_677_581e19   # 1 kpc in metres

# Minimum number of outer points needed for a meaningful slope fit
MIN_OUTER_POINTS: int = 2

OUTPUT_COLUMNS = [
    "galaxy", "logMbar", "Mgas", "Rmax", "Vmax",
    "slope_tail", "delta_f3", "env_proxy", "width_kpc", "thickness_kpc",
]

_GALAXY_TABLE_CANDIDATES = [
    "SPARC_Lelli2016c.csv",
    "SPARC_Lelli2016c.mrt",
]


# ---------------------------------------------------------------------------
# SPARC table helpers
# ---------------------------------------------------------------------------


def _find_galaxy_table(sparc_dir: Path) -> Path:
    """Return the path to the SPARC galaxy summary table inside *sparc_dir*."""
    for name in _GALAXY_TABLE_CANDIDATES:
        for prefix in (sparc_dir, sparc_dir / "raw", sparc_dir / "processed"):
            candidate = prefix / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {sparc_dir}. "
        f"Expected one of: {_GALAXY_TABLE_CANDIDATES}"
    )


def load_sparc_properties(sparc_dir: str | Path) -> pd.DataFrame:
    """Load per-galaxy structural properties from the SPARC galaxy table.

    Parameters
    ----------
    sparc_dir : str or Path
        Root SPARC directory (contains the galaxy summary table).

    Returns
    -------
    pd.DataFrame
        One row per galaxy with columns:
        ``galaxy``, ``Mgas`` (1e9 Msun), ``logMbar`` (log10 Msun),
        ``Rdisk`` (kpc).  Columns that cannot be computed (e.g. ``Rdisk``
        or ``MHI`` absent) are filled with NaN.
    """
    sparc_dir = Path(sparc_dir)
    table_path = _find_galaxy_table(sparc_dir)
    sep = "," if table_path.suffix == ".csv" else r"\s+"
    raw = pd.read_csv(table_path, sep=sep, comment="#")

    if "Galaxy" not in raw.columns:
        raise ValueError(
            f"Galaxy table {table_path} is missing the 'Galaxy' column."
        )

    df = pd.DataFrame()
    df["galaxy"] = raw["Galaxy"]

    # Stellar mass: Mstar [1e9 Msun] = upsilon × L36 [1e9 Lsun]
    if "L36" in raw.columns:
        mstar = UPSILON_DEFAULT * raw["L36"]  # 1e9 Msun
    else:
        mstar = pd.Series(np.nan, index=raw.index)

    # Gas mass: Mgas [1e9 Msun] = 1.33 × MHI
    if "MHI" in raw.columns:
        mgas = HE_CORRECTION * raw["MHI"]    # 1e9 Msun
    else:
        mgas = pd.Series(np.nan, index=raw.index)

    df["Mgas"] = mgas

    # Baryonic mass and log10(Mbar)
    mbar_msun = (mstar.fillna(0.0) + mgas.fillna(0.0)) * 1e9  # Msun
    # Only set logMbar where at least one component is available
    have_mass = mstar.notna() | mgas.notna()
    logMbar = np.where(have_mass & (mbar_msun > 0), np.log10(mbar_msun), np.nan)
    df["logMbar"] = logMbar

    # Disk scale radius
    if "Rdisk" in raw.columns:
        df["Rdisk"] = raw["Rdisk"].values
    else:
        df["Rdisk"] = np.nan

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-galaxy rotation-curve helpers
# ---------------------------------------------------------------------------


def _compute_gbar(r_kpc: np.ndarray, v_gas: np.ndarray,
                  v_disk: np.ndarray, v_bul: np.ndarray,
                  upsilon_disk: float = 1.0) -> np.ndarray:
    """Compute g_bar (baryonic centripetal acceleration) in m/s².

    Uses the quadrature-sum baryonic velocity:
        V_bar² = upsilon_disk × V_disk² + 0.7 × V_bul² + V_gas²

    Parameters
    ----------
    r_kpc : array
        Galactocentric radii (kpc).
    v_gas, v_disk, v_bul : arrays
        Velocity contributions (km/s).
    upsilon_disk : float
        Disk stellar mass-to-light ratio.

    Returns
    -------
    ndarray
        g_bar in m/s² at each radial point.
    """
    vbar2 = (upsilon_disk * v_disk**2
             + 0.7 * v_bul**2
             + v_gas**2)
    vbar2 = np.maximum(vbar2, 0.0)
    r_safe = np.maximum(r_kpc, 1e-10)
    return vbar2 * 1e6 / (r_safe * _KPC_TO_M)


def compute_rotation_stats(
    rc: pd.DataFrame,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    upsilon_disk: float = 1.0,
) -> dict:
    """Compute per-galaxy rotation-curve summary statistics.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data with columns
        ``r``, ``v_obs``, ``v_gas``, ``v_disk``, ``v_bul`` (km/s, kpc).
    a0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Outer-regime threshold: radial points with
        ``g_bar < deep_threshold × a0`` are used for the slope fit.
    upsilon_disk : float
        Disk mass-to-light ratio for g_bar computation.

    Returns
    -------
    dict
        Keys: ``Rmax`` (kpc), ``Vmax`` (km/s), ``slope_tail``
        (log-log β, or NaN if fewer than ``MIN_OUTER_POINTS`` outer points).
    """
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_gas = rc["v_gas"].values if "v_gas" in rc.columns else np.zeros_like(r)
    v_disk = rc["v_disk"].values if "v_disk" in rc.columns else np.zeros_like(r)
    v_bul = rc["v_bul"].values if "v_bul" in rc.columns else np.zeros_like(r)

    Rmax = float(np.max(r)) if len(r) > 0 else float("nan")
    Vmax = float(np.max(v_obs)) if len(v_obs) > 0 else float("nan")

    g_bar = _compute_gbar(r, v_gas, v_disk, v_bul, upsilon_disk)
    g_obs = v_obs**2 * 1e6 / (np.maximum(r, 1e-10) * _KPC_TO_M)

    valid = (g_bar > 0) & (g_obs > 0)
    outer_mask = valid & (g_bar < deep_threshold * a0)

    if outer_mask.sum() < MIN_OUTER_POINTS:
        slope_tail = float("nan")
    else:
        log_gbar = np.log10(g_bar[outer_mask])
        log_gobs = np.log10(g_obs[outer_mask])
        result = linregress(log_gbar, log_gobs)
        slope_tail = float(result.slope)

    return {"Rmax": Rmax, "Vmax": Vmax, "slope_tail": slope_tail}


def _load_rotation_curve(sparc_dir: Path, galaxy: str) -> pd.DataFrame | None:
    """Load a single galaxy rotation curve; return None if not found."""
    filename = f"{galaxy}_rotmod.dat"
    for prefix in (sparc_dir, sparc_dir / "raw"):
        candidate = prefix / filename
        if candidate.exists():
            df = pd.read_csv(
                candidate,
                sep=r"\s+",
                comment="#",
                names=["r", "v_obs", "v_obs_err", "v_gas",
                       "v_disk", "v_bul", "SBdisk", "SBbul"],
            )
            return df[["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]]
    return None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_signal_table(
    sparc_dir: str | Path,
    out: str | Path,
    env_csv: str | Path | None = None,
    upsilon: float = UPSILON_DEFAULT,
    beta_ref: float = BETA_REF,
    a0: float = A0_DEFAULT,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the per-galaxy physical signal table and write it to *out*.

    Parameters
    ----------
    sparc_dir : str or Path
        Root SPARC directory.
    out : str or Path
        Output CSV path.
    env_csv : str or Path or None
        Optional CSV with columns ``galaxy`` and ``env_proxy``.
    upsilon : float
        Stellar mass-to-light ratio at 3.6 μm (solar units) used for
        ``logMbar`` computation.  Does not affect ``slope_tail``.
    beta_ref : float
        Deep-regime reference slope for ``delta_f3 = slope_tail - beta_ref``.
    a0 : float
        Characteristic acceleration (m/s²) for the outer-regime cut.
    deep_threshold : float
        Threshold fraction of *a0* defining the outer regime.
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Signal table with ``OUTPUT_COLUMNS``.
    """
    sparc_dir = Path(sparc_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # -- galaxy properties from SPARC table --
    props = load_sparc_properties(sparc_dir)

    # Recalculate logMbar with caller-supplied upsilon (default in
    # load_sparc_properties uses UPSILON_DEFAULT; recompute only if different)
    if upsilon != UPSILON_DEFAULT:
        table_path = _find_galaxy_table(sparc_dir)
        sep = "," if table_path.suffix == ".csv" else r"\s+"
        raw = pd.read_csv(table_path, sep=sep, comment="#")
        if "L36" in raw.columns:
            mstar = upsilon * raw["L36"]
            mgas = props["Mgas"]
            mbar_msun = (mstar.fillna(0.0) + mgas.fillna(0.0)) * 1e9
            have_mass = mstar.notna() | mgas.notna()
            props["logMbar"] = np.where(
                have_mass & (mbar_msun > 0), np.log10(mbar_msun), np.nan
            )

    # -- optional env_proxy --
    env_df: pd.DataFrame | None = None
    if env_csv is not None:
        env_df = pd.read_csv(env_csv)
        if "galaxy" not in env_df.columns or "env_proxy" not in env_df.columns:
            raise ValueError(
                f"env_csv must have 'galaxy' and 'env_proxy' columns; "
                f"found: {list(env_df.columns)}"
            )

    # -- per-galaxy rotation-curve stats --
    records = []
    for _, row in props.iterrows():
        name = row["galaxy"]
        rc = _load_rotation_curve(sparc_dir, name)
        if rc is None:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found")
            continue

        stats = compute_rotation_stats(
            rc, a0=a0, deep_threshold=deep_threshold
        )
        record = {
            "galaxy": name,
            "logMbar": row["logMbar"],
            "Mgas": row["Mgas"],
            "Rmax": stats["Rmax"],
            "Vmax": stats["Vmax"],
            "slope_tail": stats["slope_tail"],
            "delta_f3": (stats["slope_tail"] - beta_ref
                         if not np.isnan(stats["slope_tail"]) else np.nan),
            "env_proxy": np.nan,
            "width_kpc": (2.5 * row["Rdisk"]
                          if not np.isnan(row["Rdisk"]) else np.nan),
            "thickness_kpc": (0.1 * row["Rdisk"]
                              if not np.isnan(row["Rdisk"]) else np.nan),
        }
        records.append(record)
        if verbose:
            st = (f"{record['slope_tail']:.3f}"
                  if not np.isnan(record["slope_tail"]) else "NaN")
            print(f"  {name}: slope_tail={st}, Rmax={record['Rmax']:.1f} kpc")

    if not records:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = pd.DataFrame(records)[OUTPUT_COLUMNS]

    # -- merge env_proxy --
    if env_df is not None and not df.empty:
        df = df.drop(columns=["env_proxy"])
        df = df.merge(
            env_df[["galaxy", "env_proxy"]],
            on="galaxy",
            how="left",
        )
        # Restore column order
        df = df[OUTPUT_COLUMNS]

    df = df.sort_values("galaxy").reset_index(drop=True)
    df.to_csv(out, index=False)

    if verbose:
        n_slope = int(df["slope_tail"].notna().sum())
        n_env = int(df["env_proxy"].notna().sum()) if not df.empty else 0
        print(
            f"\nSignal table written to {out}  "
            f"({len(df)} galaxies, {n_slope} with slope_tail, "
            f"{n_env} with env_proxy)"
        )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SCM galaxy physical signal table from SPARC data."
    )
    parser.add_argument(
        "--sparc-dir", default="data/SPARC", metavar="DIR",
        help="Root SPARC directory (default: data/SPARC).",
    )
    parser.add_argument(
        "--out", default="data/galaxy_signal_table.csv",
        help="Output CSV path (default: data/galaxy_signal_table.csv).",
    )
    parser.add_argument(
        "--env-csv", default=None, metavar="CSV",
        help="Optional CSV with columns galaxy, env_proxy.",
    )
    parser.add_argument(
        "--upsilon", type=float, default=UPSILON_DEFAULT,
        help=f"Stellar mass-to-light ratio at 3.6 μm (default: {UPSILON_DEFAULT}).",
    )
    parser.add_argument(
        "--beta-ref", type=float, default=BETA_REF, dest="beta_ref",
        help=f"Deep-regime reference β for delta_f3 (default: {BETA_REF}).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(f"Outer-regime threshold as fraction of a0 "
              f"(default: {DEEP_THRESHOLD_DEFAULT})."),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Parse arguments, run :func:`build_signal_table`, return summary dict."""
    args = _parse_args(argv)
    df = build_signal_table(
        sparc_dir=args.sparc_dir,
        out=args.out,
        env_csv=args.env_csv,
        upsilon=args.upsilon,
        beta_ref=args.beta_ref,
        a0=args.a0,
        deep_threshold=args.deep_threshold,
        verbose=not args.quiet,
    )
    n_slope = int(df["slope_tail"].notna().sum()) if not df.empty else 0
    n_env = int(df["env_proxy"].notna().sum()) if not df.empty else 0
    return {
        "n_galaxies": len(df),
        "n_slope": n_slope,
        "n_env": n_env,
        "out_path": str(args.out),
        "table": df,
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0)
