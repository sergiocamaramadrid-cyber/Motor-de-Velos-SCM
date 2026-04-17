"""
scripts/build_galaxy_signal_table.py — Galaxy physical signal table.

For each SPARC galaxy, assembles the core kinematic and structural variables
into a single per-galaxy CSV (the "galaxy signal table").

Output columns
--------------
galaxy        — galaxy name
logMbar       — log10(0.5·L36 + 1.33·MHI) in Msun; NaN if both absent
Mgas          — gas mass 1.33 × M_HI in 1e9 Msun; NaN if MHI unavailable
Rmax          — maximum observed radius (kpc)
Vmax          — maximum observed circular velocity (km/s)
slope_tail    — log-log slope of g_obs vs g_bar for r ≥ 0.7·Rmax;
                NaN when fewer than MIN_OUTER_POINTS qualify
delta_f3      — slope_tail − 0.5 (BETA_REF); NaN when slope_tail is NaN
env_proxy     — environmental density proxy; NaN if not provided
width_kpc     — 2.5 × Rdisk (kpc); NaN if Rdisk absent
thickness_kpc — 0.1 × Rdisk (kpc); NaN if Rdisk absent
outer_fit_ok  — True when slope_tail is finite (≥ MIN_OUTER_POINTS points)
n_tail_points — number of radial points used for the slope_tail fit (int)

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
* ``logMbar`` = log10(0.5·L36 + 1.33·MHI) where L36 is in 1e9 Lsun and
  MHI in 1e9 Msun.  Missing components are treated as zero only when the
  other component is present; if both are absent, logMbar is NaN.
* ``slope_tail`` uses ``upsilon_disk = 1.0`` and ``upsilon_bulge = 1.0``
  (fixed) for g_bar computation.  The outer regime is defined by
  r ≥ 0.7·Rmax (purely radial criterion, independent of g_bar).
* ``width_kpc`` and ``thickness_kpc`` use the exponential disk scale radius
  Rdisk from the SPARC catalog.  The approximation thickness ≈ 0.1·Rdisk
  is an order-of-magnitude estimate; see Kregel et al. (2002).
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
UPSILON_DEFAULT: float = 0.5    # 3.6 μm mass-to-light ratio for logMbar
HE_CORRECTION: float = 1.33     # HI → total gas (helium included)
_KPC_TO_M: float = 3.085_677_581e19   # 1 kpc in metres

# Outer regime: r >= OUTER_FRAC * Rmax
OUTER_FRAC: float = 0.7

# Minimum number of outer points needed for a meaningful slope fit
MIN_OUTER_POINTS: int = 4

OUTPUT_COLUMNS = [
    "galaxy", "logMbar", "Mgas", "Rmax", "Vmax",
    "slope_tail", "delta_f3", "env_proxy", "width_kpc", "thickness_kpc",
    "outer_fit_ok", "n_tail_points",
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
                  upsilon_disk: float = 1.0,
                  upsilon_bulge: float = 1.0) -> np.ndarray:
    """Compute g_bar (baryonic centripetal acceleration) in m/s².

    Uses the quadrature-sum baryonic velocity:
        V_bar² = upsilon_disk × V_disk² + upsilon_bulge × V_bul² + V_gas²

    Parameters
    ----------
    r_kpc : array
        Galactocentric radii (kpc).
    v_gas, v_disk, v_bul : arrays
        Velocity contributions (km/s).
    upsilon_disk : float
        Disk stellar mass-to-light ratio (default 1.0).
    upsilon_bulge : float
        Bulge stellar mass-to-light ratio (default 1.0).

    Returns
    -------
    ndarray
        g_bar in m/s² at each radial point.
    """
    vbar2 = (upsilon_disk * v_disk**2
             + upsilon_bulge * v_bul**2
             + v_gas**2)
    vbar2 = np.maximum(vbar2, 0.0)
    r_safe = np.maximum(r_kpc, 1e-10)
    return vbar2 * 1e6 / (r_safe * _KPC_TO_M)


def compute_rotation_stats(
    rc: pd.DataFrame,
    upsilon_disk: float = 1.0,
    upsilon_bulge: float = 1.0,
) -> dict:
    """Compute per-galaxy rotation-curve summary statistics.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data with columns
        ``r``, ``v_obs``, ``v_gas``, ``v_disk``, ``v_bul`` (km/s, kpc).
    upsilon_disk : float
        Disk mass-to-light ratio for g_bar computation (default 1.0).
    upsilon_bulge : float
        Bulge mass-to-light ratio for g_bar computation (default 1.0).

    Returns
    -------
    dict
        Keys: ``Rmax`` (kpc), ``Vmax`` (km/s), ``slope_tail``
        (log-log β of g_obs vs g_bar for r ≥ 0.7·Rmax, or NaN if fewer
        than ``MIN_OUTER_POINTS`` qualify), ``outer_fit_ok`` (bool),
        ``n_tail_points`` (int).
    """
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_gas = rc["v_gas"].values if "v_gas" in rc.columns else np.zeros_like(r)
    v_disk = rc["v_disk"].values if "v_disk" in rc.columns else np.zeros_like(r)
    v_bul = rc["v_bul"].values if "v_bul" in rc.columns else np.zeros_like(r)

    Rmax = float(np.max(r)) if len(r) > 0 else float("nan")
    Vmax = float(np.max(v_obs)) if len(v_obs) > 0 else float("nan")

    g_bar = _compute_gbar(r, v_gas, v_disk, v_bul, upsilon_disk, upsilon_bulge)
    g_obs = v_obs**2 * 1e6 / (np.maximum(r, 1e-10) * _KPC_TO_M)

    valid = (g_bar > 0) & (g_obs > 0)
    outer_mask = valid & (r >= OUTER_FRAC * Rmax)

    n_tail_points = int(outer_mask.sum())
    if n_tail_points < MIN_OUTER_POINTS:
        slope_tail = float("nan")
        outer_fit_ok = False
    else:
        log_gbar = np.log10(g_bar[outer_mask])
        log_gobs = np.log10(g_obs[outer_mask])
        result = linregress(log_gbar, log_gobs)
        slope_tail = float(result.slope)
        outer_fit_ok = True

    return {
        "Rmax": Rmax,
        "Vmax": Vmax,
        "slope_tail": slope_tail,
        "outer_fit_ok": outer_fit_ok,
        "n_tail_points": n_tail_points,
    }


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
        ``logMbar = log10(upsilon·L36 + 1.33·MHI)``.
        Does not affect ``slope_tail`` (which always uses upsilon_disk=1.0).
    beta_ref : float
        Deep-regime reference slope for ``delta_f3 = slope_tail - beta_ref``.
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

        stats = compute_rotation_stats(rc)
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
            "outer_fit_ok": stats["outer_fit_ok"],
            "n_tail_points": stats["n_tail_points"],
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
