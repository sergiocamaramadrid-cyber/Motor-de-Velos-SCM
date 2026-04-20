"""
scripts/catalog_acc_merge.py — Merge galaxy catalog (with environment) and
per-radial-point acceleration data, then report mass-binned RAR statistics.

The script combines two complementary datasets:

  galaxy_catalog_with_env.csv  (galaxy-level)
      Required columns: galaxy, logM, env_proxy
      ``logM`` is renamed to ``logMbar`` on load (baryonic mass in log10 M_sun).

  universal_term_comparison_full.csv  (per-radial-point)
      Required columns: galaxy, r_kpc, g_bar, g_obs, log_g_bar, log_g_obs

The merge is a left join keyed on ``galaxy``.  Points belonging to galaxies not
present in the catalog are silently dropped (logged at verbose level).

Mass-binned statistics
----------------------
The merged table is split into ``n_bins`` equal-width logMbar bins.  For each
bin the script reports:

  n_galaxies     — number of distinct galaxies
  n_points       — total radial points
  logMbar_mean   — mean logMbar of galaxies in the bin
  env_proxy_mean — mean env_proxy
  g_bar_median   — median g_bar (m/s²)
  g_obs_median   — median g_obs (m/s²)
  log_ratio_mean — mean log10(g_obs / g_bar)  [RAR offset]

Usage
-----
Default paths::

    python scripts/catalog_acc_merge.py

Custom paths::

    python scripts/catalog_acc_merge.py \\
        --catalog data/galaxy_catalog_with_env.csv \\
        --acc     results/universal_term_comparison_full.csv \\
        --n-bins  3 \\
        --out     results/catalog_acc_merge

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CATALOG_DEFAULT = "data/galaxy_catalog_with_env.csv"
ACC_DEFAULT = "results/universal_term_comparison_full.csv"
N_BINS_DEFAULT = 3
OUTER_FRAC_DEFAULT = 0.7
MIN_OUTER_POINTS = 2

CATALOG_REQUIRED = {"galaxy", "logM"}
ACC_REQUIRED = {"galaxy", "r_kpc", "g_bar", "g_obs", "log_g_bar", "log_g_obs"}

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load the galaxy catalog and rename ``logM`` to ``logMbar``.

    Parameters
    ----------
    path : str or Path
        Path to the galaxy catalog CSV.
        Must contain columns: galaxy, logM, env_proxy (env_proxy is optional
        but included in all downstream statistics when present).

    Returns
    -------
    pd.DataFrame
        Catalog with column ``logM`` renamed to ``logMbar``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Galaxy catalog not found: {path}\n"
            "Provide a valid --catalog path or place the file at "
            f"{CATALOG_DEFAULT}."
        )
    df = pd.read_csv(path)
    missing = CATALOG_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Galaxy catalog missing required columns: {missing}."
        )
    df = df.rename(columns={"logM": "logMbar"})
    return df


def load_acc(path: str | Path) -> pd.DataFrame:
    """Load the per-radial-point acceleration comparison CSV.

    Parameters
    ----------
    path : str or Path
        Path to ``universal_term_comparison_full.csv`` or equivalent.

    Returns
    -------
    pd.DataFrame
        Acceleration data with columns galaxy, r_kpc, g_bar, g_obs,
        log_g_bar, log_g_obs.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Acceleration CSV not found: {path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )
    df = pd.read_csv(path)
    missing = ACC_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Acceleration CSV missing required columns: {missing}."
        )
    return df


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_catalog_acc(catalog: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """Merge galaxy-level catalog with per-radial-point acceleration data.

    The join is a left join on the acceleration table so that only points
    whose galaxy is present in the catalog are retained.

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with at least columns ``galaxy`` and ``logMbar``.
    acc : pd.DataFrame
        Acceleration table with at least column ``galaxy``.

    Returns
    -------
    pd.DataFrame
        Merged table with all catalog columns appended to every matching
        acceleration row.  Rows with no catalog match are dropped.
    """
    merged = acc.merge(catalog, on="galaxy", how="inner")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mass-binned RAR statistics
# ---------------------------------------------------------------------------

def compute_rar_mass_bins(merged: pd.DataFrame,
                          n_bins: int = N_BINS_DEFAULT) -> list[dict]:
    """Compute mass-binned RAR statistics on the merged table.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged table from :func:`merge_catalog_acc`.  Must contain columns
        ``logMbar``, ``g_bar``, ``g_obs``, ``log_g_bar``, ``log_g_obs``,
        and ``galaxy``.
    n_bins : int
        Number of equal-width logMbar bins (default 3).

    Returns
    -------
    list of dict
        One dict per non-empty bin with keys:
        bin_lo, bin_hi, n_galaxies, n_points, logMbar_mean,
        env_proxy_mean (nan if column absent), g_bar_median, g_obs_median,
        log_ratio_mean.
    """
    if merged.empty:
        return []

    lo = float(merged["logMbar"].min())
    hi = float(merged["logMbar"].max())
    if lo == hi:
        edges = [lo - 0.5, hi + 0.5]
    else:
        edges = list(np.linspace(lo, hi, n_bins + 1))

    bins = []
    for i in range(len(edges) - 1):
        bin_lo = edges[i]
        bin_hi = edges[i + 1]
        if i < len(edges) - 2:
            mask = (merged["logMbar"] >= bin_lo) & (merged["logMbar"] < bin_hi)
        else:
            mask = (merged["logMbar"] >= bin_lo) & (merged["logMbar"] <= bin_hi)
        sub = merged[mask]
        if sub.empty:
            continue

        gal_sub = sub.drop_duplicates("galaxy")
        env_mean = (float(gal_sub["env_proxy"].mean())
                    if "env_proxy" in sub.columns else float("nan"))
        safe_gbar = sub["g_bar"].replace(0, np.nan)
        safe_gobs = sub["g_obs"].replace(0, np.nan)
        log_ratio = np.log10(safe_gobs / safe_gbar)

        bins.append({
            "bin_lo": bin_lo,
            "bin_hi": bin_hi,
            "n_galaxies": int(sub["galaxy"].nunique()),
            "n_points": int(len(sub)),
            "logMbar_mean": float(gal_sub["logMbar"].mean()),
            "env_proxy_mean": env_mean,
            "g_bar_median": float(safe_gbar.median()),
            "g_obs_median": float(safe_gobs.median()),
            "log_ratio_mean": float(log_ratio.mean()),
        })
    return bins


# ---------------------------------------------------------------------------
# Catalog-vs-acceleration overlap diagnostics
# ---------------------------------------------------------------------------

def compute_catalog_overlap(catalog: pd.DataFrame,
                             acc: pd.DataFrame) -> dict:
    """Compute galaxy-name overlap between the catalog and acceleration tables.

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with at least a ``galaxy`` column.
    acc : pd.DataFrame
        Per-radial-point acceleration table with at least a ``galaxy`` column.

    Returns
    -------
    dict with keys:
        n_catalog        — number of distinct galaxies in the catalog
        n_acc            — number of distinct galaxies in the acceleration table
        n_overlap        — galaxies present in both datasets
        n_only_catalog   — galaxies only in the catalog (no acceleration data)
        n_only_acc       — galaxies only in the acceleration table (no catalog)
        overlap          — sorted list of overlapping galaxy names
        only_catalog     — sorted list of catalog-only galaxy names
        only_acc         — sorted list of acc-only galaxy names
    """
    galaxies_cat = set(catalog["galaxy"])
    galaxies_acc = set(acc["galaxy"])

    overlap = galaxies_cat & galaxies_acc
    only_cat = galaxies_cat - galaxies_acc
    only_acc = galaxies_acc - galaxies_cat

    return {
        "n_catalog": len(galaxies_cat),
        "n_acc": len(galaxies_acc),
        "n_overlap": len(overlap),
        "n_only_catalog": len(only_cat),
        "n_only_acc": len(only_acc),
        "overlap": sorted(overlap),
        "only_catalog": sorted(only_cat),
        "only_acc": sorted(only_acc),
    }


# ---------------------------------------------------------------------------
# f_DM outer-tail computation
# ---------------------------------------------------------------------------

def compute_fdm_per_galaxy(acc: pd.DataFrame,
                           r_fraction: float = OUTER_FRAC_DEFAULT) -> pd.DataFrame:
    """Compute mean f_DM in the outer tail (r > r_fraction * r_max) per galaxy.

    The dark-matter fraction is defined as::

        f_DM = 1 - g_bar / g_obs

    where ``g_bar`` and ``g_obs`` are the mean baryonic and observed
    centripetal accelerations over the outer radial points.

    Parameters
    ----------
    acc : pd.DataFrame
        Per-radial-point acceleration table with columns
        ``galaxy``, ``r_kpc``, ``g_bar``, ``g_obs``.
    r_fraction : float
        Fraction of each galaxy's maximum radius that defines the outer tail.
        Points with ``r_kpc > r_fraction * r_max`` are used (default 0.7).

    Returns
    -------
    pd.DataFrame
        One row per galaxy with columns:
        ``galaxy``, ``r_max_kpc``, ``n_outer_points``,
        ``f_DM_out``, ``g_bar_out``, ``g_obs_out``.
        Galaxies with fewer than :data:`MIN_OUTER_POINTS` outer points or
        physically invalid accelerations are omitted.
    """
    results = []

    for galaxy, grp in acc.groupby("galaxy"):
        r_max = grp["r_kpc"].max()
        outer = grp[grp["r_kpc"] > r_fraction * r_max]

        if len(outer) < MIN_OUTER_POINTS:
            continue

        g_bar_out = float(outer["g_bar"].mean())
        g_obs_out = float(outer["g_obs"].mean())

        if g_obs_out <= 0 or g_bar_out < 0:
            continue

        f_DM_out = 1.0 - (g_bar_out / g_obs_out)

        results.append({
            "galaxy": galaxy,
            "r_max_kpc": float(r_max),
            "n_outer_points": int(len(outer)),
            "f_DM_out": f_DM_out,
            "g_bar_out": g_bar_out,
            "g_obs_out": g_obs_out,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(catalog: pd.DataFrame, acc: pd.DataFrame,
                  merged: pd.DataFrame, bins: list[dict],
                  catalog_path: str, acc_path: str) -> list[str]:
    """Format the analysis report as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — Catalog × Acceleration Merge",
        _SEP,
        f"  Catalog  : {catalog_path}",
        f"  Acc CSV  : {acc_path}",
        "",
        f"  Catalog  : shape={catalog.shape}",
        f"  Catalog columns : {list(catalog.columns)}",
        f"  Acc CSV  : shape={acc.shape}",
        f"  Acc columns     : {list(acc.columns)}",
        "",
        f"  Merged   : shape={merged.shape}",
        f"  Galaxies in merge: {merged['galaxy'].nunique()}",
        "",
        "  Mass-binned RAR statistics",
        "  " + "-" * 60,
    ]
    for b in bins:
        lines += [
            f"  logMbar [{b['bin_lo']:.2f}, {b['bin_hi']:.2f}]"
            f"  n_gal={b['n_galaxies']}  n_pts={b['n_points']}"
            f"  env={b['env_proxy_mean']:.3f}"
            f"  log(g_obs/g_bar)={b['log_ratio_mean']:+.4f}",
        ]
    lines.append(_SEP)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge galaxy catalog with per-radial-point acceleration data "
            "and report mass-binned RAR statistics."
        )
    )
    parser.add_argument(
        "--catalog", default=CATALOG_DEFAULT, metavar="CSV",
        help=f"Galaxy catalog CSV with logM and env_proxy (default: {CATALOG_DEFAULT}).",
    )
    parser.add_argument(
        "--acc", default=ACC_DEFAULT, metavar="CSV",
        help=f"Per-radial-point acceleration CSV (default: {ACC_DEFAULT}).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=N_BINS_DEFAULT, dest="n_bins",
        help=f"Number of equal-width logMbar mass bins (default: {N_BINS_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write merged.csv and report.txt to this directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Load, merge, and analyse catalog + acceleration data.

    Returns
    -------
    dict with keys:
        catalog_shape   — (n_rows, n_cols) of the galaxy catalog
        acc_shape       — (n_rows, n_cols) of the acceleration CSV
        merged_shape    — (n_rows, n_cols) of the merged table
        catalog_columns — list of catalog column names (after rename)
        acc_columns     — list of acceleration CSV column names
        n_galaxies      — number of distinct galaxies in the merged table
        bins            — list of mass-bin statistics dicts
        fdm             — DataFrame of per-galaxy outer-tail f_DM values
        overlap         — dict from compute_catalog_overlap
    """
    args = _parse_args(argv)

    catalog = load_catalog(args.catalog)
    acc = load_acc(args.acc)

    # Print shapes and columns — mirrors the exploratory pattern from the
    # original notebook (pd.DataFrame.shape / .columns) so callers can
    # quickly verify the loaded data.
    print(acc.shape)
    print(acc.columns.tolist())
    print(acc.head())

    # Overlap diagnostic
    ov = compute_catalog_overlap(catalog, acc)
    print(f"\nGalaxias en catálogo:          {ov['n_catalog']}")
    print(f"Galaxias en acc:               {ov['n_acc']}")
    print(f"Solapamiento:                  {ov['n_overlap']}")
    print(f"Solo en catálogo (sin f_DM):   {ov['n_only_catalog']}")
    print(f"Solo en acc (sin slope_tail):  {ov['n_only_acc']}")
    if ov["only_catalog"]:
        print(f"Galaxias que perderías: {ov['only_catalog']}")

    merged = merge_catalog_acc(catalog, acc)
    bins = compute_rar_mass_bins(merged, n_bins=args.n_bins)

    report_lines = format_report(
        catalog, acc, merged, bins, args.catalog, args.acc
    )
    for line in report_lines:
        print(line)

    # Compute f_DM in the outer tail per galaxy and merge with catalog
    fdm = compute_fdm_per_galaxy(acc)
    full = catalog.merge(fdm, on="galaxy", how="inner")

    print(f"\nGalaxias con f_DM calculado: {len(full)}")
    if not full.empty:
        cols = [c for c in ["galaxy", "logMbar", "env_proxy", "f_DM_out"]
                if c in full.columns]
        print(full[cols].to_string())

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_dir / "merged.csv", index=False)
        (out_dir / "report.txt").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        pd.DataFrame(bins).to_csv(out_dir / "mass_bins.csv", index=False)
        fdm.to_csv(out_dir / "fdm_per_galaxy.csv", index=False)
        # Write overlap summary as a two-column (key, value) CSV
        ov_rows = [{"key": k, "value": v}
                   for k, v in ov.items() if not isinstance(v, list)]
        pd.DataFrame(ov_rows).to_csv(out_dir / "overlap.csv", index=False)
        print(f"\n  Results written to {out_dir}")

    return {
        "catalog_shape": tuple(catalog.shape),
        "acc_shape": tuple(acc.shape),
        "merged_shape": tuple(merged.shape),
        "catalog_columns": list(catalog.columns),
        "acc_columns": list(acc.columns),
        "n_galaxies": int(merged["galaxy"].nunique()),
        "bins": bins,
        "fdm": fdm,
        "overlap": ov,
    }


if __name__ == "__main__":
    main()
