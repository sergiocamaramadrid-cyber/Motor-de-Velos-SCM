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
    """
    args = _parse_args(argv)

    catalog = load_catalog(args.catalog)
    acc = load_acc(args.acc)

    # Print shapes and columns — mirrors the exploratory pattern from the
    # original notebook (pd.DataFrame.shape / .columns) so callers can
    # quickly verify the loaded data.
    print(catalog.shape, acc.shape)
    print(list(catalog.columns))
    print(list(acc.columns))

    merged = merge_catalog_acc(catalog, acc)
    bins = compute_rar_mass_bins(merged, n_bins=args.n_bins)

    report_lines = format_report(
        catalog, acc, merged, bins, args.catalog, args.acc
    )
    for line in report_lines:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_dir / "merged.csv", index=False)
        (out_dir / "report.txt").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        pd.DataFrame(bins).to_csv(out_dir / "mass_bins.csv", index=False)
        print(f"\n  Results written to {out_dir}")

    return {
        "catalog_shape": tuple(catalog.shape),
        "acc_shape": tuple(acc.shape),
        "merged_shape": tuple(merged.shape),
        "catalog_columns": list(catalog.columns),
        "acc_columns": list(acc.columns),
        "n_galaxies": int(merged["galaxy"].nunique()),
        "bins": bins,
    }


if __name__ == "__main__":
    main()
