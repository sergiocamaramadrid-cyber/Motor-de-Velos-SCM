"""
scripts/análisis_gas.py — Gas-fraction and gas-velocity analysis for the
SCM (Motor de Velos) framework.

For each galaxy the script computes:
  - Gas fraction  f_gas = M_gas / (M_gas + M_star)
  - Log-log regression of gas-velocity contribution vs baryonic mass
  - Statistical summary of the gas-dominated subsample

Usage
-----
    python scripts/análisis_gas.py
    python scripts/análisis_gas.py --catalog data/galaxy_catalog.csv \\
                                    --sparc data/sparc_basic.csv \\
                                    --out results/resultados_gas.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_gas_fraction(sparc: pd.DataFrame) -> pd.DataFrame:
    """Add ``f_gas`` and ``logMgas`` / ``logMstar`` columns to a SPARC table.

    Parameters
    ----------
    sparc : pd.DataFrame
        Must contain ``Mgas_Msun`` and ``Mstar_Msun`` columns.

    Returns
    -------
    pd.DataFrame with additional columns: f_gas, logMgas, logMstar.
    """
    sparc = sparc.copy()
    sparc["f_gas"] = sparc["Mgas_Msun"] / (
        sparc["Mgas_Msun"] + sparc["Mstar_Msun"]
    )
    sparc["logMgas"] = np.log10(np.maximum(sparc["Mgas_Msun"], 1.0))
    sparc["logMstar"] = np.log10(np.maximum(sparc["Mstar_Msun"], 1.0))
    return sparc


def fit_gas_velocity_relation(
    sparc: pd.DataFrame,
    v_col: str = "V_gas_flat_kms",
    mass_col: str = "logMgas",
) -> dict:
    """Fit log V_gas ∝ slope · log M_gas + intercept via OLS.

    Parameters
    ----------
    sparc : pd.DataFrame
        SPARC table with computed gas fractions; must contain
        ``v_col`` and ``mass_col``.
    v_col : str
        Column with the gas flat-velocity (km/s).
    mass_col : str
        Column with log gas mass.

    Returns
    -------
    dict with keys: slope_gas, intercept_gas, r2_gas, p_valor_gas,
    stderr_gas, n.
    """
    subset = sparc.dropna(subset=[v_col, mass_col])
    subset = subset[subset[v_col] > 0]
    if len(subset) < 3:
        return {
            "slope_gas": float("nan"),
            "intercept_gas": float("nan"),
            "r2_gas": float("nan"),
            "p_valor_gas": float("nan"),
            "stderr_gas": float("nan"),
            "n": len(subset),
        }
    log_v = np.log10(subset[v_col].values)
    mass_x = subset[mass_col].values
    res = _scipy_stats.linregress(mass_x, log_v)
    return {
        "slope_gas": float(res.slope),
        "intercept_gas": float(res.intercept),
        "r2_gas": float(res.rvalue ** 2),
        "p_valor_gas": float(res.pvalue),
        "stderr_gas": float(res.stderr),
        "n": len(subset),
    }


def per_galaxy_gas_fit(
    sparc: pd.DataFrame,
    catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-galaxy gas-analysis results.

    Parameters
    ----------
    sparc : pd.DataFrame
        SPARC basic table (``sparc_basic.csv``) with gas masses and
        flat velocities.  Must contain columns: galaxy, Mstar_Msun,
        Mgas_Msun, V_gas_flat_kms.
    catalog : pd.DataFrame or None
        Galaxy catalog (``galaxy_catalog.csv``); if provided, the
        ``beta`` and ``reliable`` columns are merged in.

    Returns
    -------
    pd.DataFrame  (one row per galaxy)  with columns:
        galaxy, f_gas, logMgas, logMstar, V_gas_flat_kms,
        slope_gas, intercept_gas, p_valor_gas, r2_gas, n_puntos_gas,
        reliable_gas [, beta, reliable].
    """
    sparc = compute_gas_fraction(sparc)

    # Accept either 'V_gas_flat_kms' (preferred) or 'Vflat_kms' as fallback
    v_col_available = "V_gas_flat_kms" if "V_gas_flat_kms" in sparc.columns else "Vflat_kms"
    if v_col_available not in sparc.columns:
        raise ValueError(
            "sparc table must contain 'V_gas_flat_kms' or 'Vflat_kms' column."
        )
    # Normalise column name so downstream functions always see 'V_gas_flat_kms'
    if v_col_available != "V_gas_flat_kms":
        sparc = sparc.copy()
        sparc["V_gas_flat_kms"] = sparc[v_col_available]

    records = []
    for _, row in sparc.iterrows():
        v_val = float(row["V_gas_flat_kms"])
        log_v = np.log10(v_val) if v_val > 0 else np.nan
        records.append({
            "galaxy": row["galaxy"],
            "f_gas": row["f_gas"],
            "logMgas": row["logMgas"],
            "logMstar": row["logMstar"],
            "V_gas_flat_kms": v_val,
            "logV_gas": log_v,
        })
    df = pd.DataFrame(records)

    # Global gas-velocity relation coefficients (same for all galaxies in
    # this summary table; per-galaxy values need multi-point RC data)
    fit = fit_gas_velocity_relation(sparc)
    df["slope_gas"] = fit["slope_gas"]
    df["intercept_gas"] = fit["intercept_gas"]
    df["p_valor_gas"] = fit["p_valor_gas"]
    df["r2_gas"] = fit["r2_gas"]
    df["n_puntos_gas"] = fit["n"]
    df["reliable_gas"] = ~df["logMgas"].isna() & ~df["logV_gas"].isna()

    if catalog is not None:
        keep = [c for c in ["galaxy", "beta", "reliable"] if c in catalog.columns]
        df = df.merge(catalog[keep], on="galaxy", how="left")

    return df


def summarise_gas_analysis(df: pd.DataFrame) -> dict:
    """Return a high-level summary of the gas analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`per_galaxy_gas_fit`.

    Returns
    -------
    dict with keys: n_total, f_gas_mean, f_gas_median, f_gas_std,
    n_gas_dominated (f_gas > 0.5), slope_gas, r2_gas, p_valor_gas.
    """
    n_total = len(df)
    f_gas = df["f_gas"].dropna()
    n_gas_dom = int((f_gas > 0.5).sum())
    return {
        "n_total": n_total,
        "f_gas_mean": float(f_gas.mean()),
        "f_gas_median": float(f_gas.median()),
        "f_gas_std": float(f_gas.std()),
        "n_gas_dominated": n_gas_dom,
        "slope_gas": float(df["slope_gas"].iloc[0]) if n_total > 0 else float("nan"),
        "r2_gas": float(df["r2_gas"].iloc[0]) if n_total > 0 else float("nan"),
        "p_valor_gas": float(df["p_valor_gas"].iloc[0]) if n_total > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gas-fraction and gas-velocity analysis for the SCM framework."
    )
    parser.add_argument(
        "--sparc",
        default="data/sparc_basic.csv",
        help="Path to sparc_basic.csv (default: data/sparc_basic.csv).",
    )
    parser.add_argument(
        "--catalog",
        default="data/galaxy_catalog.csv",
        help="Path to galaxy_catalog.csv (default: data/galaxy_catalog.csv).",
    )
    parser.add_argument(
        "--out",
        default="results/resultados_gas.csv",
        help="Output CSV path (default: results/resultados_gas.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run gas analysis and write results.

    Returns
    -------
    dict with keys: df, summary, out_path.
    """
    args = _parse_args(argv)
    sparc_path = Path(args.sparc)
    catalog_path = Path(args.catalog)
    out_path = Path(args.out)

    sparc = pd.read_csv(sparc_path)
    catalog = pd.read_csv(catalog_path) if catalog_path.exists() else None

    df = per_galaxy_gas_fit(sparc, catalog)
    summary = summarise_gas_analysis(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("=" * 60)
    print("  SCM — Análisis de Gas")
    print("=" * 60)
    print(f"  N galaxias          : {summary['n_total']}")
    print(f"  f_gas media         : {summary['f_gas_mean']:.3f}")
    print(f"  f_gas mediana       : {summary['f_gas_median']:.3f}")
    print(f"  Dominadas por gas   : {summary['n_gas_dominated']}")
    print(f"  Pendiente V_gas     : {summary['slope_gas']:.4f}")
    print(f"  R²                  : {summary['r2_gas']:.4f}")
    print(f"  p-valor             : {summary['p_valor_gas']:.2e}")
    print(f"  Resultados escritos : {out_path}")
    print("=" * 60)

    return {"df": df, "summary": summary, "out_path": str(out_path)}


if __name__ == "__main__":
    main()
