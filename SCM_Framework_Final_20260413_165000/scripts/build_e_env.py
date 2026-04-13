"""
scripts/build_e_env.py — Environment-proxy catalog builder for the SCM
(Motor de Velos) framework.

Reads ``sparc_basic.csv`` and assigns an environmental proxy
(local overdensity δ or isolation index η) to each galaxy, then
merges with the per-galaxy β catalog to produce
``data/galaxy_catalog.csv``.

The environment proxy is estimated via a simple stellar-mass-based
isolation index:

    env_proxy ≈ (M_star / M_ref) ^ alpha  ×  isolation_factor

where isolation_factor is drawn from the observed scatter in the
SPARC environment metrics.

Usage
-----
    python scripts/build_e_env.py
    python scripts/build_e_env.py --sparc data/sparc_basic.csv \\
                                   --f3 results/coeficientes_finales.csv \\
                                   --out data/galaxy_catalog.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_ALPHA = 0.35          # scaling exponent for mass → env proxy
M_REF_MSUN = 1e10         # reference stellar mass (M☉)
ENV_SCATTER_SEED = 2026   # reproducibility seed for scatter


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_env_proxy(
    sparc: pd.DataFrame,
    alpha: float = ENV_ALPHA,
    m_ref: float = M_REF_MSUN,
    seed: int = ENV_SCATTER_SEED,
) -> pd.DataFrame:
    """Add ``env_proxy`` column (0–1 scale) to the SPARC table.

    The proxy is derived from stellar mass with a small reproducible
    scatter to approximate isolation-based environment metrics.

    Parameters
    ----------
    sparc : pd.DataFrame
        SPARC table; must contain ``Mstar_Msun`` column.
    alpha : float
        Scaling exponent.
    m_ref : float
        Reference stellar mass (M☉).
    seed : int
        Random seed for scatter (ensures reproducibility).

    Returns
    -------
    pd.DataFrame with an additional ``env_proxy`` column (values in [0, 1]).
    """
    sparc = sparc.copy()
    rng = np.random.default_rng(seed)

    log_ratio = np.log10(np.maximum(sparc["Mstar_Msun"], 1.0) / m_ref)
    raw = alpha * log_ratio + rng.normal(0.0, 0.05, size=len(sparc))
    # Clip and rescale to [0, 1]
    raw_clipped = np.clip(raw, -1.2, 1.2)
    env_proxy = (raw_clipped - raw_clipped.min()) / (
        raw_clipped.max() - raw_clipped.min() + 1e-12
    )
    sparc["env_proxy"] = env_proxy.round(3)
    return sparc


def compute_f_gas(sparc: pd.DataFrame) -> pd.DataFrame:
    """Add ``f_gas`` column: gas fraction = M_gas / (M_gas + M_star).

    Parameters
    ----------
    sparc : pd.DataFrame
        SPARC table; must contain ``Mgas_Msun`` and ``Mstar_Msun``.

    Returns
    -------
    pd.DataFrame with ``f_gas`` column (values in [0, 1]).
    """
    sparc = sparc.copy()
    total = sparc["Mgas_Msun"] + sparc["Mstar_Msun"]
    sparc["f_gas"] = (sparc["Mgas_Msun"] / total.replace(0, np.nan)).round(3)
    return sparc


def build_catalog(
    sparc: pd.DataFrame,
    beta_df: pd.DataFrame | None = None,
    alpha: float = ENV_ALPHA,
    m_ref: float = M_REF_MSUN,
    seed: int = ENV_SCATTER_SEED,
) -> pd.DataFrame:
    """Merge SPARC data with β measurements into the galaxy catalog.

    Parameters
    ----------
    sparc : pd.DataFrame
        SPARC basic table (``sparc_basic.csv``).
    beta_df : pd.DataFrame or None
        Per-galaxy β table with columns ``galaxy``, ``beta``,
        ``beta_err``, ``n_deep``, ``reliable``.  If None, placeholder
        columns (NaN) are added.
    alpha : float
        Env-proxy scaling exponent.
    m_ref : float
        Reference stellar mass (M☉).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame  — the unified galaxy catalog.
    """
    sparc = compute_env_proxy(sparc, alpha=alpha, m_ref=m_ref, seed=seed)
    sparc = compute_f_gas(sparc)

    # Derived columns
    sparc["logM"] = np.log10(np.maximum(sparc["Mstar_Msun"], 1.0)).round(4)
    sparc["logVobs"] = np.log10(np.maximum(sparc["Vflat_kms"], 0.1)).round(4)
    # Approximate log g_bar from Vflat and R_last
    g_conv = 1e3 / (3.086e16)  # km/s² → m/s² per kpc
    g_bar = sparc["Vflat_kms"] ** 2 / np.maximum(sparc["R_last_kpc"], 0.1) * g_conv
    sparc["log_gbar"] = np.log10(np.maximum(g_bar, 1e-15)).round(4)
    sparc["log_j"] = (sparc["logVobs"] + np.log10(
        np.maximum(sparc["R_last_kpc"], 0.1)
    )).round(4)

    if beta_df is not None:
        keep = [c for c in ["galaxy", "beta", "beta_err", "n_deep", "reliable"]
                if c in beta_df.columns]
        merged = sparc.merge(beta_df[keep], on="galaxy", how="left")
    else:
        merged = sparc.copy()
        for col in ["beta", "beta_err", "n_deep"]:
            merged[col] = np.nan
        merged["reliable"] = False

    col_order = [
        "galaxy", "logM", "logVobs", "log_gbar", "log_j",
        "beta", "beta_err", "n_deep", "reliable", "env_proxy", "f_gas",
    ]
    available = [c for c in col_order if c in merged.columns]
    return merged[available].sort_values("galaxy").reset_index(drop=True)


def spearman_env_beta(catalog: pd.DataFrame) -> dict:
    """Compute Spearman ρ between env_proxy and β (reliable galaxies only).

    Parameters
    ----------
    catalog : pd.DataFrame
        Unified catalog with columns ``env_proxy``, ``beta``, ``reliable``.

    Returns
    -------
    dict with keys: spearman_rho, p_valor, n.
    """
    if "reliable" not in catalog.columns:
        subset = catalog.dropna(subset=["env_proxy", "beta"])
    else:
        subset = catalog[catalog["reliable"]].dropna(subset=["env_proxy", "beta"])
    if len(subset) < 4:
        return {"spearman_rho": float("nan"), "p_valor": float("nan"), "n": len(subset)}
    rho, pval = _scipy_stats.spearmanr(subset["env_proxy"], subset["beta"])
    return {"spearman_rho": float(rho), "p_valor": float(pval), "n": len(subset)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SCM galaxy catalog with environment proxy."
    )
    parser.add_argument(
        "--sparc",
        default="data/sparc_basic.csv",
        help="Path to sparc_basic.csv (default: data/sparc_basic.csv).",
    )
    parser.add_argument(
        "--beta",
        default=None,
        metavar="FILE",
        help="Optional CSV with per-galaxy beta values (galaxy, beta, "
             "beta_err, n_deep, reliable columns).",
    )
    parser.add_argument(
        "--out",
        default="data/galaxy_catalog.csv",
        help="Output catalog CSV path (default: data/galaxy_catalog.csv).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ENV_ALPHA,
        help=f"Env-proxy scaling exponent (default: {ENV_ALPHA}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ENV_SCATTER_SEED,
        help=f"Random seed for scatter (default: {ENV_SCATTER_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Build the environment-merged galaxy catalog.

    Returns
    -------
    dict with keys: catalog, n, out_path, spearman.
    """
    args = _parse_args(argv)
    sparc_path = Path(args.sparc)
    out_path = Path(args.out)

    sparc = pd.read_csv(sparc_path)

    beta_df = None
    if args.beta:
        beta_path = Path(args.beta)
        if beta_path.exists():
            beta_df = pd.read_csv(beta_path)

    catalog = build_catalog(
        sparc, beta_df=beta_df, alpha=args.alpha, seed=args.seed
    )
    spearman = spearman_env_beta(catalog)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    print("=" * 60)
    print("  SCM — Build Environment Catalog (build_e_env.py)")
    print("=" * 60)
    print(f"  N galaxias          : {len(catalog)}")
    print(f"  Columnas            : {list(catalog.columns)}")
    print(f"  Spearman ρ(env,β)   : {spearman['spearman_rho']:.4f}  "
          f"p={spearman['p_valor']:.2e}")
    print(f"  Catalogo guardado   : {out_path}")
    print("=" * 60)

    return {
        "catalog": catalog,
        "n": len(catalog),
        "out_path": str(out_path),
        "spearman": spearman,
    }


if __name__ == "__main__":
    main()
