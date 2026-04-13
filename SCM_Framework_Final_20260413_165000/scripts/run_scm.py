"""
scripts/run_scm.py — SCM (Motor de Velos) end-to-end pipeline entry point.

This is the **primary** script for the SCM Framework Final release
(2026-04-13).  It orchestrates the full analysis pipeline:

  1. Load the galaxy catalog (data/galaxy_catalog.csv).
  2. Compute the deep-regime slope β distribution statistics.
  3. Fit the SCM relation: log g_obs = β · log g_bar + C.
  4. Correlate β with the environmental proxy.
  5. Write final coefficients to results/coeficientes_finales.csv.
  6. Produce the β-distribution ("figura campanada") figure.

Usage
-----
    python scripts/run_scm.py
    python scripts/run_scm.py --catalog data/galaxy_catalog.csv \\
                               --out-dir results/ \\
                               --no-fig
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

BETA_MOND = 0.5        # expected deep-regime slope (MOND / Motor de Velos)
ALPHA_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load and validate the galaxy catalog CSV.

    Parameters
    ----------
    catalog_path : str or Path
        Path to ``galaxy_catalog.csv``.

    Returns
    -------
    pd.DataFrame with columns: galaxy, logM, logVobs, log_gbar, beta,
    beta_err, n_deep, reliable, env_proxy.
    """
    df = pd.read_csv(catalog_path)
    required = {"galaxy", "logM", "logVobs", "log_gbar", "beta",
                "beta_err", "n_deep", "reliable", "env_proxy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")
    return df


def compute_beta_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics of the β distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog; must contain columns ``beta``, ``beta_err``,
        ``reliable``.

    Returns
    -------
    dict with keys: n_total, n_reliable, beta_mean, beta_median,
    beta_std, delta_mond, t_stat, p_valor, consistent_mond.
    """
    reliable = df[df["reliable"]]["beta"].dropna()
    n_total = len(df)
    n_reliable = len(reliable)

    if n_reliable == 0:
        return {
            "n_total": n_total,
            "n_reliable": 0,
            "beta_mean": float("nan"),
            "beta_median": float("nan"),
            "beta_std": float("nan"),
            "delta_mond": float("nan"),
            "t_stat": float("nan"),
            "p_valor": float("nan"),
            "consistent_mond": False,
        }

    beta_mean = float(reliable.mean())
    beta_median = float(reliable.median())
    beta_std = float(reliable.std())
    delta_mond = beta_mean - BETA_MOND

    if n_reliable >= 2:
        t_result = _scipy_stats.ttest_1samp(reliable.values, BETA_MOND)
        t_stat = float(t_result.statistic)
        p_valor = float(t_result.pvalue)
    else:
        t_stat = float("nan")
        p_valor = float("nan")

    return {
        "n_total": n_total,
        "n_reliable": n_reliable,
        "beta_mean": beta_mean,
        "beta_median": beta_median,
        "beta_std": beta_std,
        "delta_mond": delta_mond,
        "t_stat": t_stat,
        "p_valor": p_valor,
        "consistent_mond": (p_valor > ALPHA_THRESHOLD)
        if not np.isnan(p_valor) else False,
    }


def fit_scm_relation(df: pd.DataFrame) -> dict:
    """Fit the SCM relation log g_obs = β · log g_bar + C.

    Uses all galaxies with reliable β and finite log_gbar.

    Returns
    -------
    dict with keys: slope, intercept, r_value, p_value, stderr, n.
    """
    subset = df[df["reliable"]].dropna(subset=["log_gbar", "logVobs"])
    if len(subset) < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "stderr": float("nan"),
            "n": len(subset),
        }
    result = _scipy_stats.linregress(subset["log_gbar"], subset["logVobs"])
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_value": float(result.rvalue),
        "p_value": float(result.pvalue),
        "stderr": float(result.stderr),
        "n": len(subset),
    }


def compute_env_correlation(df: pd.DataFrame) -> dict:
    """Compute Spearman ρ between env_proxy and β.

    Returns
    -------
    dict with keys: spearman_rho, p_valor_spearman, n.
    """
    subset = df[df["reliable"]].dropna(subset=["env_proxy", "beta"])
    if len(subset) < 4:
        return {
            "spearman_rho": float("nan"),
            "p_valor_spearman": float("nan"),
            "n": len(subset),
        }
    rho, pval = _scipy_stats.spearmanr(subset["env_proxy"], subset["beta"])
    return {
        "spearman_rho": float(rho),
        "p_valor_spearman": float(pval),
        "n": len(subset),
    }


def build_coeficientes_csv(
    beta_stats: dict,
    scm_fit: dict,
    env_corr: dict,
    out_path: str | Path,
) -> pd.DataFrame:
    """Write coeficientes_finales.csv from the analysis results.

    Returns
    -------
    pd.DataFrame with one row per parameter.
    """
    rows = [
        {"parametro": "beta_medio",
         "valor": beta_stats["beta_mean"],
         "error": beta_stats["beta_std"] / np.sqrt(max(beta_stats["n_reliable"], 1)),
         "unidades": "adimensional",
         "descripcion": "Pendiente media del regimen profundo (F3)"},
        {"parametro": "beta_mediana",
         "valor": beta_stats["beta_median"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "Mediana de beta en galaxias fiables"},
        {"parametro": "beta_std",
         "valor": beta_stats["beta_std"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "Desviacion tipica de beta"},
        {"parametro": "delta_mond",
         "valor": beta_stats["delta_mond"],
         "error": beta_stats["beta_std"] / np.sqrt(max(beta_stats["n_reliable"], 1)),
         "unidades": "adimensional",
         "descripcion": "Diferencia respecto a beta_MOND=0.5"},
        {"parametro": "t_stat",
         "valor": beta_stats["t_stat"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "Estadistico t (H0: beta=0.5)"},
        {"parametro": "p_valor",
         "valor": beta_stats["p_valor"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "p-valor bilateral vs beta=0.5"},
        {"parametro": "N_fiables",
         "valor": beta_stats["n_reliable"],
         "error": float("nan"),
         "unidades": "galaxias",
         "descripcion": "Numero de galaxias con beta fiable"},
        {"parametro": "N_total",
         "valor": beta_stats["n_total"],
         "error": float("nan"),
         "unidades": "galaxias",
         "descripcion": "Total en catalogo"},
        {"parametro": "scm_slope",
         "valor": scm_fit["slope"],
         "error": scm_fit["stderr"],
         "unidades": "adimensional",
         "descripcion": "Pendiente ajuste SCM (log g_obs vs log g_bar)"},
        {"parametro": "scm_intercept",
         "valor": scm_fit["intercept"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "Ordenada en origen ajuste SCM"},
        {"parametro": "scm_r_value",
         "valor": scm_fit["r_value"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "Coeficiente de correlacion ajuste SCM"},
        {"parametro": "spearman_rho_env",
         "valor": env_corr["spearman_rho"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "Spearman rho (env_proxy vs beta)"},
        {"parametro": "p_valor_spearman_env",
         "valor": env_corr["p_valor_spearman"],
         "error": float("nan"),
         "unidades": "adimensional",
         "descripcion": "p-valor correlacion entorno-beta"},
    ]
    df_out = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return df_out


def generate_figura_campanada(
    df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Generate the β-distribution ('figura campanada') histogram + KDE.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog with columns ``beta`` and ``reliable``.
    out_path : str or Path
        Output PNG path.

    Returns
    -------
    Path to the saved figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError:
        print("WARNING: matplotlib not available; skipping figure.")
        return Path(out_path)

    reliable_betas = df[df["reliable"]]["beta"].dropna().values
    if len(reliable_betas) < 3:
        print("WARNING: insufficient data for figure.")
        return Path(out_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        reliable_betas, bins=18, density=True,
        color="#4472C4", alpha=0.75, edgecolor="white", linewidth=0.6,
        label=f"Galaxias (N={len(reliable_betas)})",
    )

    xs = np.linspace(reliable_betas.min() - 0.02,
                     reliable_betas.max() + 0.02, 300)
    kde = gaussian_kde(reliable_betas, bw_method=0.4)
    ax.plot(xs, kde(xs), color="#C00000", linewidth=2.0, label="KDE")

    ax.axvline(
        BETA_MOND, color="#70AD47", linewidth=1.8, linestyle="--",
        label=r"$\beta_{\mathrm{MOND}}=0.5$",
    )
    mean_beta = float(reliable_betas.mean())
    ax.axvline(
        mean_beta, color="#ED7D31", linewidth=1.8, linestyle=":",
        label=fr"$\bar{{\beta}}={mean_beta:.3f}$",
    )

    ax.set_xlabel(r"$\beta$ (pendiente régimen profundo)", fontsize=13)
    ax.set_ylabel("Densidad", fontsize=13)
    ax.set_title(
        "Distribución de la pendiente SCM — Figura Campanada",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCM Motor de Velos — end-to-end analysis pipeline. "
            "Reads data/galaxy_catalog.csv and writes results/."
        )
    )
    parser.add_argument(
        "--catalog",
        default="data/galaxy_catalog.csv",
        help="Path to galaxy_catalog.csv (default: data/galaxy_catalog.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        dest="out_dir",
        help="Output directory for results (default: results/).",
    )
    parser.add_argument(
        "--no-fig",
        action="store_true",
        dest="no_fig",
        help="Skip figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the full SCM pipeline.

    Returns
    -------
    dict with keys: catalog, beta_stats, scm_fit, env_corr, coef_path,
    figura_path.
    """
    args = _parse_args(argv)
    catalog_path = Path(args.catalog)
    out_dir = Path(args.out_dir)

    print("=" * 60)
    print("  SCM — Motor de Velos: Pipeline Final 20260413")
    print("=" * 60)

    df = load_catalog(catalog_path)
    print(f"\n[1] Catalogo cargado: {len(df)} galaxias desde {catalog_path}")

    beta_stats = compute_beta_statistics(df)
    print(f"[2] Estadisticas β:"
          f"  N_fiables={beta_stats['n_reliable']}"
          f"  β̄={beta_stats['beta_mean']:.4f}"
          f"  σ={beta_stats['beta_std']:.4f}"
          f"  p={beta_stats['p_valor']:.3f}")

    scm_fit = fit_scm_relation(df)
    print(f"[3] Ajuste SCM: slope={scm_fit['slope']:.4f}  "
          f"intercept={scm_fit['intercept']:.4f}  "
          f"r={scm_fit['r_value']:.4f}")

    env_corr = compute_env_correlation(df)
    print(f"[4] Correlacion entorno: ρ={env_corr['spearman_rho']:.4f}  "
          f"p={env_corr['p_valor_spearman']:.2e}")

    coef_path = out_dir / "coeficientes_finales.csv"
    build_coeficientes_csv(beta_stats, scm_fit, env_corr, coef_path)
    print(f"[5] Coeficientes escritos en {coef_path}")

    figura_path = out_dir / "figura_campanada.png"
    if not args.no_fig:
        generate_figura_campanada(df, figura_path)
        print(f"[6] Figura campanada guardada en {figura_path}")
    else:
        figura_path = None

    print("\n" + "=" * 60)
    if beta_stats["consistent_mond"]:
        print("  VEREDICTO: Estado A — β consistente con MOND/SCM (β=0.5)")
    else:
        print("  VEREDICTO: Estado B — β desvia de β=0.5")
    print("=" * 60)

    return {
        "catalog": df,
        "beta_stats": beta_stats,
        "scm_fit": scm_fit,
        "env_corr": env_corr,
        "coef_path": str(coef_path),
        "figura_path": str(figura_path) if figura_path else None,
    }


if __name__ == "__main__":
    main()
