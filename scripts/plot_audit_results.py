#!/usr/bin/env python3
"""
scripts/plot_audit_results.py — Visualisations for the Motor de Velos SCM audit.

Reads the 5 CSV/JSON artefacts produced by ``scripts/audit_scm.py`` and writes
six publication-ready figures to the output directory.

Figures generated
-----------------
boxplot_rmse_folds.png          — RMSE by fold for all three models (boxplot)
hist_delta_rmse.png             — Histogram of ΔRMSE (full SCM − BTFR) per galaxy
scatter_delta_vs_logM.png       — ΔRMSE vs log baryonic mass (scatter)
hist_permutation.png            — Permutation null distribution vs real RMSE
bar_coefficient_stability.png   — Coefficient stability across GroupKFold folds
scatter_resid_vs_threshold.png  — BTFR residuals vs (log g₀ − log g_bar) (hinge effect)

Usage
-----
    python scripts/plot_audit_results.py \\
        --indir  results/audit/oos_audit \\
        --outdir results/audit/figures \\
        --global-csv results/audit/sparc_global.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("paper", font_scale=1.5)

_FIG_DPI = 150
_FIGSIZE = (10, 6)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_fold_boxplot(fold_df: pd.DataFrame, outpath: Path) -> None:
    """Boxplot comparing RMSE by fold for the three models."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    melt = fold_df.melt(
        id_vars=["fold"],
        value_vars=["rmse_btfr", "rmse_no_hinge", "rmse_full"],
        var_name="Modelo",
        value_name="RMSE",
    )
    # Friendly labels
    melt["Modelo"] = melt["Modelo"].map({
        "rmse_btfr": "BTFR",
        "rmse_no_hinge": "Sin hinge",
        "rmse_full": "SCM completo",
    })
    sns.boxplot(data=melt, x="fold", y="RMSE", hue="Modelo", ax=ax)
    ax.set_title("Comparación de RMSE por fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (log)")
    ax.legend(title="Modelo")
    plt.tight_layout()
    plt.savefig(outpath, dpi=_FIG_DPI)
    plt.close()
    print(f"  [OK] {outpath}")


def plot_delta_histogram(gal_df: pd.DataFrame, outpath: Path) -> None:
    """Histogram of ΔRMSE per galaxy (full SCM − BTFR)."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    delta = gal_df["rmse_full"] - gal_df["rmse_btfr"]
    ax.hist(delta, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Mejora cero")
    ax.axvline(
        delta.median(), color="darkblue", linestyle="-", linewidth=2,
        label=f"Mediana = {delta.median():.3f}",
    )
    ax.set_xlabel("ΔRMSE (SCM − BTFR)")
    ax.set_ylabel("Número de galaxias")
    ax.set_title("Distribución de la mejora del modelo SCM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=_FIG_DPI)
    plt.close()
    print(f"  [OK] {outpath}")


def plot_delta_vs_logM(
    gal_df: pd.DataFrame,
    global_df: pd.DataFrame,
    outpath: Path,
) -> None:
    """Scatter of ΔRMSE vs log baryonic mass."""
    merged = pd.merge(gal_df, global_df[["galaxy_id", "logM"]], on="galaxy_id", how="left")
    delta = merged["rmse_full"] - merged["rmse_btfr"]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(merged["logM"], delta, alpha=0.6, edgecolor="k", s=50)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("log Masa bariónica (M☉)")
    ax.set_ylabel("ΔRMSE (SCM − BTFR)")
    ax.set_title("Dependencia de la mejora con la masa")
    plt.tight_layout()
    plt.savefig(outpath, dpi=_FIG_DPI)
    plt.close()
    print(f"  [OK] {outpath}")


def plot_permutation_distribution(
    perm_df: pd.DataFrame,
    real_rmse: float,
    outpath: Path,
) -> None:
    """Histogram of permutation RMSE vs real model RMSE."""
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(
        perm_df["perm_rmse"], bins=30, edgecolor="black",
        alpha=0.7, color="lightcoral",
    )
    ax.axvline(
        real_rmse, color="darkblue", linestyle="-", linewidth=3,
        label=f"RMSE real = {real_rmse:.4f}",
    )
    ax.axvline(
        perm_df["perm_rmse"].mean(), color="gray", linestyle="--", linewidth=2,
        label="Media permutaciones",
    )
    ax.set_xlabel("RMSE medio bajo permutación")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Permutation test: distribución nula vs valor real")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=_FIG_DPI)
    plt.close()
    print(f"  [OK] {outpath}")


def plot_coefficient_stability(coeff_df: pd.DataFrame, outpath: Path) -> None:
    """Bar chart of coefficient means ± std across GroupKFold folds."""
    params = ["a", "b", "d", "logg0"]
    # Only plot params that exist in the dataframe
    params = [p for p in params if p in coeff_df.columns]
    means = coeff_df[params].mean()
    stds = coeff_df[params].std().fillna(0)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    x = np.arange(len(params))
    ax.bar(x, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("Valor del coeficiente")
    ax.set_title("Estabilidad de coeficientes entre folds")
    plt.tight_layout()
    plt.savefig(outpath, dpi=_FIG_DPI)
    plt.close()
    print(f"  [OK] {outpath}")


def plot_residual_vs_gbar_threshold(
    gal_df: pd.DataFrame,
    global_df: pd.DataFrame,
    logg0: float,
    outpath: Path,
) -> None:
    """Scatter of BTFR residuals vs (log g₀ − log g_bar) to show the hinge effect."""
    merged = pd.merge(
        gal_df, global_df[["galaxy_id", "log_gbar", "logM"]], on="galaxy_id", how="left"
    )
    col = "resid_btfr" if "resid_btfr" in merged.columns else "rmse_btfr"
    resid = merged[col]
    x = logg0 - merged["log_gbar"]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(x, resid, alpha=0.6, edgecolor="k", s=50)
    ax.axhline(0, color="red", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--", label=f"Umbral g₀ = {logg0:.2f}")
    ax.set_xlabel("log g₀ − log g_bar")
    ax.set_ylabel("Residuo BTFR (log)")
    ax.set_title("Efecto del hinge: residuos vs proximidad al umbral")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=_FIG_DPI)
    plt.close()
    print(f"  [OK] {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate audit figures from audit_scm.py output."
    )
    parser.add_argument(
        "--indir", default="results/audit/oos_audit",
        help="Directory with audit_scm.py artefacts. "
             "Default: results/audit/oos_audit",
    )
    parser.add_argument(
        "--outdir", default="results/audit/figures",
        help="Output directory for figures. Default: results/audit/figures",
    )
    parser.add_argument(
        "--global-csv", default="results/audit/sparc_global.csv",
        dest="global_csv",
        help="Per-galaxy global CSV (for scatter plots). "
             "Default: results/audit/sparc_global.csv",
    )
    args = parser.parse_args(argv)

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------------
    fold_csv = indir / "groupkfold_metrics.csv"
    gal_csv = indir / "gal_results.csv"
    if not fold_csv.exists():
        print(f"ERROR: {fold_csv} not found.  Run audit_scm.py first.")
        return
    if not gal_csv.exists():
        print(f"ERROR: {gal_csv} not found.  Run audit_scm.py first.")
        return

    fold_df = pd.read_csv(fold_csv)
    gal_df = pd.read_csv(gal_csv)

    # ------------------------------------------------------------------
    # Optional inputs
    # ------------------------------------------------------------------
    global_df = None
    if Path(args.global_csv).exists():
        global_df = pd.read_csv(args.global_csv)
    else:
        print(f"  Advertencia: {args.global_csv} no encontrado — "
              "se omitirán gráficos que requieran logM/log_gbar.")

    perm_csv = indir / "permutation_runs.csv"
    perm_df = pd.read_csv(perm_csv) if perm_csv.exists() else None

    summary_json = indir / "permutation_summary.json"
    perm_summary: dict = {}
    if summary_json.exists():
        with open(summary_json) as f:
            perm_summary = json.load(f)

    coeff_csv = indir / "coeffs_by_fold.csv"
    coeff_df = pd.read_csv(coeff_csv) if coeff_csv.exists() else None

    # logg0: take median across folds if available, else use default
    logg0 = -10.45
    if coeff_df is not None and "logg0" in coeff_df.columns:
        logg0 = float(coeff_df["logg0"].median())

    # real SCM RMSE from fold results
    real_rmse_scm = float(fold_df["rmse_full"].mean())

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print("Generando figuras de auditoría...")

    plot_fold_boxplot(fold_df, outdir / "boxplot_rmse_folds.png")
    plot_delta_histogram(gal_df, outdir / "hist_delta_rmse.png")

    if global_df is not None:
        plot_delta_vs_logM(gal_df, global_df, outdir / "scatter_delta_vs_logM.png")

    if perm_df is not None:
        plot_permutation_distribution(
            perm_df, real_rmse_scm, outdir / "hist_permutation.png"
        )

    if coeff_df is not None:
        plot_coefficient_stability(coeff_df, outdir / "bar_coefficient_stability.png")

    if global_df is not None and coeff_df is not None:
        plot_residual_vs_gbar_threshold(
            gal_df, global_df, logg0, outdir / "scatter_resid_vs_threshold.png"
        )

    print(f"\nFiguras guardadas en {outdir}")


if __name__ == "__main__":
    main()
