"""
scripts/spectral_analysis.py — Spectral analysis of SCM velocity profiles.

Theory
------
The Motor de Velos SCM (Fluid Condensation Model) predicts that the 1-D
power spectrum of a galaxy's circular-velocity profile follows a power law:

    P(k) ∝ k^{-α}

where α is the *spectral index* encoding how kinetic energy is distributed
across spatial scales.  In the SCM framework the spectral index is expected to
correlate with the baryonic mass of the host galaxy:

    α ∝ logM^{γ},  γ > 0

and to show weaker but non-zero dependence on the local environment proxy
(overdensity δ).

This script:
  1. Loads the 13-galaxy cleaned spectral catalog.
  2. Computes Spearman rank correlations (ρ, p-value) between α and logM, and
     between α and env_proxy.
  3. Fits an OLS regression: α = a·logM + b, using the ``quality_flag == 1``
     sub-sample.
  4. Generates two diagnostic figures saved to ``../figures/``:
       - spectral_index_vs_logM.png / .pdf
       - spectral_index_histogram.png / .pdf
  5. Writes a machine-readable ``spectral_summary.csv`` to the output directory.

Usage
-----
Default paths::

    python scripts/spectral_analysis.py

Explicit options::

    python scripts/spectral_analysis.py \\
        --csv  data/scm_spectral_clean_13_galaxies.csv \\
        --out  figures

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEP = "=" * 64
CSV_DEFAULT = Path(__file__).parent.parent / "data" / "scm_spectral_clean_13_galaxies.csv"
OUT_DEFAULT = Path(__file__).parent.parent / "figures"
ALPHA_THRESHOLD = 0.05
MIN_RELIABLE = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_catalog(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the spectral catalog CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to the cleaned 13-galaxy spectral CSV.

    Returns
    -------
    pd.DataFrame
        Validated catalog with columns: galaxy, logM, V_max, sigma_v,
        slope_inner, slope_tail, spectral_index, env_proxy, quality_flag.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {csv_path}\n"
            "Expected: SCM-spectral-analysis/data/scm_spectral_clean_13_galaxies.csv"
        )

    df = pd.read_csv(csv_path)
    required = {
        "galaxy", "logM", "V_max", "sigma_v",
        "slope_inner", "slope_tail", "spectral_index", "env_proxy", "quality_flag",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    quality_only: bool = True,
) -> dict:
    """Compute Spearman ρ between two catalog columns.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog.
    x_col, y_col : str
        Column names for the two variables.
    quality_only : bool
        If True restrict to rows where ``quality_flag == 1``.

    Returns
    -------
    dict with keys: x_col, y_col, n, rho, p_value.
    """
    sub = df[df["quality_flag"] == 1] if quality_only else df
    sub = sub[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 2:
        return {"x_col": x_col, "y_col": y_col, "n": n,
                "rho": float("nan"), "p_value": float("nan")}
    rho, p_value = _scipy_stats.spearmanr(sub[x_col].values, sub[y_col].values)
    return {"x_col": x_col, "y_col": y_col, "n": n,
            "rho": float(rho), "p_value": float(p_value)}


def compute_ols(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    quality_only: bool = True,
) -> dict:
    """Fit OLS: y = slope·x + intercept.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog.
    x_col, y_col : str
        Predictor and response column names.
    quality_only : bool
        If True restrict to ``quality_flag == 1`` rows.

    Returns
    -------
    dict with keys: x_col, y_col, n, slope, intercept, stderr, r_value,
    p_value, r_squared.
    """
    sub = df[df["quality_flag"] == 1] if quality_only else df
    sub = sub[[x_col, y_col]].dropna()
    n = len(sub)
    if n < 2:
        nan = float("nan")
        return {"x_col": x_col, "y_col": y_col, "n": n,
                "slope": nan, "intercept": nan, "stderr": nan,
                "r_value": nan, "p_value": nan, "r_squared": nan}
    slope, intercept, r_value, p_value, stderr = linregress(
        sub[x_col].values, sub[y_col].values
    )
    return {
        "x_col": x_col,
        "y_col": y_col,
        "n": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "stderr": float(stderr),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "r_squared": float(r_value ** 2),
    }


def run_analysis(df: pd.DataFrame) -> dict:
    """Run the full spectral analysis on the 13-galaxy catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog as returned by :func:`load_catalog`.

    Returns
    -------
    dict with keys:
        n_galaxies         — total rows
        n_reliable         — rows with quality_flag == 1
        spearman_mass      — Spearman result for spectral_index vs logM
        spearman_env       — Spearman result for spectral_index vs env_proxy
        ols_mass           — OLS result for spectral_index ~ logM
        spectral_index_mean — mean spectral index (reliable)
        spectral_index_std  — std spectral index (reliable)
        spectral_index_min  — min spectral index (reliable)
        spectral_index_max  — max spectral index (reliable)
    """
    reliable = df[df["quality_flag"] == 1]["spectral_index"].dropna()

    spearman_mass = compute_spearman(df, "logM", "spectral_index")
    spearman_env = compute_spearman(df, "env_proxy", "spectral_index")
    ols_mass = compute_ols(df, "logM", "spectral_index")

    return {
        "n_galaxies": len(df),
        "n_reliable": len(reliable),
        "spearman_mass": spearman_mass,
        "spearman_env": spearman_env,
        "ols_mass": ols_mass,
        "spectral_index_mean": float(reliable.mean()) if len(reliable) > 0 else float("nan"),
        "spectral_index_std": float(reliable.std()) if len(reliable) > 0 else float("nan"),
        "spectral_index_min": float(reliable.min()) if len(reliable) > 0 else float("nan"),
        "spectral_index_max": float(reliable.max()) if len(reliable) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figure_scatter(
    df: pd.DataFrame,
    ols: dict,
    out_dir: Path,
) -> Path:
    """Scatter plot: spectral_index vs logM with OLS fit line.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog.
    ols : dict
        OLS result from :func:`compute_ols`.
    out_dir : Path
        Directory to write the figure to.

    Returns
    -------
    Path
        Path to the saved PNG figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["quality_flag"] == 1].dropna(subset=["logM", "spectral_index"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["logM"], sub["spectral_index"],
               color="#1f77b4", s=60, zorder=3, label="Galaxies (quality=1)")

    if not np.isnan(ols["slope"]):
        x_fit = np.linspace(sub["logM"].min(), sub["logM"].max(), 100)
        y_fit = ols["slope"] * x_fit + ols["intercept"]
        label_fit = (
            f"OLS: α = {ols['slope']:.3f}·logM + {ols['intercept']:.3f}\n"
            f"R² = {ols['r_squared']:.3f},  p = {ols['p_value']:.3f}"
        )
        ax.plot(x_fit, y_fit, color="#d62728", lw=1.8, label=label_fit)

    ax.set_xlabel(r"$\log_{10}(M_{\rm bar} / M_\odot)$", fontsize=12)
    ax.set_ylabel(r"Spectral index $\alpha$", fontsize=12)
    ax.set_title("SCM Spectral Index vs Baryonic Mass\n(13-galaxy cleaned sample)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = out_dir / "spectral_index_vs_logM.png"
    pdf_path = out_dir / "spectral_index_vs_logM.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def generate_figure_histogram(df: pd.DataFrame, out_dir: Path) -> Path:
    """Histogram of spectral indices for the reliable sub-sample.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog.
    out_dir : Path
        Directory to write the figure to.

    Returns
    -------
    Path
        Path to the saved PNG figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = df[df["quality_flag"] == 1]["spectral_index"].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub, bins=6, color="#2ca02c", edgecolor="white", alpha=0.85)
    if len(sub) > 0:
        ax.axvline(sub.mean(), color="#d62728", lw=2,
                   label=f"Mean = {sub.mean():.3f}")
    ax.set_xlabel(r"Spectral index $\alpha$", fontsize=12)
    ax.set_ylabel("Number of galaxies", fontsize=12)
    ax.set_title("Distribution of SCM Spectral Indices\n(13-galaxy cleaned sample)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    png_path = out_dir / "spectral_index_histogram.png"
    pdf_path = out_dir / "spectral_index_histogram.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(results: dict, csv_path: str) -> list[str]:
    """Format the spectral analysis report as a list of lines.

    Parameters
    ----------
    results : dict
        Output of :func:`run_analysis`.
    csv_path : str
        Input catalog path (for provenance).

    Returns
    -------
    list[str]
        Human-readable report lines.
    """
    sp_m = results["spearman_mass"]
    sp_e = results["spearman_env"]
    ols = results["ols_mass"]

    lines = [
        _SEP,
        "  Motor de Velos SCM — Spectral Analysis",
        _SEP,
        f"  Catalog      : {csv_path}",
        f"  N galaxies   : {results['n_galaxies']}",
        f"  N reliable   : {results['n_reliable']}",
        "",
        "  Spectral index α summary (reliable sub-sample):",
        f"    Mean   : {results['spectral_index_mean']:.4f}",
        f"    Std    : {results['spectral_index_std']:.4f}",
        f"    Min    : {results['spectral_index_min']:.4f}",
        f"    Max    : {results['spectral_index_max']:.4f}",
        "",
        "  Spearman ρ — spectral_index vs logM:",
        f"    ρ = {sp_m['rho']:.4f},  p = {sp_m['p_value']:.4e},  N = {sp_m['n']}",
        "",
        "  Spearman ρ — spectral_index vs env_proxy:",
        f"    ρ = {sp_e['rho']:.4f},  p = {sp_e['p_value']:.4e},  N = {sp_e['n']}",
        "",
        "  OLS — spectral_index ~ logM:",
        f"    slope     = {ols['slope']:.4f}  (SE {ols['stderr']:.4f})",
        f"    intercept = {ols['intercept']:.4f}",
        f"    R²        = {ols['r_squared']:.4f}",
        f"    p-value   = {ols['p_value']:.4e}",
    ]

    # Verdict
    sig_mass = (not np.isnan(sp_m["p_value"])) and (sp_m["p_value"] < ALPHA_THRESHOLD)
    sig_env = (not np.isnan(sp_e["p_value"])) and (sp_e["p_value"] < ALPHA_THRESHOLD)
    if sig_mass:
        lines.append(
            f"\n  ✅  α correlates significantly with logM "
            f"(ρ = {sp_m['rho']:.3f}, p = {sp_m['p_value']:.3e})"
        )
    else:
        lines.append(
            f"\n  ℹ️   No significant α–mass correlation "
            f"(ρ = {sp_m['rho']:.3f}, p = {sp_m['p_value']:.3e})"
        )
    if sig_env:
        lines.append(
            f"  ✅  α correlates significantly with env_proxy "
            f"(ρ = {sp_e['rho']:.3f}, p = {sp_e['p_value']:.3e})"
        )
    else:
        lines.append(
            f"  ℹ️   No significant α–environment correlation "
            f"(ρ = {sp_e['rho']:.3f}, p = {sp_e['p_value']:.3e})"
        )

    lines.append(_SEP)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCM spectral analysis: compute power-law spectral indices and "
            "correlations for the 13-galaxy cleaned catalog."
        )
    )
    parser.add_argument(
        "--csv", default=str(CSV_DEFAULT),
        help=f"Path to the spectral catalog CSV (default: {CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=str(OUT_DEFAULT), metavar="DIR",
        help=f"Output directory for figures and summary CSV (default: {OUT_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the spectral analysis pipeline.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    dict
        Analysis results including Spearman correlations, OLS fit, and
        descriptive statistics of the spectral index distribution.
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    df = load_catalog(csv_path)
    results = run_analysis(df)

    report_lines = format_report(results, str(csv_path))
    for line in report_lines:
        print(line)

    # Figures
    scatter_path = generate_figure_scatter(df, results["ols_mass"], out_dir)
    hist_path = generate_figure_histogram(df, out_dir)
    print(f"\n  Figures written to {out_dir}/")
    print(f"    {scatter_path.name}")
    print(f"    {hist_path.name}")

    # Summary CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for key, val in results.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                summary_rows.append({"key": f"{key}.{subkey}", "value": subval})
        else:
            summary_rows.append({"key": key, "value": val})
    pd.DataFrame(summary_rows).to_csv(out_dir / "spectral_summary.csv", index=False)
    print(f"    spectral_summary.csv")

    return results


if __name__ == "__main__":
    main()
