"""
scripts/run_analysis.py — SCM mass + distribution structural model.

Fits the structural model:

    slope_tail ~ logMbar + Sigma_resid

and reports OLS coefficients, significance, and R².

The key SCM result is:
    - logMbar is highly significant
    - Sigma_resid is significant
    - R² ≈ 0.19

Usage
-----
With default paths::

    python scripts/run_analysis.py

Explicit options::

    python scripts/run_analysis.py \\
        --data    data/scm_mass_distribution_final_dataset.csv \\
        --out-dir results \\
        --figure  figures/SCM_figure_main.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = ["logMbar", "Sigma_resid"]
TARGET_COL = "slope_tail"
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Statistical model
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the SCM mass distribution dataset.

    Parameters
    ----------
    path : str or Path
        Path to the CSV produced by ``build_dataset.py``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def run_ols_model(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> dict:
    """Fit OLS model: target ~ feature1 + feature2.

    Uses statsmodels when available (HC3 robust standard errors), falls back to
    scipy OLS otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``feature_cols`` and ``target_col``.
    feature_cols : list of str
        Predictor column names.
    target_col : str
        Response column name.

    Returns
    -------
    dict with keys:
        n, r2, r2_adj, coefs, pvalues, significant, formula
    """
    cols = feature_cols + [target_col]
    clean = df[cols].dropna()
    n = len(clean)

    if _HAS_STATSMODELS:
        formula = f"{target_col} ~ " + " + ".join(feature_cols)
        model = smf.ols(formula, data=clean).fit(cov_type="HC3")
        coefs = model.params.to_dict()
        pvalues = model.pvalues.to_dict()
        r2 = float(model.rsquared)
        r2_adj = float(model.rsquared_adj)
        result = {
            "n": n,
            "r2": round(r2, 4),
            "r2_adj": round(r2_adj, 4),
            "coefs": {k: round(float(v), 6) for k, v in coefs.items()},
            "pvalues": {k: round(float(v), 6) for k, v in pvalues.items()},
            "significant": {k: float(v) < ALPHA for k, v in pvalues.items()},
            "formula": formula,
            "backend": "statsmodels_HC3",
        }
    else:
        X = clean[feature_cols].values
        y = clean[target_col].values
        X_aug = np.column_stack([np.ones(n), X])
        coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = X.shape[1]
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

        coef_names = ["Intercept"] + feature_cols
        coefs = {name: round(float(c), 6) for name, c in zip(coef_names, coef)}

        # t-statistics via manual OLS
        mse = ss_res / (n - k - 1) if n > k + 1 else np.nan
        cov_b = mse * np.linalg.pinv(X_aug.T @ X_aug)
        se = np.sqrt(np.diag(cov_b))
        t_stat = coef / se
        pvalues = {
            name: round(float(2 * scipy_stats.t.sf(abs(t), df=n - k - 1)), 6)
            for name, t, _ in zip(coef_names, t_stat, se)
        }

        result = {
            "n": n,
            "r2": round(float(r2), 4),
            "r2_adj": round(float(r2_adj), 4),
            "coefs": coefs,
            "pvalues": pvalues,
            "significant": {k: float(v) < ALPHA for k, v in pvalues.items()},
            "formula": f"{target_col} ~ " + " + ".join(feature_cols),
            "backend": "numpy_lstsq",
        }

    return result


def compute_spearman(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> dict:
    """Compute Spearman correlations between each feature and the target.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str
    target_col : str

    Returns
    -------
    dict mapping feature name to {rho, pvalue}
    """
    results = {}
    for col in feature_cols:
        clean = df[[col, target_col]].dropna()
        rho, pval = scipy_stats.spearmanr(clean[col], clean[target_col])
        results[col] = {"rho": round(float(rho), 4), "pvalue": round(float(pval), 6)}
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary(ols: dict, spearman: dict) -> dict:
    """Combine OLS and Spearman results into a summary dict.

    Parameters
    ----------
    ols : dict
        Output of ``run_ols_model``.
    spearman : dict
        Output of ``compute_spearman``.

    Returns
    -------
    dict
    """
    return {
        "key_result": "slope_tail ~ logMbar + Sigma_resid",
        "n": ols["n"],
        "r2": ols["r2"],
        "r2_adj": ols["r2_adj"],
        "formula": ols["formula"],
        "coefficients": ols["coefs"],
        "pvalues": ols["pvalues"],
        "significant": ols["significant"],
        "spearman": spearman,
        "interpretation": {
            "logMbar": "total baryonic mass sets the scale",
            "Sigma_resid": "mass distribution modulates the structure",
        },
    }


def format_report(summary: dict) -> str:
    """Format a human-readable plain-text report.

    Parameters
    ----------
    summary : dict
        Output of ``build_summary``.

    Returns
    -------
    str
    """
    sep = "=" * 64
    lines = [
        sep,
        "SCM Framework — Mass & Distribution Result",
        sep,
        "",
        f"Key result:  {summary['key_result']}",
        f"Sample size: {summary['n']}",
        f"R²:          {summary['r2']:.4f}",
        f"R² (adj):    {summary['r2_adj']:.4f}",
        "",
        "OLS Coefficients (HC3 robust SEs when statsmodels available)",
        "-" * 64,
    ]
    for name, coef in summary["coefficients"].items():
        pval = summary["pvalues"].get(name, float("nan"))
        sig = "**" if summary["significant"].get(name) else "  "
        lines.append(f"  {name:<20s}  coef={coef:+.4f}  p={pval:.4f}  {sig}")

    lines += [
        "",
        "Spearman Correlations",
        "-" * 64,
    ]
    for feat, vals in summary["spearman"].items():
        lines.append(f"  {feat:<20s}  ρ={vals['rho']:+.4f}  p={vals['pvalue']:.4f}")

    lines += [
        "",
        "Interpretation",
        "-" * 64,
        "  logMbar:      total baryonic mass sets the scale",
        "  Sigma_resid:  mass distribution modulates the structure",
        "",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(
    df: pd.DataFrame,
    ols: dict,
    out: str | Path,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> None:
    """Generate the main SCM mass + distribution figure.

    Two-panel figure:
    - Left:  slope_tail vs logMbar (coloured by Sigma_resid)
    - Right: OLS residuals vs Sigma_resid

    Parameters
    ----------
    df : pd.DataFrame
    ols : dict
        Output of ``run_ols_model``.
    out : str or Path
        Output PNG path.
    feature_cols : list of str
    target_col : str
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cols = feature_cols + [target_col]
    clean = df[cols].dropna().copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: slope_tail vs logMbar
    ax = axes[0]
    if "Sigma_resid" in clean.columns:
        sc = ax.scatter(
            clean["logMbar"], clean[target_col],
            c=clean["Sigma_resid"], cmap="coolwarm", s=20, alpha=0.75,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=r"$\Sigma_\mathrm{resid}$")
    else:
        ax.scatter(clean["logMbar"], clean[target_col], s=20, alpha=0.75)

    # OLS line
    xgrid = np.linspace(clean["logMbar"].min(), clean["logMbar"].max(), 100)
    intercept = ols["coefs"].get("Intercept", 0.0)
    slope_logm = ols["coefs"].get("logMbar", 0.0)
    ax.plot(xgrid, intercept + slope_logm * xgrid, "k--", lw=1.5,
            label=f"OLS  R²={ols['r2']:.2f}")
    ax.set_xlabel(r"$\log M_\mathrm{bar}\ [M_\odot]$")
    ax.set_ylabel(r"slope$_\mathrm{tail}$")
    ax.set_title("A: Mass vs. outer slope")
    ax.legend(fontsize=8)

    # Panel B: residuals vs Sigma_resid
    ax = axes[1]
    if "Sigma_resid" in clean.columns:
        # Compute OLS residuals
        X = clean[feature_cols].values
        y = clean[target_col].values
        n = len(y)
        X_aug = np.column_stack([np.ones(n), X])
        coef_vec = np.array(
            [ols["coefs"].get("Intercept", 0)]
            + [ols["coefs"].get(f, 0) for f in feature_cols]
        )
        resid = y - X_aug @ coef_vec

        ax.scatter(clean["Sigma_resid"], resid, s=20, alpha=0.75,
                   edgecolors="none", color="steelblue")
        ax.axhline(0, color="k", lw=1, ls="--")
        rho, pval = scipy_stats.spearmanr(clean["Sigma_resid"], resid)
        ax.set_xlabel(r"$\Sigma_\mathrm{resid}$")
        ax.set_ylabel("OLS residual")
        ax.set_title(f"B: Distribution residual  (ρ={rho:.2f}, p={pval:.3f})")
    else:
        ax.text(0.5, 0.5, "Sigma_resid not available",
                ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(
        r"SCM: slope$_\mathrm{tail} \approx f(\log M_\mathrm{bar},\,\Sigma)$",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="SCM structural model: slope_tail ~ logMbar + Sigma_resid"
    )
    parser.add_argument(
        "--data",
        default="data/scm_mass_distribution_final_dataset.csv",
        help="Input dataset CSV (default: data/scm_mass_distribution_final_dataset.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for JSON and TXT reports (default: results)",
    )
    parser.add_argument(
        "--figure",
        default="figures/SCM_figure_main.png",
        help="Output figure path (default: figures/SCM_figure_main.png)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    if args.verbose:
        print(f"Loaded {len(df)} rows from {args.data}")

    ols = run_ols_model(df)
    spearman = compute_spearman(df)
    summary = build_summary(ols, spearman)

    json_path = out_dir / "scm_mass_distribution_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    if args.verbose:
        print(f"Summary written to {json_path}")

    report = format_report(summary)
    txt_path = out_dir / "SCM_mass_distribution_report.txt"
    with open(txt_path, "w") as fh:
        fh.write(report)
    if args.verbose:
        print(f"Report written to {txt_path}")
        print(report)

    generate_figure(df, ols, out=args.figure)
    if args.verbose:
        print(f"Figure saved to {args.figure}")

    return summary


if __name__ == "__main__":
    result = main()
