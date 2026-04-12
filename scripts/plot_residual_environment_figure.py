"""
Figura 2 — Residual Environmental Signal

Shows that the env_proxy → slope_tail correlation persists in the high-mass
regime even after removing the dependence on baryonic mass (residualisation),
ruling out the criticism that the signal is driven by mass alone.

Output:
    results/robustness/figure_residual_environment.png
    results/robustness/figure_residual_environment.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("results", "scm_master_final.csv")
OUT_DIR = os.path.join("results", "robustness")
THRESHOLD = 10.6

FIGURE_CAPTION = (
    "Residual correlation between environmental proxy and outer rotation curve "
    "slope after removing the dependence on baryonic mass. The negative "
    "correlation persists in the high-mass regime, indicating that the "
    "environmental signal is not driven by mass alone."
)


# ---------------------------------------------------------------------------
# Core functions (public API)
# ---------------------------------------------------------------------------

def compute_residuals(df, x_col, y_col, control_col):
    """Return (x_res, y_res) after regressing both on *control_col*."""
    coef_y = np.polyfit(df[control_col], df[y_col], 1)
    y_res = df[y_col] - np.polyval(coef_y, df[control_col])

    coef_x = np.polyfit(df[control_col], df[x_col], 1)
    x_res = df[x_col] - np.polyval(coef_x, df[control_col])

    return x_res, y_res


def generate_figure(x_res, y_res, out_path_base):
    """
    Plot residual scatter + regression line and save PNG + PDF.

    Parameters
    ----------
    x_res, y_res : array-like
        Residualised env_proxy and slope_tail values.
    out_path_base : str
        Path without extension; .png and .pdf are appended.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rho, p = spearmanr(x_res, y_res)
    slope, intercept, *_ = linregress(x_res, y_res)

    xfit = np.linspace(x_res.min(), x_res.max(), 100)
    yfit = slope * xfit + intercept

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x_res, y_res, color="steelblue", alpha=0.75, s=40,
               label="High-mass galaxies")
    ax.plot(xfit, yfit, color="firebrick", linewidth=1.8, label="OLS fit")

    ax.set_xlabel("env_proxy (residual)", fontsize=13)
    ax.set_ylabel("slope_tail (residual)", fontsize=13)
    ax.set_title("Residual environmental correlation (high-mass)", fontsize=12)

    p_str = f"{p:.2e}" if p >= 1e-10 else f"< 1e-10"
    ax.text(0.05, 0.95,
            f"ρ = {rho:.2f}\np = {p_str}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgrey", alpha=0.8))

    ax.legend(fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    fig.savefig(out_path_base + ".png", dpi=300)
    fig.savefig(out_path_base + ".pdf")

    return fig


def main(argv=None):
    df = pd.read_csv(DATA_PATH).dropna(
        subset=["logMbar", "env_proxy", "slope_tail"])

    high = df[df["logMbar"] >= THRESHOLD].copy()

    x_res, y_res = compute_residuals(
        high, x_col="env_proxy", y_col="slope_tail", control_col="logMbar")

    rho, p = spearmanr(x_res, y_res)

    out_base = os.path.join(OUT_DIR, "figure_residual_environment")
    fig = generate_figure(x_res, y_res, out_base)

    print(f"✅ Residual figure generated → {out_base}.png")
    print(f"   ρ = {rho:.3f}   p = {p:.2e}   N = {len(high)}")

    return dict(
        rho=rho, p=p, n=len(high),
        figure_path=out_base + ".png",
        pdf_path=out_base + ".pdf",
        figure=fig,
    )


if __name__ == "__main__":
    main()
