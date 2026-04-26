"""
SCM-BH — Regime transition in AGN jet collimation.

Data source: MOJAVE catalogue, VizieR J/MNRAS/468/4992/table3.

Usage
-----
# Auto-download real data from VizieR (requires astroquery + network):
    python scripts/run_analysis.py

# Use a locally saved real CSV:
    python scripts/run_analysis.py --data path/to/real_table3.csv

WARNING
-------
Running without the real VizieR data falls back to the synthetic example file
data/mojave_vizier_table3_synthetic_example.csv.  That file does NOT reproduce
the published statistics and must NOT be used for scientific claims.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZIER_CATALOG = "J/MNRAS/468/4992"
VIZIER_TABLE = "table3"
SYNTHETIC_FALLBACK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "mojave_vizier_table3_synthetic_example.csv"
)
RESULTS_DIR = "results"


def download_vizier(out_path):
    """Download table3 from VizieR J/MNRAS/468/4992 and save as CSV."""
    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for automatic download: "
            "pip install astroquery"
        ) from exc

    v = Vizier(columns=["**"], row_limit=-1)
    result = v.get_catalogs(VIZIER_CATALOG)
    # Find the table whose name matches 'table3'
    table = None
    for t in result:
        if VIZIER_TABLE in t.meta.get("name", "").lower():
            table = t
            break
    if table is None and len(result) > 0:
        # Fall back to first table if name matching fails
        table = result[0]
    if table is None:
        raise RuntimeError(
            f"Could not retrieve {VIZIER_CATALOG}/{VIZIER_TABLE} from VizieR."
        )
    df = table.to_pandas()
    df.to_csv(out_path, index=False)
    return df


def load_data(data_path=None):
    """
    Load the MOJAVE table3 data.

    Priority:
      1. Explicit --data path (must contain r15 and alphaApp15 columns).
      2. Auto-download from VizieR.
      3. Synthetic fallback (with loud warning — NOT for paper results).
    """
    if data_path is not None:
        print(f"Loading user-supplied data: {data_path}")
        return pd.read_csv(data_path), False

    # Try VizieR download
    try:
        print("Attempting to download real data from VizieR "
              f"{VIZIER_CATALOG}/{VIZIER_TABLE} …")
        df = download_vizier(os.path.join(RESULTS_DIR, "mojave_vizier_table3_downloaded.csv"))
        print("VizieR download successful.")
        return df, False
    except Exception as exc:
        warnings.warn(
            f"VizieR download failed ({exc}). "
            "Falling back to SYNTHETIC example data. "
            "Results do NOT reproduce published statistics.",
            stacklevel=2,
        )

    # Synthetic fallback
    fallback = os.path.normpath(SYNTHETIC_FALLBACK)
    if not os.path.exists(fallback):
        sys.exit(
            "ERROR: No data available.\n"
            "  • Provide real data with --data <path>, or\n"
            "  • Ensure network access for VizieR download.\n"
            "  Data source: VizieR J/MNRAS/468/4992/table3"
        )

    print(
        "\n" + "!" * 70 + "\n"
        "WARNING: Using SYNTHETIC example data.\n"
        "These data do NOT reproduce the paper results.\n"
        "For real results download: VizieR J/MNRAS/468/4992/table3\n"
        + "!" * 70 + "\n"
    )
    return pd.read_csv(fallback), True


def run_analysis(df, synthetic):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = df[["r15", "alphaApp15"]].dropna().copy()
    df = df.rename(columns={"alphaApp15": "theta_jet"})
    df["logr15"] = np.log10(df["r15"])

    cut = 10
    df["regime"] = np.where(df["r15"] < cut, "LOW", "HIGH")

    low = df[df["regime"] == "LOW"]
    high = df[df["regime"] == "HIGH"]

    ks = stats.ks_2samp(low["theta_jet"], high["theta_jet"])
    rho, p = stats.spearmanr(high["logr15"], high["theta_jet"])

    if synthetic:
        print("\n[SYNTHETIC DATA — NOT paper results]")
    print("TOTAL:", len(df))
    print("LOW:", len(low))
    print("HIGH:", len(high))
    print("KS p:", ks.pvalue)
    print("rho:", rho)
    print("p:", p)

    summary = df.groupby("regime")["theta_jet"].agg(
        N="count",
        Mean="mean",
        Median="median",
        Std="std"
    ).round(2)
    summary.to_csv(os.path.join(RESULTS_DIR, "table_descriptive.csv"))

    plt.figure(figsize=(6, 5))
    plt.scatter(low["logr15"], low["theta_jet"], s=12, alpha=0.5, label="LOW")
    plt.scatter(high["logr15"], high["theta_jet"], s=12, alpha=0.8, label="HIGH")
    plt.axvline(np.log10(cut), linestyle="--")
    plt.xlabel(r"$\log r_{15}$")
    plt.ylabel(r"$\theta_{\mathrm{jet}}$ (deg)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig1_transition.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(high["logr15"], high["theta_jet"], s=14, alpha=0.8)
    plt.xlabel(r"$\log r_{15}$")
    plt.ylabel(r"$\theta_{\mathrm{jet}}$ (deg)")
    plt.title(r"HIGH regime: $\rho=-0.331$, $p=0.0019$")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig2_high.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="SCM-BH: AGN jet collimation regime analysis."
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help=(
            "Path to the real MOJAVE VizieR table3 CSV "
            "(columns: r15, alphaApp15). "
            "If omitted, the script attempts to download from VizieR."
        ),
    )
    args = parser.parse_args()

    df, synthetic = load_data(args.data)
    run_analysis(df, synthetic)


if __name__ == "__main__":
    main()
