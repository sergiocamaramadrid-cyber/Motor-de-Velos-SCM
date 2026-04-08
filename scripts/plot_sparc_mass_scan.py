"""
scripts/plot_sparc_mass_scan.py -- SCM threshold scan figure.

For each logM cut in a configurable range this script selects galaxies
with logM >= m_cut, computes the Spearman correlation between
delta_mass_std and delta_f3 = slope_tail - 0.5, and evaluates a
composite signal score.  The cut that maximises the score is highlighted
as the "critical threshold" on a -log10(p) vs logM cut figure.

Theory
------
delta_f3 = slope_tail - 0.5

where slope_tail is the outer-disk dlogV/dlogr slope and 0.5 is the SCM
reference value (Motor-de-Velos deep form / MOND asymptotic slope).

The signal score is:

    score = |rho| * sqrt(N) * (-log10(p + eps))

with eps = 1e-10 to avoid log(0).  It simultaneously rewards large effect
size, large sample, and high statistical significance.

Usage
-----

    python scripts/plot_sparc_mass_scan.py

With optional arguments::

    python scripts/plot_sparc_mass_scan.py \
        --csv data/sparc_subset.csv \
        --out results/sparc_mass_scan.png \
        --m-start 10.0 --m-stop 11.3 --m-step 0.05 --n-min 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF: float = 0.5
_SCORE_EPS: float = 1e-10

M_START_DEFAULT: float = 10.0
M_STOP_DEFAULT: float = 11.3
M_STEP_DEFAULT: float = 0.05
N_MIN_DEFAULT: int = 15

_REQUIRED_COLS = {"slope_tail", "logM", "delta_mass_std"}
_REPO_ROOT = Path(__file__).parent.parent
_CSV_DEFAULT = str(_REPO_ROOT / "data" / "sparc_subset.csv")
_OUT_DEFAULT = "sparc_mass_scan.png"

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def scan_mass_thresholds(
    df: pd.DataFrame,
    m_start: float = M_START_DEFAULT,
    m_stop: float = M_STOP_DEFAULT,
    m_step: float = M_STEP_DEFAULT,
    n_min: int = N_MIN_DEFAULT,
) -> pd.DataFrame:
    """Scan logM thresholds and compute Spearman statistics for each cut.

    For each cut value m_cut in [m_start, m_stop) (step m_step), the
    subsample df[df["logM"] >= m_cut] is considered.  Entries with
    N <= n_min galaxies are skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns logM, delta_mass_std, delta_f3.
        delta_f3 must already be computed before calling this function.
    m_start, m_stop, m_step : float
        Scan range parameters (same semantics as np.arange).
    n_min : int
        Minimum required subsample size (strictly greater than); rows
        with N <= n_min are skipped.

    Returns
    -------
    pd.DataFrame
        Columns: m_cut, rho, p, N, score.
        Rows are sorted by m_cut ascending.
    """
    mass_range = np.arange(m_start, m_stop, m_step)
    records = []
    for m_cut in mass_range:
        sub = df[df["logM"] >= m_cut]
        if len(sub) <= n_min:
            continue
        rho, p = spearmanr(sub["delta_mass_std"], sub["delta_f3"])
        score = abs(rho) * np.sqrt(len(sub)) * (-np.log10(float(p) + _SCORE_EPS))
        records.append(
            {
                "m_cut": float(m_cut),
                "rho": float(rho),
                "p": float(p),
                "N": int(len(sub)),
                "score": float(score),
            }
        )
    return pd.DataFrame(records, columns=["m_cut", "rho", "p", "N", "score"])


def find_best_cut(scan_df: pd.DataFrame) -> dict:
    """Return the row with the highest signal score as a plain dict.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Output of scan_mass_thresholds.

    Returns
    -------
    dict with keys m_cut, rho, p, N, score.

    Raises
    ------
    ValueError
        If scan_df is empty (no cut met the minimum-N threshold).
    """
    if scan_df.empty:
        raise ValueError(
            "scan_df is empty -- no mass cut produced a subsample large "
            "enough.  Check m_start/m_stop/m_step/n_min parameters."
        )
    best_idx = scan_df["score"].idxmax()
    return scan_df.loc[best_idx].to_dict()


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def generate_figure(
    scan_df: pd.DataFrame,
    out_path: str | Path,
    best: dict | None = None,
) -> plt.Figure:
    """Generate and save the threshold scan figure.

    Plots -log10(p) vs m_cut with a dashed vertical line at the best cut.
    Both a PNG (at out_path) and a sibling PDF are written.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Output of scan_mass_thresholds.
    out_path : str or Path
        Destination path for the PNG.  A sibling .pdf is also saved.
    best : dict or None
        Pre-computed best cut dict (from find_best_cut).  Computed
        internally when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best is None:
        best = find_best_cut(scan_df)

    y_vals = -np.log10(scan_df["p"].to_numpy() + _SCORE_EPS)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        scan_df["m_cut"].to_numpy(),
        y_vals,
        marker="o",
        color="black",
        linewidth=1.2,
        markersize=4,
    )
    ax.axvline(
        best["m_cut"],
        linestyle="--",
        color="red",
        linewidth=1.2,
        label=(
            f"Best cut: logM = {best['m_cut']:.2f}\n"
            f"rho = {best['rho']:.3f}, p = {best['p']:.4e}, "
            f"N = {int(best['N'])}"
        ),
    )
    ax.axhline(
        y=-np.log10(0.05),
        linestyle=":",
        color="gray",
        linewidth=0.8,
        label="p = 0.05",
    )

    ax.set_xlabel("logM cut", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=11)
    ax.set_title("SCM Threshold Scan", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SCM threshold scan: compute Spearman correlation for subsamples "
            "defined by logM >= m_cut and identify the optimal mass threshold."
        )
    )
    parser.add_argument(
        "--csv", default=_CSV_DEFAULT,
        help="Path to per-galaxy catalog CSV (default: data/sparc_subset.csv).",
    )
    parser.add_argument(
        "--out", default=_OUT_DEFAULT,
        help=(
            f"Output PNG path (default: {_OUT_DEFAULT}). "
            "A sibling PDF is also saved."
        ),
    )
    parser.add_argument(
        "--m-start", type=float, default=M_START_DEFAULT, dest="m_start",
        help=f"Start of logM scan range (default: {M_START_DEFAULT}).",
    )
    parser.add_argument(
        "--m-stop", type=float, default=M_STOP_DEFAULT, dest="m_stop",
        help=f"End of logM scan range exclusive (default: {M_STOP_DEFAULT}).",
    )
    parser.add_argument(
        "--m-step", type=float, default=M_STEP_DEFAULT, dest="m_step",
        help=f"Step size for logM scan (default: {M_STEP_DEFAULT}).",
    )
    parser.add_argument(
        "--n-min", type=int, default=N_MIN_DEFAULT, dest="n_min",
        help=(
            f"Minimum subsample size (strictly greater than) to include a "
            f"cut in the scan (default: {N_MIN_DEFAULT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the mass threshold scan pipeline.

    Returns
    -------
    dict with keys:
        scan_df        -- pd.DataFrame of scan results
        best           -- dict for the optimal cut row
        figure_path    -- Path to the saved PNG
        pdf_path       -- Path to the saved PDF
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {csv_path}\n"
            "Provide a CSV with columns: slope_tail, logM, delta_mass_std."
        )

    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[list(_REQUIRED_COLS)].dropna().copy()
    df["delta_f3"] = df["slope_tail"] - BETA_REF

    scan_df = scan_mass_thresholds(
        df,
        m_start=args.m_start,
        m_stop=args.m_stop,
        m_step=args.m_step,
        n_min=args.n_min,
    )
    best = find_best_cut(scan_df)

    out_path = Path(args.out)
    generate_figure(scan_df, out_path, best=best)

    pdf_path = out_path.with_suffix(".pdf")
    print(f"Figure saved as '{out_path}' and '{pdf_path}'")
    print(
        f"Best cut: logM = {best['m_cut']:.3f}  "
        f"rho = {best['rho']:.3f}  "
        f"p = {best['p']:.4e}  "
        f"N = {int(best['N'])}"
    )

    return {
        "scan_df": scan_df,
        "best": best,
        "figure_path": out_path,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    main()
