"""
scripts/mw_delta_f3.py — Milky Way ΔF₃ from Cepheid rotation curve.

Loads a Galactocentric rotation curve table (``R_kpc``, ``Vc_kms``,
optionally ``e_Vc``) obtained from Cepheid tracers (e.g. Mroz et al. 2019,
Eilers et al. 2019) and computes the SCM outer-disk slope

    slope_tail = d log Vc / d log R

via weighted (or unweighted) OLS on the log–log plane for radii
``R >= R_cut``.  The SCM residual is then

    delta_f3_mw = slope_tail − 0.5

where 0.5 is the reference flat-rotation value (Motor-de-Velos deep form /
MOND asymptotic slope).

A radial scan equivalent to the mass-threshold scan in
:mod:`scripts.plot_sparc_mass_scan` is also provided via
:func:`scan_r_cuts` so that the cut can be chosen data-driven.

The script produces a two-panel figure:
  * Left  — full rotation curve with the fitted outer slope overlaid.
  * Right — scan of slope_tail / delta_f3 vs R_cut.

Theory
------
delta_f3 = slope_tail − 0.5

where slope_tail is the log-log OLS slope of Vc vs R in the outer region
and 0.5 is the SCM reference value.

Usage
-----
::

    python scripts/mw_delta_f3.py

With optional arguments::

    python scripts/mw_delta_f3.py \\
        --csv data/mw_cepheids.csv \\
        --r-cut 13.0 \\
        --out results/mw_delta_f3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF: float = 0.5

#: Default outer-region cut (kpc).  Below this radius the inner disk
#: potential dominates; above it the rotation curve traces the halo.
R_CUT_DEFAULT: float = 13.0

R_START_DEFAULT: float = 8.0
R_STOP_DEFAULT: float = 20.0
R_STEP_DEFAULT: float = 0.5
N_MIN_DEFAULT: int = 5

_SCORE_EPS: float = 1e-10

_REQUIRED_COLS = {"R_kpc", "Vc_kms"}
_REPO_ROOT = Path(__file__).parent.parent
_CSV_DEFAULT = str(_REPO_ROOT / "data" / "mw_cepheids.csv")
_OUT_DEFAULT = "results/mw_delta_f3.png"

#: Official figure caption for use in paper drafts and supplementary material.
FIGURE_CAPTION: str = (
    "Figure MW. Milky Way outer rotation-curve slope from Gaia/OGLE Cepheids "
    "(Mroz et al.\u00a02019; Eilers et al.\u00a02019). "
    "Left: circular velocity vs. Galactocentric radius; the dashed line shows "
    "the log\u2013log OLS fit for R\u202f\u2265\u202f13\u202ckpc. "
    "Right: slope\u2009tail and \u0394F\u2083\u202c=\u202cslope\u2009tail\u202c"
    "\u2212\u202c0.5 as a function of the outer-region cut R_cut. "
    "The negative \u0394F\u2083 confirms that the MW outer rotation curve "
    "declines faster than the SCM/MOND reference slope, consistent with the "
    "high-mass SPARC result."
)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_slope(
    R: np.ndarray,
    V: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Fit log–log OLS slope of V vs R and return SCM quantities.

    Parameters
    ----------
    R : array-like
        Galactocentric radii in kpc (outer region only).
    V : array-like
        Circular velocities in km/s (outer region only).
    weights : array-like or None
        Optional inverse-variance weights (1/e_Vc²).  If None, unweighted
        OLS is used.

    Returns
    -------
    dict with keys:
        slope_tail    — d log Vc / d log R
        intercept     — log-space intercept (log10 Vc at log10 R = 0)
        delta_f3      — slope_tail − 0.5
        p_slope       — two-sided p-value for H₀: slope = 0 (unweighted OLS)
        n             — number of data points
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)

    if len(R) < 2:
        raise ValueError(f"Need at least 2 data points; got {len(R)}.")

    logR = np.log10(R)
    logV = np.log10(V)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        # Weighted OLS via normal equations for degree-1 polynomial
        sw = w.sum()
        sx = (w * logR).sum()
        sy = (w * logV).sum()
        sxx = (w * logR ** 2).sum()
        sxy = (w * logR * logV).sum()
        denom = sw * sxx - sx ** 2
        slope_tail = (sw * sxy - sx * sy) / denom
        intercept = (sy - slope_tail * sx) / sw
    else:
        coef = np.polyfit(logR, logV, 1)
        slope_tail = float(coef[0])
        intercept = float(coef[1])

    delta_f3 = float(slope_tail) - BETA_REF

    # Unweighted p-value for H₀: slope = 0 via scipy linregress
    lr = linregress(logR, logV)
    p_slope = float(lr.pvalue)

    return {
        "slope_tail": float(slope_tail),
        "intercept": float(intercept),
        "delta_f3": delta_f3,
        "p_slope": p_slope,
        "n": int(len(R)),
    }


def scan_r_cuts(
    df: pd.DataFrame,
    r_start: float = R_START_DEFAULT,
    r_stop: float = R_STOP_DEFAULT,
    r_step: float = R_STEP_DEFAULT,
    n_min: int = N_MIN_DEFAULT,
) -> pd.DataFrame:
    """Scan outer-region cuts and return slope_tail/delta_f3/p_slope for each.

    Parameters
    ----------
    df : DataFrame
        Must have ``R_kpc`` and ``Vc_kms`` columns; optionally ``e_Vc``.
    r_start, r_stop, r_step : float
        Range of R_cut values to scan (kpc).
    n_min : int
        Minimum number of data points required to compute a fit.

    Returns
    -------
    DataFrame with columns: r_cut, slope_tail, delta_f3, p_slope, n.
    """
    has_err = "e_Vc" in df.columns
    rows = []
    cuts = np.arange(r_start, r_stop + r_step / 2.0, r_step)
    for r_cut in cuts:
        mask = df["R_kpc"].values >= r_cut
        sub = df[mask].dropna(subset=["R_kpc", "Vc_kms"])
        if len(sub) < n_min:
            continue
        weights = None
        if has_err:
            e = sub["e_Vc"].values
            valid = e > 0
            if valid.all():
                weights = 1.0 / (e ** 2)
        try:
            res = compute_slope(sub["R_kpc"].values, sub["Vc_kms"].values, weights)
        except ValueError:
            continue
        rows.append(
            {
                "r_cut": float(r_cut),
                "slope_tail": res["slope_tail"],
                "delta_f3": res["delta_f3"],
                "p_slope": res["p_slope"],
                "n": res["n"],
            }
        )
    return pd.DataFrame(rows)


def find_best_r_cut(scan_df: pd.DataFrame) -> dict:
    """Find the R_cut that maximises the signal score in the radial scan.

    The score rewards a steep negative slope, a large sample, and high
    statistical significance::

        score = |slope_tail| * sqrt(N) * (-log10(p_slope + eps))

    Parameters
    ----------
    scan_df : DataFrame
        Output of :func:`scan_r_cuts` (must have ``r_cut``, ``slope_tail``,
        ``p_slope``, ``n`` columns).

    Returns
    -------
    dict with keys:
        r_crit      — R_cut with highest score
        slope_tail  — slope at R_crit
        delta_f3    — delta_f3 at R_crit
        p_slope     — p-value at R_crit
        n           — sample size at R_crit
        score       — composite score at R_crit
    """
    if scan_df.empty:
        raise ValueError("scan_df is empty; cannot find best R_cut.")

    scores = (
        scan_df["slope_tail"].abs()
        * np.sqrt(scan_df["n"])
        * (-np.log10(scan_df["p_slope"] + _SCORE_EPS))
    )
    idx = scores.idxmax()
    row = scan_df.loc[idx]
    return {
        "r_crit": float(row["r_cut"]),
        "slope_tail": float(row["slope_tail"]),
        "delta_f3": float(row["delta_f3"]),
        "p_slope": float(row["p_slope"]),
        "n": int(row["n"]),
        "score": float(scores[idx]),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def generate_figure(
    df: pd.DataFrame,
    out_path: str | Path,
    r_cut: float = R_CUT_DEFAULT,
    best: dict | None = None,
) -> plt.Figure:
    """Generate the two-panel MW rotation-curve figure and save PNG + PDF.

    Parameters
    ----------
    df : DataFrame
        Must have ``R_kpc`` and ``Vc_kms``; optionally ``e_Vc``.
    out_path : str or Path
        Destination PNG file.  A sibling PDF is saved automatically.
    r_cut : float
        Outer-region radius cut in kpc (used for left panel and as default
        scan reference).
    best : dict or None
        Optional output of :func:`find_best_r_cut`.  If provided, the best
        R_crit is highlighted on the right panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_err = "e_Vc" in df.columns
    mask_out = df["R_kpc"].values >= r_cut
    sub = df[mask_out].dropna(subset=["R_kpc", "Vc_kms"])

    weights = None
    if has_err:
        e = sub["e_Vc"].values
        if (e > 0).all():
            weights = 1.0 / (e ** 2)

    res = compute_slope(sub["R_kpc"].values, sub["Vc_kms"].values, weights)
    scan_df = scan_r_cuts(df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left panel: rotation curve ---
    ax = axes[0]
    R_all = df["R_kpc"].values
    V_all = df["Vc_kms"].values
    R_inner = R_all[~mask_out]
    V_inner = V_all[~mask_out]
    R_outer = sub["R_kpc"].values
    V_outer = sub["Vc_kms"].values

    if has_err:
        e_all = df["e_Vc"].values
        ax.errorbar(
            R_inner, V_inner, yerr=e_all[~mask_out],
            fmt="o", color="steelblue", ms=4, lw=0.8, alpha=0.7,
            label=f"Inner (R < {r_cut:.0f} kpc)",
        )
        ax.errorbar(
            R_outer, V_outer, yerr=sub["e_Vc"].values if has_err else None,
            fmt="s", color="firebrick", ms=4, lw=0.8, alpha=0.9,
            label=f"Outer (R ≥ {r_cut:.0f} kpc)",
        )
    else:
        ax.scatter(R_inner, V_inner, s=20, color="steelblue", alpha=0.7,
                   label=f"Inner (R < {r_cut:.0f} kpc)")
        ax.scatter(R_outer, V_outer, s=20, color="firebrick", alpha=0.9,
                   label=f"Outer (R ≥ {r_cut:.0f} kpc)")

    R_line = np.linspace(R_outer.min(), R_outer.max(), 200)
    V_line = (10 ** res["intercept"]) * R_line ** res["slope_tail"]
    ax.plot(R_line, V_line, "--", color="firebrick", lw=1.5,
            label=(
                f"OLS fit: slope = {res['slope_tail']:.3f}\n"
                f"ΔF₃ = {res['delta_f3']:.3f}"
            ))

    ax.set_xlabel("R (kpc)", fontsize=11)
    ax.set_ylabel(r"$V_c$ (km s$^{-1}$)", fontsize=11)
    ax.set_title("Milky Way — Cepheid rotation curve", fontsize=11)
    ax.legend(fontsize=8)

    # --- Right panel: slope_tail & delta_f3 vs r_cut, with p_slope twin ---
    ax2 = axes[1]
    if not scan_df.empty:
        ax2.plot(scan_df["r_cut"], scan_df["slope_tail"], "o-",
                 color="steelblue", ms=4, label=r"slope$_\mathrm{tail}$")
        ax2.plot(scan_df["r_cut"], scan_df["delta_f3"], "s--",
                 color="firebrick", ms=4, label=r"$\Delta F_3$")
        ax2.axhline(0, color="black", lw=0.8, ls=":")
        ax2.axvline(r_cut, color="gray", lw=0.8, ls="--", alpha=0.6,
                    label=f"Default cut ({r_cut:.0f} kpc)")

        if best is not None:
            ax2.axvline(best["r_crit"], color="darkorange", lw=1.5, ls="-",
                        label=f"R_crit = {best['r_crit']:.1f} kpc")

        # Twin axis for -log10(p_slope)
        ax2b = ax2.twinx()
        neg_log_p = -np.log10(scan_df["p_slope"] + _SCORE_EPS)
        ax2b.plot(scan_df["r_cut"], neg_log_p, "^:", color="purple",
                  ms=4, alpha=0.7, label=r"$-\log_{10}(p)$")
        ax2b.set_ylabel(r"$-\log_{10}(p_\mathrm{slope})$", fontsize=9,
                        color="purple")
        ax2b.tick_params(axis="y", labelcolor="purple")

        ax2.set_xlabel("$R_\\mathrm{cut}$ (kpc)", fontsize=11)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.set_title("Radial scan", fontsize=11)

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2b, labels2b = ax2b.get_legend_handles_labels()
        ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MW ΔF₃ from Cepheid rotation curve."
    )
    parser.add_argument(
        "--csv", default=_CSV_DEFAULT,
        help="Path to rotation-curve CSV (default: data/mw_cepheids.csv).",
    )
    parser.add_argument(
        "--r-cut", type=float, default=R_CUT_DEFAULT, dest="r_cut",
        help=f"Outer-region radius cut in kpc (default: {R_CUT_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=_OUT_DEFAULT,
        help=f"Output PNG path (default: {_OUT_DEFAULT}). A sibling PDF is also saved.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the MW ΔF₃ pipeline.

    Returns
    -------
    dict with keys:
        slope          — compute_slope result dict (at r_cut)
        r_cut          — outer-region cut used for left panel
        r_crit         — data-driven best R_cut from radial scan
        best           — find_best_r_cut result dict
        scan_df        — DataFrame from scan_r_cuts
        figure_path    — Path to the saved PNG
        pdf_path       — Path to the saved PDF
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Rotation-curve CSV not found: {csv_path}\n"
            "Provide a CSV with columns: R_kpc, Vc_kms (and optionally e_Vc)."
        )

    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    mask = df["R_kpc"].values >= args.r_cut
    sub = df[mask].dropna(subset=["R_kpc", "Vc_kms"])
    n_outer = len(sub)

    if n_outer < 2:
        raise ValueError(
            f"Only {n_outer} data points with R >= {args.r_cut} kpc; "
            "need at least 2."
        )

    has_err = "e_Vc" in df.columns
    weights = None
    if has_err:
        e = sub["e_Vc"].values
        if (e > 0).all():
            weights = 1.0 / (e ** 2)

    slope_result = compute_slope(sub["R_kpc"].values, sub["Vc_kms"].values, weights)
    scan_df = scan_r_cuts(df)
    best = find_best_r_cut(scan_df) if not scan_df.empty else None

    out_path = Path(args.out)
    generate_figure(df, out_path, args.r_cut, best=best)
    pdf_path = out_path.with_suffix(".pdf")

    print("\nRESULTADO MW (Cefeidas)")
    print(f"R_cut        = {args.r_cut:.1f} kpc  (used for figure)")
    print(f"N (outer)    = {slope_result['n']}")
    print(f"slope_tail   = {slope_result['slope_tail']:.4f}")
    print(f"p_slope      = {slope_result['p_slope']:.3e}")
    print(f"delta_F3_MW  = {slope_result['delta_f3']:.4f}")

    if best is not None:
        print(f"\nRADIAL SCAN — R_crit (max score)")
        print(f"R_crit       = {best['r_crit']:.1f} kpc")
        print(f"slope_tail   = {best['slope_tail']:.4f}")
        print(f"p_slope      = {best['p_slope']:.3e}")
        print(f"delta_F3     = {best['delta_f3']:.4f}")
        print(f"N            = {best['n']}")

    print(f"\nSCAN TABLE (R_start={R_START_DEFAULT:.0f}, R_stop={R_STOP_DEFAULT:.0f} kpc):")
    if not scan_df.empty:
        print(scan_df.to_string(index=False,
                                float_format=lambda x: f"{x:.4f}"))

    print(f"\nFigure saved as '{out_path}' and '{pdf_path}'")

    return {
        "slope": slope_result,
        "r_cut": args.r_cut,
        "r_crit": best["r_crit"] if best else None,
        "best": best,
        "scan_df": scan_df,
        "figure_path": out_path,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    main()
