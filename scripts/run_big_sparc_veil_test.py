"""
scripts/run_big_sparc_veil_test.py — Statistical validation of the Motor de
Velos veil (SCM) model on the full SPARC catalog.

Reads the per-galaxy catalog produced by ``build_sparc_full_catalog.py`` and
applies a battery of statistical tests that together validate (or challenge)
the Motor de Velos framework predictions:

  1. **Fit quality** — fraction of galaxies with χ²_ν below a threshold.
  2. **Baryonic Tully-Fisher Relation (BTFR)** — Pearson *r* between
     log V_flat and log M_bar_BTFR; checks tightness of the predicted BTFR.
  3. **Deep-regime slope (β) test** — median β and fraction with
     |β − 0.5| < tolerance; Wilcoxon one-sample test against β = 0.5.
  4. **Velos prevalence** — fraction of galaxies with a reliable β estimate.

Outputs
-------
``<out>/results_overview.json``
    Machine-readable summary of all test statistics.
``<out>/veil_test_summary.txt``
    Human-readable plain-text summary.

Usage
-----
::

    python scripts/run_big_sparc_veil_test.py \\
        --catalog data/SPARC/sparc_full.csv \\
        --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

# ---------------------------------------------------------------------------
# Decision thresholds (can be overridden via CLI)
# ---------------------------------------------------------------------------

CHI2_GOOD_THRESHOLD = 3.0   # χ²_ν < this → "good fit"
BETA_MOND_TARGET = 0.5      # expected deep-slope for MOND / Motor de Velos
BETA_TOLERANCE = 0.1        # |β − 0.5| < this → "near-MOND"


# ---------------------------------------------------------------------------
# Core validation function
# ---------------------------------------------------------------------------

def run_veil_test(
    catalog_path: str | Path,
    out_dir: str | Path = "results",
    chi2_threshold: float = CHI2_GOOD_THRESHOLD,
    beta_target: float = BETA_MOND_TARGET,
    beta_tol: float = BETA_TOLERANCE,
    verbose: bool = True,
) -> dict:
    """Run the Motor de Velos veil statistical validation.

    Parameters
    ----------
    catalog_path : str or Path
        Full catalog CSV (output of ``build_sparc_full_catalog.py``).
    out_dir : str or Path
        Directory for output files.
    chi2_threshold : float
        χ²_ν cutoff used to define a "good fit".
    beta_target : float
        Expected deep-regime slope β (0.5 for MOND / Motor de Velos).
    beta_tol : float
        Tolerance for |β − beta_target| to count as "near-target".
    verbose : bool
        Print progress and summary.

    Returns
    -------
    dict
        Results overview (also written to ``results_overview.json``).
    """
    catalog_path = Path(catalog_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading catalog: {catalog_path}")
    df = pd.read_csv(catalog_path)
    n_total = len(df)
    if verbose:
        print(f"  {n_total} galaxies loaded")

    # ------------------------------------------------------------------
    # 1. Fit quality
    # ------------------------------------------------------------------
    chi2_vals = pd.to_numeric(df.get("chi2_reduced", pd.Series(dtype=float)),
                               errors="coerce")
    n_good_fit = int((chi2_vals < chi2_threshold).sum())
    frac_good_fit = float(n_good_fit / max(n_total, 1))
    median_chi2 = float(chi2_vals.median())

    upsilon_vals = pd.to_numeric(df.get("upsilon_disk", pd.Series(dtype=float)),
                                  errors="coerce")
    median_upsilon = float(upsilon_vals.median())

    # ------------------------------------------------------------------
    # 2. Baryonic Tully-Fisher Relation (BTFR)
    # ------------------------------------------------------------------
    btfr_r: float | None = None
    btfr_p: float | None = None
    n_btfr = 0
    if "M_bar_BTFR_Msun" in df.columns and "Vflat" in df.columns:
        mbar = pd.to_numeric(df["M_bar_BTFR_Msun"], errors="coerce")
        vflat = pd.to_numeric(df["Vflat"], errors="coerce")
        mask = np.isfinite(mbar) & np.isfinite(vflat) & (mbar > 0) & (vflat > 0)
        n_btfr = int(mask.sum())
        if n_btfr >= 3:
            r_val, p_val = pearsonr(np.log10(vflat[mask]), np.log10(mbar[mask]))
            btfr_r = round(float(r_val), 4)
            btfr_p = float(p_val)

    # ------------------------------------------------------------------
    # 3. Deep-regime slope (β) test
    # ------------------------------------------------------------------
    beta_vals = pd.to_numeric(df.get("beta", pd.Series(dtype=float)), errors="coerce")

    if "reliable" in df.columns:
        reliable_mask = df["reliable"].astype(bool) & np.isfinite(beta_vals)
    else:
        reliable_mask = np.isfinite(beta_vals)

    beta_reliable = beta_vals[reliable_mask].values
    n_reliable = len(beta_reliable)
    median_beta: float | None = (
        round(float(np.median(beta_reliable)), 4) if n_reliable > 0 else None
    )
    n_near_mond = int(np.sum(np.abs(beta_reliable - beta_target) < beta_tol))
    frac_near_mond = round(float(n_near_mond / max(n_reliable, 1)), 4)

    # Wilcoxon one-sample test: H0 = median β equals beta_target
    wilcoxon_stat: float | None = None
    wilcoxon_p: float | None = None
    if n_reliable >= 10:
        try:
            stat, p = wilcoxon(beta_reliable - beta_target, alternative="two-sided")
            wilcoxon_stat = round(float(stat), 4)
            wilcoxon_p = float(p)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 4. Velos prevalence
    # ------------------------------------------------------------------
    frac_reliable = round(float(n_reliable / max(n_total, 1)), 4)

    # ------------------------------------------------------------------
    # Assemble results dict
    # ------------------------------------------------------------------
    results: dict = {
        "n_galaxies_total": n_total,
        "fit_quality": {
            "n_good_fit": n_good_fit,
            "frac_good_fit": round(frac_good_fit, 4),
            "median_chi2_reduced": round(median_chi2, 4),
            "median_upsilon_disk": round(median_upsilon, 4),
            "chi2_threshold_used": chi2_threshold,
        },
        "btfr": {
            "n_galaxies": n_btfr,
            "pearson_r": btfr_r,
            "p_value": btfr_p,
        },
        "deep_slope": {
            "n_reliable": n_reliable,
            "median_beta": median_beta,
            "n_near_mond": n_near_mond,
            "frac_near_mond": frac_near_mond,
            "beta_target": beta_target,
            "beta_tolerance": beta_tol,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
        },
        "velos_prevalence": {
            "n_reliable_deep_slope": n_reliable,
            "frac_reliable_deep_slope": frac_reliable,
        },
    }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    json_path = out_dir / "results_overview.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    summary_path = out_dir / "veil_test_summary.txt"
    _write_summary(results, summary_path)

    if verbose:
        print("\n=== Motor de Velos Veil Test Results ===")
        print(f"  Total galaxies        : {n_total}")
        print(f"  Good fits (χ²<{chi2_threshold:.0f})    : "
              f"{n_good_fit} ({100 * frac_good_fit:.1f}%)")
        print(f"  Median χ²_ν           : {median_chi2:.4f}")
        btfr_str = f"{btfr_r:.4f}" if btfr_r is not None else "N/A"
        print(f"  BTFR Pearson r        : {btfr_str}")
        print(f"  Reliable β galaxies   : {n_reliable}")
        beta_str = f"{median_beta:.4f}" if median_beta is not None else "N/A"
        print(f"  Median β              : {beta_str}")
        print(f"  β near MOND (±{beta_tol}): "
              f"{n_near_mond} ({100 * frac_near_mond:.1f}%)")
        print(f"\nResults written to {json_path}")

    return results


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _write_summary(results: dict, path: Path) -> None:
    """Write a human-readable veil test summary to *path*."""
    fq = results["fit_quality"]
    bt = results["btfr"]
    ds = results["deep_slope"]
    vp = results["velos_prevalence"]

    lines = [
        "Motor de Velos SCM — Veil Test Summary",
        "=" * 40,
        f"Total galaxies analysed : {results['n_galaxies_total']}",
        "",
        "--- Fit Quality ---",
        f"  Good fits (χ²_ν < {fq['chi2_threshold_used']}) : "
        f"{fq['n_good_fit']} ({100 * fq['frac_good_fit']:.1f}%)",
        f"  Median χ²_ν           : {fq['median_chi2_reduced']:.4f}",
        f"  Median Υ_disk         : {fq['median_upsilon_disk']:.4f}",
        "",
        "--- Baryonic Tully-Fisher Relation ---",
        f"  N galaxies with Vflat : {bt['n_galaxies']}",
        f"  Pearson r (log Vflat vs log M_bar): {bt['pearson_r']}",
        f"  p-value               : {bt['p_value']}",
        "",
        "--- Deep-Regime Slope (β) Test ---",
        f"  Reliable β measurements  : {ds['n_reliable']}",
        f"  Median β                 : {ds['median_beta']}",
        f"  Expected (MOND/SCM)      : {ds['beta_target']}",
        (
            f"  Near MOND "
            f"(|β−{ds['beta_target']}|<{ds['beta_tolerance']}): "
            f"{ds['n_near_mond']} ({100 * ds['frac_near_mond']:.1f}%)"
        ),
        f"  Wilcoxon statistic       : {ds['wilcoxon_stat']}",
        f"  Wilcoxon p-value         : {ds['wilcoxon_p']}",
        "",
        "--- Velos Prevalence ---",
        (
            f"  Reliable deep-slope galaxies: "
            f"{vp['n_reliable_deep_slope']} "
            f"({100 * vp['frac_reliable_deep_slope']:.1f}%)"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Motor de Velos veil statistical validation on the full "
            "SPARC catalog. Produces results_overview.json with fit quality, "
            "BTFR, and deep-regime slope statistics."
        )
    )
    parser.add_argument(
        "--catalog", required=True,
        help="Full catalog CSV produced by build_sparc_full_catalog.py.",
    )
    parser.add_argument(
        "--out", default="results",
        help="Output directory (default: results).",
    )
    parser.add_argument(
        "--chi2-threshold", type=float, default=CHI2_GOOD_THRESHOLD,
        dest="chi2_threshold",
        help=f"χ²_ν cutoff for a good fit (default: {CHI2_GOOD_THRESHOLD}).",
    )
    parser.add_argument(
        "--beta-target", type=float, default=BETA_MOND_TARGET,
        dest="beta_target",
        help=f"Expected deep-regime β (default: {BETA_MOND_TARGET}).",
    )
    parser.add_argument(
        "--beta-tol", type=float, default=BETA_TOLERANCE,
        dest="beta_tol",
        help=f"Tolerance for |β - target| (default: {BETA_TOLERANCE}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = _parse_args(argv)
    run_veil_test(
        catalog_path=args.catalog,
        out_dir=args.out,
        chi2_threshold=args.chi2_threshold,
        beta_target=args.beta_target,
        beta_tol=args.beta_tol,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
