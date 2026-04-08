"""
scripts/sparc_ols_regression.py -- OLS regression analysis for the SCM
environmental correlation in the SPARC high-mass subsample.

Two nested models are estimated on galaxies with logM >= M_CRIT:

  Model 1 (simple):
      delta_f3 ~ delta_mass_std

  Model 2 (mass-controlled):
      delta_f3 ~ delta_mass_std + logM

Both models use HC3 heteroscedasticity-robust standard errors (White 1980,
MacKinnon & White 1985), which is appropriate for small-to-medium samples.

Theory
------
delta_f3 = slope_tail - 0.5

where slope_tail is the outer-disk dlogV/dlogr slope and 0.5 is the SCM
reference value (Motor-de-Velos deep form / MOND asymptotic slope).

delta_mass_std is the z-score of the angular momentum proxy, used as a
measure of the environmental tidal field.

The mass threshold M_CRIT = 10.05 is determined data-driven by the companion
script plot_sparc_mass_scan.py (maximises the composite signal score
S = |rho| * sqrt(N) * (-log10 p)).

Usage
-----

    python scripts/sparc_ols_regression.py

With optional arguments::

    python scripts/sparc_ols_regression.py \\
        --csv data/sparc_subset.csv \\
        --m-crit 10.05 \\
        --out results/ols_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_REF: float = 0.5
M_CRIT_DEFAULT: float = 10.05

_REQUIRED_COLS = {"slope_tail", "logM", "delta_mass_std"}
_REPO_ROOT = Path(__file__).parent.parent
_CSV_DEFAULT = str(_REPO_ROOT / "data" / "sparc_subset.csv")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def fit_models(df: pd.DataFrame, m_crit: float = M_CRIT_DEFAULT) -> dict:
    """Fit two OLS models on galaxies with logM >= m_crit.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with columns logM, delta_mass_std, delta_f3 (already
        computed).
    m_crit : float
        Minimum logM threshold for the subsample.

    Returns
    -------
    dict with keys:
        subsample   -- pd.DataFrame filtered to logM >= m_crit
        model1      -- fitted RegressionResultsWrapper (simple model)
        model2      -- fitted RegressionResultsWrapper (mass-controlled model)
        n           -- int, number of galaxies in subsample
        m_crit      -- float, the threshold used

    Raises
    ------
    ValueError
        If required columns are missing or the subsample has fewer than 4
        observations (cannot fit 3-parameter model).
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if "delta_f3" not in df.columns:
        missing.add("delta_f3")
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    sub = df[df["logM"] >= m_crit].copy()
    n = len(sub)
    if n < 4:
        raise ValueError(
            f"Only {n} galaxies with logM >= {m_crit}. "
            "Need at least 4 to fit both models."
        )

    y = sub["delta_f3"]

    # Model 1: simple
    X1 = sm.add_constant(sub[["delta_mass_std"]])
    model1 = sm.OLS(y, X1).fit(cov_type="HC3")

    # Model 2: with logM control
    X2 = sm.add_constant(sub[["delta_mass_std", "logM"]])
    model2 = sm.OLS(y, X2).fit(cov_type="HC3")

    return {
        "subsample": sub,
        "model1": model1,
        "model2": model2,
        "n": n,
        "m_crit": m_crit,
    }


def format_summary(result: dict) -> str:
    """Return a text summary of both fitted models.

    Parameters
    ----------
    result : dict
        Output of fit_models().

    Returns
    -------
    str
        Multi-line summary text.
    """
    lines = [
        f"SCM OLS Regression -- SPARC high-mass subsample (logM >= {result['m_crit']:.2f})",
        f"N = {result['n']} galaxies",
        "",
        "=== Model 1: Simple (delta_f3 ~ delta_mass_std) ===",
        result["model1"].summary().as_text(),
        "",
        "=== Model 2: Mass-controlled (delta_f3 ~ delta_mass_std + logM) ===",
        result["model2"].summary().as_text(),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OLS regression of delta_f3 on delta_mass_std for the SPARC "
            "high-mass subsample, with and without logM as a control variable."
        )
    )
    parser.add_argument(
        "--csv",
        default=_CSV_DEFAULT,
        help="Path to per-galaxy catalog CSV (default: data/sparc_subset.csv).",
    )
    parser.add_argument(
        "--m-crit",
        type=float,
        default=M_CRIT_DEFAULT,
        dest="m_crit",
        help=(
            f"logM threshold: only galaxies with logM >= m_crit are used "
            f"(default: {M_CRIT_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        dest="out",
        help="Optional path to save the text summary. Printed to stdout if omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the OLS regression pipeline.

    Returns
    -------
    dict with keys:
        subsample   -- pd.DataFrame of the high-mass subsample
        model1      -- fitted RegressionResultsWrapper (simple model)
        model2      -- fitted RegressionResultsWrapper (mass-controlled model)
        n           -- int, number of galaxies
        m_crit      -- float, the threshold used
        summary     -- str, formatted text summary
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {csv_path}\n"
            "Provide a CSV with columns: slope_tail, logM, delta_mass_std."
        )

    df = pd.read_csv(csv_path)
    missing_cols = _REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = df[list(_REQUIRED_COLS)].dropna().copy()
    df["delta_f3"] = df["slope_tail"] - BETA_REF

    result = fit_models(df, m_crit=args.m_crit)
    summary = format_summary(result)
    result["summary"] = summary

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary, encoding="utf-8")
        print(f"Summary saved to {out_path}", file=sys.stderr)
    else:
        print(summary)

    print(
        f"\nBest cut: logM = {result['m_crit']:.3f}  N = {result['n']}",
        file=sys.stderr,
    )

    return result


if __name__ == "__main__":
    main()
