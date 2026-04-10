"""
scripts/scm_tr_regime_test.py — SPARC mass-regime analysis for the SCM outer slope.

Splits the SPARC galaxy sample at a baryonic mass threshold (default log M = 10.05)
and tests whether the environmental proxy (env_proxy) correlates with the outer
rotation-curve slope (slope_tail, F3_SCM) in each mass regime.

Theory
------
Environmental modulation of galaxy dynamics should appear primarily in the outer
rotation curve (F3_SCM ≡ d log V / d log r).  Mass effects must be controlled
first; once galaxies are split by baryonic mass, any residual env_proxy–slope
correlation traces environmental influence.

Expected results (from paper)
------------------------------
Low-mass  (N=22): ρ≈+0.01, p≈0.98  → no signal
High-mass (N=54): ρ≈-0.44, p≈8e-4  → significant negative correlation

Usage
-----
    python scripts/scm_tr_regime_test.py

    python scripts/scm_tr_regime_test.py \\
        --csv data/sparc_env.csv \\
        --threshold 10.05 \\
        --out results/main \\
        --n-boot 1000 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGM_THRESHOLD_DEFAULT: float = 10.05
N_BOOT_DEFAULT: int = 1000
MASS_COL: str = "logMbar"
ENV_COL: str = "env_proxy"
SLOPE_COL: str = "slope_tail"

CSV_DEFAULT = "data/sparc_env.csv"
OUT_DEFAULT = "results/main"

# ---------------------------------------------------------------------------
# Named tuples
# ---------------------------------------------------------------------------

FisherComparison = namedtuple(
    "FisherComparison", ["rho1", "n1", "rho2", "n2", "z_stat", "p_two_tail"]
)

BootstrapSummary = namedtuple(
    "BootstrapSummary", ["median", "ci_lo", "ci_hi", "n_boot"]
)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def fisher_z_from_r(r: float) -> float:
    """Return Fisher's Z-transformation: atanh(r)."""
    return math.atanh(r)


def fisher_compare_correlations(
    r1: float, n1: int, r2: float, n2: int
) -> FisherComparison:
    """Compare two independent Spearman correlations via Fisher Z-test.

    Parameters
    ----------
    r1, r2 : correlation coefficients for group 1 and 2
    n1, n2 : sample sizes for group 1 and 2

    Returns
    -------
    FisherComparison namedtuple with z_stat and two-tailed p-value.
    """
    from scipy.stats import norm

    z1 = fisher_z_from_r(r1)
    z2 = fisher_z_from_r(r2)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_two_tail = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    return FisherComparison(
        rho1=r1, n1=n1, rho2=r2, n2=n2, z_stat=z_stat, p_two_tail=p_two_tail
    )


def bootstrap_spearman(
    df: pd.DataFrame,
    env_col: str,
    slope_col: str,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = 42,
) -> BootstrapSummary:
    """Bootstrap the Spearman correlation between env_col and slope_col.

    Parameters
    ----------
    df       : DataFrame containing the two columns
    env_col  : name of the environmental proxy column
    slope_col: name of the slope column
    n_boot   : number of bootstrap resamples
    seed     : random seed for reproducibility

    Returns
    -------
    BootstrapSummary namedtuple (median, ci_lo, ci_hi, n_boot).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    x = df[env_col].to_numpy()
    y = df[slope_col].to_numpy()

    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i], _ = spearmanr(x[idx], y[idx])

    median = float(np.median(boot_rhos))
    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))
    return BootstrapSummary(median=median, ci_lo=ci_lo, ci_hi=ci_hi, n_boot=n_boot)


def run_mass_scan(
    df: pd.DataFrame,
    mass_col: str,
    env_col: str,
    slope_col: str,
    scan_min: float = 8.0,
    scan_max: float = 11.5,
    scan_step: float = 0.1,
    min_n: int = 10,
) -> pd.DataFrame:
    """Scan mass thresholds and compute Spearman correlation for high-mass subset.

    For each threshold, selects galaxies with mass_col >= threshold and computes
    the Spearman correlation between env_col and slope_col.

    Parameters
    ----------
    df         : full galaxy DataFrame
    mass_col   : column name for baryonic mass
    env_col    : environmental proxy column
    slope_col  : outer slope column
    scan_min   : start of threshold scan
    scan_max   : end of threshold scan
    scan_step  : step size
    min_n      : minimum number of galaxies required to compute correlation

    Returns
    -------
    DataFrame with columns: threshold, n_high, rho_high, pval_high
    """
    thresholds = np.arange(scan_min, scan_max + scan_step / 2, scan_step)
    rows = []
    for thr in thresholds:
        sub = df[df[mass_col] >= thr]
        n = len(sub)
        if n < min_n:
            continue
        rho, pval = spearmanr(sub[env_col], sub[slope_col])
        rows.append(
            {"threshold": round(float(thr), 6), "n_high": n, "rho_high": float(rho), "pval_high": float(pval)}
        )
    return pd.DataFrame(rows, columns=["threshold", "n_high", "rho_high", "pval_high"])


def run_hc3_ols(df: pd.DataFrame, env_col: str, slope_col: str) -> dict:
    """OLS regression of slope_col on env_col with HC3 heteroscedasticity-robust SEs.

    Parameters
    ----------
    df        : DataFrame with the two columns
    env_col   : predictor column
    slope_col : outcome column

    Returns
    -------
    dict with keys: slope, intercept, slope_pval, r2
    """
    import statsmodels.api as sm

    x = df[env_col].to_numpy(dtype=float)
    y = df[slope_col].to_numpy(dtype=float)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return {
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "slope_pval": float(model.pvalues[1]),
        "r2": float(model.rsquared),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the SPARC mass-regime correlation analysis.

    Returns
    -------
    dict with keys: low, high, bootstrap, fisher
        Each of low/high is a dict: {rho, pval, n}
        bootstrap: BootstrapSummary fields as dict
        fisher: FisherComparison fields as dict
    """
    parser = argparse.ArgumentParser(
        description="SPARC mass-regime outer-slope vs env analysis"
    )
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Input CSV path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOGM_THRESHOLD_DEFAULT,
        help="log Mbar split threshold",
    )
    parser.add_argument("--out", default=OUT_DEFAULT, help="Output directory")
    parser.add_argument(
        "--n-boot", type=int, default=N_BOOT_DEFAULT, help="Bootstrap resamples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    low_df  = df[df[MASS_COL] <  args.threshold].copy()
    high_df = df[df[MASS_COL] >= args.threshold].copy()

    rho_lo, pval_lo = spearmanr(low_df[ENV_COL],  low_df[SLOPE_COL])
    rho_hi, pval_hi = spearmanr(high_df[ENV_COL], high_df[SLOPE_COL])

    boot = bootstrap_spearman(
        high_df, ENV_COL, SLOPE_COL, n_boot=args.n_boot, seed=args.seed
    )

    fisher = fisher_compare_correlations(
        r1=float(rho_lo), n1=len(low_df),
        r2=float(rho_hi), n2=len(high_df),
    )

    # Build summary DataFrame
    rows = [
        {
            "regime":     "low",
            "n":          len(low_df),
            "rho":        round(float(rho_lo), 3),
            "pval":       round(float(pval_lo), 6),
            "boot_median": "",
            "boot_ci_lo":  "",
            "boot_ci_hi":  "",
        },
        {
            "regime":     "high",
            "n":          len(high_df),
            "rho":        round(float(rho_hi), 3),
            "pval":       round(float(pval_hi), 6),
            "boot_median": round(boot.median, 3),
            "boot_ci_lo":  round(boot.ci_lo, 3),
            "boot_ci_hi":  round(boot.ci_hi, 3),
        },
    ]
    summary_df = pd.DataFrame(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scm_tr_summary.csv"
    summary_df.to_csv(out_path, index=False)

    return {
        "low":       {"rho": float(rho_lo),  "pval": float(pval_lo),  "n": len(low_df)},
        "high":      {"rho": float(rho_hi),  "pval": float(pval_hi),  "n": len(high_df)},
        "bootstrap": boot._asdict(),
        "fisher":    fisher._asdict(),
    }


if __name__ == "__main__":
    result = main()
    lo = result["low"]
    hi = result["high"]
    bt = result["bootstrap"]
    print(f"Low  mass (N={lo['n']}): rho={lo['rho']:.3f}, p={lo['pval']:.3f}")
    print(f"High mass (N={hi['n']}): rho={hi['rho']:.3f}, p={hi['pval']:.6f}")
    print(f"Bootstrap CI: [{bt['ci_lo']:.3f}, {bt['ci_hi']:.3f}], median={bt['median']:.3f}")
