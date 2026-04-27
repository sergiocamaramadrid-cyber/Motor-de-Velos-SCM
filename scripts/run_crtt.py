"""
scripts/run_crtt.py — Conditional Regime Transition Test (CRTT).

Tests whether galaxy rotation-curve dynamics exhibit a **mass-dependent regime
transition** by scanning a range of baryonic mass thresholds and comparing:

- **Global model**: single OLS fit (slope_tail ~ logMbar) on the full sample.
- **Split model**: separate OLS fits below and above each threshold.

Model selection uses the corrected Akaike Information Criterion (AICc).  A
positive ΔAIC = AICc(global) − AICc(split) favours the split model (lower AICc
wins).  Statistical significance is assessed by a label-permutation test.

Key SCM result (SPARC, N≈100)
------------------------------
- No strong global law: best global ΔAIC < 2
- Regime transition detected at logMbar ≈ 9.7
  - ΔAIC ≈ 8.8  (split model preferred)
  - p_perm ≈ 0.043

Key result (LITTLE THINGS, N=26)
---------------------------------
- Weak replication: ΔAIC ≈ 2 (not robust, low N / dwarf regime)

Outputs (written to ``--out-dir``)
------------------------------------
``crtt_summary.json``   — JSON with best threshold, ΔAIC, p_perm
``crtt_summary.txt``    — human-readable report
``crtt_scan.csv``       — per-threshold scan results

Usage
-----
SPARC::

    python scripts/run_crtt.py \\
        --data   data/processed/sparc_catalog.csv \\
        --out-dir results/sparc

LITTLE THINGS::

    python scripts/run_crtt.py \\
        --data   data/little_things_global.csv \\
        --mass-col logM --slope-col logVobs \\
        --out-dir results/little_things
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCAN_MIN: float = 8.5
SCAN_MAX: float = 11.5
SCAN_STEP: float = 0.1
MIN_N_SPLIT: int = 5      # minimum galaxies per regime for a valid split (override with --min-n-split)
N_PERM: int = 1000
PERM_SEED: int = 42

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# AICc helpers
# ---------------------------------------------------------------------------

def _ols_aicc(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit OLS y ~ x and return (slope, intercept, AICc).

    Parameters
    ----------
    x, y : np.ndarray
        Predictor and response (1-D).

    Returns
    -------
    slope, intercept, aicc : float
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.inf
    result = scipy_stats.linregress(x, y)
    y_pred = result.slope * x + result.intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    # MLE estimate of sigma²
    sigma2 = ss_res / n
    if sigma2 <= 0:
        sigma2 = 1e-30
    log_lik = -n / 2 * (np.log(2 * np.pi * sigma2) + 1)
    k = 3  # slope + intercept + sigma
    aic = 2 * k - 2 * log_lik
    # AICc correction
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    return float(result.slope), float(result.intercept), float(aicc)


def compute_split_aicc(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    min_n_split: int = MIN_N_SPLIT,
) -> float:
    """Compute AICc for the split model at a given mass threshold.

    The split model fits separate OLS lines below and above *threshold*.
    The combined AICc is the sum of the two sub-model AICc values.

    Parameters
    ----------
    x : np.ndarray
        Mass predictor (logMbar).
    y : np.ndarray
        Slope response (slope_tail).
    threshold : float
        Mass threshold (same units as x).
    min_n_split : int
        Minimum points per regime required for a valid fit.

    Returns
    -------
    float
        Combined split AICc (np.inf if either sub-sample is too small).
    """
    lo = x < threshold
    hi = ~lo
    if lo.sum() < min_n_split or hi.sum() < min_n_split:
        return np.inf
    _, _, aicc_lo = _ols_aicc(x[lo], y[lo])
    _, _, aicc_hi = _ols_aicc(x[hi], y[hi])
    return aicc_lo + aicc_hi


# ---------------------------------------------------------------------------
# Threshold scan
# ---------------------------------------------------------------------------

def run_threshold_scan(
    df: pd.DataFrame,
    mass_col: str = "logMbar",
    slope_col: str = "slope_tail",
    scan_min: float = SCAN_MIN,
    scan_max: float = SCAN_MAX,
    scan_step: float = SCAN_STEP,
    min_n_split: int = MIN_N_SPLIT,
) -> pd.DataFrame:
    """Scan mass thresholds and compute ΔAIC at each step.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *mass_col* and *slope_col*.
    mass_col, slope_col : str
    scan_min, scan_max, scan_step : float
    min_n_split : int
        Minimum points per regime for a valid split fit.

    Returns
    -------
    pd.DataFrame
        Columns: ``threshold``, ``n_lo``, ``n_hi``, ``aicc_global``,
        ``aicc_split``, ``delta_aic``.
    """
    clean = df[[mass_col, slope_col]].dropna()
    x = clean[mass_col].values
    y = clean[slope_col].values

    _, _, aicc_global = _ols_aicc(x, y)

    thresholds = np.arange(scan_min, scan_max + scan_step / 2, scan_step)
    rows = []
    for thr in thresholds:
        lo = x < thr
        n_lo = int(lo.sum())
        n_hi = int((~lo).sum())
        aicc_split = compute_split_aicc(x, y, thr, min_n_split=min_n_split)
        delta_aic = aicc_global - aicc_split  # positive → split model wins
        rows.append({
            "threshold": round(float(thr), 2),
            "n_lo": n_lo,
            "n_hi": n_hi,
            "aicc_global": round(float(aicc_global), 4),
            "aicc_split": round(float(aicc_split), 4) if np.isfinite(aicc_split) else None,
            "delta_aic": round(float(delta_aic), 4) if np.isfinite(aicc_split) else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test_transition(
    df: pd.DataFrame,
    best_threshold: float,
    mass_col: str = "logMbar",
    slope_col: str = "slope_tail",
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    min_n_split: int = MIN_N_SPLIT,
) -> float:
    """Estimate p-value for the best ΔAIC via label permutation.

    Parameters
    ----------
    df : pd.DataFrame
    best_threshold : float
    mass_col, slope_col : str
    n_perm : int
    seed : int
    min_n_split : int

    Returns
    -------
    float
        Fraction of permutations with ΔAIC ≥ observed ΔAIC.
    """
    clean = df[[mass_col, slope_col]].dropna()
    x = clean[mass_col].values
    y = clean[slope_col].values

    _, _, aicc_global = _ols_aicc(x, y)
    obs_delta = aicc_global - compute_split_aicc(x, y, best_threshold,
                                                 min_n_split=min_n_split)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        _, _, aicc_g = _ols_aicc(x, y_perm)
        aicc_s = compute_split_aicc(x, y_perm, best_threshold,
                                    min_n_split=min_n_split)
        if np.isfinite(aicc_s) and (aicc_g - aicc_s) >= obs_delta:
            count += 1

    return count / n_perm


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_crtt(
    data: str | Path,
    out_dir: str | Path,
    mass_col: str = "logMbar",
    slope_col: str = "slope_tail",
    scan_min: float = SCAN_MIN,
    scan_max: float = SCAN_MAX,
    scan_step: float = SCAN_STEP,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    verbose: bool = True,
    min_n_split: int = MIN_N_SPLIT,
) -> dict:
    """Run the full CRTT pipeline and write results.

    Parameters
    ----------
    data : str or Path
        Input catalog CSV.
    out_dir : str or Path
        Directory for output files.
    mass_col, slope_col : str
        Column names for mass predictor and slope response.
    scan_min, scan_max, scan_step : float
        Threshold scan range and step size.
    n_perm : int
        Number of permutations for the null distribution.
    seed : int
        Random seed.
    verbose : bool
    min_n_split : int
        Minimum galaxies per regime for a valid split (default: 5).

    Returns
    -------
    dict
        Keys: ``n``, ``best_threshold``, ``best_delta_aic``, ``p_perm``,
        ``aicc_global``, ``conclusion``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data)
    clean = df[[mass_col, slope_col]].dropna()
    n = len(clean)
    if verbose:
        print(f"Loaded {len(df)} rows; {n} with valid {mass_col} and {slope_col}")

    scan = run_threshold_scan(
        df, mass_col=mass_col, slope_col=slope_col,
        scan_min=scan_min, scan_max=scan_max, scan_step=scan_step,
        min_n_split=min_n_split,
    )
    scan.to_csv(out_dir / "crtt_scan.csv", index=False)

    valid = scan.dropna(subset=["delta_aic"])
    if valid.empty:
        if verbose:
            print("No valid split thresholds found.")
        summary = {"n": n, "best_threshold": None, "best_delta_aic": None,
                   "p_perm": None, "aicc_global": None,
                   "conclusion": "no_valid_split"}
    else:
        best_row = valid.loc[valid["delta_aic"].idxmax()]
        best_thr = float(best_row["threshold"])
        best_delta = float(best_row["delta_aic"])
        aicc_global = float(best_row["aicc_global"])

        if verbose:
            print(f"Best threshold: logMbar = {best_thr:.2f}  ΔAIC = {best_delta:.2f}")

        p_perm = permutation_test_transition(
            df, best_thr, mass_col=mass_col, slope_col=slope_col,
            n_perm=n_perm, seed=seed, min_n_split=min_n_split,
        )
        if verbose:
            print(f"Permutation p-value: {p_perm:.4f}")

        if best_delta < 2:
            conclusion = "no_strong_transition"
        elif best_delta >= 2 and p_perm <= 0.05:
            conclusion = "significant_transition"
        else:
            conclusion = "marginal_transition"

        summary = {
            "n": n,
            "best_threshold": best_thr,
            "best_delta_aic": round(best_delta, 4),
            "p_perm": round(p_perm, 4),
            "aicc_global": round(aicc_global, 4),
            "conclusion": conclusion,
        }

    # Write JSON
    with open(out_dir / "crtt_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Write text report
    lines = [
        _SEP,
        "SCM — Conditional Regime Transition Test (CRTT)",
        _SEP,
        f"Input:          {data}",
        f"Sample size:    {n}",
        f"Mass column:    {mass_col}",
        f"Slope column:   {slope_col}",
        "",
    ]
    if summary["best_threshold"] is not None:
        lines += [
            f"Best threshold: logMbar = {summary['best_threshold']:.2f}",
            f"ΔAIC:           {summary['best_delta_aic']:.4f}",
            f"p_perm:         {summary['p_perm']:.4f}",
            f"AICc (global):  {summary['aicc_global']:.4f}",
            "",
            f"Conclusion:     {summary['conclusion']}",
        ]
    else:
        lines.append("Conclusion: no valid split thresholds found.")
    lines += ["", _SEP]
    report = "\n".join(lines)

    with open(out_dir / "crtt_summary.txt", "w") as fh:
        fh.write(report)
    if verbose:
        print(report)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Conditional Regime Transition Test (CRTT) for SCM."
    )
    parser.add_argument("--data", default="data/processed/sparc_catalog.csv",
                        help="Input catalog CSV")
    parser.add_argument("--out-dir", default="results/sparc",
                        help="Output directory")
    parser.add_argument("--mass-col", default="logMbar",
                        help="Mass predictor column (default: logMbar)")
    parser.add_argument("--slope-col", default="slope_tail",
                        help="Slope response column (default: slope_tail)")
    parser.add_argument("--scan-min", type=float, default=SCAN_MIN)
    parser.add_argument("--scan-max", type=float, default=SCAN_MAX)
    parser.add_argument("--scan-step", type=float, default=SCAN_STEP)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=PERM_SEED)
    parser.add_argument("--min-n-split", type=int, default=MIN_N_SPLIT,
                        help=f"Min points per regime (default: {MIN_N_SPLIT})")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args(argv)

    return run_crtt(
        data=args.data,
        out_dir=args.out_dir,
        mass_col=args.mass_col,
        slope_col=args.slope_col,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        scan_step=args.scan_step,
        n_perm=args.n_perm,
        seed=args.seed,
        verbose=args.verbose,
        min_n_split=args.min_n_split,
    )


if __name__ == "__main__":
    main()
