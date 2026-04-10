"""
scripts/scm_tr_yang.py — Yang-proxy validation of the SPARC mass-regime result.

Repeats the scm_tr_regime_test analysis using the Yang group-catalogue
environmental proxy (delta_mass_std: standardised halo mass offset) instead
of the SPARC env_proxy column.

Theory
------
If the env_proxy signal in the high-mass SPARC regime is real, it should
replicate with an independent environmental metric.  The Yang catalogue
provides delta_mass_std, a standardised measure of the galaxy's offset from
the mean halo mass at fixed stellar mass.

Expected results (from paper)
------------------------------
Low-mass  regime: no significant correlation
High-mass regime: significant negative correlation (env modulation)

Usage
-----
    python scripts/scm_tr_yang.py

    python scripts/scm_tr_yang.py \\
        --csv data/yang_dataset.csv \\
        --threshold 10.05 \\
        --out results/yang
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.scm_tr_regime_test import (
    LOGM_THRESHOLD_DEFAULT,
    MASS_COL,
    SLOPE_COL,
    bootstrap_spearman,
    fisher_compare_correlations,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YANG_ENV_COL: str = "delta_mass_std"

CSV_DEFAULT = "data/yang_dataset.csv"
OUT_DEFAULT = "results/yang"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the Yang-proxy mass-regime correlation analysis.

    Returns
    -------
    dict with keys: low, high, bootstrap, fisher, out_path
    """
    parser = argparse.ArgumentParser(
        description="Yang-proxy mass-regime outer-slope vs delta_mass_std analysis"
    )
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Input CSV path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=LOGM_THRESHOLD_DEFAULT,
        help="log Mbar split threshold",
    )
    parser.add_argument("--out", default=OUT_DEFAULT, help="Output directory")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    low_mask  = df[MASS_COL] <  args.threshold
    high_mask = df[MASS_COL] >= args.threshold

    df = df.copy()
    df["regime"] = np.where(high_mask, "high", "low")

    low_df  = df[low_mask].copy()
    high_df = df[high_mask].copy()

    rho_lo, pval_lo = spearmanr(low_df[YANG_ENV_COL],  low_df[SLOPE_COL])
    rho_hi, pval_hi = spearmanr(high_df[YANG_ENV_COL], high_df[SLOPE_COL])

    boot = bootstrap_spearman(high_df, YANG_ENV_COL, SLOPE_COL)

    fisher = fisher_compare_correlations(
        r1=float(rho_lo), n1=len(low_df),
        r2=float(rho_hi), n2=len(high_df),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scm_tr_yang_dataset.csv"
    df.to_csv(out_path, index=False)

    return {
        "low":      {"rho": float(rho_lo), "pval": float(pval_lo), "n": len(low_df)},
        "high":     {"rho": float(rho_hi), "pval": float(pval_hi), "n": len(high_df)},
        "bootstrap": boot._asdict(),
        "fisher":    fisher._asdict(),
        "out_path":  str(out_path),
    }


if __name__ == "__main__":
    result = main()
    lo = result["low"]
    hi = result["high"]
    bt = result["bootstrap"]
    print(f"Yang low  (N={lo['n']}): rho={lo['rho']:.3f}, p={lo['pval']:.3f}")
    print(f"Yang high (N={hi['n']}): rho={hi['rho']:.3f}, p={hi['pval']:.6f}")
    print(f"Bootstrap CI: [{bt['ci_lo']:.3f}, {bt['ci_hi']:.3f}]")
