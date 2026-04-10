"""
run_environment_analysis.py
---------------------------
Reproducible analysis of environmental modulation of outer disk kinematics (ΔF₃)
in the SPARC galaxy sample within the SCM framework.

Input:  data/galaxy_catalog_with_env.csv  (columns: galaxy, logM, env_proxy, delta_f3)
Output: results/paper1_environment/tables/summary_results.csv  (written when run as main)

Usage
-----
    python scripts/run_environment_analysis.py
    python scripts/run_environment_analysis.py --input data/galaxy_catalog_with_env.csv

Public API
----------
    load_catalog(csv_path) -> pd.DataFrame
    run_analysis(df, mass_col, env_col, slope_col, mass_threshold) -> dict
    main(argv=None) -> dict
"""

import argparse
import csv
import os
import sys

import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT = os.path.join("data", "galaxy_catalog_with_env.csv")
DEFAULT_OUTPUT = os.path.join("results", "paper1_environment", "tables", "summary_results.csv")
MASS_THRESHOLD = 10.6  # log(M/M_sun), split point for mass sub-samples


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_catalog(csv_path: str) -> pd.DataFrame:
    """Load and validate the galaxy catalog CSV.

    Required columns: galaxy, logM, env_proxy, delta_f3.
    Rows with NaN in any required column are dropped.
    """
    required = {"logM", "env_proxy", "delta_f3"}
    df = pd.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    before = len(df)
    df = df.dropna(subset=list(required))
    if len(df) < before:
        print(f"[run_environment_analysis] Dropped {before - len(df)} rows with NaN.", file=sys.stderr)
    return df.reset_index(drop=True)


def run_analysis(
    df: pd.DataFrame,
    mass_col: str = "logM",
    env_col: str = "env_proxy",
    slope_col: str = "delta_f3",
    mass_threshold: float = MASS_THRESHOLD,
) -> dict:
    """Compute global and mass-split Spearman correlations.

    Returns a dict with keys:
        global_n, global_rho, global_p,
        low_n, low_rho, low_p,
        high_n, high_rho, high_p,
        mass_threshold
    """
    x = df[env_col].values
    y = df[slope_col].values

    rho, p = spearmanr(x, y)

    low  = df[df[mass_col] <  mass_threshold]
    high = df[df[mass_col] >= mass_threshold]

    rho_low,  p_low  = spearmanr(low[env_col],  low[slope_col])  if len(low)  > 2 else (float("nan"), float("nan"))
    rho_high, p_high = spearmanr(high[env_col], high[slope_col]) if len(high) > 2 else (float("nan"), float("nan"))

    return {
        "global_n":   len(df),
        "global_rho": round(rho,      4),
        "global_p":   round(p,        6),
        "low_n":      len(low),
        "low_rho":    round(rho_low,  4),
        "low_p":      round(p_low,    4),
        "high_n":     len(high),
        "high_rho":   round(rho_high, 4),
        "high_p":     round(p_high,   4),
        "mass_threshold": mass_threshold,
    }


def _print_results(res: dict) -> None:
    print("GLOBAL")
    print("N =",   res["global_n"])
    print("rho =", res["global_rho"])
    print("p =",   res["global_p"])

    print("\nLOW MASS")
    print(res["low_n"], res["low_rho"], res["low_p"])

    print("\nHIGH MASS")
    print(res["high_n"], res["high_rho"], res["high_p"])


def _write_summary_csv(res: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = [
        ["subset",      "n",           "rho_spearman",  "p_spearman", "mass_threshold"],
        ["global",      res["global_n"], res["global_rho"], res["global_p"], ""],
        ["low_mass",    res["low_n"],    res["low_rho"],    res["low_p"],    f"logM < {res['mass_threshold']}"],
        ["high_mass",   res["high_n"],   res["high_rho"],   res["high_p"],   f"logM >= {res['mass_threshold']}"],
    ]
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[run_environment_analysis] Results written to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="SCM environmental modulation analysis")
    parser.add_argument("--input",     default=DEFAULT_INPUT,  help="Input catalog CSV")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT, help="Output summary CSV")
    parser.add_argument("--threshold", default=MASS_THRESHOLD, type=float,
                        help="log(M) mass-split threshold (default: %(default)s)")
    args = parser.parse_args(argv)

    df  = load_catalog(args.input)
    res = run_analysis(df, mass_threshold=args.threshold)

    _print_results(res)
    _write_summary_csv(res, args.output)

    return res


if __name__ == "__main__":
    main()
