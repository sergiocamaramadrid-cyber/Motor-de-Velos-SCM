#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCM-TR: Environmental Regime Transition test suite

Implements:
1) Low-mass vs high-mass Spearman correlations
2) Fisher z test for difference between correlations
3) Bootstrap CI for high-mass correlation
4) Continuous mass-threshold scan
5) Optional HC3 robust OLS in the high-mass regime

Expected columns by default:
- logMbar
- env_proxy
- slope_tail

Example:
python scripts/scm_tr_regime_test.py \
  --input results/paper1_environment/data/galaxy_catalog_with_env.csv \
  --outdir results/scm_tr \
  --low-cut 10.0 \
  --high-cut 10.1 \
  --bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


DEFAULT_MASS_COL = "logMbar"
DEFAULT_ENV_COL = "env_proxy"
DEFAULT_SLOPE_COL = "slope_tail"


@dataclass
class RegimeStats:
    label: str
    threshold_rule: str
    n: int
    rho_spearman: float
    p_value: float


@dataclass
class FisherComparison:
    z1: float
    z2: float
    se_diff: float
    z_stat: float
    p_two_sided: float
    n1: int
    n2: int


@dataclass
class BootstrapSummary:
    n_boot: int
    rho_observed: float
    ci95_low: float
    ci95_high: float
    frac_negative: float


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCM-TR regime transition tests")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--mass-col", default=DEFAULT_MASS_COL)
    parser.add_argument("--env-col", default=DEFAULT_ENV_COL)
    parser.add_argument("--slope-col", default=DEFAULT_SLOPE_COL)

    parser.add_argument("--low-cut", type=float, default=10.0,
                        help="Low-mass regime uses mass < low_cut")
    parser.add_argument("--high-cut", type=float, default=10.1,
                        help="High-mass regime uses mass > high_cut")

    parser.add_argument("--scan-min", type=float, default=9.5)
    parser.add_argument("--scan-max", type=float, default=10.5)
    parser.add_argument("--scan-step", type=float, default=0.05)
    parser.add_argument("--min-n", type=int, default=8,
                        help="Minimum sample size required in a scan subset")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run-hc3", action="store_true",
                        help="Run HC3 robust OLS in high-mass regime")
    return parser.parse_args(argv)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fisher_z_from_r(r: float) -> float:
    # Avoid infinities at |r|=1
    eps = 1e-12
    r = min(max(r, -1 + eps), 1 - eps)
    return 0.5 * math.log((1 + r) / (1 - r))


def fisher_compare_correlations(r1: float, n1: int, r2: float, n2: int) -> FisherComparison:
    if n1 <= 3 or n2 <= 3:
        raise ValueError("Fisher comparison requires n > 3 in both groups")

    z1 = fisher_z_from_r(r1)
    z2 = fisher_z_from_r(r2)
    se_diff = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se_diff
    p_two_sided = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    return FisherComparison(
        z1=z1,
        z2=z2,
        se_diff=se_diff,
        z_stat=z_stat,
        p_two_sided=p_two_sided,
        n1=n1,
        n2=n2,
    )


def clean_dataframe(df: pd.DataFrame, mass_col: str, env_col: str, slope_col: str) -> pd.DataFrame:
    needed = [mass_col, env_col, slope_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df[needed].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def spearman_stats(df: pd.DataFrame, env_col: str, slope_col: str,
                   label: str, rule: str) -> RegimeStats:
    n = len(df)
    if n < 2:
        return RegimeStats(label, rule, n, np.nan, np.nan)

    rho, p = spearmanr(df[env_col].to_numpy(), df[slope_col].to_numpy())
    return RegimeStats(label, rule, n, float(rho), float(p))


def bootstrap_spearman(df: pd.DataFrame, env_col: str, slope_col: str,
                       n_boot: int, seed: int) -> BootstrapSummary:
    rng = np.random.default_rng(seed)
    n = len(df)
    if n < 3:
        raise ValueError("Bootstrap requires at least 3 rows")

    x = df[env_col].to_numpy()
    y = df[slope_col].to_numpy()

    rho_obs, _ = spearmanr(x, y)

    boot = np.empty(n_boot, dtype=float)
    idx = np.arange(n)

    for i in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        rho_i, _ = spearmanr(x[sample_idx], y[sample_idx])
        boot[i] = rho_i

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    frac_negative = float(np.mean(boot < 0.0))

    return BootstrapSummary(
        n_boot=n_boot,
        rho_observed=float(rho_obs),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        frac_negative=frac_negative,
    )


def run_mass_scan(df: pd.DataFrame,
                  mass_col: str,
                  env_col: str,
                  slope_col: str,
                  scan_min: float,
                  scan_max: float,
                  scan_step: float,
                  min_n: int) -> pd.DataFrame:
    cuts = np.arange(scan_min, scan_max + 0.5 * scan_step, scan_step)
    rows: List[Dict] = []

    for cut in cuts:
        high = df[df[mass_col] > cut].copy()
        low = df[df[mass_col] < cut].copy()

        row: Dict = {"mass_cut": float(cut)}

        if len(high) >= min_n:
            rho_h, p_h = spearmanr(high[env_col], high[slope_col])
            row["n_high"] = int(len(high))
            row["rho_high"] = float(rho_h)
            row["p_high"] = float(p_h)
            row["minus_log10_p_high"] = float(-np.log10(max(p_h, 1e-300)))
        else:
            row["n_high"] = int(len(high))
            row["rho_high"] = np.nan
            row["p_high"] = np.nan
            row["minus_log10_p_high"] = np.nan

        if len(low) >= min_n:
            rho_l, p_l = spearmanr(low[env_col], low[slope_col])
            row["n_low"] = int(len(low))
            row["rho_low"] = float(rho_l)
            row["p_low"] = float(p_l)
        else:
            row["n_low"] = int(len(low))
            row["rho_low"] = np.nan
            row["p_low"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def run_hc3_ols(df: pd.DataFrame, env_col: str, slope_col: str) -> Dict:
    if not HAS_STATSMODELS:
        return {"available": False}

    X = sm.add_constant(df[[env_col]])
    y = df[slope_col]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    return {
        "available": True,
        "n": int(len(df)),
        "beta_env": float(model.params[env_col]),
        "beta_env_se_hc3": float(model.bse[env_col]),
        "beta_env_t_hc3": float(model.tvalues[env_col]),
        "beta_env_p_hc3": float(model.pvalues[env_col]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_plots(scan_df: pd.DataFrame,
               low_cut: float,
               high_cut: float,
               outdir: str) -> None:
    # Plot 1: rho_high vs mass_cut
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = scan_df["rho_high"].notna()
    ax.plot(scan_df.loc[valid, "mass_cut"], scan_df.loc[valid, "rho_high"])
    ax.axvline(low_cut, linestyle="--")
    ax.axvline(high_cut, linestyle="--")
    ax.set_xlabel("Mass threshold (logMbar cut)")
    ax.set_ylabel("Spearman rho (high-mass subset)")
    ax.set_title("SCM-TR mass scan: high-mass correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mass_scan_rho_high.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "mass_scan_rho_high.pdf"))
    plt.close(fig)

    # Plot 2: -log10(p_high) vs mass_cut
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = scan_df["minus_log10_p_high"].notna()
    ax.plot(scan_df.loc[valid, "mass_cut"], scan_df.loc[valid, "minus_log10_p_high"])
    ax.axhline(-np.log10(0.05), linestyle="--")
    ax.axvline(low_cut, linestyle="--")
    ax.axvline(high_cut, linestyle="--")
    ax.set_xlabel("Mass threshold (logMbar cut)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("SCM-TR mass scan: significance of high-mass correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mass_scan_logp_high.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "mass_scan_logp_high.pdf"))
    plt.close(fig)


def write_summary_text(summary: Dict, outpath: str) -> None:
    lines = []
    lines.append("SCM-TR regime test summary")
    lines.append("=" * 32)
    lines.append("")

    low = summary["low_regime"]
    high = summary["high_regime"]
    fisher = summary.get("fisher_comparison")
    boot = summary.get("bootstrap_high")
    hc3 = summary.get("hc3_high")

    lines.append("Low-mass regime")
    lines.append(f"  Rule: {low['threshold_rule']}")
    lines.append(f"  N: {low['n']}")
    lines.append(f"  Spearman rho: {low['rho_spearman']:.6f}")
    lines.append(f"  p-value: {low['p_value']:.6g}")
    lines.append("")

    lines.append("High-mass regime")
    lines.append(f"  Rule: {high['threshold_rule']}")
    lines.append(f"  N: {high['n']}")
    lines.append(f"  Spearman rho: {high['rho_spearman']:.6f}")
    lines.append(f"  p-value: {high['p_value']:.6g}")
    lines.append("")

    if fisher is not None:
        lines.append("Fisher comparison")
        lines.append(f"  z_stat: {fisher['z_stat']:.6f}")
        lines.append(f"  p_two_sided: {fisher['p_two_sided']:.6g}")
        lines.append("")

    if boot is not None:
        lines.append("Bootstrap (high-mass regime)")
        lines.append(f"  n_boot: {boot['n_boot']}")
        lines.append(f"  rho_observed: {boot['rho_observed']:.6f}")
        lines.append(f"  CI95: [{boot['ci95_low']:.6f}, {boot['ci95_high']:.6f}]")
        lines.append(f"  frac_negative: {boot['frac_negative']:.6f}")
        lines.append("")

    if hc3 is not None and hc3.get("available", False):
        lines.append("HC3 robust OLS (high-mass regime)")
        lines.append(f"  beta_env: {hc3['beta_env']:.6f}")
        lines.append(f"  beta_env_se_hc3: {hc3['beta_env_se_hc3']:.6f}")
        lines.append(f"  beta_env_p_hc3: {hc3['beta_env_p_hc3']:.6g}")
        lines.append(f"  r2: {hc3['r2']:.6f}")
        lines.append("")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None) -> Dict:
    args = parse_args(argv)
    ensure_dir(args.outdir)

    df_raw = pd.read_csv(args.input)
    df = clean_dataframe(df_raw, args.mass_col, args.env_col, args.slope_col)

    low_df = df[df[args.mass_col] < args.low_cut].copy()
    high_df = df[df[args.mass_col] > args.high_cut].copy()

    low_stats = spearman_stats(
        low_df, args.env_col, args.slope_col,
        label="low_mass",
        rule=f"{args.mass_col} < {args.low_cut}"
    )

    high_stats = spearman_stats(
        high_df, args.env_col, args.slope_col,
        label="high_mass",
        rule=f"{args.mass_col} > {args.high_cut}"
    )

    fisher_summary = None
    if (
        low_stats.n > 3 and high_stats.n > 3
        and np.isfinite(low_stats.rho_spearman)
        and np.isfinite(high_stats.rho_spearman)
    ):
        fisher_summary = asdict(
            fisher_compare_correlations(
                low_stats.rho_spearman, low_stats.n,
                high_stats.rho_spearman, high_stats.n
            )
        )

    bootstrap_summary = None
    if high_stats.n >= 3:
        bootstrap_summary = asdict(
            bootstrap_spearman(
                high_df, args.env_col, args.slope_col,
                n_boot=args.bootstrap,
                seed=args.seed
            )
        )

    scan_df = run_mass_scan(
        df=df,
        mass_col=args.mass_col,
        env_col=args.env_col,
        slope_col=args.slope_col,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        scan_step=args.scan_step,
        min_n=args.min_n,
    )
    scan_path = os.path.join(args.outdir, "mass_scan.csv")
    scan_df.to_csv(scan_path, index=False)

    make_plots(scan_df, args.low_cut, args.high_cut, args.outdir)

    hc3_summary = None
    if args.run_hc3:
        hc3_summary = run_hc3_ols(high_df, args.env_col, args.slope_col)

    summary = {
        "input_csv": args.input,
        "n_total_clean": int(len(df)),
        "columns": {
            "mass_col": args.mass_col,
            "env_col": args.env_col,
            "slope_col": args.slope_col,
        },
        "cuts": {
            "low_cut": args.low_cut,
            "high_cut": args.high_cut,
        },
        "low_regime": asdict(low_stats),
        "high_regime": asdict(high_stats),
        "fisher_comparison": fisher_summary,
        "bootstrap_high": bootstrap_summary,
        "hc3_high": hc3_summary,
        "scan_csv": scan_path,
    }

    save_json(summary, os.path.join(args.outdir, "scm_tr_summary.json"))
    write_summary_text(summary, os.path.join(args.outdir, "scm_tr_summary.txt"))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
