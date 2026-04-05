"""
scripts/final_analysis.py — Unified final analysis across SPARC, LITTLE THINGS,
and Yang with the data each dataset provides.

This is the paper-level synthesis script.  It orchestrates four analysis blocks
and writes a single cross-dataset summary:

  Block A — SPARC F3 distribution
      Reads a per-galaxy F3 catalog (friction_slope / beta column).
      Reports: N, β_mean ± std, β_median, t-test p-value vs β = 0.5,
      and MOND-consistency flag.

  Block B — LITTLE THINGS blind test + F3-equivalent β
      Applies the BTFR and interpolation models to the LITTLE THINGS
      global dataset (data/little_things_global.csv) and also derives
      a sample-level F3-equivalent slope using the global kinematic
      quantities available (g_obs from Vflat and specific angular
      momentum j):

          log(g_obs) = 3 · log(Vflat_km/s) − log(j_kpc·km/s) + C_unit

      Then fits:  log(g_obs) = β_LT · log(g_bar) + intercept

      Interpretation: β_LT ≈ 0.5 confirms that the LITTLE THINGS sample
      is in the deep-MOND regime, consistent with SPARC.

  Block C — SPARC + Yang robustness (BLOQUE FINAL)
      If an environmental catalog with a `delta_mass` column is provided
      via ``--env-catalog``, delegates to
      :func:`scripts.f3_robustness.run_robustness` and reports:
        - β_env (controlled regression HC3)
        - p_perm (stratified permutation)
        - ΔAIC 95 % CI (bootstrap)

  Block D — Cross-dataset β comparison
      Collects the β values from each dataset/method and prints a unified
      comparison table highlighting consistency with MOND (β ≈ 0.5).

Usage
-----
With a pre-computed SPARC F3 catalog and Yang env catalog::

    python scripts/final_analysis.py \\
        --f3-catalog  results/f3_catalog_real.csv \\
        --env-catalog results/delta_mass_yang_sparc.csv \\
        --out         results/final_analysis

With LITTLE THINGS only (no SPARC catalog)::

    python scripts/final_analysis.py \\
        --lt-csv   data/little_things_global.csv \\
        --out      results/final_analysis

All three datasets::

    python scripts/final_analysis.py \\
        --f3-catalog  results/f3_catalog_real.csv \\
        --env-catalog results/delta_mass_yang_sparc.csv \\
        --lt-csv      data/little_things_global.csv \\
        --n-perms     1000 \\
        --n-boot      1000 \\
        --seed        42 \\
        --out         results/final_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, ttest_1samp

# ---------------------------------------------------------------------------
# Physics / unit constants
# ---------------------------------------------------------------------------

KPC_TO_M: float = 3.085677581e19   # meters per kiloparsec (IAU 2012)
KMS_TO_MS: float = 1.0e3           # m/s per km/s
A0_DEFAULT: float = 1.2e-10        # characteristic acceleration m/s²

# log10 unit-conversion offset for LT g_obs derivation
# g_obs = Vflat_ms³ / j_m²s  →  need to convert kms/kpckms to SI
_LT_GOBS_OFFSET: float = (
    3.0 * math.log10(KMS_TO_MS) - math.log10(KPC_TO_M * KMS_TO_MS)
)

# MOND deep-regime expected β
BETA_MOND: float = 0.5
ALPHA: float = 0.05              # significance level

_SEP = "=" * 72

# Default path for the LITTLE THINGS global CSV
_DEFAULT_LT_CSV: str = str(
    Path(__file__).parent.parent / "data" / "little_things_global.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first matching column name from *aliases*, or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    return None


# ---------------------------------------------------------------------------
# Block A — SPARC F3 distribution
# ---------------------------------------------------------------------------

def _sparc_f3_block(f3_catalog_path: Path) -> dict:
    """Read per-galaxy F3 catalog and return β distribution statistics.

    Parameters
    ----------
    f3_catalog_path : Path
        Per-galaxy catalog CSV with columns ``friction_slope``/``beta``
        and ``reliable``/``velo_inerte_flag``.

    Returns
    -------
    dict with keys: dataset, n_galaxies, n_reliable, beta_mean, beta_median,
                    beta_std, t_stat, p_value, consistent_mond
    """
    df = pd.read_csv(f3_catalog_path)

    beta_col = _resolve_col(df, ["friction_slope", "beta", "f3"])
    if beta_col is None:
        raise ValueError(
            f"SPARC catalog has no F3 column (tried friction_slope/beta/f3).\n"
            f"Found columns: {list(df.columns)}"
        )

    reliable_col = _resolve_col(df, ["reliable", "velo_inerte_flag"])
    if reliable_col is None:
        # Fall back to all rows
        reliable = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
        warnings.warn(
            "No 'reliable' column found; using all rows.", RuntimeWarning, stacklevel=2
        )
    else:
        reliable = df[reliable_col].astype(bool)

    betas = pd.to_numeric(df[beta_col], errors="coerce")
    rel_betas = betas[reliable].dropna()

    n_galaxies = len(df)
    n_reliable = len(rel_betas)

    if n_reliable >= 2:
        beta_mean = float(rel_betas.mean())
        beta_median = float(rel_betas.median())
        beta_std = float(rel_betas.std())
        t_stat, p_value = ttest_1samp(rel_betas.values, BETA_MOND)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        beta_mean = beta_median = beta_std = float("nan")
        t_stat = p_value = float("nan")

    consistent_mond = (p_value > ALPHA) if math.isfinite(p_value) else False

    return {
        "dataset": "SPARC",
        "n_galaxies": n_galaxies,
        "n_reliable": n_reliable,
        "beta_mean": beta_mean,
        "beta_median": beta_median,
        "beta_std": beta_std,
        "t_stat": t_stat,
        "p_value": p_value,
        "consistent_mond": consistent_mond,
    }


# ---------------------------------------------------------------------------
# Block B — LITTLE THINGS blind test + F3-equivalent β
# ---------------------------------------------------------------------------

def compute_lt_gobs(
    logVobs: np.ndarray,
    log_j: np.ndarray,
) -> np.ndarray:
    """Derive log10(g_obs) from flat-velocity and specific angular momentum.

    At the flat-rotation regime:

        g_obs = V_flat² / r_eff   with   r_eff ≈ j / V_flat

    so  g_obs = V_flat³ / j.

    In log10 with Vflat in km/s and j in kpc·km/s:

        log10(g_obs / m·s⁻²) = 3·log10(V_flat_kms) − log10(j_kpckms)
                              + (3·log10(KMS_TO_MS) − log10(KPC_TO_M·KMS_TO_MS))

    Parameters
    ----------
    logVobs : array_like
        log10(Vflat / km/s).
    log_j : array_like
        log10(j / kpc·km/s).

    Returns
    -------
    ndarray
        log10(g_obs / m·s⁻²).
    """
    logVobs = np.asarray(logVobs, dtype=float)
    log_j = np.asarray(log_j, dtype=float)
    return 3.0 * logVobs - log_j + _LT_GOBS_OFFSET


def _lt_block(lt_csv_path: Path) -> dict:
    """Run LITTLE THINGS analysis.

    Computes:
    1. BTFR and interpolation model predictions (re-uses
       :mod:`scripts.blind_test_little_things` logic).
    2. F3-equivalent β for the whole LT sample.

    Parameters
    ----------
    lt_csv_path : Path
        Path to the LITTLE THINGS global CSV (must contain
        galaxy_id, logM, logVobs, log_gbar, log_j).

    Returns
    -------
    dict with keys: dataset, n_galaxies, beta_lt, beta_lt_err,
                    beta_lt_r, beta_lt_p,
                    rmse_btfr, rmse_interp, wilcoxon_p_interp,
                    consistent_mond
    """
    from scripts.blind_test_little_things import (
        load_dataset,
        predict_logv_btfr,
        predict_logv_interp,
    )
    from scipy.stats import wilcoxon as _wilcoxon

    df = load_dataset(lt_csv_path)
    n = len(df)

    # --- F3-equivalent β ---
    log_gobs = compute_lt_gobs(df["logVobs"].values, df["log_j"].values)
    log_gbar = df["log_gbar"].values
    valid = np.isfinite(log_gobs) & np.isfinite(log_gbar)
    sl, ic, r_val, p_val, se = linregress(log_gbar[valid], log_gobs[valid])
    consistent_mond = bool(
        abs(sl - BETA_MOND) < 2 * se  # β within 2σ of 0.5
    )

    # --- Blind test ---
    logV_btfr = predict_logv_btfr(df["logM"].values)
    logV_interp = predict_logv_interp(df["log_gbar"].values, df["log_j"].values)
    res_btfr = logV_btfr - df["logVobs"].values
    res_interp = logV_interp - df["logVobs"].values
    rmse_btfr = float(np.sqrt(np.mean(res_btfr**2)))
    rmse_interp = float(np.sqrt(np.mean(res_interp**2)))

    diff = np.abs(res_interp) - np.abs(res_btfr)
    nz = diff[diff != 0.0]
    wp_interp = float("nan")
    if len(nz) >= 5:
        try:
            _, wp_interp = _wilcoxon(nz, alternative="less")
            wp_interp = float(wp_interp)
        except Exception:
            pass

    return {
        "dataset": "LITTLE_THINGS",
        "n_galaxies": n,
        "beta_lt": float(sl),
        "beta_lt_err": float(se),
        "beta_lt_r": float(r_val),
        "beta_lt_p": float(p_val),
        "rmse_btfr": rmse_btfr,
        "rmse_interp": rmse_interp,
        "wilcoxon_p_interp": wp_interp,
        "consistent_mond": consistent_mond,
    }


# ---------------------------------------------------------------------------
# Block C — SPARC + Yang robustness
# ---------------------------------------------------------------------------

def _yang_robustness_block(
    f3_catalog_path: Path,
    env_catalog_path: Path,
    n_perms: int = 1000,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Run BLOQUE FINAL robustness analysis (SPARC × Yang).

    Parameters
    ----------
    f3_catalog_path : Path
        SPARC per-galaxy F3 catalog.
    env_catalog_path : Path
        Yang environmental proxy catalog (must contain galaxy + delta_mass).
    n_perms, n_boot : int
        Permutation and bootstrap counts.
    seed : int
        Master RNG seed.

    Returns
    -------
    dict with keys: dataset, reg_*, perm_*, boot_*
        (prefixed sub-dicts from :func:`scripts.f3_robustness.run_robustness`)
    """
    from scripts.f3_robustness import run_robustness

    f3_df = pd.read_csv(f3_catalog_path)
    env_df = pd.read_csv(env_catalog_path)

    join_key_f3 = _resolve_col(f3_df, ["galaxy", "name", "galname"])
    join_key_env = _resolve_col(env_df, ["galaxy", "name", "galname"])
    if join_key_f3 is None or join_key_env is None:
        raise ValueError(
            "Both the F3 catalog and env catalog must contain a 'galaxy' "
            f"(or 'name'/'galname') column for joining."
        )

    merged = f3_df.merge(
        env_df.rename(columns={join_key_env: join_key_f3}),
        on=join_key_f3,
        how="inner",
    )

    if "delta_mass" not in merged.columns:
        raise ValueError(
            "delta_mass column not found after merging F3 + env catalog. "
            f"env catalog columns: {list(env_df.columns)}"
        )

    reg, perm, boot = run_robustness(
        merged,
        n_perms=n_perms,
        n_boot=n_boot,
        seed=seed,
    )

    out: dict = {"dataset": "SPARC_Yang"}
    out.update({f"reg_{k}": v for k, v in reg.items()})
    out.update({f"perm_{k}": v for k, v in perm.items()})
    out.update({f"boot_{k}": v for k, v in boot.items()})
    return out, reg, perm, boot


# ---------------------------------------------------------------------------
# Block D — Cross-dataset β comparison table
# ---------------------------------------------------------------------------

def _build_comparison_table(
    sparc: dict | None,
    lt: dict | None,
) -> pd.DataFrame:
    """Build a unified β comparison DataFrame.

    Each row represents one dataset/method.  Columns:
      dataset, N, beta, beta_err, p_value_vs_05, consistent_mond
    """
    rows = []

    if sparc is not None:
        rows.append({
            "dataset": "SPARC",
            "method": "deep-regime log-log slope",
            "N": sparc.get("n_reliable", float("nan")),
            "beta": sparc.get("beta_median", float("nan")),
            "beta_std": sparc.get("beta_std", float("nan")),
            "p_vs_mond": sparc.get("p_value", float("nan")),
            "consistent_mond": sparc.get("consistent_mond", False),
        })

    if lt is not None:
        rows.append({
            "dataset": "LITTLE THINGS",
            "method": "F3-equiv from g_obs=Vflat³/j",
            "N": lt.get("n_galaxies", float("nan")),
            "beta": lt.get("beta_lt", float("nan")),
            "beta_std": lt.get("beta_lt_err", float("nan")),
            "p_vs_mond": lt.get("beta_lt_p", float("nan")),
            "consistent_mond": lt.get("consistent_mond", False),
        })

    if not rows:
        return pd.DataFrame(
            columns=["dataset", "method", "N", "beta", "beta_std",
                     "p_vs_mond", "consistent_mond"]
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_final_report(
    sparc: dict | None,
    lt: dict | None,
    yang: dict | None,
    reg: dict | None,
    perm: dict | None,
    boot: dict | None,
    comparison: pd.DataFrame,
) -> list[str]:
    """Format the four-block final report."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — FINAL ANALYSIS",
        "  SPARC · LITTLE THINGS · Yang",
        _SEP,
        "",
    ]

    # ─── Block A ─────────────────────────────────────────────────────────
    lines += [
        "─" * 72,
        "  BLOCK A — SPARC F3 distribution",
        "─" * 72,
    ]
    if sparc is None:
        lines.append("  [SKIPPED] No --f3-catalog provided.")
    else:
        lines += [
            f"  N galaxies (total)    : {sparc['n_galaxies']}",
            f"  N reliable β          : {sparc['n_reliable']}",
            f"  β mean ± std          : {sparc['beta_mean']:.4f} ± {sparc['beta_std']:.4f}",
            f"  β median              : {sparc['beta_median']:.4f}",
            f"  t-test p (vs β=0.5)   : {sparc['p_value']:.4e}",
            f"  MOND-consistent       : {'✅ YES' if sparc['consistent_mond'] else '❌ NO'}",
        ]

    lines += [""]

    # ─── Block B ─────────────────────────────────────────────────────────
    lines += [
        "─" * 72,
        "  BLOCK B — LITTLE THINGS blind test + F3-equivalent β",
        "─" * 72,
    ]
    if lt is None:
        lines.append("  [SKIPPED] No --lt-csv provided.")
    else:
        lines += [
            f"  N galaxies            : {lt['n_galaxies']}",
            "",
            f"  F3-equiv β (LT)       : {lt['beta_lt']:.4f} ± {lt['beta_lt_err']:.4f}",
            f"  Pearson r             : {lt['beta_lt_r']:.4f}   (p = {lt['beta_lt_p']:.2e})",
            f"  MOND-consistent       : {'✅ YES (|β−0.5| < 2σ)' if lt['consistent_mond'] else '⚠️  NO'}",
            "",
            "  Blind-test predictions:",
            f"    RMSE BTFR           : {lt['rmse_btfr']:.4f} dex",
            f"    RMSE interp         : {lt['rmse_interp']:.4f} dex",
            f"    Wilcoxon p (interp) : {lt['wilcoxon_p_interp']:.4f}"
            if math.isfinite(lt['wilcoxon_p_interp'])
            else f"    Wilcoxon p (interp) : N/A",
        ]

    lines += [""]

    # ─── Block C ─────────────────────────────────────────────────────────
    lines += [
        "─" * 72,
        "  BLOCK C — SPARC × Yang robustness (BLOQUE FINAL)",
        "─" * 72,
    ]
    if yang is None:
        lines.append("  [SKIPPED] No --env-catalog provided (or no --f3-catalog).")
    else:
        if reg and reg.get("statsmodels_available", True):
            daic = reg.get("delta_aic", float("nan"))
            lines += [
                f"  N galaxies (joined)   : {reg.get('n_galaxies', 'n/a')}",
                f"  Controls              : {reg.get('controls_used', [])}",
                f"  β_env (HC3 OLS)       : {reg.get('beta_env', float('nan')):.4f} "
                f"± {reg.get('beta_env_se', float('nan')):.4f}",
                f"  p_env                 : {reg.get('p_env', float('nan')):.4e}",
                f"  ΔAIC (base→full)      : {daic:.3f}  "
                f"({'✅ full preferred' if daic > 2 else '⚠️  weak'})",
            ]
        if perm:
            lines += [
                f"  Spearman ρ (obs)      : {perm.get('obs_rho', float('nan')):.4f}  "
                f"(p = {perm.get('obs_pval', float('nan')):.2e})",
                f"  p_perm (stratified)   : {perm.get('p_perm', float('nan')):.4f}  "
                f"({'✅ signal persists' if perm.get('p_perm', 1) < 0.05 else '⚠️  check'})",
            ]
        if boot and boot.get("statsmodels_available", True):
            frac = boot.get("frac_above_threshold", float("nan"))
            frac_s = f"{100*frac:.1f}%" if math.isfinite(frac) else "n/a"
            lines += [
                f"  Bootstrap ΔAIC mean   : {boot.get('boot_mean_delta_aic', float('nan')):.3f}",
                f"  Bootstrap ΔAIC 95%CI  : [{boot.get('ci_lo', float('nan')):.3f}, "
                f"{boot.get('ci_hi', float('nan')):.3f}]",
                f"  Fraction ΔAIC > 2     : {frac_s}",
            ]

    lines += [""]

    # ─── Block D ─────────────────────────────────────────────────────────
    lines += [
        "─" * 72,
        "  BLOCK D — Cross-dataset β comparison",
        "─" * 72,
    ]
    if comparison.empty:
        lines.append("  No datasets available for comparison.")
    else:
        lines += [
            f"  {'Dataset':<18} {'Method':<35} {'N':>5} {'β':>7} {'σ(β)':>7} "
            f"{'MOND?':>7}",
            "  " + "-" * 70,
        ]
        for _, row in comparison.iterrows():
            mond = "✅" if row["consistent_mond"] else "❌"
            beta_s = f"{row['beta']:.4f}" if math.isfinite(float(row["beta"])) else "  NaN"
            std_s = f"{row['beta_std']:.4f}" if math.isfinite(float(row["beta_std"])) else "  NaN"
            lines.append(
                f"  {str(row['dataset']):<18} {str(row['method']):<35} "
                f"{int(row['N']):>5} {beta_s:>7} {std_s:>7} {mond:>7}"
            )
        lines += [
            "",
            f"  Expected MOND value: β = {BETA_MOND:.1f}",
        ]

    lines += ["", _SEP]
    return lines


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_final_analysis(
    f3_catalog_path: Path | None = None,
    lt_csv_path: Path | None = None,
    env_catalog_path: Path | None = None,
    n_perms: int = 1000,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Run all four blocks and return a nested results dict.

    Parameters
    ----------
    f3_catalog_path : Path, optional
        SPARC F3 per-galaxy catalog (Block A + C).
    lt_csv_path : Path, optional
        LITTLE THINGS global CSV (Block B).
    env_catalog_path : Path, optional
        Yang environmental proxy CSV with delta_mass column (Block C).
    n_perms, n_boot : int
        Permutation / bootstrap counts for Block C.
    seed : int
        Master RNG seed.

    Returns
    -------
    dict with keys:
        sparc, lt, yang_flat, reg, perm, boot, comparison, report_lines
    """
    sparc: dict | None = None
    lt: dict | None = None
    yang: dict | None = None
    reg: dict | None = None
    perm: dict | None = None
    boot: dict | None = None

    if f3_catalog_path is not None:
        sparc = _sparc_f3_block(f3_catalog_path)

    if lt_csv_path is not None:
        lt = _lt_block(lt_csv_path)

    if f3_catalog_path is not None and env_catalog_path is not None:
        yang, reg, perm, boot = _yang_robustness_block(
            f3_catalog_path,
            env_catalog_path,
            n_perms=n_perms,
            n_boot=n_boot,
            seed=seed,
        )

    comparison = _build_comparison_table(sparc, lt)
    report_lines = format_final_report(sparc, lt, yang, reg, perm, boot, comparison)

    return {
        "sparc": sparc,
        "lt": lt,
        "yang": yang,
        "reg": reg,
        "perm": perm,
        "boot": boot,
        "comparison": comparison,
        "report_lines": report_lines,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified final analysis across SPARC, LITTLE THINGS, and Yang "
            "with the data each dataset provides."
        )
    )
    parser.add_argument(
        "--f3-catalog", default=None, dest="f3_catalog",
        metavar="FILE",
        help="SPARC per-galaxy F3 catalog CSV (friction_slope/beta + reliable).",
    )
    parser.add_argument(
        "--env-catalog", default=None, dest="env_catalog",
        metavar="FILE",
        help="Yang environmental proxy CSV with 'galaxy' and 'delta_mass' columns.",
    )
    parser.add_argument(
        "--lt-csv",
        default=_DEFAULT_LT_CSV,
        dest="lt_csv",
        metavar="FILE",
        help=(
            "LITTLE THINGS global CSV "
            "(default: data/little_things_global.csv)."
        ),
    )
    parser.add_argument(
        "--n-perms", type=int, default=1000, dest="n_perms",
        help="Permutation count for Block C (default: 1000).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=1000, dest="n_boot",
        help="Bootstrap count for Block C (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write results to this directory (log + JSON + CSV).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: parse args, run analysis, print and optionally write."""
    args = _parse_args(argv)

    # Resolve paths
    f3_path: Path | None = Path(args.f3_catalog) if args.f3_catalog else None
    env_path: Path | None = Path(args.env_catalog) if args.env_catalog else None
    lt_path_raw = Path(args.lt_csv)
    lt_path: Path | None = lt_path_raw if lt_path_raw.exists() else None

    # Validate supplied paths
    for label, p in [("--f3-catalog", f3_path), ("--env-catalog", env_path)]:
        if p is not None and not p.exists():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)
    if lt_path is None and args.lt_csv:
        # Only warn; LT block is optional
        warnings.warn(
            f"LITTLE THINGS CSV not found at {args.lt_csv}; Block B will be skipped.",
            RuntimeWarning,
            stacklevel=1,
        )

    results = run_final_analysis(
        f3_catalog_path=f3_path,
        lt_csv_path=lt_path,
        env_catalog_path=env_path,
        n_perms=args.n_perms,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    for line in results["report_lines"]:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Log
        (out_dir / "final_analysis.log").write_text(
            "\n".join(results["report_lines"]) + "\n", encoding="utf-8"
        )

        # JSON (comparison DF → list for JSON serialisation)
        serial: dict = {}
        for k, v in results.items():
            if k == "comparison":
                serial[k] = v.to_dict(orient="records") if v is not None else []
            elif k == "report_lines":
                serial[k] = v
            else:
                serial[k] = v
        with (out_dir / "final_analysis.json").open("w", encoding="utf-8") as fh:
            json.dump(serial, fh, indent=2, allow_nan=True)

        # CSV comparison table
        if not results["comparison"].empty:
            results["comparison"].to_csv(
                out_dir / "cross_dataset_beta.csv", index=False
            )

        print(f"\n  Results written to {out_dir}/")

    return results


if __name__ == "__main__":
    main()
