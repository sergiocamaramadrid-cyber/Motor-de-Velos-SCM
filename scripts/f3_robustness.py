"""
scripts/f3_robustness.py — BLOQUE FINAL: controlled regression, stratified
permutation, and bootstrap ΔAIC for the F3–environment correlation.

Three robustness blocks that close the main referee objections before
submission:

  Block 1 — Controlled regression
      F3 ~ log_M_bar + log_Rmax + δ_mass (HC3 robust OLS)
      Reports: β_env, p-value_env, ΔAIC(base vs full)

  Block 2 — Stratified permutation
      Shuffles δ_mass *within* stellar-mass bins (N_BINS=3) to preserve
      the M_bar distribution while destroying the environmental signal.
      Reports: observed ρ_Spearman, empirical p-value (p_perm), CI.

  Block 3 — Bootstrap ΔAIC
      1 000 galaxy-level bootstrap resamples.
      Reports: mean ΔAIC, 95% CI, fraction of resamples with ΔAIC > 2.

Column resolution
-----------------
The script resolves flexible column names so it works on catalogs produced
by both ``generate_f3_catalog.py`` (columns: beta / friction_slope) and any
join with an environmental proxy table.

  F3 observable   : ``friction_slope`` > ``beta`` > ``f3``
  Baryonic mass   : ``log_M_bar`` | ``log_mbar`` | log10(``M_bar_BTFR_Msun``)
  Max radius      : ``Rmax_kpc`` | ``rmax_kpc`` | ``r_max_kpc`` | ``Rmax``
  Environment     : ``delta_mass``

If ``log_M_bar`` or ``Rmax`` columns are absent the controlled regression and
stratified permutation degrade gracefully (mass/radius controls are dropped
or the permutation falls back to simple shuffle).

Usage
-----
Run on a joined catalog containing F3 and delta_mass::

    python scripts/f3_robustness.py \\
        --catalog results/f3_env_joined.csv \\
        --out     results/f3_robustness

Or join two catalogs on the fly::

    python scripts/f3_robustness.py \\
        --catalog   results/f3_catalog_real.csv \\
        --env-catalog results/delta_mass_yang_sparc.csv \\
        --out       results/f3_robustness \\
        --n-perms   1000 \\
        --n-boot    1000 \\
        --seed      42
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import statsmodels.formula.api as smf  # type: ignore[import-untyped]
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

_SEP = "=" * 70

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

N_PERMS_DEFAULT = 1000
N_BOOT_DEFAULT = 1000
SEED_DEFAULT = 42
N_MASS_BINS = 3
DELTA_AIC_STRONG_THRESHOLD = 2.0   # conventional threshold for "strong support"

# Rmax values above this are assumed to be in raw kpc; take log10 before use.
# Values below it are assumed to already be in log-scale or normalised units.
_RMAX_LINEAR_THRESHOLD = 10.0

# Minimum stellar mass in M_sun used as lower clip when deriving log_M_bar
# from a linear mass column (avoids log(0) for empty/erroneous entries).
_MIN_MASS_MSUN = 1.0


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

_F3_ALIASES = ["friction_slope", "beta", "f3"]
_LOG_MBAR_ALIASES = ["log_M_bar", "log_mbar", "log_Mbar"]
_RMAX_ALIASES = ["Rmax_kpc", "rmax_kpc", "r_max_kpc", "Rmax", "rmax"]


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first matching column name from *aliases*, or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def _prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Resolve columns and create standardised working frame.

    Returns
    -------
    work : pd.DataFrame
        Cleaned frame with canonical column names:
        ``f3``, ``delta_mass``, ``log_M_bar`` (optional), ``log_Rmax`` (optional).
    col_map : dict
        Mapping of canonical name → original source column (None if absent).
    """
    df = df.copy()
    col_map: dict[str, str | None] = {}

    # --- F3 observable ---
    f3_col = _resolve_column(df, _F3_ALIASES)
    if f3_col is None:
        raise ValueError(
            "No F3 column found. Expected one of: "
            + ", ".join(_F3_ALIASES)
        )
    col_map["f3"] = f3_col

    # --- delta_mass ---
    if "delta_mass" not in df.columns:
        raise ValueError("Column 'delta_mass' not found in catalog.")
    col_map["delta_mass"] = "delta_mass"

    # --- log baryonic mass ---
    log_mbar_col = _resolve_column(df, _LOG_MBAR_ALIASES)
    if log_mbar_col is not None:
        col_map["log_M_bar"] = log_mbar_col
    elif "M_bar_BTFR_Msun" in df.columns:
        df["_log_M_bar_derived"] = np.log10(
            df["M_bar_BTFR_Msun"].clip(lower=_MIN_MASS_MSUN)
        )
        col_map["log_M_bar"] = "_log_M_bar_derived"
    else:
        col_map["log_M_bar"] = None

    # --- log max radius ---
    rmax_col = _resolve_column(df, _RMAX_ALIASES)
    if rmax_col is not None:
        col_map["log_Rmax"] = rmax_col
    else:
        col_map["log_Rmax"] = None

    # Build standardised columns
    work = pd.DataFrame(index=df.index)
    work["f3"] = pd.to_numeric(df[col_map["f3"]], errors="coerce")
    work["delta_mass"] = pd.to_numeric(df[col_map["delta_mass"]], errors="coerce")

    if col_map["log_M_bar"] is not None:
        work["log_M_bar"] = pd.to_numeric(df[col_map["log_M_bar"]], errors="coerce")
    if col_map["log_Rmax"] is not None:
        raw_rmax = pd.to_numeric(df[col_map["log_Rmax"]], errors="coerce")
        # If values look like raw kpc (> _RMAX_LINEAR_THRESHOLD), take log10; otherwise store as-is
        if raw_rmax.median(skipna=True) > _RMAX_LINEAR_THRESHOLD:
            work["log_Rmax"] = np.log10(raw_rmax.clip(lower=0.01))
        else:
            work["log_Rmax"] = raw_rmax

    work = work.dropna(subset=["f3", "delta_mass"])
    return work, col_map


# ---------------------------------------------------------------------------
# Block 1 — Controlled OLS regression
# ---------------------------------------------------------------------------

def controlled_regression(
    df: pd.DataFrame,
) -> dict:
    """OLS regression F3 ~ controls + delta_mass_std with HC3 robust errors.

    Parameters
    ----------
    df : pd.DataFrame
        Working frame with columns ``f3``, ``delta_mass``, and optionally
        ``log_M_bar``, ``log_Rmax``.

    Returns
    -------
    dict with keys:
        n_galaxies, formula_base, formula_full,
        beta_env, beta_env_se, t_env, p_env,
        aic_base, aic_full, delta_aic,
        r2_base, r2_full,
        controls_used, statsmodels_available
    """
    if not _HAS_STATSMODELS:
        warnings.warn(
            "statsmodels not available; controlled regression skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"statsmodels_available": False}

    df = df.copy()

    # Standardise delta_mass for coefficient comparability
    dm_mean = df["delta_mass"].mean()
    dm_std = df["delta_mass"].std()
    df["delta_mass_std"] = (df["delta_mass"] - dm_mean) / (dm_std if dm_std > 0 else 1.0)

    # Decide which controls are available
    controls = []
    if "log_M_bar" in df.columns and df["log_M_bar"].notna().sum() > 5:
        controls.append("log_M_bar")
    if "log_Rmax" in df.columns and df["log_Rmax"].notna().sum() > 5:
        controls.append("log_Rmax")

    if controls:
        df_fit = df[["f3", "delta_mass_std"] + controls].dropna()
    else:
        df_fit = df[["f3", "delta_mass_std"]].dropna()

    if len(df_fit) < 10:
        warnings.warn(
            f"Only {len(df_fit)} rows after dropna; regression may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    ctrl_str = " + ".join(controls) if controls else "1"
    formula_base = f"f3 ~ {ctrl_str}"
    formula_full = f"f3 ~ {ctrl_str} + delta_mass_std"

    base_res = smf.ols(formula_base, data=df_fit).fit(cov_type="HC3")
    full_res = smf.ols(formula_full, data=df_fit).fit(cov_type="HC3")

    delta_aic = base_res.aic - full_res.aic  # positive → full model preferred

    beta_env = float(full_res.params.get("delta_mass_std", float("nan")))
    beta_env_se = float(full_res.bse.get("delta_mass_std", float("nan")))
    t_env = float(full_res.tvalues.get("delta_mass_std", float("nan")))
    p_env = float(full_res.pvalues.get("delta_mass_std", float("nan")))

    return {
        "n_galaxies": len(df_fit),
        "formula_base": formula_base,
        "formula_full": formula_full,
        "controls_used": controls,
        "beta_env": beta_env,
        "beta_env_se": beta_env_se,
        "t_env": t_env,
        "p_env": p_env,
        "aic_base": float(base_res.aic),
        "aic_full": float(full_res.aic),
        "delta_aic": float(delta_aic),
        "r2_base": float(base_res.rsquared),
        "r2_full": float(full_res.rsquared),
        "statsmodels_available": True,
    }


# ---------------------------------------------------------------------------
# Block 2 — Stratified permutation test
# ---------------------------------------------------------------------------

def stratified_permutation(
    df: pd.DataFrame,
    n_perms: int = N_PERMS_DEFAULT,
    rng: np.random.Generator | None = None,
    n_bins: int = N_MASS_BINS,
) -> dict:
    """Permutation test: shuffle delta_mass within stellar-mass bins.

    Preserves the marginal distribution of M_bar while destroying the
    environmental signal.  Falls back to simple global shuffle when
    ``log_M_bar`` is unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        Working frame with at least ``f3`` and ``delta_mass``.
    n_perms : int
        Number of permutations.
    rng : np.random.Generator, optional
        Random number generator (created with SEED_DEFAULT if None).
    n_bins : int
        Number of mass-quantile bins for stratified shuffle.

    Returns
    -------
    dict with keys:
        n_galaxies, n_perms,
        obs_rho, obs_pval,
        p_perm, ci_lo_rho, ci_hi_rho,
        perm_rho_mean, perm_rho_std,
        stratified
    """
    if rng is None:
        rng = np.random.default_rng(SEED_DEFAULT)

    work = df[["f3", "delta_mass"]].dropna().copy()
    has_mass = "log_M_bar" in df.columns and df["log_M_bar"].notna().sum() > 5
    if has_mass:
        work["log_M_bar"] = df.loc[work.index, "log_M_bar"]

    obs_result = spearmanr(work["f3"].values, work["delta_mass"].values)
    obs_rho = float(obs_result.statistic)
    obs_pval = float(obs_result.pvalue)

    # Build mass bins for stratified shuffle
    if has_mass:
        work_nonan = work.dropna(subset=["log_M_bar"])
        try:
            bins_series = pd.qcut(
                work_nonan["log_M_bar"], q=n_bins, labels=False, duplicates="drop"
            )
        except Exception:
            bins_series = None
        stratified = bins_series is not None
    else:
        work_nonan = work.copy()
        bins_series = None
        stratified = False

    perm_rhos: list[float] = []
    for _ in range(n_perms):
        dm_perm = work_nonan["delta_mass"].values.copy()

        if stratified and bins_series is not None:
            for bin_idx in range(int(bins_series.max()) + 1):
                mask = (bins_series == bin_idx).values
                dm_perm[mask] = rng.permutation(dm_perm[mask])
        else:
            dm_perm = rng.permutation(dm_perm)

        rho_perm = float(spearmanr(work_nonan["f3"].values, dm_perm).statistic)
        perm_rhos.append(rho_perm)

    perm_arr = np.array(perm_rhos)
    p_perm = float(np.mean(perm_arr >= obs_rho))
    ci_lo = float(np.percentile(perm_arr, 2.5))
    ci_hi = float(np.percentile(perm_arr, 97.5))

    return {
        "n_galaxies": len(work_nonan),
        "n_perms": n_perms,
        "obs_rho": obs_rho,
        "obs_pval": obs_pval,
        "p_perm": p_perm,
        "ci_lo_rho": ci_lo,
        "ci_hi_rho": ci_hi,
        "perm_rho_mean": float(perm_arr.mean()),
        "perm_rho_std": float(perm_arr.std()),
        "stratified": stratified,
    }


# ---------------------------------------------------------------------------
# Block 3 — Bootstrap ΔAIC
# ---------------------------------------------------------------------------

def bootstrap_delta_aic(
    df: pd.DataFrame,
    n_boot: int = N_BOOT_DEFAULT,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap galaxy-level ΔAIC (AIC_base − AIC_full).

    Each bootstrap resample draws N galaxies with replacement and fits
    the base and full OLS models.  The 95 % CI of ΔAIC excludes zero when
    the environmental term is robustly supported.

    Parameters
    ----------
    df : pd.DataFrame
        Working frame (same as for controlled_regression).
    n_boot : int
        Number of bootstrap resamples.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict with keys:
        n_galaxies, n_boot,
        observed_delta_aic, boot_mean_delta_aic,
        ci_lo, ci_hi,
        frac_above_threshold,
        statsmodels_available
    """
    if not _HAS_STATSMODELS:
        warnings.warn(
            "statsmodels not available; bootstrap ΔAIC skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"statsmodels_available": False}

    if rng is None:
        rng = np.random.default_rng(SEED_DEFAULT)

    df = df.copy()
    dm_mean = df["delta_mass"].mean()
    dm_std = df["delta_mass"].std()
    df["delta_mass_std"] = (df["delta_mass"] - dm_mean) / (dm_std if dm_std > 0 else 1.0)

    controls = []
    if "log_M_bar" in df.columns and df["log_M_bar"].notna().sum() > 5:
        controls.append("log_M_bar")
    if "log_Rmax" in df.columns and df["log_Rmax"].notna().sum() > 5:
        controls.append("log_Rmax")

    ctrl_str = " + ".join(controls) if controls else "1"
    formula_base = f"f3 ~ {ctrl_str}"
    formula_full = f"f3 ~ {ctrl_str} + delta_mass_std"

    fit_cols = ["f3", "delta_mass_std"] + controls
    df_fit = df[fit_cols].dropna().reset_index(drop=True)

    # Observed ΔAIC
    observed_delta_aic: float
    try:
        obs_base = smf.ols(formula_base, data=df_fit).fit()
        obs_full = smf.ols(formula_full, data=df_fit).fit()
        observed_delta_aic = float(obs_base.aic - obs_full.aic)
    except Exception:
        observed_delta_aic = float("nan")

    # Bootstrap
    delta_aics: list[float] = []
    n = len(df_fit)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = df_fit.iloc[idx].reset_index(drop=True)
        try:
            b_base = smf.ols(formula_base, data=sample).fit(disp=0)
            b_full = smf.ols(formula_full, data=sample).fit(disp=0)
            delta_aics.append(float(b_base.aic - b_full.aic))
        except Exception:
            continue  # skip degenerate resamples

    boot_arr = np.array(delta_aics)
    ci_lo = float(np.percentile(boot_arr, 2.5)) if len(boot_arr) else float("nan")
    ci_hi = float(np.percentile(boot_arr, 97.5)) if len(boot_arr) else float("nan")
    boot_mean = float(boot_arr.mean()) if len(boot_arr) else float("nan")
    frac_above = float(np.mean(boot_arr > DELTA_AIC_STRONG_THRESHOLD)) if len(boot_arr) else float("nan")

    return {
        "n_galaxies": n,
        "n_boot": n_boot,
        "n_boot_valid": len(boot_arr),
        "observed_delta_aic": observed_delta_aic,
        "boot_mean_delta_aic": boot_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "frac_above_threshold": frac_above,
        "statsmodels_available": True,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    reg: dict,
    perm: dict,
    boot: dict,
) -> list[str]:
    """Format the three-block robustness report."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — F3 Robustness Analysis (BLOQUE FINAL)",
        _SEP,
        "",
        "─" * 70,
        "  BLOCK 1 — Controlled OLS regression (HC3 robust errors)",
        "─" * 70,
    ]

    if not reg.get("statsmodels_available", True):
        lines.append("  [SKIPPED] statsmodels not installed.")
    else:
        lines += [
            f"  N galaxies      : {reg.get('n_galaxies', 'n/a')}",
            f"  Controls used   : {reg.get('controls_used', [])}",
            f"  Formula (base)  : {reg.get('formula_base', 'n/a')}",
            f"  Formula (full)  : {reg.get('formula_full', 'n/a')}",
            "",
            f"  β_env           : {reg.get('beta_env', float('nan')):.4f}",
            f"  SE(β_env)       : {reg.get('beta_env_se', float('nan')):.4f}",
            f"  t(β_env)        : {reg.get('t_env', float('nan')):.3f}",
            f"  p(β_env)        : {reg.get('p_env', float('nan')):.4e}",
            "",
            f"  AIC (base)      : {reg.get('aic_base', float('nan')):.2f}",
            f"  AIC (full)      : {reg.get('aic_full', float('nan')):.2f}",
            f"  ΔAIC            : {reg.get('delta_aic', float('nan')):.3f}  "
            f"({'✅ full model preferred' if reg.get('delta_aic', 0) > DELTA_AIC_STRONG_THRESHOLD else '⚠️ weak evidence'})",
            f"  R² (base)       : {reg.get('r2_base', float('nan')):.4f}",
            f"  R² (full)       : {reg.get('r2_full', float('nan')):.4f}",
        ]

    lines += [
        "",
        "─" * 70,
        "  BLOCK 2 — Stratified permutation test",
        "─" * 70,
        f"  Stratified       : {perm.get('stratified', False)} "
        f"({'mass bins' if perm.get('stratified') else 'global shuffle'})",
        f"  N galaxies       : {perm.get('n_galaxies', 'n/a')}",
        f"  N permutations   : {perm.get('n_perms', 'n/a')}",
        "",
        f"  Observed ρ       : {perm.get('obs_rho', float('nan')):.4f}  "
        f"(p_Spearman = {perm.get('obs_pval', float('nan')):.4e})",
        f"  Null ρ mean      : {perm.get('perm_rho_mean', float('nan')):.4f}  "
        f"± {perm.get('perm_rho_std', float('nan')):.4f}",
        f"  Null ρ 95% CI    : [{perm.get('ci_lo_rho', float('nan')):.4f}, "
        f"{perm.get('ci_hi_rho', float('nan')):.4f}]",
        f"  p_perm           : {perm.get('p_perm', float('nan')):.4f}  "
        f"({'✅ signal persists' if perm.get('p_perm', 1) < 0.05 else '⚠️ not significant'})",
    ]

    lines += [
        "",
        "─" * 70,
        "  BLOCK 3 — Bootstrap ΔAIC",
        "─" * 70,
    ]

    if not boot.get("statsmodels_available", True):
        lines.append("  [SKIPPED] statsmodels not installed.")
    else:
        frac = boot.get("frac_above_threshold", float("nan"))
        frac_str = f"{100*frac:.1f}%" if not np.isnan(frac) else "n/a"
        lines += [
            f"  N galaxies       : {boot.get('n_galaxies', 'n/a')}",
            f"  N boot (valid)   : {boot.get('n_boot_valid', 'n/a')} / {boot.get('n_boot', 'n/a')}",
            "",
            f"  Observed ΔAIC    : {boot.get('observed_delta_aic', float('nan')):.3f}",
            f"  Bootstrap mean   : {boot.get('boot_mean_delta_aic', float('nan')):.3f}",
            f"  95% CI           : [{boot.get('ci_lo', float('nan')):.3f}, "
            f"{boot.get('ci_hi', float('nan')):.3f}]",
            f"  Fraction > {DELTA_AIC_STRONG_THRESHOLD:.0f}   : {frac_str}  "
            f"({'✅ robust' if not np.isnan(frac) and frac > 0.80 else '⚠️ check CI'})",
        ]

    lines += ["", _SEP]
    return lines


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_robustness(
    df: pd.DataFrame,
    n_perms: int = N_PERMS_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> tuple[dict, dict, dict]:
    """Run all three robustness blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-merged catalog with F3 and delta_mass columns.
    n_perms, n_boot : int
        Permutation and bootstrap counts.
    seed : int
        Master RNG seed (used to derive seeds for each block).

    Returns
    -------
    (reg_results, perm_results, boot_results) : tuple[dict, dict, dict]
    """
    work, _col_map = _prepare_dataframe(df)

    rng_perm = np.random.default_rng(seed)
    rng_boot = np.random.default_rng(seed + 1)

    reg = controlled_regression(work)
    perm = stratified_permutation(work, n_perms=n_perms, rng=rng_perm)
    boot = bootstrap_delta_aic(work, n_boot=n_boot, rng=rng_boot)

    return reg, perm, boot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BLOQUE FINAL: controlled regression + stratified permutation + "
            "bootstrap ΔAIC for the F3–environment correlation."
        )
    )
    parser.add_argument(
        "--catalog", required=True,
        help=(
            "Per-galaxy catalog CSV with F3 (friction_slope / beta) and optionally "
            "log_M_bar, Rmax_kpc, delta_mass columns."
        ),
    )
    parser.add_argument(
        "--env-catalog", default=None, dest="env_catalog",
        help=(
            "Optional separate CSV with delta_mass column to join on 'galaxy'. "
            "Used when delta_mass is not in --catalog."
        ),
    )
    parser.add_argument(
        "--n-perms", type=int, default=N_PERMS_DEFAULT, dest="n_perms",
        help=f"Number of permutations (default: {N_PERMS_DEFAULT}).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=N_BOOT_DEFAULT, dest="n_boot",
        help=f"Number of bootstrap resamples (default: {N_BOOT_DEFAULT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED_DEFAULT,
        help=f"Master RNG seed (default: {SEED_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write results to this directory (CSV + JSON + log).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[dict, dict, dict]:
    """Entry point: parse args, run, print and optionally write results."""
    args = _parse_args(argv)
    cat_path = Path(args.catalog)

    if not cat_path.exists():
        print(f"ERROR: catalog not found: {cat_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(cat_path)

    if args.env_catalog is not None:
        env_path = Path(args.env_catalog)
        if not env_path.exists():
            print(f"ERROR: env-catalog not found: {env_path}", file=sys.stderr)
            sys.exit(1)
        env_df = pd.read_csv(env_path)
        join_key = next(
            (c for c in ["galaxy", "name", "galname"] if c in env_df.columns),
            None,
        )
        if join_key is None:
            print(
                "ERROR: env-catalog must contain a 'galaxy' column for joining.",
                file=sys.stderr,
            )
            sys.exit(1)
        df_key = next(
            (c for c in ["galaxy", "name", "galname"] if c in df.columns), None
        )
        if df_key is None:
            print(
                "ERROR: --catalog must contain a 'galaxy' column for joining.",
                file=sys.stderr,
            )
            sys.exit(1)
        df = df.merge(
            env_df.rename(columns={join_key: df_key}),
            on=df_key,
            how="inner",
        )

    reg, perm, boot = run_robustness(
        df,
        n_perms=args.n_perms,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    report_lines = format_report(reg, perm, boot)
    for line in report_lines:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Log
        log_path = out_dir / "f3_robustness.log"
        log_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        # JSON
        results_json = {"regression": reg, "permutation": perm, "bootstrap": boot}
        json_path = out_dir / "f3_robustness.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(results_json, fh, indent=2, allow_nan=True)

        # CSV (flat, one row)
        flat = {f"reg_{k}": v for k, v in reg.items()}
        flat.update({f"perm_{k}": v for k, v in perm.items()})
        flat.update({f"boot_{k}": v for k, v in boot.items()})
        pd.DataFrame([flat]).to_csv(out_dir / "f3_robustness_summary.csv", index=False)

        print(f"\n  Results written to {out_dir}/")

    return reg, perm, boot


if __name__ == "__main__":
    main()
