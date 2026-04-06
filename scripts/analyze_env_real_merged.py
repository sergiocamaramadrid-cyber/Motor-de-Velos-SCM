"""
scripts/analyze_env_real_merged.py — Environmental-correlation analysis on the
pre-merged SPARC + Chae environment CSV.

Reads the output of ``build_env_real_input.py`` (or any CSV with the same
schema) and performs:

1. **OLS base model**  — ``delta_f3 ~ logM`` with HC3 heteroskedasticity-
   robust standard errors.
2. **Spearman + permutation test** — correlation between base-model residuals
   and the environmental proxy ``e_env`` (``n_perms`` shuffles of ``e_env``).
3. **OLS full model**  — ``delta_f3 ~ logM + e_env`` with HC3.
4. **Model comparison** — ΔAIC, ΔBIC, ΔR² (full − base).

Input schema
------------
The CSV must contain at minimum:

* ``galaxy_name`` **or** ``galaxy`` — galaxy identifier (string).
* ``logM``      — log₁₀ baryonic mass proxy.
* ``delta_f3``  — β − 0.5 (deviation from MOND deep-regime expectation).
* ``e_env``     — Chae environmental proxy.

An optional ``e_env_err`` column is retained in the per-galaxy output table
when present but does not influence the statistical models.

Outputs (written to ``out_dir``)
---------------------------------
``env_real_merged_table.csv``
    Per-galaxy table: galaxy_name, logM, delta_f3, e_env, residual_base,
    (e_env_err when present).
``env_real_merged_summary.csv``
    One-row machine-readable summary of all statistics.
``env_real_merged_summary.txt``
    Human-readable summary report.

Usage
-----
CLI::

    python scripts/analyze_env_real_merged.py \\
        --input  results/env_real/sparc_f3_chae_merged.csv \\
        --out    results/env_real/ \\
        --n-perms 2000 \\
        --seed   42

Programmatic (keyword API)::

    from scripts.analyze_env_real_merged import main as analyze_main

    analyze_main(
        input_path="results/env_real/sparc_f3_chae_merged.csv",
        out_dir="results/env_real/",
        n_perms=2000,
        seed=42,
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of permutations for the environment proxy shuffle test
DEFAULT_N_PERMS: int = 1000

#: Output file stems
_TABLE_STEM = "env_real_merged_table.csv"
_SUMMARY_CSV_STEM = "env_real_merged_summary.csv"
_SUMMARY_TXT_STEM = "env_real_merged_summary.txt"

_SEP = "=" * 68


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_merged_csv(path: str | Path) -> pd.DataFrame:
    """Load the pre-merged SPARC + Chae environment CSV.

    Accepts ``galaxy_name`` or ``galaxy`` as the identifier column (the former
    takes precedence).  The returned DataFrame always has a ``galaxy_name``
    column.

    Required columns (in addition to the identifier): ``logM``, ``delta_f3``,
    ``e_env``.  An optional ``e_env_err`` column is passed through unchanged.

    Parameters
    ----------
    path : str or Path
        Path to the merged CSV produced by ``build_env_real_input.py``.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame ready for analysis.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Merged CSV not found: {path}")

    df = pd.read_csv(path)

    # Normalise galaxy identifier column
    if "galaxy_name" not in df.columns and "galaxy" in df.columns:
        df = df.rename(columns={"galaxy": "galaxy_name"})
    elif "galaxy_name" not in df.columns:
        raise ValueError(
            f"Merged CSV {path} must contain 'galaxy_name' or 'galaxy' column; "
            f"found: {list(df.columns)}"
        )

    required = {"logM", "delta_f3", "e_env"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Merged CSV {path} missing columns: {sorted(missing)}; "
            f"found: {list(df.columns)}"
        )

    return df.copy()


# ---------------------------------------------------------------------------
# OLS models
# ---------------------------------------------------------------------------

def fit_ols_base(df: pd.DataFrame):
    """Fit the base OLS model: ``delta_f3 ~ logM`` with HC3.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``delta_f3`` and ``logM`` columns.

    Returns
    -------
    statsmodels RegressionResultsWrapper
        Fitted model result (HC3 covariance).
    """
    model = smf.ols("delta_f3 ~ logM", data=df)
    return model.fit(cov_type="HC3")


def fit_ols_full(df: pd.DataFrame):
    """Fit the full OLS model: ``delta_f3 ~ logM + e_env`` with HC3.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``delta_f3``, ``logM``, and ``e_env`` columns.

    Returns
    -------
    statsmodels RegressionResultsWrapper
        Fitted model result (HC3 covariance).
    """
    model = smf.ols("delta_f3 ~ logM + e_env", data=df)
    return model.fit(cov_type="HC3")


# ---------------------------------------------------------------------------
# Spearman + permutation test
# ---------------------------------------------------------------------------

def compute_spearman_permutation(
    residuals: np.ndarray,
    e_env: np.ndarray,
    n_perms: int = DEFAULT_N_PERMS,
    seed: int | None = None,
) -> dict[str, Any]:
    """Spearman correlation between base-model residuals and ``e_env``,
    with a permutation-based p-value.

    Parameters
    ----------
    residuals : array_like
        Residuals from the base OLS model.
    e_env : array_like
        Environmental proxy values (same length as *residuals*).
    n_perms : int
        Number of permutations (default: 1000).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``rho``    — Spearman correlation coefficient.
        ``p``      — Two-tailed asymptotic p-value from scipy.
        ``p_perm`` — Permutation-based p-value (fraction of |ρ_perm| ≥ |ρ|).
    """
    residuals = np.asarray(residuals, dtype=float)
    e_env = np.asarray(e_env, dtype=float)

    rho, p = _scipy_stats.spearmanr(residuals, e_env)

    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_perms):
        perm = rng.permutation(e_env)
        rho_perm, _ = _scipy_stats.spearmanr(residuals, perm)
        if abs(rho_perm) >= abs(rho):
            count_extreme += 1
    p_perm = count_extreme / n_perms

    return {"rho": float(rho), "p": float(p), "p_perm": float(p_perm)}


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compute_model_comparison(result_base, result_full) -> dict[str, float]:
    """Compute ΔAIC, ΔBIC, and ΔR² (full − base).

    Uses the *unrobust* AIC/BIC from statsmodels (which are based on the
    log-likelihood, not the robust covariance).

    Parameters
    ----------
    result_base : statsmodels RegressionResultsWrapper
        Fitted base model.
    result_full : statsmodels RegressionResultsWrapper
        Fitted full model.

    Returns
    -------
    dict with keys:
        ``delta_aic``  — AIC(full) − AIC(base).  Negative = full model better.
        ``delta_bic``  — BIC(full) − BIC(base).  Negative = full model better.
        ``delta_r2``   — R²(full) − R²(base).    Positive = full model better.
        ``coef_env``   — Coefficient on ``e_env`` in the full model.
        ``p_env``      — HC3 p-value for ``e_env`` in the full model.
    """
    delta_aic = float(result_full.aic - result_base.aic)
    delta_bic = float(result_full.bic - result_base.bic)
    delta_r2 = float(result_full.rsquared - result_base.rsquared)
    coef_env = float(result_full.params.get("e_env", float("nan")))
    p_env = float(result_full.pvalues.get("e_env", float("nan")))

    return {
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "delta_r2": delta_r2,
        "coef_env": coef_env,
        "p_env": p_env,
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_env_real_merged(
    df: pd.DataFrame,
    n_perms: int = DEFAULT_N_PERMS,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the full environmental-correlation analysis on the merged DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_merged_csv`.
    n_perms : int
        Number of permutations for the shuffle test.
    seed : int or None
        Random seed for the permutation test.

    Returns
    -------
    dict with keys:
        ``N``          — Number of galaxies analysed.
        ``rho``        — Spearman ρ (residuals vs e_env).
        ``p``          — Asymptotic Spearman p-value.
        ``p_perm``     — Permutation p-value.
        ``delta_aic``  — ΔAIC (full − base).
        ``delta_bic``  — ΔBIC (full − base).
        ``delta_r2``   — ΔR² (full − base).
        ``coef_env``   — β coefficient on e_env.
        ``p_env``      — HC3 p-value for e_env.
        ``df_table``   — Per-galaxy DataFrame with residuals appended.
    """
    df_fit = df.dropna(subset=["delta_f3", "logM", "e_env"]).copy()

    result_base = fit_ols_base(df_fit)
    result_full = fit_ols_full(df_fit)

    df_fit = df_fit.copy()
    df_fit["residual_base"] = result_base.resid.values

    spearman = compute_spearman_permutation(
        df_fit["residual_base"].values,
        df_fit["e_env"].values,
        n_perms=n_perms,
        seed=seed,
    )
    comparison = compute_model_comparison(result_base, result_full)

    # Build per-galaxy output table
    table_cols = ["galaxy_name", "logM", "delta_f3", "e_env", "residual_base"]
    if "e_env_err" in df_fit.columns:
        table_cols.append("e_env_err")
    df_table = df_fit[table_cols].reset_index(drop=True)

    return {
        "N": int(len(df_fit)),
        "rho": spearman["rho"],
        "p": spearman["p"],
        "p_perm": spearman["p_perm"],
        **comparison,
        "df_table": df_table,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _format_summary_report(stats: dict[str, Any], input_path: str) -> list[str]:
    """Return the human-readable summary as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — Environmental Analysis (Merged CSV)",
        _SEP,
        f"  Input        : {input_path}",
        f"  N galaxies   : {stats['N']}",
        "",
        "  --- Spearman (residuals vs e_env) ---",
        f"  ρ            : {stats['rho']:+.4f}",
        f"  p (asymp.)   : {stats['p']:.4e}",
        f"  p (perm.)    : {stats['p_perm']:.4f}",
        "",
        "  --- OLS full (delta_f3 ~ logM + e_env, HC3) ---",
        f"  coef_env     : {stats['coef_env']:+.4f}",
        f"  p_env        : {stats['p_env']:.4e}",
        "",
        "  --- Model comparison (full − base) ---",
        f"  ΔAIC         : {stats['delta_aic']:+.4f}",
        f"  ΔBIC         : {stats['delta_bic']:+.4f}",
        f"  ΔR²          : {stats['delta_r2']:+.4f}",
        _SEP,
    ]
    return lines


def save_outputs(
    stats: dict[str, Any],
    out_dir: str | Path,
    input_path: str,
) -> None:
    """Write table CSV, summary CSV, and summary TXT to *out_dir*.

    Parameters
    ----------
    stats : dict
        Return value of :func:`analyze_env_real_merged`.
    out_dir : str or Path
        Output directory (created if necessary).
    input_path : str
        Original input path string (used in the report only).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats["df_table"].to_csv(out_dir / _TABLE_STEM, index=False)

    summary_row = {k: v for k, v in stats.items() if k != "df_table"}
    pd.DataFrame([summary_row]).to_csv(out_dir / _SUMMARY_CSV_STEM, index=False)

    report_lines = _format_summary_report(stats, input_path)
    (out_dir / _SUMMARY_TXT_STEM).write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    for line in report_lines:
        print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Environmental-correlation analysis on the pre-merged "
            "SPARC + Chae environment CSV."
        )
    )
    parser.add_argument(
        "--input", dest="input_path", default=None,
        help="Path to the merged CSV (galaxy_name, logM, delta_f3, e_env).",
    )
    parser.add_argument(
        "--out", dest="out_dir", default=None, metavar="DIR",
        help="Output directory for table + summary files.",
    )
    parser.add_argument(
        "--n-perms", dest="n_perms", type=int, default=DEFAULT_N_PERMS,
        help=f"Number of permutations (default: {DEFAULT_N_PERMS}).",
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=None,
        help="Random seed for reproducibility (default: None).",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    input_path: str | None = None,
    out_dir: str | None = None,
    n_perms: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Entry point for CLI and programmatic use.

    Keyword arguments take precedence over parsed *argv* values.  When any
    keyword argument is provided and *argv* is ``None``, argparse receives an
    empty list (``[]``) so that ``sys.argv`` is not inadvertently consumed.

    Parameters
    ----------
    argv : list[str] or None
        Command-line argument list.  Pass ``[]`` or omit to use only keyword
        arguments.
    input_path : str or None
        Path to the merged input CSV.
    out_dir : str or None
        Output directory.  When ``None`` the outputs are not saved.
    n_perms : int or None
        Number of permutations.  Defaults to :data:`DEFAULT_N_PERMS`.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Statistics dict as returned by :func:`analyze_env_real_merged`
        (without the ``df_table`` key — use the written CSV for table access).

    Raises
    ------
    ValueError
        If *input_path* is ``None`` after merging argv + kwargs.
    """
    kwargs_provided = any(
        x is not None for x in [input_path, out_dir, n_perms, seed]
    )
    if kwargs_provided and argv is None:
        argv = []

    args = _parse_args(argv)

    resolved_input = input_path if input_path is not None else args.input_path
    resolved_out = out_dir if out_dir is not None else args.out_dir
    resolved_n_perms = n_perms if n_perms is not None else args.n_perms
    resolved_seed = seed if seed is not None else args.seed

    if resolved_input is None:
        raise ValueError(
            "Required argument not provided: --input / input_path"
        )

    df = load_merged_csv(resolved_input)
    stats = analyze_env_real_merged(df, n_perms=resolved_n_perms, seed=resolved_seed)

    if resolved_out is not None:
        save_outputs(stats, resolved_out, resolved_input)

    return {k: v for k, v in stats.items() if k != "df_table"}


if __name__ == "__main__":
    main()
