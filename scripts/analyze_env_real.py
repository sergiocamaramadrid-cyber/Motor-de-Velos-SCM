"""
scripts/analyze_env_real.py — OLS regression analysis of delta_f3 vs.
stellar mass and external environment proxy.

Loads a crossmatched CSV containing per-galaxy values for stellar mass
(``logM``), kinematic F3 residual (``delta_f3``), and an observational
environment proxy (``e_env``).  Fits two nested OLS models:

  * **Base model**:  delta_f3 ~ logM
  * **Full model**:  delta_f3 ~ logM + e_env

Both models use HC3 heteroscedasticity-robust standard errors.

A permutation test (shuffle ``e_env`` in fixed residuals) quantifies whether
the observed Spearman rank correlation between mass-model residuals and
``e_env`` is statistically significant beyond chance.

Outputs (written to ``--out`` directory):
  * ``env_real_analysis_table.csv`` — input table augmented with
    ``pred_mass`` and ``residual_mass`` columns.
  * ``env_real_summary.csv``        — single-row summary of key statistics.
  * ``env_real_summary.txt``        — human-readable key: value summary.

CLI usage
---------
::

    python scripts/analyze_env_real.py \\
        --input results/crossmatched_env.csv \\
        --out   results/env_real/

Required input columns
----------------------
``galaxy_name``, ``logM``, ``delta_f3``, ``e_env``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


def load_crossmatched_table(path: str | Path) -> pd.DataFrame:
    """Load and validate the crossmatched input table.

    Parameters
    ----------
    path : str | Path
        Path to a CSV with columns ``galaxy_name``, ``logM``,
        ``delta_f3``, and ``e_env``.

    Returns
    -------
    pd.DataFrame
        Cleaned table with non-finite rows dropped.

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    df = pd.read_csv(path)

    required = ["galaxy_name", "logM", "delta_f3", "e_env"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["galaxy_name"] = out["galaxy_name"].astype(str).str.strip()
    for col in ["logM", "delta_f3", "e_env"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["galaxy_name", "logM", "delta_f3", "e_env"]).reset_index(drop=True)
    return out


def fit_mass_only(df: pd.DataFrame):
    """Fit OLS: delta_f3 ~ logM (HC3 robust SEs).

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``logM`` and ``delta_f3``.

    Returns
    -------
    tuple[RegressionResultsWrapper, pd.Series, pd.Series]
        Fitted model, predicted values, and residuals.
    """
    X = sm.add_constant(df[["logM"]])
    y = df["delta_f3"]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    pred = model.predict(X)
    resid = y - pred
    return model, pred, resid


def fit_full(df: pd.DataFrame):
    """Fit OLS: delta_f3 ~ logM + e_env (HC3 robust SEs).

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``logM``, ``delta_f3``, and ``e_env``.

    Returns
    -------
    RegressionResultsWrapper
        Fitted full model.
    """
    X = sm.add_constant(df[["logM", "e_env"]])
    y = df["delta_f3"]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


def permutation_pvalue(
    residual: pd.Series,
    env: pd.Series,
    n_perms: int = 1000,
    seed: int | None = 42,
) -> tuple[float, float, list[float], float]:
    """Compute Spearman ρ and a permutation-based two-sided p-value.

    Shuffles ``env`` *n_perms* times and counts how often the permuted
    |ρ| exceeds the observed |ρ|.

    Parameters
    ----------
    residual : pd.Series
        Mass-model residuals.
    env : pd.Series
        Environment proxy values aligned with *residual*.
    n_perms : int
        Number of permutations (default 1000).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, list[float], float]
        ``(rho_obs, p_spearman, perm_rhos, p_perm)``
    """
    rho_obs, p_obs = spearmanr(residual, env)
    rng = np.random.default_rng(seed)
    perm = []

    env_values = env.to_numpy()
    residual_values = residual.to_numpy()

    for _ in range(n_perms):
        shuffled = rng.permutation(env_values)
        rho_perm, _ = spearmanr(residual_values, shuffled)
        perm.append(rho_perm)

    p_perm = float(np.mean(np.abs(perm) >= abs(rho_obs)))
    return float(rho_obs), float(p_obs), perm, p_perm


def summarize(
    df: pd.DataFrame,
    model_base,
    model_full,
    rho: float,
    p_spear: float,
    p_perm: float,
) -> dict:
    """Collect key statistics into a flat dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table (used only for row count).
    model_base : RegressionResultsWrapper
        Fitted base (mass-only) model.
    model_full : RegressionResultsWrapper
        Fitted full (mass + env) model.
    rho : float
        Spearman ρ between mass residuals and ``e_env``.
    p_spear : float
        Parametric Spearman p-value.
    p_perm : float
        Permutation-based two-sided p-value.

    Returns
    -------
    dict
        Flat summary with keys: N, rho_residual_env, p_spearman, p_perm,
        aic_base, aic_full, delta_aic, coef_logM_full, p_logM_full,
        coef_env_full, p_env_full, r2_base, r2_full.
    """
    return {
        "N": int(len(df)),
        "rho_residual_env": rho,
        "p_spearman": p_spear,
        "p_perm": p_perm,
        "aic_base": float(model_base.aic),
        "aic_full": float(model_full.aic),
        "delta_aic": float(model_base.aic - model_full.aic),
        "coef_logM_full": float(model_full.params["logM"]),
        "p_logM_full": float(model_full.pvalues["logM"]),
        "coef_env_full": float(model_full.params["e_env"]),
        "p_env_full": float(model_full.pvalues["e_env"]),
        "r2_base": float(model_base.rsquared),
        "r2_full": float(model_full.rsquared),
    }


def save_outputs(
    df: pd.DataFrame,
    summary: dict,
    out_dir: str | Path,
) -> None:
    """Write analysis table and summary files to *out_dir*.

    Creates three files:
    * ``env_real_analysis_table.csv``
    * ``env_real_summary.csv``
    * ``env_real_summary.txt``

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table (with ``pred_mass`` / ``residual_mass`` columns).
    summary : dict
        Flat summary dictionary produced by :func:`summarize`.
    out_dir : str | Path
        Directory to write outputs (created if absent).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "env_real_analysis_table.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "env_real_summary.csv", index=False)

    with open(out_dir / "env_real_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "OLS regression: delta_f3 ~ logM vs. delta_f3 ~ logM + e_env, "
            "with permutation p-value for the environment effect."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Crossmatched CSV with galaxy_name, logM, delta_f3, e_env",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_crossmatched_table(args.input)

    model_base, pred, resid = fit_mass_only(df)
    df["pred_mass"] = pred
    df["residual_mass"] = resid

    rho, p_spear, _, p_perm = permutation_pvalue(
        df["residual_mass"],
        df["e_env"],
        n_perms=args.n_perms,
        seed=args.seed,
    )

    model_full = fit_full(df)

    summary = summarize(
        df=df,
        model_base=model_base,
        model_full=model_full,
        rho=rho,
        p_spear=p_spear,
        p_perm=p_perm,
    )

    print(model_base.summary())
    print(model_full.summary())
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    save_outputs(df, summary, args.out)


if __name__ == "__main__":
    main()
