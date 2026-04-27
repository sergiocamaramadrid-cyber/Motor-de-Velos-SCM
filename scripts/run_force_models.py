"""
scripts/run_force_models.py — Compare force-model families on SPARC galaxies.

Tests three force models against the observed outer-regime slope distribution:

  mond        — MOND deep-regime (β = 0.5 slope in log g_obs vs log g_bar)
  velos       — Motor de Velos baseline (V² = V_bar² + a0·r)
  newtonian   — Pure Newtonian (no dark matter, no modification)

Model comparison uses AICc evaluated on per-galaxy outer-slope residuals.  A
regime-dependent pattern emerges if the best model changes across the mass range.

Outputs (written to ``--out-dir``)
------------------------------------
``force_model_comparison.csv``  — per-galaxy best model and AICc values
``force_model_summary.json``    — aggregate statistics
``force_model_summary.txt``     — human-readable report

Usage
-----
::

    python scripts/run_force_models.py \\
        --data    data/processed/sparc_catalog.csv \\
        --out-dir results/sparc
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

A0: float = 1.2e-10          # characteristic acceleration (m/s²)
KPC_TO_M: float = 3.085677581e19
_CONV: float = 1e6 / KPC_TO_M  # (km/s)²/kpc → m/s²

EXPECTED_SLOPE_MOND: float = 0.5
EXPECTED_SLOPE_NEWTON: float = 1.0   # Keplerian outer regime
EXPECTED_SLOPE_VELOS: float = 0.5    # same deep limit as MOND

_SEP = "=" * 64

_DATA_DEFAULT = "data/processed/sparc_catalog.csv"
_OUT_DEFAULT = "results/sparc"


# ---------------------------------------------------------------------------
# Model residuals
# ---------------------------------------------------------------------------

def model_residual(
    slope_obs: float,
    logMbar: float,
    model: str,
    slope_col_mean: float = 0.0,
) -> float:
    """Compute the squared residual of a model's slope prediction.

    Each model predicts an expected outer slope; the residual measures
    how far the observed slope deviates.

    Parameters
    ----------
    slope_obs : float
        Observed outer-regime log-slope (slope_tail).
    logMbar : float
        Log10 baryonic mass (Msun).
    model : str
        One of ``'mond'``, ``'velos'``, ``'newtonian'``.
    slope_col_mean : float
        Mean of observed slopes (used for newtonian intercept correction).

    Returns
    -------
    float
        Squared residual.
    """
    if model == "mond":
        expected = EXPECTED_SLOPE_MOND
    elif model == "velos":
        expected = EXPECTED_SLOPE_VELOS
    elif model == "newtonian":
        # Newtonian flat-rotation prediction: slope ≈ 0 in the outer disc
        expected = 0.0
    else:
        raise ValueError(f"Unknown model: {model!r}")
    return (slope_obs - expected) ** 2


def _aicc_from_residuals(residuals: np.ndarray, k: int = 2) -> float:
    """Compute AICc from a vector of squared residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Squared residuals (one per galaxy).
    k : int
        Number of free parameters in the model.

    Returns
    -------
    float
    """
    n = len(residuals)
    if n < k + 2:
        return np.inf
    ss = float(np.sum(residuals))
    sigma2 = ss / n
    if sigma2 <= 0:
        sigma2 = 1e-30
    log_lik = -n / 2 * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * k - 2 * log_lik
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    return float(aicc)


# ---------------------------------------------------------------------------
# Per-galaxy comparison
# ---------------------------------------------------------------------------

def compare_models_per_galaxy(
    df: pd.DataFrame,
    slope_col: str = "slope_tail",
    mass_col: str = "logMbar",
) -> pd.DataFrame:
    """Assign the best force model to each galaxy.

    Parameters
    ----------
    df : pd.DataFrame
    slope_col : str
    mass_col : str

    Returns
    -------
    pd.DataFrame
        Original rows plus columns ``res_mond``, ``res_velos``,
        ``res_newtonian``, ``best_model``.
    """
    clean = df[[slope_col, mass_col]].dropna()
    slope_mean = float(clean[slope_col].mean())

    models = ["mond", "velos", "newtonian"]
    result = df.copy()
    for m in models:
        result[f"res_{m}"] = result.apply(
            lambda row: model_residual(
                row[slope_col], row[mass_col], m, slope_mean
            ) if pd.notna(row[slope_col]) and pd.notna(row[mass_col])
            else np.nan,
            axis=1,
        )

    res_cols = [f"res_{m}" for m in models]
    valid_mask = result[res_cols].notna().all(axis=1)
    result["best_model"] = pd.Series(dtype=object)
    best_labels = (
        result.loc[valid_mask, res_cols]
        .idxmin(axis=1)
        .str.replace("res_", "", regex=False)
    )
    result["best_model"] = result["best_model"].astype(object)
    result.loc[valid_mask, "best_model"] = best_labels.values
    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_model_stats(
    df: pd.DataFrame,
    slope_col: str = "slope_tail",
    mass_col: str = "logMbar",
) -> dict:
    """Compute per-model AICc and fraction-best statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``compare_models_per_galaxy``.
    slope_col, mass_col : str

    Returns
    -------
    dict
        Keys: ``n``, ``aicc_mond``, ``aicc_velos``, ``aicc_newtonian``,
        ``best_model_global``, ``fraction_best``.
    """
    models = ["mond", "velos", "newtonian"]
    clean = df.dropna(subset=[f"res_{m}" for m in models])
    n = len(clean)

    aicc_vals = {}
    for m in models:
        aicc_vals[f"aicc_{m}"] = round(_aicc_from_residuals(clean[f"res_{m}"].values), 4)

    best_global = min(aicc_vals, key=aicc_vals.get).replace("aicc_", "")

    fraction_best = {}
    for m in models:
        fraction_best[m] = round(float((clean["best_model"] == m).mean()), 4)

    return {
        "n": n,
        **aicc_vals,
        "best_model_global": best_global,
        "fraction_best_per_galaxy": fraction_best,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_force_models(
    data: str | Path = _DATA_DEFAULT,
    out_dir: str | Path = _OUT_DEFAULT,
    slope_col: str = "slope_tail",
    mass_col: str = "logMbar",
    verbose: bool = True,
) -> dict:
    """Run force-model comparison and write results.

    Parameters
    ----------
    data : str or Path
    out_dir : str or Path
    slope_col, mass_col : str
    verbose : bool

    Returns
    -------
    dict
        Aggregate statistics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data)
    if verbose:
        print(f"Loaded {len(df)} galaxies from {data}")

    df_cmp = compare_models_per_galaxy(df, slope_col=slope_col, mass_col=mass_col)
    df_cmp.to_csv(out_dir / "force_model_comparison.csv", index=False)

    stats = aggregate_model_stats(df_cmp, slope_col=slope_col, mass_col=mass_col)

    with open(out_dir / "force_model_summary.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    lines = [
        _SEP,
        "SCM — Force Model Comparison",
        _SEP,
        f"Input:   {data}",
        f"N:       {stats['n']}",
        "",
        "AICc per model (lower = better)",
        "-" * 40,
        f"  MOND:       {stats['aicc_mond']:.4f}",
        f"  Velos:      {stats['aicc_velos']:.4f}",
        f"  Newtonian:  {stats['aicc_newtonian']:.4f}",
        "",
        f"Best model (global AICc): {stats['best_model_global']}",
        "",
        "Fraction of galaxies where each model is best",
        "-" * 40,
    ]
    for m, frac in stats["fraction_best_per_galaxy"].items():
        lines.append(f"  {m:<12s}: {frac:.3f}")
    lines += ["", _SEP]
    report = "\n".join(lines)

    with open(out_dir / "force_model_summary.txt", "w") as fh:
        fh.write(report)
    if verbose:
        print(report)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Compare MOND / Velos / Newtonian force models on SPARC catalog."
    )
    parser.add_argument("--data", default=_DATA_DEFAULT, help="Input catalog CSV")
    parser.add_argument("--out-dir", default=_OUT_DEFAULT, help="Output directory")
    parser.add_argument("--slope-col", default="slope_tail")
    parser.add_argument("--mass-col", default="logMbar")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args(argv)

    return run_force_models(
        data=args.data,
        out_dir=args.out_dir,
        slope_col=args.slope_col,
        mass_col=args.mass_col,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
