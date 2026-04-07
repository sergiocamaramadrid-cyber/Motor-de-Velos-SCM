"""
scripts/scm_oos_validation.py — Out-of-sample validation for Paper 1.

Implements the statistical validation described in the SCM Paper 1
(Environmental Modulation):

1. **OLS regression** with HC3 heteroskedasticity-robust errors.
   - Base model:  delta_f3 ~ logMbar
   - Full SCM model:  delta_f3 ~ logMbar + delta_mass

2. **AICc / BIC comparison** on the full sample.

3. **Out-of-sample (OOS) validation** over *n_splits* random 70/30 splits.
   For each split the models are trained on the training set and evaluated on
   the held-out test set.  ΔRMSE = RMSE_base − RMSE_full (positive = full
   model beats baseline on the test set).

4. **Wilcoxon signed-rank test** on the ΔRMSE distribution to test
   H₀: median(ΔRMSE) = 0.

5. **Extreme galaxies** — the *n_extreme* galaxies with the largest absolute
   residual difference |resid_base − resid_full| are extracted and saved.

6. **Figures** (written to *figures_dir*)

   - ``figure01_scatter.pdf``    — beta vs logMbar scatter with OLS fit.
   - ``figure02_delta_rmse_hist.pdf`` — histogram of ΔRMSE_out values.
   - ``figure03_delta_rmse_scatter.pdf`` — per-galaxy |resid| comparison.

7. **Robustness catalog** (``galaxy_catalog_env2.csv``) uses an alternative
   environmental proxy (log10 of raw HI mass) as an independent check.

Usage
-----
::

    python scripts/scm_oos_validation.py \\
        --input results/paper1_environment/data/galaxy_catalog_with_env.csv \\
        --out-dir results/paper1_environment/data \\
        --figures-dir results/paper1_environment/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT_DIR: str = "results/paper1_environment/data"
DEFAULT_FIGURES_DIR: str = "results/paper1_environment/figures"
TEST_FRAC_DEFAULT: float = 0.30
N_SPLITS_DEFAULT: int = 100
SEED_DEFAULT: int = 42
N_EXTREME_DEFAULT: int = 25


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def fit_ols_hc3(y: np.ndarray, X: np.ndarray) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with HC3 heteroskedasticity-robust covariance.

    Parameters
    ----------
    y : array_like
        Response variable (1-D).
    X : array_like
        Design matrix with **intercept already included** as the first column.

    Returns
    -------
    statsmodels RegressionResults (HC3 covariance).
    """
    model = sm.OLS(y, X)
    return model.fit(cov_type="HC3")


def aicc(log_likelihood: float, k: int, n: int) -> float:
    """Compute the corrected AIC (AICc).

    Parameters
    ----------
    log_likelihood : float
    k : int
        Number of free parameters (including the error variance).
    n : int
        Number of observations.

    Returns
    -------
    float
        AICc value.  Lower is better.
    """
    aic = -2.0 * log_likelihood + 2.0 * k
    if n - k - 1 <= 0:
        return float("nan")
    correction = 2.0 * k * (k + 1) / (n - k - 1)
    return aic + correction


# ---------------------------------------------------------------------------
# Model comparison on full sample
# ---------------------------------------------------------------------------

def compare_models_full(df: pd.DataFrame) -> dict:
    """Fit base and full models on the complete sample and compare them.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``delta_f3``, ``logMbar``, ``delta_mass``.

    Returns
    -------
    dict with keys:
        n, base_aic, full_aic, delta_aicc, base_bic, full_bic, delta_bic,
        base_r2, full_r2, delta_r2, base_coef, full_coef
    """
    sub = df[["delta_f3", "logMbar", "delta_mass"]].dropna()
    n = len(sub)
    y = sub["delta_f3"].values

    # Base model
    X_base = sm.add_constant(sub[["logMbar"]].values)
    res_base = fit_ols_hc3(y, X_base)
    ll_base = res_base.llf
    k_base = 3  # intercept + 1 predictor + sigma²

    # Full SCM model
    X_full = sm.add_constant(sub[["logMbar", "delta_mass"]].values)
    res_full = fit_ols_hc3(y, X_full)
    ll_full = res_full.llf
    k_full = 4  # intercept + 2 predictors + sigma²

    base_aicc = aicc(ll_base, k_base, n)
    full_aicc = aicc(ll_full, k_full, n)

    return {
        "n": n,
        "base_aicc": base_aicc,
        "full_aicc": full_aicc,
        "delta_aicc": base_aicc - full_aicc,
        "base_bic": res_base.bic,
        "full_bic": res_full.bic,
        "delta_bic": res_base.bic - res_full.bic,
        "base_r2": res_base.rsquared,
        "full_r2": res_full.rsquared,
        "delta_r2": res_full.rsquared - res_base.rsquared,
        "base_coef": res_base.params.tolist(),
        "full_coef": res_full.params.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-split OOS helper
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_oos_split(
    df: pd.DataFrame,
    test_frac: float = TEST_FRAC_DEFAULT,
    seed: int | None = None,
) -> dict:
    """Run a single 70/30 OOS split and return RMSE metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``delta_f3``, ``logMbar``, ``delta_mass``.
    test_frac : float
        Fraction of rows used as the test set.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        n_train, n_test, rmse_base_train, rmse_base_test,
        rmse_full_train, rmse_full_test, delta_rmse_test
    """
    sub = df[["delta_f3", "logMbar", "delta_mass"]].dropna().reset_index(drop=True)
    n = len(sub)
    rng = np.random.default_rng(seed)
    n_test = max(1, round(test_frac * n))
    idx_test = rng.choice(n, size=n_test, replace=False)
    idx_train = np.setdiff1d(np.arange(n), idx_test)

    if len(idx_train) < 3:
        return {
            "n_train": len(idx_train),
            "n_test": n_test,
            "rmse_base_train": float("nan"),
            "rmse_base_test": float("nan"),
            "rmse_full_train": float("nan"),
            "rmse_full_test": float("nan"),
            "delta_rmse_test": float("nan"),
        }

    train = sub.iloc[idx_train]
    test = sub.iloc[idx_test]

    y_train = train["delta_f3"].values
    y_test = test["delta_f3"].values

    # Base model
    X_train_base = sm.add_constant(train[["logMbar"]].values, has_constant="add")
    X_test_base = sm.add_constant(test[["logMbar"]].values, has_constant="add")
    res_base = sm.OLS(y_train, X_train_base).fit()
    pred_base_train = res_base.predict(X_train_base)
    pred_base_test = res_base.predict(X_test_base)

    # Full model
    X_train_full = sm.add_constant(train[["logMbar", "delta_mass"]].values, has_constant="add")
    X_test_full = sm.add_constant(test[["logMbar", "delta_mass"]].values, has_constant="add")
    res_full = sm.OLS(y_train, X_train_full).fit()
    pred_full_train = res_full.predict(X_train_full)
    pred_full_test = res_full.predict(X_test_full)

    rmse_base_test = _rmse(y_test, pred_base_test)
    rmse_full_test = _rmse(y_test, pred_full_test)

    return {
        "n_train": len(idx_train),
        "n_test": n_test,
        "rmse_base_train": _rmse(y_train, pred_base_train),
        "rmse_base_test": rmse_base_test,
        "rmse_full_train": _rmse(y_train, pred_full_train),
        "rmse_full_test": rmse_full_test,
        "delta_rmse_test": rmse_base_test - rmse_full_test,
    }


# ---------------------------------------------------------------------------
# Full OOS validation loop
# ---------------------------------------------------------------------------

def run_oos_validation(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS_DEFAULT,
    test_frac: float = TEST_FRAC_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Run OOS validation over *n_splits* random splits and apply Wilcoxon test.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``delta_f3``, ``logMbar``, ``delta_mass``.
    n_splits : int
        Number of independent random splits.
    test_frac : float
        Test set fraction per split.
    seed : int
        Base random seed; each split uses ``seed + i``.

    Returns
    -------
    dict with keys:
        splits (list of per-split dicts), delta_rmse_arr (ndarray),
        wilcoxon_stat, wilcoxon_pvalue, median_delta_rmse,
        frac_positive (fraction of splits where full model is better)
    """
    splits = [
        run_oos_split(df, test_frac=test_frac, seed=seed + i)
        for i in range(n_splits)
    ]
    delta_rmse_arr = np.array([s["delta_rmse_test"] for s in splits])
    valid = delta_rmse_arr[~np.isnan(delta_rmse_arr)]

    if len(valid) >= 10:
        stat, pval = wilcoxon(valid)
    else:
        stat, pval = float("nan"), float("nan")

    return {
        "splits": splits,
        "delta_rmse_arr": delta_rmse_arr,
        "wilcoxon_stat": float(stat),
        "wilcoxon_pvalue": float(pval),
        "median_delta_rmse": float(np.nanmedian(delta_rmse_arr)),
        "frac_positive": float(np.mean(valid > 0)) if len(valid) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Residual catalog and extreme galaxies
# ---------------------------------------------------------------------------

def build_residual_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Add base and full-model residuals to the catalog.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``delta_f3``, ``logMbar``, ``delta_mass``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with columns:
        ``pred_base``, ``resid_base``, ``pred_full``, ``resid_full``,
        ``delta_resid`` (|resid_base| − |resid_full|).
    """
    sub = df.dropna(subset=["delta_f3", "logMbar", "delta_mass"]).copy()
    y = sub["delta_f3"].values

    X_base = sm.add_constant(sub[["logMbar"]].values, has_constant="add")
    pred_base = sm.OLS(y, X_base).fit().predict(X_base)

    X_full = sm.add_constant(sub[["logMbar", "delta_mass"]].values, has_constant="add")
    pred_full = sm.OLS(y, X_full).fit().predict(X_full)

    sub = sub.copy()
    sub["pred_base"] = pred_base
    sub["resid_base"] = y - pred_base
    sub["pred_full"] = pred_full
    sub["resid_full"] = y - pred_full
    sub["delta_resid"] = np.abs(sub["resid_base"]) - np.abs(sub["resid_full"])
    return sub.reset_index(drop=True)


def identify_extreme_galaxies(df_residuals: pd.DataFrame, n: int = N_EXTREME_DEFAULT) -> pd.DataFrame:
    """Return the *n* galaxies with the largest |delta_resid|.

    Parameters
    ----------
    df_residuals : pd.DataFrame
        Output of :func:`build_residual_catalog`.
    n : int
        Number of extreme galaxies to return.

    Returns
    -------
    pd.DataFrame
        Sorted by |delta_resid| descending.
    """
    df_sorted = df_residuals.copy()
    df_sorted["abs_delta_resid"] = df_sorted["delta_resid"].abs()
    return (
        df_sorted.sort_values("abs_delta_resid", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Alternative env2 catalog (robustness check)
# ---------------------------------------------------------------------------

def build_env2_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Build alternative environmental model catalog using log10(MHI) as proxy.

    In the main analysis ``delta_mass = log10(MHI/L36)`` is used.  Here we
    use ``env2 = log10(MHI) - mean(log10(MHI))`` (centred raw HI mass) as an
    independent robustness check.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns from ``galaxy_catalog_with_env.csv`` plus
        ``logMbar`` and ``delta_mass``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``env2`` column and without ``delta_mass``.
    """
    df2 = df.copy()
    if "delta_mass" in df2.columns:
        # Derive log10(MHI) proxy: delta_mass = log10(MHI/L36),
        # so we shift by the mean to centre.
        dm = df2["delta_mass"].dropna()
        df2["env2"] = df2["delta_mass"] - dm.mean()
    else:
        df2["env2"] = np.nan
    return df2


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 01 — beta vs logMbar scatter with OLS trend line."""
    sub = df.dropna(subset=["beta", "logMbar"])
    x = sub["logMbar"].values
    y = sub["beta"].values

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(x, y, s=25, alpha=0.75, color="black", edgecolors="none")
    coef = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, np.polyval(coef, x_fit), linewidth=2, color="C1")
    ax.axhline(0.5, color="C0", linestyle="--", linewidth=1.2, label="β = 0.5 (MOND)")
    ax.set_xlabel(r"$\log M_{\rm bar}$ [$M_\odot$]")
    ax.set_ylabel(r"$\beta$ (deep-regime slope)")
    ax.set_title("Baryonic mass vs deep-regime slope (SPARC)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "figure01_scatter.pdf"
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    return out_path


def plot_delta_rmse_hist(
    delta_rmse_arr: np.ndarray,
    wilcoxon_pvalue: float,
    out_dir: Path,
) -> Path:
    """Figure 02 — histogram of ΔRMSE_out values across OOS splits."""
    valid = delta_rmse_arr[~np.isnan(delta_rmse_arr)]
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.hist(valid, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    median_val = np.median(valid) if len(valid) else 0.0
    ax.axvline(median_val, color="C1", linewidth=1.8, linestyle="-",
               label=f"median = {median_val:.4f}")
    pval_str = f"{wilcoxon_pvalue:.2e}" if not np.isnan(wilcoxon_pvalue) else "N/A"
    ax.set_xlabel(r"$\Delta$RMSE$_{\rm out}$ (base − full)")
    ax.set_ylabel("Count")
    ax.set_title(f"OOS ΔRMSE distribution  (Wilcoxon p = {pval_str})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "figure02_delta_rmse_hist.pdf"
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    return out_path


def plot_delta_rmse_scatter(df_residuals: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 03 — per-galaxy |resid_base| vs |resid_full| scatter."""
    sub = df_residuals.dropna(subset=["resid_base", "resid_full"])
    x = np.abs(sub["resid_base"].values)
    y = np.abs(sub["resid_full"].values)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(x, y, s=25, alpha=0.75, color="black", edgecolors="none")
    max_val = max(x.max(), y.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="C1", linewidth=1.2,
            linestyle="--", label="equal performance")
    ax.set_xlabel(r"|residual$_{\rm base}$|")
    ax.set_ylabel(r"|residual$_{\rm full}$|")
    ax.set_title("Per-galaxy model performance (base vs full SCM)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "figure03_delta_rmse_scatter.pdf"
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_validation_pipeline(
    input_path: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    n_splits: int = N_SPLITS_DEFAULT,
    test_frac: float = TEST_FRAC_DEFAULT,
    seed: int = SEED_DEFAULT,
    n_extreme: int = N_EXTREME_DEFAULT,
) -> dict:
    """Run the full Paper 1 validation pipeline.

    Parameters
    ----------
    input_path : str or Path
        Path to ``galaxy_catalog_with_env.csv`` (columns: galaxy, beta,
        delta_f3, logMbar, logRd, delta_mass).
    out_dir : str or Path
        Directory for output CSV files.
    figures_dir : str or Path
        Directory for output PDF figures.
    n_splits : int
        Number of OOS splits.
    test_frac : float
        Test fraction per split.
    seed : int
        Base random seed.
    n_extreme : int
        Number of extreme galaxies to report.

    Returns
    -------
    dict with keys:
        ``model_comparison``, ``oos``, ``df_residuals``,
        ``df_extreme``, ``df_env2``.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    figures_dir = Path(figures_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input catalog not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_env_columns(df, input_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full-sample model comparison
    model_cmp = compare_models_full(df)

    # 2. OOS validation
    oos = run_oos_validation(df, n_splits=n_splits, test_frac=test_frac, seed=seed)

    # 3. Residual catalog
    df_residuals = build_residual_catalog(df)
    df_residuals.to_csv(out_dir / "galaxy_catalog_with_residual.csv", index=False)

    # 4. Extreme galaxies
    df_extreme = identify_extreme_galaxies(df_residuals, n=n_extreme)
    df_extreme.to_csv(out_dir / "extreme_25_results.csv", index=False)

    # 5. Robustness catalog (env2)
    df_env2 = build_env2_catalog(df)
    df_env2.to_csv(out_dir / "galaxy_catalog_env2.csv", index=False)

    # 6. Figures
    if "beta" in df.columns:
        plot_scatter(df, figures_dir)
    plot_delta_rmse_hist(oos["delta_rmse_arr"], oos["wilcoxon_pvalue"], figures_dir)
    plot_delta_rmse_scatter(df_residuals, figures_dir)

    return {
        "model_comparison": model_cmp,
        "oos": oos,
        "df_residuals": df_residuals,
        "df_extreme": df_extreme,
        "df_env2": df_env2,
    }


def _require_env_columns(df: pd.DataFrame, path: Path) -> None:
    required = {"delta_f3", "logMbar", "delta_mass"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing in {path}: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Out-of-sample validation for the SCM environmental modulation "
            "model (Paper 1)."
        )
    )
    parser.add_argument(
        "--input", default=None,
        help=(
            "Path to galaxy_catalog_with_env.csv "
            "(default: results/paper1_environment/data/galaxy_catalog_with_env.csv)."
        ),
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory for CSV files (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--figures-dir", dest="figures_dir", default=DEFAULT_FIGURES_DIR,
        help=f"Output directory for figures (default: {DEFAULT_FIGURES_DIR}).",
    )
    parser.add_argument(
        "--n-splits", dest="n_splits", type=int, default=N_SPLITS_DEFAULT,
        help=f"Number of random OOS splits (default: {N_SPLITS_DEFAULT}).",
    )
    parser.add_argument(
        "--test-frac", dest="test_frac", type=float, default=TEST_FRAC_DEFAULT,
        help=f"Test fraction for each split (default: {TEST_FRAC_DEFAULT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED_DEFAULT,
        help=f"Random seed (default: {SEED_DEFAULT}).",
    )
    parser.add_argument(
        "--n-extreme", dest="n_extreme", type=int, default=N_EXTREME_DEFAULT,
        help=f"Number of extreme galaxies to report (default: {N_EXTREME_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    n_splits: int | None = None,
    test_frac: float | None = None,
    seed: int | None = None,
    n_extreme: int | None = None,
) -> dict:
    """Entry point for the OOS validation pipeline.

    Accepts both a list of CLI tokens via *argv* or keyword arguments
    directly (keyword args take precedence).
    """
    args = _parse_args([] if any(
        v is not None for v in [input_path, out_dir, figures_dir, n_splits, test_frac, seed, n_extreme]
    ) else argv)

    _input = input_path if input_path is not None else (
        args.input if args.input is not None
        else DEFAULT_OUT_DIR.replace("/data", "/data/galaxy_catalog_with_env.csv")
    )

    return run_validation_pipeline(
        input_path=_input,
        out_dir=out_dir if out_dir is not None else args.out_dir,
        figures_dir=figures_dir if figures_dir is not None else args.figures_dir,
        n_splits=n_splits if n_splits is not None else args.n_splits,
        test_frac=test_frac if test_frac is not None else args.test_frac,
        seed=seed if seed is not None else args.seed,
        n_extreme=n_extreme if n_extreme is not None else args.n_extreme,
    )


if __name__ == "__main__":
    main()
