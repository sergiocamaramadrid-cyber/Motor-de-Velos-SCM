"""
scripts/scm_oos_validation.py — Out-of-sample (OOS) 70/30 validation for the
SCM F3 model.

This script implements the referee-proof OOS generalisation test required for
the SPARC MNRAS paper (Table 4 / Results section).

Physical motivation
-------------------
The SCM deep-regime model predicts that the friction slope β ≡
d log v_obs / d log v_bar converges to 0.5 in the deep-velos limit.  Any
per-galaxy deviation Δβ = β_fitted − 0.5 (``delta_from_mond``) reflects
either genuine physical scatter or systematic environmental modulation (the
F3 term).

For the OOS test we ask: **does knowing the population-level mean β from
70 % of galaxies improve predictions for the held-out 30 %?**

  Base model   — predict β = 0.5 for every galaxy (pure MOND prior)
  SCM  model   — predict β = mean(β_train) (learned population prior)

Per-galaxy metric:
  rmse_base_i  = |β_i − 0.5|
  rmse_scm_i   = |β_i − μ_train|
  δRMSE_i      = rmse_scm_i − rmse_base_i   (< 0 ⟹ SCM wins)

Alternative modes
-----------------
Three operating modes are auto-detected from the input columns (in priority order):

1. **direct_pred** — input has per-row predictions (multiple rows per galaxy).
   The script splits galaxies 70/30, then computes per-galaxy RMSE and MAE
   from ``y_true`` vs ``pred_base`` / ``pred_scm``.

2. **direct_rmse** — input has one row per galaxy with pre-computed
   ``rmse_base`` / ``rmse_scm`` columns (km/s from rotation-curve fits).

3. **beta_proxy** — input has one row per galaxy with ``friction_slope`` (β);
   base model predicts β = 0.5 (MOND prior), SCM model predicts β = mean(β_train).

Outputs
-------
  <out>/oos_generalization_results.csv   — per-galaxy, per-seed results
  <out>/oos_summary.txt                  — paper-ready summary (Table 4 block)
  <out>/hist_delta_rmse_out.png/.pdf     — histogram of δRMSE across test galaxies
  <out>/hist_delta_logl_out.png/.pdf     — histogram of per-galaxy log-likelihood ratio

Usage
-----
::

    # direct_pred mode (row-level predictions):
    python scripts/scm_oos_validation.py \\
        --input results/predictions/sparc_predictions.csv \\
        --split 0.7 \\
        --out results/scm_oos

    # Multiple seeds for robustness:
    python scripts/scm_oos_validation.py \\
        --input results/delta_f3/sparc_delta_f3_catalog.csv \\
        --split 0.7 \\
        --seeds 42 43 44 45 46 \\
        --out results/scm_oos

Input column contract
---------------------
**direct_pred mode** (any row count per galaxy):
  galaxy identifier  — galaxy | name | galname
  y_true             — y_true | target | y | observed | value_true
  pred_base          — pred_base | yhat_base | base_pred | pred_baseline
  pred_scm           — pred_scm | yhat_scm | scm_pred | pred_model

**direct_rmse mode** (one row per galaxy):
  galaxy             — (same aliases as above)
  rmse_base          — per-galaxy RMSE of baseline rotation-curve fit
  rmse_scm           — per-galaxy RMSE of SCM rotation-curve fit

**beta_proxy mode** (one row per galaxy):
  galaxy             — (same aliases as above)
  friction_slope     — per-galaxy fitted β (alias: beta)
  friction_slope_err — β fitting uncertainty (alias: beta_err, optional)

Optional:
  delta_from_mond / delta_f3  — β − 0.5 (recognised in beta_proxy mode)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_BETA = 0.5          # MOND / deep-velos prediction
DEFAULT_SPLIT = 0.70         # training fraction
DEFAULT_SEEDS = [42]         # default random seeds
MIN_TEST_GALAXIES = 5        # minimum viable test-set size for Wilcoxon

# Column name aliases — checked in priority order (first match in the DataFrame wins)
_COL_GALAXY_ALIASES = ("galaxy", "name", "galname")
_COL_BETA = ("friction_slope", "beta")
_COL_BETA_ERR = ("friction_slope_err", "beta_err")
_COL_DELTA = ("delta_from_mond", "delta_f3")
_COL_RMSE_BASE = ("rmse_base",)
_COL_RMSE_SCM = ("rmse_scm",)
# direct_pred mode aliases — checked in priority order (first match wins)
_COL_Y_TRUE = ("y_true", "target", "y", "observed", "value_true")
_COL_PRED_BASE = ("pred_base", "yhat_base", "base_pred", "pred_baseline")
_COL_PRED_SCM = ("pred_scm", "yhat_scm", "scm_pred", "pred_model")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_catalog(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet catalog into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    """Return the first column from *names* that is present in *df*."""
    for name in names:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def validate_input(df: pd.DataFrame) -> dict[str, str]:
    """Validate the input catalog and return a mapping of role → actual column name.

    Operating modes are detected in priority order:

    1. **direct_pred** — ``y_true`` (or alias) + ``pred_base`` (or alias) +
       ``pred_scm`` (or alias) are present.  Input may have multiple rows per
       galaxy; per-galaxy RMSE/MAE are computed inside :func:`_aggregate_direct_pred`.
    2. **direct_rmse** — ``rmse_base`` + ``rmse_scm`` are present (one row per
       galaxy with pre-computed metrics).
    3. **beta_proxy** — ``friction_slope`` / ``beta`` is present (one row per
       galaxy; base model = β = 0.5, SCM model = training-set mean β).

    Raises
    ------
    ValueError
        If required columns are missing or the mode cannot be determined.
    """
    # Galaxy column — check aliases in priority order
    galaxy_col = _first_col(df, _COL_GALAXY_ALIASES)
    if galaxy_col is None:
        raise ValueError(
            f"Input catalog missing galaxy identifier column. "
            f"Expected one of: {list(_COL_GALAXY_ALIASES)}. "
            f"Available columns: {list(df.columns)}"
        )
    col_map: dict[str, str] = {"galaxy": galaxy_col}

    # Priority 1: direct_pred mode (row-level predictions)
    y_true_col = _first_col(df, _COL_Y_TRUE)
    pred_base_col = _first_col(df, _COL_PRED_BASE)
    pred_scm_col = _first_col(df, _COL_PRED_SCM)
    if y_true_col is not None and pred_base_col is not None and pred_scm_col is not None:
        col_map["mode"] = "direct_pred"
        col_map["y_true"] = y_true_col
        col_map["pred_base"] = pred_base_col
        col_map["pred_scm"] = pred_scm_col
        return col_map

    # Priority 2: direct_rmse mode (pre-computed per-galaxy RMSE)
    rmse_base_col = _first_col(df, _COL_RMSE_BASE)
    rmse_scm_col = _first_col(df, _COL_RMSE_SCM)
    if rmse_base_col is not None and rmse_scm_col is not None:
        col_map["mode"] = "direct_rmse"
        col_map["rmse_base"] = rmse_base_col
        col_map["rmse_scm"] = rmse_scm_col
        return col_map

    # Priority 3: beta_proxy mode
    beta_col = _first_col(df, _COL_BETA)
    beta_err_col = _first_col(df, _COL_BETA_ERR)
    if beta_col is None:
        raise ValueError(
            "Cannot determine operating mode.  Expected one of:\n"
            "  (1) direct_pred  — columns y_true (or alias) + pred_base (or alias)"
            " + pred_scm (or alias)\n"
            "  (2) direct_rmse  — columns rmse_base + rmse_scm\n"
            "  (3) beta_proxy   — column friction_slope (or beta)\n"
            f"Available columns: {list(df.columns)}"
        )
    col_map["mode"] = "beta_proxy"
    col_map["beta"] = beta_col
    if beta_err_col is not None:
        col_map["beta_err"] = beta_err_col

    # Derive or locate delta column
    delta_col = _first_col(df, _COL_DELTA)
    if delta_col is not None:
        col_map["delta"] = delta_col

    return col_map


# ---------------------------------------------------------------------------
# Per-galaxy metric computation
# ---------------------------------------------------------------------------


def _compute_per_galaxy_metrics(
    df: pd.DataFrame,
    col_map: dict[str, str],
    mu_train: float,
) -> pd.DataFrame:
    """Compute per-galaxy OOS metrics for the test split.

    Parameters
    ----------
    df : pd.DataFrame
        Test-split galaxies.
    col_map : dict
        Column role → actual column name (from ``validate_input``).
    mu_train : float
        Mean β (or RMSE-equivalent) from the training set (SCM prediction).

    Returns
    -------
    pd.DataFrame with columns:
        galaxy, rmse_base, rmse_scm, delta_rmse, delta_logL
    """
    rows = []
    for _, row in df.iterrows():
        galaxy = row[col_map["galaxy"]]

        if col_map["mode"] == "direct_rmse":
            rmse_base = float(row[col_map["rmse_base"]])
            rmse_scm = float(row[col_map["rmse_scm"]])
        else:
            beta_obs = float(row[col_map["beta"]])
            rmse_base = abs(beta_obs - EXPECTED_BETA)
            rmse_scm = abs(beta_obs - mu_train)

        delta_rmse = rmse_scm - rmse_base

        # Approximate log-likelihood ratio under Gaussian errors with variance
        # σ² ∝ RMSE².  Under this assumption:
        #   log L(model) ∝ −0.5 × (residual/σ)²  →  δ log L ≈ 0.5 × (ε_base² − ε_scm²) / σ²
        # where σ is approximated by rmse_scm (the narrower of the two).
        # δ log L > 0 ⟹ SCM model is more likely; δ log L < 0 ⟹ base is more likely.
        eps = 1e-12
        delta_logL = 0.5 * (rmse_base**2 - rmse_scm**2) / (max(rmse_scm, eps) ** 2)

        rows.append({
            "galaxy": galaxy,
            "rmse_base": rmse_base,
            "rmse_scm": rmse_scm,
            "delta_rmse": delta_rmse,
            "delta_logL": delta_logL,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# direct_pred aggregation helper
# ---------------------------------------------------------------------------


def _aggregate_direct_pred(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Aggregate row-level predictions into per-galaxy RMSE and MAE.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the full catalog for a given split (train or test), with
        multiple rows per galaxy containing columns y_true, pred_base, pred_scm.
    col_map : dict
        Column role → actual column name (from ``validate_input``).

    Returns
    -------
    pd.DataFrame with one row per galaxy and columns:
        galaxy, rmse_base, rmse_scm, mae_base, mae_scm, n_rows
    """
    galaxy_col = col_map["galaxy"]
    y_true_col = col_map["y_true"]
    pred_base_col = col_map["pred_base"]
    pred_scm_col = col_map["pred_scm"]

    rows = []
    for galaxy, group in df.groupby(galaxy_col, sort=False):
        y = group[y_true_col].to_numpy(dtype=float)
        yhat_base = group[pred_base_col].to_numpy(dtype=float)
        yhat_scm = group[pred_scm_col].to_numpy(dtype=float)

        # Keep only rows where all three values are finite
        mask = np.isfinite(y) & np.isfinite(yhat_base) & np.isfinite(yhat_scm)
        y = y[mask]
        yhat_base = yhat_base[mask]
        yhat_scm = yhat_scm[mask]

        if len(y) == 0:
            continue

        rmse_base = float(np.sqrt(np.mean((y - yhat_base) ** 2)))
        rmse_scm = float(np.sqrt(np.mean((y - yhat_scm) ** 2)))
        mae_base = float(np.mean(np.abs(y - yhat_base)))
        mae_scm = float(np.mean(np.abs(y - yhat_scm)))

        rows.append({
            "galaxy": galaxy,
            "rmse_base": rmse_base,
            "rmse_scm": rmse_scm,
            "mae_base": mae_base,
            "mae_scm": mae_scm,
            "n_rows": len(y),
        })

    return pd.DataFrame(rows)


def _run_single_seed(
    df: pd.DataFrame,
    col_map: dict[str, str],
    train_frac: float,
    seed: int,
) -> dict:
    """Run OOS validation for a single random seed.

    Returns
    -------
    dict with per-seed summary and per-galaxy results.
    """
    galaxies = df[col_map["galaxy"]].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(galaxies)
    n_train = max(1, int(len(shuffled) * train_frac))
    train_gals = set(shuffled[:n_train])
    test_gals = set(shuffled[n_train:])

    df_train = df[df[col_map["galaxy"]].isin(train_gals)].copy()
    df_test = df[df[col_map["galaxy"]].isin(test_gals)].copy()

    n_train_gals = len(df_train[col_map["galaxy"]].unique())
    n_test_gals = len(df_test[col_map["galaxy"]].unique())

    # Training-set mean (SCM population prior) and per-galaxy test metrics
    if col_map["mode"] == "direct_pred":
        # Aggregate row-level predictions into per-galaxy RMSE/MAE
        agg_test = _aggregate_direct_pred(df_test, col_map)
        # _aggregate_direct_pred always outputs a normalised "galaxy" column
        # regardless of the original alias (e.g. "galname").  We therefore
        # override col_map["galaxy"] in the temporary col_map so that
        # _compute_per_galaxy_metrics reads the correct column from the
        # aggregated DataFrame.
        _rmse_col_map = dict(col_map)
        _rmse_col_map["mode"] = "direct_rmse"
        _rmse_col_map["galaxy"] = "galaxy"   # normalised by _aggregate_direct_pred
        _rmse_col_map["rmse_base"] = "rmse_base"
        _rmse_col_map["rmse_scm"] = "rmse_scm"
        per_gal = _compute_per_galaxy_metrics(agg_test, _rmse_col_map, mu_train=0.0)
        # Attach MAE columns
        per_gal = per_gal.merge(
            agg_test[["galaxy", "mae_base", "mae_scm"]], on="galaxy", how="left"
        )
        per_gal["delta_mae"] = per_gal["mae_scm"] - per_gal["mae_base"]
        # mu_train not used in direct modes but stored for info
        mu_train = 0.0
    elif col_map["mode"] == "direct_rmse":
        # Use mean rmse_scm from training as SCM model prediction for test
        mu_train = float(df_train[col_map["rmse_scm"]].mean())
        per_gal = _compute_per_galaxy_metrics(df_test, col_map, mu_train)
    else:
        beta_col = col_map["beta"]
        mu_train = float(df_train[beta_col].mean())
        per_gal = _compute_per_galaxy_metrics(df_test, col_map, mu_train)
    per_gal["seed"] = seed

    n_test = len(per_gal)
    n_improved = int((per_gal["delta_rmse"] < 0).sum())

    # Aggregate RMSE over test set
    rmse_base_agg = float(np.sqrt(np.mean(per_gal["rmse_base"] ** 2)))
    rmse_scm_agg = float(np.sqrt(np.mean(per_gal["rmse_scm"] ** 2)))
    delta_rmse_median = float(per_gal["delta_rmse"].median())

    # Wilcoxon signed-rank test (one-sided, H1: median δRMSE < 0)
    wilcoxon_p = float("nan")
    if n_test >= MIN_TEST_GALAXIES:
        try:
            _, wilcoxon_p = wilcoxon(
                per_gal["delta_rmse"].to_numpy(),
                alternative="less",
                zero_method="wilcox",
            )
        except Exception:
            wilcoxon_p = float("nan")

    summary = {
        "seed": seed,
        "n_train": n_train_gals,
        "n_test": n_test_gals,
        "mu_train": mu_train,
        "rmse_base_out": rmse_base_agg,
        "rmse_scm_out": rmse_scm_agg,
        "delta_rmse_out": rmse_scm_agg - rmse_base_agg,
        "delta_rmse_median": delta_rmse_median,
        "n_improved": n_improved,
        "pct_improved": 100.0 * n_improved / n_test if n_test > 0 else float("nan"),
        "wilcoxon_p": wilcoxon_p,
    }
    return {"summary": summary, "per_galaxy": per_gal}


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------


def run_oos_validation(
    input_path: str | Path,
    out_dir: str | Path,
    train_frac: float = DEFAULT_SPLIT,
    seeds: list[int] | None = None,
    *,
    no_figures: bool = False,
) -> dict:
    """Run OOS validation and write all outputs.

    Parameters
    ----------
    input_path : str | Path
        Path to per-galaxy catalog (CSV or Parquet).
    out_dir : str | Path
        Output directory.  Created if absent.
    train_frac : float
        Fraction of galaxies used for training (default 0.70).
    seeds : list[int] | None
        Random seeds.  Defaults to ``[42]``.
    no_figures : bool
        If True, skip matplotlib figure generation (useful in headless tests).

    Returns
    -------
    dict with keys:
        ``all_summaries``  — list of per-seed summary dicts
        ``per_galaxy_df``  — combined per-galaxy DataFrame
        ``aggregate``      — aggregated statistics across seeds (paper numbers)
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_catalog(input_path)
    if df.empty:
        raise ValueError(f"Input catalog is empty: {input_path}")

    col_map = validate_input(df)

    # Filter out rows with NaN in the key prediction columns
    if col_map["mode"] == "beta_proxy":
        beta_col = col_map["beta"]
        n_before = len(df)
        df = df[df[beta_col].notna()].copy()
        n_after = len(df)
        if n_after < n_before:
            print(
                f"  [OOS] Dropped {n_before - n_after} rows with NaN {beta_col}; "
                f"{n_after} rows remain."
            )
    elif col_map["mode"] == "direct_rmse":
        for rmse_col in (col_map["rmse_base"], col_map["rmse_scm"]):
            df = df[df[rmse_col].notna()].copy()
    else:
        # direct_pred mode: drop rows where any key column is NaN
        for key_col in (col_map["y_true"], col_map["pred_base"], col_map["pred_scm"]):
            df = df[df[key_col].notna()].copy()

    n_galaxies = len(df[col_map["galaxy"]].unique())
    print(f"  [OOS] {n_galaxies} galaxies | mode={col_map['mode']} | "
          f"train={train_frac:.0%} | seeds={seeds}")

    all_summaries = []
    all_per_galaxy = []

    for seed in seeds:
        result = _run_single_seed(df, col_map, train_frac, seed)
        all_summaries.append(result["summary"])
        all_per_galaxy.append(result["per_galaxy"])

    per_galaxy_df = pd.concat(all_per_galaxy, ignore_index=True)

    # Aggregate across seeds
    summary_df = pd.DataFrame(all_summaries)

    n_valid_seeds = len(summary_df.dropna(subset=["wilcoxon_p"]))
    pct_improved_agg = float(summary_df["pct_improved"].mean())
    delta_rmse_median_agg = float(summary_df["delta_rmse_median"].mean())
    wilcoxon_p_agg = float(summary_df["wilcoxon_p"].median())
    n_test_agg = int(summary_df["n_test"].median())

    aggregate = {
        "n_valid": n_test_agg,
        "pct_improved": pct_improved_agg,
        "delta_rmse_median": delta_rmse_median_agg,
        "wilcoxon_p": wilcoxon_p_agg,
        "n_valid_seeds": n_valid_seeds,
    }

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------

    # 1. Per-galaxy CSV
    per_gal_path = out_dir / "oos_generalization_results.csv"
    per_galaxy_df.to_csv(per_gal_path, index=False)
    print(f"  [OOS] Per-galaxy results → {per_gal_path}")

    # 2. Per-seed summary CSV
    summary_path = out_dir / "oos_seed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # 3. Text summary (paper Table 4 block)
    _write_text_summary(aggregate, summary_df, out_dir / "oos_summary.txt", col_map)

    # 4. Figures
    if not no_figures:
        _write_figures(per_galaxy_df, out_dir)

    return {
        "all_summaries": all_summaries,
        "per_galaxy_df": per_galaxy_df,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def _write_text_summary(
    aggregate: dict,
    summary_df: pd.DataFrame,
    path: Path,
    col_map: dict[str, str],
) -> None:
    n_valid = aggregate["n_valid"]
    pct = aggregate["pct_improved"]
    n_improved = int(round(n_valid * pct / 100))
    median_delta = aggregate["delta_rmse_median"]
    p_val = aggregate["wilcoxon_p"]
    mode = col_map.get("mode", "unknown")
    units = " (km/s)" if mode == "direct_rmse" else " (β units, proxy)"

    n_total = summary_df["n_train"].iloc[0] + summary_df["n_test"].iloc[0]
    text = textwrap.dedent(f"""
    ============================================================
    OOS VALIDATION SUMMARY — SCM F3 model
    ============================================================

    Mode          : {mode}
    Seeds         : {list(summary_df['seed'])}
    Train fraction: {summary_df['n_train'].iloc[0]} / {n_total} galaxies

    --- Paper Table 4 numbers ---
    N_valid (test galaxies, median across seeds) : {n_valid}
    Improved cases (δRMSE < 0)                  : {n_improved}/{n_valid}  ({pct:.1f}%)
    Median δRMSE{units:<30}: {median_delta:+.4f}
    Wilcoxon p-value (one-sided, "less")        : {p_val:.4g}

    --- MNRAS Results skeleton ---
    Using the SPARC sample (N = {n_valid} galaxies after cuts),
    we find that the SCM model improves out-of-sample performance
    in {n_improved}/{n_valid} cases ({pct:.1f}%).

    The median improvement in RMSE is
    ΔRMSEout = {median_delta:+.4f}{units},
    with a one-sided Wilcoxon test yielding p = {p_val:.4g}.

    ============================================================
    """).strip()

    path.write_text(text)
    print(f"  [OOS] Summary → {path}")
    print()
    print(text)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _write_figures(per_galaxy_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [OOS] matplotlib not available — skipping figures.")
        return

    _plot_histogram(
        per_galaxy_df["delta_rmse"],
        xlabel="δRMSE (SCM − Base)",
        title="OOS δRMSE distribution",
        stem="hist_delta_rmse_out",
        out_dir=out_dir,
    )

    _plot_histogram(
        per_galaxy_df["delta_logL"],
        xlabel="δ log L (SCM − Base)",
        title="OOS δ log-likelihood distribution",
        stem="hist_delta_logl_out",
        out_dir=out_dir,
    )


def _plot_histogram(
    values: "pd.Series",
    xlabel: str,
    title: str,
    stem: str,
    out_dir: Path,
    bins: int = 25,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values.dropna(), bins=bins, color="steelblue", edgecolor="white",
            linewidth=0.5)
    ax.axvline(0, color="crimson", linewidth=1.5, linestyle="--", label="zero (no change)")
    median_val = values.median()
    ax.axvline(median_val, color="darkorange", linewidth=1.5, linestyle=":",
               label=f"median = {median_val:+.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  [OOS] Figure → {out_dir / stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Out-of-sample 70/30 validation for the SCM F3 model.  "
            "Produces the four referee-proof numbers required for MNRAS Table 4."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
            # Minimal run (single seed):
            python scripts/scm_oos_validation.py \\
                --input results/delta_f3/sparc_delta_f3_catalog.csv \\
                --split 0.7 --out results/scm_oos

            # Multiple seeds:
            python scripts/scm_oos_validation.py \\
                --input results/delta_f3/sparc_delta_f3_catalog.csv \\
                --split 0.7 --seeds 42 43 44 45 46 --out results/scm_oos
        """),
    )
    parser.add_argument(
        "--input", required=True, metavar="FILE",
        help="Per-galaxy catalog (CSV or Parquet).",
    )
    parser.add_argument(
        "--split", type=float, default=DEFAULT_SPLIT, metavar="FRAC",
        help=f"Training fraction (default: {DEFAULT_SPLIT}).  Test fraction = 1 − FRAC.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, metavar="N",
        help=f"Random seeds for galaxy splitting (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument(
        "--out", default="results/scm_oos", metavar="DIR",
        help="Output directory (default: results/scm_oos).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    return run_oos_validation(
        input_path=args.input,
        out_dir=args.out,
        train_frac=args.split,
        seeds=args.seeds,
        no_figures=args.no_figures,
    )


if __name__ == "__main__":
    main()
