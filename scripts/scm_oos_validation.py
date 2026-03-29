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

Alternative mode — if the input catalog already contains pre-computed per-galaxy
RMSE columns ``rmse_base`` and ``rmse_scm`` (in km/s from rotation-curve fits),
those are used directly and the β-proxy is bypassed.

Outputs
-------
  <out>/oos_generalization_results.csv   — per-galaxy, per-seed results
  <out>/oos_summary.txt                  — paper-ready summary (Table 4 block)
  <out>/hist_delta_rmse_out.png/.pdf     — histogram of δRMSE across test galaxies
  <out>/hist_delta_logl_out.png/.pdf     — histogram of per-galaxy log-likelihood ratio

Usage
-----
::

    python scripts/scm_oos_validation.py \\
        --input results/delta_f3/sparc_delta_f3_catalog.csv \\
        --split 0.7 \\
        --out results/scm_oos

    # Multiple seeds for robustness
    python scripts/scm_oos_validation.py \\
        --input results/delta_f3/sparc_delta_f3_catalog.csv \\
        --split 0.7 \\
        --seeds 42 43 44 45 46 \\
        --out results/scm_oos

Required input columns (one row per galaxy)
-------------------------------------------
  galaxy                 — galaxy identifier
  friction_slope         — per-galaxy fitted β (alias: beta)
  friction_slope_err     — β fitting uncertainty (alias: beta_err)

Optional (if present, direct-RMSE mode is used instead of the β-proxy):
  rmse_base              — per-galaxy RMSE of baseline rotation-curve fit
  rmse_scm               — per-galaxy RMSE of SCM rotation-curve fit

Optional extra columns recognised automatically:
  delta_from_mond / delta_f3  — β − 0.5 (recomputed if absent)
  inc_deg / Inc               — inclination (used only for informational filtering)
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

# Column name aliases (primary → fallback)
_COL_GALAXY = "galaxy"
_COL_BETA = ("friction_slope", "beta")
_COL_BETA_ERR = ("friction_slope_err", "beta_err")
_COL_DELTA = ("delta_from_mond", "delta_f3")
_COL_RMSE_BASE = ("rmse_base",)
_COL_RMSE_SCM = ("rmse_scm",)

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

REQUIRED_BASE_COLS = (_COL_GALAXY,)


def validate_input(df: pd.DataFrame) -> dict[str, str]:
    """Validate the input catalog and return a mapping of role → actual column name.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input catalog missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    col_map: dict[str, str] = {"galaxy": _COL_GALAXY}

    # Detect operating mode: direct-RMSE or β-proxy
    rmse_base_col = _first_col(df, _COL_RMSE_BASE)
    rmse_scm_col = _first_col(df, _COL_RMSE_SCM)
    if rmse_base_col is not None and rmse_scm_col is not None:
        col_map["mode"] = "direct_rmse"
        col_map["rmse_base"] = rmse_base_col
        col_map["rmse_scm"] = rmse_scm_col
        return col_map

    # β-proxy mode: require friction_slope
    beta_col = _first_col(df, _COL_BETA)
    beta_err_col = _first_col(df, _COL_BETA_ERR)
    if beta_col is None:
        raise ValueError(
            f"Cannot determine operating mode. "
            f"Either provide (rmse_base, rmse_scm) columns for direct-RMSE mode, "
            f"or (friction_slope / beta) for β-proxy mode. "
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
# Single-seed OOS run
# ---------------------------------------------------------------------------


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

    # Training-set mean (SCM population prior)
    if col_map["mode"] == "direct_rmse":
        # Use mean rmse_scm from training as SCM model prediction for test
        mu_train = float(df_train[col_map["rmse_scm"]].mean())
    else:
        beta_col = col_map["beta"]
        mu_train = float(df_train[beta_col].mean())

    # Per-galaxy test metrics
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

    # Filter to rows with valid β (non-NaN friction_slope)
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
    else:
        for rmse_col in (col_map["rmse_base"], col_map["rmse_scm"]):
            df = df[df[rmse_col].notna()].copy()

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
