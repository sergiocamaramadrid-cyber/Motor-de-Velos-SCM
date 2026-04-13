"""
scripts/run_scm.py — Motor de Velos SCM — Main analysis pipeline.

Detects and validates the negative correlation between environmental density
(``env_proxy``) and the external rotation-curve slope (``F3_SCM``) in
high-mass galaxies (logM ≥ 10.5).

Analysis steps
--------------
1. Load galaxy catalog.
2. Split into high-mass (logM ≥ 10.5) and low-mass sub-samples.
3. Ordinary Least Squares regression: F3_SCM ~ env_proxy.
4. Bootstrap (stratified, n=5 000) confidence intervals.
5. Permutation test (n=10 000) for the null hypothesis rho=0.
6. Ridge-regression stability scan (alpha grid).
7. Control regressions: add logMbar and F_gas as covariates.
8. Spearman correlation.
9. Write machine-readable JSON, human-readable TXT, mass-scan CSV.
10. Generate scatter figure (PNG + PDF).

Usage
-----
    python scripts/run_scm.py
    python scripts/run_scm.py --catalog data/galaxy_catalog_with_env.csv
    python scripts/run_scm.py --out results/scm --seed 42

Output files (in ``--out`` directory, default ``results/scm``):
    scm_summary.json          machine-readable results dict
    scm_summary.txt           human-readable report
    mass_scan.csv             mass-threshold sensitivity scan
    env_slope_scatter.png     scatter figure
    env_slope_scatter.pdf     scatter figure (publication quality)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASS_THRESHOLD_DEFAULT = 10.5
N_BOOT_DEFAULT = 5_000
N_PERM_DEFAULT = 10_000
SEED_DEFAULT = 42
RIDGE_ALPHAS_DEFAULT = (0.01, 0.1, 1.0, 10.0, 100.0)
CATALOG_DEFAULT = "data/galaxy_catalog_with_env.csv"
OUT_DEFAULT = "results/scm"

# Required catalog columns
_REQUIRED_COLS = {"logM", "logMbar", "F3_SCM", "env_proxy"}
_OPTIONAL_COL_GAS = "F_gas"

_SEP = "=" * 68

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load the galaxy catalog from *path*.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file with at least the columns
        ``logM``, ``logMbar``, ``F3_SCM``, ``env_proxy``.

    Returns
    -------
    pd.DataFrame
        Catalog with complete rows only (rows with any NaN are dropped).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )
    df = df.dropna(subset=list(_REQUIRED_COLS)).reset_index(drop=True)
    return df


def split_by_mass(df: pd.DataFrame, threshold: float = MASS_THRESHOLD_DEFAULT
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into high-mass and low-mass sub-samples.

    Parameters
    ----------
    df : pd.DataFrame
        Full galaxy catalog (must contain column ``logM``).
    threshold : float
        log-mass boundary.  Galaxies with ``logM >= threshold`` go into the
        high-mass sample.

    Returns
    -------
    (high_df, low_df) : tuple of DataFrames
    """
    high = df[df["logM"] >= threshold].copy().reset_index(drop=True)
    low = df[df["logM"] < threshold].copy().reset_index(drop=True)
    return high, low


# ---------------------------------------------------------------------------
# Statistical estimators
# ---------------------------------------------------------------------------


def compute_ols(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
) -> dict:
    """Ordinary Least Squares regression y ~ x.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
        Column names.

    Returns
    -------
    dict with keys ``coeff``, ``intercept``, ``se``, ``t_stat``, ``p_value``,
    ``r_squared``, ``n``.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    n = len(x)
    slope, intercept, r_value, p_value, stderr = _scipy_stats.linregress(x, y)
    return {
        "coeff": float(slope),
        "intercept": float(intercept),
        "se": float(stderr),
        "t_stat": float(slope / stderr) if stderr > 0 else float("nan"),
        "p_value": float(p_value),
        "r_squared": float(r_value ** 2),
        "n": int(n),
    }


def compute_spearman(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
) -> dict:
    """Spearman rank-correlation between *x_col* and *y_col*.

    Returns
    -------
    dict with keys ``rho`` and ``p_value``.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    rho, p = _scipy_stats.spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p)}


def bootstrap_ols(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Bootstrap 95 % confidence interval for the OLS slope.

    Resamples rows with replacement *n_boot* times and computes the OLS slope
    for each resample.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``coeff_mean``, ``coeff_std``, ``ci_low``, ``ci_high``,
    ``n_boot``, ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    slopes: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = df.iloc[idx]
        slope, *_ = _scipy_stats.linregress(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
        )
        slopes.append(float(slope))

    arr = np.array(slopes)
    return {
        "coeff_mean": float(arr.mean()),
        "coeff_std": float(arr.std()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_boot": n_boot,
        "seed": seed,
    }


def permutation_test(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
    n_perm: int = N_PERM_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Permutation test for H0: slope(y ~ x) = 0.

    Randomly shuffles *x_col* *n_perm* times and counts how often the
    absolute OLS slope exceeds the observed value.

    Returns
    -------
    dict with keys ``observed_coeff``, ``p_value``, ``n_perm``, ``seed``.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    obs_slope, *_ = _scipy_stats.linregress(x, y)
    abs_obs = abs(float(obs_slope))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        x_perm = rng.permutation(x)
        s, *_ = _scipy_stats.linregress(x_perm, y)
        if abs(s) >= abs_obs:
            count += 1

    p_value = (count + 1) / (n_perm + 1)
    return {
        "observed_coeff": float(obs_slope),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "seed": seed,
    }


def ridge_regression(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
    alphas: tuple[float, ...] = RIDGE_ALPHAS_DEFAULT,
) -> dict:
    """Ridge regression stability scan across *alphas*.

    Uses the closed-form Ridge solution::

        beta_ridge = (X'X + alpha*I)^{-1} X'y

    where X is mean-centred before computing the inverse.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
    alphas : sequence of float
        Regularisation strengths to scan.

    Returns
    -------
    dict with key ``scan`` (list of dicts with ``alpha``, ``coeff``,
    ``intercept``) and ``coeff_at_alpha_1`` (coefficient at alpha=1.0).
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean

    scan = []
    coeff_alpha1 = float("nan")
    for alpha in alphas:
        denom = float(np.dot(xc, xc)) + alpha
        coeff = float(np.dot(xc, yc)) / denom
        intercept = y_mean - coeff * x_mean
        entry = {"alpha": float(alpha), "coeff": coeff, "intercept": intercept}
        scan.append(entry)
        if abs(alpha - 1.0) < 1e-9:
            coeff_alpha1 = coeff

    return {"scan": scan, "coeff_at_alpha_1": coeff_alpha1}


def control_regression(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
    control_col: str = "logMbar",
) -> dict:
    """OLS with an additional control variable.

    Fits y ~ x + control via matrix OLS.

    Returns
    -------
    dict with keys ``coeff_env``, ``coeff_control``, ``intercept``,
    ``p_env``, ``p_control``, ``r_squared``, ``n``, ``control_col``.
    """
    sub = df[[x_col, control_col, y_col]].dropna()
    n = len(sub)
    if n < 3:
        nan = float("nan")
        return {
            "coeff_env": nan, "coeff_control": nan, "intercept": nan,
            "p_env": nan, "p_control": nan, "r_squared": nan,
            "n": n, "control_col": control_col,
        }

    x = sub[x_col].to_numpy(dtype=float)
    ctrl = sub[control_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)

    X = np.column_stack([np.ones(n), x, ctrl])
    # OLS via normal equations
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        nan = float("nan")
        return {
            "coeff_env": nan, "coeff_control": nan, "intercept": nan,
            "p_env": nan, "p_control": nan, "r_squared": nan,
            "n": n, "control_col": control_col,
        }

    intercept, coeff_env, coeff_ctrl = float(beta[0]), float(beta[1]), float(beta[2])
    y_hat = X @ beta
    residuals = y - y_hat
    sse = float(np.dot(residuals, residuals))
    sst = float(np.var(y, ddof=0) * n)
    r_squared = 1.0 - sse / sst if sst > 0 else float("nan")

    df_model = n - 3  # 3 parameters
    s2 = sse / df_model if df_model > 0 else float("nan")
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se_env = float(np.sqrt(cov[1, 1]))
        se_ctrl = float(np.sqrt(cov[2, 2]))
    except np.linalg.LinAlgError:
        se_env = se_ctrl = float("nan")

    t_env = coeff_env / se_env if se_env > 0 else float("nan")
    t_ctrl = coeff_ctrl / se_ctrl if se_ctrl > 0 else float("nan")

    if df_model > 0 and not (
        np.isnan(t_env) or np.isnan(t_ctrl)
    ):
        p_env = float(2 * _scipy_stats.t.sf(abs(t_env), df=df_model))
        p_ctrl = float(2 * _scipy_stats.t.sf(abs(t_ctrl), df=df_model))
    else:
        p_env = p_ctrl = float("nan")

    return {
        "coeff_env": coeff_env,
        "coeff_control": coeff_ctrl,
        "intercept": intercept,
        "p_env": p_env,
        "p_control": p_ctrl,
        "r_squared": r_squared,
        "n": int(n),
        "control_col": control_col,
    }


def mass_threshold_scan(
    df: pd.DataFrame,
    x_col: str = "env_proxy",
    y_col: str = "F3_SCM",
    mass_col: str = "logM",
    scan_min: float = 9.5,
    scan_max: float = 11.0,
    scan_step: float = 0.25,
    min_n: int = 10,
) -> pd.DataFrame:
    """Sensitivity of the env–slope signal to the mass threshold.

    For each threshold in the scan grid, computes OLS slope and p-value for
    the high-mass (≥ threshold) sub-sample.

    Returns
    -------
    pd.DataFrame with columns ``threshold``, ``n``, ``coeff``, ``p_value``.
    """
    thresholds = np.arange(scan_min, scan_max + scan_step / 2, scan_step)
    rows: list[dict] = []
    for thr in thresholds:
        sub = df[df[mass_col] >= thr]
        if len(sub) < min_n:
            rows.append({
                "threshold": round(float(thr), 4),
                "n": int(len(sub)),
                "coeff": float("nan"),
                "p_value": float("nan"),
            })
            continue
        ols = compute_ols(sub, x_col=x_col, y_col=y_col)
        rows.append({
            "threshold": round(float(thr), 4),
            "n": ols["n"],
            "coeff": ols["coeff"],
            "p_value": ols["p_value"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def generate_figure(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    coeff: float,
    intercept: float,
    ci_low: float,
    ci_high: float,
    spearman_rho: float,
    p_value: float,
    out_path: Path,
) -> None:
    """Scatter plot of y vs x with OLS fit and 95 % bootstrap CI band.

    Saves PNG and PDF.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = coeff * x_fit + intercept
    y_ci_low = ci_low * x_fit + intercept
    y_ci_high = ci_high * x_fit + intercept

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(x, y, s=28, alpha=0.72, color="steelblue",
               edgecolors="none", zorder=3)
    ax.plot(x_fit, y_fit, color="C1", linewidth=2.0, zorder=4,
            label=f"OLS  β = {coeff:.3f}")
    ax.fill_between(x_fit, y_ci_low, y_ci_high, color="C1",
                    alpha=0.15, zorder=2, label="95 % boot CI")

    sign = "−" if p_value < 0.001 else f"{p_value:.3f}"
    ax.text(
        0.97, 0.97,
        f"ρ = {spearman_rho:.3f}  p = {p_value:.3f}  n = {len(x)}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
    )
    ax.set_xlabel("Environmental density  (env_proxy)", fontsize=12)
    ax.set_ylabel(r"Outer slope  $F_3^{\mathrm{SCM}}$", fontsize=12)
    ax.set_title("SCM — env_proxy vs F3_SCM  (logM ≥ 10.5)", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)
    fig.tight_layout()

    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(results: dict) -> str:
    """Format *results* dict as a human-readable text report."""
    ols = results["ols"]
    boot = results["bootstrap"]
    perm = results["permutation"]
    sp = results["spearman"]
    ctrl_mass = results["control_mass"]
    ctrl_gas = results.get("control_gas", {})
    n_hi = results["n_high_mass"]
    n_lo = results["n_low_mass"]
    threshold = results["mass_threshold"]

    lines = [
        _SEP,
        "  Motor de Velos SCM — Resumen de resultados",
        _SEP,
        f"  Catálogo          : {results['catalog_path']}",
        f"  N total           : {n_hi + n_lo}",
        f"  Umbral de masa    : logM ≥ {threshold}",
        f"  N alta masa       : {n_hi}",
        f"  N baja masa       : {n_lo}",
        "",
        "  ── OLS simple (F3_SCM ~ env_proxy, alta masa) ──────────────────",
        f"  β_env             : {ols['coeff']:+.4f}",
        f"  Error estándar    : {ols['se']:.4f}",
        f"  t                 : {ols['t_stat']:+.4f}",
        f"  p-valor           : {ols['p_value']:.4f}",
        f"  R²                : {ols['r_squared']:.4f}",
        "",
        "  ── Bootstrap (IC 95 %) ─────────────────────────────────────────",
        f"  β_env (media boot): {boot['coeff_mean']:+.4f}",
        f"  IC95 %            : [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]",
        f"  Remuestras        : {boot['n_boot']}",
        "",
        "  ── Test de permutación ─────────────────────────────────────────",
        f"  β_obs             : {perm['observed_coeff']:+.4f}",
        f"  p permutación     : {perm['p_value']:.4f}",
        f"  N permutaciones   : {perm['n_perm']}",
        "",
        "  ── Correlación de Spearman ─────────────────────────────────────",
        f"  ρ                 : {sp['rho']:+.4f}",
        f"  p                 : {sp['p_value']:.4f}",
        "",
        "  ── Control por masa bariónica (logMbar) ─────────────────────────",
        f"  β_env             : {ctrl_mass['coeff_env']:+.4f}  (p={ctrl_mass['p_env']:.4f})",
        f"  β_logMbar         : {ctrl_mass['coeff_control']:+.4f}  (p={ctrl_mass['p_control']:.4f})",
        f"  R²                : {ctrl_mass['r_squared']:.4f}",
    ]
    if ctrl_gas:
        lines += [
            "",
            "  ── Control por contenido de gas (F_gas) ─────────────────────────",
            f"  β_env             : {ctrl_gas['coeff_env']:+.4f}  (p={ctrl_gas['p_env']:.4f})",
            f"  β_F_gas           : {ctrl_gas['coeff_control']:+.4f}  (p={ctrl_gas['p_control']:.4f})",
            f"  R²                : {ctrl_gas['r_squared']:.4f}",
        ]
    lines += ["", _SEP]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    """Run the full Motor de Velos SCM analysis.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Uses ``sys.argv[1:]`` if *None*.

    Returns
    -------
    dict
        Full results dictionary (also written to JSON/TXT in *--out*).
    """
    parser = argparse.ArgumentParser(
        description="Motor de Velos SCM — main analysis pipeline."
    )
    parser.add_argument(
        "--catalog", default=CATALOG_DEFAULT,
        help=f"Path to galaxy catalog CSV (default: {CATALOG_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUT_DEFAULT,
        help=f"Output directory (default: {OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=MASS_THRESHOLD_DEFAULT,
        help=f"log-mass threshold for high-mass split (default: {MASS_THRESHOLD_DEFAULT}).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=N_BOOT_DEFAULT,
        help=f"Bootstrap resamples (default: {N_BOOT_DEFAULT}).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=N_PERM_DEFAULT,
        help=f"Permutation resamples (default: {N_PERM_DEFAULT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED_DEFAULT,
        help=f"Random seed (default: {SEED_DEFAULT}).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load catalog
    df = load_catalog(args.catalog)

    # 2. Split by mass
    high_df, low_df = split_by_mass(df, threshold=args.threshold)

    # 3–8. Statistical tests on high-mass sub-sample
    ols = compute_ols(high_df)
    boot = bootstrap_ols(high_df, n_boot=args.n_boot, seed=args.seed)
    perm = permutation_test(high_df, n_perm=args.n_perm, seed=args.seed)
    ridge = ridge_regression(high_df)
    sp = compute_spearman(high_df)
    ctrl_mass = control_regression(high_df, control_col="logMbar")
    ctrl_gas: dict = {}
    if _OPTIONAL_COL_GAS in high_df.columns:
        ctrl_gas = control_regression(high_df, control_col=_OPTIONAL_COL_GAS)

    # 9. Mass-threshold sensitivity scan
    scan_df = mass_threshold_scan(df)
    scan_df.to_csv(out_dir / "mass_scan.csv", index=False)

    # 10. Assemble results
    results: dict = {
        "catalog_path": str(args.catalog),
        "mass_threshold": args.threshold,
        "n_total": int(len(df)),
        "n_high_mass": int(len(high_df)),
        "n_low_mass": int(len(low_df)),
        "ols": ols,
        "bootstrap": boot,
        "permutation": perm,
        "ridge": ridge,
        "spearman": sp,
        "control_mass": ctrl_mass,
        "control_gas": ctrl_gas,
    }

    # Write machine-readable JSON
    (out_dir / "scm_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write human-readable report
    report = format_report(results)
    print(report)
    (out_dir / "scm_summary.txt").write_text(report + "\n", encoding="utf-8")

    # 11. Generate figure
    fig_path = out_dir / "env_slope_scatter.png"
    generate_figure(
        df=high_df,
        x_col="env_proxy",
        y_col="F3_SCM",
        coeff=ols["coeff"],
        intercept=ols["intercept"],
        ci_low=boot["ci_low"],
        ci_high=boot["ci_high"],
        spearman_rho=sp["rho"],
        p_value=ols["p_value"],
        out_path=fig_path,
    )

    return results


if __name__ == "__main__":
    main()
