"""
Run BIG-SPARC deep-regime β test from a per-point catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_1samp


REQUIRED_COLUMNS = {"galaxy", "g_obs", "g_bar"}
EXPECTED_BETA = 0.5
G0_DEFAULT = 1.2e-10
DEEP_THRESHOLD_DEFAULT = 0.3
BOOTSTRAP_ITERS_DEFAULT = 2000
BOOTSTRAP_SEED_DEFAULT = 42
CV_FOLDS_DEFAULT = 5


def compute_beta_from_curve(sub: pd.DataFrame) -> dict:
    """Compute β = dlog(g_obs)/dlog(g_bar) for one galaxy."""
    clean = sub[["g_obs", "g_bar"]].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[(clean["g_obs"] > 0) & (clean["g_bar"] > 0)]
    deep_max = DEEP_THRESHOLD_DEFAULT * G0_DEFAULT
    clean = clean[clean["g_bar"] < deep_max]
    n = len(clean)

    if n < 2:
        return {
            "n_points": int(n),
            "beta": float("nan"),
            "beta_stderr": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
        }

    x = np.log10(clean["g_bar"].to_numpy())
    y = np.log10(clean["g_obs"].to_numpy())
    if np.allclose(np.std(x), 0.0):
        return {
            "n_points": int(n),
            "beta": float("nan"),
            "beta_stderr": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
        }

    slope, _, r_value, p_value, stderr = linregress(x, y)
    return {
        "n_points": int(n),
        "beta": float(slope),
        "beta_stderr": float(stderr),
        "r_value": float(r_value),
        "p_value": float(p_value),
    }


def _fit_environmental_regression(beta_catalog: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    needed = {"beta", "logSigmaHI_out", "logMbar"}
    if not needed.issubset(beta_catalog.columns):
        return {}, np.array([]), np.array([])

    reg = beta_catalog[list(needed)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(reg) < 3:
        return {}, np.array([]), np.array([])

    y = reg["beta"].to_numpy(dtype=float)
    x1 = reg["logSigmaHI_out"].to_numpy(dtype=float)
    x2 = reg["logMbar"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(reg)), x1, x2])

    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coefs
    residuals = y - y_hat
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")

    return {
        "n_samples": int(len(reg)),
        "coef_intercept": float(coefs[0]),
        "coef_logSigmaHI_out": float(coefs[1]),
        "coef_logMbar": float(coefs[2]),
        "r2": r2,
    }, x, y


def _bootstrap_regression_coefs(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    if len(y) == 0:
        return pd.DataFrame(columns=["iter", "intercept", "b_logSigmaHI_out", "c_logMbar"])

    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    for i in range(iterations):
        idx = rng.choice(n, size=n, replace=True)
        xb = x[idx]
        yb = y[idx]
        coefs, *_ = np.linalg.lstsq(xb, yb, rcond=None)
        rows.append(
            {
                "iter": i + 1,
                "intercept": float(coefs[0]),
                "b_logSigmaHI_out": float(coefs[1]),
                "c_logMbar": float(coefs[2]),
            }
        )
    return pd.DataFrame(rows)


def _kfold_cv(x: np.ndarray, y: np.ndarray, k_folds: int, seed: int) -> pd.DataFrame:
    if len(y) < 2:
        return pd.DataFrame(columns=["fold", "n_train", "n_test", "rmse", "mae"])

    k = min(k_folds, len(y))
    if k < 2:
        return pd.DataFrame(columns=["fold", "n_train", "n_test", "rmse", "mae"])

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    rows = []
    for i, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
        if len(train_idx) < 3 or len(test_idx) == 0:
            continue
        coefs, *_ = np.linalg.lstsq(x[train_idx], y[train_idx], rcond=None)
        pred = x[test_idx] @ coefs
        err = y[test_idx] - pred
        rows.append(
            {
                "fold": i,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def _write_regression_summary(path: Path, summary: dict) -> None:
    if not summary:
        txt = "Insufficient data for environmental regression.\n"
    else:
        txt = "\n".join(
            [
                "Environmental model: beta = a + b*logSigmaHI_out + c*logMbar",
                f"n_samples={summary['n_samples']}",
                f"a_intercept={summary['coef_intercept']:.6f}",
                f"b_logSigmaHI_out={summary['coef_logSigmaHI_out']:.6f}",
                f"c_logMbar={summary['coef_logMbar']:.6f}",
                f"r2={summary['r2']:.6f}",
            ]
        ) + "\n"
    path.write_text(txt, encoding="utf-8")


def _plot_beta_vs_sigma(beta_catalog: pd.DataFrame, out_path: Path) -> None:
    if "logSigmaHI_out" not in beta_catalog.columns:
        plt.figure(figsize=(7, 4.5))
        plt.text(0.5, 0.5, "Missing logSigmaHI_out", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    sub = beta_catalog[["logSigmaHI_out", "beta"]].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(7, 4.5))
    if len(sub) >= 2:
        x = sub["logSigmaHI_out"].to_numpy(dtype=float)
        y = sub["beta"].to_numpy(dtype=float)
        slope, intercept, *_ = linregress(x, y)
        xgrid = np.linspace(np.min(x), np.max(x), 100)
        ygrid = intercept + slope * xgrid
        sigma = float(np.std(y - (intercept + slope * x), ddof=1)) if len(y) > 2 else 0.0

        plt.scatter(x, y, alpha=0.8, label="Galaxies")
        plt.plot(xgrid, ygrid, color="tab:red", label="Linear fit")
        plt.fill_between(
            xgrid,
            ygrid - 1.96 * sigma,
            ygrid + 1.96 * sigma,
            color="tab:red",
            alpha=0.15,
            label="~95% band",
        )
    elif len(sub) == 1:
        plt.scatter(sub["logSigmaHI_out"], sub["beta"], alpha=0.8, label="Galaxies")
    else:
        plt.text(0.5, 0.5, "No valid beta/sigma points", ha="center", va="center")
        plt.axis("off")

    plt.xlabel("logSigmaHI_out")
    plt.ylabel("beta")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_bootstrap_hist(draws: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    if len(draws) > 0:
        plt.hist(draws, bins=30, alpha=0.8, color="tab:blue")
    else:
        plt.text(0.5, 0.5, "No bootstrap draws", ha="center", va="center")
        plt.axis("off")
    plt.xlabel("Bootstrap mean(beta)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _bootstrap_mean_beta(beta_values: np.ndarray, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(beta_values)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(beta_values, size=n, replace=True)
        draws[i] = float(np.mean(sample))
    return {
        "iterations": int(iterations),
        "mean_of_means": float(np.mean(draws)),
        "std_of_means": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
    }


def run_test(catalog_path: Path, out_dir: Path, bootstrap_iters: int, seed: int) -> dict:
    print("Cargando catálogo...")
    df = pd.read_csv(catalog_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {sorted(missing)}")
    print(f"Galaxias en catálogo: {df['galaxy'].nunique()}")

    rows = []
    for galaxy, sub in df.groupby("galaxy", sort=True):
        stats = compute_beta_from_curve(sub)
        stats["galaxy"] = galaxy
        if "logMbar" in sub.columns:
            stats["logMbar"] = float(sub["logMbar"].iloc[0])
        if "logSigmaHI_out" in sub.columns:
            stats["logSigmaHI_out"] = float(sub["logSigmaHI_out"].iloc[0])
        rows.append(stats)

    beta_catalog = pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    beta_catalog.to_csv(out_dir / "beta_catalog.csv", index=False)

    valid = beta_catalog["beta"].dropna().to_numpy(dtype=float)
    if len(valid) == 0:
        raise ValueError("No valid β values were computed from catalog.")

    t_stat, p_value = ttest_1samp(valid, EXPECTED_BETA, nan_policy="omit")
    bootstrap = _bootstrap_mean_beta(valid, iterations=bootstrap_iters, seed=seed)
    rng = np.random.default_rng(seed)
    bootstrap_draws = np.array(
        [np.mean(rng.choice(valid, size=len(valid), replace=True)) for _ in range(bootstrap_iters)],
        dtype=float,
    )

    reg_summary, reg_x, reg_y = _fit_environmental_regression(beta_catalog)
    _write_regression_summary(out_dir / "regression_summary.txt", reg_summary)
    boot_coefs = _bootstrap_regression_coefs(reg_x, reg_y, iterations=bootstrap_iters, seed=seed)
    boot_coefs.to_csv(out_dir / "bootstrap_coefs.csv", index=False)
    cv_df = _kfold_cv(reg_x, reg_y, k_folds=CV_FOLDS_DEFAULT, seed=seed)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    _plot_beta_vs_sigma(beta_catalog, out_dir / "beta_vs_sigmaHI_with_ci.png")
    _plot_bootstrap_hist(bootstrap_draws, out_dir / "bootstrap_beta_hist.png")

    overview = {
        "catalog": str(catalog_path),
        "n_galaxies": int(len(beta_catalog)),
        "n_valid_beta": int(len(valid)),
        "expected_beta": EXPECTED_BETA,
        "mean_beta": float(np.mean(valid)),
        "median_beta": float(np.median(valid)),
        "std_beta": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        "t_stat_vs_0_5": float(t_stat) if np.isfinite(t_stat) else float("nan"),
        "p_value_vs_0_5": float(p_value) if np.isfinite(p_value) else float("nan"),
        "bootstrap": bootstrap,
        "environmental_regression": reg_summary,
        "cv_mean_rmse": float(cv_df["rmse"].mean()) if not cv_df.empty else float("nan"),
        "cv_mean_mae": float(cv_df["mae"].mean()) if not cv_df.empty else float("nan"),
    }

    (out_dir / "results_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    bootstrap_txt = "\n".join(
        [
            f"n_galaxies={overview['n_galaxies']}",
            f"n_valid_beta={overview['n_valid_beta']}",
            f"mean_beta={overview['mean_beta']:.6f}",
            f"median_beta={overview['median_beta']:.6f}",
            f"std_beta={overview['std_beta']:.6f}",
            f"t_stat_vs_0_5={overview['t_stat_vs_0_5']:.6f}",
            f"p_value_vs_0_5={overview['p_value_vs_0_5']:.6g}",
            f"bootstrap_iterations={bootstrap['iterations']}",
            f"bootstrap_mean_of_means={bootstrap['mean_of_means']:.6f}",
            f"bootstrap_std_of_means={bootstrap['std_of_means']:.6f}",
            f"bootstrap_ci95=[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]",
            f"cv_folds={len(cv_df)}",
            f"cv_mean_rmse={overview['cv_mean_rmse']:.6f}" if np.isfinite(overview["cv_mean_rmse"]) else "cv_mean_rmse=nan",
            f"cv_mean_mae={overview['cv_mean_mae']:.6f}" if np.isfinite(overview["cv_mean_mae"]) else "cv_mean_mae=nan",
        ]
    )
    (out_dir / "bootstrap_stats.txt").write_text(bootstrap_txt + "\n", encoding="utf-8")
    return overview


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BIG-SPARC β test pipeline.")
    parser.add_argument("--catalog", required=True, help="Input CSV with at least galaxy,g_obs,g_bar.")
    parser.add_argument("--out", default="results", help="Output directory for result artifacts.")
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=BOOTSTRAP_ITERS_DEFAULT,
        dest="bootstrap_iters",
        help=f"Bootstrap iterations (default: {BOOTSTRAP_ITERS_DEFAULT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED_DEFAULT,
        help=f"Random seed for bootstrap (default: {BOOTSTRAP_SEED_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    return run_test(
        catalog_path=Path(args.catalog),
        out_dir=Path(args.out),
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
