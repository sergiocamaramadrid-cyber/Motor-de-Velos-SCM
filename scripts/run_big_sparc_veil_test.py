"""
Run BIG-SPARC deep-regime β test from a per-point catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_1samp


REQUIRED_COLUMNS = {"galaxy", "g_obs", "g_bar"}
EXPECTED_BETA = 0.5
BOOTSTRAP_ITERS_DEFAULT = 2000
BOOTSTRAP_SEED_DEFAULT = 42


def compute_beta_from_curve(sub: pd.DataFrame) -> dict:
    """Compute β = dlog(g_obs)/dlog(g_bar) for one galaxy."""
    clean = sub[["g_obs", "g_bar"]].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[(clean["g_obs"] > 0) & (clean["g_bar"] > 0)]
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
    df = pd.read_csv(catalog_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {sorted(missing)}")

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
