#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KPC_TO_M = 3.085677581e19
CONV = 1e6 / KPC_TO_M


def _nu_simple(x: np.ndarray) -> np.ndarray:
    safe = np.maximum(x, 1e-12)
    return 1.0 / (1.0 - np.exp(-np.sqrt(safe)))


def _delta_rmse(g_bar: np.ndarray, g_obs: np.ndarray, a0: float) -> float:
    x = g_bar / a0
    g_pred_baseline = _nu_simple(x) * g_bar
    g_pred_scm = g_bar + a0
    rmse_baseline = float(np.sqrt(np.mean((g_obs - g_pred_baseline) ** 2)))
    rmse_scm = float(np.sqrt(np.mean((g_obs - g_pred_scm) ** 2)))
    return rmse_scm - rmse_baseline


def run_shuffled_control(
    comparison_csv: Path,
    out_csv: Path,
    n_shuffles: int = 1000,
    seed: int = 42,
    a0: float = 1.2e-10,
) -> pd.DataFrame:
    df = pd.read_csv(comparison_csv)
    required = {"g_bar", "g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {comparison_csv}: {sorted(missing)}")

    g_bar = df["g_bar"].to_numpy(dtype=float)
    g_obs = df["g_obs"].to_numpy(dtype=float)

    delta_real = _delta_rmse(g_bar, g_obs, a0=a0)
    rng = np.random.default_rng(seed)
    shuffled_deltas: list[float] = []
    scm_wins = 0
    for _ in range(int(n_shuffles)):
        shuffled_obs = rng.permutation(g_obs)
        delta = _delta_rmse(g_bar, shuffled_obs, a0=a0)
        shuffled_deltas.append(delta)
        if delta < 0:
            scm_wins += 1

    shuffled_arr = np.asarray(shuffled_deltas, dtype=float)
    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "n_shuffles": int(n_shuffles),
                "delta_rmse_real": float(delta_real),
                "delta_rmse_shuffled_mean": float(np.mean(shuffled_arr)),
                "delta_rmse_shuffled_std": float(np.std(shuffled_arr, ddof=1)) if len(shuffled_arr) > 1 else 0.0,
                "scm_preference_ratio_shuffled": float(scm_wins / max(int(n_shuffles), 1)),
                "scm_preference_ratio_expected": 0.5,
            }
        ]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run shuffled control test on SCM OOS comparison table.")
    p.add_argument(
        "--comparison-csv",
        default="results/universal_term_comparison_full.csv",
        help="Input CSV with columns g_bar and g_obs.",
    )
    p.add_argument(
        "--out",
        default="results/shuffled_control_results.csv",
        help="Output CSV path.",
    )
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--a0", type=float, default=1.2e-10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_shuffled_control(
        comparison_csv=Path(args.comparison_csv),
        out_csv=Path(args.out),
        n_shuffles=int(args.n_shuffles),
        seed=int(args.seed),
        a0=float(args.a0),
    )
    row = summary.iloc[0]
    print(f"delta_rmse_real={row['delta_rmse_real']}")
    print(f"scm_preference_ratio_shuffled={row['scm_preference_ratio_shuffled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
