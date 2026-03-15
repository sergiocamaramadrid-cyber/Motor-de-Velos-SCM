#!/usr/bin/env python3
"""
oos_environment_validation.py

Out-of-sample comparison:
  baseline: delta_f3 ~ logMbar + logRd
  full:     delta_f3 ~ logMbar + logRd + logSigmaHI_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["delta_f3", "logSigmaHI_out", "logMbar", "logRd"]


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    n_initial = int(len(df))
    n_used = int(len(clean))
    n_removed = int(n_initial - n_used)
    return clean, n_initial, n_used, n_removed


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    x_train = np.column_stack([np.ones(len(train)), train[predictors].to_numpy(dtype=float)])
    y_train = train["delta_f3"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    x_test = np.column_stack([np.ones(len(test)), test[predictors].to_numpy(dtype=float)])
    return x_test @ beta


def _gaussian_logl(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    sigma2 = float(np.mean(err**2))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return float("nan")
    n = len(err)
    return float(-0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0))


def run_oos(
    df: pd.DataFrame,
    test_size: float = 0.3,
    repeats: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    n = len(df)
    n_test = max(1, int(round(test_size * n)))
    for i in range(repeats):
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        if len(train_idx) < 5:
            continue
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_true = test["delta_f3"].to_numpy(dtype=float)

        y_pred_base = _fit_predict(train, test, ["logMbar", "logRd"])
        y_pred_full = _fit_predict(train, test, ["logMbar", "logRd", "logSigmaHI_out"])

        rmse_base = float(np.sqrt(np.mean((y_true - y_pred_base) ** 2)))
        rmse_full = float(np.sqrt(np.mean((y_true - y_pred_full) ** 2)))
        mae_base = float(np.mean(np.abs(y_true - y_pred_base)))
        mae_full = float(np.mean(np.abs(y_true - y_pred_full)))
        logl_base = _gaussian_logl(y_true, y_pred_base)
        logl_full = _gaussian_logl(y_true, y_pred_full)

        rows.append(
            {
                "repeat_id": int(i),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "rmse_out_baseline": rmse_base,
                "rmse_out_full": rmse_full,
                "mae_out_baseline": mae_base,
                "mae_out_full": mae_full,
                "delta_rmse_out": rmse_full - rmse_base,
                "delta_logL_out": logl_full - logl_base,
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    outdir: Path,
    per_repeat: pd.DataFrame,
    n_initial: int,
    n_used: int,
    n_removed: int,
    repeats: int,
    test_size: float,
    seed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(outdir / "oos_repeats.csv", index=False)

    summary = {
        "n_initial": int(n_initial),
        "n_used": int(n_used),
        "n_removed_nan": int(n_removed),
        "repeats_requested": int(repeats),
        "repeats_used": int(len(per_repeat)),
        "test_size": float(test_size),
        "seed": int(seed),
        "RMSE_out_baseline_mean": float(per_repeat["rmse_out_baseline"].mean()),
        "RMSE_out_full_mean": float(per_repeat["rmse_out_full"].mean()),
        "MAE_out_baseline_mean": float(per_repeat["mae_out_baseline"].mean()),
        "MAE_out_full_mean": float(per_repeat["mae_out_full"].mean()),
        "delta_RMSE_out_mean": float(per_repeat["delta_rmse_out"].mean()),
        "delta_logL_out_mean": float(per_repeat["delta_logL_out"].mean()),
    }
    pd.DataFrame([summary]).to_csv(outdir / "oos_summary.csv", index=False)
    with (outdir / "oos_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6.5, 4.0))
    plt.hist(per_repeat["delta_rmse_out"], bins=20)
    plt.axvline(float(np.median(per_repeat["delta_rmse_out"])), linestyle="--", color="tab:red")
    plt.xlabel("ΔRMSE_out = RMSE_full - RMSE_baseline")
    plt.ylabel("Count")
    plt.title("OOS ΔRMSE_out distribution")
    plt.tight_layout()
    plt.savefig(outdir / "hist_delta_rmse_out.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sparc_175_master.csv")
    parser.add_argument("--out", default="results/oos_environment")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    check_columns(df)
    clean, n_initial, n_used, n_removed = prepare_data(df)
    per_repeat = run_oos(clean, test_size=args.test_size, repeats=args.repeats, seed=args.seed)
    save_outputs(
        Path(args.out),
        per_repeat,
        n_initial=n_initial,
        n_used=n_used,
        n_removed=n_removed,
        repeats=args.repeats,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
