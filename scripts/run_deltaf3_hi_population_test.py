from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REQUIRED_COLUMNS = {"galaxy", "delta_f3", "logSigmaHI_out"}
CONTROL_COLUMNS = ["logMstar", "Rdisk", "inclination"]


def _fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict:
    x = df[x_cols].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(df)), x])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coefs
    resid = y - y_hat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")
    return {
        "intercept": float(coefs[0]),
        "coefficients": {name: float(coefs[i + 1]) for i, name in enumerate(x_cols)},
        "r2": r2,
        "predictions": y_hat,
    }


def _choose_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    available_controls = [c for c in CONTROL_COLUMNS if c in df.columns]
    full_features = ["logSigmaHI_out", *available_controls]
    full_data = df[["delta_f3", *full_features]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(full_data) >= len(full_features) + 1:
        return full_features, full_data

    fallback_features = ["logSigmaHI_out"]
    fallback_data = df[["delta_f3", *fallback_features]].replace([np.inf, -np.inf], np.nan).dropna()
    return fallback_features, fallback_data


def _run_oos(reg_df: pd.DataFrame, features: list[str], seed: int, test_fraction: float) -> dict:
    n = len(reg_df)
    if n < 3:
        return {
            "train_n": 0,
            "test_n": 0,
            "coef_logSigmaHI_out": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "baseline_rmse": float("nan"),
            "delta_rmse": float("nan"),
            "coef_positive": False,
        }

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = min(max(1, int(round(n * test_fraction))), n - 1)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    train = reg_df.iloc[train_idx]
    test = reg_df.iloc[test_idx]

    model = _fit_ols(train, y_col="delta_f3", x_cols=features)
    coef_sigma = model["coefficients"]["logSigmaHI_out"]

    x_test = np.column_stack([np.ones(len(test)), test[features].to_numpy(dtype=float)])
    pred = x_test @ np.array([model["intercept"], *[model["coefficients"][f] for f in features]])
    y_test = test["delta_f3"].to_numpy(dtype=float)
    err = y_test - pred

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    baseline_pred = float(np.mean(train["delta_f3"]))
    baseline_rmse = float(np.sqrt(np.mean((y_test - baseline_pred) ** 2)))

    return {
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "coef_logSigmaHI_out": float(coef_sigma),
        "rmse": rmse,
        "mae": mae,
        "baseline_rmse": baseline_rmse,
        "delta_rmse": float(baseline_rmse - rmse),
        "coef_positive": bool(coef_sigma > 0),
    }


def run_analysis(table_path: Path, out_dir: Path, seed: int = 42, test_fraction: float = 0.3) -> dict:
    df = pd.read_csv(table_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Population table missing required columns: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    scatter_df = (
        df[["delta_f3", "logSigmaHI_out"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    plt.figure(figsize=(7, 4.5))
    if not scatter_df.empty:
        plt.scatter(scatter_df["logSigmaHI_out"], scatter_df["delta_f3"], alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No valid points", ha="center", va="center")
        plt.axis("off")
    plt.xlabel("logSigmaHI_out")
    plt.ylabel("delta_f3")
    plt.tight_layout()
    plt.savefig(out_dir / "deltaf3_vs_logSigmaHI_out.png", dpi=150)
    plt.close()

    if len(scatter_df) >= 2:
        pearson_r, pearson_p = pearsonr(scatter_df["logSigmaHI_out"], scatter_df["delta_f3"])
        spearman_rho, spearman_p = spearmanr(scatter_df["logSigmaHI_out"], scatter_df["delta_f3"])
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")
    correlation = {
        "n_samples": int(len(scatter_df)),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
    }

    features, reg_df = _choose_features(df)
    if len(reg_df) >= len(features) + 1:
        reg_fit = _fit_ols(reg_df, y_col="delta_f3", x_cols=features)
        regression = {
            "features": features,
            "n_samples": int(len(reg_df)),
            "intercept": reg_fit["intercept"],
            "coefficients": reg_fit["coefficients"],
            "r2": reg_fit["r2"],
        }
    else:
        regression = {
            "features": features,
            "n_samples": int(len(reg_df)),
            "intercept": float("nan"),
            "coefficients": {f: float("nan") for f in features},
            "r2": float("nan"),
        }

    oos = _run_oos(reg_df, features=features, seed=seed, test_fraction=test_fraction)

    summary = {
        "input_table": str(table_path),
        "n_rows": int(len(df)),
        "correlation": correlation,
        "regression": regression,
        "oos_70_30": oos,
    }
    (out_dir / "deltaf3_hi_population_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full population ΔF3 vs logSigmaHI_out analysis."
    )
    parser.add_argument("--table", required=True, help="Input master table CSV.")
    parser.add_argument(
        "--out",
        default="results/big_sparc_f3_hi_analysis",
        help="Output directory for plots and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 70/30 split.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        dest="test_fraction",
        help="Out-of-sample test fraction (default: 0.3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    return run_analysis(
        table_path=Path(args.table),
        out_dir=Path(args.out),
        seed=args.seed,
        test_fraction=args.test_fraction,
    )


if __name__ == "__main__":
    main()
