#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/SPARC}"
MASTER_CSV="${MASTER_CSV:-$ROOT_DIR/data/sparc_175_master.csv}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
FIG_DIR="${RESULTS_DIR}/figures"

mkdir -p "${RESULTS_DIR}" "${FIG_DIR}"

python "${ROOT_DIR}/scripts/ingest_big_sparc_contract.py" \
  --data-root "${DATA_ROOT}" \
  --out "${MASTER_CSV}"

python - "${MASTER_CSV}" "${RESULTS_DIR}" "${FIG_DIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fit_simple(x: np.ndarray, y: np.ndarray) -> dict:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    resid = y - pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "intercept": float(coef[0]),
        "coef_logSigmaHI_out": float(coef[1]),
        "r2": float(1.0 - (sse / sst)) if sst > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid**2))),
    }


master_csv = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
fig_dir = Path(sys.argv[3])

df = pd.read_csv(master_csv)

delta = df[["delta_f3", "logSigmaHI_out"]].replace([np.inf, -np.inf], np.nan).dropna()
beta = df[["beta", "logSigmaHI_out"]].replace([np.inf, -np.inf], np.nan).dropna()

if len(delta) >= 2:
    delta_fit = _fit_simple(delta["logSigmaHI_out"].to_numpy(float), delta["delta_f3"].to_numpy(float))
else:
    delta_fit = {"intercept": float("nan"), "coef_logSigmaHI_out": float("nan"), "r2": float("nan"), "rmse": float("nan")}

if len(beta) >= 2:
    beta_fit = _fit_simple(beta["logSigmaHI_out"].to_numpy(float), beta["beta"].to_numpy(float))
else:
    beta_fit = {"intercept": float("nan"), "coef_logSigmaHI_out": float("nan"), "r2": float("nan"), "rmse": float("nan")}

delta_out = pd.DataFrame([{"model": "delta_f3 ~ logSigmaHI_out", **delta_fit, "n_samples": int(len(delta))}])
beta_out = pd.DataFrame([{"model": "beta ~ logSigmaHI_out", **beta_fit, "n_samples": int(len(beta))}])
delta_out.to_csv(results_dir / "delta_f3_regression.csv", index=False)
beta_out.to_csv(results_dir / "beta_regression.csv", index=False)

rng = np.random.default_rng(42)

if len(beta) >= 3:
    idx = np.arange(len(beta))
    rng.shuffle(idx)
    n_test = max(1, int(round(0.3 * len(beta))))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train = beta.iloc[train_idx]
    test = beta.iloc[test_idx]

    train_fit = _fit_simple(train["logSigmaHI_out"].to_numpy(float), train["beta"].to_numpy(float))
    test_x = test["logSigmaHI_out"].to_numpy(float)
    test_y = test["beta"].to_numpy(float)
    pred_full = train_fit["intercept"] + train_fit["coef_logSigmaHI_out"] * test_x
    pred_base = np.full_like(test_y, fill_value=float(train["beta"].mean()))
    rmse_full = float(np.sqrt(np.mean((test_y - pred_full) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((test_y - pred_base) ** 2)))
else:
    train_fit = {"r2": float("nan")}
    rmse_full = float("nan")
    rmse_baseline = float("nan")

if len(beta) >= 2:
    boot = []
    for _ in range(1000):
        sample = beta.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**32 - 1)))
        fit = _fit_simple(sample["logSigmaHI_out"].to_numpy(float), sample["beta"].to_numpy(float))
        boot.append(fit["coef_logSigmaHI_out"])
    boot_arr = np.asarray(boot, dtype=float)
    ci95_low = float(np.percentile(boot_arr, 2.5))
    ci95_high = float(np.percentile(boot_arr, 97.5))
else:
    ci95_low = float("nan")
    ci95_high = float("nan")

overview = {
    "n_galaxies": int(len(df)),
    "environmental_coefficient": {
        "name": "coef(logSigmaHI_out)",
        "value": beta_fit["coef_logSigmaHI_out"],
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
    },
    "oos": {
        "r2_full": train_fit["r2"],
        "r2_baseline": 0.0,
        "rmse_full": rmse_full,
        "rmse_baseline": rmse_baseline,
    },
}
(results_dir / "results_overview.json").write_text(
    json.dumps(overview, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

plt.figure(figsize=(6, 4))
if len(beta) > 0:
    plt.scatter(beta["logSigmaHI_out"], beta["beta"], alpha=0.8)
if len(beta) >= 2:
    xg = np.linspace(beta["logSigmaHI_out"].min(), beta["logSigmaHI_out"].max(), 100)
    yg = beta_fit["intercept"] + beta_fit["coef_logSigmaHI_out"] * xg
    plt.plot(xg, yg, color="tab:red")
plt.xlabel("logSigmaHI_out")
plt.ylabel("beta")
plt.tight_layout()
plt.savefig(fig_dir / "beta_vs_logSigmaHI_out.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
if len(delta) > 0:
    plt.scatter(delta["logSigmaHI_out"], delta["delta_f3"], alpha=0.8)
if len(delta) >= 2:
    xg = np.linspace(delta["logSigmaHI_out"].min(), delta["logSigmaHI_out"].max(), 100)
    yg = delta_fit["intercept"] + delta_fit["coef_logSigmaHI_out"] * xg
    plt.plot(xg, yg, color="tab:red")
plt.xlabel("logSigmaHI_out")
plt.ylabel("delta_f3")
plt.tight_layout()
plt.savefig(fig_dir / "deltaf3_vs_logSigmaHI_out.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
if len(beta) > 0:
    plt.hist(beta["beta"], bins=20, alpha=0.85)
else:
    plt.text(0.5, 0.5, "No valid beta values", ha="center", va="center")
    plt.axis("off")
plt.xlabel("beta")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(fig_dir / "beta_distribution.png", dpi=150)
plt.close()
PY

echo "[OK] Pipeline completed."
