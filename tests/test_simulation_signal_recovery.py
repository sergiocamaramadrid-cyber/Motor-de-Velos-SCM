from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_simulation_recovers_injected_negative_sigma_signal() -> None:
    rng = np.random.default_rng(42)
    n = 1200

    log_sigma_star_out = rng.normal(0.0, 1.0, n)
    log_mb = 0.45 * log_sigma_star_out + rng.normal(0.0, 1.0, n)
    log_rd = 0.30 * log_mb + rng.normal(0.0, 1.0, n)
    incl = 60.0 + 10.0 * rng.normal(0.0, 1.0, n)
    q = 2.0 + 0.2 * rng.normal(0.0, 1.0, n)

    noise = rng.normal(0.0, 0.11, n)
    y = (
        0.4192
        - 0.1855 * log_sigma_star_out
        + 0.0550 * log_mb
        - 0.0508 * log_rd
        - 0.0010 * incl
        + 0.0901 * q
        + noise
    )

    x = pd.DataFrame(
        {
            "log_Sigma_star_out": log_sigma_star_out,
            "log_Mb": log_mb,
            "log_Rd": log_rd,
            "i": incl,
            "Q": q,
        }
    )

    idx = rng.permutation(n)
    cut = int(0.7 * n)
    train_idx, test_idx = idx[:cut], idx[cut:]

    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
    coef_sigma = float(model.params["log_Sigma_star_out"])
    p_sigma = float(model.pvalues["log_Sigma_star_out"])

    y_pred_full = model.predict(sm.add_constant(x_test))
    y_pred_baseline = np.full_like(y_test, fill_value=float(np.mean(y_train)))

    rmse_full = _rmse(y_test, y_pred_full)
    rmse_baseline = _rmse(y_test, y_pred_baseline)

    sse_full = float(np.sum((y_test - y_pred_full) ** 2))
    sse_baseline = float(np.sum((y_test - y_pred_baseline) ** 2))
    sst = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2_full = 1.0 - sse_full / sst
    r2_baseline = 1.0 - sse_baseline / sst

    boot_coefs = []
    for _ in range(250):
        b = rng.integers(0, len(x_train), len(x_train))
        m = sm.OLS(y_train[b], sm.add_constant(x_train.iloc[b])).fit()
        boot_coefs.append(float(m.params["log_Sigma_star_out"]))
    ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])

    assert coef_sigma < 0.0
    assert p_sigma < 1e-10
    assert ci_high < 0.0
    assert r2_full > r2_baseline
    assert rmse_full < rmse_baseline
