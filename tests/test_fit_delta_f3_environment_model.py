import numpy as np
import pandas as pd
import pytest

from scripts.fit_delta_f3_environment_model import check_columns, prepare_data, run_model


def test_missing_columns_fails():
    df = pd.DataFrame({"delta_f3": [0.1], "logSigmaHI_out": [0.2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        check_columns(df)


def test_nan_filtering_counts():
    df = pd.DataFrame(
        {
            "delta_f3": [0.1, np.nan, 0.2],
            "logSigmaHI_out": [0.3, 0.4, 0.5],
            "logMbar": [9.0, 9.1, np.nan],
            "logRd": [0.5, 0.6, 0.7],
            "inclination": [45.0, 50.0, 55.0],
        }
    )
    clean, n_initial, n_used, n_removed = prepare_data(df)
    assert n_initial == 3
    assert n_used == 1
    assert n_removed == 2
    assert len(clean) == 1


def test_synthetic_signal_recovery():
    np.random.seed(0)
    n = 200
    x1 = np.random.uniform(0.0, 1.0, n)  # logSigmaHI_out
    x2 = np.random.uniform(9.0, 11.0, n)  # logMbar
    x3 = np.random.uniform(0.2, 1.2, n)  # logRd
    x4 = np.random.uniform(30.0, 80.0, n)  # inclination
    y = 0.5 + 1.5 * x1 + 0.1 * x2 - 0.7 * x3 + 0.02 * x4
    df = pd.DataFrame(
        {
            "delta_f3": y,
            "logSigmaHI_out": x1,
            "logMbar": x2,
            "logRd": x3,
            "inclination": x4,
        }
    )
    coef_df, summary = run_model(df, n_bootstrap=100, seed=123)
    coef = dict(zip(coef_df["variable"], coef_df["coefficient"]))
    assert np.isclose(coef["intercept"], 0.5, atol=1e-6)
    assert np.isclose(coef["logSigmaHI_out"], 1.5, atol=1e-6)
    assert np.isclose(coef["logMbar"], 0.1, atol=1e-6)
    assert np.isclose(coef["logRd"], -0.7, atol=1e-6)
    assert np.isclose(coef["inclination"], 0.02, atol=1e-6)
    assert np.isclose(float(summary["R2"]), 1.0)
