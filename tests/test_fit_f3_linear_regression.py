import numpy as np
import pandas as pd

from scripts.fit_f3_linear_regression import run_regression


def test_linear_recovery():
    """The least-squares fit should exactly recover a noiseless linear model."""
    np.random.seed(0)

    n = 100

    logSigmaHI_out = np.random.rand(n)
    logMbar = np.random.rand(n)
    logRd = np.random.rand(n)

    f3 = (
        2.0 * logSigmaHI_out
        + 3.0 * logMbar
        - 1.0 * logRd
        + 5.0
    )

    df = pd.DataFrame(
        {
            "F3": f3,
            "logSigmaHI_out": logSigmaHI_out,
            "logMbar": logMbar,
            "logRd": logRd,
        }
    )

    intercept, coefs, r2, _ = run_regression(df)

    assert np.isclose(intercept, 5.0, atol=1e-6)
    assert np.allclose(coefs, [2.0, 3.0, -1.0], atol=1e-6)
    assert np.isclose(r2, 1.0)
