import math
import numpy as np
import pandas as pd

from scripts import compare_nu_models as mod


def test_log_likelihood_gaussian():
    v_obs = np.array([1.0, 2.0, 3.0, 4.0])
    v_obs_err = np.array([0.1, 0.1, 0.1, 0.1])
    v_pred_good = np.array([1.0, 2.0, 3.0, 4.0])
    v_pred_worse = np.array([1.5, 2.5, 3.5, 4.5])

    ll_good = mod.log_likelihood(v_obs, v_obs_err, v_pred_good)
    ll_worse = mod.log_likelihood(v_obs, v_obs_err, v_pred_worse)

    assert isinstance(ll_good, float)
    assert math.isfinite(ll_good)
    # peor ajuste → menor logL
    assert ll_worse < ll_good


def test_aicc_penalizes_more_params():
    logL = -10.0
    n = 50

    aicc_k2 = mod.aicc(logL, k=2, n=n)
    aicc_k5 = mod.aicc(logL, k=5, n=n)

    # más parámetros → peor (AICc más alto)
    assert aicc_k5 > aicc_k2


def test_run_csv_comparison_roundtrip(tmp_path):
    path = tmp_path / "data.csv"

    df = pd.DataFrame(
        {
            "galaxy": ["NGC2403", "NGC3198", "DDO154"],
            "chi2_reduced": [1.1, 0.9, 1.3],
            "n_points": [30, 25, 20],
        }
    )
    df.to_csv(path, index=False)

    result_df, winner = mod.run_csv_comparison(path, tmp_path)

    assert result_df is not None
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0
    assert isinstance(winner, str)
    assert len(winner) > 0
