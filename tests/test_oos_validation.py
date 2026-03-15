from __future__ import annotations

import pandas as pd

from scripts.run_oos_validation import run_oos


def test_run_oos_returns_bootstrap_metrics() -> None:
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G2", "G2", "G3", "G3"],
            "g_bar": [1e-11, 2e-11, 1.5e-11, 2.2e-11, 1.2e-11, 1.9e-11],
            "g_obs": [1.05e-11, 2.2e-11, 1.55e-11, 2.3e-11, 1.22e-11, 2.01e-11],
        }
    )

    out = run_oos(df, seed=7, n_bootstrap=10, a0=1.2e-10)
    assert not out.empty
    for col in [
        "rmse_out_baseline",
        "rmse_out_scm",
        "delta_rmse_out",
        "logL_out_baseline",
        "logL_out_scm",
        "delta_logL_out",
    ]:
        assert col in out.columns
