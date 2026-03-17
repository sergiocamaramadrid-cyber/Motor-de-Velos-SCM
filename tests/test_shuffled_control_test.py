from pathlib import Path

import pandas as pd

from shuffled_control_test import run_shuffled_control


def test_shuffled_control_generates_summary(tmp_path: Path):
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame(
        {
            "g_bar": [1.0e-11, 2.0e-11, 3.0e-11, 4.0e-11, 5.0e-11, 6.0e-11],
            "g_obs": [1.1e-11, 2.1e-11, 2.8e-11, 4.2e-11, 4.9e-11, 6.1e-11],
        }
    ).to_csv(comparison, index=False)

    out_csv = tmp_path / "shuffled_control_results.csv"
    summary = run_shuffled_control(
        comparison_csv=comparison,
        out_csv=out_csv,
        n_shuffles=50,
        seed=123,
    )

    assert out_csv.exists()
    assert len(summary) == 1
    assert {"preference_mean", "preference_std", "n_shuffles", "status"}.issubset(summary.columns)
    assert 0.0 <= float(summary.loc[0, "preference_mean"]) <= 1.0
    assert summary.loc[0, "status"] == "ok"
