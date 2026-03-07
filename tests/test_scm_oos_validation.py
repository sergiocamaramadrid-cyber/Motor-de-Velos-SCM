from __future__ import annotations

import pandas as pd

from scripts.scm_oos_validation import run_oos_validation


def test_run_oos_validation_writes_expected_artifacts(tmp_path):
    csv_path = tmp_path / "comparison.csv"
    out_dir = tmp_path / "oos"
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G1", "G2", "G2", "G2"],
            "r_kpc": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "g_bar": [1.0e-11, 2.0e-11, 3.0e-11, 1.5e-11, 2.5e-11, 3.5e-11],
            "g_obs": [1.3e-11, 2.4e-11, 3.7e-11, 1.8e-11, 2.9e-11, 4.1e-11],
        }
    )
    df.to_csv(csv_path, index=False)

    n_gal, median_delta, p_value = run_oos_validation(csv_path, out_dir, a0=1.2e-10)

    assert n_gal == 2
    assert isinstance(median_delta, float)
    assert isinstance(p_value, float)

    result_csv = out_dir / "oos_generalization_results.csv"
    assert result_csv.exists()
    out_df = pd.read_csv(result_csv)
    assert set(["galaxy", "n_out", "rmse_out_baseline", "rmse_out_scm", "delta_rmse_out"]).issubset(
        out_df.columns
    )
    assert len(out_df) == 2

    assert (out_dir / "hist_delta_rmse_out.pdf").exists()
    log_path = out_dir / "oos_terminal_log.txt"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "n_galaxies = 2" in log_text
    assert "median ΔRMSE_out" in log_text
    assert "p-value Wilcoxon" in log_text
