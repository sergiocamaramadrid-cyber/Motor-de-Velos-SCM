from __future__ import annotations

import json

import pandas as pd

from scripts.run_big_sparc_veil_test import run_test


def test_run_big_sparc_veil_test_writes_expected_outputs(tmp_path):
    catalog = tmp_path / "big_sparc_catalog.csv"
    pd.DataFrame(
        [
            {"galaxy": "G1", "g_obs": 2.5e-11, "g_bar": 1.5e-11, "logMbar": 10.1, "logSigmaHI_out": 0.1},
            {"galaxy": "G1", "g_obs": 2.0e-11, "g_bar": 1.2e-11, "logMbar": 10.1, "logSigmaHI_out": 0.1},
            {"galaxy": "G1", "g_obs": 1.6e-11, "g_bar": 9.0e-12, "logMbar": 10.1, "logSigmaHI_out": 0.1},
            {"galaxy": "G2", "g_obs": 1.9e-11, "g_bar": 1.4e-11, "logMbar": 9.9, "logSigmaHI_out": 0.05},
            {"galaxy": "G2", "g_obs": 1.5e-11, "g_bar": 1.0e-11, "logMbar": 9.9, "logSigmaHI_out": 0.05},
            {"galaxy": "G2", "g_obs": 1.1e-11, "g_bar": 7.0e-12, "logMbar": 9.9, "logSigmaHI_out": 0.05},
        ]
    ).to_csv(catalog, index=False)

    out_dir = tmp_path / "results"
    overview = run_test(catalog, out_dir, bootstrap_iters=200, seed=123)

    beta_catalog = out_dir / "beta_catalog.csv"
    overview_json = out_dir / "results_overview.json"
    bootstrap_txt = out_dir / "bootstrap_stats.txt"

    assert beta_catalog.exists()
    assert overview_json.exists()
    assert bootstrap_txt.exists()

    parsed = json.loads(overview_json.read_text(encoding="utf-8"))
    assert parsed["n_galaxies"] == 2
    assert parsed["n_valid_beta"] == 2
    assert "bootstrap" in parsed
    assert overview["n_galaxies"] == 2
