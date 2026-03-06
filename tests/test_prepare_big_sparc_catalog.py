from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prepare_big_sparc_catalog import _KILOPARSEC_TO_METERS, prepare_catalog


def test_prepare_catalog_passthrough_when_g_columns_exist(tmp_path):
    inp = tmp_path / "input.csv"
    out = tmp_path / "big_sparc_catalog.csv"
    pd.DataFrame(
        [
            {"galaxy": "A", "g_obs": 2.0e-11, "g_bar": 1.1e-11, "logMbar": 10.2},
            {"galaxy": "A", "g_obs": 1.7e-11, "g_bar": 0.9e-11, "logMbar": 10.2},
            {"galaxy": "B", "g_obs": 1.4e-11, "g_bar": 0.8e-11, "logMbar": 9.9},
        ]
    ).to_csv(inp, index=False)

    df = prepare_catalog(inp, out)
    assert out.exists()
    assert list(df.columns) == ["galaxy", "g_obs", "g_bar", "logMbar"]
    assert len(df) == 3


def test_prepare_catalog_converts_contract_columns_to_acceleration(tmp_path):
    inp = tmp_path / "contract.csv"
    out = tmp_path / "big_sparc_catalog.csv"
    pd.DataFrame(
        [
            {"galaxy": "G1", "r_kpc": 1.0, "vobs_kms": 100.0, "vbar_kms": 80.0},
            {"galaxy": "G1", "r_kpc": 2.0, "vobs_kms": 120.0, "vbar_kms": 95.0},
            {"galaxy": "G2", "r_kpc": 1.5, "vobs_kms": 90.0, "vbar_kms": 70.0},
        ]
    ).to_csv(inp, index=False)

    df = prepare_catalog(inp, out)
    assert out.exists()
    assert list(df.columns) == ["galaxy", "g_obs", "g_bar"]
    assert len(df) == 3
    assert np.all(df["g_obs"] > 0.0)
    assert np.all(df["g_bar"] > 0.0)
    expected_g_obs_first = ((100.0 * 1_000.0) ** 2) / (1.0 * _KILOPARSEC_TO_METERS)
    expected_g_bar_first = ((80.0 * 1_000.0) ** 2) / (1.0 * _KILOPARSEC_TO_METERS)
    assert np.isclose(df.iloc[0]["g_obs"], expected_g_obs_first)
    assert np.isclose(df.iloc[0]["g_bar"], expected_g_bar_first)
