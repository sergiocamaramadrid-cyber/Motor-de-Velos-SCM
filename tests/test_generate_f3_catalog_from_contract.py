from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import generate_f3_catalog_from_contract


def test_validate_columns_raises_when_required_missing():
    df = pd.DataFrame({"galaxy": ["G1"], "r": [1.0], "gbar": [1e-10]})

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_f3_catalog_from_contract.validate_columns(
            df, ["galaxy", "r", "gbar", "gobs"]
        )


def test_compute_f3_handles_zero_gbar_with_eps_floor():
    gobs = np.array([2.0, 5.0])
    gbar = np.array([1.0, 0.0])

    out = generate_f3_catalog_from_contract.compute_f3(gobs, gbar)

    assert np.isclose(out[0], 1.0)
    assert np.isfinite(out[1])
    assert out[1] > 0


def test_build_catalog_filters_invalid_rows_and_computes_delta_f3():
    df = pd.DataFrame(
        {
            "galaxy": ["B", "A", "A", "A", "B"],
            "r": [2.0, 2.0, 1.0, -1.0, 1.0],
            "gbar": [2.0, 4.0, 2.0, 1.0, 0.0],
            "gobs": [6.0, 12.0, 4.0, 10.0, 8.0],
            "SB": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    out = generate_f3_catalog_from_contract.build_catalog(df)

    assert out["galaxy"].tolist() == ["A", "A", "B"]
    assert out["r"].tolist() == [1.0, 2.0, 2.0]
    assert "SB" in out.columns

    expected_f3 = [(4.0 - 2.0) / 2.0, (12.0 - 4.0) / 4.0, (6.0 - 2.0) / 2.0]
    assert np.allclose(out["F3"].to_numpy(), np.array(expected_f3))

    assert np.isnan(out.loc[0, "delta_f3"])
    assert np.isclose(out.loc[1, "delta_f3"], expected_f3[1] - expected_f3[0])
    assert np.isnan(out.loc[2, "delta_f3"])


def test_build_catalog_accepts_velocity_contract_columns():
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1"],
            "r_kpc": [1.0, 2.0],
            "vobs_kms": [30.0, 40.0],
            "vbar_kms": [10.0, 20.0],
        }
    )

    out = generate_f3_catalog_from_contract.build_catalog(df)
    expected_gobs = np.array([9.724778419796843e-10, 6.483185613197896e-10])
    expected_gbar = np.array([1.0805309355329826e-10, 1.620796403299474e-10])
    expected_f3 = (expected_gobs - expected_gbar) / expected_gbar

    for col in [
        "gobs",
        "gbar",
        "F3",
        "delta_f3",
        "f3_scm",
        "fit_ok",
        "quality_flag",
        "beta",
        "beta_err",
        "reliable",
        "friction_slope",
        "velo_inerte_flag",
    ]:
        assert col in out.columns

    assert np.allclose(out["gobs"].to_numpy(), expected_gobs)
    assert np.allclose(out["gbar"].to_numpy(), expected_gbar)
    assert np.allclose(out["F3"].to_numpy(), expected_f3)
