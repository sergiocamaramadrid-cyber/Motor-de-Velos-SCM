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
