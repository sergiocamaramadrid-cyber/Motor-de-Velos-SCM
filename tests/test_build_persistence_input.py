import pandas as pd
import pytest

from scripts.experimental.build_persistence_input import build_persistence_input


def test_build_persistence_input_creates_ratio_and_mass_columns():
    df = pd.DataFrame(
        {
            "galaxy": ["A", "B"],
            "logMbar": [9.0, 10.0],
            "g_obs": [1.0e-12, 2.0e-12],
            "g_bar": [0.5e-12, 1.0e-12],
        }
    )

    out = build_persistence_input(df, max_gbar=1.0e-11)

    assert {"Mbar", "logMbar", "g_obs", "g_bar", "r"}.issubset(out.columns)
    assert len(out) == 2
    assert out["r"].iloc[0] == pytest.approx(2.0)


def test_build_persistence_input_requires_core_columns():
    df = pd.DataFrame({"logMbar": [9.0]})
    with pytest.raises(ValueError, match="required columns"):
        build_persistence_input(df)
