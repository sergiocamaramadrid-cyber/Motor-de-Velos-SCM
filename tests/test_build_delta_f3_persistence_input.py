import pandas as pd
import pytest

from scripts.experimental.build_delta_f3_persistence_input import build_delta_f3_persistence_input


def test_build_delta_f3_persistence_input_happy_path_filters_and_sorts():
    df = pd.DataFrame(
        {
            "galaxy": ["B", "A", "A", "A", "C"],
            "logMbar": [9.2, 8.8, 9.0, 9.1, 9.3],
            "delta_f3": [0.1, 0.2, float("inf"), 0.3, 0.4],
            "fit_ok": [True, True, True, False, True],
            "reliable": [True, True, True, True, True],
            "quality_flag": ["ok", "ok", "ok", "bad", "ok"],
        }
    )

    out = build_delta_f3_persistence_input(df)

    assert out.columns.tolist() == ["galaxy", "order_var", "delta_f3", "fit_ok", "reliable", "quality_flag"]
    assert out["galaxy"].tolist() == ["A", "B", "C"]
    assert out["order_var"].tolist() == [8.8, 9.2, 9.3]
    assert out["delta_f3"].tolist() == [0.2, 0.1, 0.4]


def test_build_delta_f3_persistence_input_accepts_fallback_columns():
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G2"],
            "r_kpc": [2.0, 1.0, 1.5],
            "DeltaF3": [0.03, 0.01, 0.02],
        }
    )

    out = build_delta_f3_persistence_input(df)

    assert out.columns.tolist() == ["galaxy", "order_var", "delta_f3"]
    assert out["order_var"].tolist() == [1.0, 2.0, 1.5]


def test_build_delta_f3_persistence_input_requires_galaxy():
    df = pd.DataFrame({"logMbar": [9.0, 9.1, 9.2], "delta_f3": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="galaxy"):
        build_delta_f3_persistence_input(df)


def test_build_delta_f3_persistence_input_requires_minimum_rows():
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G2", "G3"],
            "logMbar": [9.0, 9.1, 9.2],
            "delta_f3": [0.1, 0.2, 0.3],
            "fit_ok": [True, False, False],
        }
    )
    with pytest.raises(ValueError, match="Too few valid rows"):
        build_delta_f3_persistence_input(df)
