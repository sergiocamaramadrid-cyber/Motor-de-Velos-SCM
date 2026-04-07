from pathlib import Path

import pandas as pd

from scripts.generate_env_figure import find_columns, try_load


def test_find_columns_exact_match():
    df = pd.DataFrame(
        {
            "delta_mass_std": [0.1, 0.2],
            "slope_tail": [0.3, 0.4],
        }
    )

    xcol, ycol = find_columns(df)

    assert xcol == "delta_mass_std"
    assert ycol == "slope_tail"


def test_find_columns_heuristic():
    df = pd.DataFrame(
        {
            "delta_mass_proxy": [0.1, 0.2],
            "outer_slope_tail": [0.3, 0.4],
        }
    )

    xcol, ycol = find_columns(df)

    assert xcol == "delta_mass_proxy"
    assert ycol == "outer_slope_tail"


def test_find_columns_none_when_no_match():
    df = pd.DataFrame(
        {
            "mass": [1, 2],
            "velocity": [3, 4],
        }
    )

    xcol, ycol = find_columns(df)

    assert xcol is None
    assert ycol is None


def test_try_load_too_small(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"delta_mass_std": [0.1], "slope_tail": [0.2]}).to_csv(path, index=False)

    loaded = try_load(path)

    assert loaded is None


def test_try_load_returns_none_on_bad_extension(tmp_path):
    path = tmp_path / "not_csv.txt"
    path.write_text("delta_mass_std,slope_tail\n0.1,0.2\n0.2,0.3\n")

    loaded = try_load(path)

    assert loaded is None
