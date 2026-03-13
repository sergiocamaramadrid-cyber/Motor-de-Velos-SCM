from pathlib import Path
import pandas as pd
import pytest


def test_f3_catalog_columns():
    path = Path("results/combined/f3_combined_catalog.csv")

    if not path.exists():
        pytest.skip("f3_combined_catalog.csv not generated in this environment")

    df = pd.read_csv(path)

    expected = {
        "galaxy",
        "f3_scm",
        "delta_f3",
        "n_tail_points",
        "fit_ok",
    }

    assert expected.issubset(set(df.columns))
