from pathlib import Path
import pytest
import pandas as pd


def test_f3_catalog_columns():
    path = Path("results/combined/f3_combined_catalog.csv")

    if not path.exists():
        pytest.skip("Artifact not present: results/combined/f3_combined_catalog.csv")

    df = pd.read_csv(path)

    expected = {
        "galaxy",
        "f3_scm",
        "delta_f3",
        "tail_slope",
        "n_tail_points",
        "tail_rmin",
        "fit_method",
        "fit_ok",
        "fit_ok_reason",
        "quality_flag",
    }

    assert expected.issubset(set(df.columns))
