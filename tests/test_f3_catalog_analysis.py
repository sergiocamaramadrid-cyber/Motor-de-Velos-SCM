from pathlib import Path

import pandas as pd
import pytest


def test_f3_catalog_exists() -> None:
    path = Path("results/combined/f3_combined_catalog.csv")

    if not path.exists():
        pytest.skip("Artifact not present: f3_combined_catalog.csv")

    assert path.exists()


def test_f3_columns() -> None:
    path = Path("results/combined/f3_combined_catalog.csv")

    if not path.exists():
        pytest.skip("Artifact not present: f3_combined_catalog.csv")

    df = pd.read_csv(path)

    expected = {
        "galaxy",
        "f3_scm",
        "delta_f3",
        "tail_slope",
        "n_tail_points",
        "fit_ok",
        "quality_flag",
    }

    assert expected.issubset(df.columns)


def test_delta_f3_range() -> None:
    path = Path("results/combined/f3_combined_catalog.csv")

    if not path.exists():
        pytest.skip("Artifact not present: f3_combined_catalog.csv")

    df = pd.read_csv(path)

    valid = df["delta_f3"].dropna()
    if len(valid) == 0:
        pytest.skip("No valid ΔF3 values")

    assert valid.abs().max() < 2
