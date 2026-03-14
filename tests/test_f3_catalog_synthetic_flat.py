from pathlib import Path
import pytest
import pandas as pd


def test_synthetic_flat_delta_f3_is_near_zero():
    path = Path("results/f3_catalog_synthetic_flat.csv")

    if not path.exists():
        pytest.skip("Artifact not present: results/f3_catalog_synthetic_flat.csv")

    df = pd.read_csv(path)

    required = {"f3_scm", "delta_f3"}
    missing = required - set(df.columns)
    if missing:
        pytest.skip(f"Missing required columns in artifact: {sorted(missing)}")

    if df["f3_scm"].dropna().empty:
        pytest.skip("No valid f3_scm values in synthetic flat artifact")

    valid = df["delta_f3"].dropna()
    if len(valid) == 0:
        pytest.skip("No valid delta_f3 values in synthetic flat artifact")

    mean_delta = float(valid.mean())

    assert abs(mean_delta) < 0.05, (
        f"Expected mean(delta_f3) near 0 for synthetic flat catalog, got {mean_delta:.6f}"
    )
