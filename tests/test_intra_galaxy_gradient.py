from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.intra_galaxy_gradient import analyze_single_galaxy
from scripts.intra_galaxy_gradient import run_analysis
from scripts.intra_galaxy_gradient import save_outputs


def _synthetic_galaxy(galaxy: str, n: int = 8) -> pd.DataFrame:
    r = np.linspace(1.0, 8.0, n)
    gbar = np.linspace(1e-12, 4e-12, n)
    sb = np.linspace(40.0, 10.0, n)
    gobs = gbar * (1.2 + 0.02 * np.arange(n))
    return pd.DataFrame({"galaxy": galaxy, "r": r, "gbar": gbar, "gobs": gobs, "SB": sb})


def test_analyze_single_galaxy_requires_min_rings() -> None:
    df = _synthetic_galaxy("G1", n=5)
    assert analyze_single_galaxy(df) is None


def test_run_analysis_produces_expected_columns() -> None:
    df = pd.concat([_synthetic_galaxy("G1", n=8), _synthetic_galaxy("G2", n=9)], ignore_index=True)
    out = run_analysis(df)

    assert len(out) == 2
    expected = {"galaxy", "a_grad", "b_gbar", "c", "rmse", "rmse_null", "delta_rmse", "n_rings", "n_pairs"}
    assert expected.issubset(out.columns)
    assert np.isfinite(out["a_grad"]).all()
    assert np.isfinite(out["b_gbar"]).all()


def test_run_analysis_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="Missing required radial columns"):
        run_analysis(pd.DataFrame({"galaxy": ["G1"], "r": [1.0]}))


def test_analyze_single_galaxy_handles_duplicate_r_without_crash() -> None:
    df = _synthetic_galaxy("G1", n=8)
    df.loc[3, "r"] = df.loc[2, "r"]
    out = analyze_single_galaxy(df)
    assert out is not None


def test_save_outputs_writes_csv_txt_png(tmp_path: Path) -> None:
    res_df = pd.DataFrame(
        [
            {"galaxy": "G1", "a_grad": -0.1, "b_gbar": 0.3, "c": 0.01, "rmse": 0.1, "rmse_null": 0.2, "delta_rmse": -0.1, "n_rings": 8, "n_pairs": 7},
            {"galaxy": "G2", "a_grad": 0.2, "b_gbar": 0.1, "c": -0.02, "rmse": 0.15, "rmse_null": 0.18, "delta_rmse": -0.03, "n_rings": 9, "n_pairs": 8},
        ]
    )

    save_outputs(res_df, str(tmp_path))

    assert (tmp_path / "intra_galaxy_fits.csv").exists()
    assert (tmp_path / "summary.txt").exists()
    assert (tmp_path / "coef_hist.png").exists()
