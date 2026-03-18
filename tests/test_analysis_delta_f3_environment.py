from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis_delta_f3_environment import compute_aicc, run_analysis


def test_compute_aicc_handles_small_sample_and_zero_rss():
    assert np.isinf(compute_aicc(1.0, n=3, k=2))
    finite_val = compute_aicc(0.0, n=20, k=3)
    assert np.isfinite(finite_val)


def test_run_analysis_outputs_and_detects_environment_signal(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 250
    log_mbar = rng.uniform(9.0, 11.0, n)
    rdisk = rng.uniform(0.5, 5.0, n)
    incl = rng.uniform(30.0, 80.0, n)
    log_sigma_hi = rng.uniform(-1.0, 1.0, n)

    noise = rng.normal(0.0, 0.03, n)
    delta_f3 = 0.4 + 0.09 * log_mbar - 0.12 * rdisk + 0.01 * incl + 0.75 * log_sigma_hi + noise

    df = pd.DataFrame(
        {
            "delta_f3": delta_f3,
            "logMbar": log_mbar,
            "Rdisk": rdisk,
            "inclination": incl,
            "logSigmaHI_out": log_sigma_hi,
        }
    )

    input_csv = tmp_path / "input.csv"
    outdir = tmp_path / "out"
    df.to_csv(input_csv, index=False)

    results = run_analysis(str(input_csv), str(outdir))

    assert (outdir / "results.txt").exists()
    assert results["Delta_AICc"] < 0
    assert results["Delta_RMSE"] < 0
    assert abs(results["coef_HI"]) > 0.1


def test_run_analysis_fails_when_required_column_missing(tmp_path: Path):
    df = pd.DataFrame(
        {
            "delta_f3": [0.1, 0.2],
            "logMbar": [9.5, 10.1],
            "Rdisk": [1.1, 1.2],
            "inclination": [45.0, 50.0],
        }
    )
    input_csv = tmp_path / "missing.csv"
    df.to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="Falta columna: logSigmaHI_out"):
        run_analysis(str(input_csv), str(tmp_path / "out"))
