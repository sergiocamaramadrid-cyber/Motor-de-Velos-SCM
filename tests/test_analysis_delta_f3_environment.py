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
    log_sigma_hi = rng.uniform(-1.0, 1.0, n)

    noise = rng.normal(0.0, 0.03, n)
    delta_f3 = 0.4 + 0.09 * log_mbar + 0.75 * log_sigma_hi + noise

    df = pd.DataFrame(
        {
            "delta_f3": delta_f3,
            "logMbar": log_mbar,
            "logSigmaHI_out": log_sigma_hi,
        }
    )

    input_csv = tmp_path / "input.csv"
    outdir = tmp_path / "out"
    df.to_csv(input_csv, index=False)

    results = run_analysis(str(input_csv), str(outdir))

    assert (outdir / "results.txt").exists()
    assert results["delta_aicc"] < 0
    assert results["delta_rmse"] < 0
    assert results["coef_hi"] > 0


def test_run_analysis_builds_delta_from_f3_when_missing(tmp_path: Path):
    rng = np.random.default_rng(5)
    n = 120
    mbar_linear = rng.uniform(9.0, 11.0, n)
    sigma_hi_linear = rng.uniform(-0.8, 0.8, n)
    noise = rng.normal(0.0, 0.02, n)
    f3 = 0.2 + 0.08 * mbar_linear + 0.65 * sigma_hi_linear + noise

    df = pd.DataFrame(
        {
            "Mbar": mbar_linear,
            "SigmaHI_out": sigma_hi_linear,
            "F3": f3,
        }
    )
    input_csv = tmp_path / "sample.csv"
    outdir = tmp_path / "out"
    df.to_csv(input_csv, index=False)

    results = run_analysis(str(input_csv), str(outdir))
    assert (outdir / "results.txt").exists()
    assert np.isfinite(results["delta_aicc"])


def test_run_analysis_fails_when_required_column_missing(tmp_path: Path):
    df = pd.DataFrame(
        {
            "delta_f3": [0.1, 0.2],
            "logMbar": [9.5, 10.1],
        }
    )
    input_csv = tmp_path / "missing.csv"
    df.to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="No valid column found for HI"):
        run_analysis(str(input_csv), str(tmp_path / "out"))
