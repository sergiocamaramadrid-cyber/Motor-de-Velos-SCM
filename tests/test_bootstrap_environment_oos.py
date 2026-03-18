from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.bootstrap_environment_oos import run_bootstrap


def _synthetic_df(n: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_mbar = rng.uniform(9.0, 11.0, n)
    log_sigma_hi = rng.uniform(-1.0, 1.0, n)
    noise = rng.normal(0.0, 0.04, n)
    delta_f3 = 0.15 + 0.08 * log_mbar + 0.7 * log_sigma_hi + noise
    return pd.DataFrame(
        {
            "delta_f3": delta_f3,
            "logMbar": log_mbar,
            "logSigmaHI_out": log_sigma_hi,
        }
    )


def test_run_bootstrap_writes_outputs_and_detects_signal(tmp_path: Path):
    df = _synthetic_df()
    input_csv = tmp_path / "sparc_175_master.csv"
    outdir = tmp_path / "results"
    df.to_csv(input_csv, index=False)

    metrics = run_bootstrap(
        input_csv=str(input_csv),
        outdir=str(outdir),
        n_boot=80,
        test_size=0.3,
        seed=42,
    )

    assert (outdir / "resultados_bootstrap.txt").exists()
    assert (outdir / "bootstrap_HI.png").exists()
    assert metrics["delta_rmse_mean"] < 0
    assert metrics["prop_improve"] > 0.5
    assert abs(metrics["coef_hi_mean"]) > 0.05


def test_run_bootstrap_derives_delta_f3_from_f3_when_missing(tmp_path: Path):
    rng = np.random.default_rng(11)
    n = 120
    log_mbar = np.linspace(9.0, 10.8, n)
    log_sigma_hi = rng.uniform(-0.5, 0.5, n)
    f3 = 0.3 * np.arange(n) + 0.5 * log_sigma_hi
    df = pd.DataFrame(
        {
            "F3": f3,
            "logMbar": log_mbar,
            "logSigmaHI_out": log_sigma_hi,
        }
    )

    input_csv = tmp_path / "sparc_175_master.csv"
    outdir = tmp_path / "results"
    df.to_csv(input_csv, index=False)

    metrics = run_bootstrap(
        input_csv=str(input_csv),
        outdir=str(outdir),
        n_boot=20,
        test_size=0.3,
        seed=4,
    )

    assert (outdir / "resultados_bootstrap.txt").exists()
    assert np.isfinite(metrics["p_empirical"])


def test_run_bootstrap_requires_target_or_f3(tmp_path: Path):
    df = pd.DataFrame(
        {
            "logMbar": [10.0, 10.1],
            "logSigmaHI_out": [0.2, 0.3],
        }
    )
    input_csv = tmp_path / "sparc_175_master.csv"
    df.to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="requiere 'delta_f3' o, alternativamente, 'F3'"):
        run_bootstrap(input_csv=str(input_csv), outdir=str(tmp_path / "results"), n_boot=5)
