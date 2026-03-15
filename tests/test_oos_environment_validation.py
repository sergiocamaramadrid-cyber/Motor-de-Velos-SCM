import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.oos_environment_validation import (
    check_columns,
    prepare_data,
    run_oos,
    save_outputs,
)


def _synthetic_catalog(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    log_mbar = rng.uniform(9.0, 11.0, n)
    log_rd = rng.uniform(0.2, 1.1, n)
    log_sigma = rng.uniform(0.0, 1.2, n)
    noise = rng.normal(0.0, 0.04, n)
    delta_f3 = 0.3 + 0.08 * log_mbar - 0.5 * log_rd + 0.7 * log_sigma + noise
    return pd.DataFrame(
        {
            "delta_f3": delta_f3,
            "logSigmaHI_out": log_sigma,
            "logMbar": log_mbar,
            "logRd": log_rd,
        }
    )


def test_oos_outputs_and_schema(tmp_path: Path):
    df = _synthetic_catalog()
    df.loc[0, "delta_f3"] = np.nan
    clean, n_initial, n_used, n_removed = prepare_data(df)
    assert n_removed == 1

    per_repeat = run_oos(clean, repeats=20, test_size=0.3, seed=42)
    outdir = tmp_path / "oos_environment"
    save_outputs(
        outdir=outdir,
        per_repeat=per_repeat,
        n_initial=n_initial,
        n_used=n_used,
        n_removed=n_removed,
        repeats=20,
        test_size=0.3,
        seed=42,
    )

    assert (outdir / "oos_repeats.csv").exists()
    assert (outdir / "oos_summary.csv").exists()
    assert (outdir / "oos_summary.json").exists()
    assert (outdir / "hist_delta_rmse_out.pdf").exists()

    summary_csv = pd.read_csv(outdir / "oos_summary.csv")
    assert {"RMSE_out_baseline_mean", "RMSE_out_full_mean", "delta_RMSE_out_mean", "delta_logL_out_mean"}.issubset(
        summary_csv.columns
    )
    summary_json = json.loads((outdir / "oos_summary.json").read_text(encoding="utf-8"))
    assert summary_json["n_removed_nan"] == 1
    assert summary_json["repeats_used"] > 0
    assert summary_json["delta_RMSE_out_mean"] < 0


def test_missing_columns_raise():
    bad = pd.DataFrame({"delta_f3": [0.1], "logMbar": [9.5], "logRd": [0.5]})
    try:
        check_columns(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised
