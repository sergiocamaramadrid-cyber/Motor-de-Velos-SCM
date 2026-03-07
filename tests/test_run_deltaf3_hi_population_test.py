from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.run_deltaf3_hi_population_test import run_analysis


def test_run_analysis_writes_outputs_and_keeps_positive_hi_term(tmp_path):
    table = tmp_path / "big_sparc_f3_hi.csv"
    out_dir = tmp_path / "analysis"

    rows = []
    for i in range(30):
        x_hi = -0.30 + 0.02 * i
        logm = 9.8 + 0.01 * i
        rd = 1.5 + 0.03 * i
        inc = 45.0 + 0.8 * i
        delta = (
            0.20 * x_hi
            + 0.03 * logm
            - 0.01 * rd
            + 0.001 * inc
            + 0.002 * np.sin(i)
        )
        rows.append(
            {
                "galaxy": f"G{i:03d}",
                "delta_f3": delta,
                "deep_slope": 0.5 + delta,
                "n_tail_points": 5,
                "tail_r_min": 2.0,
                "tail_r_max": 10.0,
                "logSigmaHI_out": x_hi,
                "logMstar": logm,
                "Rdisk": rd,
                "inclination": inc,
            }
        )
    pd.DataFrame(rows).to_csv(table, index=False)

    summary = run_analysis(table_path=table, out_dir=out_dir, seed=7)

    assert (out_dir / "deltaf3_hi_scatter.png").exists()
    summary_json = out_dir / "deltaf3_hi_population_summary.json"
    assert summary_json.exists()
    loaded = json.loads(summary_json.read_text(encoding="utf-8"))
    assert loaded["n_rows"] == 30
    assert loaded["correlation"]["n_samples"] == 30

    assert summary["regression"]["features"] == [
        "logSigmaHI_out",
        "logMstar",
        "Rdisk",
        "inclination",
    ]
    assert summary["regression"]["coefficients"]["logSigmaHI_out"] > 0.0
    assert summary["oos_70_30"]["coef_logSigmaHI_out"] > 0.0


def test_run_analysis_falls_back_to_simple_regression_when_controls_missing(tmp_path):
    table = tmp_path / "big_sparc_f3_hi_min.csv"
    out_dir = tmp_path / "analysis_min"

    x = np.linspace(-0.2, 0.3, 12)
    delta = 0.4 * x - 0.01
    pd.DataFrame(
        {
            "galaxy": [f"G{i:02d}" for i in range(len(x))],
            "delta_f3": delta,
            "deep_slope": 0.5 + delta,
            "n_tail_points": 5,
            "tail_r_min": 1.0,
            "tail_r_max": 6.0,
            "logSigmaHI_out": x,
        }
    ).to_csv(table, index=False)

    summary = run_analysis(table_path=table, out_dir=out_dir, seed=1)

    assert summary["regression"]["features"] == ["logSigmaHI_out"]
    assert np.isfinite(summary["correlation"]["pearson_r"])
    assert summary["oos_70_30"]["test_n"] > 0
