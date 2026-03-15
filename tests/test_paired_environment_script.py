from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "test_paired_environment.py"


def _make_catalog(path: Path) -> None:
    rows = []
    for i in range(6):
        base_m = 9.5 + 0.1 * i
        base_r = 0.5 + 0.03 * i
        rows.append(
            {
                "galaxy": f"G{i:02d}A",
                "delta_f3": 0.1 + 0.02 * i,
                "F3": 0.8 + 0.02 * i,
                "logSigmaHI_out": 0.6 + 0.05 * i,
                "logMbar": base_m,
                "logRd": base_r,
                "fit_ok": True,
                "n_tail_points": 5,
                "inclination": 55.0,
                "quality_flag": "good",
            }
        )
        rows.append(
            {
                "galaxy": f"G{i:02d}B",
                "delta_f3": 0.06 + 0.02 * i,
                "F3": 0.76 + 0.02 * i,
                "logSigmaHI_out": 0.3 + 0.05 * i,
                "logMbar": base_m,
                "logRd": base_r,
                "fit_ok": True,
                "n_tail_points": 5,
                "inclination": 55.0,
                "quality_flag": "good",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_paired_environment_script_generates_expected_outputs(tmp_path: Path) -> None:
    input_csv = tmp_path / "sparc_175_master.csv"
    out_dir = tmp_path / "paired_environment"
    _make_catalog(input_csv)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--in",
        str(input_csv),
        "--out",
        str(out_dir),
        "--calipers",
        "0.5",
        "--radial-cuts",
        "0.7",
        "--main-radial-cut",
        "0.7",
        "--bootstrap-n",
        "20",
        "--placebo-n",
        "20",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    expected = [
        out_dir / "paired_sample.csv",
        out_dir / "paired_stats_summary.csv",
        out_dir / "paired_bootstrap.csv",
        out_dir / "placebo_tests.csv",
        out_dir / "delta_f3_vs_delta_logSigmaHI.png",
        out_dir / "run_metadata.json",
    ]
    for p in expected:
        assert p.exists(), f"Missing output file: {p}"

    sample = pd.read_csv(out_dir / "paired_sample.csv")
    assert not sample.empty
    assert {"delta_logSigma", "delta_delta_f3", "delta_F3"}.issubset(sample.columns)

    summary = pd.read_csv(out_dir / "paired_stats_summary.csv")
    assert not summary.empty
    assert "status" in summary.columns
    assert (summary["status"] == "ok").any()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["script"] == "scripts/test_paired_environment.py"
    assert metadata["best_pair_selection"]["radial_cut"] == 0.7
