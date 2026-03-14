from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "run_f3_pipeline.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )


def test_unknown_arg_fails_fast() -> None:
    cp = _run("--does-not-exist")
    assert cp.returncode == 1
    assert "Argumento desconocido" in cp.stdout


def test_missing_input_fails() -> None:
    cp = _run("--input", str(REPO_ROOT / "no_such_input.parquet"))
    assert cp.returncode == 1
    assert "No se encuentra el archivo de entrada" in cp.stdout


def test_generates_and_validates_catalog(tmp_path: Path) -> None:
    input_csv = tmp_path / "contract.csv"
    out_dir = tmp_path / "out"

    df = pd.DataFrame(
        [
            {"galaxy": "G1", "r_kpc": 1.0, "vobs_kms": 31.0, "vobs_err_kms": 2.0, "vbar_kms": 10.0},
            {"galaxy": "G1", "r_kpc": 2.0, "vobs_kms": 37.0, "vobs_err_kms": 2.0, "vbar_kms": 20.0},
            {"galaxy": "G1", "r_kpc": 3.0, "vobs_kms": 43.0, "vobs_err_kms": 2.0, "vbar_kms": 30.0},
            {"galaxy": "G1", "r_kpc": 4.0, "vobs_kms": 49.0, "vobs_err_kms": 2.0, "vbar_kms": 40.0},
            {"galaxy": "G2", "r_kpc": 1.0, "vobs_kms": 15.0, "vobs_err_kms": 2.0, "vbar_kms": 10.0},
            {"galaxy": "G2", "r_kpc": 2.0, "vobs_kms": 24.0, "vobs_err_kms": 2.0, "vbar_kms": 20.0},
            {"galaxy": "G2", "r_kpc": 3.0, "vobs_kms": 32.0, "vobs_err_kms": 2.0, "vbar_kms": 30.0},
            {"galaxy": "G2", "r_kpc": 4.0, "vobs_kms": 39.0, "vobs_err_kms": 2.0, "vbar_kms": 40.0},
        ]
    )
    df.to_csv(input_csv, index=False)

    cp = _run("--input", str(input_csv), "--out", str(out_dir))

    assert cp.returncode == 0, cp.stdout + "\n" + cp.stderr
    out_csv = out_dir / "f3_catalog.csv"
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    canonical = {"f3_scm", "delta_f3", "fit_ok", "quality_flag"}
    legacy = {"beta", "beta_err", "reliable", "friction_slope", "velo_inerte_flag"}
    assert canonical.issubset(out_df.columns)
    assert legacy.issubset(out_df.columns)
    assert "[VALIDACIÓN SCM]" in cp.stdout
    assert "min=" in cp.stdout and "max=" in cp.stdout and "std=" in cp.stdout
