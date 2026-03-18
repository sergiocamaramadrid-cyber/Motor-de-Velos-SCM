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
            {"galaxy": "G1", "r": 1.0, "gobs": 31.0, "gbar": 10.0},
            {"galaxy": "G1", "r": 2.0, "gobs": 37.0, "gbar": 20.0},
            {"galaxy": "G1", "r": 3.0, "gobs": 43.0, "gbar": 30.0},
            {"galaxy": "G1", "r": 4.0, "gobs": 49.0, "gbar": 40.0},
            {"galaxy": "G2", "r": 1.0, "gobs": 15.0, "gbar": 10.0},
            {"galaxy": "G2", "r": 2.0, "gobs": 24.0, "gbar": 20.0},
            {"galaxy": "G2", "r": 3.0, "gobs": 32.0, "gbar": 30.0},
            {"galaxy": "G2", "r": 4.0, "gobs": 39.0, "gbar": 40.0},
        ]
    )
    df.to_csv(input_csv, index=False)

    cp = _run("--input", str(input_csv), "--out", str(out_dir))

    assert cp.returncode == 0, cp.stdout + "\n" + cp.stderr
    out_csv = out_dir / "f3_catalog.csv"
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert {"F3", "delta_f3"}.issubset(out_df.columns)
    assert "[VALIDACIÓN SCM]" in cp.stdout
    assert "min=" in cp.stdout and "max=" in cp.stdout and "std=" in cp.stdout
