from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_consolidar_sparc_v1_alias_generates_v1_catalog(tmp_path):
    sparc_dir = tmp_path / "SPARC"
    rotmod_dir = sparc_dir / "rotmod"
    rotmod_dir.mkdir(parents=True)
    out_csv = tmp_path / "results" / "rotation_curves-v1.0.csv"

    sample_rotation_data = [1.0, 100.0, 2.0, 30.0, 40.0, 50.0]  # r, v_obs, err, v_gas, v_disk, v_bulge
    pd.DataFrame([sample_rotation_data]).to_csv(
        rotmod_dir / "NGC2403_rotmod.dat", sep=" ", index=False, header=False
    )

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "consolidar_sparc_v1.py"),
        "--input",
        str(sparc_dir),
        "--out",
        str(out_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr or result.stdout
    assert out_csv.exists()
