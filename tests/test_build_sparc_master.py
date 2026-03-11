from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_sparc_master_alias_generates_full_catalog(tmp_path):
    sparc_dir = tmp_path / "SPARC"
    rotmod_dir = sparc_dir / "rotmod"
    rotmod_dir.mkdir(parents=True)
    out_csv = tmp_path / "results" / "sparc_full.csv"

    pd.DataFrame(
        [
            {
                "Galaxy": "NGC2403",
                "L_3.6": 2.0,
                "MHI": 1.0,
                "RHI": 10.0,
            }
        ]
    ).to_csv(sparc_dir / "SPARC_Lelli2016c.csv", index=False)

    pd.DataFrame([[1.0, 100.0, 2.0, 40.0, 60.0, 10.0]]).to_csv(
        rotmod_dir / "NGC2403_rotmod.dat", sep=" ", index=False, header=False
    )

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "build_sparc_master.py"),
        "--data-root",
        str(sparc_dir),
        "--out",
        str(out_csv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr or result.stdout
    assert out_csv.exists()
