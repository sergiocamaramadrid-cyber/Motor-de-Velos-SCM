from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_sparc_master_alias_generates_full_catalog(tmp_path):
    sparc_dir = tmp_path / "SPARC"
    metadata_dir = sparc_dir / "metadata"
    rotmod_dir = sparc_dir / "rotmod"
    metadata_dir.mkdir(parents=True)
    rotmod_dir.mkdir(parents=True)

    for name in ("MassModels_Lelli2016c.mrt", "RAR.mrt", "RARbins.mrt", "CDR_Lelli2016b.mrt"):
        pd.DataFrame([{"Galaxy": "NGC2403"}]).to_csv(metadata_dir / name, index=False)

    pd.DataFrame(
        [
            {
                "Galaxy": "NGC2403",
                "L_3.6": 2.0,
                "MHI": 1.0,
                "RHI": 10.0,
                "Rdisk": 2.5,
                "Inc": 60.0,
            }
        ]
    ).to_csv(metadata_dir / "SPARC_Lelli2016c.csv", index=False)

    pd.DataFrame(
        [
            [1.0, 100.0, 2.0, 40.0, 60.0, 10.0],
            [2.0, 87.05505633, 2.0, 35.0, 55.0, 8.0],
            [3.0, 80.27202793, 2.0, 30.0, 50.0, 7.0],
            [4.0, 75.78582833, 2.0, 28.0, 48.0, 6.0],
            [5.0, 72.47857831, 2.0, 26.0, 46.0, 5.0],
            [6.0, 69.89300212, 2.0, 24.0, 44.0, 4.0],
        ]
    ).to_csv(
        rotmod_dir / "NGC2403_rotmod.dat", sep=" ", index=False, header=False
    )

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "build_sparc_master.py"),
        "--data-root",
        str(sparc_dir),
        "--out",
        str(tmp_path / "sparc_175_master.csv"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr or result.stdout
    out_csv = tmp_path / "sparc_175_master.csv"
    assert out_csv.exists()
    out_df = pd.read_csv(out_csv)
    assert {"galaxy", "F3", "logSigmaHI_out", "logMbar", "logRd"}.issubset(out_df.columns)
