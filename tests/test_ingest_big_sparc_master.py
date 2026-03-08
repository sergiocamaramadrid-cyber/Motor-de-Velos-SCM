from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ingest_big_sparc_contract.py"


def _write_rotmod(path: Path, base_v: float) -> None:
    path.write_text(
        "\n".join(
            [
                "# r vobs ev gas disk bul",
                f"1.0 {base_v:.1f} 2.0 20.0 70.0 0.0",
                f"2.0 {base_v+2:.1f} 2.0 19.0 72.0 0.0",
                f"3.0 {base_v+3:.1f} 2.0 18.0 73.0 0.0",
                f"4.0 {base_v+3.5:.1f} 2.0 17.0 73.5 0.0",
                f"5.0 {base_v+4:.1f} 2.0 16.0 74.0 0.0",
            ]
        ),
        encoding="utf-8",
    )


def test_cli_builds_master_and_sanity_csv(tmp_path: Path) -> None:
    data_root = tmp_path / "SPARC"
    rotmod = data_root / "rotmod"
    rotmod.mkdir(parents=True)
    _write_rotmod(rotmod / "GAL_A.dat", 100.0)
    _write_rotmod(rotmod / "GAL_B.dat", 120.0)

    out_csv = tmp_path / "results" / "sparc_175_master.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-root",
            str(data_root),
            "--out",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert out_csv.exists()

    sanity_csv = out_csv.with_name("sparc_175_master_sanity.csv")
    assert sanity_csv.exists()

    master = pd.read_csv(out_csv)
    assert set(["galaxy", "F3_SCM", "delta_f3", "beta", "logSigmaHI_out_proxy"]).issubset(master.columns)
    assert len(master) == 2


def test_cli_fails_without_rotmod_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "SPARC"
    data_root.mkdir()

    out_csv = tmp_path / "results" / "sparc_175_master.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-root",
            str(data_root),
            "--out",
            str(out_csv),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode != 0
    assert "No existe directorio de curvas" in (result.stderr + result.stdout)
