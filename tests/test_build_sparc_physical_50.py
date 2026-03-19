from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_build_sparc_physical_50_generates_expected_catalog(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_sparc_physical_50.py"
    )

    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)

    out_dir = tmp_path / "data" / "SPARC" / "rotmod"
    files = sorted(out_dir.glob("G_PHYSICAL_*_rotmod.dat"))
    assert len(files) == 50

    sample = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(sample) == 16
    assert all(len(line.split()) == 6 for line in sample)
