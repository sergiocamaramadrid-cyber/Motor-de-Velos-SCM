from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_build_sparc_synthetic_175_physical_generates_expected_catalog(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_sparc_synthetic_175_physical.py"
    )

    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)

    out_dir = tmp_path / "data" / "SPARC_synthetic" / "rotmod"
    files = sorted(out_dir.glob("G_SYNTH_*_rotmod.dat"))
    assert len(files) == 175

    sample = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(sample) == 12
    assert all(len(line.split()) == 6 for line in sample)
