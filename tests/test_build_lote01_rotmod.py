from __future__ import annotations

import runpy
from pathlib import Path


def test_build_lote01_rotmod_writes_expected_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        "/home/runner/work/Motor-de-Velos-SCM/Motor-de-Velos-SCM/scripts/build_lote01_rotmod.py",
        run_name="__main__",
    )

    rotmod_dir = tmp_path / "data" / "SPARC" / "rotmod"
    files = sorted(rotmod_dir.glob("*_rotmod.dat"))
    assert len(files) == 10

    sample = rotmod_dir / "NGC0024_rotmod.dat"
    lines = sample.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 7
    assert lines[0] == "1.14 55.4 5 12.1 48.2 0"

