from __future__ import annotations

import runpy
from pathlib import Path


def test_build_lote03_rotmod_writes_expected_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_lote03_rotmod.py"

    runpy.run_path(
        str(script_path),
        run_name="__main__",
    )

    rotmod_dir = tmp_path / "data" / "SPARC" / "rotmod"
    files = sorted(rotmod_dir.glob("*_rotmod.dat"))
    assert len(files) == 10

    sample = rotmod_dir / "NGC3198_rotmod.dat"
    lines = sample.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    assert lines[0] == "0.92 55 5 11.5 45.2 0"
