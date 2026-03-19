from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_lote01_rotmod.py"

    spec = importlib.util.spec_from_file_location("build_lote01_rotmod", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_expected_number_of_rotmod_files(tmp_path):
    module = _load_module()

    out_dir = tmp_path / "data" / "SPARC" / "rotmod"
    module.OUT_DIR = out_dir
    module.OUT_DIR.mkdir(parents=True, exist_ok=True)

    module.main()

    files = sorted(out_dir.glob("*_rotmod.dat"))
    assert len(files) == len(module.DATA) == 10

    expected_names = {
        "NGC0024_rotmod.dat",
        "NGC0055_rotmod.dat",
        "NGC0247_rotmod.dat",
        "NGC0300_rotmod.dat",
        "NGC0801_rotmod.dat",
        "NGC2841_rotmod.dat",
        "NGC2903_rotmod.dat",
        "NGC2976_rotmod.dat",
        "NGC3741_rotmod.dat",
        "NGC4013_rotmod.dat",
    }
    assert {p.name for p in files} == expected_names


def test_written_file_has_expected_content_for_known_galaxy(tmp_path):
    module = _load_module()

    out_dir = tmp_path / "data" / "SPARC" / "rotmod"
    module.OUT_DIR = out_dir
    module.OUT_DIR.mkdir(parents=True, exist_ok=True)

    module.main()

    target = out_dir / "NGC0024_rotmod.dat"
    assert target.exists()

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(module.DATA["NGC0024"]) == 7
    assert lines[0] == "1.14 55.4 5 12.1 48.2 0"
    assert lines[-1] == "10.26 114.5 2 49.2 106.3 0"
