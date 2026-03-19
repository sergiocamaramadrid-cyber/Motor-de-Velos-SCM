from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "build_lote04_rotmod.py"

    spec = importlib.util.spec_from_file_location("build_lote04_rotmod", script_path)
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
        "NGC4559_rotmod.dat",
        "NGC4725_rotmod.dat",
        "NGC5005_rotmod.dat",
        "NGC5033_rotmod.dat",
        "NGC5371_rotmod.dat",
        "NGC5907_rotmod.dat",
        "NGC6195_rotmod.dat",
        "NGC6674_rotmod.dat",
        "NGC7331_rotmod.dat",
        "NGC7814_rotmod.dat",
    }
    assert {p.name for p in files} == expected_names


def test_written_file_has_expected_content_for_known_galaxy(tmp_path):
    module = _load_module()

    out_dir = tmp_path / "data" / "SPARC" / "rotmod"
    module.OUT_DIR = out_dir
    module.OUT_DIR.mkdir(parents=True, exist_ok=True)

    module.main()

    target = out_dir / "NGC4559_rotmod.dat"
    assert target.exists()

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(module.DATA["NGC4559"]) == 7
    assert lines[0] == "0.8 65.2 5 12.4 55.1 0"
    assert lines[-1] == "16.8 122.1 3 72.1 112.4 0"
