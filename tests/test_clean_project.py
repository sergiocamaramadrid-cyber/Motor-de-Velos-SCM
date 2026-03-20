from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_clean_project_resets_results_and_removes_pycache(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "clean_project.py"

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "artifact.txt").write_text("temporary", encoding="utf-8")

    pycache_root = tmp_path / "__pycache__"
    pycache_root.mkdir()
    (pycache_root / "x.pyc").write_bytes(b"0")

    nested_pycache = tmp_path / "pkg" / "__pycache__"
    nested_pycache.mkdir(parents=True)
    (nested_pycache / "y.pyc").write_bytes(b"1")

    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)

    assert not results_dir.exists()
    assert not pycache_root.exists()
    assert not nested_pycache.exists()


def test_clean_project_full_removes_extra_targets(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "clean_project.py"

    sp = tmp_path / "results" / "SPARC"
    oos = tmp_path / "results" / "oos_validation"
    final = tmp_path / "results" / "scm_results_final"
    sp.mkdir(parents=True)
    oos.mkdir(parents=True)
    final.mkdir(parents=True)

    subprocess.run([sys.executable, str(script), "--full"], cwd=tmp_path, check=True)

    assert not sp.exists()
    assert not oos.exists()
    assert not final.exists()


def test_clean_project_does_not_remove_protected_paths(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "clean_project.py"

    protected_sparc = tmp_path / "data" / "SPARC"
    protected_sparc.mkdir(parents=True)
    (protected_sparc / "keep.txt").write_text("x", encoding="utf-8")

    protected_scripts = tmp_path / "scripts"
    protected_scripts.mkdir(parents=True)
    (protected_scripts / "keep.py").write_text("print('x')", encoding="utf-8")

    subprocess.run([sys.executable, str(script), "--full"], cwd=tmp_path, check=True)

    assert protected_sparc.exists()
    assert (protected_sparc / "keep.txt").exists()
    assert protected_scripts.exists()
    assert (protected_scripts / "keep.py").exists()


def test_clean_project_prints_messages_in_spanish(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "clean_project.py"
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Ejecutando limpieza estándar" in completed.stdout
    assert "Limpieza estándar completada" in completed.stdout
