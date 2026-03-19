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

    assert results_dir.exists()
    assert list(results_dir.iterdir()) == []
    assert not pycache_root.exists()
    assert not nested_pycache.exists()

