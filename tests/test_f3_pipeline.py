from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_writes_framework_summary(tmp_path: Path) -> None:
    summary = tmp_path / "framework_summary.json"
    result = subprocess.run(
        [sys.executable, "scripts/run_full_framework_pipeline.py", "--summary", str(summary)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert "status" in payload
    assert "steps" in payload
    assert isinstance(payload["steps"], list)
