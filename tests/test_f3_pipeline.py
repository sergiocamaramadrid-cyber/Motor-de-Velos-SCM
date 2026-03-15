from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_writes_framework_summary(tmp_path: Path) -> None:
    summary = tmp_path / "framework_summary.json"
    generated_paths = [
        REPO_ROOT / "figures" / "f3_distribution.png",
        REPO_ROOT / "figures" / "f3_environment_scatter.png",
        REPO_ROOT / "figures" / "f3_vs_logMbar.png",
        REPO_ROOT / "figures" / "f3_vs_sigmaHI.png",
        REPO_ROOT / "results" / "regression" / "f3_regression_summary.csv",
        REPO_ROOT / "results" / "validation" / "sparc_catalog_validation_report.csv",
        REPO_ROOT / "results" / "reproducibility" / "universal_term_comparison_full.csv",
    ]
    try:
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
    finally:
        for path in generated_paths:
            if path.exists():
                path.unlink()
