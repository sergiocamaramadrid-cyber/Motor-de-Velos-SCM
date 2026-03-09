from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "plot_beta_vs_environment.py"


def test_plot_beta_vs_environment_writes_figure(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "galaxy": ["G1", "G2", "G3", "G4"],
            "logSigmaHI_out": [-0.4, -0.2, 0.0, 0.2],
            "beta": [0.48, 0.50, 0.53, 0.55],
        }
    ).to_csv(results_dir / "per_galaxy_beta.csv", index=False)

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={"RESULTS_DIR": str(results_dir)},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (results_dir / "fig_beta_environment.png").exists()


def test_plot_beta_vs_environment_fails_with_missing_columns(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame({"beta": [0.5, 0.52, 0.54]}).to_csv(
        results_dir / "per_galaxy_beta.csv",
        index=False,
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={"RESULTS_DIR": str(results_dir)},
    )

    assert result.returncode != 0
    assert "Faltan columnas requeridas" in (result.stderr + result.stdout)
