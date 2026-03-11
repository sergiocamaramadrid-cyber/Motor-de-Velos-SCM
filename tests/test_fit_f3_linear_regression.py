from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fit_f3_linear_regression import fit_f3_model


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fit_f3_linear_regression.py"


def _make_master(path: Path) -> None:
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    x3 = np.array([1.5, 0.5, 3.0, 1.0, 2.5])
    y = 1.5 * x1 - 0.75 * x2 + 2.0 * x3 + 4.0

    df = pd.DataFrame(
        {
            "logSigmaHI_out": x1,
            "logMbar": x2,
            "logRd": x3,
            "F3": y,
        }
    )
    df.to_csv(path, index=False)


def test_fit_f3_model_recovers_coefficients(tmp_path: Path) -> None:
    csv_path = tmp_path / "sparc_175_master.csv"
    _make_master(csv_path)

    model = fit_f3_model(csv_path)

    assert np.allclose(model.coef_, np.array([1.5, -0.75, 2.0]), atol=1e-9)
    assert np.isclose(model.intercept_, 4.0, atol=1e-9)


def test_cli_prints_coefficients_and_intercept(tmp_path: Path) -> None:
    csv_path = tmp_path / "sparc_175_master.csv"
    _make_master(csv_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(csv_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "coeficientes:" in result.stdout
    assert "intercepto:" in result.stdout


def test_cli_uses_default_input_name_in_cwd(tmp_path: Path) -> None:
    csv_path = tmp_path / "sparc_175_master.csv"
    _make_master(csv_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "coeficientes:" in result.stdout
    assert "intercepto:" in result.stdout
