from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from analisis_persistencia_delta_f3 import build_inter_galaxy_pairs, build_intra_galaxy_pairs


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analisis_persistencia_delta_f3.py"


def test_build_intra_galaxy_pairs_constructs_consecutive_delta_pairs() -> None:
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2"],
            "r_kpc": [1, 2, 3, 4, 1, 2, 3, 4],
            "F3": [1.0, 2.0, 4.0, 7.0, 1.0, 1.5, 2.5, 4.0],
        }
    )

    out = build_intra_galaxy_pairs(df, galaxy_col="galaxy", f3_col="F3", order_col="r_kpc")

    assert len(out) == 4
    g1 = out[out["galaxy"] == "G1"].reset_index(drop=True)
    assert g1["delta_f3_i"].tolist() == [1.0, 2.0]
    assert g1["delta_f3_j"].tolist() == [2.0, 3.0]
    assert g1["order_i"].tolist() == pytest.approx([1.5, 2.5])
    assert g1["order_j"].tolist() == pytest.approx([2.5, 3.5])
    g2 = out[out["galaxy"] == "G2"].reset_index(drop=True)
    assert g2["delta_f3_i"].tolist() == [0.5, 1.0]
    assert g2["delta_f3_j"].tolist() == [1.0, 1.5]
    assert g2["order_i"].tolist() == pytest.approx([1.5, 2.5])
    assert g2["order_j"].tolist() == pytest.approx([2.5, 3.5])


def test_build_inter_galaxy_pairs_uses_global_ordering() -> None:
    df = pd.DataFrame(
        {
            "galaxy": ["A", "B", "C", "D", "E"],
            "logMbar": [9.1, 8.8, 9.3, 9.0, 9.2],
            "F3": [1.3, 1.0, 1.9, 1.5, 1.7],
        }
    )

    out = build_inter_galaxy_pairs(df, galaxy_col="galaxy", f3_col="F3", order_col="logMbar")

    assert len(out) == 3
    assert {"delta_f3_i", "delta_f3_j", "order_i", "order_j"}.issubset(out.columns)
    assert out["order_i"].tolist() == pytest.approx([8.9, 9.05, 9.15])
    assert out["order_j"].tolist() == pytest.approx([9.05, 9.15, 9.25])
    assert out["delta_f3_i"].tolist() == pytest.approx([0.5, -0.2, 0.4])
    assert out["delta_f3_j"].tolist() == pytest.approx([-0.2, 0.4, 0.2])


def test_build_inter_galaxy_pairs_accepts_precomputed_delta_column_without_zero_fill() -> None:
    df = pd.DataFrame(
        {
            "galaxy": ["A", "B", "C", "D", "E"],
            "logMbar": [8.8, 9.0, 9.1, 9.2, 9.3],
            # First value undefined (as diff output), plus a real zero to filter
            "delta_f3": [float("nan"), 0.5, 0.0, 0.4, 0.2],
            "F3": [1.0, 1.5, 1.5, 1.9, 2.1],
        }
    )

    out = build_inter_galaxy_pairs(
        df.dropna(subset=["delta_f3"]),
        galaxy_col="galaxy",
        f3_col="F3",
        order_col="logMbar",
        delta_col="delta_f3",
    )

    assert len(out) == 1
    assert out["delta_f3_i"].tolist() == pytest.approx([0.4])
    assert out["delta_f3_j"].tolist() == pytest.approx([0.2])


def test_cli_generates_pairs_models_summary_and_figure(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"

    rows = []
    for galaxy, offset in [("G1", 0.0), ("G2", 0.5)]:
        for r, f3 in [(1, 1.0 + offset), (2, 1.4 + offset), (3, 2.0 + offset), (4, 2.9 + offset), (5, 4.1 + offset)]:
            rows.append(
                {
                    "galaxy": galaxy,
                    "r_kpc": r,
                    "logMbar": 9.0 + 0.1 * r,
                    "F3": f3,
                    "fit_ok": True,
                    "reliable": True,
                }
            )
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_csv),
        "--mode",
        "intra-galaxy",
        "--filter-fit-ok",
        "--filter-reliable",
        "--outdir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    pairs_csv = out_dir / "delta_f3_pairs.csv"
    models_csv = out_dir / "delta_f3_model_comparison.csv"
    summary_txt = out_dir / "delta_f3_summary.txt"
    fig_png = out_dir / "delta_f3_persistence_fit.png"
    boot_csv = out_dir / "delta_f3_bootstrap_quadratic.csv"

    for path in [pairs_csv, models_csv, summary_txt, fig_png, boot_csv]:
        assert path.exists(), f"Missing output: {path}"

    models = pd.read_csv(models_csv)
    assert set(models["model"]) == {"nulo", "linear", "quadratic"}
    assert models["aicc"].is_monotonic_increasing
    assert models.iloc[0]["aicc"] == pytest.approx(models["aicc"].min())


def test_cli_inter_galaxy_requires_delta_f3_column(tmp_path: Path) -> None:
    input_csv = tmp_path / "input_missing_delta.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "galaxy": ["A", "B", "C", "D"],
            "logMbar": [9.0, 9.1, 9.2, 9.3],
            "F3": [1.0, 1.2, 1.4, 1.7],
        }
    ).to_csv(input_csv, index=False)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_csv),
        "--mode",
        "inter-galaxy",
        "--outdir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode != 0
    assert "delta_col (inter-galaxy)" in (result.stdout + result.stderr)


def test_cli_inter_galaxy_prints_expected_model_and_bootstrap_blocks(tmp_path: Path) -> None:
    input_csv = tmp_path / "input_inter.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "galaxy": ["A", "B", "C", "D", "E", "F"],
            "logMbar": [8.9, 9.0, 9.1, 9.2, 9.3, 9.4],
            "F3": [1.0, 1.3, 1.5, 1.9, 2.2, 2.5],
            "delta_f3": [0.1, 0.3, 0.2, 0.4, 0.3, 0.2],
        }
    ).to_csv(input_csv, index=False)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(input_csv),
        "--mode",
        "inter-galaxy",
        "--outdir",
        str(out_dir),
        "--n-boot",
        "50",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    console = result.stdout + result.stderr
    assert "--- COMPARACIÓN DE MODELOS (AICc) ---" in console
    assert "Mejor modelo" in console
    assert "ΔAICc cuadrático-nulo" in console
    assert "--- BOOTSTRAP 95% ---" in console
    assert "b:" in console
