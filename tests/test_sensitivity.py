"""Tests for src.sensitivity."""
import pathlib

from src.sensitivity import main, run_sensitivity
from src.scm_models import SCMConfig


def test_run_sensitivity_default_alphas(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "SPARC"
    data_dir.mkdir()
    out_dir = tmp_path / "sens"
    cfg = SCMConfig(data_dir=data_dir, out_dir=out_dir)
    results = run_sensitivity(cfg)
    assert len(results) == 3
    assert all(r.score > 0 for r in results)
    assert (out_dir / "sensitivity_results.csv").exists()


def test_sensitivity_main_returns_zero(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "SPARC"
    data_dir.mkdir()
    out_dir = tmp_path / "sens"
    rc = main(["--data-dir", str(data_dir), "--out", str(out_dir)])
    assert rc == 0
