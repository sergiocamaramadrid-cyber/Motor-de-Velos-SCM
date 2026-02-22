"""Tests for src.scm_analysis."""
import pathlib

from src.scm_analysis import main, run_analysis
from src.scm_models import SCMConfig


def test_run_analysis_creates_outputs(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "SPARC"
    data_dir.mkdir()
    out_dir = tmp_path / "results"
    cfg = SCMConfig(data_dir=data_dir, out_dir=out_dir)
    results = run_analysis(cfg)
    assert len(results) == 1
    assert (out_dir / "universal_term_comparison_full.csv").exists()
    assert (out_dir / "executive_summary.txt").exists()
    assert (out_dir / "top10_universal.tex").exists()


def test_main_returns_zero(tmp_path: pathlib.Path) -> None:
    data_dir = tmp_path / "SPARC"
    data_dir.mkdir()
    out_dir = tmp_path / "results"
    rc = main(["--data-dir", str(data_dir), "--out", str(out_dir)])
    assert rc == 0
