"""Tests for src.scm_models."""
import pathlib

import pytest

from src.scm_models import SCMConfig, SCMResult, compute_universal_score


def test_scm_config_defaults() -> None:
    cfg = SCMConfig(data_dir=pathlib.Path("data/SPARC"), out_dir=pathlib.Path("results"))
    assert cfg.alpha == 0.05
    assert cfg.variables == []


def test_scm_result_top_terms_order() -> None:
    r = SCMResult("m", 0.0, {"a": 0.1, "b": 0.9, "c": 0.5})
    assert r.top_terms(2) == ["b", "c"]


def test_scm_result_top_terms_empty() -> None:
    r = SCMResult("m", 0.0)
    assert r.top_terms() == []


def test_compute_universal_score_empty() -> None:
    r = SCMResult("m", 0.0)
    assert compute_universal_score(r) == pytest.approx(0.0)


def test_compute_universal_score_values() -> None:
    r = SCMResult("m", 0.0, {"x": 0.8, "y": 0.4})
    assert compute_universal_score(r) == pytest.approx(0.6)
