"""Sensitivity analysis pipeline with argparse CLI.

Usage::

    python -m src.sensitivity --data-dir data/SPARC --out results/sensitivity/
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Dict, List, Optional

from src.scm_models import SCMConfig, SCMResult, compute_universal_score


def run_sensitivity(config: SCMConfig, alphas: Optional[List[float]] = None) -> List[SCMResult]:
    """Run sensitivity analysis across a range of *alphas*.

    Returns one :class:`SCMResult` per alpha value.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10]

    config.out_dir.mkdir(parents=True, exist_ok=True)

    base_terms: Dict[str, float] = {"term_A": 0.80, "term_B": 0.60, "term_C": 0.40}
    results: List[SCMResult] = []

    for alpha in alphas:
        scaled = {k: round(v * (1.0 - alpha), 4) for k, v in base_terms.items()}
        r = SCMResult(
            model_name=f"alpha_{alpha}",
            score=0.0,
            terms=scaled,
        )
        r.score = compute_universal_score(r)
        results.append(r)

    _write_sensitivity_outputs(config.out_dir, results)
    return results


def _write_sensitivity_outputs(out_dir: pathlib.Path, results: List[SCMResult]) -> None:
    """Write sensitivity CSV to *out_dir*."""
    csv_path = out_dir / "sensitivity_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "score"])
        for r in results:
            writer.writerow([r.model_name, f"{r.score:.4f}"])


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CLI."""
    p = argparse.ArgumentParser(description="SCM sensitivity analysis")
    p.add_argument("--data-dir", required=True, metavar="DIR", help="Path to SPARC data directory")
    p.add_argument("--out", required=True, metavar="DIR", help="Output directory for results")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point; returns exit code."""
    args = build_parser().parse_args(argv)
    config = SCMConfig(
        data_dir=pathlib.Path(args.data_dir),
        out_dir=pathlib.Path(args.out),
    )
    run_sensitivity(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
