from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_comparison_csv(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G2", "G2", "G3", "G3"],
            "r_kpc": [1, 2, 1, 2, 1, 2],
            "g_bar": [1e-11, 2e-11, 1.5e-11, 2.2e-11, 1.2e-11, 1.9e-11],
            "g_obs": [1.1e-11, 2.3e-11, 1.55e-11, 2.35e-11, 1.25e-11, 2.0e-11],
            "log_g_bar": np.log10([1e-11, 2e-11, 1.5e-11, 2.2e-11, 1.2e-11, 1.9e-11]),
            "log_g_obs": np.log10([1.1e-11, 2.3e-11, 1.55e-11, 2.35e-11, 1.25e-11, 2.0e-11]),
        }
    )
    df.to_csv(path, index=False)
    return path


def _ensure_compare_csv(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "galaxy": ["G1", "G2", "G3"],
            "chi2_reduced": [1.1, 0.9, 1.2],
            "n_points": [20, 18, 22],
        }
    ).to_csv(path, index=False)
    return path


def _run(cmd: list[str]) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    finished = datetime.now(timezone.utc)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-20:]),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducibility checks for key framework scripts")
    parser.add_argument("--out", default="results/reproducibility/repro_report.json", help="Output report JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_csv = _ensure_comparison_csv(Path("results/reproducibility/universal_term_comparison_full.csv"))
    compare_csv = _ensure_compare_csv(Path("results/reproducibility/compare_nu_input.csv"))

    commands = [
        [sys.executable, "scripts/deep_slope_test.py", "--csv", str(comparison_csv), "--out", "results/reproducibility"],
        [sys.executable, "scripts/compare_nu_models.py", "--csv", str(compare_csv), "--out", "results/reproducibility"],
        [sys.executable, "scripts/fit_f3_linear_regression.py", "--input", "results/SPARC/sparc_175_master_sample.csv"],
    ]

    results = [_run(cmd) for cmd in commands]
    status = "ok" if all(r["returncode"] == 0 for r in results) else "failed"

    report = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Reproducibility report written to {out_path}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
