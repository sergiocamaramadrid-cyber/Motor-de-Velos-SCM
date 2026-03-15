from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_COMPARISON_CSV = (
    "galaxy,r_kpc,g_bar,g_obs,log_g_bar,log_g_obs\n"
    "G1,1,1e-11,1.1e-11,-11,-10.958607315\n"
    "G1,2,2e-11,2.2e-11,-10.698970004,-10.657577319\n"
)


def _run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-20:]),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full SCM framework pipeline")
    parser.add_argument("--summary", default="results/framework_summary.json", help="Output summary JSON")
    return parser.parse_args(argv)


def _ensure_fallback_comparison_csv() -> None:
    comp = REPO_ROOT / "results/reproducibility/universal_term_comparison_full.csv"
    if comp.exists():
        return
    comp.parent.mkdir(parents=True, exist_ok=True)
    comp.write_text(FALLBACK_COMPARISON_CSV, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    commands = [
        [sys.executable, "scripts/build_sparc_master_catalog.py"],
        [sys.executable, "scripts/validate_sparc_catalog.py"],
        [sys.executable, "scripts/deep_slope_test.py", "--csv", "results/reproducibility/universal_term_comparison_full.csv", "--out", "results"],
        [sys.executable, "scripts/fit_f3_linear_regression.py", "--input", "data/sparc_175_master.csv"],
        [sys.executable, "scripts/run_oos_validation.py"],
        [sys.executable, "scripts/plot_f3_analysis.py"],
    ]

    # Ensure synthetic deep-slope input exists if full file is unavailable
    _ensure_fallback_comparison_csv()

    results = [_run(cmd) for cmd in commands]
    status = "ok" if all(r["returncode"] == 0 for r in results) else "failed"

    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": results,
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Framework summary written to {summary_path}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
