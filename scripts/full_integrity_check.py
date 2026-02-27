"""
scripts/full_integrity_check.py — Full integrity check for the SCM Framework.

Runs a comprehensive integrity audit of the Motor de Velos SCM v0.6.1
framework and writes the results to:

    reports/integrity_report_v0.6.1.json

Usage
-----
From the repository root::

    python scripts/full_integrity_check.py

The exit code is 0 if the integrity check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERSION = "0.6.1"
_M81_GROUP = ["M81", "M82", "NGC2403", "NGC3077", "NGC2976", "IC2574"]
_XI_MIN = 1.34
_XI_MAX = 1.48


def run_full_integrity_check(repo_root: Path | None = None) -> dict:
    """Execute all integrity checks and return a structured report dict.

    Parameters
    ----------
    repo_root : Path, optional
        Root of the repository.  Defaults to the parent of this script's
        directory (i.e., the project root).

    Returns
    -------
    dict
        A report dictionary suitable for serialisation to JSON.
    """
    root = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent

    audits_dir = root / "audits" / "validated"
    report_csv = root / "reports" / "batch_results_m81_group.csv"
    config_yaml = root / "config" / "defaults.yaml"

    report: dict = {
        "framework_version": _VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "summary": {},
    }

    all_passed = True

    # ------------------------------------------------------------------
    # 1. Duplicate files in audits/validated
    # ------------------------------------------------------------------
    dup_result: dict = {"status": "OK", "duplicates": []}
    if audits_dir.exists():
        seen: dict[str, list[str]] = {}
        for f in sorted(audits_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                key = data.get("galaxy", f.stem)
            except (json.JSONDecodeError, OSError):
                key = f.stem
            seen.setdefault(key, []).append(f.name)
        duplicates = {k: v for k, v in seen.items() if len(v) > 1}
        if duplicates:
            dup_result["status"] = "FAIL"
            dup_result["duplicates"] = [
                {"galaxy": k, "files": v} for k, v in duplicates.items()
            ]
            all_passed = False
    else:
        dup_result["status"] = "FAIL"
        dup_result["error"] = "audits/validated directory not found"
        all_passed = False

    report["checks"]["duplicate_files"] = dup_result

    # ------------------------------------------------------------------
    # 2. Coherence between audits and reports
    # ------------------------------------------------------------------
    coherence_result: dict = {"status": "OK", "missing_json": [], "missing_csv": []}

    # Check each expected JSON exists and has a valid xi
    xi_values: dict[str, float] = {}
    for galaxy in _M81_GROUP:
        json_path = audits_dir / f"{galaxy}_v{_VERSION}.json"
        if not json_path.exists():
            coherence_result["missing_json"].append(f"{galaxy}_v{_VERSION}.json")
            coherence_result["status"] = "FAIL"
            all_passed = False
        else:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                xi = float(data.get("xi", float("nan")))
                xi_values[galaxy] = xi
            except (json.JSONDecodeError, ValueError):
                coherence_result["status"] = "FAIL"
                coherence_result.setdefault("invalid_json", []).append(f"{galaxy}_v{_VERSION}.json")
                all_passed = False

    # Check report CSV
    if not report_csv.exists():
        coherence_result["status"] = "FAIL"
        coherence_result["missing_csv"].append("batch_results_m81_group.csv")
        all_passed = False
    else:
        try:
            df = pd.read_csv(report_csv)
            csv_galaxies = df["galaxy"].tolist() if "galaxy" in df.columns else []
            for g in _M81_GROUP:
                if g not in csv_galaxies:
                    coherence_result["missing_csv"].append(g)
                    coherence_result["status"] = "FAIL"
                    all_passed = False
        except Exception as exc:  # noqa: BLE001
            coherence_result["status"] = "FAIL"
            coherence_result["csv_error"] = str(exc)
            all_passed = False

    report["checks"]["audit_report_coherence"] = coherence_result

    # ------------------------------------------------------------------
    # 3. ξ consistency
    # ------------------------------------------------------------------
    xi_result: dict = {"status": "OK", "xi_values": xi_values, "out_of_range": []}
    for galaxy, xi in xi_values.items():
        if not (_XI_MIN <= xi <= _XI_MAX):
            xi_result["out_of_range"].append(
                {"galaxy": galaxy, "xi": xi, "expected_range": [_XI_MIN, _XI_MAX]}
            )
            xi_result["status"] = "FAIL"
            all_passed = False

    report["checks"]["xi_consistency"] = xi_result

    # ------------------------------------------------------------------
    # 4. config/defaults.yaml calibration
    # ------------------------------------------------------------------
    calib_result: dict = {"status": "OK"}
    if not config_yaml.exists():
        calib_result["status"] = "FAIL"
        calib_result["error"] = "config/defaults.yaml not found"
        all_passed = False
    else:
        try:
            text = config_yaml.read_text(encoding="utf-8")
            cfg = yaml.safe_load(text)
            xi_global_count = sum(
                1 for line in text.splitlines()
                if line.strip().startswith("xi_global")
            )
            calib_result["xi_global"] = cfg.get("xi_global")
            calib_result["xi_global_occurrences"] = xi_global_count
            if xi_global_count != 1:
                calib_result["status"] = "FAIL"
                calib_result["error"] = (
                    f"xi_global appears {xi_global_count} times; expected exactly 1"
                )
                all_passed = False
            elif cfg.get("xi_global") is None:
                calib_result["status"] = "FAIL"
                calib_result["error"] = "xi_global key missing from parsed YAML"
                all_passed = False
        except Exception as exc:  # noqa: BLE001
            calib_result["status"] = "FAIL"
            calib_result["error"] = str(exc)
            all_passed = False

    report["checks"]["calibration"] = calib_result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    report["summary"] = {
        "integrity_check": "PASSED" if all_passed else "FAILED",
        "duplicate_files": "NONE" if not dup_result["duplicates"] else "FOUND",
        "missing_files": (
            "NONE"
            if not coherence_result["missing_json"] and not coherence_result["missing_csv"]
            else "FOUND"
        ),
        "calibration_conflicts": (
            "NONE" if calib_result["status"] == "OK" else "FOUND"
        ),
        "framework_status": "CONSISTENT" if all_passed else "INCONSISTENT",
    }

    return report


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    report = run_full_integrity_check(root)

    # Write JSON report
    out_path = root / "reports" / "integrity_report_v0.6.1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Print summary
    summary = report["summary"]
    print(f"Integrity check: {summary['integrity_check']}")
    print(f"Duplicate files: {summary['duplicate_files']}")
    print(f"Missing files: {summary['missing_files']}")
    print(f"Calibration conflicts: {summary['calibration_conflicts']}")
    print(f"Framework status: {summary['framework_status']}")
    print(f"\nFull report written to: {out_path}")

    sys.exit(0 if summary["integrity_check"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
