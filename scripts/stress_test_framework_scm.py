#!/usr/bin/env python3
"""
Stress Test Framework SCM

Defines the observational domain of applicability for the SCM framework and
exports a reproducible stress-test table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STATUS_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
STATUS_FUTURE_EXTENSION = "FUTURE_EXTENSION_CANDIDATE"
STATUS_FRAMEWORK_READY = "FRAMEWORK_READY"
MAX_REDSHIFT_FOR_READY = 1.0
MIN_QUALITY_FOR_READY = 0.60


def build_stress_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "object": "Cloud-9",
                "rotation_supported": False,
                "resolved_rotation_curve": False,
                "redshift": 0.0,
                "quality_score": 0.30,
            },
            {
                "object": "Aquarius III",
                "rotation_supported": False,
                "resolved_rotation_curve": False,
                "redshift": 0.0,
                "quality_score": 0.25,
            },
            {
                "object": "Platypus",
                "rotation_supported": True,
                "resolved_rotation_curve": True,
                "redshift": 1.8,
                "quality_score": 0.70,
            },
            {
                "object": "NGC 2403",
                "rotation_supported": True,
                "resolved_rotation_curve": True,
                "redshift": 0.0,
                "quality_score": 0.95,
            },
        ]
    )


def clasificar_framework(row: pd.Series) -> str:
    if (not bool(row["rotation_supported"])) or (not bool(row["resolved_rotation_curve"])):
        return STATUS_OUT_OF_DOMAIN
    if float(row["redshift"]) > MAX_REDSHIFT_FOR_READY or float(row["quality_score"]) < MIN_QUALITY_FOR_READY:
        return STATUS_FUTURE_EXTENSION
    return STATUS_FRAMEWORK_READY


def run_stress_test() -> pd.DataFrame:
    df = build_stress_test_data().copy()
    df["framework_status"] = df.apply(clasificar_framework, axis=1)
    return df


def export_results(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "object",
        "framework_status",
        "rotation_supported",
        "resolved_rotation_curve",
        "redshift",
        "quality_score",
    ]
    df[export_cols].to_csv(output_csv, index=False)


def main() -> None:
    df = run_stress_test()
    out_csv = Path(__file__).resolve().parent / "stress_test_results.csv"
    export_results(df, out_csv)
    print(f"[OK] Stress test exported: {out_csv}")


if __name__ == "__main__":
    main()
