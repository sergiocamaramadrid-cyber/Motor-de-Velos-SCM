from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_STD_FOR_ZSCORE = 1e-10


def _default_input() -> Path:
    for p in [Path("data/sparc_175_master.csv"), Path("data/sparc_175_master_sample.csv")]:
        if p.exists():
            return p
    return Path("results/SPARC/sparc_175_master_sample.csv")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality control checks for F3 catalog")
    parser.add_argument("--input", default=str(_default_input()), help="Input catalog CSV")
    parser.add_argument("--out", default="results/quality/f3_quality_report.csv", help="Output report CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Missing input: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    issues: list[dict[str, object]] = []

    if "F3" in df.columns:
        std = float(df["F3"].std(ddof=0))
        if np.isfinite(std) and std > MIN_STD_FOR_ZSCORE:
            z = (df["F3"] - df["F3"].mean()) / std
            outliers = z.abs() > 3.0
            for idx in df.index[outliers]:
                issues.append({"issue_type": "outlier", "row_index": int(idx), "details": f"F3 z-score={z.loc[idx]:.3f}"})

    if {"fit_ok"}.issubset(df.columns):
        failed = ~df["fit_ok"].astype(bool)
        for idx in df.index[failed]:
            issues.append({"issue_type": "fit_failed", "row_index": int(idx), "details": "fit_ok=False"})

    if "beta_err" in df.columns:
        high_err = pd.to_numeric(df["beta_err"], errors="coerce") > 0.5
        for idx in df.index[high_err.fillna(False)]:
            issues.append({"issue_type": "extreme_error", "row_index": int(idx), "details": f"beta_err={df.loc[idx, 'beta_err']}"})

    if {"beta", "F3"}.issubset(df.columns):
        bad = (pd.to_numeric(df["beta"], errors="coerce") - pd.to_numeric(df["F3"], errors="coerce")).abs() > 0.2
        for idx in df.index[bad.fillna(False)]:
            issues.append({"issue_type": "beta_f3_inconsistency", "row_index": int(idx), "details": f"beta={df.loc[idx, 'beta']} F3={df.loc[idx, 'F3']}"})

    out_df = pd.DataFrame(issues)
    if out_df.empty:
        out_df = pd.DataFrame([{"issue_type": "ok", "row_index": -1, "details": "No quality issues"}])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Quality report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
