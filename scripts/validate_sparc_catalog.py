from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "logSigmaHI_out",
    "logMbar",
    "logRd",
    "F3",
    "f3_scm",
    "delta_f3",
    "fit_ok",
    "quality_flag",
    "beta",
    "beta_err",
    "reliable",
    "friction_slope",
    "velo_inerte_flag",
]

RANGE_RULES = {
    "logSigmaHI_out": (-5.0, 5.0),
    "logMbar": (0.0, 15.0),
    "logRd": (-5.0, 5.0),
    "F3": (-5.0, 5.0),
    "f3_scm": (-5.0, 5.0),
    "delta_f3": (-5.0, 5.0),
    "beta": (-5.0, 5.0),
    "beta_err": (0.0, 5.0),
    "friction_slope": (-5.0, 5.0),
}


def _default_input() -> Path:
    candidates = [
        Path("data/sparc_175_master.csv"),
        Path("data/sparc_175_master_sample.csv"),
        Path("results/SPARC/sparc_175_master_sample.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def validate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, object]] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for col in missing:
        issues.append(
            {
                "issue_type": "missing_column",
                "row_index": -1,
                "column": col,
                "value": "",
                "details": "Required column is missing",
            }
        )

    present_required = [c for c in REQUIRED_COLUMNS if c in df.columns]

    for col in present_required:
        null_mask = df[col].isna()
        for idx in df.index[null_mask]:
            issues.append(
                {
                    "issue_type": "nan_value",
                    "row_index": int(idx),
                    "column": col,
                    "value": "NaN",
                    "details": "Required value is NaN",
                }
            )

    for col, (low, high) in RANGE_RULES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad_mask = values.notna() & ((values < low) | (values > high))
        for idx in df.index[bad_mask]:
            issues.append(
                {
                    "issue_type": "out_of_range",
                    "row_index": int(idx),
                    "column": col,
                    "value": float(values.loc[idx]),
                    "details": f"Expected range [{low}, {high}]",
                }
            )

    if "galaxy" in df.columns:
        dup_mask = df["galaxy"].duplicated(keep=False)
    else:
        dup_mask = df.duplicated(keep=False)

    for idx in df.index[dup_mask]:
        issues.append(
            {
                "issue_type": "duplicate",
                "row_index": int(idx),
                "column": "galaxy" if "galaxy" in df.columns else "row",
                "value": str(df.loc[idx, "galaxy"]) if "galaxy" in df.columns else "duplicated_row",
                "details": "Duplicate entry detected",
            }
        )

    if {"fit_ok", "reliable"}.issubset(df.columns):
        bad = (~df["fit_ok"].astype(bool)) & (df["reliable"].astype(bool))
        for idx in df.index[bad]:
            issues.append(
                {
                    "issue_type": "inconsistent_flags",
                    "row_index": int(idx),
                    "column": "fit_ok/reliable",
                    "value": f"fit_ok={df.loc[idx, 'fit_ok']} reliable={df.loc[idx, 'reliable']}",
                    "details": "fit_ok=False cannot have reliable=True",
                }
            )

    if {"fit_ok", "quality_flag"}.issubset(df.columns):
        bad = (~df["fit_ok"].astype(bool)) & (df["quality_flag"].astype(str).str.lower() == "ok")
        for idx in df.index[bad]:
            issues.append(
                {
                    "issue_type": "inconsistent_flags",
                    "row_index": int(idx),
                    "column": "fit_ok/quality_flag",
                    "value": f"fit_ok={df.loc[idx, 'fit_ok']} quality_flag={df.loc[idx, 'quality_flag']}",
                    "details": "quality_flag='ok' inconsistent when fit_ok=False",
                }
            )

    if {"beta", "friction_slope"}.issubset(df.columns):
        beta = pd.to_numeric(df["beta"], errors="coerce")
        fr = pd.to_numeric(df["friction_slope"], errors="coerce")
        bad = (beta.notna() & fr.notna() & (np.abs(beta - fr) > 1e-6))
        for idx in df.index[bad]:
            issues.append(
                {
                    "issue_type": "inconsistent_flags",
                    "row_index": int(idx),
                    "column": "beta/friction_slope",
                    "value": f"beta={beta.loc[idx]} friction_slope={fr.loc[idx]}",
                    "details": "beta and friction_slope should match",
                }
            )

    out = pd.DataFrame(issues)
    if out.empty:
        out = pd.DataFrame(
            [
                {
                    "issue_type": "ok",
                    "row_index": -1,
                    "column": "",
                    "value": "",
                    "details": "No validation issues",
                }
            ]
        )
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SPARC master catalog.")
    parser.add_argument("--input", default=str(_default_input()), help="Input SPARC catalog CSV")
    parser.add_argument(
        "--out",
        default="results/validation/sparc_catalog_validation_report.csv",
        help="Output validation report CSV",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    report = validate_catalog(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"Validation report written to {out_path}")

    has_issues = not (len(report) == 1 and report.iloc[0]["issue_type"] == "ok")
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
